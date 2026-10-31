import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    """Causal group cover with training/development-fitted transition incidence."""
    def __init__(self,c):
        super().__init__();self.c=c
        sensors=c.get('input_shape',[40,6])[-1];classes=c.get('classes',3);width=c.get('width',64)
        groups=[[i] for i in range(sensors)]+c.get('paired_groups',[[i,i+1] for i in range(0,sensors-1,2)])
        members=torch.zeros(len(groups),sensors)
        for i,group in enumerate(groups):members[i,group]=1
        self.register_buffer('members',members)
        self.register_buffer('incidence',torch.ones(classes,classes,len(groups)))
        self.register_buffer('transition',torch.ones(classes,classes)-torch.eye(classes))
        self.cell=nn.GRUCell(3*sensors,width);self.second=nn.GRUCell(width,width);self.head=nn.Linear(width,classes)
    def prepare(self,data,development):
        """Fit pairwise logistic maps on train and threshold accuracy on development."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        def features(values):
            x=values['x'];window=self.c.get('feature_window',5)
            return np.stack([x[:,max(0,t-window+1):t+1].mean(1) for t in range(x.shape[1])],1).reshape(-1,x.shape[-1])
        train_x,dev_x=features(data),features(development)
        train_y,dev_y=data['state'].ravel(),development['state'].ravel()
        incidence=np.zeros(self.incidence.shape,np.float32)
        for first in range(len(incidence)):
            for second in range(first+1,len(incidence)):
                fit=np.isin(train_y,[first,second]);check=np.isin(dev_y,[first,second])
                if len(np.unique(train_y[fit]))<2 or len(np.unique(dev_y[check]))<2:continue
                for group,members in enumerate(self.members.cpu().numpy().astype(bool)):
                    model=LogisticRegression(C=100,max_iter=300,random_state=13)
                    model.fit(train_x[fit][:,members],train_y[fit])
                    value=balanced_accuracy_score(dev_y[check],model.predict(dev_x[check][:,members]))
                    incidence[first,second,group]=incidence[second,first,group]=value>=self.c.get('distinction_threshold',.78)
        transitions=np.zeros(self.transition.shape,np.float32)
        for source,target in zip(data['state'][:,:-1].ravel(),data['state'][:,1:].ravel()):
            if source!=target:transitions[int(source),int(target)]+=1
        transitions/=np.maximum(transitions.sum(-1,keepdims=True),1)
        self.incidence.copy_(torch.as_tensor(incidence,device=self.incidence.device));self.transition.copy_(torch.as_tensor(transitions,device=self.transition.device))
    def forward(self,b):
        if self.c.get('physical',False):
            from .physical import forward_model
            return forward_model(self,b)
        x=b['x'].float();n,time,sensors=x.shape;variant=self.c.get('variant','full');window=self.c.get('window',20);probe_interval=self.c.get('probe_interval',10)
        h=x.new_zeros(n,self.head.in_features);h2=h.clone();age=x.new_zeros(n,*self.transition.shape);previous=x.new_zeros(n,sensors);elapsed=previous.clone();remaining=x.new_zeros(n)
        predictions=[];readouts=[];attempts=[];unknown=[]
        for tick in range(time):
            if tick%window==0:
                allowance=int(self.c.get('fraction',.55)*window*sensors);remaining.fill_(allowance)
                required=sum(sensors for future in range(tick,min(time,tick+window)) if future%probe_interval==0)
                if variant not in {'no_probe','baseline'} and allowance<required:raise ValueError('Read allowance cannot fund mandatory probes')
            prior=self.head(h2).softmax(-1);urgency=prior[:,:,None]*self.transition[None]
            if variant!='no_age':urgency=urgency*(1+(age/probe_interval).clamp_max(1))
            selected=x.new_zeros(n,sensors)
            for row in range(n):
                stop=min(time,(tick//window+1)*window)
                reserved=sum(sensors for future in range(tick+1,stop) if future%probe_interval==0) if variant!='no_probe' else 0
                available=max(0,int(remaining[row])-reserved)
                probe=tick%probe_interval==0 and variant!='no_probe'
                budget=sensors if variant=='baseline' else min(available,sensors if probe else max(1,int(self.c.get('fraction',.55)*sensors)))
                covered=torch.zeros_like(age[row],dtype=torch.bool)
                if probe or variant=='baseline':selected[row,:budget]=1
                else:
                    while selected[row].sum()<budget:
                        added=self.members*(1-selected[row]);cost=(added*(1+self.c.get('wake_cost',.2)*(1-previous[row]))).sum(-1)
                        if variant=='no_wake':cost=added.sum(-1)
                        benefit=((~covered)[:,:,None]*urgency[row,:,:,None]*self.incidence).sum((0,1))
                        legal=(added.sum(-1)>0)&(added.sum(-1)+selected[row].sum()<=budget)
                        utility=(benefit/cost.clamp_min(1e-8)).masked_fill(~legal,-1)
                        if utility.max()<=0:break
                        group=int(utility.argmax());selected[row]=torch.maximum(selected[row],self.members[group]);covered|=self.incidence[:,:,group].bool()
                remaining[row]-=selected[row].sum()
            attempts.append(selected.clone())
            if 'available' in b:selected=selected*b['available'][:,tick].to(selected.dtype)
            completed=(self.members[None]*selected[:,None]).sum(-1)==self.members.sum(-1)[None]
            covered=(self.incidence[None].bool()&completed[:,None,None]).any(-1)
            age=(age+1)*(~covered)
            required=(prior[:,:,None]>.1)&(self.transition[None]>0)
            unknown.append((required&(age>probe_interval+self.c.get('confirmation',2))).any(-1).any(-1))
            elapsed=(elapsed+1)*(1-selected)
            h=self.cell(torch.cat([x[:,tick]*selected,selected,elapsed/probe_interval],-1),h)
            h2=self.second(h,h2);predictions.append(self.head(h2));readouts.append(selected);previous=selected
        logits=torch.stack(predictions,1)
        return {'probability':logits[:,-1].softmax(-1),'timeline_logits':logits,'readout':torch.stack(readouts,1),'attempted_readout':torch.stack(attempts,1),'unknown':torch.stack(unknown,1)}
    def objective(self,b,p):
        labels=b['state'].long().flatten();valid=labels>=0
        return F.cross_entropy(p['timeline_logits'].flatten(0,1)[valid],labels[valid])



def demo(seed,n,c):
    rng=np.random.default_rng(seed);state=(np.arange(40)//10)%3;states=np.tile(state,(n,1));x=np.eye(3)[states].repeat(2,axis=-1)+rng.normal(0,.1,(n,40,6));return {'x':x.astype('float32'),'y':states[:,-1],'state':states,'group':np.arange(n)}


def measure(p,d,c):
    predicted=p['timeline_logits'].argmax(-1)
    if 'unknown' in p:predicted=np.where(p['unknown'],-1,predicted)
    correct=[];false=0;missed=0;delays=[];deadline=c.get('deadline_steps',10);confirmation=c.get('confirmation',2)
    for row,(labels,truth) in enumerate(zip(predicted,d['state'])):
        acquired=p['readout'][row].sum(-1)>0
        events=np.flatnonzero((truth[1:]!=truth[:-1])&(truth[1:]>=0)&(truth[:-1]>=0))+1;notifications=[t for t in range(confirmation,len(labels)) if labels[t]>=0 and acquired[t-confirmation+1:t+1].all() and np.all(labels[t-confirmation+1:t+1]==labels[t]) and labels[t-confirmation]!=labels[t]];used=set()
        for index,event in enumerate(events):
            end=events[index+1] if index+1<len(events) else len(labels)
            valid=[t for t in notifications if event<=t<end and labels[t]==truth[event] and t not in used]
            if valid:t=min(valid);used.add(t);delays.append(t-event);correct.append(t-event<=deadline)
            else:missed+=1;correct.append(False)
        false+=len(notifications)-len(used)
    from sklearn.metrics import f1_score
    hours=len(predicted)*predicted.shape[1]*c.get('interval_s',1.)/3600
    attempted=p.get('attempted_readout',p['readout'])
    wakes=np.sum(np.diff(np.pad(attempted,((0,0),(1,0),(0,0))),axis=1)>0)
    wake_rate=float(wakes/max(hours*attempted.shape[-1],1e-12))
    energy=.15+.85*float(attempted.mean())+.0003*wake_rate
    return {'normalized_endpoint_energy_proxy':energy,'wakes_per_endpoint_hour':wake_rate,'unknown_fraction':float(np.mean(predicted<0)),'attempted_readout_fraction':float(p.get('attempted_readout',p['readout']).mean()),'state_macro_f1':float(f1_score(d['state'][d['state']>=0],predicted[d['state']>=0],labels=np.unique(d['state'][d['state']>=0]),average='macro',zero_division=0)),'false_notifications_per_hour':false/max(hours,1e-12),'readout_fraction':float(p['readout'].mean()),'transition_recall':float(len(delays)/max(len(delays)+missed,1)),'deadline_success':float(np.mean(correct)) if correct else 0.,'false_notifications':false,'p95_detected_delay_steps':float(np.quantile(delays,.95)) if delays else None}
