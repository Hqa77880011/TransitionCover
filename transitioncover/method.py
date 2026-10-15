import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;shape=c.get('input_shape',[40,6]);s=shape[-1];classes=c.get('classes',3);w=c.get('width',24)
        self.cell=nn.GRUCell(2*s,w);self.head=nn.Linear(w,classes);self.register_buffer('incidence',torch.ones(classes,classes,s))
    def prepare(self,data):
        means=[]
        for label in range(self.incidence.shape[0]):
            mask=data['state']==label;values=data['x'][mask];means.append(np.mean(values,axis=0) if len(values) else np.zeros(self.incidence.shape[-1]))
        means=np.stack(means);scale=np.std(data['x'].reshape(-1,data['x'].shape[-1]),axis=0).clip(1e-6);self.incidence.copy_(torch.tensor(abs(means[:,None]-means[None,:])/scale>.5,device=self.incidence.device))
    def forward(self,b):
        x=b['x'].float();n,time,s=x.shape;h=x.new_zeros(n,self.head.in_features);age=x.new_zeros(n,self.incidence.shape[0],self.incidence.shape[1]);previous=x.new_zeros(n,s);remaining=x.new_full((n,),int(self.c.get('fraction',.55)*self.c.get('window',20)*s));predictions=[];readouts=[]
        for t in range(time):
            if t%self.c.get('window',20)==0:remaining.fill_(int(self.c.get('fraction',.55)*self.c.get('window',20)*s))
            prior=self.head(h).softmax(-1);urgency=prior[:,:,None]*prior[:,None,:]
            if self.c['variant']!='no_age':urgency=urgency*(1+(age/self.c.get('probe_interval',10)).clamp_max(1))
            selected=x.new_zeros(n,s)
            probe=t%self.c.get('probe_interval',10)==0 and self.c['variant']!='no_probe'
            for row in range(n):
                budget=min(int(remaining[row]),s if probe else max(1,int(self.c.get('fraction',.55)*s)))
                if self.c['variant']=='baseline':budget=s
                covered=torch.zeros_like(age[row],dtype=torch.bool)
                for _ in range(budget):
                    benefit=((~covered)[:,:,None]*urgency[row,:,:,None]*self.incidence).sum((0,1));cost=1+self.c.get('wake_cost',.2)*(1-previous[row])
                    if self.c['variant']=='no_wake':cost=torch.ones_like(cost)
                    utility=(benefit/cost).masked_fill(selected[row]>0,-1);channel=int(utility.argmax());selected[row,channel]=1;covered|=self.incidence[:,:,channel].bool()
                remaining[row]-=int(selected[row].sum());age[row]=(age[row]+1)*(~covered)
            h=self.cell(torch.cat([x[:,t]*selected,selected],-1),h);predictions.append(self.head(h));readouts.append(selected);previous=selected
        logits=torch.stack(predictions,1);return {'probability':logits[:,-1].softmax(-1),'timeline_logits':logits,'readout':torch.stack(readouts,1)}
    def objective(self,b,p):return F.cross_entropy(p['timeline_logits'].flatten(0,1),b['state'].long().flatten())


def demo(seed,n,c):
    rng=np.random.default_rng(seed);state=(np.arange(40)//10)%3;states=np.tile(state,(n,1));x=np.eye(3)[states].repeat(2,axis=-1)+rng.normal(0,.1,(n,40,6));return {'x':x.astype('float32'),'y':states[:,-1],'state':states,'group':np.arange(n)}


def measure(p,d,c):
    predicted=p['timeline_logits'].argmax(-1);correct=[];false=0;missed=0;delays=[];deadline=c.get('probe_interval',10);confirmation=c.get('confirmation',2)
    for labels,truth in zip(predicted,d['state']):
        events=np.flatnonzero(truth[1:]!=truth[:-1])+1;notifications=[t for t in range(confirmation,len(labels)) if np.all(labels[t-confirmation+1:t+1]==labels[t]) and labels[t-confirmation]!=labels[t]];used=set()
        for event in events:
            valid=[t for t in notifications if t>=event and labels[t]==truth[event] and t not in used]
            if valid:t=min(valid);used.add(t);delays.append(t-event);correct.append(t-event<=deadline)
            else:missed+=1;correct.append(False)
        false+=len(notifications)-len(used)
    return {'readout_fraction':float(p['readout'].mean()),'transition_recall':float(len(delays)/max(len(delays)+missed,1)),'deadline_success':float(np.mean(correct)) if correct else 0.,'false_notifications':false,'p95_detected_delay_steps':float(np.quantile(delays,.95)) if delays else None}
