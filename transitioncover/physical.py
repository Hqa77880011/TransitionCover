"""Causal completion-time replay with hold, probe, quarantine and burst budgets."""
import argparse
import heapq
import json
from pathlib import Path
import numpy as np
import torch
from .method import Model,measure


def replay(model,x,availability=None):
    """Schedule before seeing current values; release observations only at completion."""
    c=model.c;time,sensors=x.shape;dt=c.get('interval_s',1.);window=c.get('window',300);probe=c.get('probe_interval',20)
    settle=c.get('settling_s',3.);transfer=c.get('transfer_s',.5);acquire=c.get('acquisition_s',1.)
    minimum=max(1,int(np.ceil(max(c.get('minimum_on_s',3.),settle+transfer+acquire)/dt)))
    device=next(model.parameters()).device;hidden=torch.zeros(1,model.head.in_features,device=device);second=hidden.clone()
    active=np.zeros(sensors,bool);hold=np.full(sensors,-1);wake=np.full(sensors,-np.inf);last_ack=np.full(sensors,-1.);last_attempt=np.full(sensors,-1.);quarantine=np.zeros(sensors,bool);latest=np.zeros(sensors,np.float32);last_complete=np.full(sensors,-np.inf)
    age=np.zeros(model.transition.shape);members=model.members.detach().cpu().numpy().astype(bool);incidence=model.incidence.detach().cpu().numpy().astype(bool);transition=model.transition.detach().cpu().numpy();group_credit=np.full(len(members),-np.inf)
    spent={};pending=[];sequence=0;log=[];logits=[];readouts=[];attempts=[];unknown=[];remaining_record=[]
    def allowance(slot):return int(c.get('fraction',.55)*window*sensors)
    def commitment_ok(endpoints,tick):
        obligations={}
        for future in range(tick,min(time,max(tick+minimum,int(hold.max())+1))):
            held=(hold>future)|endpoints
            obligations[future//window]=obligations.get(future//window,0)+int(held.sum())
        return all(spent.get(key,0)+value<=allowance(key) for key,value in obligations.items())
    for tick in range(time):
        now=tick*dt;completed=np.zeros(sensors,bool)
        while pending and pending[0][0]<=now:
            completion,_,sensor,value,success=heapq.heappop(pending)
            if success:
                latest[sensor]=value;last_complete[sensor]=completion;last_ack[sensor]=now;quarantine[sensor]=False;completed[sensor]=True
        age+=1
        for group,mask in enumerate(members):
            times=last_complete[mask]
            if len(times) and np.min(times)>group_credit[group] and np.max(times)-np.min(times)<=c.get('feature_window',5)*dt:
                age[incidence[:,:,group]]=0;group_credit[group]=np.min(times)
        timeout=c.get('communication_timeout_s',6.)
        quarantine|=(last_attempt>=0)&(now-np.maximum(last_ack,wake)>timeout)
        with torch.no_grad():prior=model.head(second).softmax(-1)[0].cpu().numpy()
        required=(prior[:,None]>.1)&(transition>0)
        pressure=bool(np.any(required&(age>=.8*probe)));due=tick%probe==0
        slot=tick//window;balance=allowance(slot)-spent.get(slot,0)
        end=min(time,(slot+1)*window)
        mandatory=sum(sensors*min(minimum,end-future) for future in range(tick+1,end) if future%probe==0)
        reserve=0 if pressure or due or c.get('variant')=='no_reserve' else int(c.get('burst_reserve',.1)*allowance(slot))
        requested=hold>tick;covered=np.zeros_like(age,dtype=bool)
        usable=~quarantine
        retry=(tick%c.get('retry_interval',probe)==0)
        if due or retry and quarantine.any():
            proposed=np.ones(sensors,bool)
            if commitment_ok(proposed,tick):requested=proposed
        else:
            urgency=prior[:,None]*transition*(1+np.minimum(age/probe,1))
            while True:
                best=None;utility=0.
                for group,mask in enumerate(members):
                    if np.any(mask&~usable):continue
                    union=requested|mask;extra=union&~requested
                    if not extra.any() or int(union.sum())+mandatory+reserve>balance or not commitment_ok(union,tick):continue
                    benefit=np.sum(urgency*(~covered)*incidence[:,:,group]);cost=np.sum(extra*(1+c.get('wake_cost',.2)*(~active)))
                    if benefit/max(cost,1e-8)>utility:best=group;utility=benefit/max(cost,1e-8)
                if best is None:break
                requested|=members[best];covered|=incidence[:,:,best]
        # Existing holds are charged first; their admission reserved future slots.
        if requested.sum()>balance:raise RuntimeError('Committed holds exceeded reserved allowance')
        new=requested&~active;wake[new]=now;hold[new]=tick+minimum
        spent[slot]=spent.get(slot,0)+int(requested.sum());active=requested.copy();attempts.append(requested.copy())
        for sensor in np.flatnonzero(requested):
            last_attempt[sensor]=now
            if now-wake[sensor]<settle:continue
            completion=now+acquire+transfer
            success=bool(availability is None or availability[tick,sensor])
            heapq.heappush(pending,(completion,sequence,int(sensor),float(x[tick,sensor]),success));sequence+=1
        elapsed=np.minimum(now-last_complete,probe*dt)/(probe*dt);elapsed[~np.isfinite(elapsed)]=1
        feature=torch.as_tensor(np.r_[latest*completed,completed.astype(float),elapsed],dtype=torch.float32,device=device)[None]
        hidden=model.cell(feature,hidden);second=model.second(hidden,second);logits.append(model.head(second)[0]);readouts.append(completed)
        expired=bool(np.any(required&(age>probe+c.get('confirmation',2))))
        unknown.append(expired);remaining_record.append(balance-int(requested.sum()))
        log.append({'tick':tick,'wakes':np.flatnonzero(new).tolist(),'quarantined':np.flatnonzero(quarantine).tolist(),'probe_due':due,'probe_admitted':bool(due and requested.all()),'completed':np.flatnonzero(completed).tolist(),'burst_released':pressure,'remaining_allowance':remaining_record[-1]})
    return torch.stack(logits),np.asarray(readouts),np.asarray(attempts),np.asarray(unknown),log


def forward_model(model,batch):
    outputs=[];readouts=[];attempts=[];unknown=[];ledger=[]
    for row in range(len(batch['x'])):
        availability=batch.get('available');availability=None if availability is None else availability[row].detach().cpu().numpy()
        logits,read,attempt,missing,records=replay(model,batch['x'][row].detach().cpu().numpy(),availability)
        outputs.append(logits);readouts.append(read);attempts.append(attempt);unknown.append(missing);ledger.append(records)
    model.last_completion_ledger=ledger
    logits=torch.stack(outputs);device=logits.device
    return {'timeline_logits':logits,'probability':logits[:,-1].softmax(-1),'readout':torch.as_tensor(np.asarray(readouts),device=device),'attempted_readout':torch.as_tensor(np.asarray(attempts),device=device),'unknown':torch.as_tensor(np.asarray(unknown),device=device)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True);parser.add_argument('--model',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    state=torch.load(args.model,map_location='cpu',weights_only=True);config=dict(state['config'],physical=True);model=Model(config).eval();model.load_state_dict(state['state'])
    with np.load(args.input,allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
    with torch.no_grad():prediction={key:value.cpu().numpy() for key,value in forward_model(model,{'x':torch.as_tensor(data['x']),**({'available':torch.as_tensor(data['available'])} if 'available' in data else {})}).items()}
    metrics=measure(prediction,data,config)
    if 'source' in data:
        records={}
        for source in np.unique(data['source']):
            selected=data['source']==source;records[str(source)]=measure({key:value[selected] for key,value in prediction.items()},{key:value[selected] for key,value in data.items() if len(value)==len(selected)},config)
        metrics={'per_source':records,'aggregation':'equal_source_weight',**{key:float(np.mean([record[key] for record in records.values()])) for key in records[next(iter(records))] if all(isinstance(record[key],(int,float)) for record in records.values())}}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'predictions.npz',**prediction);(out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');(out/'completion_ledger.json').write_text(json.dumps(model.last_completion_ledger,indent=2),encoding='utf-8')


if __name__=='__main__':main()
