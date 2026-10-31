"""Completion causality, holds, faults, reserved budgets and learned gradients."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transitioncover.method import Model,demo
from transitioncover.physical import replay
root=Path(__file__).resolve().parents[1];out=root/'runs/physical_check';out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(13)
c=json.loads((root/'physical_config.json').read_text(encoding='utf-8'));c.update(width=4,window=40,probe_interval=20,input_shape=[40,6],classes=3);data=demo(13,4,c);model=Model(c);model.prepare(data,data)
available=np.ones_like(data['x'],bool);available[:,:16]=False
_,read,attempt,missing,log=replay(model,data['x'][0],available[0]);assert not read[:5].any();assert log[7]['quarantined'];assert any(not row['quarantined'] and row['completed'] for row in log[25:]);assert attempt.sum()<=int(c['fraction']*40*6)
for row in log:
 for sensor in row['wakes']:
  start=row['tick'];assert attempt[start:min(start+5,40),sensor].all()
batch={key:torch.as_tensor(value) for key,value in data.items()};batch['available']=torch.as_tensor(available)
model.train();before=model.cell.weight_ih.detach().clone();prediction=model(batch);loss=model.objective(batch,prediction);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);optimizer.zero_grad();loss.backward();assert torch.isfinite(loss) and model.cell.weight_ih.grad.norm()>0;optimizer.step();assert not torch.equal(before,model.cell.weight_ih)
model.eval();torch.save({'config':c,'state':model.state_dict()},out/'model.pt');data['available']=available;data['source']=np.array(['A','A','B','B']);np.savez_compressed(out/'test.npz',**data)
subprocess.run([sys.executable,'-m','transitioncover.physical','--input',str(out/'test.npz'),'--model',str(out/'model.pt'),'--output',str(out/'evaluation')],cwd=root,check=True)
metrics=json.loads((out/'evaluation/metrics.json').read_text());assert metrics['aggregation']=='equal_source_weight' and set(metrics['per_source'])=={'A','B'}
print('4.5s completions, >=3s holds, fault quarantine/re-entry, per-window budget, physical-path backward/update and source-equal replay CLI passed')
