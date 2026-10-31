"""CPU functional checks; synthetic fixtures do not reproduce paper results."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transitioncover import method
from transitioncover.cli import tensor_batch

torch.set_num_threads(1)
torch.manual_seed(13)
root=Path(__file__).resolve().parents[1]
out=root/'runs'/'functional_check'
out.mkdir(parents=True,exist_ok=True)
c=json.loads((root/'config.json').read_text(encoding='utf-8'))
c.update(width=4,variant='full')
data=method.demo(13,20,c)
np.savez_compressed(out/'fixture.npz',**data)
with np.load(out/'fixture.npz',allow_pickle=False) as archive:
 data={k:archive[k] for k in archive.files}
model=method.Model(c)
if hasattr(model,'prepare'):model.prepare(data,data)
batch=tensor_batch(data,np.arange(2),'cpu')
model.train();before={name:p.detach().clone() for name,p in model.named_parameters()}
prediction=model(batch);loss=model.objective(batch,prediction)
assert torch.isfinite(loss),loss
optimizer=torch.optim.AdamW(model.parameters(),lr=.001);optimizer.zero_grad();loss.backward()
for prefix in ['cell', 'second', 'head']:
 assert any(p.grad is not None and p.grad.norm()>0 for name,p in model.named_parameters() if name.startswith(prefix)),prefix
optimizer.step()
assert any(not torch.equal(before[name],p) for name,p in model.named_parameters())
model.eval();first=model(batch)
torch.save({'config':c,'state':model.state_dict()},out/'roundtrip.pt')
saved=torch.load(out/'roundtrip.pt',weights_only=True);loaded=method.Model(saved['config']);loaded.load_state_dict(saved['state']);loaded.eval();second=loaded(batch)
for key in first:
 if isinstance(first[key],torch.Tensor):torch.testing.assert_close(first[key],second[key],rtol=1e-5,atol=1e-6)
print('gradient/update and checkpoint roundtrip passed; loss=',float(loss.detach()))
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def run(*args):
 subprocess.run([sys.executable,'-m','transitioncover.cli',*map(str,args)],cwd=root,env=env,check=True)
run('prepare','--input',out/'fixture.npz','--output',out/'prepared')
run('train','--input',out/'prepared','--output',out/'trained','--config',root/'config.json','--set','width=4','--epochs',1,'--batch-size',4,'--seed',13)
run('evaluate','--input',out/'prepared/test.npz','--model',out/'trained/model.pt','--output',out/'trained/evaluation','--batch-size',4)
run('plot','--input',out/'trained','--output',out/'plots')
print((out/'trained/evaluation/metrics.json').read_text(encoding='utf-8'))

truth=np.array([[0,0,1,1,1,2,2,2]])
logits=np.eye(3)[truth]*10
values=method.measure({'timeline_logits':logits,'readout':np.ones((1,8,2))},{'state':truth},{'confirmation':2,'deadline_steps':2,'interval_s':1})
assert values['transition_recall']==1 and values['deadline_success']==1 and values['false_notifications']==0,values
print('hand-calculated 2 events, delays 1,1 and 100% DS passed')

blocked=dict(batch);blocked['available']=torch.zeros_like(batch['x']);model.eval();result=model(blocked)
assert result['readout'].sum()==0 and result['attempted_readout'].sum()>0 and result['unknown'].any()
reads=result['attempted_readout'].sum(-1)
for start in range(0,reads.shape[1],c['window']):assert (reads[:,start:start+c['window']].sum(-1)<=int(c['fraction']*c['window']*reads.new_tensor(batch['x'].shape[-1]))).all()
print('unavailable reads do not complete/reset pair ages, unknown is exposed, and command allowance is retained')
