"""Reconstruct the constrained cell from paired Float32 native displacement words."""
import json,re
from pathlib import Path
import numpy as np

def load_capture():
    info=json.loads(Path('cell-observer/identity.json').read_bytes())
    txt=Path('cell-trial/stderr.log').read_text()
    stage={}
    for m in re.finditer(r'cell_trial root=1 iteration=15 phase=(\d) global=(\d+) ((?:[-+eE0-9.,]+ ){13}[-+eE0-9.,]+)',txt):
        phase,n,p=m.groups()
        v=np.array([[np.float32(x) for x in w.split(',')] for w in p.split()],dtype='f8')
        stage.setdefault(int(phase),{})[int(n)-info['offset']]=v
    return stage
def run():
    cap=load_capture();a=np.load('cavity.npz');row=a['selected'];t=a['tet'][a['rows']==row][0];idx=np.searchsorted(a['ids'],t);ref=a['xyz'][0,idx];dm=ref[1:]-ref[0];dt=7.8125e-6
    F=lambda u:np.eye(3)+np.linalg.solve(dm,u[1:]-u[0]);out={}
    for phase,rows in cap.items():
        data=np.stack([rows[n]for n in t]);u=data[:,6,:3]+data[:,7,:3];v=data[:,9,:3];du=data[:,0,:3];curr=data[:,12,:3]+data[:,13,:3]
        if (np.abs(data[:,4,:3]-ref)).max()>2e-6:raise ValueError('Reference frame mismatch')
        out[phase]=dict(accepted_J=np.linalg.det(F(u)),current_J=np.linalg.det(F(curr)),raw_endpoint_J=np.linalg.det(F(u+dt*(v+du))),correction_rms_m_per_s=np.sqrt(np.mean((du**2).sum(1))),point_error_m=float(np.max(abs(u+ref-a['xyz'][2,idx]))))
    Path('measured-cell.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':run()
