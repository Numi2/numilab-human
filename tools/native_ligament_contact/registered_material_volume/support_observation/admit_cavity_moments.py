"""Retain only anatomical P1-admissible cavities that conserve the loaded and reference mass moments."""
import json,sys
from pathlib import Path
import numpy as np
def run(original,stage,proposals,basis,out):
    out.mkdir();p=json.loads(proposals.read_bytes());record=json.loads((stage/'transfer.json').read_bytes());ov=np.load(stage/'overlay.npz');basis=json.loads(basis.read_bytes())
    bad={r for v in basis['roles'].values()for r in v['inadmissible_label_rows']};T=[np.loadtxt(x/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype=np.int32)-1 for x in (original,stage)]
    oldmass,newmass=[np.fromfile(x/'cell-mass.f64','<f8')for x in (original,stage)]
    poses=[np.loadtxt(original/n,skiprows=1,usecols=(1,2,3))for n in ('mesh.node','current.node')]
    keep=[];records=[]
    for cavity in p['patches']:
        rows=np.array(cavity['old']);dst=np.unique(ov['target'][np.isin(ov['source'],rows)])
        moments=[]
        for x in poses:
            center=x[T[0][rows][0,0]];scale=oldmass[rows].sum()*np.linalg.norm(x[T[0][rows]]-center,axis=2).max()
            delta=newmass[dst]@(x[T[1][dst]].mean(1)-center)-oldmass[rows]@(x[T[0][rows]].mean(1)-center)
            moments.append(float(np.max(abs(delta))/scale))
        inadmissible=len(bad.intersection(map(int,dst)))
        accepted=max(moments)<=1e-10 and not inadmissible
        records.append(dict(rows=rows.tolist(),cells=len(dst),relative_moments=moments,inadmissible_P1_labels=inadmissible,admitted=accepted))
        if accepted:keep.append(cavity)
    if not keep:raise ValueError('No compatible cavity')
    p['patches']=keep;p.pop('remaining_on_recorded_positions',None)
    (out/'cavities.json').write_text(json.dumps(p,indent=2)+'\n');(out/'admission.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
