from pathlib import Path
import csv,sys,json
import numpy as np

def run(p47,snapshot,out):
    with snapshot.open() as f:rows=[r for r in csv.DictReader(f) if r['field']=='material_node']
    xyz=np.array([[float(r[c]) for c in 'xyz'] for r in rows]);off=0;stats={}
    for sid in (29,30):
        d=np.load(p47/'projected'/f'{sid}.npz');v=d['vertices'].astype(float);t=d['cells'];use=np.flatnonzero(d['cell_volumes']>0)
        curr=xyz[off:off+len(v)];off+=len(v)
        det=lambda x:np.linalg.det(x[t[:,1:]]-x[t[:,:1]])
        ref=det(v);J=det(curr)/ref;fraction=6*d['cell_volumes']/ref
        full_mass=np.bincount(t.ravel(),weights=np.repeat(ref/24,4),minlength=len(v));actual=np.bincount(t.ravel(),weights=d['shape_integrals'].ravel(),minlength=len(v))
        rows=[]
        for i in use[np.argsort(J[use])[:12]]:
            rows.append(dict(cell=int(i),J=J[i],occupancy=fraction[i],volume_ml=d['cell_volumes'][i]*1e6,nodes=t[i].tolist(),node_support_fraction=(actual[t[i]]/full_mass[t[i]]).tolist()))
        stats[sid]=rows
    out.write_text(json.dumps(stats,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
