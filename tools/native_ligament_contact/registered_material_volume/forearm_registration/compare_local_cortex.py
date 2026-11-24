"""Intersect the full incident patch before and after registration with the same 185 rigid cortical solids."""
import json,sys
from pathlib import Path
import numpy as np
from measure_bone_tissue_overlap import run as overlaps
from compile_connected_joint import digest
def run(volume,fit,source,frame,out):
    out.mkdir(exist_ok=True)
    rec=json.loads((fit/'fit.json').read_bytes())
    mesh=np.loadtxt(volume/'mesh.ele',skiprows=1,dtype='i4');nodes=np.array(rec['nodes_0_based'])
    rows=np.flatnonzero(np.isin(mesh[:,1:5]-1,nodes).any(1));patch=mesh[rows].copy()
    patch[:,0]=np.arange(1,len(rows)+1);ele=out/'mesh.ele'
    np.savetxt(ele,patch,fmt='%d',header=f'{len(patch)} 4 1',comments='')
    records={}
    for k,p in [('before',volume/'current.node'),('after',fit/'current.node')]:
        s=out/k;s.mkdir(exist_ok=True)
        for name,target in [('mesh.node',p),('mesh.ele',ele)]:
            if not (s/name).exists():(s/name).symlink_to(target.resolve())
        result=out/(k+'-bones');overlaps(source,result,frame,None,s)
        records[k]=json.loads((result/'result.json').read_bytes())['results']
        if len(records[k])!=185 or any('skin' not in r or 'fat' not in r for r in records[k].values()):raise ValueError('Incomplete cortical coverage')
    totals={};changed=[]
    for role in ['skin','fat']:
        a,b=([r[role]for r in records[k].values()]for k in ['before','after']);totals[role]={'before_mm3':sum(a),'after_mm3':sum(b)}
        for bone,r in records['before'].items():
            u,v=r[role],records['after'][bone][role]
            if abs(u-v)>1e-8:changed.append(dict(bone=bone,role=role,before_mm3=u,after_mm3=v))
    d=dict(pose=digest(frame),rows=rows.tolist(),bone_count=len(records['after']),sums=totals,changed_pairs=changed)
    (out/'result.json').write_text(json.dumps(d,indent=2)+'\n');print(totals,changed)
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
