"""Check source donor/region ownership against the exact native cell preference rule."""
import json,sys
from pathlib import Path
import numpy as np
from register_material_surfaces import sha


def run(mesh,support,out):
    labels=np.loadtxt(mesh/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    receipt=json.loads((mesh/'receipt.json').read_bytes())
    boundary=min(r['first_cell'] for r in receipt['regions'])
    output={}
    for mat,tag in ((15,'skin'),(16,'fat')):
        rows=np.flatnonzero(labels[:,1]==mat)
        sr=json.loads((support/(tag+'-p1-receipt.json')).read_bytes())
        p=Path(sr['p1_one_hop_path'])
        if sha(p)!=sr['p1_one_hop_sha256']:raise ValueError('Changed P1')
        a=np.load(p);off=a['cell_group_offsets'];g=a['cell_group_ids'];tbl=a['group_tuples']
        f=a['cell_group_p1_score_numerators']/a['cell_group_p1_score_denominators']
        total=np.add.reduceat(f,off[:-1]);high=np.maximum.reduceat(f,off[:-1])
        cells=np.repeat(np.arange(len(rows)),np.diff(off))
        equal=np.all(tbl[g]==labels[rows[cells],1:4],axis=1)
        score=np.maximum.reduceat(np.where(equal,f,0),off[:-1])
        bad=np.flatnonzero(score<high-1e-12)
        wrong=np.flatnonzero(abs(total-1)>1e-12)
        unique=np.add.reduceat((f==high[cells]).astype(int),off[:-1])==1
        o=dict(role_cells=len(rows),sum_failures=len(wrong),primary_failures=len(bad),
               previous_connectivity=int((rows[bad]<boundary).sum()),rebuilt_region=int((rows[bad]>=boundary).sum()),
               zero_current_support=int((score[bad]==0).sum()),unique_max=int(unique[bad].sum()),
               witness_global=rows[bad[:15]].tolist(),first=[])
        for j in bad[:8]:
            ranks=np.arange(off[j],off[j+1]);rank=ranks[f[ranks]==high[j]]
            o['first'].append(dict(global_cell=int(rows[j]),current_tuple=labels[rows[j],1:4].tolist(),
                          current_fraction=float(score[j]),maximum_fraction=float(high[j]),highest_source_tuples=tbl[g[rank]].tolist()))
        output[tag]=o;print(tag,{k:v for k,v in o.items() if k!='first'},flush=True)
    out.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
