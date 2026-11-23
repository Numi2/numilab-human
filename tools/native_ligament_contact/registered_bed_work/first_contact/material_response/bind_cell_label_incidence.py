"""Rebind only operational corners touched by primary-tuple changes, preserving all historical fields."""
import json,sys
from pathlib import Path
import numpy as np
from register_material_surfaces import pinned,sha

def run(launch,labels,out):
    env=json.loads(launch.read_bytes())['environment']
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    old=np.loadtxt(pinned(env,'CHILD_LABELS','CHILD_LABELS_SHA256'),delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    e=json.loads((labels/'result.json').read_bytes())['environment'];fp=pinned(e,'CHILD_LABELS','CHILD_LABELS_SHA256')
    new=np.loadtxt(fp,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4');changed=np.any(new!=old,axis=1)
    if np.any(new[:,0]!=old[:,0]):raise ValueError('Cell material changed')
    t=np.loadtxt(pinned(env,'TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256'),skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    n=int(t.max()+1);index=np.argsort(t.ravel(),kind='stable');off=np.r_[0,np.cumsum(np.bincount(t.ravel(),minlength=n))]
    rows=pinned(env,'COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256');out.mkdir();outpath=out/'combined-node-map.csv';touched=set();relabeled=set()
    with rows.open() as inp,outpath.open('x') as dst:
        dst.write(inp.readline())
        for line in inp:
            fields=line.rstrip().split(',');c=int(fields[43])
            if not changed[c]:dst.write(line);continue
            node=int(fields[0])-1;corners=index[off[node]:off[node+1]]
            if not np.array_equal([int(fields[40]),int(fields[41])],old[c,1:]):raise ValueError('Old corner identity differs')
            same=corners[np.all(new[corners//4]==old[c],axis=1)]
            if len(same):corner=int(same[0])
            else:
                corner=4*c+int(np.flatnonzero(t[c]==node)[0]);relabeled.add(node)
            z,k=divmod(corner,4);fields[40:42]=list(map(str,new[z,1:]));fields[43]=str(z)
            fields[44:48]=[str(int(v==k)) for v in range(4)]
            if any(float(a)!=0 for a in fields[48:52]):raise ValueError('Operational point requires a projection')
            dst.write(','.join(fields)+'\n');touched.add(node)
    envpin={prefix+'COMBINED_NODE_MAP_CSV':str(outpath),prefix+'COMBINED_NODE_MAP_SHA256':sha(outpath)}
    receipt=dict(changed_cell_labels=int(changed.sum()),updated_incident_nodes=len(touched),updated_operational_tuples=len(relabeled),source=sha(rows),environment=envpin)
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print({k:v for k,v in receipt.items() if k!='environment'},flush=True)
if __name__=='__main__':run(*(Path(z).resolve() for z in sys.argv[1:]))
