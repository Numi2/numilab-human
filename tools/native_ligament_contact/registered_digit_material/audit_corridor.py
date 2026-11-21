"""Measure the dorsal material corridor and force-site enclosure on the connected tetrahedral boundary."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from material_context import get_layers
from hand_geometry import Surface

def audit(work,mat,trial,history,sites,output):
    ref=get_layers(work/'qualified/native-coupled-006/native/resting-human.mrvpack',work/'qualified/motion-008/step-8.receipt.json')
    x=np.load(mat)['x'];y=np.load(trial);tree=cKDTree(x)
    layers={}
    for k,(v,faces) in ref.items():
        distance,ids=tree.query(v)
        if distance.max()!=0 or len(np.unique(ids))!=len(v):raise ValueError('Material boundary is not the source of the exact native mesh')
        layers[k]=Surface(y[ids],faces)
    data=[]
    for t in json.loads(history.read_text())['targets']:
        q=np.array(t['origin']);n=np.array(t['normal'])
        deep=layers['deep'];hits=deep.ray(q+1e-8*n,n,.025)
        enclosed=deep.inside(q)
        measured=1000*(hits[0][0]+1e-8) if enclosed and hits else None
        data.append(dict(label=t['label'],cortex_enclosed=enclosed,before_mm=t['before_mm'],after_mm=measured,required_mm=t['target_mm']))
    source=json.loads(sites.read_text());site_after=[]
    for row in source:
        p=np.array(row['world'])
        site_after.append(dict(name=row['name'],order=row['order'],inside={k:s.inside(p) for k,s in layers.items()}))
    result=dict(rows=data,sites=site_after,
                initially_sufficient=sum(r['before_mm']>=r['required_mm'] for r in data),
                now_sufficient=sum(r['after_mm'] is not None and r['after_mm']>=r['required_mm'] for r in data),
                previously_exposed=sum(not s['layers']['outer']['inside'] for s in source),
                now_exposed=sum(not s['inside']['outer'] for s in site_after),
                cortex_outside_deep=sum(not r['cortex_enclosed'] for r in data))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('rows','sites')},flush=True)
if __name__=='__main__':audit(*map(Path,sys.argv[1:]))
