"""Measure the last free digital tendon segment in the same entrance pose as each inferred footprint."""
import json,sys
from pathlib import Path
import numpy as np
from hand_geometry import Surface
from material_context import read_inputs,get_layers

def intervals(a,b,surface):
    delta=b-a;length=float(np.linalg.norm(delta))
    if length<=0:raise ValueError('Zero length route')
    hits=surface.ray(a,delta/length,length)
    # Record all face crossings. Endpoint ownership is determined separately.
    cuts=[0.]+[u for u,i in hits if 1e-9<u<length-1e-9]+[length]
    inner=sum(v-u for u,v in zip(cuts,cuts[1:]) if surface.inside(a+(u+v)/(2*length)*delta))
    return dict(total_mm=length*1000,inside_mm=inner*1000,interior_hits=[i for u,i in hits if 1e-9<u<length-1e-9])

def run(w,manifest,c,foot,out):
    routes,_,bones,_,_=read_inputs(w/'current-package.json',manifest,c/'emitted-native/stderr.log')
    shell={k:Surface(*v) for k,v in get_layers(w/'qualified/native-coupled-006/native/resting-human.mrvpack',w/'qualified/motion-008/step-8.receipt.json').items()}
    slots={k:Surface(*v) for k,v in bones.items()};bbox={k:(s.v.min(0),s.v.max(0)) for k,s in slots.items()}
    results={}
    for row in json.loads(foot.read_text())['footprints']:
        name=row['muscle'];r=routes[name];p=r[-2]['world'];entry={}
        for label,q in [('authored',r[-1]['world']),('inferred',np.array(row['lines'][1]['samples'][1]['world_m']))]:
            lo,hi=np.minimum(p,q)-1e-9,np.maximum(p,q)+1e-9
            contacts={}
            for key,(a,b) in bbox.items():
                if np.any(a>hi) or np.any(b<lo):continue
                v=intervals(p,q,slots[key])
                if v['inside_mm']>0 or v['interior_hits']:contacts[key]=v
            layers={k:intervals(p,q,s) for k,s in shell.items()}
            entry[label]=dict(solid_bones=contacts,layers=layers)
        results[name]=entry
    out.write_text(json.dumps(results,indent=2)+'\n')
    for name,r in results.items():
        print(name,[(key,round(sum(i['inside_mm'] for i in row['solid_bones'].values()),4),round(row['layers']['outer']['total_mm']-row['layers']['outer']['inside_mm'],4)) for key,row in r.items()])
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
