"""Compare individual triangle-pair identities at the retained full-volume initial pose."""
import json,sys
from pathlib import Path
import numpy as np
from screen_material_transport import inputs
from bone_overlap import intersections

def run(scene,loader,receipt,bone,out):
    _,banks,bs,_,pins=inputs(scene,loader,receipt,bone);rows=[]
    for b in (8,9):
        old,new,f=bs[b];low=np.minimum(old.min(0),new.min(0));hi=np.maximum(old.max(0),new.max(0))
        for (sem,sid),(v,g) in banks.items():
            if sem==51004 and sid==b or np.any(low>v.max(0)) or np.any(hi<v.min(0)):continue
            a,c=(set(map(tuple,intersections(x,f,v,g))) for x in (old,new))
            if a or c:
                rows.append(dict(bone=b,semantic=sem,stable=sid,before=len(a),after=len(c),added=len(c-a),removed=len(a-c),added_first=sorted(c-a)[:1],removed_first=sorted(a-c)[:1]))
    out.write_text(json.dumps(dict(inputs=pins,rows=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
