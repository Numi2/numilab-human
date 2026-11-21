"""Independently reintersect both emitted material partitions at binary64 and Float32 precision."""
import json,sys
from pathlib import Path
import numpy as np
from prepare_material_interface import read_inputs,solid,close_remote_boundary
from prepare_posterior_muscles import closed
from compare_hand_interfaces import query
from material_context import sha

def run(cfg,rc,cap,base,folder,exe,out):
    surfaces,_=read_inputs(cfg,rc,cap,base);bones={}
    for name in ('bone_1','bone_2','bone_3','bone_18','bone_19'):
        bones[name]=solid(*surfaces[name],name)[0]
    x,f=surfaces['skin_deep'];roi=np.maximum(*(surfaces['muscle_'+str(j)][0].max(0) for j in (29,30)))
    x,f,_=close_remote_boundary(x,f,roi);deep=solid(x,f,'deep')[0]
    report=dict(predicate_sha256=sha(exe),sides={})
    glutes={}
    for precision in ('f8','f4'):
        glutes[precision]={}
        for sid in (29,30):
            p=np.load(folder/('muscle_%d.npz'%sid));v=p['vertices'].astype(precision);f=p['triangles']
            geom=solid(v,f,str(sid))[0];glutes[precision][sid]=geom
            own=('bone_1','bone_2','bone_18') if sid==29 else ('bone_1','bone_3','bone_19')
            r=dict(cortex_intersection_ml={k:(geom^bones[k]).volume()*1.e6 for k in own},outside_deep_fat_ml=(geom-deep).volume()*1.e6,topology=closed(v,f))
            if precision=='f4':r['exact_nonadjacent_self_intersections']=len(query(exe,v,f,np.ones(len(f),bool)))
            report['sides'].setdefault(sid,{})[precision]=r
        report[precision+'_bilateral_overlap_ml']=(glutes[precision][29]^glutes[precision][30]).volume()*1.e6
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
