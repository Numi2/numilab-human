from pathlib import Path
import json,sys
import numpy as np
from fast_skin_intersection_audit import install_for_float32, owner

def run(path,out):
    z=np.load(path);x=z['vertex_world_m'];f=z['faces'];install_for_float32()
    records,_=owner.float32_surface_records(x,f)
    report=owner._audit_pair_prepared_first(owner._prepare_surface_aabb(records),records,same_surface=True)
    details=[]
    for a,b in report['triangle_pairs']:
        details.append(dict(rows=[a,b],bank=z['cortex_bank'][[a,b]].tolist(),marker=z['marker'][[a,b]].tolist(),
                            parent=z['parent_face'][[a,b]].tolist(),ids=f[[a,b]].tolist()))
    out.write_text(json.dumps(dict(count=report['count'],pairs=details),indent=2)+'\n')
    print('crossings',len(details),flush=True)

if __name__=='__main__':run(*map(Path,sys.argv[1:]))
