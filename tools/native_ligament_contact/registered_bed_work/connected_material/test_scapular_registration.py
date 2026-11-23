"""Test the existing cortex registration on held-out force attachments at the supported opening."""
import json
from pathlib import Path
import runpy
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from audit_route_clearance import TriangleDistance,Surface
from run_bed_work import digest
from verify_native_extensor_pair import loaded


def screen(root,fit,reader):
    case=root/'pose-one';_,poses,data=loaded(case,True,False)
    ref=json.loads((root/'material-source/scapular-frame.json').read_text())
    route=json.loads((root/'material-source/route-clearance.json').read_text())
    sys.path.insert(0,str(reader.parent/'src'));api=runpy.run_path(str(reader))
    bones={k[1]:(v,f) for k,_,v,f in api['primitives'](data,poses) if k[0]==51004}
    fits={r['bone']:r for r in json.loads(fit.read_text())}
    with np.load(root/'material-source/corrected/outer-boundaries.npz') as z:skin=Surface(z['loaded_xyz'],z['triangles'])
    results=[]
    for source in ref['bilateral']:
        t=fits[source['source']];body=source['body'];p=poses[body];turn=Rotation.from_quat(p['quaternion_xyzw']);joint=np.array(p['position_m'])
        if t['body']!=body or t['scale']!=1:raise ValueError('Unmatched cortical fit')
        native,cells=bones[t['sid']];cortex=TriangleDistance(native,cells)
        rot=Rotation.from_quat(t['rotation_xyzw']);shift=np.array(t['translation_m'])
        rows=[]
        for r in route['route_roles']:
            k=r['site'];site=route['sites'][str(k)]
            if site['body']!=body or any(g['site']==k for g in rows):continue
            pos=np.array(site['world_m']);local=turn.inv().apply(pos-joint)
            new=turn.apply(rot.inv().apply(local-shift))+joint
            e=skin.query(new)
            rows.append(dict(site=k,role=r['kind'],route=r['route'],before_cortex_m=cortex.nearest(pos),
                             after_cortex_m=cortex.nearest(new),before_skin_side=site['side'],after_skin_side=e['side'],
                             after_skin_distance_m=e['distance_m'],displacement_m=float(np.linalg.norm(new-pos))))
        tested,=[r for r in rows if r['site']==source['source_site']]
        if abs(source['assembled_cortex_distance_m']-tested['before_cortex_m'])>1e-7:raise ValueError('Wrong assembled cortex')
        results.append(dict(body=body,bone=source['source'],source_fit=digest(fit),surface_heldout_rms_m=t['heldout_after_rms_m'],
                     before_sites_outside=sum(r['before_skin_side']=='outside' for r in rows),
                     after_sites_outside=sum(r['after_skin_side']=='outside' for r in rows),deltoid_origin=tested,sites=rows))
    return dict(source_comparison=digest(root/'material-source/scapular-frame.json'),bilateral=results,
                scope='Frozen-pose geometric test of the inverse of the prior independent rigid cortex fit. Changes to tendon length, wrapping and moment arms require native force-path testing. No edited sites enter the launcher.')
if __name__=='__main__':
    p=Path(__file__).resolve().parent
    (p/'material-source/scapular-site-screen.json').write_text(json.dumps(screen(p,Path(sys.argv[1]),Path(sys.argv[2])),indent=2,allow_nan=False)+'\n')
