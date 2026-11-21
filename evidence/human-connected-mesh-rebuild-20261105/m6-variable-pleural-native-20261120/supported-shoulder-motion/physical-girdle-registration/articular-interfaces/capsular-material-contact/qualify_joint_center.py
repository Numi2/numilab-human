"""Test whether native shoulder rotation alone can reconcile the independently registered joint centers."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from source_pair_audit import inputs
from shoulder_inputs import digest

def run(launch,refs,motion,donor,fits,sites,capture,out):
    _,poses,rest,_=inputs(launch,refs,motion);poses[0]=rest
    poses[-4]={r['body_index']:np.asarray(r['position_m']+r['quaternion_xyzw']) for r in json.loads(capture.read_text())['accepted_body_poses']}
    z=np.load(donor);reg={r['sid']:r for r in json.loads(fits.read_text())}
    rows=[]
    for r in json.loads(sites.read_text())['rows']:
        side='r' if r['side'] else 'l'
        pivot=z['shoulder_rot_'+side+'_center']
        if np.linalg.norm(pivot-z['shoulder_elv_'+side+'_center'])>1e-8:raise ValueError('Donor glenohumeral axes have distinct centers')
        centers=[]
        for e in r['attachments']:
            fit=reg[e['stable']];q=rest[e['body']];R=Rotation.from_quat(q[3:])
            v=z[fit['bone']+'_local'];w=z[fit['bone']+'_world']
            if np.max(abs(q[:3]+R.apply(v)-w))>1e-6:raise ValueError('Source pose does not reproduce the donor mesh')
            body_center=R.inv().apply(pivot-q[:3])
            mapped=Rotation.from_quat(fit['rotation_xyzw']).inv().apply(body_center-fit['translation_m'])/fit['scale']
            centers.append(dict(owner=e['body'],bank=e['stable'],donor_center_body_m=body_center.tolist(),
                 atlas_center_body_m=mapped.tolist(),offset_radius_m=np.linalg.norm(mapped-body_center)))
        samples=[]
        for st,p in poses.items():
            def transform(e,k):
                q=p[e['owner']];return q[:3]+Rotation.from_quat(q[3:]).apply(e[k])
            d=np.linalg.norm(transform(centers[0],'donor_center_body_m')-transform(centers[1],'donor_center_body_m'))
            delta=transform(centers[0],'atlas_center_body_m')-transform(centers[1],'atlas_center_body_m')
            samples.append((st,d,np.linalg.norm(delta),delta.tolist()))
        rows.append(dict(side=side,centers=centers,samples=len(samples),donor_pivot_max_separation_m=max(x[1] for x in samples),
             rotation_only_lower_bound_m=abs(centers[0]['offset_radius_m']-centers[1]['offset_radius_m']),
             mapped_center_separation_range_m=[min(x[2] for x in samples),max(x[2] for x in samples)],
             mapped_separation_native_entrance_m=next(x[3] for x in samples if x[0]==-4)))
    out.write_text(json.dumps(dict(source_donor_sha256=digest(donor),registration_sha256=digest(fits),capture_sha256=digest(capture),rows=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
