"""Measure all cortical solids on the verified accepted registered tissue and rigid endpoints."""
from pathlib import Path
import json,sys
from compile_connected_joint import digest
from measure_bone_tissue_overlap import run as overlaps
def run(owner,src,case,motion,control,out):
    out.mkdir(exist_ok=True);s=out/'source';s.mkdir(exist_ok=True)
    m=json.loads(motion.read_bytes())
    receipt=case/f"native-traces/accepted-geometry/step-{m['accepted_step']}.receipt.json"
    if digest(receipt)!=m['receipt_sha256'] or digest(case/'physical-accepted.node')!=m['physical_frame']:raise ValueError('Unmatched accepted state')
    r=json.loads(receipt.read_bytes());body=r['accepted_body_poses'];frame=out/'rigid-pose.json'
    if [r['body_index'] for r in body]!=list(range(len(body))):raise ValueError('Incomplete rigid state')
    for n,p in [('launch.json',case/'launch.json'),('source/mesh.node',case/'physical-accepted.node'),('source/mesh.ele',src/'mesh.ele')]:
        if not (out/n).exists():(out/n).symlink_to(p.resolve())
    frame.write_text(json.dumps(dict(receipt_sha256=digest(receipt),all_bodies=[dict(body_index=row['body_index'],current=dict(position_world_m=row['position_m'],orientation_world_xyzw=row['quaternion_xyzw']))for row in body]))+'\n')
    overlaps(owner,out/'solids',frame,None,s);a=json.loads((control/'result.json').read_bytes());b=json.loads((out/'solids/result.json').read_bytes())
    names=set(a['results']);results={}
    if names!=set(b['results']) or len(names)!=185 or a['elements']!=b['elements']:raise ValueError('Different anatomical scope')
    for k in ['skin','fat']:
        results[k]={label:sum(r[k]for r in z['results'].values())for label,z in [('prior_mm3',a),('actual_mm3',b)]}
    report=dict(bones=len(names),results=results,pose=digest(frame),accepted_time_s=m['accepted_s'],source=digest(control/'result.json'),raw=digest(out/'solids/result.json'))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(results)
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))
