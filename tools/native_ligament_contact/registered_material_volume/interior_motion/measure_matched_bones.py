"""Compare the new accepted skin and fat with all registered cortical solids."""
import argparse,json
from pathlib import Path
from compile_connected_joint import digest
from measure_bone_tissue_overlap import run as overlaps

def run(parent,stage,case,motion,out):
    v=stage/'witness-material/volume';z=json.loads(motion.read_bytes())
    receipt=case/f"native-traces/accepted-geometry/step-{z['accepted_step']}.receipt.json"
    state=json.loads(receipt.read_bytes());body=state['accepted_body_poses']
    if digest(receipt)!=z['receipt_sha256'] or [r['body_index'] for r in body]!=list(range(len(body))):
        raise ValueError('Unmatched rigid endpoint')
    native=case/'physical-accepted.node'
    if digest(native)!=z['physical_frame']:raise ValueError('Material coordinates changed')
    src=out/'source';src.mkdir(exist_ok=True,parents=True)
    for key,p in [('mesh.node',native),('mesh.ele',v/'mesh.ele')]:
        dst=src/key
        if not dst.exists():dst.symlink_to(p)
        if dst.resolve()!=p.resolve():raise ValueError('Stale audit input')
    frame=case/'accepted-registered-bone-pose.json'
    frame.write_text(json.dumps(dict(receipt_sha256=digest(receipt),all_bodies=[dict(body_index=r['body_index'],current=dict(position_world_m=r['position_m'],orientation_world_xyzw=r['quaternion_xyzw'])) for r in body]))+'\n')
    j=parent.parents[2];results=out/'volume'
    overlaps(j.parent/'ligament-equilibrium-070',results,frame,None,src)
    r=json.loads((results/'result.json').read_bytes());prev=json.loads((parent/'witness-material/motion-bone-volumes/result.json').read_bytes())
    if r['capture']!=digest(frame) or r['mesh']!=digest(native) or r['elements']!=digest(v/'mesh.ele') or r['results'].keys()!=prev['results'].keys():
        raise ValueError('Bone coverage mismatch')
    metrics={}
    for s in ('skin','fat'):
        after=sum(row[s] for row in r['results'].values());before=sum(row[s] for row in prev['results'].values())
        metrics[s]=dict(before_mm3=before,after_mm3=after,difference_mm3=after-before)
    (out/'summary.json').write_text(json.dumps(dict(material_endpoint=digest(native),bone_count=len(r['results']),summed_pair_volumes=metrics,
          scope='Pairwise solid intersections in the accepted endpoint, summed across anatomical bones.'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parent','stage','case','motion','out'):p.add_argument(k,type=lambda x:Path(x).resolve())
    run(**vars(p.parse_args()))
