"""Return the source faces and registered cortical positions behind an actual sampled crossing."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_native_finger_cortex import sha
def run(work,control,case,output):
    sys.path[:0]=[str(work/'resting-surface-clearance-001/src'),str(work/'resting-surface-clearance-001'),str(work/'resting-clearance-and-atrium-005')]
    import audit
    from fast_skin_intersection_audit import install_for_float32
    install_for_float32()
    initial=np.load(control/'initial-cortical.npz');r=json.loads((control/'step-0.receipt.json').read_text());members=json.loads((control/'initial-cortical.json').read_text())['member_by_stable_id']
    poses={p['body_index']:p for p in r['accepted_body_poses']};coords={};faces={};keys={}
    for name,index,_,_ in members.values():
        if name.startswith(('4proxph','5proxph')):
            p=poses[index];keys[index]=name;faces[name]=initial[name+'_f']
            coords[name]=Rotation.from_quat(p['quaternion_xyzw']).inv().apply(initial[name+'_v'].astype(float)-p['position_m'])
    states={}
    for row in csv.DictReader((case/'resting-body-motion-partition.csv').open()):
        b=int(row['body_index'])
        if b not in keys:continue
        n=keys[b];q=[float(row['orientation_'+a]) for a in 'xyzw'];t=[float(row['root_'+a+'_m'])+float(row['com_relative_root_'+a+'_m']) for a in 'xyz']
        states.setdefault(int(row['accepted_step']),{})[n]=Rotation.from_quat(q).apply(coords[n])+t
    data={}
    for k,points in states.items():
        out={}
        for side in ['l','r']:
            bank=['%dproxph_%s'%(i,side) for i in [4,5]]
            records=[audit.clearance._exact_surface_records(points[name].astype('<f4'),faces[name]) for name in bank]
            found=audit.intersections._audit_pair_prepared_first(audit.intersections._prepare_surface_aabb(records[0]),records[1])
            out[side]=[dict(face_ids=[int(a),int(b)],vertices=[faces[bank[0]][a].tolist(),faces[bank[1]][b].tolist()]) for a,b in found['triangle_pairs']]
        data[k]=out
    output.write_text(json.dumps(dict(native_initial_pack=r['pack_file_sha256'],sample_trace=sha(case/'resting-body-motion-partition.csv'),witnesses=data),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
