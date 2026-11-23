"""Prepare the supported native scene in the registered whole-volume joint pose and certify both skeletons."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

def load(p):
    return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def pose_args(argv):
    entries={}
    for i,arg in enumerate(argv):
        if arg!='--pose-q': continue
        key=int(argv[i+1])
        if key in entries or not np.isfinite(float(argv[i+2])):
            raise ValueError('Invalid articulated pose override')
        entries[key]=i+2
    return entries

def prepare(supported,full,out):
    launch=load(supported);reference=load(full)
    if launch.get('headless',True) or reference['environment'].get('NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM')!='1':
        raise ValueError('Expected the supported native viewer and a registered full-volume launch')
    a=launch['argv'];b=reference['argv']
    for i in (1, 2):
        if sha(Path(a[i])) != launch['pins'][a[i]] or sha(Path(b[i])) != reference['pins'][b[i]]:
            raise ValueError('Unbound native articulated or muscle input')
        if launch['pins'][a[i]] != reference['pins'][b[i]]:
            raise ValueError('Different articulated or muscle source')
    target=pose_args(a);source=pose_args(b)
    if target.keys()!=source.keys():raise ValueError('Different joint override coverage')
    for q in target:
        a[target[q]]=b[source[q]]
    out.mkdir();launch['parent_launch']=sha(supported)
    launch['reference_material_pose_launch_sha256']=sha(full)
    (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')

def opening_poses(case, paired=None):
    receipt=load(case/'terminal.receipt.json')
    rows=receipt['initial_anatomical_registration']['body_poses']
    by={r['body_index']:r for r in rows}
    count=receipt['accepted_body_pose_count']
    if len(by)!=len(rows) or not by or not set(by)<=set(range(count)):
        raise ValueError('Invalid registered source poses')
    if len(by)==count:
        return by
    if paired is None:
        raise ValueError('A complete source pose capture is required for this native registration')
    extra=load(paired/'terminal.receipt.json')
    if (extra['base_pack_file_sha256']!=receipt['base_pack_file_sha256']
            or extra['accepted_body_pose_count']!=count):
        raise ValueError('Different native opening frames')
    owner=load(case/'launch.json'); other=load(paired/'launch.json')
    if owner['argv'][0:4]!=other['argv'][0:4] or owner['pins'][owner['argv'][0]]!=other['pins'][other['argv'][0]]:
        raise ValueError('Different native skeleton or runtime')
    a=pose_args(owner['argv']);b=pose_args(other['argv'])
    if ({k:float(owner['argv'][i]) for k,i in a.items()}!={k:float(other['argv'][i]) for k,i in b.items()}
            or owner['environment'].get('NUMI_EXPERIMENT_INITIAL_ROOT_POSE')!=other['environment'].get('NUMI_EXPERIMENT_INITIAL_ROOT_POSE')):
        raise ValueError('Different native starting pose')
    complete=opening_poses(paired)
    if any(row!=complete[i] for i,row in by.items()):
        raise ValueError('Exported native body rows disagree')
    return complete

def certify(full,case,out,initial_pose_capture=None):
    base=load(full/'native-initial-bodies.json');received=load(case/'terminal.receipt.json')
    src={v['body_index']:v['current'] for v in base['all_bodies']}
    tgt=opening_poses(case,initial_pose_capture)
    if (len(src)!=len(base['all_bodies'])
            or set(src)!=set(tgt) or sorted(tgt)!=list(range(received['accepted_body_pose_count']))):
        raise ValueError('Different rigid body coverage')
    full_receipt=load(full/'terminal.receipt.json')
    for row in full_receipt['initial_anatomical_registration']['body_poses']:
        recorded=src[row['body_index']]
        for key,native_key in [('position_m','position_world_m'),('quaternion_xyzw','orientation_world_xyzw')]:
            if not np.array_equal(np.asarray(row[key],'f4'),np.asarray(recorded[native_key],'f4')):
                raise ValueError('Full-volume skeleton differs from native receipt')
    root=1
    Q=Rotation.from_quat(tgt[root]['quaternion_xyzw'])*Rotation.from_quat(src[root]['orientation_world_xyzw']).inv()
    shift=tgt[root]['position_m']-Q.apply(src[root]['position_world_m'])
    dx=[];angle=[]
    for i in sorted(src):
        dx.append(np.linalg.norm(Q.apply(src[i]['position_world_m'])+shift-tgt[i]['position_m']))
        relative=Q*Rotation.from_quat(src[i]['orientation_world_xyzw'])*Rotation.from_quat(tgt[i]['quaternion_xyzw']).inv()
        angle.append(relative.magnitude())
    meshes=load(full/'verified-endpoint.json')['material_audits']
    opening=[r for r in meshes if r['phase']=='reference']
    if {int(r['material']) for r in opening}!={15,16} or len(opening)!=2:
        raise ValueError('Expected both skin and fat opening material audits')
    for row in opening:
        if int(row['cells'])<=0 or any(int(row[k])!=0 for k in ('inverted','nonfinite','volume_le_1e_18','shape_lt_1e_5')):
            raise ValueError('Invalid whole-volume material at release')
    mass_path,=(full/'native-traces').glob('registered-volume-fractional-mass-*.json')
    mass=load(mass_path)
    if mass['status']!='PASS' or mass['carrier_count']<=0 or mass['projection_rest_basis_rejects']!=0:
        raise ValueError('Invalid physical source mass mapping')
    cert=dict(full_volume_release=sha(full/'native-initial-bodies.json'),supported_release_receipt=sha(case/'terminal.receipt.json'),
              source_physical_nodes=sha(full/'physical-reference.node'),source_mass_receipt=sha(mass_path),mass_mapping=mass['mass_mapping_path_sha256'],bodies=len(src),orientation_xyzw=Q.as_quat().tolist(),
              world_translation_m=shift.tolist(),maximum_joint_position_delta_m=float(max(dx)),
              maximum_body_angle_delta_rad=float(max(angle)),passed=bool(max(dx)<=2e-6 and max(angle)<=2e-6),
              scope='Initial skeletal frame parity. The whole-volume reference can retain its coordinates and use this single external rigid transform. Dynamic material transfer and the supported skin remain separate checks.')
    if initial_pose_capture is not None:
        cert['paired_initial_pose_receipt']=sha(initial_pose_capture/'terminal.receipt.json')
    out.write_text(json.dumps(cert,indent=2)+'\n')
    return cert['passed']
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name,keys in [('prepare',('supported','full','out')),('check',('full','case','out'))]:
        p=sub.add_parser(name)
        for k in keys:p.add_argument(k,type=Path)
        if name=='check':p.add_argument('--initial-pose-capture',type=Path)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='prepare':prepare(**args)
    elif not certify(**args):raise SystemExit(1)
