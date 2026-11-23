"""Reconcile original MyoSim scapular cortex with the actual native force-site coordinate frame."""
import hashlib
import json
import struct
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_route_clearance import Surface,TriangleDistance
from run_bed_work import digest


def rigid(x,y):
    xc=x-x.mean(0);yc=y-y.mean(0);u,s,v=np.linalg.svd(xc.T@yc)
    if s[-1]<1e-6:raise ValueError('Insufficient source landmark span')
    proper=np.eye(3);proper[-1,-1]=np.linalg.det(u@v);R=u@proper@v
    return R,y.mean(0)-x.mean(0)@R


def evaluate(base,source,audit,archive):
    read=lambda p:json.loads(p.read_text())
    case=base/'pose-one';job=read(case/'launch.json');args=job['argv']
    tend=Path(args[args.index('--tendon-payload')+1]);myo=Path(args[2]);raw=myo.read_bytes()
    native=read(tend.with_suffix('.manifest.json'))
    if native['payload']['sha256']!=digest(tend) or native['source']['myosim_muscle_payload_sha256']!=digest(myo) or digest(myo)!=job['pins'][str(myo)]:
        raise ValueError('Changed native muscles or tendons')
    ref=read(audit);spec=read(source/'model-source.json');source_hash=digest(archive)
    if source_hash!=spec['source_archive_sha256'] or source_hash!=ref['source']['archive_sha256']:raise ValueError('Different MyoSim sources')
    poses={r['body_index']:r for r in read(case/'terminal.receipt.json')['initial_anatomical_registration']['body_poses']}
    bone=Path(next(a for a in args if a.endswith('.nhbones')))
    bone_meta=read(bone.with_suffix('.manifest.json'))
    mf=bone_meta['source']['runtime_reference']['manifest']
    if digest(mf['file'])!=mf['sha256'] or digest(bone)!=job['pins'][str(bone)]:raise ValueError('Changed body frame')
    inertial={d['core_body_index']:d for d in read(Path(mf['file']))['core_tree']['source_body_records']}
    actual={(r['muscle'],r['endpoint']):r for r in native['endpoints']}
    material=base/'material-source'
    supports=read(material/'bone-enclosure.json')['exposed_endpoints']
    results=[]
    with np.load(material/'corrected/outer-boundaries.npz') as s,np.load(source/'source-bones.npz') as z:
        skin=Surface(s['loaded_xyz'],s['triangles'])
        for side in ('r','l'):
            name='scapula_'+side;origin='DELT3' if side=='r' else 'DELT3_l'
            pairs=[(r,actual[(r['muscle'],r['endpoint'])]) for r in ref['endpoints'] if r['source_body_name']==name]
            ids={n['body_index'] for r,n in pairs}
            if len(ids)!=1:raise ValueError('Source scapula split across native bodies')
            body,=ids;pose=poses[body]
            x=np.array([r['source_site_position_body_m'] for r,n in pairs])
            y=np.array([struct.unpack_from('<I3f',raw,76+16*n['source_site_index'])[1:] for r,n in pairs])
            if not np.array_equal(y,np.array([n['source_local_point_m'] for r,n in pairs])):raise ValueError('Different muscle payload coordinates')
            source_xyz=z[name+'_local'];f=z[name+'_faces'];identity,=[t for t in spec['bones'] if t['body']==name]
            if hashlib.sha256(z[name+'_world'].tobytes()).hexdigest()!=identity['sha256_world_float64'] or len(f)!=identity['faces']:
                raise ValueError('Unmatched cortical source')
            errors=[]
            for i in range(len(x)):
                keep=np.arange(len(x))!=i;R,t=rigid(x[keep],y[keep]);errors.append(float(np.linalg.norm(x[i]@R+t-y[i])))
            if max(errors)>3e-7:raise ValueError('A single body frame does not explain the source sites')
            R,t=rigid(x,y)
            # model_surfaces stored cortex in the inertial frame, the frame of the native force-site rows.
            frame=inertial[body]
            world=Rotation.from_quat(frame['default_inertial_quaternion_world_xyzw']).apply(source_xyz)+frame['default_com_position_world_m']
            if np.max(np.linalg.norm(world-z[name+'_world'],axis=1))>1e-5:raise ValueError('Source cortex is not in the native inertial frame')
            v=source_xyz
            triangles=v[f]
            good=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>1e-14
            cortex=TriangleDistance(v,f[good])
            i,=[j for j,(r,n) in enumerate(pairs) if r['muscle']==origin and r['endpoint']=='origin']
            record,n=pairs[i];dist=cortex.nearest(y[i])
            if abs(dist-record['nearest_source_bone_mesh']['distance_m'])>1e-6:raise ValueError(('Independent source cortical distance disagrees',side,dist,record['nearest_source_bone_mesh']['distance_m']))
            target,=[r for r in supports if r['route']==origin and r['kind']=='origin']
            if target['site']!=n['source_site_index']:raise ValueError('Changed target attachment')
            world=Rotation.from_quat(pose['quaternion_xyzw']).apply(v)+pose['position_m']
            envelope=[skin.query(p) for p in world];outside=[p for p in envelope if p['side']=='outside']
            results.append(dict(body=body,source=name,source_site=n['source_site_index'],muscle=origin,matched_sites=len(x),
                                max_heldout_coordinate_error_m=max(errors),source_cortex_distance_m=dist,
                                assembled_cortex_distance_m=target['cortical_distance_m'],assembled_tendon_admission=n['admission_reason'],
                                archived_source_cortex_distance_m=record['nearest_source_bone_mesh']['distance_m'],
                                native_frame_row_rotation=R.tolist(),native_frame_translation_m=t.tolist(),
                                source_mesh_vertices=len(world),source_vertices_outside_loaded_skin=len(outside),
                                source_vertices_indeterminate=sum(p['side']=='indeterminate' for p in envelope),
                                source_max_skin_exposure_m=max((p['distance_m'] for p in outside),default=0.)))
    return dict(source_archive=source_hash,reference_cortex_arrays=digest(source/'source-bones.npz'),source_proximity=digest(audit),
                source_model=digest(source/'model-source.json'),source_muscles=digest(myo),source_tendons=digest(tend),bilateral=results,
                scope='Static source-cortex comparison with a proper rigid frame validated by leaving every identified site out in turn. This measures the registration error of the assembled force-to-bone interface. Original cortex tested against the current loaded skin with the same native opening skeleton.')
if __name__=='__main__':
    p=Path(__file__).resolve().parent
    (p/'material-source/scapular-frame.json').write_text(json.dumps(evaluate(p,*(Path(q).resolve() for q in sys.argv[1:])),indent=2,allow_nan=False)+'\n')
