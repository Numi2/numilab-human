"""Re-register both scapular origins and their private wrap frames in the existing native muscle and tendon inputs."""
from pathlib import Path
import copy
import json
import struct
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from numilab_human import model as owner
from inspect_shoulder_source import nearest
from qualify_deltoid_route import program
from recount_registered_crossings import digest,read


def cortical_circle(v,f,center,radius,target):
    tri=v[f];a,b,c=tri.transpose(1,0,2);ab=b-a;ac=c-a
    n=np.cross(ab,ac);norm=np.linalg.norm(n,axis=1)
    n/=np.maximum(norm[:,None],1.e-30)
    h=np.einsum('ij,ij->i',a-center,n)
    projection=center+h[:,None]*n;rad=np.sqrt(np.maximum(0.,radius**2-h*h))
    # In each triangle the feasible locus is an arc. The nearest point to the
    # anatomical prediction is either radial to that prediction or an arc end.
    choices=[]
    for idx in np.flatnonzero((norm>0)&(np.abs(h)<=radius)):
        vertices=tri[idx];z=projection[idx];u=target-z;u-=n[idx]*(u@n[idx])
        if np.linalg.norm(u)>0:
            p=z+rad[idx]*u/np.linalg.norm(u)
            ux=np.linalg.lstsq(np.stack((ab[idx],ac[idx]),axis=1),p-a[idx],rcond=None)[0]
            if min(ux)>=0 and sum(ux)<=1:choices.append((idx,p))
        for j in range(3):
            x=vertices[j];d=vertices[(j+1)%3]-x;A=d@d;B=(x-center)@d;C=np.sum((x-center)**2)-radius**2
            disc=B*B-A*C
            if A==0 or disc<0:continue
            for t in ((-B-np.sqrt(disc))/A,(-B+np.sqrt(disc))/A):
                if 0<=t<=1:choices.append((idx,x+t*d))
    if not choices:raise ValueError('Cortical surface misses source length sphere')
    return np.array([k for k,p in choices]),np.array([p for k,p in choices])


def pack_envelope(e,body):
    coords=[v for xyz in e['node_local_points_m'] for v in (*xyz,0.)]
    maps=[v for mat in e['force_maps'] for row in mat for v in (*row,0.)]
    return struct.pack('<4I68f',body,e['bone_stable_id'],e['source_triangle_index'],4,*coords,*maps,
        e['surface_distance_m'],e['patch_radius_m'],e['sampled_total_force_amplification'],e['l2_force_amplification'])


def register(folder,data,lib,out):
    launch=read(data/'bed-unscaled-four/launch.json');args=launch['argv']
    myo=Path(args[2]);tendon=Path(args[args.index('--tendon-payload')+1]);bones=Path(args[3])
    reg=read(folder/'source-comparison.json')
    if any(digest(p)!=reg[k] or digest(p)!=launch['pins'][str(p)] for p,k in ((myo,'routes'),(tendon,'tendon'),(bones,'bone'))):raise ValueError('Native identity mismatch')
    muscle=bytearray(myo.read_bytes());t=bytearray(tendon.read_bytes());meta=read(tendon.with_suffix('.manifest.json'))
    mh=struct.unpack_from('<8s9I32s',muscle);wo=76+16*mh[4];ro=wo+64*mh[5]
    th=list(struct.unpack_from('<8s10I32s32s32s',t));old_count=th[6]
    route=np.frombuffer(bytes(muscle),'<u4',4*mh[6],ro).reshape(-1,4)
    if th[3]!=mh[3] or th[-2].hex()!=reg['routes']:raise ValueError('Tendon reference mismatch')
    z=np.load(folder/'correspondence.npz');report=[];receipts=[]
    skinmesh=np.load(data.parent/'native-input.npz');from measure_forearm_enclosure import Enclosure
    skin=Enclosure(lib,np.loadtxt(data/'current-native-forearm/mesh.node',skiprows=1,usecols=(1,2,3),dtype='f4')[skinmesh['node_O']],skinmesh['face_O'])
    poses=read(data/'bed-unscaled-four/native-initial-bodies.json')['all_bodies']
    try:
        for reference in reg['fit']:
            side=reference['side'];sc='scapula_'+side;bs=reference['body'];rawrow=reference['native_row'];sid=rawrow[5]
            muscle_name='DELT3' if side=='r' else 'DELT3_l'
            p=program(data,side);seq=p['seq'];oi,wi,vi=[idx for _,idx in seq[:3]]
            x0=p['sites'][oi][1];via=p['sites'][vi][1];ss=p['wrap'][wi][3];rad=np.linalg.norm(x0-via)
            if any(np.count_nonzero((route[:,0]==1)&(route[:,1]==u))+np.count_nonzero(route[:,2]==u)!=1 for u in (oi,ss)) or np.count_nonzero((route[:,0]==2)&(route[:,1]==wi))!=1:
                raise ValueError('Attempted to move a shared route landmark')
            V,F=z[sc+'_v'],z[sc+'_f'];target=z[sc+'_cortical_prediction'][0]
            faces,points=cortical_circle(V,F,via,rad,target)
            atlas=R.from_quat(rawrow[9:13]).inv().apply(points-rawrow[6:9])/rawrow[13]
            _,dist,_=nearest((z[sc+'_spinal_muscle'],z[sc+'_spinal_muscle_f']),atlas,lib)
            pose=poses[bs]['current'];world=lambda a:R.from_quat(pose['orientation_world_xyzw']).apply(a)+pose['position_world_m']
            inside,depth=skin.query(world(points))
            ok=(dist<=.002)&(inside==1)&(depth>.001)
            if not ok.any():raise ValueError('No anatomy-supported equal-length attachment')
            i=int(np.argmin(np.where(ok,np.linalg.norm(points-target,axis=1),np.inf)))
            point=points[i].astype('f4').astype(float)
            u=(x0-via)/rad;v=(point-via)/np.linalg.norm(point-via)
            axis=np.cross(u,v);angle=np.arctan2(np.linalg.norm(axis),u@v)
            rotation=R.from_rotvec(axis/np.linalg.norm(axis)*angle)
            def convert(x):return via+rotation.apply(x-via)
            unchanged_wi=p['wrap'][wi];offset=wo+64*wi
            geom=struct.unpack_from('<2I14f',muscle,offset)
            rotated=rotation.as_matrix()@np.array(geom[7:]).reshape(3,3)
            struct.pack_into('<2I14f',muscle,offset,*geom[:4],*convert(unchanged_wi[1]),*rotated.ravel())
            for node,pos in ((oi,point),(ss,convert(p['sites'][ss][1]))):
                struct.pack_into('<I3f',muscle,76+16*node,bs,*pos)
            surf=dict(body_index=bs,stable_id=sid,member_id=('FJ3384' if side=='r' else 'FJ3279'),vertices=V.tolist(),triangles=F.tolist())
            settings=meta['admission']
            e,why=owner._numi_human_tendon_surface_envelope_connected(point.tolist(),surf,settings['maximum_surface_distance_m'],
                settings['maximum_patch_radius_m'],settings['maximum_sampled_total_force_amplification'])
            if not e:raise ValueError((muscle_name,why))
            new_index=th[6];th[6]+=1
            mi=212 if side=='r' else 275
            ep_index,tag=next((j,k)for j,k in enumerate(meta['endpoints']) if k['muscle']==muscle_name and k['endpoint']=='origin')
            ep=struct.unpack_from('<8I8f',t,144+64*ep_index)
            if ep[:2]!=(mi,0) or ep[3:6]!=(oi,bs,0):raise ValueError('Changed source endpoint')
            struct.pack_into('<8I8f',t,144+64*ep_index,*ep[:5],2,new_index,sid,*point,e['surface_distance_m'],
                e['sampled_total_force_amplification'],e['patch_radius_m'],e['moment_residual_m'],0.)
            t.extend(pack_envelope(e,bs))
            label,sd=skin.query(world(np.array(e['node_local_points_m'])))
            if (label!=1).any():raise ValueError('New force nodes escape physical skin')
            tag.update(source_local_point_m=point.tolist(),resolved_local_point_m=point.tolist(),attachment_mode='registered_bone_distributed_envelope',
                admission_reason='admitted',surface={k:v for k,v in e.items() if k not in ('force_maps','resolved_local_point_m')})
            report.append(dict(muscle=muscle_name,body=bs,site=oi,via=vi,wrap=wi,wrap_side_site=ss,
                cortical_source_face=int(faces[i]),atlas_spinal_deltoid_clearance_mm=1000*dist[i],origin_depth_mm=1000*depth[i],
                node_depth_mm=(1000*sd).tolist(),reference_origin_m=x0.tolist(),origin_m=point.tolist(),
                initial_source_cortex_mm=reference['baseline_mm'],source_to_target_displacement_mm=1000*np.linalg.norm(point-x0),
                current_cortex_distance_mm=1000*e['surface_distance_m'],wrap_rotation_xyzw=rotation.as_quat().tolist(),
                invariant_span_error_m=abs(np.linalg.norm(point-via)-rad),force_error=e['force_residual'],moment_error_m=e['moment_residual_m'],
                amplification=e['sampled_total_force_amplification'],patch_radius_m=e['patch_radius_m']))
    finally:skin.close()
    out.mkdir(exist_ok=True);mp=out/myo.name;tp=out/tendon.name
    mp.write_bytes(muscle)
    th[-2]=bytes.fromhex(digest(mp));struct.pack_into('<8s10I32s32s32s',t,0,*th)
    tp.write_bytes(t);meta['payload'].update(sha256=digest(tp),bytes=len(t))
    coverage=meta['coverage']
    coverage['registered_bone_distributed_envelope_count']=th[6]
    coverage['distributed_surface_envelope_count']=th[6]
    coverage['source_site_point_fallback_count']=th[5]-th[6]
    reasons=meta['admission']['rejection_counts']
    reasons['surface_distance_exceeds_gate']-=len(report)
    if sum(reasons.values())!=coverage['source_site_point_fallback_count']:raise ValueError('Incorrect point attachment accounting')
    coverage['surface_coverage_fraction']=th[6]/th[5]
    if th[6]!=old_count+2:raise ValueError('Wrong number of new material attachments')
    methods={'connected_geodesic_compass_vertices':'compass_vertex_envelope_count',
             'connected_geodesic_topology_aware_exact_surface_points':'topology_aware_exact_surface_envelope_count'}
    for tag in meta['endpoints']:
        if tag['muscle'] in ('DELT3','DELT3_l') and tag['endpoint']=='origin':
            method=tag['surface']['surface_patch_method']
            coverage[methods[method]]+=1

    meta['source']['myosim_muscle_payload_sha256']=digest(mp)
    rc=dict(parent_muscle=reg['routes'],parent_tendon=reg['tendon'],source_bone=reg['bone'],source_spinal_deltoids=reg['muscle'],
        muscle=digest(mp),tendon=digest(tp),original_envelopes=old_count,registered_envelopes=th[6],bilateral=report)
    meta['source']['regional_spinal_deltoid_registration']=rc
    (out/'registration.json').write_text(json.dumps(rc,indent=2)+'\n')
    tp.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    for row in report:print(row['muscle'],row['origin_depth_mm'],row['atlas_spinal_deltoid_clearance_mm'],row['invariant_span_error_m'],flush=True)
if __name__=='__main__':register(*(Path(p).resolve()for p in sys.argv[1:]))
