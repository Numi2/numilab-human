"""Evaluate material energy and cortical interface compatibility along a hash-bound native trajectory."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pose_glutes import parts,model,states
from material_context import sha
from prepare_material_interface import solid,read_inputs
from bind_material_ports import gram
from pelvic_specimen import properties

def frame(r):
    return np.array(r['position_m']+r['quaternion_xyzw'])
def local(points,p):
    shape=points.shape
    return Rotation.from_quat(p[3:]).inv().apply((points-p[:3]).reshape(-1,3)).reshape(shape)
def main(capture,fullcfg,base,run,projected,ports,material,out):
    spec=properties(material);out.mkdir(exist_ok=True);rc=run/'terminal.receipt.json';pr=capture/'step-8.receipt.json';trace=run/'resting-body-motion-partition.csv'
    cfg=json.loads((run/'launch.json').read_text())['argv'];asset=Path(cfg[cfg.index('--soft-tissue-payload')+1]);record=json.loads((run/'glutes/motion.json').read_text())
    if not record['recorded_pose_replay_qualified'] or (sha(rc),sha(asset),sha(trace))!=(record['receipt_sha256'],record['source_payload_sha256'],record['trace_sha256']):raise ValueError('Unverified motion')
    full={p['body_index']:frame(p) for p in json.loads(pr.read_text())['initial_anatomical_registration']['body_poses']}
    receipt=json.loads(rc.read_text());end={p['body_index']:frame(p) for p in receipt['accepted_body_poses']};last=receipt['accepted_step']
    poses=states(trace,rc)
    # Sacrum is an independent owner. Recover the same frozen COM offset convention as the femoral and pelvic frames.
    for row in csv.DictReader(trace.open()):
        if int(row['body_index'])==1:
            k=int(row['accepted_step']);poses[k][1]=np.r_[[float(row['root_'+a+'_m'])+float(row['com_relative_root_'+a+'_m']) for a in 'xyz'],[float(row['orientation_'+a]) for a in 'xyzw']]
    initial=poses[0];correction={int(k):np.array(v) for k,v in record['frame_correction_local_m'].items()}
    correction[1]=local(end[1][:3],poses[last][1]);frames=sorted(poses)
    for k in frames[1:]:
        for b,v in correction.items():poses[k][b][:3]+=Rotation.from_quat(poses[k][b][3:]).apply(v)
    for b in correction:
        np.testing.assert_allclose(poses[last][b][:3],end[b][:3],atol=2.e-7,rtol=0.)
        if abs(np.dot(poses[last][b][3:],end[b][3:]))<1.-2.e-6:raise ValueError('Donor end orientation drift')
    surfs,_=read_inputs(fullcfg,pr,capture,base);bones={}
    for name in ('bone_1','bone_2','bone_3','bone_18','bone_19'):
        x,f=surfs[name];b={'bone_1':1,'bone_2':131,'bone_3':145,'bone_18':128,'bone_19':128}[name]
        xx=local(x,full[b]);bones[name]=solid(xx,f,name)[0]
    output=dict(material_sha256=sha(material),source_registration=sha(pr),native_trajectory=sha(trace),native_terminal=sha(rc),frames=len(frames),end_s=receipt['accepted_time_s'],sides={})
    for sid in (29,30):
        p=np.load(projected/(str(sid)+'.npz'));m=np.load(ports/(str(sid)+'.npz'));v=p['vertices'].astype(float);t=p['cells'];vol=p['cell_volumes'];use=vol>0;t=t[use];vol=vol[use]
        rest=np.swapaxes(v[t[:,1:]]-v[t[:,:1]],1,2);inv=np.linalg.inv(rest)
        data=parts(asset,sid);owners=m['target_body'];names=m['target_bank'];points=p['port_vertices'];weights=p['port_shape'].mean(axis=1);node=m['muscle_nodes'];bank={}
        for name in np.unique(names):
            ix=np.flatnonzero(names==name)
            if name=='skin_deep':continue
            b=int(owners[ix[0]]);face=local(m['target_triangle'][ix].astype(float),full[b])
            n=np.cross(face[:,1]-face[:,0],face[:,2]-face[:,0])
            # The Boolean exposes the bone from the cavity side; derive outward normal from the muscle port winding.
            rest_port=local(points[ix],full[b]);outer=-np.cross(rest_port[:,1]-rest_port[:,0],rest_port[:,2]-rest_port[:,0])
            norm=np.linalg.norm(outer,axis=1);sel=norm>1.e-18
            ix=ix[sel];bank[name]=(b,ix,face[sel],outer[sel]/norm[sel,None])
        stats={name:dict(maximum_inward_plane_offset_mm=0.,witness=None,confirmed_inside_samples=[]) for name in bank}
        js=[np.inf,0.];energy=0;bad=0
        for step in frames:
            pose=poses[step];x=model(data,pose,1)[0].astype('f4').astype(float)
            if step in (0,last):
                x=np.load(run/'glutes'/('%s-%d.npz'%('entrance' if step==0 else 'accepted',sid)))['xyz'].astype(float)
            # Current P1 Jacobians use the independently registered cut material volumes.
            F=np.swapaxes(x[t[:,1:]]-x[t[:,:1]],1,2)@inv;J=np.linalg.det(F)
            js=[min(js[0],J.min()),max(js[1],J.max())];bad+=int(((J<spec["jmin"])|(J>spec["jmax"])).sum());lg=np.log(J)
            psi=.5*spec["mu"]*((F*F).sum((1,2))-3.-2*lg)+.5*spec["lam"]*lg*lg
            energy=max(energy,float(vol@psi))
            # Four scheduled checks plus the actual terminal geometry, covering early release and resting motion.
            if step not in (0,frames[1],frames[len(frames)//4],frames[len(frames)//2],last):continue
            pos=np.einsum('fk,fkj->fj',weights,x[node])
            for name,(b,ix,face,n) in bank.items():
                query=local(pos[ix],pose[b]);normal_offset=np.einsum('ij,ij->i',query-face[:,0],n)
                shape=gram(query[:,None],face)[:,0,:]
                valid=(shape.min(1)>=0)&(shape.max(1)<=1)&(normal_offset < -1e-6)
                for idx in np.flatnonzero(valid)[np.argsort(normal_offset[valid])[:4]]:
                    point=query[idx];direction=np.array([[1.,.173,.281],[-.137,.85,.319]])
                    counts=[len(bones[name].ray_cast(point,point+d)) for d in direction]
                    if not all(c%2==1 for c in counts):continue
                    off=-1000*normal_offset[idx]
                    row=dict(step=int(step),port_row=int(ix[idx]),bone_local_point_m=point.tolist(),rays=counts,inward_parent_plane_mm=off)
                    stats[name]['confirmed_inside_samples'].append(row)
                    if off>stats[name]['maximum_inward_plane_offset_mm']:stats[name].update(maximum_inward_plane_offset_mm=off,witness=row)
        output['sides'][sid]=dict(samples=len(frames),cells=len(t),J_min=js[0],J_max=js[1],volume_gate_failures=bad,max_integrated_passive_energy_J=energy,interfaces=stats)
        print(sid,js,bad,[(k,z['maximum_inward_plane_offset_mm']) for k,z in stats.items()],flush=True)
    (out/'motion.json').write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
