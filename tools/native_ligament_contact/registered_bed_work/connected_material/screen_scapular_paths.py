"""Compare wrapped MyoSim tendon excursions under the proposed body-local registration."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from run_bed_work import digest


def screen(root,fit_path,archive):
    from myo_sim.build.compose import build_model
    import mujoco
    proof=json.loads((root/'material-source/scapular-frame.json').read_text());fits=json.loads(fit_path.read_text())
    if digest(archive)!=proof['source_archive'] or mujoco.__version__!='3.12.0':raise ValueError('Different source model')
    models=[build_model('myofullbody') for _ in range(2)];base,trial=models
    if (base.nq,base.nv,base.nu,base.ntendon)!=(129,128,416,424):raise ValueError('Wrong model identity')
    wrap_types=(int(mujoco.mjtWrap.mjWRAP_CYLINDER),int(mujoco.mjtWrap.mjWRAP_SPHERE))
    wrap_ids=set(int(i) for i in base.wrap_objid[np.isin(base.wrap_type,wrap_types)])
    transformed=[]
    for s in proof['bilateral']:
        fit,=[r for r in fits if r['bone']==s['source']]
        if s['body']!=fit['body'] or fit['scale']!=1:raise ValueError('Nonrigid or mismatched source fit')
        body=base.body(s['source']).id
        A=np.array(s['native_frame_row_rotation']);t=np.array(s['native_frame_translation_m'])
        Q=Rotation.from_quat(fit['rotation_xyzw']).inv();shift=np.array(fit['translation_m'])
        rot=Rotation.from_matrix(A) * Q * Rotation.from_matrix(A.T)
        origin=(Q.apply(t-shift)-t)@A.T
        sites=np.flatnonzero(base.site_bodyid==body)
        for i in sites:trial.site_pos[i]=rot.apply(base.site_pos[i])+origin
        wraps=sorted(i for i in wrap_ids if base.geom_bodyid[i]==body)
        for i in wraps:
            trial.geom_pos[i]=rot.apply(base.geom_pos[i])+origin
            q=base.geom_quat[i]
            changed=(rot*Rotation.from_quat(q[[1,2,3,0]])).as_quat()
            trial.geom_quat[i]=changed[[3,0,1,2]]
        if not len(sites) or not len(wraps):raise ValueError('Missing source routing domain')
        transformed.append(dict(body=s['source'],sites=len(sites),wraps=len(wraps)))
    ds=[mujoco.MjData(m) for m in models]
    # Traverse the source polynomial equality graph for both held-out poses and derivatives.
    dependencies={}
    for i in range(base.neq):
        if not base.eq_active0[i]:continue
        if base.eq_type[i]!=int(mujoco.mjtEq.mjEQ_JOINT):raise ValueError('Unsupported source equality')
        child=int(base.jnt_qposadr[base.eq_obj1id[i]])
        parent=int(base.jnt_qposadr[base.eq_obj2id[i]]) if base.eq_obj2id[i]>=0 else None
        if child in dependencies:raise ValueError('Duplicate dependent coordinate')
        dependencies[child]=(parent,base.eq_data[i,:5].copy())
    order=[];remaining=dict(dependencies)
    while remaining:
        ready=[k for k,(parent,_) in remaining.items() if parent not in remaining]
        if not ready:raise ValueError('Cyclic or inconsistent source equalities')
        for k in ready:
            order.append((k,*remaining.pop(k)))
    def project(q):
        q=q.copy()
        for k,parent,poly in order:
            x=q[parent]-base.qpos0[parent] if parent is not None else 0.
            q[k]=base.qpos0[k]+np.polynomial.polynomial.polyval(x,poly)
        return q
    def evaluate(j,q):
        m=models[j];d=ds[j];d.qpos[:]=project(q)
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_tendon(m,d)
        return d.ten_length.copy()
    cases=[[.10,0.,0.],[.25,0.,0.],[.10,.12,0.],[.10,0.,.12],[.35,.15,-.15]]
    tol=.001;measures=[];affected=set()
    for angles in cases:
        q=base.qpos0.copy()
        dofs=[]
        for side in ('r','l'):
            for name,angle in zip(('shoulder_elv_','elv_angle_','shoulder_rot_'),angles):
                joint=base.joint(name+side);address=int(joint.qposadr[0])
                if address in dependencies:raise ValueError('Perturbed dependent shoulder coordinate')
                if not joint.range[0]<angle<joint.range[1]:raise ValueError('Invalid held-out joint pose')
                q[address]=angle;dofs.append((name+side,address))
        ds[0].qpos[:]=project(q)
        mujoco.mj_fwdPosition(base,ds[0])
        eq=ds[0].efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        eq_residual=float(np.max(abs(ds[0].efc_pos[eq])))
        if int(sum(eq))!=len(order) or eq_residual>1e-8:raise ValueError(('Off-manifold test pose',eq_residual))
        lengths=[evaluate(j,q) for j in (0,1)];diff=np.abs(lengths[1]-lengths[0]);idx=int(diff.argmax())
        if not np.isfinite(diff).all():raise ValueError('Invalid source route')
        affected.update(np.flatnonzero(diff>1e-7).tolist())
        record=dict(angles_rad=angles,source_equality_rows=len(order),maximum_equality_residual=eq_residual,peak_length_change_m=float(diff[idx]),length_witness=base.tendon(idx).name,
                    base_length_m=float(lengths[0][idx]),candidate_length_m=float(lengths[1][idx]))
        signs=[];peak=0.;derivative_noise=0.
        for name,address in dofs:
            deriv=[]
            for j in (0,1):
                estimates=[]
                for h in (2e-5,1e-5):
                    left=q.copy();right=q.copy();left[address]-=h;right[address]+=h
                    estimates.append(-(evaluate(j,right)-evaluate(j,left))/(2*h))
                if not all(np.isfinite(x).all() for x in estimates):raise ValueError('Nonfinite path derivative')
                derivative_noise=max(derivative_noise,float(np.max(abs(estimates[1]-estimates[0]))))
                deriv.append(estimates)
            a,b=[x[-1] for x in deriv];change=abs(a-b);i=int(change.argmax())
            if change[i]>peak:
                peak=float(change[i]);record['moment_witness']=dict(joint=name,tendon=base.tendon(i).name,baseline_m=float(a[i]),candidate_m=float(b[i]))
            for i in np.flatnonzero((a*b<0)&(abs(a)>tol)&(abs(b)>tol)&(deriv[0][0]*a>0)&(deriv[1][0]*b>0)&(abs(deriv[0][0]-a)<1e-5)&(abs(deriv[1][0]-b)<1e-5)):
                signs.append(dict(joint=name,tendon=base.tendon(i).name,baseline_m=float(a[i]),candidate_m=float(b[i])))
        record.update(peak_moment_arm_change_m=peak,opposite_direction_pairs=signs,peak_step_halving_change_m=derivative_noise)
        measures.append(record)
    return dict(source=proof['source_archive'],rigid_fit=digest(fit_path),source_frame=digest(root/'material-source/scapular-frame.json'),mujoco=mujoco.__version__,
                transformed=transformed,changed_tendons=len(affected),cases=measures,
                scope='Offline native-source kinematic comparison, five source-equality-projected joint poses. The rigid map was applied to scapula sites and actual tendon wrapping geometries together. Central differences evaluate complete wrapped tendon lengths. No dynamics were advanced or applied to the Apple GPU scene.')
if __name__=='__main__':
    p=Path(__file__).resolve().parent
    (p/'material-source/scapular-paths.json').write_text(json.dumps(screen(p,Path(sys.argv[1]),Path(sys.argv[2])),indent=2,allow_nan=False)+'\n')
