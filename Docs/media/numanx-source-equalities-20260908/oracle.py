"""Pinned MuJoCo row oracle and independent FP64 Schur-elimination check.
No rollout, native solver, physical outcome, or performance qualification.
"""
from pathlib import Path
import hashlib,json,struct
import numpy as np
import mujoco
from myo_sim.build.compose import build_model

BASE=Path(__file__).resolve().parent
raw=(BASE/'myosim-fullbody-joint-equalities-v2.nheq').read_bytes()
head=struct.unpack_from('<8s10I32s',raw)
rows=[struct.unpack_from('<4I24f',raw,80+112*i) for i in range(head[4])]
export=json.loads((BASE/'myosim-export-nheq2.json').read_text())
source_rows=export['joint_equalities']
m=build_model('myofullbody');m.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
assert mujoco.__version__=='3.12.0' and int(m.opt.integrator)==0
# Promote canonical packed records into the source engine's exact row inputs.
# Inertias/dynamics are not compared or changed, and no simulation is stepped.
core_to_source_v={i:i for i in range(6)}
for r,s in zip(rows,source_rows):
    ei=s['id'];dep=s['dependent_joint'];master=s['master_joint']
    m.eq_data[ei,:5]=r[6:11];m.eq_solref[ei]=r[12:14];m.eq_solimp[ei]=r[16:21]
    m.qpos0[m.jnt_qposadr[dep]]=r[4]
    m.dof_invweight0[m.jnt_dofadr[dep]]=r[24]
    core_to_source_v[r[1]]=int(m.jnt_dofadr[dep])
    if master>=0:
        m.qpos0[m.jnt_qposadr[master]]=r[5]
        m.dof_invweight0[m.jnt_dofadr[master]]=r[25]
        core_to_source_v[r[3]]=int(m.jnt_dofadr[master])

def mm(a,b):
    # Deterministic NumPy scalar contraction avoids Accelerate BLAS floating-status
    # warnings observed after this MuJoCo process despite finite matrix results.
    return np.einsum('ik,kj->ij' if b.ndim==2 else 'ij,j->i', a, b, optimize=False)

rng=np.random.default_rng(190807)
max_errors={'position':0.0,'jacobian':0.0,'reference_acceleration':0.0,'regularizer':0.0}
last=None
for case in range(4):
    d=mujoco.MjData(m);d.qpos[:]=m.qpos0
    if case:
        for s in source_rows:
            for joint in [s['dependent_joint'],s['master_joint']]:
                if joint>=0:
                    d.qpos[m.jnt_qposadr[joint]]+=rng.uniform(-.4,.4)
        d.qvel[:]=rng.normal(0,.2,m.nv)
    m.opt.timestep=0.0001 if case!=3 else 0.02 # exercise refsafe time constant clamp
    mujoco.mj_forward(m,d)
    J=np.zeros((51,128));pos=np.zeros(51);R=np.zeros(51);aref=np.zeros(51)
    for i,(r,s) in enumerate(zip(rows,source_rows)):
        dep=s['dependent_joint'];master=s['master_joint'];yi=int(m.jnt_qposadr[dep]);vi=int(m.jnt_dofadr[dep]);
        x=d.qpos[m.jnt_qposadr[master]]-r[5] if master>=0 else 0.0
        p=sum(r[6+k]*x**k for k in range(5));der=sum(k*r[6+k]*x**(k-1) for k in range(1,5)) if master>=0 else 0.0
        phi=d.qpos[yi]-r[4]-p;J[i,r[1]]=1
        jv=d.qvel[vi]
        if master>=0:
            J[i,r[3]]=-der;jv-=der*d.qvel[m.jnt_dofadr[master]]
        imp=min(max(r[16],.0001),.9999);dw=min(max(r[17],.0001),.9999)
        t=max(r[12],2*m.opt.timestep);B=2/(dw*t);K=1/(dw*dw*t*t*r[13]*r[13])
        pos[i]=phi;aref[i]=-B*jv-K*imp*phi;R[i]=max(1e-15,(1-imp)/imp*(r[24]+r[25]))
        erow=int(np.flatnonzero((d.efc_type==0)&(d.efc_id==s['id']))[0])
        source_J=d.efc_J.reshape(d.nefc,m.nv)[erow]
        expected_J=np.zeros(m.nv)
        for cv,sv in core_to_source_v.items():expected_J[sv]=J[i,cv]
        max_errors['position']=max(max_errors['position'],abs(phi-d.efc_pos[erow]))
        max_errors['reference_acceleration']=max(max_errors['reference_acceleration'],abs(aref[i]-d.efc_aref[erow]))
        max_errors['regularizer']=max(max_errors['regularizer'],abs(R[i]-d.efc_R[erow]))
        max_errors['jacobian']=max(max_errors['jacobian'],np.max(np.abs(expected_J-source_J)))
    if case==0: initial_max_defect=float(np.max(np.abs(pos)))
    last=(J,R,aref)

J,R,aref=last
# Coupled Schur check uses an independent generic SPD A0 and a nonzero external impulse.
# Compare the full179x179 dual KKT solve with eliminated128x128 primal solve.
C=rng.normal(size=(128,128));A=mm(C.T,C)/128+np.diag(np.geomspace(.03,3,128))
v0=rng.normal(0,.1,128);vfree=v0+rng.normal(0,.02,128);F=rng.normal(0,.05,128)
h=1e-4;b=mm(J,v0)+h*aref
KKT=np.block([[A,J.T],[J,-np.diag(R)]])
sol=np.linalg.solve(KKT,np.r_[F,b-mm(J,vfree)])
B=A+mm(J.T,J/R[:,None]);rhs=F-mm(J.T,(mm(J,vfree)-b)/R)
dv=np.linalg.solve(B,rhs);lam=(mm(J,vfree+dv)-b)/R
rv=F-mm(A,dv)-mm(J.T,lam);rl=b-mm(J,vfree+dv)+R*lam
p=rng.normal(size=128);eps=1e-6
def residual(z):return F-mm(A,z)-mm(J.T,(mm(J,vfree+z)-b)/R)
fd=-(residual(dv+eps*p)-residual(dv-eps*p))/(2*eps)
result={
 'schema':'numi.human.nheq2-fp64-oracle.v1','scope':'CPU row semantics and algebra only; no native or physical outcome proof',
 'mujoco_version':mujoco.__version__,'payload_sha256':hashlib.sha256(raw).hexdigest(),
 'canonical_fp32_inputs_promoted_to_mujoco_fp64':True,'cases':4,'rows_per_case':51,
 'source_row_max_absolute_errors':{k:float(v) for k,v in max_errors.items()},
 'initial_max_position_defect':initial_max_defect,
 'eliminated_vs_full_kkt_max_velocity_error':float(np.max(abs(dv-sol[:128]))),
 'eliminated_vs_full_kkt_max_dual_error':float(np.max(abs(lam-sol[128:]))),
 'primal_residual_inf':float(np.max(abs(rv))),'dual_residual_inf':float(np.max(abs(rl))),
 'operator_central_difference_relative_error':float(np.linalg.norm(fd-mm(B,p))/np.linalg.norm(mm(B,p))),
 'J_rank':int(np.linalg.matrix_rank(J)),'reduced_nullity':128-int(np.linalg.matrix_rank(J)),
 'R_min':float(min(R)),'R_max':float(max(R)),
 'source_urls':['https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_core_constraint.c'],
}
assert max_errors['reference_acceleration']<1e-9 and max_errors['position']<1e-12 and max_errors['jacobian']<1e-12 and max_errors['regularizer']<1e-12
assert result['eliminated_vs_full_kkt_max_velocity_error']<1e-10
assert result['primal_residual_inf']<1e-9
assert result['operator_central_difference_relative_error']<1e-8
(BASE/'oracle-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
