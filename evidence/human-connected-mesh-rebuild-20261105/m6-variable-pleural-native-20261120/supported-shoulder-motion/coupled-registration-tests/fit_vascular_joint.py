"""Constrain the shoulder-frame search by the arm and the two registered vascular interfaces."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from compare_registration import prepare
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32

def fit(launch,ref,motion,archive,frames,vascular):
    cfg,_,meshes,an,poses,default,data=prepare(launch,ref,motion,archive)
    cal=json.loads(frames.read_text()); inputs=np.load(vascular); reports=[]
    install_for_float32()
    for ss,r in zip(an['candidates'],cal):
        s,h=(data[k] for k in ss['stable_ids'])
        ulna=next(row for row in data.values() if row[0]==('FJ3286' if ss['side']==0 else 'FJ3391'))
        # Reference distal forearm site identified from the distal end of the
        # ulna on its measured principal axis, oriented away from the humerus.
        uv=ulna[2]; axis=np.linalg.svd(uv-uv.mean(axis=0),full_matrices=False)[2][0]
        if (ulna[2].mean(axis=0)-h[2].mean(axis=0))@axis<0:axis=-axis
        u=ulna[5][uv@axis>=np.quantile(uv@axis,.98)].mean(axis=0)
        pivot=np.asarray(r['reference_pivot_m'])
        old=h[6].apply(pivot)*h[7]+h[8]; new=s[6].apply(pivot)*s[7]+s[8]
        atlas_r=s[6]*h[6].inv()
        p=default[s[1]]; rp=Rotation.from_quat(p[3:])
        pairs=[]
        if ss['side']==0:
            for k in (475,477):
                pp,qq,ll=[inputs[f'{a}{k}'].astype(float) for a in ['p','q','L']]
                assert ll.shape==(len(pp),len(qq))
                pairs.append((pp,qq,ll))
        def metric(rot):
            R=Rotation.from_rotvec(rot)
            wrist=np.linalg.norm(new+R.apply(u-old)-u)
            bounds=[float(np.max(cdist(pp,new+R.apply(qq-old))/ll)) for pp,qq,ll in pairs]
            return float(wrist),bounds
        def distance_jac(x):
            return np.array(metric(x)[1])
        def objective(v):
            wrist,bounds=metric(v)
            return (wrist/.05)**2+.05*np.dot(v,v)
        guesses=[np.zeros(3),atlas_r.as_rotvec()]
        solved=[]
        for guess in guesses:
            opt=minimize(objective,guess,method='SLSQP',constraints=([dict(type='ineq',fun=lambda v:1-distance_jac(v))] if pairs else []),
                         bounds=[(-np.pi/2,np.pi/2)]*3,options=dict(ftol=1e-11,maxiter=160))
            solved.append((opt.x,bool(opt.success)))
        # Include the anatomical orientation reference and the pure translation
        # as controls. All remaining angle samples interpolate the same proper rotations.
        best=min(solved,key=lambda item:(not item[1],objective(item[0])))
        control_rot=Rotation.from_rotvec(best[0]); diff=atlas_r*control_rot.inv()
        attempts=[]
        for t in (0,.125,.25,.375,.5,.625,.75,.875,1):
            rot=Rotation.from_rotvec(diff.as_rotvec()*t)*control_rot
            wrist,bounds=metric(rot.as_rotvec())
            local_r=rp.inv()*rot*rp; local_dx=rp.inv().apply(rot.apply(p[:3]-old)+new-p[:3])
            counts={}
            for k,pose in [(0,default),*((k,poses[k]) for k in [32,256,8000,-3])]:
                sa=pose[s[1]];ha=pose[h[1]]
                rs=Rotation.from_quat(sa[3:]); sh=sa[:3]+rs.apply(s[4])
                hu=ha[:3]+Rotation.from_quat(ha[3:]).apply(h[4])
                hu=sa[:3]+rs.apply(local_r.apply(rs.inv().apply(hu-sa[:3]))+local_dx)
                counts[k]=len(intersections(sh,s[3],hu,h[3]))
            attempts.append(dict(atlas_orientation_fraction=t,wrist_displacement_m=wrist,
                necessary_fixed_material_stretches=bounds,scapula_humerus_triangle_pairs=counts,
                parent_rotation_xyzw=local_r.as_quat().tolist(),parent_translation_base_m=local_dx.tolist()))
            print(ss['side'],t,round(wrist,4),np.round(bounds,3),counts,flush=True)
        reports.append(dict(side=ss['side'],constrained_solver_converged=best[1],attempts=attempts))
    return reports
if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(fit(*args),indent=2)+'\n')
