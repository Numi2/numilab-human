"""Register both donors to one material reference and interpolate between anatomical attachment regions."""
import json,struct,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pose_glutes import parts,states,model,plane
from preserve_muscle_cells import det
from compare_hand_interfaces import query
from material_context import sha

def bound_weights(x,f,sites,depth,band,existing):
    # First-ring neighborhoods are fixed to the donor of each named attachment.
    nearest=cKDTree(x);regions=[]
    for tag in ('origin','insertion'):
        seeds=nearest.query(np.array([p for k,p in sites.items() if k.endswith(tag)]))[1]
        regions.append(np.unique(f[np.any(np.isin(f,seeds),axis=1)]))
    proximal=np.union1d(regions[0],np.flatnonzero(depth<band))
    distal=regions[1]
    if len(np.intersect1d(proximal,distal)):raise ValueError('Conflicting material regions')
    center=x.mean(0);shape=np.column_stack(((x-center)/.1,np.ones(len(x))))
    A=np.vstack((-shape[proximal],shape[distal]));target=np.r_[np.zeros(len(proximal)),np.ones(len(distal))]
    def cost(q):return .5*q[:3]@q[:3],np.r_[q[:3],0.]
    sol=minimize(cost,np.zeros(4),jac=True,constraints=[dict(type='ineq',fun=lambda q:A@q-target,jac=lambda q:A)],method='SLSQP',options=dict(maxiter=200,ftol=1e-12))
    if (A@sol.x-target).min()<-1e-7:raise ValueError('Attachment bands are not separable')
    u=np.clip(shape@sol.x,0.,1.);u=u*u*(3.-2.*u)
    u[proximal]=0.;u[distal]=1.
    return u,regions,dict(coefficients=sol.x.tolist(),coordinate_center_m=center.tolist(),coordinate_scale_m=.1,medial_region_m=band,origin_patch=len(regions[0]),insertion_patch=len(distal),total_pelvic_patch=len(proximal))
def run(cfg,trace,receipt,volume,exe,out):
    out.mkdir(exist_ok=True);args=json.loads(cfg.read_text())['argv'];asset=Path(args[args.index('--soft-tissue-payload')+1])
    fr=states(trace,receipt);org,axis=plane(fr,0)
    site_report=json.loads((volume/'binding-parity.json').read_text())
    results={};data_set={};inside={}
    for sid in (29,30):
        data=parts(asset,sid);src,weights,donors,faces,aff=data;side=-1 if sid==29 else 1
        X=model(data,fr[0],side)[0];depth=side*(X-org)@axis
        R=[];T=[]
        for j,b in enumerate(donors):
            p=fr[0][b];rot=Rotation.from_quat(p[3:]).as_matrix()
            R.append(rot@aff[j][0]);T.append(p[:3]+rot@aff[j][1])
        # Keep the atlas-to-pelvis transform. Both donor frames must reconstruct
        # the same registered material point before their dynamic rotations.
        neutral=np.linalg.solve(R[0],(X-T[0]).T).T.astype('f4')
        scale=float(np.linalg.det(R[0])**(1/3));p1=fr[0][donors[1]];rot1=Rotation.from_quat(p1[3:]).as_matrix()
        t1=rot1.T@(T[0]-p1[:3]);q1=Rotation.from_matrix(rot1.T@R[0]/scale).as_quat()
        bind=np.array([*t1,*q1,scale],dtype='f4')
        new_aff=[aff[0],(Rotation.from_quat(bind[3:7]).as_matrix()*bind[7],bind[:3])]
        tets=np.loadtxt(volume/str(sid)/'source.1.ele',skiprows=1,dtype=int)[:,1:5]
        reference=np.loadtxt(volume/str(sid)/'source.1.node',skiprows=1)[:,1:4];den=det(reference,tets)
        for band in (.008,.015):
            w,regions,reg=bound_weights(X,faces,site_report[str(sid)]['actuator_sites_world_m'],depth,band,weights)
            label='band%d'%round(band*1000);ww=np.column_stack((1-w,w)).astype('f4')
            d=(neutral,ww,donors,faces,new_aff)
            shot=np.linalg.norm(X-model(d,fr[0],side)[0],axis=1).max()
            frames={k:model(d,pose,side)[0].astype('f4') for k,pose in fr.items()}
            rr=np.array([det(x.astype(float),tets)/den for x in frames.values()])
            stats=dict(maximum_entrance_error_m=float(shot),inversion_count_max_per_sample=int((rr<=0).sum(1).max()),ever_inverted_cells=int(np.any(rr<=0,axis=0).sum()),minimum_determinant_ratio=float(rr.min()),registration=reg)
            results.setdefault(label,{})[sid]=stats;data_set.setdefault(label,{})[sid]=(frames,faces,neutral,ww,bind,X)
            np.savez_compressed(out/f'{sid}-{label}.npz',vertices=neutral,weights=ww,secondary_binding=bind,origins=regions[0],insertions=regions[1])
    for label,sides in data_set.items():
        X,f,source,*_=sides[29];Y,g,other,*_=sides[30]
        both=np.vstack((f,g+len(source)));N=len(f);mask=np.ones(len(both),bool)
        for k in (0,32,256,512,1024,2048,4096,8000):
            hits=query(exe,np.vstack((X[k],Y[k])),both,mask)
            results[label].setdefault('sampled_crossings',{})[k]=[sum(a<N<=b for a,b in hits),sum(b<N for a,b in hits),sum(a>=N for a,b in hits)]
    result=dict(source=sha(asset),native_pose_trace=sha(trace),reference_receipt=sha(receipt),results=results)
    (out/'study.json').write_text(json.dumps(result,indent=2)+'\n')
    print({l:[z[s]['ever_inverted_cells'] for s in (29,30)] for l,z in results.items()})
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
