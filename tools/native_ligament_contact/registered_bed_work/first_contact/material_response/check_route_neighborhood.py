"""Check whether each registered cortical path can stay in its contact features as attachment bodies move."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,nnls
from scipy.spatial.transform import Rotation
from audit_fiber_load_bounds import read_pose
from rebind_ligament_material import banks
from certify_cortical_paths import Cortex,verify,line_length
from compile_connected_joint import digest
from switch_cortical_features import insert_cortical_contact

def run(parent,sections,routes,out,translation_mm=.25,rotation_deg=.25):
    src=next((parent/'supported-bones').glob('*.nhbones'));reference=parent/'native-ligament-material/stderr.log'
    payload=banks(src);poses=read_pose(reference)['bodyPoses'];rows={}
    trials=[('reference',np.zeros(3),np.zeros(3))]
    for i,label in enumerate('xyz'):
        axis=np.eye(3)[i]
        for sign in (-1,1):
            trials.append((f'translate_{sign}_{label}',sign*axis*translation_mm,np.zeros(3)))
            trials.append((f'rotate_{sign}_{label}',np.zeros(3),sign*axis*np.deg2rad(rotation_deg)))
    for file in sorted(routes.glob('*.json')):
        r=json.loads(file.read_text());s=json.loads((sections/file.name).read_text())
        if s['bones']!=digest(src) or s['pose']!=digest(reference):raise ValueError('Input generations differ')
        y0=np.array(r['world_m'])*1000;edges=r['edges'];b0=edges[0]['body'];other=[p['body'] for p in s['sites'] if 'body' in p]
        if len(other)!=2:raise ValueError(s['sites'])
        start_id,end_id=other
        owner={sid:b for b,sid in payload}
        # The one-millimetre envelope covers both 0.25 mm translations and the tested rotations.
        # Select owners from the full 185 source banks, including banks absent from the fitted path.
        ids=[];swept=1.
        for (body,sid),(faces,v,_) in payload.items():
            p=np.array(poses[body]);verts=Rotation.from_quat(p[3:]).apply(v)*1000+p[:3]*1000
            reach=translation_mm+np.linalg.norm(verts-p[:3]*1000,axis=1).max()*np.deg2rad(rotation_deg) if body==end_id else 0.
            if np.all(verts.max(0)+reach>=y0.min(0)-swept) and np.all(verts.min(0)-reach<=y0.max(0)+swept):ids.append(sid)
        pts={sid:Cortex(owner[sid],sid,payload[owner[sid],sid],poses[owner[sid]]) for sid in ids}
        center=np.asarray(poses[end_id][:3])*1000;seed=np.array([e['weight'] for e in edges]);probes=[]
        for name,dp,angle in trials:
            Q=Rotation.from_rotvec(angle);fwd=lambda v:Q.apply(v-center)+center+dp
            moving={sid:o for sid,o in pts.items() if o.body==end_id}
            local=dict(pts)
            for sid,o in moving.items():
                pose=np.asarray(poses[end_id]);rot=Q*Rotation.from_quat(pose[3:]);local[sid]=Cortex(end_id,sid,payload[end_id,sid],np.r_[(center+dp)*.001,rot.as_quat()])
            E=[];N=[]
            for e in edges:
                surf=local[e['stable_id']];fe,v,_=payload[e['body'],e['stable_id']]
                coord=surf.rotation.apply(v[e['source_edge']])*1000+surf.center
                E.append(coord);N.append(surf.normal[e['incident_faces']])
            E=np.array(E);du=E[:,1]-E[:,0];y=y0.copy();y[-1]=fwd(y[-1])
            if np.any(np.max(abs(y-y0),axis=0)>swept):raise ValueError('Broadphase margin exceeded')
            def fun(t):
                y[1:-1]=E[:,0]+t[:,None]*du
                L,d=line_length(y);return L,np.einsum('ij,ij->i',d,du)
            fit=minimize(fun,seed,method='L-BFGS-B',jac=True,bounds=[(0,1)]*len(seed),options={'ftol':2e-15,'gtol':2e-9,'maxiter':800,'maxls':64,'maxcor':16})
            L,g=fun(fit.x);_,d=line_length(y);res=[]
            if np.any(y<y0.min(0)-swept) or np.any(y>y0.max(0)+swept):raise ValueError('Optimized path went outside the tested envelope')
            for j,(e,t,n) in enumerate(zip(edges,fit.x,N)):
                if min(t,1-t)<1e-6:
                    o=local[e['stable_id']];v=e['source_edge'][int(t>.5)];node=np.flatnonzero(o.first==v)[0]
                    n=o.normal[(o.faces==node).any(1)]
                res.append(nnls(n.T,d[j])[1])
            crosses=verify(y,local.values());initial=list(crosses);new_feature=None;chain=edges
            if crosses:
                candidate=insert_cortical_contact(y,E,N,edges,fit.x,local,crosses)
                if candidate is not None:
                    L,err,y,chain,weights,new_feature=candidate;res=[err];crosses=verify(y,local.values())
            loaded=L*.001>r['slack_m'];ok=not crosses and (not loaded or max(res)<1e-4)
            # Envelope theorem: differentiate the minimized length against force at the departure/arrival sites.
            owners=[start_id]+[e['body'] for e in chain]+[end_id];split=[j for j in range(len(owners)-1) if owners[j]!=owners[j+1]]
            if len(split)!=1:raise ValueError('Multiple free spans')
            k=split[0];p0,p1=y[k:k+2];unit=(p1-p0)/np.linalg.norm(p1-p0)
            probes.append(dict(probe=name,loaded=bool(loaded),passes=bool(ok),solver_success=bool(fit.success),length_m=L*.001,max_reaction_residual=float(max(res)),center_world_m=((center+dp)*.001).tolist(),world_vertices_m=(y*.001).tolist(),point_owners=owners,free_span_direction=unit.tolist(),target_moment_arm_mm=np.cross(p1-center-dp,unit).tolist(),original_penetrations=initial,added_feature=new_feature,penetrations=crosses))
        rows[r['fiber']]=dict(broadphase_bones=ids,passed=sum(k['passes'] for k in probes),total=len(probes),probes=probes)
        print(r['fiber'],rows[r['fiber']]['passed'],len(probes),flush=True)
    out.write_text(json.dumps(dict(bones=digest(src),pose=digest(reference),translation_mm=translation_mm,rotation_deg=rotation_deg,trials=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:5]),*map(float,sys.argv[5:]))
