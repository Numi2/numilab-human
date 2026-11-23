"""Transfer both physical hands with a shared displacement field and bound the exact cell Jacobians."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,vstack,hstack,eye,csr_matrix
from register_limb_field import component,tetra_jac
from compare_supported_postures import poses
from compile_connected_joint import digest

def run(h,work,old_scene,pose,out):
    out.mkdir();root=h/'closed-native/native';x=np.loadtxt(root/'mesh.node',skiprows=1,usecols=(1,2,3))
    n=len(x);data=np.loadtxt(root/'mesh.ele',skiprows=1,dtype='i4');t=data[:,1:5]-1;mat=data[:,5];mass=np.fromfile(root/'cell-mass.f64','<f8')
    p=poses(old_scene/'native-initial-bodies.json');q=poses(pose);y=x.copy();mask=np.zeros(n,bool);stats={};parts={}
    wall=np.load(h/'cortical-registration/registered-cortical-facets.npz')['cell_side_material_bank_nodes'][:,-3:]
    pins=np.unique(np.r_[wall.ravel(),json.load(open(h/'scene-registration/registration.json'))['first_contact_new']])
    for side,hum,elb,hand,other in [('l',91,92,95,45),('r',41,42,45,95)]:
        w=q[hand,:3]-p[hand,:3];axis=p[elb,:3]-p[hum,:3];axis/=np.linalg.norm(axis)
        coord=(x-p[hum,:3])@axis;seed=np.argmin(np.linalg.norm(x-p[hand,:3],axis=1));opp=np.argmin(np.linalg.norm(x-p[other,:3],axis=1))
        for station in np.linspace(-.10,.26,38):
            arm=component(t,coord>station,seed,n)
            if arm.any() and not arm[opp] and x[arm,1].min()>p[hand,1]-.25 and not (arm&mask).any():break
        else:raise ValueError('No separate distal limb')
        ids=np.flatnonzero(arm[t].any(1));nodes,inv=np.unique(t[ids],return_inverse=True);tl=inv.reshape(-1,4)
        # Snap the rigid translation to one common binary grid. Most of the fine
        # mesh then preserves its original edge vectors exactly in world Float32.
        grid=np.spacing(abs(x[nodes]).max(0).astype('f4')).astype(float)
        D=np.rint(w/grid)*grid
        target=np.clip((coord[nodes]-station)/.18,0.,1.)
        fixed=(~arm[nodes])|(coord[nodes]<=station+.002)|np.isin(nodes,pins)
        rigid=(target>=1)&~fixed
        free=~(fixed|rigid);u=np.flatnonzero(free);values=np.zeros(len(nodes));values[rigid]=1
        active=free[tl].any(1);rows=tl[active];index=ids[active]
        J=tetra_jac(x[nodes],rows);det=np.linalg.det(J);v=det/6
        gradients=np.linalg.inv(J).transpose(0,2,1);grad=np.concatenate((-gradients.sum(1)[:,None,:],gradients),1)
        # det(F)=1+D.grad(w) exactly before emission. A single sparse linear
        # constraint enforces both orientation and the physical volume band.
        G=coo_matrix(((grad@D).ravel(),(np.repeat(np.arange(len(rows)),4),rows.ravel())),shape=(len(rows),len(nodes))).tocsr()
        g=G[:,u];c=1+G@values
        I=eye(len(u),format='csr');zero=csr_matrix(g.shape)
        A=vstack([hstack([g,zero]),hstack([-g,zero]),hstack([I,-I]),hstack([-I,-I])],format='csr')
        rho=mass[index]/v;lo=np.where(mat[index]==15,900.,850.);hi=np.where(mat[index]==15,1100.,950.)
        # Try the original mass/density band first. If the requested reference
        # configuration cannot carry that distribution, retain the construction
        # witness and solve a bounded geometry trial separately.
        # This is a deformed initial pose. The installed reference mass and
        # constitutive rest mesh stay exactly at their imported values.
        lower=np.full(len(v),.410);upper=np.full(len(v),2.45)
        rhs=np.r_[upper-c,c-lower,target[u],-target[u]]
        logs=[];best=None;edges=[(a,b) for a in range(3) for b in range(a+1,4)]
        for it in range(32):
            res=linprog(np.r_[np.zeros(len(u)),np.ones(len(u))],A_ub=A,b_ub=rhs,bounds=[(0,1)]*len(u)+[(0,None)]*len(u),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
            logs.append([it,res.message])
            if not res.success:break
            values[u]=res.x[:len(u)]
            trial=x[nodes]+values[:,None]*D
            dot=1+G@values
            constraints=[];new_bounds=[]
            for a,b in edges:
                edge=trial[rows[:,b]]-trial[rows[:,a]];length=np.linalg.norm(edge,axis=1)
                target_shape=1.0002e-5
                need=target_shape*length**3/det
                bad=np.flatnonzero(need-dot>3e-8)
                if not len(bad):continue
                # Each cubed edge length is a convex scalar function of two
                # vertex weights. Its supporting hyperplane excludes this
                # violating point without removing a valid solution.
                slope=3*target_shape*length[bad]*(edge[bad]@D)/det[bad]
                E=coo_matrix((np.r_[-slope,slope],(np.tile(np.arange(len(bad)),2),np.r_[rows[bad,a],rows[bad,b]])),shape=(len(bad),len(nodes))).tocsr()
                C=(E-G[bad])[:,u];constraints.append(hstack([C,csr_matrix(C.shape)]))
                new_bounds.append(dot[bad]-need[bad]+C@values[u])
            if not constraints:best='bounded_pose';break
            A=vstack([A,*constraints],format='csr');rhs=np.r_[rhs,np.concatenate(new_bounds)]
        if best is None:raise ValueError(f'No shape and Jacobian certified transition on {side}: {logs}')
        ideal=x[nodes]+values[:,None]*D;yy=ideal.astype('f4').astype(float);y[nodes]=yy;mask[nodes[~fixed]]=True
        J0=tetra_jac(x[nodes],tl);J1=tetra_jac(yy,tl);v0=np.linalg.det(J0)/6;v1=np.linalg.det(J1)/6;raw=np.linalg.det(tetra_jac(ideal,tl))/6
        ratio=v1/v0;stable=rigid[tl].all(1)
        parts[side+'_nodes']=nodes;parts[side+'_w']=values;parts[side+'_rigid']=rigid
        stats[side]=dict(applied_displacement_m=D.tolist(),translation_roundoff_m=float(np.linalg.norm(D-w)),nodes=int(arm.sum()),transition_dofs=len(u),rigid_cells=int(stable.sum()),transition_rows=len(index),linear_solver=logs,selected=best,nonpositive_ideal=int((raw<=0).sum()),inverted_float32=int((v1<=0).sum()),inverted_rigid=int(((v1<=0)&stable).sum()),min_ratio=float(ratio.min()),rigid_exact_volume=int((v0[stable]==v1[stable]).sum()),min_material_z=float(yy[:,2].min()))
        print(side,stats[side],flush=True)
    np.savez_compressed(out/'material-transport.npz',nodes=np.flatnonzero(mask),positions=y[mask].astype('f4'))
    np.savez_compressed(out/'regions.npz',**parts)
    (out/'result.json').write_text(json.dumps(dict(source=digest(root/'mesh.node'),pose=digest(pose),regions=stats,transport=digest(out/'material-transport.npz')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
