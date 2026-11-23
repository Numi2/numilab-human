"""Condition the inferred deep-fat boundary locally, preserving the material axes and cell mass."""
import csv
import json
import re
from itertools import product
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack, eye, csr_matrix
from fit_dorsal_material import inputs
from registered_density import density_bounds
from register_material_surfaces import sha
from fit_posterior_equilibrium import materials
from supported_material_admission import certify, stress_limits


def scores(X,U,mass,density):
    edge = X[:,1:]-X[:,:1]
    volume = np.linalg.det(edge)/6
    # The perturbation acts only on reference coordinates. The measured relative
    # displacements are transported, never fed back into the native dynamics.
    Y=X+U
    dy=Y[:,1:]-Y[:,:1]
    v=np.linalg.det(dy)/6
    shape=lambda p,vol: 6*vol/np.maximum.reduce([np.linalg.norm(p[:,i]-p[:,j],axis=1) for i in range(4) for j in range(i)])**3
    s=np.zeros((len(X),10))
    s[:,0]=volume/1e-18;s[:,1]=shape(X,volume)/1e-5
    s[:,2]=v/1e-18;s[:,3]=shape(Y,v)/1e-5
    s[:,4]=volume*density[:,1]/mass
    s[:,5]=mass/(density[:,0]*np.maximum(volume,1e-30))
    good=(volume>0)&(v>0)
    if good.any():
        F=np.linalg.solve(edge[good],dy[good])
        stretches=np.linalg.svd(F,compute_uv=False)
        j=v[good]/volume[good]
        s[good,6]=j/.4;s[good,7]=2.5/j
        s[good,8]=stretches[:,-1]*1.5;s[good,9]=1.5/stretches[:,0]
    return s


def propose(trial,mesh,build,witnesses,dest,reroute=False,second=None):
    cfg=json.loads((trial/'launch.json').read_bytes())
    x,y,t,role,pins=inputs(trial)
    if not np.array_equal(x,y):raise ValueError('Expected an unstressed reference at release')
    X=x.astype(float)
    e=cfg['environment'];info=json.loads(build.read_bytes())
    header=Path(info['source_header']['path'])
    if sha(header)!=info['source_header']['sha256']:raise ValueError('Stale attachment registration')
    anchors={int(s) for s in re.findall(r'\{\d+u,(\d+)u,',header.read_text().split('constexpr Anchor anchors[]={',1)[1].split('};',1)[0])}
    if len(anchors)!=14:raise ValueError('Expected all attachment positions')
    rec=json.loads((mesh/'receipt.json').read_bytes())
    if sha(mesh/'cell-mass.f64')!=rec['cell_mass_sha256'] or pins['elements_sha256']!=sha(mesh/'mesh.ele'):
        raise ValueError('Unmatched physical mass')
    masses=np.fromfile(mesh/'cell-mass.f64',dtype='<f8')
    p=trial/'physical-accepted.node'
    observed=np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    w=json.loads(witnesses.read_bytes())
    if sha(p)!=w['accepted_pose']:raise ValueError('Stale endpoint')
    U=observed-X
    secondary=None
    if second is not None:
        if sha(second/'physical-reference.node')!=sha(trial/'physical-reference.node'):
            raise ValueError('Unmatched material release')
        secondary=np.loadtxt(second/'physical-accepted.node',skiprows=1,usecols=(1,2,3))-X
    bad=np.array([r['cell'] for r in w['failures']])
    if secondary is not None:
        alternate=json.loads((second/'cell-witnesses.json').read_bytes())
        if sha(second/'physical-accepted.node')!=alternate['accepted_pose']:raise ValueError('Stale secondary witnesses')
        bad=np.union1d(bad,[r['cell'] for r in alternate['failures']])
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_'
    bind=Path(e[prefix+'CSV'])
    if sha(bind)!=e[prefix+'SHA256']:raise ValueError('Wrong node identities')
    deep=np.zeros(len(X),bool)
    with bind.open() as f:
        for row in csv.DictReader(f):
            if row['registered_surface_layer'] in ('D','I'):deep[int(row['joint_node_id'])-1]=True
    deep[list(anchors)]=False
    owner=Path(cfg['compiled_material_owner'])
    ranges=np.array([density_bounds(owner,m) for m in (15,16)])
    seed=np.unique(t[bad]);star=np.any(np.isin(t,seed),axis=1);free=np.unique(t[star]);free=free[deep[free]]
    Cid=np.flatnonzero(np.any(np.isin(t,free),axis=1))
    C=t[Cid];m=masses[Cid];bounds=ranges[role[Cid]-15];u=U[C]
    def evaluate(pts,selected=None):
        if selected is None:selected=slice(None)
        primary=scores(pts,u[selected],m[selected],bounds[selected])
        if secondary is None:return primary
        return np.column_stack((primary,scores(pts,secondary[C[selected]],m[selected],bounds[selected])[:,[2,3,6,7,8,9]]))
    original=X.copy()
    col=np.full(len(X),-1,dtype=int);col[free]=np.arange(len(free))
    nv=len(free)*3
    history=[]
    baseline=evaluate(X[C])
    # Some emitted densities can lie on a physical limit up to rounding.
    # Lock any existing roundoff discrepancy; the candidate cannot enlarge it.
    floor=np.ones_like(baseline)
    floor[:,4:6]=np.minimum(floor[:,4:6],baseline[:,4:6])
    floor[:,4:6]-=1e-10
    if reroute:
        floor[role[Cid]==16,4:6]=0.
    def rank(s):
        violation=np.minimum(np.divide(s,floor,out=np.ones_like(s),where=floor>0),1.)
        return (int(np.sum(np.min(violation,axis=1)<1)),float(np.sum((1-violation)**2)),float(-violation.min()))
    for turn in range(25):
        pts=X[C];s=evaluate(pts)
        current=rank(s)
        print(turn,current,flush=True)
        if current[0]==0:break
        near=(s<1.5)&(floor>0)
        # Preserve each already valid inequality, encourage failed ones to approach
        # its physical threshold, and minimize the largest source displacement.
        i,metric=np.where(near)
        # Local central differences in micrometres; output is checked at exact Float32.
        coeff=np.zeros((len(i),4,3))
        for c in range(4):
            for k in range(3):
                h=1e-9
                plus=pts.copy();minus=pts.copy()
                plus[:,c,k]+=h;minus[:,c,k]-=h
                coeff[:,c,k]=(evaluate(plus)[i,metric]-evaluate(minus)[i,metric])/(2*h*1e6)
        vertex=col[C[i]]
        rr,cc,kk=np.where(np.broadcast_to(vertex[:,:,None]>=0,coeff.shape))
        A=coo_matrix((-coeff[rr,cc,kk],(rr,3*vertex[rr,cc]+kk)),shape=(len(i),nv)).tocsr()
        old=s[i,metric];target=floor[i,metric];g=np.maximum(0,target-old)
        failed=np.flatnonzero(g>0)
        # A separate progress variable lets a solvable witness improve even if a
        # neighboring source constraint blocks another. Penalize source movement.
        nfail=len(failed)
        improvement=coo_matrix((g[failed]*1.08,(failed,np.arange(nfail))),shape=(len(i),nfail))
        A=hstack((A,improvement,csr_matrix((len(i),nv))))
        unit=eye(nv);blank=csr_matrix((nv,nfail))
        A=vstack((A,hstack((unit,blank,-unit)),hstack((-unit,blank,-unit))),format='csr')
        best=None
        for trust in (.25,1.,4.,16.):
            cost=np.r_[np.zeros(nv),-np.ones(nfail),np.full(nv,1e-4/trust)]
            fit=linprog(cost,A_ub=A,b_ub=np.r_[np.maximum(0.,old-target),np.zeros(2*nv)],bounds=[(-trust,trust)]*nv+[(0.,1.)]*nfail+[(0.,trust)]*nv,method='highs')
            if not fit.success or fit.fun>=-1e-7:continue
            d=fit.x[:nv].reshape(-1,3)*1e-6
            for scale in (1.,.5,.25,.125):
                moved=(X[free]+scale*d).astype('f4').astype(float)
                if np.max(np.linalg.norm(moved-original[free],axis=1))>50e-6:continue
                prev=X[free].copy();X[free]=moved
                score=evaluate(X[C]);X[free]=prev
                if np.any(score[:,4:6]<floor[:,4:6]) or np.any(score[:,:2]<1):continue
                key=rank(score)
                if key>=current or best is not None and key>=best[0]:continue
                best=key,moved
        if best is None:break
        X[free]=best[1];history.append(best[0])
    # Descend on the emitted Float32 grid after a continuous update rounds
    # away. Only score candidate vertices incident to remaining failures.
    if reroute:
        for iteration in range(40):
            all_scores=evaluate(X[C])
            current=rank(all_scores)
            if not current[0]:break
            bad_local=np.flatnonzero(np.any((all_scores<floor)&(floor>0),axis=1))
            best=None
            for node in np.intersect1d(free,np.unique(C[bad_local])):
                incident=np.flatnonzero(np.any(C==node,axis=1));local=C[incident]
                orig=X[node].astype('f4')
                step_plus=np.nextafter(orig,np.float32(np.inf))-orig
                step_minus=orig-np.nextafter(orig,np.float32(-np.inf))
                for direction in product((-1,0,1),repeat=3):
                    d=np.array(direction)
                    if not d.any():continue
                    for count in (1,4,16):
                        point=(orig+count*d*np.where(d>=0,step_plus,step_minus)).astype('f4').astype(float)
                        if np.linalg.norm(point-original[node])>50e-6:continue
                        local_x=X[local].copy()
                        local_x[local==node]=point
                        sc=evaluate(local_x,incident)
                        if np.any(sc[:,:2]<1) or np.any(sc[:,4:6]<floor[incident,4:6]):continue
                        candidate=all_scores.copy();candidate[incident]=sc
                        key=rank(candidate)
                        if key>=current or best is not None and key>=best[0]:continue
                        best=(key,node,point)
            if best is None:break
            key,node,point=best;X[node]=point;history.append(key)
            print('Float32',iteration,key,flush=True)
    changed=free[np.any(X[free]!=original[free],axis=1)]
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'changed-coordinates.npz',ids=changed,points=X[changed].astype('f4'))
    props,files=materials(owner)
    result=dict(source_pins=pins,endpoint_sha256=sha(p),free_deep_nodes=len(free),incident_cells=len(C),changed=changed.tolist(),
            max_node_change_m=float(np.linalg.norm(X[free]-original[free],axis=1).max()),baseline_density_ratio_min=baseline[:,4:6].min(axis=0).tolist(),
            density_reroute_required=reroute,preservation='identical connectivity, node identities, outer skin and attachment coordinates',history=history,
            patch_sha256=sha(dest/'changed-coordinates.npz'))
    for name,pose in [('reference',X),('transported_displacement',X+U)]:
        result[name]=certify(X,pose,t,role,props,stress_limits(files))
    if secondary is not None:
        result['second_native_sha256']=sha(second/'physical-accepted.node')
        result['subdivided_displacement']=certify(X,X+secondary,t,role,props,stress_limits(files))
    (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(len(changed),result['transported_displacement'],flush=True)


if __name__=='__main__':propose(*(Path(a).resolve() for a in sys.argv[1:6]),reroute='--reroute' in sys.argv[6:],second=Path(sys.argv[7]).resolve() if len(sys.argv)==8 else None)
