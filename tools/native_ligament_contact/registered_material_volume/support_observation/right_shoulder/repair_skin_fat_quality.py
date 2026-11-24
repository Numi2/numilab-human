"""Fit local cavities to the same physical skin/fat at reference, opening and accepted states."""
from itertools import combinations
from pathlib import Path
import json,sys
import numpy as np
from scipy.sparse import csr_matrix
from prepare_shoulder import read
from compile_connected_joint import digest
from repair_material_cavities import fan
from repair_infraspinatus import boundary,orient
from supported_material_admission import geom
from certify_interior_motion import response
from transfer_cavities import intersection_fraction


def incident(t,n):
    slots=t.ravel()
    ids=np.repeat(np.arange(len(t),dtype=np.int32),4)
    return csr_matrix((np.ones(len(slots),dtype=np.uint8),(slots,ids)),shape=(n,len(t)))


def alternatives(fr,t,start):
    root=t[start];neighbors=set();groups=[]
    for size in (3,2):
        for key in combinations(root,size):
            ids=np.flatnonzero(np.isin(t,key).sum(1)==size)
            if size==3:neighbors.update(ids.tolist())
            if size==2:groups.append(ids)
            if len(ids)!=5-size:continue
            old=t[ids];u=np.unique(old)
            if len(u)!=5:continue
            axis=np.setdiff1d(u,key)
            n=orient(fr[0],np.array([list(axis)+list(p)for p in combinations(key,size-1)]))
            if boundary(n)==boundary(old):yield ids,n,fr,dict(type='bistellar')
    for count in range(1,len(neighbors)):
        groups.extend(np.array(sorted([start]+list(p)))for p in combinations(neighbors-{start},count))
    for ids in groups:
        if len(ids)<2:continue
        z=fan(fr,t[ids])
        if z is not None:
            _,n,xx,r=z;yield ids,n,xx,r


def trial(j,out):
    out.mkdir(exist_ok=True)
    case=j/'posterior-interface-082/native-pair';vol=j/'conditioned/witness-material/volume';witness=read(case/'material-check.json')
    cfg=read(case/'launch.json');env=cfg['environment']
    files=[vol/'mesh.node',Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']),case/'physical-accepted.node']
    if [digest(f)for f in files]!=[witness[k]for k in ('reference','loaded','physical_frame')]:raise ValueError('Mismatched physical fields')
    XYZ=[np.loadtxt(p,skiprows=1,usecols=(1,2,3)) for p in files]
    raw=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1;roles=raw[:,4]
    mass=np.fromfile(vol/'cell-mass.f64','<f8');C=incident(t,len(XYZ[0]))
    par=read(j.parent/'registration-strain.json')['materials'];edits=[];outcomes={};busy=set()
    targets=sorted(s for n in ('skin','fat') for s in witness['roles'][n]['failing_cells'])
    for q in targets:
        counts=dict(considered=0,geometric=0,mechanical=0,mass=0);outcomes[q]=counts
        ii=C[t[q]].indices;ii=np.unique(ii);ii=ii[roles[ii]==roles[q]]
        nodes,u=np.unique(t[ii],return_inverse=True);u=u.reshape(-1,4)
        pts=[p[nodes] for p in XYZ];start=np.searchsorted(ii,q)
        X,Y,Z=pts;props=par[str(roles[q])]['registered'];mu,lam=[props[k]for k in ('source_mu_pa','source_logJ_coefficient_pa')]
        if geom(Z,u[[start]])[1][0]>=1e-5:raise ValueError('Unexpected source witness')
        best=None
        for ids,fresh,frames,r in alternatives(pts,u,start):
            if busy.intersection(ii[ids]):continue
            counts['considered']+=1
            # Check after the source loader rounds new material sample coordinates.
            if r['type']!='bistellar':
                frames=[x.copy() for x in frames]
                for x in frames[:2]:x[-1]=x[-1].astype('f4')
            org=u[ids];init=[geom(x,org) for x in pts];qual=[geom(x,fresh) for x in frames]
            if any(np.min(v)<=1e-18 or np.min(s)<1e-5 for v,s in qual):continue
            if any(np.min(a/qual[0][0])<.4 or np.max(a/qual[0][0])>2.5 for a,h in qual[1:]):continue
            if any(abs(c[0].sum()-e[0].sum())>max(1e-18,c[0].sum()*1e-6) for c,e in zip(init,qual)):continue
            counts['geometric']+=1
            # Virtual work at original nodal coordinates includes the exact transpose of the new interpolation.
            mapping=np.eye(len(nodes))
            if r['type']!='bistellar':
                extra=np.zeros(len(nodes));extra[r['nodes']]=r['weights'];mapping=np.vstack((mapping,extra))
            checks=[]
            for k in (1,2):
                e0,f0,_=response(pts[0],pts[k],org,mu,lam);e1,f1,_=response(frames[0],frames[k],fresh,mu,lam)
                checks.append(dict(force_error_N=float(np.linalg.norm(mapping.T@f1-f0)),force_norm_N=float(np.linalg.norm(f0)),energy_change_J=e1-e0,energy_J=e0))
            if any(v['force_error_N']>max(1e-10,.01*v['force_norm_N']) or abs(v['energy_change_J'])>max(1e-14,.05*abs(v['energy_J']))for v in checks):continue
            counts['mechanical']+=1
            ol=init[0][0];nv=qual[0][0]
            w=np.array([[intersection_fraction(a,b,frames[0])for b in fresh]for a in org])
            if max(abs(w.sum(1)-1))>3e-9 or max(abs(w.T@ol/nv-1))>3e-9:continue
            counts['mass']+=1;weight=w/w.sum(1)[:,None];m=weight.T@mass[ii[ids]]
            old_moment=mass[ii[ids]]@(pts[0][org].mean(1)-pts[0][org[0,0]])
            new_moment=m@(frames[0][fresh].mean(1)-pts[0][org[0,0]])
            if np.linalg.norm(new_moment-old_moment)>1e-11 or abs(m.sum()-mass[ii[ids]].sum())>1e-13:continue
            new_id=len(XYZ[0])+sum(z['type']!='bistellar' for z in edits)
            ix=np.r_[nodes,new_id]
            row=dict(type=r['type'],role=int(roles[q]),witness=q,source_cells=ii[ids].tolist(),cells=ix[fresh].tolist(),
                source_minimum_shape=min(np.min(h)for a,h in init),minimum_shape=min(np.min(h)for a,h in qual),
                overlap_fractions=weight.tolist(),mass_kg=m.tolist(),first_moment_error_kg_m=float(np.linalg.norm(new_moment-old_moment)),response=checks)
            if r['type']!='bistellar':row.update(point=new_id,parent_nodes=nodes[r['nodes']].tolist(),barycentric=r['weights'],coordinates=[x[-1].tolist() for x in frames])
            if best is None or row['minimum_shape']>best['minimum_shape']:best=row
        if best:busy.update(best['source_cells']);edits.append(best)
        print(q,counts,'selected',bool(best),flush=True)
    root=dict(witness=digest(case/'material-check.json'),frames=[digest(p)for p in files],mesh=digest(vol/'mesh.ele'),cell_mass=digest(vol/'cell-mass.f64'),targets=targets,edits=edits,outcomes=outcomes)
    (out/'result.json').write_text(json.dumps(root,indent=2,allow_nan=False)+'\n')
    print('selected',len(edits),'repaired',len(set(targets)&busy),flush=True)
if __name__=='__main__':trial(*(Path(a).resolve() for a in sys.argv[1:]))
