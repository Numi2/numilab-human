#!/usr/bin/env python3
"""Bounded joint-coordinate fallback for stalled 2-4-node quality patches.

This proposes local Float32 point coordinates only. Callers still run full source,
all185, boundary/flux, volume, cross-material and native admission audits.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, importlib.util, sys, time, traceback
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parents[1]
BUILDER=HERE/'repair_coupled_quality_points_722_v25h_unified_inner_back_ordinary.py'
BUILDER_SHA='6e285a7bc64afdaaf94b7075f0f2fb46d6f36671d058151d8b32c78e30e5ab4c'

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def _load_pinned_builder():
    if _sha(BUILDER)!=BUILDER_SHA:raise RuntimeError('pinned v25h producer changed')
    spec=importlib.util.spec_from_file_location('v25h_joint_small_patch_subject',BUILDER)
    if spec is None or spec.loader is None:raise RuntimeError('cannot import pinned v25h producer')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

_OWNER=_load_pinned_builder()
Q=_OWNER.Q

_ACCEPTED_PRIOR_STATUSES={'REJECT_NO_GLOBAL_LEX_PROGRESS','REJECT_CURRENT_OR_IMMUTABLE_GOOD_REGRESSION',
 'PATCH_SOLVER_EXCEPTION','NO_GLOBAL_LEX_PROGRESS','NODE_FALLBACK_STALLED','NO_JOINT_LINEAR_PROGRESS',
 'NODE_FALLBACK_STALLED_WITHOUT_ACCEPTANCE','REJECT_NO_LOCAL_LEX_PROGRESS','LP_NO_CANDIDATE',
 'NO_LEXICOGRAPHIC_PROGRESS','NO_COMPONENT_LEX_PROGRESS','UNCHANGED'}

def _log_deficit(qratio,vratio):
    q=np.asarray(qratio,dtype=np.float64);v=np.asarray(vratio,dtype=np.float64)
    tiny=np.finfo(np.float64).tiny
    return float(np.maximum(0.,-np.log(np.maximum(q,tiny))).sum(dtype=np.float64)
        +np.maximum(0.,-np.log(np.maximum(v,tiny))).sum(dtype=np.float64))

def _full_star(tets,point_count,free_rows):
    off,by=_OWNER._point_cell_incidence(np.asarray(tets,np.int64),int(point_count))
    return np.asarray(_OWNER._incident_cells_by_rows(np.asarray(free_rows,np.int64),off,by),np.int64)

def try_joint_small_patch_slsqp(points,tetrahedra,free_rows,seed_bad_cell_rows,
        current_passed,baseline_passed,remaining_l1_cap_by_point,prior_failures,*,
        separation_rows=None,fixed_rows=(),allowed_free_rows=None,
        incidence_offsets=None,incidence_by_node=None,maxiter=30,time_limit_s=1.5):
    """Fit 2-4 shared point coordinates jointly after the local LP and node fallback stall.

    `free_rows` are global shared-point rows. `remaining_l1_cap_by_point` is the
    remaining per-point cumulative path budget. `prior_failures` must record a
    failed coupled LP and an attempted node fallback with zero accepted nodes.
    Returns `(None, diagnostic)` on every rejected/unsupported path.
    """
    p=np.asarray(points)
    t=np.asarray(tetrahedra,np.int64)
    free=np.asarray(sorted(set(map(int,np.asarray(free_rows,np.int64).reshape(-1)))),np.int64)
    seeds=np.asarray(sorted(set(map(int,np.asarray(seed_bad_cell_rows,np.int64).reshape(-1)))),np.int64)
    cur=np.asarray(current_passed,bool);base=np.asarray(baseline_passed,bool)
    caps_all=np.asarray(remaining_l1_cap_by_point,dtype=np.float64)
    def reject(status,**kw):return None,{'schema':'h706.joint-small-patch-slsqp.v3','status':status,**kw}
    if p.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():return reject('REJECT_POINT_DOMAIN')
    if t.ndim!=2 or t.shape[1]!=4 or (len(t) and (t.min()<0 or t.max()>=len(p))):return reject('REJECT_TET_DOMAIN')
    if cur.shape!=(len(t),) or base.shape!=(len(t),):return reject('REJECT_QUALITY_MASK_SHAPE')
    if caps_all.shape!=(len(p),) or not np.isfinite(caps_all).all():return reject('REJECT_CAP_SHAPE')
    if not isinstance(prior_failures,dict) or prior_failures.get('lp_attempted') is not True or prior_failures.get('node_fallback_attempted') is not True or prior_failures.get('node_fallback_accepted',0)!=0:
        return reject('SKIP_NOT_AFTER_STALLED_LP_AND_NODE_FALLBACK')
    if prior_failures.get('lp_status') not in _ACCEPTED_PRIOR_STATUSES:return reject('SKIP_PRIOR_LP_STATUS_NOT_STALLED',prior_lp_status=prior_failures.get('lp_status'))
    if not 2<=len(free)<=4:return reject('SKIP_PATCH_SIZE_OUTSIDE_2_TO_4_FREE_NODES',free_nodes=int(len(free)))
    if not 1<=len(seeds)<=4 or (len(seeds) and (seeds.min()<0 or seeds.max()>=len(t))):return reject('SKIP_SEED_PATCH_SIZE_OUTSIDE_1_TO_4',seed_cells=int(len(seeds)))
    if len(free) and (free.min()<0 or free.max()>=len(p)):return reject('REJECT_FREE_ROW_DOMAIN')
    if np.any(caps_all[free]<=0) or np.any(caps_all[free]>5e-4):return reject('REJECT_REMAINING_CAP_OUTSIDE_POLICY',maximum_cap_m=float(caps_all[free].max(initial=0.)))
    fixed=set(map(int,fixed_rows))
    if any(r<0 or r>=len(p) for r in fixed):return reject('REJECT_FIXED_ROW_DOMAIN')
    if fixed.intersection(map(int,free)):return reject('REJECT_FREE_ROW_INTERSECTS_FIXED_SCOPE')
    if allowed_free_rows is not None and not set(map(int,free)).issubset(set(map(int,np.asarray(allowed_free_rows,np.int64)))):return reject('REJECT_FREE_ROW_OUTSIDE_AUTHORIZED_PATCH')
    if np.any(cur[seeds]):return reject('REJECT_SEED_CELL_NOT_CURRENTLY_BAD')
    seed_vertices=set(map(int,np.asarray(t[seeds],np.int64).reshape(-1)))
    if not set(map(int,free)).issubset(seed_vertices):return reject('REJECT_FREE_ROW_NOT_A_SEED_BAD_CELL_CORNER')
    free_corner_counts=np.asarray([np.count_nonzero(np.isin(t[int(r)],free)) for r in seeds],np.int64)
    if np.any((free_corner_counts<2)|(free_corner_counts>4)):
        return reject('SKIP_SEED_BAD_CELL_OUTSIDE_2_TO_4_FREE_CORNERS',free_corner_counts=free_corner_counts.tolist())
    if (incidence_offsets is None)!=(incidence_by_node is None):return reject('REJECT_INCOMPLETE_SHARED_INCIDENCE_CACHE')
    if incidence_offsets is None:
        cells=_full_star(t,len(p),free)
    else:
        io=np.asarray(incidence_offsets,np.int64);ib=np.asarray(incidence_by_node,np.int64)
        if io.shape!=(len(p)+1,) or (len(io) and (io[0]!=0 or io[-1]!=len(ib))) or np.any(np.diff(io)<0):
            return reject('REJECT_SHARED_INCIDENCE_CACHE_SHAPE')
        cells=np.asarray(_OWNER._incident_cells_by_rows(free,io,ib),np.int64)
    if not len(cells) or not set(map(int,seeds)).issubset(set(map(int,cells))):return reject('REJECT_INCOMPLETE_POINT_STAR',affected_cells=int(len(cells)))
    local_t=t[cells]
    if not np.any(~cur[cells]):return reject('SKIP_NO_BAD_CELL_IN_FULL_STAR')
    caps=caps_all[free].copy();k=len(free);m=len(cells);nvar=3*k;zdim=2*nvar
    lookup=np.full(len(p),-1,np.int64);lookup[free]=np.arange(k,dtype=np.int64)
    p64=p.astype(np.float64);x0=p64[free].copy()
    d0,q0=Q._M.measurements(p,local_t);d0=np.asarray(d0,np.float64);q0=np.asarray(q0,np.float64)
    if not np.isfinite(d0).all() or not np.isfinite(q0).all() or np.any(d0<=0):return reject('REJECT_BASELINE_LOCAL_NONPOSITIVE')
    req_good=cur[cells]|base[cells]
    qratio0=q0/1.e-5;vratio0=d0/6.e-18
    qfloor=np.minimum(qratio0,1.001);vfloor=np.minimum(vratio0,1.001)
    xyz0=p64[local_t]
    pairs=np.asarray([(a,b) for a in range(4) for b in range(a+1,4)],np.int64)
    edge0=np.linalg.norm(xyz0[:,pairs[:,0]]-xyz0[:,pairs[:,1]],axis=2)
    longest0=edge0.max(axis=1)
    det_scale=np.maximum.reduce((np.abs(d0),1.e-5*np.maximum(longest0,1.e-30)**3,np.full(m,6.e-18),np.full(m,1.e-30)))
    positive_floor=np.maximum(1.e-30,det_scale*1.e-14)
    bad=~cur[cells]
    start=time.monotonic();deadline=start+float(time_limit_s)
    if time_limit_s<=0 or time_limit_s>3.0 or maxiter<=0 or maxiter>60:return reject('REJECT_UNBOUNDED_SOLVER_BUDGET')
    sep_records=[]
    if separation_rows is not None:
        sepobj=Q._normalise_separation_rows(separation_rows)
        selected=set()
        for node in free:selected.update(sepobj['row_indices_by_point'].get(int(node),()))
        for ri in sorted(selected):
            r=sepobj['constraints'][int(ri)];nodes=np.asarray(r['point_rows'],np.int64);vec=np.asarray(r['vectors'],np.float64)
            origin=np.asarray(r['origin'],np.float64);coeff=np.zeros(nvar,np.float64)
            for node,v in zip(nodes,vec):
                j=int(lookup[int(node)]) if 0<=int(node)<len(lookup) else -1
                if j>=0:coeff[3*j:3*j+3]+=caps[j]*v
            lhs0=float(np.einsum('ij,ij->',vec,p64[nodes]-origin[None,:]))
            sep_records.append((float(r['upper_bound'])-lhs0,coeff))
    def unpack(z):
        y=np.asarray(z[:nvar],np.float64).reshape(k,3)
        work=p64.copy();work[free]=x0+caps[:,None]*y
        return y,work
    def evaluate(y,with_grad=True):
        if time.monotonic()>deadline:raise TimeoutError
        work=p64.copy();work[free]=x0+caps[:,None]*np.asarray(y,np.float64).reshape(k,3)
        d,q=Q._M.measurements(work,local_t);d=np.asarray(d,np.float64);q=np.asarray(q,np.float64)
        gradx=Q.determinant_gradients(work,local_t)
        det_y=np.zeros((m,nvar),np.float64)
        nodeix=lookup[local_t]
        for c in range(4):
            j=nodeix[:,c];sel=j>=0
            if np.any(sel):det_y[np.flatnonzero(sel)[:,None],3*j[sel,None]+np.arange(3)[None,:]]=gradx[sel,c,:]*caps[j[sel],None]
        xyz=work[local_t]
        edgevec=xyz[:,pairs[:,0],:]-xyz[:,pairs[:,1],:]
        lens=np.linalg.norm(edgevec,axis=2)
        maxix=np.argmax(lens,axis=1);rowix=np.arange(m);L=np.maximum(lens[rowix,maxix],1.e-30)
        dL_y=np.zeros((m,nvar),np.float64)
        ea=pairs[maxix,0];eb=pairs[maxix,1];unit=edgevec[rowix,maxix]/L[:,None]
        for c in range(4):
            j=nodeix[:,c];sel=j>=0
            if np.any(sel):
                sign=np.where(ea[sel]==c,1.,np.where(eb[sel]==c,-1.,0.))
                dL_y[np.flatnonzero(sel)[:,None],3*j[sel,None]+np.arange(3)[None,:]]=unit[sel]*sign[:,None]*caps[j[sel],None]
        qden=1.e-5*L**3;qr=d/qden;vr=d/6.e-18
        qgrad=det_y/qden[:,None]-(3.*d/(1.e-5*L**4))[:,None]*dL_y
        return work,d,q,qr,vr,det_y,qgrad
    def objective(z):
        _,_,_,qr,vr,_,_=evaluate(z[:nvar],False)
        return _log_deficit(qr[bad],vr[bad])
    def objective_jac(z):
        _,_,_,qr,vr,det_y,qgrad=evaluate(z[:nvar],True)
        gy=np.zeros(nvar,np.float64)
        aq=bad&(qr<1.)&(qr>np.finfo(np.float64).tiny);av=bad&(vr<1.)&(vr>np.finfo(np.float64).tiny)
        if np.any(aq):gy-=np.sum(qgrad[aq]/qr[aq,None],axis=0)
        if np.any(av):gy-=np.sum(det_y[av]/vr[av,None]/6.e-18,axis=0)
        return np.r_[gy,np.zeros(nvar,np.float64)]
    # Linear absolute-value slack constraints: u >= +/-y and sum(abs(y_i)) <= 1 per point.
    def constraints(z):
        y=z[:nvar].reshape(k,3);u=z[nvar:].reshape(k,3)
        _,d,q,qr,vr,_,_=evaluate(y.reshape(-1),False)
        vals=[(u-y).reshape(-1),(u+y).reshape(-1),1.-u.sum(axis=1),d/det_scale-positive_floor/det_scale,
              qr[req_good]-qfloor[req_good],vr[req_good]-vfloor[req_good]]
        if sep_records:
            vals.append(np.asarray([rhs-coeff@y.reshape(-1) for rhs,coeff in sep_records],np.float64))
        return np.concatenate(vals)
    def constraints_jac(z):
        y=z[:nvar].reshape(k,3);u=z[nvar:].reshape(k,3)
        _,d,q,qr,vr,det_y,qgrad=evaluate(y.reshape(-1),True)
        ng=int(np.count_nonzero(req_good));ns=len(sep_records)
        J=np.zeros((7*k+m+2*ng+ns,zdim),np.float64);I=np.eye(nvar,dtype=np.float64)
        J[0:nvar,0:nvar]=-I;J[0:nvar,nvar:]=I
        J[nvar:2*nvar,0:nvar]=I;J[nvar:2*nvar,nvar:]=I
        for j in range(k):J[2*nvar+j,nvar+3*j:nvar+3*j+3]=-1.
        currow=7*k
        J[currow:currow+m,0:nvar]=det_y/det_scale[:,None];currow+=m
        J[currow:currow+ng,0:nvar]=qgrad[req_good];currow+=ng
        J[currow:currow+ng,0:nvar]=det_y[req_good]/6.e-18;currow+=ng
        for rhs,coeff in sep_records:J[currow,0:nvar]=-coeff;currow+=1
        return J
    z0=np.zeros(zdim,np.float64);bounds=[(-1.,1.)]*nvar+[(0.,1.)]*nvar
    try:
        from scipy.optimize import minimize
        result=minimize(objective,z0,method='SLSQP',jac=objective_jac,bounds=bounds,
            constraints=[{'type':'ineq','fun':constraints,'jac':constraints_jac}],
            options={'maxiter':int(maxiter),'ftol':1.e-10,'disp':False})
    except TimeoutError:return reject('REJECT_PATCH_TIME_LIMIT',affected_cells=int(m),free_nodes=int(k))
    except Exception as exc:return reject('REJECT_SLSQP_EXCEPTION',exception=repr(exc),traceback=traceback.format_exc(limit=8),affected_cells=int(m),free_nodes=int(k))
    if time.monotonic()>deadline:return reject('REJECT_PATCH_TIME_LIMIT',affected_cells=int(m),free_nodes=int(k),iterations=int(getattr(result,'nit',0)))
    _,work=unpack(result.x);trial=np.asarray(work,np.float32)
    # Recover only F32 per-point L1-cap overshoot, moving coordinates toward the current point.
    for j,node in enumerate(free):
        try:
            recovered,_=Q.recover_f32_trust_rounding(p[node:node+1],trial[node:node+1],float(caps[j]))
            trial[node]=recovered[0]
        except Exception as exc:return reject('REJECT_F32_CAP_RECOVERY',node=int(node),exception=repr(exc))
    changed=np.flatnonzero(np.any(trial.view(np.uint32)!=p.view(np.uint32),axis=1))
    if not len(changed):return reject('REJECT_NO_F32_MOVE',optimizer_success=bool(result.success),iterations=int(result.nit))
    if not set(map(int,changed)).issubset(set(map(int,free))):return reject('REJECT_MOVED_POINT_OUTSIDE_PATCH')
    step=np.abs(trial[free].astype(np.float64)-p[free].astype(np.float64)).sum(axis=1)
    if np.any(step>caps+1.e-15):return reject('REJECT_F32_CAP_EXCEEDED',maximum_L1_m=float(step.max(initial=0.)))
    nd,nq=Q._M.measurements(trial,local_t);ng=np.asarray(Q._M.passes(nd,nq),bool)
    if not np.isfinite(nd).all() or not np.isfinite(nq).all() or np.any(nd<=0):return reject('REJECT_F32_NONPOSITIVE_OR_NONFINITE')
    if np.any((cur[cells]|base[cells])&~ng):return reject('REJECT_CURRENT_OR_BASELINE_GOOD_REGRESSION',regressions=int(np.count_nonzero((cur[cells]|base[cells])&~ng)))
    exact=Q._exact_positive(trial,local_t)
    if any(int(x)<=0 for x in exact):return reject('REJECT_EXACT_F32_NONPOSITIVE')
    witness_violations=Q._separation_violation_rows(trial,separation_rows,point_filter=free)
    if witness_violations:return reject('REJECT_EXACT_WITNESS_REGRESSION',violating_witness_rows=list(map(int,witness_violations[:64])),violating_witness_count=int(len(witness_violations)))
    dscore0=_log_deficit(qratio0[bad],vratio0[bad]);dn=np.asarray(nd,np.float64)/np.maximum(np.max(np.stack([np.linalg.norm(trial[local_t[:,a]]-trial[local_t[:,b]],axis=1) for a in range(4) for b in range(a+1,4)],axis=0),axis=0)**3,np.finfo(float).tiny)/1.e-5
    vnew=np.asarray(nd,np.float64)/6.e-18;dscore1=_log_deficit(dn[bad],vnew[bad])
    bad0=int(np.count_nonzero(~cur[cells]));bad1=int(np.count_nonzero(~ng))
    count_gain=bad1<bad0;score_gain=bad1==bad0 and dscore0-dscore1>max(1.e-8,1.e-6*dscore0)
    if not (count_gain or score_gain):return reject('REJECT_NO_LOCAL_LEX_PROGRESS',bad_before=bad0,bad_after=bad1,score_before=dscore0,score_after=dscore1,optimizer_success=bool(result.success),iterations=int(result.nit))
    return trial,{'schema':'h706.joint-small-patch-slsqp.v3','status':'PASS_JOINT_PATCH_PROPOSAL','physical_admission':False,'native_steps':0,
        'fallback_scope':'failed local patch only; 2-4 free rows, 1-4 seed bad cells','prior_failures':prior_failures,'affected_cells':cells.tolist(),'seed_bad_cells':seeds.tolist(),
        'free_rows':free.tolist(),'seed_free_corner_counts':free_corner_counts.tolist(),'moved_rows':changed.tolist(),'full_point_star_verified':True,'maximum_remaining_l1_m':float(caps.max(initial=0.)),
        'maximum_step_l1_m':float(step.max(initial=0.)),'bad_before':bad0,'bad_after':bad1,'score_before':dscore0,'score_after':dscore1,
        'optimizer_success':bool(result.success),'optimizer_message':str(result.message),'iterations':int(result.nit),'solver_elapsed_s':float(time.monotonic()-start),
        'analytic_jacobians':True,'normalized_variable_parameterization':'delta_i=remaining_l1_cap_i*y_i','quality_objective':'sum log deficits for current bad cells; max-edge q and min-volume','all_current_and_baseline_good_incident_cells_preserved':True,
        'all_local_exact_f32_determinants_positive':True,'all_local_f32_native_passes_recomputed':True,'all_attached_separation_witnesses_rechecked':True,
        'shared_point_cell_incidence_reused':incidence_offsets is not None,
        'requires_external_full_805_and_global_boundary_flux_volume_audits':True,'mass_reconciliation':'PENDING'}
