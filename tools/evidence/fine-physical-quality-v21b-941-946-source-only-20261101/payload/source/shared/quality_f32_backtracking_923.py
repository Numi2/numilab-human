#!/usr/bin/env python3
"""Full-star Float32 backtracking for one already-computed SLSQP candidate."""
from __future__ import annotations
import numpy as np

def _accepted(value):
    return bool(value.get("accepted", False)) if isinstance(value, dict) else bool(value)

def backtrack_f32_candidate(points_f32, candidate64, tetrahedra, free_rows, l1_caps_m,
                           current_passed, baseline_passed, measure, passes, *,
                           progress, exact_positive=None, separation_violations=None,
                           immutable_baseline_points_f32=None, optimizer_status=None,
                           q_min=1.0e-5, volume_min_m3=1.0e-18, max_halvings=6,
                           cap_tolerance_m=1.0e-15):
    """Try alpha=1,1/2,...,1/64 with unchanged owner predicates and role-progress callback.

    measure(points,tets)->(det6, normalized_det); passes(det6,normalized_det) is
    the unchanged native gate. progress(trial32,det6,nd,alpha) applies the existing
    role count/score policy. Caller must pass the complete incident-cell star.
    """
    p0=np.asarray(points_f32); c64=np.asarray(candidate64,dtype=np.float64)
    t=np.asarray(tetrahedra,dtype=np.int64); free=np.asarray(free_rows,dtype=np.int64)
    cur=np.asarray(current_passed,dtype=bool); base=np.asarray(baseline_passed,dtype=bool)
    if p0.dtype!=np.float32 or p0.ndim!=2 or p0.shape[1]!=3: raise ValueError("points_f32 must be Nx3")
    if c64.shape!=p0.shape or not np.isfinite(c64).all(): raise ValueError("candidate64 shape/nonfinite")
    if t.ndim!=2 or t.shape[1]!=4 or np.any(t<0) or np.any(t>=len(p0)): raise ValueError("invalid Mx4 tetrahedra")
    if cur.shape!=(len(t),) or base.shape!=(len(t),): raise ValueError("good masks must cover full star")
    if free.ndim!=1 or len(np.unique(free))!=len(free) or np.any(free<0) or np.any(free>=len(p0)): raise ValueError("invalid free rows")
    caps=np.asarray(l1_caps_m,dtype=np.float64)
    if caps.ndim==0: caps=np.full(len(free),float(caps))
    if caps.shape!=(len(free),) or np.any(~np.isfinite(caps)) or np.any(caps<0): raise ValueError("invalid L1 caps")
    mask=np.zeros(len(p0),bool); mask[free]=True
    p64=p0.astype(np.float64)
    if not np.array_equal(c64[~mask].view(np.uint64),p64[~mask].view(np.uint64)):
        raise ValueError("candidate changes fixed rows")
    req=cur|base
    d0,q0=measure(p64,t); d0=np.asarray(d0,float); q0=np.asarray(q0,float)
    imm_d=imm_q=None
    if immutable_baseline_points_f32 is not None:
        ib=np.asarray(immutable_baseline_points_f32)
        if ib.dtype!=np.float32 or ib.shape!=p0.shape: raise ValueError("bad immutable baseline")
        imm_d,imm_q=measure(ib.astype(np.float64),t); imm_d=np.asarray(imm_d,float); imm_q=np.asarray(imm_q,float)
    attempts=[]; first_regression=None
    for k in range(int(max_halvings)+1):
        alpha=2.0**(-k)
        x64=p64.copy(); x64[free]=p64[free]+alpha*(c64[free]-p64[free])
        x32=np.asarray(x64,dtype=np.float32)
        step=np.abs(x32[free].astype(np.float64)-p64[free]).sum(axis=1,dtype=np.float64)
        maxstep=float(step.max(initial=0.0))
        if np.any(step>caps+cap_tolerance_m):
            attempts.append({"alpha":alpha,"status":"REJECT_L1_CAP","maximum_l1_step_m":maxstep}); continue
        d64,q64=measure(x64,t); d32,q32=measure(x32,t)
        d64=np.asarray(d64,float); q64=np.asarray(q64,float); d32=np.asarray(d32,float); q32=np.asarray(q32,float)
        g64=np.asarray(passes(d64,q64),bool); g32=np.asarray(passes(d32,q32),bool)
        if np.any(~np.isfinite(d32)) or np.any(~np.isfinite(q32)) or np.any(d32<=0):
            attempts.append({"alpha":alpha,"status":"REJECT_F32_NONPOSITIVE_OR_NONFINITE"}); continue
        reg=np.flatnonzero(req&~g32)
        if len(reg):
            if first_regression is None:
                witnesses=[]
                fset=set(map(int,free)); used=set()
                for r in reg[:16]:
                    nodes=np.asarray(t[int(r)],dtype=np.int64); used.update(int(n) for n in nodes if int(n) in fset)
                    item={"local_cell_row":int(r),"tet_point_rows":nodes.tolist(),
                          "current_input":{"det6_m3":float(d0[r]),"volume_m3":float(d0[r]/6),
                              "normalized_det":float(q0[r]),"current_passed":bool(cur[r]),
                              "immutable_baseline_passed":bool(base[r])},
                          "candidate_float64":{"det6_m3":float(d64[r]),"volume_m3":float(d64[r]/6),
                              "normalized_det":float(q64[r]),"native_pass":bool(g64[r])},
                          "candidate_float32":{"det6_m3":float(d32[r]),"volume_m3":float(d32[r]/6),
                              "normalized_det":float(q32[r]),"native_pass":bool(g32[r])},
                          "float32_minus_float64":{"det6_m3":float(d32[r]-d64[r]),
                              "normalized_det":float(q32[r]-q64[r])},
                          "regression_required_by_current_or_immutable_baseline":True}
                    if q_min is not None:
                        item["normalized_det_threshold"]=float(q_min)
                        item["candidate64_q_margin"]=float(q64[r]-q_min)
                        item["candidate32_q_margin"]=float(q32[r]-q_min)
                    if volume_min_m3 is not None:
                        item["volume_threshold_m3"]=float(volume_min_m3)
                        item["candidate64_volume_margin_m3"]=float(d64[r]/6-volume_min_m3)
                        item["candidate32_volume_margin_m3"]=float(d32[r]/6-volume_min_m3)
                    if imm_d is not None:
                        item["immutable_baseline"]={"det6_m3":float(imm_d[r]),
                            "volume_m3":float(imm_d[r]/6),"normalized_det":float(imm_q[r])}
                    witnesses.append(item)
                coords={str(n):{"input_f32_m":p0[n].astype(np.float64).tolist(),
                    "candidate64_m":x64[n].tolist(),"candidate32_m":x32[n].astype(np.float64).tolist()}
                    for n in sorted(used)}
                first_regression={"alpha":alpha,
                    "candidate64_passed_same_native_predicate_on_regressed_rows":bool(np.all(g64[reg])),
                    "optimizer_status":dict(optimizer_status or {}),
                    "regressed_good_cell_witness":{"rows":witnesses,"free_point_coordinates":coords}}
            attempts.append({"alpha":alpha,"status":"REJECT_F32_GOOD_CELL_REGRESSION",
                "regression_count":int(len(reg))}); continue
        if exact_positive is not None:
            e=exact_positive(x32,t)
            if isinstance(e,(bool,np.bool_)): ok=bool(e)
            else: ok=not any(int(v)<=0 for v in e)
            if not ok:
                attempts.append({"alpha":alpha,"status":"REJECT_EXACT_NONPOSITIVE"}); continue
        if separation_violations is not None:
            bad=separation_violations(x32)
            if bad is not None and len(bad):
                attempts.append({"alpha":alpha,"status":"REJECT_SEPARATION","violation_count":int(len(bad))}); continue
        pd=progress(x32,d32,q32,alpha)
        if not _accepted(pd):
            attempts.append({"alpha":alpha,"status":"REJECT_NO_ROLE_COUNT_OR_SCORE_PROGRESS",
                "progress":pd if isinstance(pd,dict) else None}); continue
        attempts.append({"alpha":alpha,"status":"ACCEPT","maximum_l1_step_m":maxstep,
            "progress":pd if isinstance(pd,dict) else None})
        return x32,{"schema":"h706.quality-f32-full-star-backtracking.v1",
            "status":"PASS_BOUNDED_F32_BACKTRACKED_PROPOSAL","physical_admission":False,"native_steps":0,
            "alpha":alpha,"attempts":attempts,"first_f32_good_regression":first_regression,
            "optimizer_status":dict(optimizer_status or {}),"full_supplied_incident_star_replayed":True,
            "all_current_and_baseline_good_preserved":True,"unchanged_native_predicates":True,
            "no_threshold_or_trust_cap_relaxation":True,"maximum_l1_step_m":maxstep}
    return None,{"schema":"h706.quality-f32-full-star-backtracking.v1",
        "status":"NO_ACCEPTED_BACKTRACKED_PROPOSAL","physical_admission":False,"native_steps":0,
        "attempts":attempts,"first_f32_good_regression":first_regression,
        "optimizer_status":dict(optimizer_status or {}),"full_supplied_incident_star_replayed":True,
        "all_current_and_baseline_good_preserved":True,"unchanged_native_predicates":True,
        "no_threshold_or_trust_cap_relaxation":True}

