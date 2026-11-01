#!/usr/bin/env python3
"""Bounded one-ULP correction on exact witness coordinates for Float32 candidates."""
from __future__ import annotations
import heapq
import itertools
import numpy as np

SCHEMA = "h706.quality-f32-witness-ulp-search.v2"
MAX_CHANGED_COORDINATES = 4
MAX_STATES = 256

def _log_deficit(det, q):
    d=np.asarray(det,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    tiny=np.finfo(np.float64).tiny
    return float(np.maximum(0.,-np.log(np.maximum(q,tiny)/1.e-5)).sum(dtype=np.float64)
        +np.maximum(0.,-np.log(np.maximum(d,tiny)/6.e-18)).sum(dtype=np.float64))

def _finite_positive(exact_positive, points, tetrahedra):
    values=exact_positive(points,tetrahedra)
    if isinstance(values,(bool,np.bool_)):return bool(values)
    return not any(int(v)<=0 for v in values)

def _install_state(work, start, changed, coordinate_keys):
    # Reset the small eligible coordinate set; never copy the complete domain per state.
    for r,a in coordinate_keys:work[r,a]=start[r,a]
    for r,a,direction in changed:
        target=np.float32(np.inf if direction>0 else -np.inf)
        work[r,a]=np.nextafter(start[r,a],target,dtype=np.float32)

def search_witness_ulp_correction(points_f32, rounded_trial_f32, tetrahedra,
        free_rows, l1_caps_m, current_passed, baseline_passed,
        measure, passes, *, exact_positive, separation_violations, progress,
        witness_point_rows, alpha, max_changed_coordinates=4, max_states=256,
        max_ulp_steps_per_coordinate=1):
    """Search one-step nextafter choices on free witness-tet coordinates.

    Nonpositive rounded starts are explicitly unsupported. This helper corrects
    small Float32 rounding losses from positive candidates; it never admits
    nonpositive geometry.
    """
    p=np.asarray(points_f32);start=np.asarray(rounded_trial_f32)
    t=np.asarray(tetrahedra,np.int64);free=np.asarray(free_rows,np.int64)
    cur=np.asarray(current_passed,bool);base=np.asarray(baseline_passed,bool)
    caps=np.asarray(l1_caps_m,np.float64)
    if p.dtype!=np.float32 or start.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or start.shape!=p.shape:
        raise ValueError("Float32 ULP input point arrays must be matching Nx3")
    if t.ndim!=2 or t.shape[1]!=4 or (len(t) and (t.min()<0 or t.max()>=len(p))):
        raise ValueError("invalid ULP tetrahedron domain")
    if cur.shape!=(len(t),) or base.shape!=(len(t),):raise ValueError("ULP quality mask shape mismatch")
    if free.ndim!=1 or len(np.unique(free))!=len(free) or np.any(free<0) or np.any(free>=len(p)):
        raise ValueError("invalid ULP free-row list")
    if caps.shape!=(len(free),) or np.any(~np.isfinite(caps)) or np.any(caps<0):
        raise ValueError("invalid ULP cap list")
    if max_changed_coordinates<1 or max_changed_coordinates>MAX_CHANGED_COORDINATES or max_states<1 or max_states>MAX_STATES:
        raise ValueError("ULP search budget is outside the fixed bound")
    if max_ulp_steps_per_coordinate!=1:raise ValueError("only one nextafter step per coordinate is permitted")

    start_det,start_q=measure(start,t)
    start_det=np.asarray(start_det,np.float64);start_q=np.asarray(start_q,np.float64)
    exact_start=_finite_positive(exact_positive,start,t)
    if (not np.isfinite(start_det).all() or not np.isfinite(start_q).all()
            or np.any(start_det<=0) or not exact_start):
        exact_values=exact_positive(start,t)
        exact_min=(bool(exact_values) if isinstance(exact_values,(bool,np.bool_))
                   else min((int(v) for v in exact_values),default=None))
        return None,{"schema":SCHEMA,"status":"UNSUPPORTED_ROUNDED_START_NONPOSITIVE",
            "explored_states":0,"start_float32_determinants_finite":bool(np.isfinite(start_det).all()),
            "start_float32_normalized_quality_finite":bool(np.isfinite(start_q).all()),
            "start_float32_min_signed_determinant":float(start_det.min(initial=np.inf)),
            "start_exact_positive":bool(exact_start),"start_exact_min_determinant":exact_min,
            "reason":"ULP correction is limited to positive rounded proposals"}

    allowed=set(map(int,free));witness=sorted(set(map(int,witness_point_rows))&allowed)
    if not witness:return None,{"schema":SCHEMA,"status":"NO_AUTHORIZED_WITNESS_COORDINATES",
        "witness_point_rows":sorted(set(map(int,witness_point_rows))),"explored_states":0}
    coordinate_keys=tuple((r,a) for r in witness for a in range(3))
    req=cur|base
    attempts=0;seen=set();heap=[];serial=itertools.count();work=start.copy()

    def signature(changed):return tuple(sorted(changed))
    def inspect(changed):
        nonlocal attempts
        _install_state(work,start,changed,coordinate_keys);attempts+=1
        if not np.isfinite(work).all():return None
        steps=np.abs(work[free].astype(np.float64)-p[free].astype(np.float64)).sum(axis=1,dtype=np.float64)
        if np.any(steps>caps+1.e-15):return None
        det,q=measure(work,t);det=np.asarray(det,np.float64);q=np.asarray(q,np.float64)
        if not np.isfinite(det).all() or not np.isfinite(q).all() or np.any(det<=0):return None
        good=np.asarray(passes(det,q),bool)
        if good.shape!=(len(t),) or not _finite_positive(exact_positive,work,t):return None
        sep=separation_violations(work)
        if sep is not None and len(sep):return None
        regressions=int(np.count_nonzero(req&~good));bad=int(np.count_nonzero(~good))
        score=_log_deficit(det,q)
        return (regressions,bad,score,len(changed),signature(changed)),det,q,good,steps
    def enqueue(changed):
        nonlocal attempts
        if attempts>=int(max_states):return False
        sig=signature(changed)
        if sig in seen:return False
        seen.add(sig)
        evaluated=inspect(sig)
        if evaluated is None:return False
        rank,det,q,good,steps=evaluated
        heapq.heappush(heap,(rank,next(serial),sig,(det,q,good,steps)))
        return True

    enqueue(tuple())
    while heap and attempts<=int(max_states):
        rank,_,changed,payload=heapq.heappop(heap)
        det,q,good,steps=payload;_install_state(work,start,changed,coordinate_keys)
        if rank[0]==0:
            pd=progress(work,det,q,float(alpha))
            accepted=bool(pd.get("accepted",False)) if isinstance(pd,dict) else bool(pd)
            if accepted:
                delta=np.abs(work[free].astype(np.float64)-p[free].astype(np.float64)).sum(axis=1,dtype=np.float64)
                changed_rows=sorted(set(r for r,a,direction in changed))
                coords=[{"point_row":int(r),"axis":int(a),
                    "from_bits":int(start[r,a].view(np.uint32)),"to_bits":int(work[r,a].view(np.uint32)),
                    "direction":"up" if direction>0 else "down"} for r,a,direction in changed]
                return work.copy(),{"schema":SCHEMA,"status":"PASS_BOUNDED_WITNESS_ULP_CORRECTION",
                    "alpha":float(alpha),"witness_point_rows":witness,"changed_coordinates":coords,
                    "changed_point_rows":changed_rows,"changed_coordinate_count":len(changed),
                    "max_ulp_steps_per_coordinate":1,"max_changed_coordinates":int(max_changed_coordinates),
                    "max_states":int(max_states),"explored_states":int(attempts),"unique_states":int(len(seen)),
                    "initial_rounded_point_bits":{str(r):start[r].view(np.uint32).astype(np.uint32).tolist() for r in changed_rows},
                    "accepted_point_bits":{str(r):work[r].view(np.uint32).astype(np.uint32).tolist() for r in changed_rows},
                    "final_progress":pd if isinstance(pd,dict) else {"accepted":accepted},
                    "final_native_bad_count":int(rank[1]),"final_log_deficit":float(rank[2]),
                    "maximum_l1_step_m":float(delta.max(initial=0.)),
                    "all_returned_gates_rechecked":True,"thresholds_or_caps_changed":False}
        if len(changed)>=int(max_changed_coordinates) or attempts>=int(max_states):continue
        changedset=set(changed)
        for r,a in coordinate_keys:
            if (r,a,1) in changedset or (r,a,-1) in changedset:continue
            for direction in (1,-1):
                if attempts>=int(max_states):break
                enqueue(tuple(sorted(changedset|{(r,a,direction)})))
            if attempts>=int(max_states):break
    return None,{"schema":SCHEMA,"status":"NO_ACCEPTED_WITNESS_ULP_CORRECTION",
        "alpha":float(alpha),"witness_point_rows":witness,"explored_states":int(attempts),
        "unique_states":int(len(seen)),"max_changed_coordinates":int(max_changed_coordinates),
        "max_states":int(max_states),"max_ulp_steps_per_coordinate":1,
        "reason":"bounded one-ULP witness-coordinate search found no state passing unchanged full-star gates"}
