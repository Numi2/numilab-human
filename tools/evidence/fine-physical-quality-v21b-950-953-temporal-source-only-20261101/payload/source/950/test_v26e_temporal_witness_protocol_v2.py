#!/usr/bin/env python3
"""Recompute a rejected prior F32 witness before authorizing expansion rows."""
from pathlib import Path
import importlib.util
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v26e_temporal_verifier_fixture_v2",HERE/"verify_ordinary_allrole_v26e_witness_ulp_temporal.py")
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
P=V.M;Q=P.Q
x0=np.float32(1e6);ya=np.float32(0.);za=np.float32(.1);zb=np.float32(.4)
Lx=np.float32(3135.0625);Ly=np.float32(3150.076904296875);Lb=np.float32(4000.)
points=np.asarray([[x0,ya,zb],[x0+Lx,ya,zb],[x0,ya+Ly,zb],[x0+1.5*Lx,1.5*Ly,za],
    [x0,0.,0.],[x0+Lb,0.,0.],[x0,Lb,0.],[x0+1000.,1000.,za]],np.float32)
tets=np.asarray([[0,2,1,3],[4,5,6,7]],np.int64)
det,q=Q._M.measurements(points.astype(np.float64),tets)
good=np.asarray(Q._M.passes(det,q),bool)
assert good.tolist()==[True,False]
zulp=float(np.nextafter(za,np.float32(np.inf),dtype=np.float32)-za)
candidate64=points.astype(np.float64);candidate64[3,0]+=.04;candidate64[3,2]+=.6*zulp;candidate64[7,2]+=2.6*zulp
separation=[{"point_rows":np.asarray([7]),"vectors":np.asarray([[0.,0.,1.]]),"origin":points[7].astype(np.float64),"upper_bound":1.e-6}]
def no_progress(*args):return {"accepted":False,"global_bad_before":1,"global_bad_candidate":1,"count_gain":False,"score_gain":False}
optimizer={"status":"SYNTHETIC_REJECTED_RAW_WITNESS","affected_cells":[0,1],"free_rows":[3,7]}
free=np.asarray([3,7],np.int64)
_,prior_f32=P.F32_BACKTRACK.backtrack_f32_candidate(points,candidate64,tets,free,np.asarray([.2,.2]),
    good,good,lambda x,t:Q._M.measurements(x,t),lambda d0,q0:Q._M.passes(d0,q0),progress=no_progress,
    exact_positive=lambda x,t:Q._exact_positive(x,t),
    separation_violations=lambda x:Q._separation_violation_rows(x,separation,point_filter=free),
    immutable_baseline_points_f32=points,optimizer_status=optimizer)
assert prior_f32["status"]=="NO_ACCEPTED_BACKTRACKED_PROPOSAL",prior_f32
assert prior_f32["first_f32_good_regression"]["alpha"]==1.
prior_f32["first_f32_good_regression"]["optimizer_status"].update({"affected_cells":[0,1],"free_rows":[3,7]})
previous={"affected_cells":[0,1],"free_rows":[3,7],"f32_backtracking":prior_f32}
prior_witness=prior_f32["first_f32_good_regression"]["regressed_good_cell_witness"]
assert prior_witness["rows"][0]["local_cell_row"]==0
assert np.asarray(Q._M.passes(*Q._M.measurements(points.astype(np.float64),tets))).tolist()==[True,False]
# The next expanded solve has a different witness, proving the receipt is temporal.
current_attempt_witness_local_rows=[1]
assert current_attempt_witness_local_rows!=[r["local_cell_row"] for r in prior_witness["rows"]]
selection={"status":"WITNESS_DIRECTED_EXPANSION_READY","raw64_witness_predicate_passed":True,
 "witness_local_cell_rows":[0],"seed_free_rows":[0,4],"witness_point_rows":[3],
 "selected_witness_extra_rows":[3],"selected_rows":[0,3,4],
 "support_by_selected_row":{"3":{"witness_rows":[0]}}}
free={0,3,4};patch=free|{1,2,5,6,7};star=set(range(8))
result=V.verify_witness_selection_from_previous_report(selection,previous,free,patch,star,tets,set(),points)
assert result["status"]=="PASS_PREVIOUS_REPORT_WITNESS_SELECTION",result
assert result["raw_witness_replay"]["mode"]=="rejected_coordinate_ledger"
# Falsify the raw-F64 ledger: independent recomputation must reject it.
bad_previous={"affected_cells":[0,1],"free_rows":[3,7],"f32_backtracking":dict(prior_f32)}
bad_f32=dict(prior_f32);bad_first=dict(prior_f32["first_f32_good_regression"]);bad_witness=dict(bad_first["regressed_good_cell_witness"])
bad_coords=dict(bad_witness["free_point_coordinates"]);bad_row=dict(bad_coords["3"]);bad_candidate=list(bad_row["candidate64_m"]);bad_candidate[2]+=1.e-4;bad_row["candidate64_m"]=bad_candidate;bad_coords["3"]=bad_row;bad_witness["free_point_coordinates"]=bad_coords;bad_first["regressed_good_cell_witness"]=bad_witness;bad_f32["first_f32_good_regression"]=bad_first;bad_previous["f32_backtracking"]=bad_f32
try:
 V.verify_witness_selection_from_previous_report(selection,bad_previous,free,patch,star,tets,set(),points)
 raise AssertionError("mismatched raw coordinate ledger was accepted")
except ValueError as e:
 assert "rejected witness Float32 rounding differs" in str(e),str(e)
# Unauthorized extras remain rejected despite a valid prior witness.
bad=dict(selection);bad["selected_witness_extra_rows"]=[5];bad["witness_point_rows"]=[3,5];bad["selected_rows"]=[0,3,4,5];bad["support_by_selected_row"]={"5":{"witness_rows":[0]}}
try:
 V.verify_witness_selection_from_previous_report(bad,previous,set(bad["selected_rows"]),patch,star,tets,set(),points)
 raise AssertionError("unwitnessed extra was accepted")
except ValueError as e:
 assert "preceding witnessed authorized rows" in str(e),str(e)
print("PASS_TEMPORAL_SELECTION_PRIOR_REJECTED_REPORT_RAW_REPLAY")
