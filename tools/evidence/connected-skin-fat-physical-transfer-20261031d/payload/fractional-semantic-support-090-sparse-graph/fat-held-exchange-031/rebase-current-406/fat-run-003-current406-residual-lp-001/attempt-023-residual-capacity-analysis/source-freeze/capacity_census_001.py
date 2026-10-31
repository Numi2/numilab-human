from pathlib import Path
import hashlib,json,time,resource
import numpy as np
R=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001")
base=R/"attempt-022-source-exhaustion-reserved-batch"
out=R/"attempt-023-residual-capacity-analysis"
graph=R/"attempt-005/lp-input-current406.npz"
target=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz")
support=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/connected-region-mass-support-143-current406-root/fat-connected-region-support.npz")
M=base/"run-001/merged-held-centroid-noncentroid-source-exhaustion.npz"
stages=[
 R/"attempt-012-centroid-partial-batch/partial-centroid-additions.npz",
 R/"attempt-014-noncentroid-partial-batch/partial-noncentroid-additions.npz",
 R/"attempt-021-source-exhaustion-batch/run-001/proposal.npz",
 base/"run-001/proposal.npz"]
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
t0=time.time()
with np.load(graph,allow_pickle=False) as z:
 P=np.asarray(z["parent_role_rows"],np.int32); C=np.asarray(z["child_rows"],np.int32); B=np.asarray(z["barycentric"],np.float64); D=np.asarray(z["distance_m"],np.float64); sitecap=np.asarray(z["column_mass_scale_kg"],np.float64); vol=np.asarray(z["child_volume_hybrid_m3"],np.float64)
with np.load(M,allow_pickle=False) as z:
 q=np.asarray(z["parent_corner_residual_after_kg"],np.float64); childmass=np.asarray(z["child_current_mass_kg"],np.float64); parentmass=np.asarray(z["parent_source_mass_kg"],np.float64)
used=np.zeros(len(P),np.float64)
for fp in stages:
 with np.load(fp,allow_pickle=False) as z:
  rr=np.asarray(z["source_candidate_site_rows"],np.int64); mm=np.asarray(z["flow_mass_kg"],np.float64)
  if rr.shape!=mm.shape or np.any(rr<0) or np.any(rr>=len(P)): raise RuntimeError("stage site row shape/domain")
  np.add.at(used,rr,mm)
site_rem=sitecap-used
if np.any(site_rem < -64*np.finfo(float).eps*np.maximum(sitecap,np.finfo(float).tiny)): raise RuntimeError("site cap exceeded")
site_rem=np.maximum(site_rem,0)
need=np.maximum(850.0*vol-childmass,0)
head=950.0*vol-childmass
n=len(P); capsum=np.zeros(len(vol),np.float64); cornercap=np.zeros_like(q); cap_total=0.; nonzero=0; eligible=0; by_receiver_n=np.zeros(len(vol),np.int32)
for lo in range(0,n,200000):
 hi=min(n,lo+200000); p=P[lo:hi]; c=C[lo:hi]; b=B[lo:hi]
 valid=np.all(np.isfinite(b),axis=1)&np.all(b>=0,axis=1)&(np.abs(b.sum(axis=1)-1)<=2e-14)&(D[lo:hi]<=.019353)&(site_rem[lo:hi]>0)&(head[c]>0)
 ratio=np.full_like(b,np.inf)
 np.divide(q[p],b,out=ratio,where=b>0)
 src=ratio.min(axis=1)
 valid &= np.isfinite(src)&(src>0)
 maxflow=np.minimum(site_rem[lo:hi],np.minimum(head[c],src))
 maxflow[~valid]=0.
 nonzero+=int(np.count_nonzero(maxflow>0))
 cap_total+=float(maxflow.sum())
 np.add.at(capsum,c,maxflow)
 for k in range(4): np.add.at(cornercap[:,k],p,maxflow*b[:,k])
 by_receiver_n += np.bincount(c,weights=(maxflow>0).astype(np.int32),minlength=len(vol)).astype(np.int32)
short=need>0
direct_capacity=np.minimum(capsum,head)
covers=direct_capacity+1e-12*np.maximum(need,1e-18)>=need
uncovered=np.flatnonzero(short&~covers)
covered=np.flatnonzero(short&covers)
percorner_ratio=np.divide(cornercap,q,out=np.zeros_like(q),where=q>0)
report={"schema":"numi.human.current406-source-only-residual-capacity-census.v1","status":"SOURCE_ONLY_RESIDUAL_CAPACITY_BOUND_DIAGNOSTIC","physical_admission":False,"native_steps":0,"complete_allocation":False,"scope":"Optimistic independent per-site upper-capacity census on pinned 406 graph. Sums single-site caps separately by receiver and source corner, so shared parent/receiver coupling is not solved; passing a bound is not a feasible allocation. Inputs and current406 geometry only.","inputs":{str(x):sha(x) for x in [graph,target,support,M,*stages]},"counts":{"graph_rows":int(n),"rows_with_positive_local_capacity":nonzero,"rows_with_residual_corner_support":eligible,"children_with_lower_deficit":int(short.sum()),"deficit_children_with_direct_site_capacity_bound_cover":int(len(covered)),"deficit_children_below_direct_site_capacity_bound":int(len(uncovered)),"deficit_children_no_capacity":int(np.count_nonzero(short&(capsum<=0))),"prior_stage_rows":[int(np.load(x,allow_pickle=False)["flow_mass_kg"].shape[0]) for x in stages],"capacity_census_chunk_rows":200000},"mass_kg":{"source_corner_residual":float(q.sum()),"receiver_lower_deficit":float(need.sum()),"receiver_upper_headroom":float(np.maximum(head,0).sum()),"sum_independent_site_row_caps":cap_total,"sum_deficit_receiver_individual_caps":float(np.minimum(capsum[short],np.maximum(head[short],0)).sum()),"uncovered_deficit_total":float(need[uncovered].sum()),"covered_deficit_total":float(need[covered].sum())},"metrics":{"max_child_deficit_kg":float(need.max(initial=0)),"max_single_child_capacity_kg":float(capsum.max(initial=0)),"max_parent_corner_capacity_over_q":float(percorner_ratio.max(initial=0)),"total_source_corner_capacity_over_q_sum":float(cornercap.sum()/max(q.sum(),1e-300)),"worst_uncovered_child_rows":sorted([{"child":int(c),"need_kg":float(need[c]),"direct_capacity_bound_kg":float(min(capsum[c],max(head[c],0))),"site_rows":int(by_receiver_n[c])} for c in uncovered],key=lambda x:(-x["need_kg"],x["child"]))[:100],"worst_covered_child_rows":sorted([{"child":int(c),"need_kg":float(need[c]),"direct_capacity_bound_kg":float(min(capsum[c],max(head[c],0))),"site_rows":int(by_receiver_n[c])} for c in covered],key=lambda x:(-x["need_kg"],x["child"]))[:20]},"elapsed_seconds":time.time()-t0,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
reportpath=out/"report.json";reportpath.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"counts":report["counts"],"mass_kg":report["mass_kg"],"metrics":{k:v for k,v in report["metrics"].items() if not k.startswith("worst_")},"elapsed_seconds":report["elapsed_seconds"],"peak_rss_bytes":report["peak_rss_bytes"],"report":str(reportpath),"sha256":sha(reportpath)},indent=2))
