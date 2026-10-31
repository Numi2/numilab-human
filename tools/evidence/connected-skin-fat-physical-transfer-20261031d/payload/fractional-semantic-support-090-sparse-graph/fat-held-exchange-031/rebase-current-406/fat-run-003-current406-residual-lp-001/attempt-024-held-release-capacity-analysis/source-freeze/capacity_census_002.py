from pathlib import Path
import hashlib,json,time,resource
import numpy as np
R=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001")
base=R/"attempt-022-source-exhaustion-reserved-batch"
out=R/"attempt-024-held-release-capacity-analysis"
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
n=len(P); capsum=np.zeros(len(vol),np.float64); cornercap=np.zeros_like(q); cap_total=0.; nonzero=0; eligible=0; fullcap_total=0.; full_nonzero=0; fullcapsum=np.zeros(len(vol),np.float64); potential_by_parent=np.zeros(len(q),np.float64); by_receiver_n=np.zeros(len(vol),np.int32); qfull=parentmass[:,None]*.25
for lo in range(0,n,200000):
 hi=min(n,lo+200000); p=P[lo:hi]; c=C[lo:hi]; b=B[lo:hi]
 valid=np.all(np.isfinite(b),axis=1)&np.all(b>=0,axis=1)&(np.abs(b.sum(axis=1)-1)<=2e-14)&(D[lo:hi]<=.019353)&(site_rem[lo:hi]>0)&(head[c]>0)
 ratio=np.full_like(b,np.inf)
 np.divide(q[p],b,out=ratio,where=b>0)
 src=ratio.min(axis=1)
 valid &= np.isfinite(src)&(src>0)
 eligible+=int(np.count_nonzero(valid))
 maxflow=np.minimum(site_rem[lo:hi],np.minimum(head[c],src))
 fullratio=np.full_like(b,np.inf)
 np.divide(qfull[p],b,out=fullratio,where=b>0)
 fullsrc=fullratio.min(axis=1)
 fullgood=np.all(np.isfinite(b),axis=1)&np.all(b>=0,axis=1)&(np.abs(b.sum(axis=1)-1)<=2e-14)&(D[lo:hi]<=.019353)&(fullsrc>0)
 fullcap=np.minimum(sitecap[lo:hi],np.minimum(950.0*vol[c],fullsrc))
 fullcap[~fullgood]=0.
 full_nonzero+=int(np.count_nonzero(fullcap>0)); fullcap_total+=float(fullcap.sum())
 np.add.at(fullcapsum,c,fullcap)
 deficitedge=need[c]>0
 np.add.at(potential_by_parent,p[deficitedge],fullcap[deficitedge])
 maxflow[~valid]=0.
 nonzero+=int(np.count_nonzero(maxflow>0))
 cap_total+=float(maxflow.sum())
 np.add.at(capsum,c,maxflow)
 for k in range(4): np.add.at(cornercap[:,k],p,maxflow*b[:,k])
 by_receiver_n += np.bincount(c,weights=(maxflow>0).astype(np.int32),minlength=len(vol)).astype(np.int32)
short=need>0
full_direct_capacity=np.minimum(fullcapsum,950.0*vol)
full_covers=full_direct_capacity+1e-12*np.maximum(need,1e-18)>=need
full_uncovered=np.flatnonzero(short&~full_covers)
with np.load(M,allow_pickle=False) as z:
 held_by_parent=np.bincount(np.asarray(z["flow_parent_role_rows"],np.int32),weights=np.asarray(z["flow_mass_kg"],np.float64),minlength=len(q))
top=[]
for pp in np.argsort(-potential_by_parent)[:200]:
 if potential_by_parent[pp]<=0: break
 top.append({"parent_role_row":int(pp),"global_source330_row":int(pp+14316),"held_mass_kg":float(held_by_parent[pp]),"remaining_M4_kg":float(q[pp].sum()),"independent_fullrelease_capacity_to_deficits_kg":float(potential_by_parent[pp])})
direct_capacity=np.minimum(capsum,head)
covers=direct_capacity+1e-12*np.maximum(need,1e-18)>=need
uncovered=np.flatnonzero(short&~covers)
covered=np.flatnonzero(short&covers)
percorner_ratio=np.divide(cornercap,q,out=np.zeros_like(q),where=q>0)
report={"schema":"numi.human.current406-source-only-residual-capacity-census.v1","status":"SOURCE_ONLY_RESIDUAL_CAPACITY_BOUND_DIAGNOSTIC","physical_admission":False,"native_steps":0,"complete_allocation":False,"scope":"Optimistic independent per-site upper-capacity census on pinned 406 graph. Sums single-site caps separately by receiver and source corner, so shared parent/receiver coupling is not solved; passing a bound is not a feasible allocation. Inputs and current406 geometry only.","inputs":{str(x):sha(x) for x in [graph,target,support,M,*stages]},"counts":{"graph_rows":int(n),"rows_with_positive_local_capacity":nonzero,"rows_with_residual_corner_support":eligible,"rows_with_optimistic_fullrelease_capacity":full_nonzero,"children_with_lower_deficit":int(short.sum()),"deficit_children_with_direct_site_capacity_bound_cover":int(len(covered)),"deficit_children_below_direct_site_capacity_bound":int(len(uncovered)),"deficit_children_no_capacity":int(np.count_nonzero(short&(capsum<=0))),"deficit_children_uncovered_even_by_allheldrelease_bound":int(len(full_uncovered)),"prior_stage_rows":[int(np.load(x,allow_pickle=False)["flow_mass_kg"].shape[0]) for x in stages],"capacity_census_chunk_rows":200000},"mass_kg":{"source_corner_residual":float(q.sum()),"receiver_lower_deficit":float(need.sum()),"receiver_upper_headroom":float(np.maximum(head,0).sum()),"sum_independent_site_row_caps":cap_total,"allheldrelease_independent_site_capacity_bound":fullcap_total,"allheldrelease_deficit_child_capacity_bound":float(np.minimum(fullcapsum[short],950.0*vol[short]).sum()),"deficit_mass_uncovered_by_allheldrelease_bound_kg":float(need[full_uncovered].sum()),"sum_deficit_receiver_individual_caps":float(np.minimum(capsum[short],np.maximum(head[short],0)).sum()),"uncovered_deficit_total":float(need[uncovered].sum()),"covered_deficit_total":float(need[covered].sum())},"metrics":{"max_child_deficit_kg":float(need.max(initial=0)),"max_single_child_capacity_kg":float(capsum.max(initial=0)),"max_parent_corner_capacity_over_q":float(percorner_ratio.max(initial=0)),"total_source_corner_capacity_over_q_sum":float(cornercap.sum()/max(q.sum(),1e-300)),"top_held_parents_with_deficit_capacity":top,"worst_uncovered_child_rows":sorted([{"child":int(c),"need_kg":float(need[c]),"direct_capacity_bound_kg":float(min(capsum[c],max(head[c],0))),"site_rows":int(by_receiver_n[c])} for c in uncovered],key=lambda x:(-x["need_kg"],x["child"]))[:100],"worst_covered_child_rows":sorted([{"child":int(c),"need_kg":float(need[c]),"direct_capacity_bound_kg":float(min(capsum[c],max(head[c],0))),"site_rows":int(by_receiver_n[c])} for c in covered],key=lambda x:(-x["need_kg"],x["child"]))[:20]},"elapsed_seconds":time.time()-t0,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
reportpath=out/"report.json";reportpath.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"counts":report["counts"],"mass_kg":report["mass_kg"],"metrics":{k:v for k,v in report["metrics"].items() if not k.startswith("worst_")},"elapsed_seconds":report["elapsed_seconds"],"peak_rss_bytes":report["peak_rss_bytes"],"report":str(reportpath),"sha256":sha(reportpath)},indent=2))
