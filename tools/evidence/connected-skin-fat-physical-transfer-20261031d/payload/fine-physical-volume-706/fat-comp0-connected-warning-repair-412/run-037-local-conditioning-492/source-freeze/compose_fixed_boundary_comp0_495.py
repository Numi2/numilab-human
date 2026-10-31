#!/usr/bin/env python3
"""Compose proven fixed-boundary comp0 connectivity deltas onto the exact SAT parent."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, sys
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
R=H/"fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492"
P=H/"fat-comp0-connected-warning-repair-412/run-036-reused484-volume-splice-488/result-004/fat-comp0-reused484-spliced-volume-proposal.npz"
PR=P.with_name("report.json")
LOCAL=R/"input-comp0-replacement.npz"
LOCAL_RECEIPT=R/"input-comp0-replacement.receipt.json"
BATCH=R/"connectivity-307"
BATCH_RECEIPT=BATCH/"receipt.json"
VERIFY327=R/"verify-327-fat/receipt.json"
FACE=R/"connectivity-313/fixed-boundary-volume-proposal.npz"
FACEREPORT=R/"connectivity-313/report.json"
VERIFY374=R/"verify-374/receipt.json"
NOPROGRESS=R/"connectivity-313-second/report.json"
CAPS=H/"direct_connected_skin_caps_288_root.py"
CORE=H/"fixed_boundary_volume_flip_198_root.py"
SCRIPTS={
 "batch307":(H/"batch_degree3_and_polygon_flips_307_root.py","623db8e9fb002c7d93f7c8b9b31900aecff62b9b53ef175fa38936b1a16858e8"),
 "verify327":(H/"verify_fixed_boundary_connectivity_batch_327_root.py","fc1ea9f8d84149a759006b7532d1e93d46752dfdb5ec91c95aff1d20b4112151"),
 "flip313":(H/"fixed_boundary_face_flips_npz_313_root.py","224935f5978f9f52c4b413dde4ea31bbe8e52ae6ca27a399547d136d04095ab1"),
 "verify374":(H/"verify_face_flips_generic_374_root.py","f297b4315865eb2c526e8d6ff3be30bb8f0649515ced791ba2d5cc29c3150e18"),
 "caps288":(CAPS,"4b7c46955fa286db38c6ee5ec1dc91ac427e97ed5184cbe964bc7934640d0791"),
 "predicate198":(CORE,"650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"),
}
PINS={
 str(P):"ed7d84228d5b1d11e9fbf0c39af296c6ace54ce0add867c5cc7b527c46820a08",
 str(PR):"6db9a54d18aca2cb482bbe3d53c64e243d50c8958be78db6fdfe422e13a2b8b5",
 str(LOCAL):"0b13733f7f19e387da8ab1904972224770b58b433b491dfe91aa34e3890a2562",
 str(LOCAL_RECEIPT):"c0258653c1ae95182e374ee34ad8ddfbffbeac6928639af95618788adc7cca7c",
 str(BATCH_RECEIPT):"5e247a2db5df72b5adec577446425c33bf9c425dd0619f9da43379efbe04e729",
 str(VERIFY327):"08d4ef9f469dec3913e76b9ebd2e364ca97b036597bb888957a1cf9ffda7d989",
 str(FACE):"f842d681108b86c97cce9d8769b989aebfc87493de1f500f586798039a7641f2",
 str(FACEREPORT):"aaca5cea61c89913b49308ea0bf0df44a91766db2037dfcee358a219b5cb02db",
 str(VERIFY374):"83a32ac3a3b26108f3b69de0083648ad98b77b66bc8bef94415181902d9a7508",
 str(NOPROGRESS):"f583bbaa24480e1e1c6828afea93c0ed3bd23faa007a2eb1511bca28fd40f02b",
}
def sha(path):
 with open(path,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def loadnpz(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def boundary_faces(t,pat):
 faces=np.asarray(t,np.int64)[:,pat].reshape(-1,3)
 sf=np.sort(faces,axis=1);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]));ss=sf[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];counts=np.diff(np.r_[starts,len(faces)])
 if np.any(counts>2):raise ValueError("nonmanifold face incidence in composed domain")
 signs=1-2*(((faces[:,0]>faces[:,1]).astype(np.int8)+(faces[:,0]>faces[:,2]).astype(np.int8)+(faces[:,1]>faces[:,2]).astype(np.int8))%2)
 summed=np.add.reduceat(signs[order],starts)
 if np.any(summed[counts==2]!=0):raise ValueError("same-direction internal face incidence")
 rows=starts[counts==1]
 return faces[order[rows]]
def oriented_chain(faces):
 out={}
 for f in np.asarray(faces,np.int64).reshape(-1,3):
  a,b,c=map(int,f);key=tuple(sorted((a,b,c)));sgn=1 if ((a>b)+(a>c)+(b>c))%2==0 else -1
  out[key]=out.get(key,0)+sgn
 return {k:v for k,v in out.items() if v}
def chain_sha(chain):
 rows=np.asarray([(k[0],k[1],k[2],int(s)) for k,s in sorted(chain.items())],dtype="<i8")
 return hashlib.sha256(rows.tobytes()).hexdigest()
def edge_keys(t,npoints,patterns):
 e=np.sort(np.asarray(t,np.int64)[:,patterns].reshape(-1,2),axis=1)
 return np.unique(e[:,0].astype(np.int64)*int(npoints)+e[:,1].astype(np.int64))
def exact_new_edges(old,new,pp,outside_edges,patterns,npoints):
 fresh=np.setdiff1d(edge_keys(new,npoints,patterns),edge_keys(old,npoints,patterns),assume_unique=True)
 if not len(fresh):return 0
 ix=np.searchsorted(outside_edges,fresh);hit=ix<len(outside_edges)
 hit[hit]&=outside_edges[ix[hit]]==fresh[hit]
 if np.any(hit):raise ValueError("operation introduces an edge already present in retained mesh: "+repr(fresh[np.flatnonzero(hit)[:4]].tolist()))
 return len(fresh)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 for p,w in PINS.items():
  g=sha(Path(p))
  if g!=w:raise ValueError(f"pinned input hash mismatch: {p} {g}")
 for _,(p,w) in SCRIPTS.items():
  g=sha(p)
  if g!=w:raise ValueError(f"owner source hash mismatch: {p} {g}")
 started=time.monotonic();m=loadmod("comp0fixed198",CORE);caps=loadmod("comp0caps288",CAPS)
 base=loadnpz(P);loc=loadnpz(LOCAL)
 parent_report=json.loads(PR.read_text());br=json.loads(BATCH_RECEIPT.read_text());vr327=json.loads(VERIFY327.read_text());fr=json.loads(FACEREPORT.read_text());vr374=json.loads(VERIFY374.read_text());nr=json.loads(NOPROGRESS.read_text())
 if parent_report.get("status")!="PASS_EXACT_COMPOSED_COMP0_TARGET_VOLUME_SOURCE_ONLY" or parent_report.get("physical_admission") is not False:raise ValueError("parent candidate receipt is not source-only pass")
 if parent_report["exact"]["retained_and_new_sat"].get("pass") is not True:raise ValueError("parent exact SAT proof absent")
 if vr327.get("status")!="PASS_INDEPENDENT_FIXED_BOUNDARY_VOLUME_CONNECTIVITY_REPLAY" or vr327.get("role")!="Fat" or vr327.get("operations")!=443:raise ValueError("Fat role-correct 327 receipt mismatch")
 if vr374.get("status")!="PASS_INDEPENDENT_SOURCE198_FACE_FLIP_REPLAY" or vr374.get("operations")!=5:raise ValueError("374 receipt mismatch")
 if fr.get("passing_operations")!=5 or nr.get("passing_operations")!=0:raise ValueError("313 receipts mismatch")
 pp=np.asarray(base["points_world_m"],np.float32);tfull=np.asarray(base["tetrahedra"],np.int64);mfull=np.asarray(base["material_indices"],np.int32)
 if not np.array_equal(pp,base["points_world_m_f32"]):raise ValueError("parent canonical F32 point view differs")
 localp=np.asarray(loc["points_world_m"],np.float32);cur_t=np.asarray(loc["tetrahedra"],np.int64);cur_m=np.asarray(loc["material_indices"],np.int32)
 rows=np.asarray(base["comp0_replacement_output_cell_rows"],np.int64);source_rows=np.asarray(loc["source_global_cell_rows"],np.int64)
 if len(rows)!=13675 or not np.array_equal(rows,source_rows):raise ValueError("local source rows differ from parent comp0 replacement rows")
 if not np.array_equal(localp,pp) or not np.array_equal(tfull[rows],cur_t) or not np.array_equal(mfull[rows],cur_m):raise ValueError("local patch does not exactly extract parent cells")
 if not np.all(cur_m==16):raise ValueError("local material changed")
 ancestry=[{int(r)} for r in rows]
 patterns=np.asarray(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)),np.int64)
 keep_full=np.ones(len(tfull),bool);keep_full[rows]=False
 outside_edges=edge_keys(tfull[keep_full],len(pp),patterns)
 edge_checked=0
 history=br["history"]
 if len(history)!=4 or br.get("passing_operations")!=443 or br.get("bad_cells_before")!=1954 or br.get("bad_cells_after")!=1481:raise ValueError("307 batch summary mismatch")
 for h in history:
  k=int(h["pass"]);folder=BATCH/("pass-%03d"%k);delta=folder/"exact-connectivity-delta.npz";rp=folder/"report.json"
  if sha(delta)!=h["delta_sha256"] or sha(rp)!=h["report_sha256"]:raise ValueError("307 delta/report pin mismatch")
  rr=json.loads(rp.read_text())
  with np.load(delta,allow_pickle=False) as z:d={x:np.array(z[x],copy=True) for x in z.files}
  parents=d["replacement_parent_cell_rows"].astype(np.int64);po=d["replacement_parent_offsets"].astype(np.int64);news=d["replacement_tetrahedra"].astype(np.int64);no=d["replacement_tet_offsets"].astype(np.int64);newm=d["replacement_material_indices"].astype(np.int32)
  if po[0]!=0 or no[0]!=0 or po[-1]!=len(parents) or no[-1]!=len(news) or len(po)!=len(no):raise ValueError("307 offsets malformed")
  if len(np.unique(parents))!=len(parents) or np.any(parents<0) or np.any(parents>=len(cur_t)):raise ValueError("307 parent rows invalid or duplicated")
  used_vertices=set();op_anc=[]
  for oi in range(len(po)-1):
   pr=parents[po[oi]:po[oi+1]];nt=news[no[oi]:no[oi+1]]
   if not len(pr) or not len(nt) or len(set(map(int,cur_m[pr])))!=1 or np.any(newm[no[oi]:no[oi+1]]!=cur_m[pr[0]]):raise ValueError("307 operation material/shape mismatch")
   verts=set(map(int,cur_t[pr].ravel()))
   if verts&used_vertices:raise ValueError("307 operation vertex stars overlap")
   used_vertices.update(verts)
   if m.certify(pp,cur_t[pr],nt)!=(True,"PASS"):raise ValueError("307 exact local operation certificate failed")
   edge_checked+=exact_new_edges(cur_t[pr],nt,pp,outside_edges,patterns,len(pp))
   op_anc.append(set().union(*(ancestry[int(r)] for r in pr)))
  removed=np.zeros(len(cur_t),bool);removed[parents]=True;kept=np.flatnonzero(~removed)
  nxt_t=np.vstack([cur_t[kept],news]);nxt_m=np.r_[cur_m[kept],newm]
  nxt_anc=[ancestry[int(i)] for i in kept]
  for oi in range(len(po)-1):nxt_anc.extend([op_anc[oi].copy() for _ in range(no[oi+1]-no[oi])])
  if len(nxt_anc)!=len(nxt_t) or len(nxt_m)!=len(nxt_t):raise ValueError("307 reconstruction shape mismatch")
  if int(np.count_nonzero(~m.passes(*m.measurements(pp,nxt_t))))!=int(rr["bad_cells_after"]):raise ValueError("307 replay quality mismatch")
  if len(nxt_t)!=int(rr["output_cells"]):raise ValueError("307 replay output-cell count mismatch")
  cur_t,cur_m,ancestry=nxt_t,nxt_m,nxt_anc
 last307=Path(br["final_proposal"])
 with np.load(last307,allow_pickle=False) as z:last={x:np.array(z[x],copy=True) for x in z.files}
 if sha(last307)!=br["final_proposal_sha256"] or not np.array_equal(cur_t,last["tetrahedra"]) or not np.array_equal(cur_m,last["material_indices"]):raise ValueError("307 final proposal differs from replay")
 if not np.array_equal(last["points_world_m"],pp):raise ValueError("307 moved points")
 face=loadnpz(FACE);parents=np.asarray(face["replacement_parent_cell_rows"],np.int64);new=np.asarray(face["replacement_tetrahedra"],np.int64);keep=np.asarray(face["retained_input_cell_rows"],np.int64)
 if parents.shape!=(5,2) or new.shape!=(5,3,4) or len(keep)!=len(cur_t)-10:raise ValueError("313 map malformed")
 used_vertices=set();face_anc=[]
 for oi,pr in enumerate(parents):
  verts=set(map(int,cur_t[pr].ravel()))
  if verts&used_vertices:raise ValueError("313 operation vertex stars overlap")
  used_vertices.update(verts)
  nt=new[oi]
  if m.certify(pp,cur_t[pr],nt)!=(True,"PASS"):raise ValueError("313 exact local operation certificate failed")
  edge_checked+=exact_new_edges(cur_t[pr],nt,pp,outside_edges,patterns,len(pp))
  face_anc.append(set().union(*(ancestry[int(r)] for r in pr)))
 if not np.array_equal(keep,np.setdiff1d(np.arange(len(cur_t),dtype=np.int64),parents.ravel())):raise ValueError("313 retained row map malformed")
 final_t=np.vstack([cur_t[keep],new.reshape(-1,4)])
 final_m=np.r_[cur_m[keep],np.concatenate([np.full(3,cur_m[pr[0]],np.int32) for pr in parents])]
 final_anc=[ancestry[int(i)] for i in keep]
 for u in face_anc:final_anc.extend([u.copy() for _ in range(3)])
 if not np.array_equal(final_t,face["tetrahedra"]) or not np.array_equal(final_m,face["material_indices"]):raise ValueError("313 final proposal differs from exact replay")
 if len(final_anc)!=len(final_t) or any(not x for x in final_anc):raise ValueError("conditioned cell ancestry missing")
 if not np.array_equal(face["points_world_m"],pp):raise ValueError("313 moved points")
 initial_local_chain=oriented_chain(boundary_faces(np.asarray(loc["tetrahedra"],np.int64),m.PAT))
 final_local_boundary=boundary_faces(final_t,m.PAT)
 if oriented_chain(final_local_boundary)!=initial_local_chain:raise ValueError("conditioned local oriented boundary changed")
 if oriented_chain(final_local_boundary)!=oriented_chain(loc["desired_boundary_faces"]):raise ValueError("local boundary differs from pinned refined shell")
 local_flux_before=caps.exact_volume_int(pp,np.asarray(loc["tetrahedra"],np.int64));local_flux_after=caps.exact_volume_int(pp,final_t)
 if local_flux_before!=local_flux_after:raise ValueError("conditioned exact local signed volume changed")
 used=np.unique(final_t.ravel());lookup=np.full(len(pp),-1,np.int64);lookup[used]=np.arange(len(used));ip=m.integer_points(pp[used])
 exact_det6=[m.exact_det(ip,[int(lookup[int(q)]) for q in tet]) for tet in final_t]
 if any(x<=0 for x in exact_det6):raise ValueError("conditioned exact F32 tetrahedron nonpositive")
 local_det,local_quality=m.measurements(pp,final_t)
 if np.any(local_det<=0) or np.any(final_m!=16):raise ValueError("conditioned local F32/material gate failed")
 keep_rows=np.flatnonzero(keep_full)
 outt=np.vstack([tfull[keep_rows],final_t]);outm=np.r_[mfull[keep_rows],final_m]
 if len(outt)!=len(tfull)-len(rows)+len(final_t):raise ValueError("composed cell count mismatch")
 if not np.array_equal(pp,base["points_world_m"]) or len(pp)!=len(base["points_world_m"]):raise ValueError("point domain changed")
 oldb=boundary_faces(tfull,m.PAT);newb=boundary_faces(outt,m.PAT)
 old_chain=oriented_chain(oldb);new_chain=oriented_chain(newb)
 if old_chain!=new_chain:raise ValueError("global oriented boundary changed")
 target=np.asarray(base["proposed_current_desired_boundary_faces"],np.int64)
 if new_chain!=oriented_chain(target):raise ValueError("global boundary does not match pinned target")
 canon=np.sort(outt,axis=1);order=np.lexsort((canon[:,3],canon[:,2],canon[:,1],canon[:,0]))
 if np.any(np.all(canon[order[1:]]==canon[order[:-1]],axis=1)):raise ValueError("duplicate tetrahedron after composition")
 gdet,gq=m.measurements(pp,outt)
 if np.any(gdet<=0) or np.any(outm!=16):raise ValueError("composed global F32 positive/material gate failed")
 old_to_new=np.full(len(tfull),-1,np.int64);old_to_new[keep_rows]=np.arange(len(keep_rows),dtype=np.int64)
 payload=dict(base);payload["tetrahedra"]=outt;payload["material_indices"]=outm;payload["boundary_faces"]=newb
 payload["output_cell_to_input_cell"]=np.r_[np.asarray(base["output_cell_to_input_cell"],np.int64)[keep_rows],np.full(len(final_t),-1,np.int64)]
 old_i2o=np.asarray(base["input_cell_to_output_cell"],np.int64);i2o=np.full_like(old_i2o,-1);valid=old_i2o>=0;i2o[valid]=old_to_new[old_i2o[valid]];payload["input_cell_to_output_cell"]=i2o
 payload["retained_input_cell_rows"]=np.asarray(base["retained_input_cell_rows"],np.int64)
 payload["comp0_output_cell_to_parent_D480_rows"]=np.r_[np.asarray(base["comp0_output_cell_to_parent_D480_rows"],np.int64)[keep_rows],np.full(len(final_t),-1,np.int64)]
 payload["comp0_output_cell_to_source480_rows"]=np.r_[np.asarray(base["comp0_output_cell_to_source480_rows"],np.int64)[keep_rows],np.full(len(final_t),-1,np.int64)]
 new_patch_rows=np.arange(len(keep_rows),len(outt),dtype=np.int64)
 payload["comp0_replacement_output_cell_rows"]=new_patch_rows
 payload["comp0_replacement_output_tetrahedra_global_node_rows"]=final_t
 payload["comp0_replacement_output_material_indices"]=final_m
 parent_report=json.loads(PR.read_text());old_local_bad=int(parent_report["quality"]["old_local_failures"])
 new_local_bad=int(np.count_nonzero(~m.passes(local_det,local_quality)))
 payload["comp0_quality_delta"]=np.asarray([new_local_bad-old_local_bad],np.int64)
 anc_offsets=np.r_[0,np.cumsum([len(x) for x in final_anc])].astype(np.int64)
 anc_rows=np.asarray([q for group in final_anc for q in sorted(group)],np.int64)
 payload["comp0_conditioned_cell_ancestor_offsets"]=anc_offsets
 payload["comp0_conditioned_cell_ancestor_parent_output_cell_rows"]=anc_rows
 payload["comp0_conditioning_source_candidate_sha256"]=np.asarray([sha(P)])
 payload["comp0_conditioning_parent_union_D480_rows"]=np.asarray(base["comp0_selected_parent_cell_rows_in_D480"],np.int64)
 payload["comp0_conditioning_parent_union_source480_rows"]=np.asarray(base["comp0_selected_parent_cell_rows_in_source480"],np.int64)
 payload["comp0_conditioning_sat_basis"]=np.asarray(["inherited parent SAT; exact local boundary/volume connectivity deltas; no global SAT rerun"])
 old_patch_det,old_patch_q=m.measurements(pp,np.asarray(loc["tetrahedra"],np.int64));old_patch_bad=int(np.count_nonzero(~m.passes(old_patch_det,old_patch_q)))
 if old_patch_bad!=1954 or new_local_bad!=1476:raise ValueError("local quality totals differ from receipts")
 out.mkdir(parents=True,exist_ok=False);candidate=out/"fat-comp0-conditioned-volume-proposal.npz";np.savez_compressed(candidate,**payload)
 with np.load(candidate,allow_pickle=False) as z:
  for k,want in (("points_world_m",pp),("tetrahedra",outt),("material_indices",outm),("boundary_faces",newb),("proposed_current_desired_boundary_faces",target),("comp0_conditioned_cell_ancestor_offsets",anc_offsets),("comp0_conditioned_cell_ancestor_parent_output_cell_rows",anc_rows)):
   if not np.array_equal(z[k],want):raise ValueError("serialized candidate mismatch "+k)
 report={
  "schema":"numi.fat614.comp0-conditioned-fixed-boundary-connectivity.v1",
  "status":"PASS_EXACT_LOCAL_CONNECTIVITY_COMPOSITION_SOURCE_ONLY",
  "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
  "inputs":{"parent_candidate":{"path":str(P),"sha256":sha(P)},"parent_report":{"path":str(PR),"sha256":sha(PR)},"local_input":{"path":str(LOCAL),"sha256":sha(LOCAL)},"307_batch_receipt":{"path":str(BATCH_RECEIPT),"sha256":sha(BATCH_RECEIPT)},"327_Fat_receipt":{"path":str(VERIFY327),"sha256":sha(VERIFY327)},"313_report":{"path":str(FACEREPORT),"sha256":sha(FACEREPORT)},"374_receipt":{"path":str(VERIFY374),"sha256":sha(VERIFY374)},"second_313_no_progress_report":{"path":str(NOPROGRESS),"sha256":sha(NOPROGRESS)},"exact_SAT_owner":{"path":str(CAPS),"sha256":sha(CAPS)}},
  "operations":{"fixed_boundary_edge_stars_307":443,"fixed_boundary_face_flips_313":5,"followup_313_accepted":0,"exact_per_operation_volume_boundary_replay":"PASS","no_point_motion":True,"new_interior_global_edges_checked_against_retained":edge_checked,"global_existing_edge_conflicts":0},
  "local_patch":{"input_cells":len(loc["tetrahedra"]),"conditioned_cells":len(final_t),"input_union_quality_failures":old_patch_bad,"conditioned_union_quality_failures":new_local_bad,"quality_failure_reduction":old_patch_bad-new_local_bad,"material":16,"boundary_faces":len(final_local_boundary),"oriented_boundary_sha256":chain_sha(initial_local_chain),"boundary_unchanged":True,"exact_signed_det6_before":str(local_flux_before),"exact_signed_det6_after":str(local_flux_after),"exact_volume_unchanged":True,"all_exact_determinants_positive":True,"all_F32_determinants_positive":True},
  "whole_candidate":{"points":len(pp),"cells_before":len(tfull),"cells_after":len(outt),"global_quality_failures_after":int(np.count_nonzero(~m.passes(gdet,gq))),"all_F32_determinants_positive":True,"material_16":True,"duplicate_tets":0,"oriented_boundary_faces":len(newb),"global_oriented_boundary_matches_parent":True,"current_target_boundary_matches":True,"target_faces":len(target),"exact_sat":{"status":"INHERITED_FROM_EXACT_PARENT","parent_candidate_sha256":sha(P),"parent_sat_report_sha256":sha(PR),"parent_checked_pairs":int(parent_report["exact"]["retained_and_new_sat"]["checked_pairs"]),"rerun_after_connectivity_changes":False,"basis":"exact local positive fixed-boundary and exact-volume operation replay; no claim of a second whole-mesh SAT run"}},
  "lineage":{"conditioned_tet_to_parent_output_cell_map":"CSR in candidate comp0_conditioned_cell_ancestor_offsets / comp0_conditioned_cell_ancestor_parent_output_cell_rows","patch_source_union_D480_rows":len(payload["comp0_conditioning_parent_union_D480_rows"]),"patch_source_union_source480_rows":len(payload["comp0_conditioning_parent_union_source480_rows"]),"per_tet_direct_D480_parent":"not asserted; complete cavity-level parent union retained"},
  "candidate":{"path":str(candidate),"sha256":sha(candidate),"size_bytes":candidate.stat().st_size},
  "limitations":"Source-only geometric construction/conditioning evidence. Native quality failures remain; no mass mapping or physical admission.",
  "producer_sha256":sha(Path(__file__)),"elapsed_s":time.monotonic()-started}
 (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k not in ("inputs","lineage")},indent=2))
if __name__=="__main__":main()

