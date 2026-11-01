#!/usr/bin/env python3
"""Narrow negative controls for retained empty-cap transfer and exact typed-I lineage."""
from pathlib import Path
from itertools import combinations
import hashlib, importlib.util, json, sys
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=H/"quality-partial-cavity-684-root"
OWNER=D/"paired_connected_boundary_patch_batch_883_targeted_source_edge_innerflip_root.py"
def sha(p): return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
spec=importlib.util.spec_from_file_location("cap_owner_983",OWNER)
B=importlib.util.module_from_spec(spec);sys.modules[spec.name]=B;spec.loader.exec_module(B)
ENG=B.ENG;T=B.T;M=B.M;A=B.A
PARENT=H/"all-quality-972-last-fat-fixed-775-qualified-root/verified-state.json"
QUAL=H/"all-quality-984-retained-I-quad-qualified-root"
PROPOSAL=H/"all-quality-982-retained-I-quad-root"
RECEIPT=QUAL/"receipt.json"
STATE=QUAL/"verified-state.json"
REPORT=PROPOSAL/"report.json"
JOURNAL=PROPOSAL/"retained-seed-patch-journal.json"
REPLAY=PROPOSAL/"independent-retained-seed-replay.json"
OUT=D/"retained_cap_transfer_983_controls.json"

def need(ok,msg):
 if not bool(ok): raise ValueError(msg)

def check_empty_retirement_metadata(volume,selected,source_rows):
 selected=np.asarray(selected,np.int64).reshape(-1)
 need(len(selected)==1,"empty target retirement must name exactly one cap cell")
 need(np.asarray(volume["replacement_tetrahedra"],np.int64).reshape((-1,4)).shape[0]==0,
      "empty target retirement has replacement tetrahedra")
 retired=np.asarray(volume["connected_boundary_patch_empty_target_retirement_cell_rows"],np.int64).reshape(-1)
 need(np.array_equal(retired,selected),"wrong retired cell row")
 gotfaces=np.asarray(volume["connected_boundary_patch_empty_target_retirement_source_face_rows"],np.int64).reshape(-1)
 need(np.array_equal(gotfaces,np.asarray(source_rows,np.int64).reshape(-1)),"empty retirement source-face identity mismatch")
 rp=np.asarray(volume["replacement_parent_offsets"],np.int64)
 rr=np.asarray(volume["replacement_parent_cell_rows"],np.int64)
 need(np.array_equal(rp,np.asarray([0],np.int64)) and rr.size==0,"empty retirement has nonempty replacement-parent CSR")
 inverse=np.asarray(volume["input_cell_to_output_cell"],np.int64)
 need(selected[0]<len(inverse) and inverse[selected[0]]==-1,"retired source cap is not mapped to an explicit removed parent")
 return True

def check_no_good_regression(old_pass,keep_mask,new_pass):
 old=np.asarray(old_pass,bool).reshape(-1);keep=np.asarray(keep_mask,bool).reshape(-1);new=np.asarray(new_pass,bool).reshape(-1)
 need(len(old)==len(keep),"old-pass and keep masks differ in length")
 nk=int(keep.sum())
 need(len(new)>=nk,"new pass mask is shorter than retained material cells")
 need(not np.any(old[keep]&~new[:nk]),"formerly-good retained material cell regressed")
 need(int(np.count_nonzero(~new))<=int(np.count_nonzero(~old)),"whole material native bad count regressed")
 return True

def exact_det_sum(points,tets):
 pp=np.asarray(points,np.float32);tt=np.asarray(tets,np.int64).reshape((-1,4))
 ip={int(i):ENG.S.S.ip(pp[int(i)]) for i in np.unique(tt.ravel())}
 vals=[int(ENG.S.S.det(*(ip[int(i)] for i in tet))) for tet in tt]
 return vals,sum(vals)

def check_empty_cap_transfer(points,full_tets,selected_rows,replacement_tets,old_faces,new_faces,retirement_rows,target_override=None):
 pp=np.asarray(points,np.float32);tets=np.asarray(full_tets,np.int64).reshape((-1,4))
 sel=np.asarray(selected_rows,np.int64).reshape(-1);new=np.asarray(replacement_tets,np.int64).reshape((-1,4))
 oldf=np.asarray(old_faces,np.int64).reshape((-1,3));newf=np.asarray(new_faces,np.int64).reshape((-1,3))
 need(len(sel)==1,"empty cap is only allowed for a single explicit cap cell")
 need(np.array_equal(np.asarray(retirement_rows,np.int64).reshape(-1),sel),"wrong retired tet for empty target")
 oldsub=tets[sel];oldbd=np.asarray(T.m.boundary(oldsub),np.int64);oldchain=ENG.chain(oldbd)
 for f in oldf:
  need(oldchain.get(tuple(sorted(map(int,f))),0)==ENG.fsign(f),"changed source face is not on retired cap boundary")
 local_new=np.asarray(ENG.oriented_boundary_replace(oldbd,list(zip(oldf,newf))),np.int64)
 need(len(new)==0 and len(local_new)==0 and not ENG.chain(local_new),
      "empty local target is not exactly the source-flip cap shell")
 keep=np.ones(len(tets),bool);keep[sel]=False
 retained=tets[keep]
 need(len(retained)>0,"empty cap would remove the entire material domain")
 combined=np.vstack((retained,new)).astype(np.int64)
 need(len(np.unique(np.sort(combined,axis=1),axis=0))==len(combined),"duplicate full-domain tetrahedra")
 oldbd_full=np.asarray(T.m.boundary(tets),np.int64)
 target_expected=np.asarray(ENG.oriented_boundary_replace(oldbd_full,list(zip(oldf,newf))),np.int64)
 target_new=target_expected if target_override is None else np.asarray(target_override,np.int64).reshape((-1,3))
 need(ENG.chain(target_new)==ENG.chain(target_expected),"full target differs from exact changed-target construction")
 actual=np.asarray(T.m.boundary(combined),np.int64)
 need(ENG.chain(actual)==ENG.chain(target_new),"retained full material domain does not match exact changed target")
 need(T.closed(target_new)[0],"changed full-material target is not closed")
 dets,total=exact_det_sum(pp,combined)
 need(len(dets)==len(combined) and min(dets)>0,"retained full material domain is not entirely positive")
 flux=int(ENG.exact_boundary_flux(pp,target_new))
 need(total==flux,"full material exact tet volume differs from changed target flux")
 oldd,oldtotal=exact_det_sum(pp,tets)
 oldflux=int(ENG.exact_boundary_flux(pp,oldbd_full))
 need(oldtotal==oldflux,"old full material domain exact volume does not match old target")
 delta=int(flux-oldflux)
 need(total-oldtotal==delta,"retained full material exact volume delta does not match changed target")
 dd,qq=M.measurements(pp,combined)
 need(np.all(np.asarray(dd)>0),"native readback found a nonpositive material tet")
 native=np.asarray(M.passes(dd,qq),bool)
 need(np.all(native),"retained full material domain has a native quality failure")
 oldd_f,oldq=M.measurements(pp,tets);oldpass=np.asarray(M.passes(oldd_f,oldq),bool)
 check_no_good_regression(oldpass,keep,native)
 return {"status":"PASS_EXPLICIT_EMPTY_CAP_TRANSFER","retained_cells":int(len(retained)),
         "retired_cells":int(len(sel)),"replacement_cells":int(len(new)),
         "full_target_chain":True,"full_domain_positive":True,"full_domain_native_pass":True,
         "exact_flux":str(flux),"exact_volume_delta":str(delta)}

def csr_values(z,offset_key,lanes,row):
 off=np.asarray(z[offset_key],np.int64);lo,hi=map(int,off[int(row):int(row)+2])
 if offset_key=="quality_star_patch_union_offsets": keys=ENG.TUN
 else: keys=ENG.ANC
 return {tuple(int(z[k][j]) for k in keys) for j in range(lo,hi)}

def check_source_typed_lineage(parent,out,changed_rows,new_faces,parent_path):
 pf=np.asarray(parent["faces"],np.int64);of=np.asarray(out["faces"],np.int64)
 rows=np.asarray(changed_rows,np.int64).reshape(-1);nf=np.asarray(new_faces,np.int64).reshape((-1,3))
 need(len(rows)==len(nf) and len(set(map(int,rows)))==len(rows),"typed-I changed face roster mismatch")
 want=pf.copy();want[rows]=nf
 need(np.array_equal(of,want),"typed-I face rows differ from exact patch")
 for k in ("source_input_face_rows","source_input_vertex_quotient"):
  if k in parent or k in out:
   need(k in parent and k in out and np.array_equal(np.asarray(parent[k]),np.asarray(out[k])),
        "typed-I immediate source identity changed: "+k)
 changed=set(map(int,rows))
 q0=np.asarray(parent["quality_star_face_output_to_input_row"],np.int64)
 q1=np.asarray(out["quality_star_face_output_to_input_row"],np.int64)
 expected=q0.copy();expected[rows]=-1
 need(np.array_equal(q1,expected),"typed-I changed face exact-parent map mismatch")
 pu=set().union(*(csr_values(parent,"quality_star_patch_union_offsets",ENG.TUN,int(r)) for r in rows))
 pa=set().union(*(csr_values(parent,"face_ancestry_offsets",ENG.ANC,int(r)) for r in rows))
 for r in rows:
  need(csr_values(out,"quality_star_patch_union_offsets",ENG.TUN,int(r))==pu,"typed-I changed face parent union mismatch")
  need(csr_values(out,"face_ancestry_offsets",ENG.ANC,int(r))==pa,"typed-I changed face ancestry mismatch")
  first=sorted(pu)[0]
  for j,k in enumerate(ENG.TSC):
   need(int(np.asarray(out[k])[int(r)])==int(first[j]),"typed-I direct source identity does not equal canonical parent union: "+k)
 psha=sha(parent_path)
 for k in ("typed_parent_candidate_path","source_current_candidate_path"):
  if k in out:
   need(str(np.asarray(out[k]).reshape(-1)[0])==str(Path(parent_path).resolve()),"typed-I parent path alias mismatch: "+k)
 for k in ("typed_parent_candidate_sha256","source_current_candidate_sha256"):
  if k in out:
   need(str(np.asarray(out[k]).reshape(-1)[0])==psha,"typed-I parent SHA alias mismatch: "+k)
 return True

def make_cap_fixture():
 points=np.asarray([(x,y,z) for z in (0.,2.) for y in (0.,2.) for x in (0.,2.)],np.float32)
 raw=np.asarray([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]],np.int64)
 tets=raw.copy()
 for j,tet in enumerate(tets):
  value=int(ENG.S.S.det(*(ENG.S.S.ip(points[int(i)]) for i in tet)))
  if value<0:tets[j,2],tets[j,3]=tets[j,3],tets[j,2]
  if value==0:raise AssertionError("cube fixture contains a degenerate tetrahedron")
 fullbd=np.asarray(T.m.boundary(tets),np.int64)
 if not T.closed(fullbd)[0]:raise AssertionError("cube fixture parent boundary must be closed")
 cap=tets[0:1].copy();capbd=np.asarray(T.m.boundary(cap),np.int64)
 fullkeys={tuple(sorted(map(int,f))) for f in fullbd}
 oldf=np.asarray([f for f in capbd if tuple(sorted(map(int,f))) in fullkeys],np.int64)
 internal=np.asarray([f for f in capbd if tuple(sorted(map(int,f))) not in fullkeys],np.int64)
 if oldf.shape!=(2,3) or internal.shape!=(2,3):raise AssertionError("fixture cap must have two source and two internal faces")
 keep=tets[1:].copy();restbd=np.asarray(T.m.boundary(keep),np.int64)
 restlut={tuple(sorted(map(int,f))):f for f in restbd}
 newf=np.asarray([restlut[tuple(sorted(map(int,f)))] for f in internal],np.int64)
 target=np.asarray(ENG.oriented_boundary_replace(fullbd,list(zip(oldf,newf))),np.int64)
 if not T.closed(target)[0] or ENG.chain(target)!=ENG.chain(restbd):
  raise AssertionError("fixture empty-cap source flip must equal the retained material boundary")
 return points,tets,np.asarray([0],np.int64),np.empty((0,4),np.int64),oldf,newf

def expect_fail(fn,needle):
 try:fn()
 except (ValueError,AssertionError) as e:
  if needle not in str(e):raise AssertionError("expected failure containing "+needle+", got "+str(e))
  return str(e)
 raise AssertionError("negative control unexpectedly passed: "+needle)

def main():
 receipt=json.loads(RECEIPT.read_text());state=json.loads(STATE.read_text());parent_state=json.loads(PARENT.read_text())
 report=json.loads(REPORT.read_text());journal=json.loads(JOURNAL.read_text());replay=json.loads(REPLAY.read_text())
 need(receipt["status"]=="PASS_ENDPOINT_CAVITY_EXACT_REPLAY_775_808_SOURCE_ONLY","candidate is not 775/808 qualified source-only")
 need(receipt["source_proof"] and receipt["surface_crossings"]==0,"fresh source proof is incomplete")
 need(state.get("physical_admission") is False and int(state.get("native_steps",0))==0,"candidate must remain source-only")
 need(state.get("mass_mapping")=="NOT_RUN","mass mapping must remain explicitly pending")
 need(report["independent_replay_status"]=="PASS_INDEPENDENT_RETAINED_SEED_REPLAY","retained-seed replay not PASS")
 need(replay["status"]=="PASS_INDEPENDENT_RETAINED_SEED_REPLAY","readback replay receipt mismatch")
 need(report["paired_exact_signed_delta"]=="0","paired exact volume transfer did not cancel")
 need(report["paired_quality_bad_before"]==5 and report["paired_quality_bad_after"]==4,"candidate quality summary differs from pinned 982 evidence")
 jr=journal[0];source_rows=np.asarray(jr["source_changed_face_rows"],np.int64)
 psrc=B.load_npz(parent_state["source_surfaces"]["I"]["path"])
 osrc=B.load_npz(state["source_surfaces"]["I"]["path"])
 expected_nf=np.asarray(jr["source_patch_new_faces"][0],np.int64)
 check_source_typed_lineage(psrc,osrc,source_rows,expected_nf,parent_state["source_surfaces"]["I"]["path"])
 cap_rows=np.asarray(jr["empty_target_retirement"]["input_cell_rows"],np.int64)
 fatref=state["Fat_current"];fat=B.load_npz(fatref["path"])
 check_empty_retirement_metadata(fat,cap_rows,source_rows)
 # Adversarial metadata controls against the actual serialized candidate.
 wrong=dict(fat);wrong["connected_boundary_patch_empty_target_retirement_cell_rows"]=np.asarray([cap_rows[0]+1],np.int64)
 wrong_retire_error=expect_fail(lambda:check_empty_retirement_metadata(wrong,cap_rows,source_rows),"wrong retired cell row")
 wrongface=dict(fat);wrongface["connected_boundary_patch_empty_target_retirement_source_face_rows"]=source_rows[::-1].copy()
 wrong_typed_error=expect_fail(lambda:check_empty_retirement_metadata(wrongface,cap_rows,source_rows),"source-face identity mismatch")
 # A synthetic two-sided cap transfer exercises the full empty-target rule independently of the 770k-cell arrays.
 pp,tt,sel,nt,oldf,newf=make_cap_fixture()
 positive=check_empty_cap_transfer(pp,tt,sel,nt,oldf,newf,sel)
 wrong_tet=expect_fail(lambda:check_empty_cap_transfer(pp,tt,np.asarray([1]),nt,oldf,newf,np.asarray([1])),"changed source face is not on retired cap boundary")
 unchanged_target=expect_fail(lambda:check_empty_cap_transfer(pp,tt,sel,nt,oldf,oldf,sel),"empty local target is not exactly")
 fullbd=np.asarray(T.m.boundary(tt),np.int64)
 valid_target=np.asarray(ENG.oriented_boundary_replace(fullbd,list(zip(oldf,newf))),np.int64)
 tampered_target=valid_target.copy();tampered_target[0]=tampered_target[0,[0,2,1]]
 changed_target_error=expect_fail(lambda:check_empty_cap_transfer(pp,tt,sel,nt,oldf,newf,sel,target_override=tampered_target),"full target differs from exact changed-target construction")
 single_cell=tt[:1].copy()
 all_retired=expect_fail(lambda:check_empty_cap_transfer(pp,single_cell,np.asarray([0]),nt,oldf,newf,np.asarray([0])),"empty cap would remove the entire material domain")
 # The same quality guard is used for full material replay; force a former-good retained cell to fail.
 oldpass=np.asarray([True,True,False],bool);keep=np.asarray([True,True,False],bool)
 check_no_good_regression(oldpass,keep,np.asarray([True,True,False],bool))
 oldgood_error=expect_fail(lambda:check_no_good_regression(oldpass,keep,np.asarray([True,False,False],bool)),"formerly-good retained material cell regressed")
 # Mutate exact ancestry and immediate typed-I row maps; both must fail closed.
 badsrc=dict(osrc);lane="face_ancestry_parent_face_rows";badsrc[lane]=np.asarray(osrc[lane]).copy()
 changed=int(source_rows[0]);lo=int(osrc["face_ancestry_offsets"][changed]);badsrc[lane][lo]+=1
 ancestry_error=expect_fail(lambda:check_source_typed_lineage(psrc,badsrc,source_rows,expected_nf,parent_state["source_surfaces"]["I"]["path"]),"typed-I changed face ancestry mismatch")
 badsrc2=dict(osrc);badsrc2["source_input_face_rows"]=np.asarray(osrc["source_input_face_rows"]).copy();badsrc2["source_input_face_rows"][changed]+=1
 typed_row_error=expect_fail(lambda:check_source_typed_lineage(psrc,badsrc2,source_rows,expected_nf,parent_state["source_surfaces"]["I"]["path"]),"typed-I immediate source identity changed")
 result={"schema":"h706.retained-cap-transfer-negative-controls.v1","status":"PASS_RETAINED_CAP_TRANSFER_CONTROLS",
  "candidate_receipt":{"path":str(RECEIPT),"sha256":sha(RECEIPT)},
  "candidate_report":{"path":str(REPORT),"sha256":sha(REPORT)},
  "candidate_state":{"path":str(STATE),"sha256":sha(STATE)},
  "retained_patch_journal":{"path":str(JOURNAL),"sha256":sha(JOURNAL)},
  "retained_patch_replay":{"path":str(REPLAY),"sha256":sha(REPLAY)},
  "source_owner":{"path":str(OWNER),"sha256":sha(OWNER)},
  "controls":{"actual_candidate_775_808_proof":"PASS","synthetic_empty_cap_full_domain_positive_exact_target":"PASS",
   "wrong_retired_cell_rejected":wrong_retire_error,"wrong_retirement_source_face_rejected":wrong_typed_error,
   "wrong_tet_cap_rejected":wrong_tet,"wrong_changed_target_rejected":unchanged_target,
   "full_domain_target_mutation_rejected":changed_target_error,
   "empty_entire_material_domain_rejected":all_retired,"former_good_quality_regression_rejected":oldgood_error,
   "changed_source_ancestry_mutation_rejected":ancestry_error,"typed_I_immediate_face_row_mutation_rejected":typed_row_error},
  "candidate_boundary_cases":{"Fat_empty_target_retirement":True,"Fat_remaining_declared_domain_cells":288,
   "Skin_replacement_cells":2,"paired_signed_volume_delta":"0"},
  "physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"}
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
