#!/usr/bin/env python3
"""Exact-key splice of the frozen TetGen 484 component into the D480 volume."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
R=H/"fat-comp0-connected-warning-repair-412"
SRC=R/"run-033-target-rebind-480/result-002/fat-current-typed-target-rebound.npz"
SHELL=R/"run-034-compact-prep-481/preparation-002/connected-cavity-shells.npz"
PREP=R/"run-034-compact-prep-481/preparation-002/report.json"
DIAG=R/"run-035-current-target-480-comp0-interior128/diagnostic-003/report.json"
CASE=R/"run-035-current-target-480-comp0-interior128/component-0000"
REMESH=H/"remesh_fat614_compact_cavities_484_root.py"
VOL=H/"d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/transfer-result-001/fat-D480-short-edge-quotient-volume-proposal.npz"
VOLREPORT=H/"d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/transfer-result-001/report.json"
PINS={
 str(SRC):"1bdff4e0007fdc2ab5867830f20ded3f30c70a1bac5a9ffaa0b66577db7913b2",
 str(SHELL):"c5f9889ab26e152144652ae51fde8ce7eff6a673038cc110b3f34f0cd7ba46b8",
 str(PREP):"8de29b9c440759b4c15fe71402355578601db8eb99182eec111a378e8ac8f835",
 str(DIAG):"f7d372d186e2b8aafbbfc78594a2b9dfc63d8a11ce5d329dd440a78603099c1a",
 str(REMESH):"edcdb4d52b179c6f06885c9ba5de3c738f2d9fb4ef3f8ab3433692048eacfde5",
 str(VOL):"40c90ad65ac924511e5bd98da79f338d15df18839acb099e1c13a6f65058a88c",
}
def sha(p):return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def loadnpz(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def f32key(q):return np.ascontiguousarray(q,dtype=np.float32).view(np.uint32).tobytes()
def table(path):
 lines=[q.strip().split() for q in Path(path).read_text().splitlines() if q.strip() and not q.lstrip().startswith("#")]
 n=int(lines[0][0]);a=np.asarray([[float(x) for x in r] for r in lines[1:]],np.float64)
 if len(a)!=n:raise ValueError("TetGen table row count mismatch: "+str(path))
 return a
def face_key(f):
 a,b,c=map(int,f);return min((a,b,c),(b,c,a),(c,a,b))
def face_sign(f):
 a,b,c=map(int,f);k=face_key(f)
 return 1 if (a,b,c) in (k,(k[1],k[2],k[0]),(k[2],k[0],k[1])) else -1
def oc(faces):
 c=Counter()
 for f in np.asarray(faces,np.int64).reshape(-1,3):c[(face_key(f),face_sign(f))]+=1
 return c
def edgechain(faces):
 c=Counter()
 for a,b,d in np.asarray(faces,np.int64).reshape(-1,3):
  for u,v in ((int(a),int(b)),(int(b),int(d)),(int(d),int(a))):
   c[tuple(sorted((u,v)))]+=1 if u<v else -1
 return {k:v for k,v in c.items() if v}
def exactflux(ip,faces,lookup):
 s=0
 for f in np.asarray(faces,np.int64):
  a,b,c=[ip[lookup[int(v)]] for v in f]
  s+=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
 return s
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--pair-budget",type=int,default=2000000);a=ap.parse_args()
 out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 tic=time.monotonic()
 for p,want in PINS.items():
  got=sha(Path(p))
  if got!=want:raise ValueError(f"input hash mismatch {p}: {got}")
 vr=json.loads(VOLREPORT.read_text())
 if vr.get("candidate",{}).get("sha256")!=PINS[str(VOL)] or vr.get("status")!="PASS_D480_INDEXED379_VOLUME_QUOTIENT_PROPOSAL" or vr.get("physical_admission") is not False:raise ValueError("D480 transfer report mismatch")
 old=loadnpz(SRC);v=loadnpz(VOL);s=loadnpz(SHELL)
 Rm=loadmod("remesh484_splice_488",REMESH);m=Rm.m;M=m.m
 caps=loadmod("caps288_splice_488",H/"direct_connected_skin_caps_288_root.py")
 ep=H/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
 sys.path.insert(0,str(ep.parent.parent));exact=__import__("numilab_human.cardiac_cavity_intersections",fromlist=["_audit_pair"])
 p0=np.asarray(v["points_world_m"],np.float32);pold=np.asarray(old["points_world_m"],np.float32)
 t0=np.asarray(v["tetrahedra"],np.int64);m0=np.asarray(v["material_indices"],np.int32)
 if p0.shape!=(253223,3) or not np.array_equal(p0,pold) or not np.array_equal(p0,np.asarray(v["points_world_m_f32"],np.float32)):raise ValueError("D480 point prefix differs from prepared 480 input")
 if t0.shape!=(802860,4) or len(m0)!=len(t0) or np.any(m0!=16):raise ValueError("unexpected D480 volume domain/material")
 srcrows=np.asarray(s["input_cell_rows"][s["cell_offsets"][0]:s["cell_offsets"][1]],np.int64)
 oldt=np.asarray(old["tetrahedra"],np.int64);oldm=np.asarray(old["material_indices"],np.int32)
 if len(srcrows)!=14048 or len(np.unique(srcrows))!=len(srcrows):raise ValueError("prepared source cell rows invalid")
 keyrow={}
 for i,(tet,mat) in enumerate(zip(t0,m0)):
  k=(tuple(map(int,tet)),int(mat))
  if k in keyrow:raise ValueError("duplicate D480 ordered tet/material key")
  keyrow[k]=i
 rows=[]
 for r in srcrows:
  k=(tuple(map(int,oldt[int(r)])),int(oldm[int(r)]))
  if k not in keyrow:raise ValueError(f"prepared cell key lost in D480 transfer: row {r}")
  rows.append(keyrow[k])
 rows=np.asarray(rows,np.int64)
 if len(np.unique(rows))!=len(rows):raise ValueError("prepared rows map nonuniquely")
 # Map the frozen TetGen 484 outputs by exact F32 coordinates, appending only genuinely new nodes.
 nd=table(CASE/"cavity.1.node");el=table(CASE/"cavity.1.ele");fa=table(CASE/"cavity.1.face")
 ids=nd[:,0].astype(np.int64);x64=nd[:,1:4];x32=x64.astype(np.float32)
 if len(np.unique(ids))!=len(ids):raise ValueError("duplicate TetGen node IDs")
 idloc={int(q):i for i,q in enumerate(ids)}
 lookup0={}
 for i,q in enumerate(p0):
  k=f32key(q)
  if k in lookup0:raise ValueError("duplicate source F32 coordinate")
  lookup0[k]=i
 gm=np.full(len(ids),-1,np.int64);newpts=[];newlookup={}
 for i,q in enumerate(x32):
  k=f32key(q)
  if k in lookup0:gm[i]=lookup0[k]
  elif k in newlookup:gm[i]=newlookup[k]
  else:
   gm[i]=len(p0)+len(newpts);newlookup[k]=int(gm[i]);newpts.append(q.copy())
 if len(newpts)!=44:raise ValueError(f"expected 44 new F32 points, got {len(newpts)}")
 eids=el[:,1:5].astype(np.int64)
 try:ot=np.asarray([[idloc[int(q)] for q in r] for r in eids],np.int64)
 except KeyError as e:raise ValueError(f"unknown TetGen element node {e}")
 if len(ot)!=13675 or el.shape[1]<6:raise ValueError("TetGen element domain changed")
 matnew=el[:,5].astype(np.int32)
 if np.any(matnew!=16):raise ValueError("TetGen material not 16")
 pp=np.vstack([p0,np.asarray(newpts,np.float32)]);gt=gm[ot]
 if np.any(gt<0):raise ValueError("unmapped output tet vertex")
 # The only appended boundary point is a declared D target-edge refinement point.
 face_node_ids=set(map(int,fa[1:,1:4].astype(np.int64).ravel()))
 new_boundary=sorted({int(gm[idloc[q]]) for q in face_node_ids if f32key(x32[idloc[q]]) not in lookup0})
 if new_boundary!=[253251]:raise ValueError(f"expected boundary F32 point at global row253251, got {new_boundary}")
 bp=253251;wantbp=np.asarray([-0.29879412055015564,-0.9244550466537476,0.10537811368703842],np.float32)
 if not np.array_equal(pp[bp].view(np.uint32),wantbp.view(np.uint32)) or hashlib.sha256(pp[bp].tobytes()).hexdigest()!="5719b56f5a0569b0fd73efd2212fc6ae7631084c70e5a03787068e716ab4a0b6":raise ValueError("boundary point bits/hash differ")
 # Reproduce the local shell and exact 2->4 face split from the output tetrahedra.
 grows=np.asarray(s["global_point_rows"][s["point_offsets"][0]:s["point_offsets"][1]],np.int64)
 lf=np.asarray(s["local_faces"][s["face_offsets"][0]:s["face_offsets"][1]],np.int64);shell=grows[lf]
 parents=np.asarray([[131938,132166,245998],[131938,132165,132166]],np.int64)
 shell_rows=[];target=np.asarray(v["proposed_current_desired_boundary_faces"],np.int64);target_rows=[]
 for f in parents:
  k=tuple(sorted(map(int,f)))
  hs=[i for i,q in enumerate(shell) if tuple(sorted(map(int,q)))==k]
  ht=[i for i,q in enumerate(target) if tuple(sorted(map(int,q)))==k]
  if len(hs)!=1 or len(ht)!=1:raise ValueError("target split parent face not uniquely present in shell and current target")
  shell_rows.append(hs[0]);target_rows.append(ht[0])
 for sr,tr in zip(shell_rows,target_rows):
  if oc(shell[sr:sr+1])!=oc(target[tr:tr+1]):raise ValueError("shell/target parent orientation mismatch")
 oblocal=m.boundary(ot);ob=gm[oblocal]
 patch=ob[np.any(ob==bp,axis=1)]
 if patch.shape!=(4,3) or set(map(int,patch.ravel()))!={131938,132166,132165,245998,bp}:raise ValueError("TetGen target patch is not the expected 2-to-4 split")
 if edgechain(shell[shell_rows])!=edgechain(patch):raise ValueError("patch outer directed edge chain differs")
 shell2=np.vstack([np.asarray([f for i,f in enumerate(shell) if i not in set(shell_rows)],np.int64),patch])
 if oc(shell2)!=oc(ob):raise ValueError("exact local shell/cut chain differs after boundary refinement")
 # Exact point-location check for the 43 interior Steiner nodes in the refined shell.
 used_shell=np.unique(shell2);sl=np.full(len(pp),-1,np.int64);sl[used_shell]=np.arange(len(used_shell));local_shell=sl[shell2]
 prepared=exact.prepare_float32_point_location_meters(pp[used_shell],local_shell)
 interior=np.asarray([q for q in range(len(p0),len(pp)) if q!=bp],np.int64);locations=Counter();badloc=[]
 for q in interior:
  loc=exact.point_location_prepared_indexed(tuple(Fraction(float(x)) for x in pp[int(q)]),prepared)
  locations[str(loc.get("location"))]+=1
  if loc.get("location")!="inside":badloc.append((int(q),loc))
 if badloc:raise ValueError(f"new interior points outside refined shell: {badloc[:3]}")
 # Exact revised-shell self intersection and exact tetrahedron positivity/flux.
 rec,_=exact.float32_surface_records(pp,shell2,source_seam_groups=());selfcheck=exact._audit_pair(rec,rec,same_surface=True)
 if int(selfcheck.get("count",-1))!=0:raise ValueError(f"refined shell self intersection: {selfcheck}")
 det,q=M.measurements(x32,ot)
 if np.any(det<=0):raise ValueError("new F32 tetrahedron nonpositive")
 uniq=np.unique(gt);ilook={int(g):i for i,g in enumerate(uniq)};ip=M.integer_points(pp[uniq])
 exactdet=[M.exact_det(ip,[ilook[int(x)] for x in tet]) for tet in gt]
 if any(x<=0 for x in exactdet):raise ValueError("new exact dyadic determinant nonpositive")
 used=np.unique(np.r_[shell.ravel(),shell2.ravel(),gt.ravel()]);ip2=M.integer_points(pp[used]);glook={int(g):i for i,g in enumerate(used)}
 oldflux=exactflux(ip2,shell,glook);newflux=exactflux(ip2,shell2,glook);tetflux=sum(exactdet)
 if newflux!=tetflux:raise ValueError("exact tetrahedral flux differs from refined shell")
 od,oq=M.measurements(p0,t0[rows]);oldbad=int(np.count_nonzero(~M.passes(od,oq)));newbad=int(np.count_nonzero(~M.passes(det,q)))
 # Rebase and compose by exact ordered tet+material key.
 removed=np.zeros(len(t0),bool);removed[rows]=True;keep=np.flatnonzero(~removed)
 outt=np.vstack([t0[keep],gt]);outm=np.r_[m0[keep],matnew]
 bound=m.boundary(outt)
 oldtarget=np.asarray(v["proposed_current_desired_boundary_faces"],np.int64)
 target_keep=np.asarray([i for i in range(len(oldtarget)) if i not in set(target_rows)],np.int64)
 refined_target=np.vstack([oldtarget[target_keep],patch])
 miss=oc(refined_target)-oc(bound);extra=oc(bound)-oc(refined_target)
 if miss or extra:raise ValueError(f"whole-volume oriented boundary mismatch vs refined current target: miss={sum(miss.values())}, extra={sum(extra.values())}")
 # Exact new-vs-retained and new-vs-new positive-volume overlap test.
 tree=caps.TetBVH(p0,t0)
 sat=caps.check_intersections(pp,t0,gt,rows,[],tree,max_pairs=a.pair_budget)
 if not sat.get("pass"):raise ValueError(f"exact retained/new SAT failed: {sat}")
 if np.any(outm!=16):raise ValueError("composed material changed")
 if len(np.unique(np.sort(outt,axis=1),axis=0))!=len(outt):raise ValueError("duplicate unordered tetrahedra after composition")
 gd,gq=M.measurements(pp,outt);global_bad=int(np.count_nonzero(~M.passes(gd,gq)))
 if np.any(gd<=0):raise ValueError("composed volume has nonpositive F32 tetrahedra")
 # Full row maps: immediate current D480 parent plus source 480 row lineage.
 in2out=np.full(len(t0),-1,np.int64);in2out[keep]=np.arange(len(keep),dtype=np.int64)
 out2parent=np.r_[keep,np.full(len(gt),-1,np.int64)]
 out2source=np.r_[np.asarray(v["output_cell_to_input_cell"],np.int64)[keep],np.full(len(gt),-1,np.int64)]
 node2parent=np.r_[np.arange(len(p0),dtype=np.int64),np.full(len(pp)-len(p0),-1,np.int64)]
 # Update the target face list and its source metadata; synthetic D face IDs remain explicitly unmapped pending fresh 397 rebind.
 groups=np.asarray(v["proposed_current_desired_boundary_source_group"],np.int8)
 sr=np.asarray(v["proposed_current_desired_boundary_source_rows"],np.int64)
 sfr=np.asarray(v["proposed_current_desired_boundary_source_face_rows"],np.int64)
 if not(len(groups)==len(sr)==len(sfr)==len(oldtarget)):raise ValueError("target crosswalk arrays malformed")
 groups=np.r_[groups[target_keep],np.ones(4,np.int8)];sr=np.r_[sr[target_keep],np.full(4,-1,np.int64)];sfr=np.r_[sfr[target_keep],np.full(4,-1,np.int64)]
 payload=dict(v)
 payload.update({
  "points_world_m":pp,"points_world_m_f32":pp.copy(),"tetrahedra":outt,"material_indices":outm,"boundary_faces":bound,
  "proposed_current_desired_boundary_faces":refined_target,"proposed_current_desired_boundary_source_group":groups,
  "proposed_current_desired_boundary_source_rows":sr,"proposed_current_desired_boundary_source_face_rows":sfr,
  "input_cell_to_output_cell":in2out,"output_cell_to_input_cell":out2parent,"retained_input_cell_rows":keep,
  "output_node_to_input_node":node2parent,
  "comp0_parent_candidate_path":np.asarray([str(VOL)]),"comp0_parent_candidate_sha256":np.asarray([sha(VOL)]),
  "comp0_prepared_source_candidate_path":np.asarray([str(SRC)]),"comp0_prepared_source_candidate_sha256":np.asarray([sha(SRC)]),
  "comp0_selected_parent_cell_rows_in_D480":rows,"comp0_selected_parent_cell_rows_in_source480":srcrows,
  "comp0_source480_to_D480_current_cell_rows":rows,
  "comp0_replacement_output_cell_rows":np.arange(len(keep),len(outt),dtype=np.int64),
  "comp0_replacement_output_tetrahedra_global_node_rows":gt,"comp0_replacement_output_material_indices":matnew,
  "comp0_added_global_point_rows":np.arange(len(p0),len(pp),dtype=np.int64),
  "comp0_boundary_steiner_global_point_row":np.asarray([bp],np.int64),"comp0_boundary_steiner_coordinate_f32":pp[bp],
  "comp0_boundary_refined_target_faces":patch,"comp0_retired_target_faces_global":parents,
  "comp0_old_selected_shell_faces_global":shell,"comp0_refined_selected_shell_faces_global":shell2,
  "comp0_tetgen_node_to_global_rows":gm,"comp0_tetgen_output_tets_local_node_rows":ot,
  "comp0_output_cell_to_parent_D480_rows":out2parent,"comp0_output_cell_to_source480_rows":out2source,
  "comp0_exact_sat_checked_pairs":np.asarray([sat.get("checked_pairs",0)],np.int64),
  "comp0_exact_shell_flux_det6":np.asarray([str(newflux)],dtype="U256"),
  "comp0_exact_tet_flux_det6":np.asarray([str(tetflux)],dtype="U256"),
  "comp0_quality_delta":np.asarray([newbad-oldbad],np.int64),
 })
 outdir=a.output.resolve();outdir.mkdir(parents=True,exist_ok=False)
 candidate=outdir/"fat-comp0-reused484-spliced-volume-proposal.npz";np.savez_compressed(candidate,**payload)
 saved=loadnpz(candidate)
 for k,arr in (("points_world_m",pp),("tetrahedra",outt),("material_indices",outm),("boundary_faces",bound),("proposed_current_desired_boundary_faces",refined_target)):
  if not np.array_equal(saved[k],arr):raise ValueError(f"serialized candidate mismatch {k}")
 report={
  "schema":"numi.fat614.comp0-reused-484-tetgen-volume-splice.v1","status":"PASS_EXACT_COMPOSED_COMP0_TARGET_VOLUME_SOURCE_ONLY",
  "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
  "inputs":{"source480":{"path":str(SRC),"sha256":sha(SRC)},"D480_volume":{"path":str(VOL),"sha256":sha(VOL),"report_path":str(VOLREPORT),"report_sha256":sha(VOLREPORT)},"prepared_shell":{"path":str(SHELL),"sha256":sha(SHELL),"prep_report_path":str(PREP),"prep_report_sha256":sha(PREP)},"TetGen_node":{"path":str(CASE/"cavity.1.node"),"sha256":sha(CASE/"cavity.1.node")},"TetGen_ele":{"path":str(CASE/"cavity.1.ele"),"sha256":sha(CASE/"cavity.1.ele")},"TetGen_face":{"path":str(CASE/"cavity.1.face"),"sha256":sha(CASE/"cavity.1.face")},"diagnostic":{"path":str(DIAG),"sha256":sha(DIAG)},"remesher":{"path":str(REMESH),"sha256":sha(REMESH)}},
  "rebase":{"method":"exact ordered tet corner tuple + material","prepared_source_rows":len(srcrows),"D480_current_rows":len(rows),"all_selected_exact_keys_recovered":True,"all_points_bitwise_unchanged":True},
  "mesh":{"TetGen_nodes":len(ids),"TetGen_tets":len(gt),"output_boundary_faces":len(ob),"new_point_count":len(newpts),"interior_point_count":len(interior),"boundary_point_row":bp,"boundary_point_f32":pp[bp].tolist(),"boundary_point_f32_sha256":hashlib.sha256(pp[bp].tobytes()).hexdigest(),"all_F32_positive":True,"all_exact_positive":True,"material_16":True,"duplicate_tets":0},
  "boundary_refinement":{"two_retired_target_faces":[f.tolist() for f in parents],"four_child_boundary_faces":patch.tolist(),"outer_oriented_chain_preserved":True,"selected_shell_missing":0,"selected_shell_extra":0,"whole_target_missing":0,"whole_target_extra":0,"fresh_D_target_rebind_required":True},
  "exact":{"new_tet_flux_det6":str(tetflux),"refined_shell_flux_det6":str(newflux),"flux_equal":True,"shell_self_intersection_pairs":int(selfcheck.get("count",-1)),"retained_and_new_sat":sat,"interior_point_locations":dict(locations),"all_interior_points_inside_refined_shell":True},
  "quality":{"old_local_failures":oldbad,"new_local_failures":newbad,"local_failure_delta":newbad-oldbad,"global_failures_after":global_bad,"native_gate_changed":False},
  "candidate":{"path":str(candidate),"sha256":sha(candidate),"size_bytes":candidate.stat().st_size,"points":len(pp),"cells":len(outt),"actual_boundary_faces":len(bound),"target_boundary_faces":len(refined_target)},
  "limitations":"This is an explicit inferred D boundary refinement and positive volume splice, not exact old-surface support and not native/physical/mass admission. The new point is 24.1139 nm off the old edge and 20.6437 nm from the old source-face plane; 43 interior points carry construction ancestry pending source330/P1/mass rebind."
 }
 rp=outdir/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+"\n")
 print(json.dumps({"status":report["status"],"candidate":str(candidate),"sha256":sha(candidate),"report":str(rp),"report_sha256":sha(rp),"boundary":{"actual":len(bound),"target":len(refined_target),"missing":0,"extra":0},"local_quality":[oldbad,newbad,newbad-oldbad],"global_quality_failures":global_bad,"sat":sat,"time_s":time.monotonic()-tic},indent=2))
if __name__=="__main__":main()
