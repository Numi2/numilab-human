#!/usr/bin/env python3
from __future__ import annotations
import gc,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
W=Path("/Users/n/numi-human-exact-normalization-m6-1348");R=Path("/Users/n/numi-human-retained-delivery-20261009")
BASE=W/"reports/temporalis-clearance-002/skin-clip-clearance-026/run-003/result-001/temporalis-163-skin-clearance-candidate-026.npz"
OUT=W/"reports/temporalis-clearance-002/skin-clip-clearance-035/run-004/result-001"
MAX_RSS=2_500_000_000
sys.path.insert(0,str(W/"src"))
from numilab_human import common_atlas_skin_clearance as ca
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human import cardiac_cavity_partition as cp
m_spec=importlib.util.spec_from_file_location("clearance026",W/"scripts/temporalis_skin_clip_clearance_026.py");m=importlib.util.module_from_spec(m_spec);m_spec.loader.exec_module(m)
def req(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def volume(v,f):
 t=np.asarray(v,dtype=np.float64)[np.asarray(f,dtype=np.int64)];terms=np.einsum("ij,ij->i",t[:,0],np.cross(t[:,1],t[:,2]),optimize=False)/6.0;return float(__import__("math").fsum(float(x) for x in terms))
def normals(v,f):
 tri=np.asarray(v,dtype=np.float64)[np.asarray(f,dtype=np.int64)];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);out=np.zeros_like(v,dtype=np.float64)
 for col in range(3):np.add.at(out,f[:,col],cross)
 length=np.linalg.norm(out,axis=1);req(np.isfinite(length).all() and np.all(length>1e-14),"collapsed candidate has undefined vertex normal");return (out/length[:,None]).astype("<f4")
def main():
 OUT.mkdir(parents=True,exist_ok=False);progress=OUT/"progress.jsonl";start=time.monotonic()
 req(BASE.is_file(),"base candidate missing");base_sha=sha(BASE)
 with np.load(BASE,allow_pickle=False) as z:
  oldp=np.asarray(z["positions_m"],dtype="<f4");oldf=np.asarray(z["faces"],dtype=np.int64);old_ancestry=json.loads(str(z["face_ancestry_json"].item()));base_meta=json.loads(str(z["candidate_provenance_json"].item()))
 req(len(oldp)==5164 and len(oldf)==10336 and len(old_ancestry)==len(oldf),"pinned 026 candidate population changed")
 keep,drop=4807,4766
 edge_faces=np.flatnonzero(np.any(oldf==keep,axis=1)&np.any(oldf==drop,axis=1));req(len(edge_faces)==2,"selected collapse edge is not exactly two-face incident")
 nk=set(oldf[np.any(oldf==keep,axis=1)].ravel())-{keep};nd=set(oldf[np.any(oldf==drop,axis=1)].ravel())-{drop};common=nk&nd;opposite=set(oldf[edge_faces].ravel())-{keep,drop};req(common==opposite,"selected collapse edge fails link condition")
 newf=oldf.copy();newf[newf==drop]=keep;face_keep=np.asarray([len(set(map(int,x)))==3 for x in newf]);req(int((~face_keep).sum())==2 and set(np.flatnonzero(~face_keep))==set(map(int,edge_faces)),"edge contraction removed unexpected face rows")
 removed_ancestry=[old_ancestry[i] for i in np.flatnonzero(~face_keep)];ancestry=[row for row,yes in zip(old_ancestry,face_keep) if yes];newf=newf[face_keep]
 used,inv=np.unique(newf,return_inverse=True);faces=inv.reshape((-1,3)).astype(np.int64);source=oldp[used].copy();ancestry_source=[oldp[keep].astype(float),oldp[drop].astype(float)]
 moved=float(np.linalg.norm(ancestry_source[0]-ancestry_source[1]));req(moved<0.00025,"selected edge contraction displacement exceeds 0.25 mm")
 # Keep the geometrically bone-safe endpoint 4807; 4766 is now omitted by the link-condition-preserving contraction.
 arr=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]));arr.sort(axis=1);edges,counts=np.unique(arr,axis=0,return_counts=True);req(np.all(counts==2),"contracted source is not edge-closed")
 tri0=oldp[oldf].astype(float);tri1=source[faces].astype(float)
 oldcross=np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]);newcross=np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]);aff=np.flatnonzero(np.any(np.isin(oldf[face_keep],keep),axis=1));req(np.all(np.einsum("ij,ij->i",oldcross[face_keep][aff],newcross[aff])>0),"contracted survivor faces reverse orientation in bound coordinates")
 binding_indices=np.zeros((len(source),4),dtype="<u4");binding_indices[:,1:]=0xffffffff;weights=np.zeros((len(source),4),dtype="<f4");weights[:,0]=1.0
 f_spec=importlib.util.spec_from_file_location("fh",m.FORWARD);helper=importlib.util.module_from_spec(f_spec);f_spec.loader.exec_module(helper);tissue=helper.load_tissue(m.TISS);bindrow=helper.row_data(tissue,159)
 req(int(tissue["bindings"][bindrow["fb"]]["core"])==23,"registered binding row159/core23 changed")
 v6=np.column_stack((source,normals(source,faces))).astype("<f4");step0_receipt=json.loads((m.CURRENT8/"step-0.receipt.json").read_text());step0_poses=m.body_poses(step0_receipt,"current8",0);step0_world=helper.forward(source,binding_indices,weights,bindrow,tissue,step0_poses).astype("<f4")
 rec0,uniq0,qf0=m.mesh_records(step0_world,faces);self0=ci._audit_pair(rec0,rec0,same_surface=True);top0=cp._validate_output_topology(rec0)
 req(self0["count"]==0 and top0["closed_oriented_2_manifold"] and top0["component_count"]==1 and volume(uniq0,qf0)>0,"contracted step0 source self/topology/volume gate failed")
 out_candidate=OUT/"temporalis-163-skin-clearance-candidate-035.npz";provenance=dict(base_meta);provenance.update({"schema":"offline.temporalis.skin-clipped-clearance-candidate.v2","base_candidate_026_sha256":base_sha,"topology_repair":{"operation":"link-condition-preserving edge contraction","retained_vertex_index_in_026":keep,"removed_vertex_index_in_026":drop,"edge_length_m":moved,"max_vertex_displacement_m":moved,"incident_face_indices_in_026":edge_faces.tolist(),"removed_face_ancestry":removed_ancestry,"runtime_skin_modified":False,"runtime_ocular_openings_changed":False,"target_safe_direction":"retain endpoint 4807; collapse 4766 into 4807"}})
 np.savez_compressed(out_candidate,positions_m=source,vertices6=v6,faces=faces,binding_indices=binding_indices,weights=weights,step0_world_f32=step0_world,face_ancestry_json=np.asarray(json.dumps(ancestry,sort_keys=True)),stable_id=np.asarray([163],dtype="<u4"),core_body_index=np.asarray([23],dtype="<u4"),binding_record_sha256=np.asarray([hashlib.sha256(bindrow["binding_bytes"]).hexdigest()]),candidate_provenance_json=np.asarray(json.dumps(provenance,sort_keys=True)))
 captures=[];current8_ok=True;stop_reason=None;input_pins={str(BASE):base_sha,str(m.CLIPPED):sha(m.CLIPPED),str(m.CLIP_REPORT):sha(m.CLIP_REPORT),str(m.CAP_REPORT):sha(m.CAP_REPORT),str(m.TISS):sha(m.TISS),str(m.FORWARD):sha(m.FORWARD),str(W/"src/numilab_human/common_atlas_skin_clearance.py"):sha(W/"src/numilab_human/common_atlas_skin_clearance.py"),str(W/"src/numilab_human/cardiac_cavity_intersections.py"):sha(W/"src/numilab_human/cardiac_cavity_intersections.py"),str(W/"src/numilab_human/cardiac_cavity_partition.py"):sha(W/"src/numilab_human/cardiac_cavity_partition.py")}
 for idx,(group,step,folder) in enumerate(m.capture_rows(),1):
  if idx==9 and not current8_ok:
   stop_reason="current8_nonadmitting";break
  pack=folder/f"step-{step}.mrvpack";receipt_path=folder/f"step-{step}.receipt.json";req(pack.is_file() and receipt_path.is_file(),f"missing {group}/{step} capture")
  receipt=json.loads(receipt_path.read_text());req(int(receipt.get("accepted_step",-1))==step and receipt.get("physical_endpoint")=="accepted",f"{group}/{step}: endpoint is not accepted")
  poses=m.body_poses(receipt,group,step);world=helper.forward(source,binding_indices,weights,bindrow,tissue,poses).astype("<f4")
  rec,unique,qf=m.mesh_records(world,faces);self_a=ci._audit_pair(rec,rec,same_surface=True);top=cp._validate_output_topology(rec)
  keys=m.parse_pack_keys(pack);xyz,target_data,pack_info=ca._pack_surfaces(pack,keys);target_faces={k:np.asarray(v["faces"],dtype=np.int64) for k,v in target_data.items()};audits=ca._target_intersection_audit(rec,target_faces,xyz);nonzero={k:int(v["count"]) for k,v in audits.items() if int(v["count"])>0};present132=sorted(f"{s}:{i}" for s,i in keys if i==132)
  if group=="current8":req(len(audits)==866,f"{group}/{step}: expected current156+bone target population 866, got {len(audits)}");req(bool(present132),f"{group}/{step}: repaired left132 target is absent")
  row={"capture":idx,"group":group,"step":step,"pack_sha256":sha(pack),"receipt_sha256":sha(receipt_path),"accepted_body_pose_count":len(receipt.get("accepted_body_poses",[])),"registered_pose_count":len(receipt.get("accepted_registered_body_poses",[])),"required_head_cores_present":True,"surface_key_132_present":present132,"all_target_surface_count":len(audits),"pack_info":pack_info,"world_f32_sha256":hashlib.sha256(world.tobytes()).hexdigest(),"self_intersection_count":int(self_a["count"]),"closed_oriented_2_manifold":bool(top["closed_oriented_2_manifold"]),"component_count":int(top["component_count"]),"signed_volume_m3":volume(unique,qf),"nonzero_target_counts":nonzero,"clear":bool(self_a["count"]==0 and top["closed_oriented_2_manifold"] and top["component_count"]==1 and not nonzero),"rss_bytes_after_audit":m.rss_bytes()}
  captures.append(row);m.emit(progress,{"event":"capture_audited",**row})
  if group=="current8" and not row["clear"]:current8_ok=False
  if row["rss_bytes_after_audit"]>MAX_RSS:raise RuntimeError(f"RSS ceiling exceeded at {group}/{step}: {row[rss_bytes_after_audit]}")
  del world,rec,unique,qf,self_a,top,xyz,target_data,target_faces,audits,poses;gc.collect();print(json.dumps({"event":"capture_audited","capture":idx,"group":group,"step":step,"clear":row["clear"],"self_pairs":row["self_intersection_count"],"target_hits":sum(nonzero.values()),"rss_bytes":row["rss_bytes_after_audit"],"elapsed_s":time.monotonic()-start},sort_keys=True),flush=True)
 status="candidate_clear_all25" if len(captures)==25 and all(x["clear"] for x in captures) else ("candidate_clear_current8" if len(captures)==8 and current8_ok else ("trial_not_admitted" if all(x["clear"] for x in captures) else "trial_not_admitted"))
 report={"schema":"offline.temporalis.skin-clip-clearance-admission.v2","status":status,"candidate_npz":str(out_candidate),"candidate_npz_sha256":sha(out_candidate),"base_candidate_026_sha256":base_sha,"current8_passed":current8_ok,"all25_captures":len(captures),"all25_captures_clear":bool(len(captures)==25 and all(x["clear"] for x in captures)),"stop_reason":stop_reason,"capture_groups":{"current8":8,"K1":1,"P011-baseline":8,"P011-intervention":8},"stable_id":163,"native_tissue_binding_row":159,"core_body_index":23,"active_head_core_ids_required":[20,23,29,79],"runtime_skin_modified":False,"runtime_ocular_openings_changed":False,"temporary_authoring_target_used_at_runtime":False,"edge_contraction":{"retained_vertex_index_in_026":keep,"removed_vertex_index_in_026":drop,"max_displacement_m":moved,"removed_face_ancestry":removed_ancestry,"link_condition_passed":True,"exact_step0_self_pairs":int(self0["count"]),"step0_closed_oriented":bool(top0["closed_oriented_2_manifold"]),"step0_components":int(top0["component_count"])},"input_sha256":{**input_pins,str(Path(__file__)):sha(Path(__file__))},"captures":captures,"elapsed_seconds":time.monotonic()-start,"max_rss_bytes":m.rss_bytes()}
 m.atomic_json(OUT/"report.json",report);print(json.dumps({"event":"finished","status":status,"captures":len(captures),"elapsed_s":report["elapsed_seconds"],"report":str(OUT/"report.json")},sort_keys=True),flush=True)
if __name__=="__main__":main()
