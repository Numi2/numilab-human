#!/Users/numi/numi-human-prep-venv-20261005/bin/python
from pathlib import Path
import hashlib, importlib, importlib.util, json, sys, time, traceback
import numpy as np

P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BASE=P/"fine-physical-volume-706"
RUN=BASE/"local-microedge-condition-001"
OUT=RUN/"exact-gates-014-current-all185-F32-F64"
OWNER=P/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
ALL185=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
BONE_HELPER=P/"bone-barrier-remeshing-609/bone_barrier.py"
MANIFEST=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
EXACT010=RUN/"exact-gates-010/report.json"
EXPECTED={
 "owner":"f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d",
 "all185":"e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5",
 "bone_helper":"734527f060dcae186ae1b9609eca1ab95768d952aa0e9e8c7af76cae882b236b",
 "manifest":"0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c",
 "exact010":"810a2f07d589eb3646155c277840ce0e98455b965cfe1d935e01f36d7e29acbc",
}
SURFACES={
 "O":(BASE/"local-microedge-condition-001/candidate-O-microcollapsed.npz","1b3820a1a7aa46c3bed0f1475c0f68a9929a2e1353097a3cc4a28240b39844fe"),
 "I":(BASE/"local-microedge-saved-operation-replay-011/I/candidate-I-saved-op-replay.npz","9721102247157eeb42c59633fda59d63fd5ba48ad11ba4e63987834136711e31"),
 "D":(BASE/"local-microedge-saved-operation-replay-013/D/candidate-D-saved-op-replay.npz","6200c66e255caccafaf04aae9669df21769d3a56e9e991091ec3cdaa18c15766"),
}
FACE_OFFSETS={"O":0,"I":300000,"D":600000}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
def atomic(path,obj):
 tmp=Path(str(path)+".tmp");write(tmp,obj);tmp.replace(path)
def role_records(owner,points,faces,face_offset):
 raw,_=owner.float32_surface_records(np.asarray(points,np.float32).astype(np.float64),np.asarray(faces,np.int64),source_seam_groups=())
 return [(r[0],r[1],r[2],face_offset+i,r[4]) for i,r in enumerate(raw)]
def disjoint_records(owner,points,faces):
 raw,_=owner.float32_surface_records(np.asarray(points,np.float32).astype(np.float64),np.asarray(faces,np.int64),source_seam_groups=())
 return [(r[0],r[1],r[2],-1-i,tuple(-1-int(v) for v in r[4])) for i,r in enumerate(raw)]
def unit_regression(owner):
 pts=np.asarray([[.010,.020,.030],[.011,.020,.030],[.010,.021,.030]],np.float32)
 faces=np.asarray([[0,1,2]],np.int64)
 tissue=role_records(owner,pts,faces,100)
 bone=disjoint_records(owner,pts,faces)
 blo=np.nextafter(pts.min(axis=0).astype(np.float64),-np.inf)
 bhi=np.nextafter(pts.max(axis=0).astype(np.float64),np.inf)
 rlo=np.asarray([min(r[1][k] for r in tissue) for k in range(3)],np.float64)
 rhi=np.asarray([max(r[2][k] for r in tissue) for k in range(3)],np.float64)
 old_mixed=bool(np.all(bhi>=rlo) and np.all(rhi>=blo))
 result=owner._audit_pair_prepared_first(owner._prepare_surface_aabb(bone),tissue,same_surface=False)
 if old_mixed or int(result.get("count",-1))!=1:
  raise AssertionError(f"unit regression failed old_mixed={old_mixed} exact={result.get(count)}")
 return {"status":"PASS","fixture":"identical 10 mm-scale F32 triangle with disjoint role IDs","old_mixed_units_prefilter_candidates":old_mixed,"same_integer_lattice_exact_pairs":int(result["count"]),"role_bound_integer_magnitude":max(abs(int(x)) for x in rlo),"native_bound_metre_magnitude":float(np.max(np.abs(blo)))}

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);(OUT/"source").mkdir()
 source=Path(__file__);(OUT/"source"/source.name).write_bytes(source.read_bytes())
 report={"schema":"numi.fine-physical-volume.current-surfaces-vs-all185-F32-F64-common-lattice.v1","status":"PREPARING","physical_admission":False,"runtime_created":0,"accepted_physical_steps":0,"inputs":{},"stages":{},"roles":{},"witness_files":{}}
 def stage(name,status,**fields):
  report["stages"][name]={"status":status,**fields};atomic(OUT/"progress.json",report)
 for name,path in (("exact_owner",OWNER),("all185_native_F32",ALL185),("bone_helper",BONE_HELPER),("all185_manifest",MANIFEST),("exact_gates_010",EXACT010)):
  got=sha(path)
  if got!=EXPECTED[name.replace("exact_owner","owner").replace("all185_native_F32","all185").replace("bone_helper","bone_helper").replace("all185_manifest","manifest").replace("exact_gates_010","exact010")]:
   raise ValueError(f"{name} input SHA mismatch: {got}")
  report["inputs"][name]={"path":str(path),"sha256":got}
 for role,(path,expected) in SURFACES.items():
  if sha(path)!=expected:raise ValueError(f"{role} candidate pin mismatch")
  report["inputs"][role+"_candidate"]={"path":str(path),"sha256":expected}
 prior=json.loads(EXACT010.read_text())
 for role in "OID":
  if prior.get("full_current_self_results",{}).get(role,{}).get("count")!=0:raise ValueError(f"{role} self/cross prior receipt not zero")
 for label in ("O-I","O-D","I-D"):
  if prior.get("full_current_changed_cross_results",{}).get(label,{}).get("count")!=0:raise ValueError(f"{label} prior cross receipt not zero")
 report["inherited_valid_geometry"]={"source_report":str(EXACT010),"source_sha256":EXPECTED["exact010"],"full_current_self_pairs":{"O":0,"I":0,"D":0},"full_current_cross_pairs":{"O-I":0,"O-D":0,"I-D":0},"source_all185_result":"INVALID_MIXED_UNITS; not inherited"}
 sys.path.insert(0,str(OWNER.parent.parent))
 owner=importlib.import_module("numilab_human.cardiac_cavity_intersections")
 if sha(Path(owner.__file__))!=EXPECTED["owner"]:raise ValueError("package-loaded exact owner SHA mismatch")
 bone_mod=load_module("bone_barrier_current_014",BONE_HELPER)
 regression=unit_regression(owner)
 report["unit_regression"]=regression
 stage("mixed_units_negative_control",regression["status"],**{k:v for k,v in regression.items() if k!="status"})
 native=load_npz(ALL185);np32=np.asarray(native["points_world_f32"],np.float32);nf=np.asarray(native["faces"],np.int64)
 bank_offsets=np.asarray(native["bank_face_offsets"],np.int64)
 if np32.shape!=(234812,3) or nf.shape!=(469136,3) or bank_offsets.shape!=(186,) or int(bank_offsets[0])!=0 or int(bank_offsets[-1])!=len(nf):
  raise ValueError("native F32 185-bank source schema mismatch")
 receipt_manifest=json.loads(MANIFEST.read_text())
 if len(receipt_manifest.get("operands",[]))!=185:raise ValueError("F64 manifest is not all 185 banks")
 # The existing owner loader maps both Float32 replay and exact Float64 operands
 # onto denominator 2^149 integer coordinates before building any AABB.
 point_offset=max(len(load_npz(p)["points_world_m"]) for p,_ in SURFACES.values())+1000
 t0=time.monotonic()
 barrier,barrier_receipt=bone_mod.load_bone_barrier_records(owner,root=P,vertex_offset=point_offset,expected_native_sha256=EXPECTED["all185"],expected_manifest_sha256=EXPECTED["manifest"])
 load_s=time.monotonic()-t0
 if len(barrier)!=938272 or barrier_receipt.get("record_count")!=938272 or barrier_receipt.get("bank_coordinate_domains")!=370:
  raise ValueError("combined 185 F32/F64 barrier record inventory mismatch")
 if any(int(r[3])>=0 for r in barrier):raise ValueError("bone face IDs are not disjoint negative IDs")
 if any(any(int(v)<point_offset for v in r[4]) for r in barrier):raise ValueError("bone vertex ID namespace overlaps role rows")
 f32_probe=bone_mod.exact_common_lattice_key(tuple(float(x) for x in np32[int(nf[0,0])]))
 if tuple(barrier[0][0][0])!=f32_probe:raise ValueError("loaded F32 record is not in expected common lattice")
 first_f64=next(r for r in barrier_receipt["ranges"] if r["domain"]=="pinned_f64_exact")
 operand=receipt_manifest["operands"][0]
 with np.load(operand["operand_path"],allow_pickle=False) as z:
  fp=np.asarray(z["points_world_m_f64"]);ff=np.asarray(z["faces_f64"],np.int64)
  f64_probe=bone_mod.exact_common_lattice_key(tuple(float(x) for x in fp[int(ff[0,0])]))
 if tuple(barrier[int(first_f64["negative_face_index_start"])][0][0])!=f64_probe:
  raise ValueError("loaded F64 record is not in same common lattice")
 report["inputs"]["native_F32_schema"]={"points":list(np32.shape),"faces":list(nf.shape),"banks":185,"face_offsets_sha256":hashlib.sha256(bank_offsets.tobytes()).hexdigest()}
 report["all185_record_receipt"]=barrier_receipt
 report["all185_records"]={"count":len(barrier),"F32_records":469136,"F64_records":469136,"banks_per_domain":185,"distinct_body_bone_domains":370,"vertex_namespace_start":point_offset,"one_shared_exact_integer_lattice":True,"F32_unit_probe":"PASS","F64_unit_probe":"PASS","bbox_prefilter":"none; full exact AABB records used"}
 stage("all185_F32_F64_records_prepared","PASS",records=len(barrier),banks_per_domain=185,record_loader_seconds=load_s,coordinate_domain="owner common 2^-149 integer lattice")
 t0=time.monotonic();index=owner._prepare_surface_aabb(barrier);index_s=time.monotonic()-t0
 report["all185_index"]={"records":len(barrier),"build_seconds":index_s,"coordinate_domain":"integer lattice","meter_prefilter":"not used"}
 stage("full_all185_AABB_index_prepared","PASS",records=len(barrier),seconds=index_s)
 domain_ranges=barrier_receipt["ranges"]
 total_pairs=0
 for role,(path,expected) in SURFACES.items():
  dat=load_npz(path);pts=np.asarray(dat["points_world_m"],np.float32);faces=np.asarray(dat["faces"],np.int64)
  q=role_records(owner,pts,faces,FACE_OFFSETS[role])
  if len(q)!=len(faces):raise ValueError(f"{role} current record coverage mismatch")
  started=time.monotonic()
  result=owner._audit_pair_prepared_first(index,q,same_surface=False)
  raw=np.asarray(result.get("triangle_pairs",[]),np.int64).reshape((-1,2))
  converted=[];by_domain={"source_order_native_f32_replay":0,"pinned_f64_exact":0};by_bank={}
  for barrier_face_id,role_face_id in raw.tolist():
   record_index=-1-int(barrier_face_id)
   role_row=int(role_face_id)-FACE_OFFSETS[role]
   if not (0<=record_index<len(barrier) and 0<=role_row<len(faces)):raise ValueError("returned exact pair outside mapped domains")
   matches=[rg for rg in domain_ranges if int(rg["negative_face_index_start"])<=record_index<int(rg["negative_face_index_end"])]
   if len(matches)!=1:raise ValueError("exact pair record has ambiguous source bank range")
   rg=matches[0];domain=rg["domain"];domain_code=0 if domain=="source_order_native_f32_replay" else 1
   local_face=record_index-int(rg["negative_face_index_start"])
   converted.append((role_row,record_index,domain_code,local_face,int(rg["body"]),int(rg["bone"])))
   by_domain[domain]+=1
   by_bank[(domain,int(rg["body"]),int(rg["bone"]))]=by_bank.get((domain,int(rg["body"]),int(rg["bone"])),0)+1
  arr=np.asarray(converted,np.int64).reshape((-1,6));pair_path=OUT/f"{role}-whole-current-vs-all185-F32-F64-pairs.npy";np.save(pair_path,arr)
  info={"status":"PASS" if len(arr)==0 else "FAIL","candidate_path":str(path),"candidate_sha256":expected,"current_points":len(pts),"current_faces":len(faces),"barrier_records":len(barrier),"F32_pairs":by_domain["source_order_native_f32_replay"],"F64_pairs":by_domain["pinned_f64_exact"],"total_pairs":len(arr),"aabb_candidates":int(result.get("aabb_candidate_pairs",-1)),"overlap_banks":len(by_bank),"overlap_bank_pair_counts":{f"{d}:body{b}:bone{n}":c for (d,b,n),c in sorted(by_bank.items())},"elapsed_s":time.monotonic()-started,"pair_columns":["current_role_face_row","barrier_global_record_index","domain_code_0_F32_1_F64","barrier_local_face_row","body","bone"],"pairs_path":str(pair_path),"pairs_sha256":sha(pair_path),"first_20_pairs":arr[:20].tolist()}
  report["roles"][role]=info;report["witness_files"][role]={"path":str(pair_path),"sha256":sha(pair_path),"count":len(arr)};total_pairs+=len(arr)
  stage(role+"_full_current_vs_all185_F32_F64",info["status"],**{k:v for k,v in info.items() if k not in ("status","first_20_pairs","overlap_bank_pair_counts")})
 report["all185_total_pairs"]=total_pairs
 report["status"]="PASS_WHOLE_CURRENT_OID_VS_ALL185_F32_F64" if total_pairs==0 else "FAIL_EXACT_WHOLE_CURRENT_NATIVE_BONE_PAIRS"
 report["limitations"]=["Whole-current O/I/D surfaces were each queried against the complete 469136-face native F32 plus 469136-face pinned F64 barriers from 185 banks per domain, using one common exact integer lattice and disjoint namespaces.","The exact-gates-010 all185 result is invalid due to a meter-versus-lattice bounding prefilter and is not reused. Its full-current self/cross exact-predicate results are inherited and remain independently valid.","No closed-solid containment, TetGen, source330 child registration, mass transport, attachment-force, physical admission, or native Runtime claim."]
 write(OUT/"report.json",report);write(OUT/"progress.json",report)
 print(json.dumps({"status":report["status"],"out":str(OUT),"report_sha256":sha(OUT/"report.json"),"unit_regression":regression,"record_counts":report["all185_records"],"roles":{r:{k:v[k] for k in ("current_faces","F32_pairs","F64_pairs","total_pairs","aabb_candidates","elapsed_s","pairs_sha256")} for r,v in report["roles"].items()}},indent=2),flush=True)
if __name__=="__main__":
 try:main()
 except Exception as e:
  out=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/local-microedge-condition-001/exact-gates-014-current-all185-F32-F64")
  out.mkdir(parents=True,exist_ok=True);err={"status":"RUN_FAILED_RETAINED","error":repr(e),"traceback":traceback.format_exc(),"physical_admission":False,"runtime_created":0,"accepted_physical_steps":0};write(out/"failure.json",err);print(json.dumps(err,sort_keys=True),flush=True);raise

