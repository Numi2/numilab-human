#!/usr/bin/env python3
"""785: correct 781 typed-union representative indexing and persist accepted ops first."""
from pathlib import Path
import argparse, hashlib, importlib.util, inspect, json, os, sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE/"coupled_anchor_collapse_783.py"
BASE_SHA="b78358fccf208a12a24b229608c10c231817a8f8c15e4e4c0da45f9352585841"
ENGINE_SHA="739e41b0fb4a3bd7d7eff2b9688657bad325d294185f7ae22a25af9aa2b186eb"
OLD="for j,k in enumerate(A.T):data[k]=np.asarray([u[j] for u in unions],np.int64)"
NEW="for j,k in enumerate(A.T):data[k]=np.asarray([u[0][j] for u in unions],np.int64)"
ACTIVE=None

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

if sha(BASE)!=BASE_SHA: raise RuntimeError("frozen 783 source changed")
spec=importlib.util.spec_from_file_location("coupled_anchor_collapse_783_for_785",BASE)
if spec is None or spec.loader is None: raise RuntimeError("cannot import 783")
G=importlib.util.module_from_spec(spec);sys.modules[spec.name]=G;spec.loader.exec_module(G)
M=G.M; A=M.A; B=M.B; ENGINE=Path(M.__file__).resolve()
if sha(ENGINE)!=ENGINE_SHA: raise RuntimeError("frozen 781 engine SHA mismatch")
ORIGINAL_TYPED=M.typed_source_output
ORIGINAL_RUN=M.run_from_config
src=inspect.getsource(ORIGINAL_TYPED)
if src.count(OLD)!=1: raise RuntimeError("expected one known scalar-lineage indexing expression")
exec(compile(src.replace(OLD,NEW,1),str(Path(__file__).resolve()),"exec"),M.__dict__)
FIXED_TYPED=M.typed_source_output

def _jsonable(x):
 if isinstance(x,dict): return {str(k):_jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [_jsonable(v) for v in x]
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,Path): return str(x)
 if x is None or isinstance(x,(str,int,float,bool)): return x
 return repr(x)

def _journal(parent,ops,points,faces,faceparents):
 if ACTIVE is None: raise RuntimeError("accepted ops lack a 785 pre-serialization journal destination")
 out=ACTIVE["out"];out.mkdir(parents=True,exist_ok=True)
 arr={"output_surface_points_world_m_f32":np.asarray(points,np.float32),
      "output_surface_faces":np.asarray(faces,np.int64),
      "output_surface_parent_face_rows":np.asarray(faceparents,np.int64)}
 recs=[]
 for oi,op in enumerate(ops):
  q={"index":oi,"keep":int(op["keep"]),"drop":int(op["drop"]),
     "anchor_f32":np.asarray(op["anchor_f32"],np.float32).tolist(),"fit":_jsonable(op["fit"]),
     "native_bad_gain_by_role":_jsonable(op["native_bad_gain_by_role"]),
     "source_min_face_apex_bound":float(op["source_min_face_apex_bound"]),
     "exact_paired_delta":str(op["exact_paired_delta"]),"volume_data":{}}
  for name in ("surface_full_endpoint_face_rows","surface_retired_face_rows",
               "surface_changed_output_face_rows","surface_star_vertices"):
   key=f"op{oi:04d}_{name}";arr[key]=np.asarray(op[name],np.int64);q[name+"_key"]=key
  for role in ("Skin","Fat"):
   for name in ("volume_star_cells","volume_star_points"):
    key=f"op{oi:04d}_{role}_{name}";arr[key]=np.asarray(op[name][role],np.int64);q[name+"_"+role+"_key"]=key
   st=op["volumes"][role];r={}
   for name in ("new_tets","materials","parent_rows","old_rows","point_rows"):
    key=f"op{oi:04d}_{role}_{name}";arr[key]=np.asarray(st[name]);r[name+"_key"]=key
    r[name+"_shape"]=list(arr[key].shape);r[name+"_dtype"]=str(arr[key].dtype)
   for name in ("keep_volume_node","drop_volume_node","old_bad","exact_delta","delta_m3"):r[name]=_jsonable(st[name])
   q["volume_data"][role]=r
  recs.append(q)
 p=out/"accepted-operations-pre-serialization.npz";tmp=out/"accepted-operations-pre-serialization.tmp"
 with tmp.open("wb") as f: np.savez(f,**arr);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
 manifest={"schema":"h706.coupled-anchor-collapse-pre-serialization-journal.v1",
   "status":"PASS_ACCEPTED_OPERATIONS_PERSISTED_BEFORE_TYPED_SERIALIZATION",
   "operation_count":len(recs),"source_run_config":ACTIVE["config"],
   "parent_surface":ACTIVE["parent_surface"],
   "source_783":{"path":str(BASE),"sha256":BASE_SHA},
   "source_781":{"path":str(ENGINE),"sha256":ENGINE_SHA},
   "source_785":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)},
   "journal_npz":{"path":str(p),"sha256":sha(p),"array_count":len(arr)},
   "operations":recs,"scope":"replay payload only; no independent geometry/source qualification",
   "physical_admission":False,"native_steps":0}
 mp=out/"accepted-operations-pre-serialization.json";mt=out/"accepted-operations-pre-serialization.json.tmp"
 mt.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 with mt.open("rb+") as f:os.fsync(f.fileno())
 mt.replace(mp)
 ACTIVE["journal"]={"path":str(mp),"sha256":sha(mp),"npz_path":str(p),"npz_sha256":sha(p)}

def _serialize(parent,ops,points,faces,faceparents):
 if ops:_journal(parent,ops,points,faces,faceparents)
 return FIXED_TYPED(parent,ops,points,faces,faceparents)
M.typed_source_output=_serialize

def run_from_config_785(config_path,out_dir,edges_override=None,prune_receipt=None):
 global ACTIVE
 cp=Path(config_path).resolve();od=Path(out_dir).resolve();cfg=json.loads(cp.read_text())
 ACTIVE={"out":od,"config":{"path":str(cp),"sha256":sha(cp)},"parent_surface":cfg.get("surface_I"),"journal":None}
 try:
  result=ORIGINAL_RUN(cp,od,edges_override=edges_override,prune_receipt=prune_receipt)
  result["serialization_fix_785"]={"path":str(Path(__file__).resolve()),"sha256":sha(__file__),
    "representative_rule":"first tuple of sorted complete 7-lane face union; all union/ancestry CSR retained",
    "before":OLD,"after":NEW,"journal":ACTIVE["journal"]}
  state=od/"state.json"
  if state.exists():
   s=json.loads(state.read_text());s["serialization_fix_785"]=result["serialization_fix_785"]
   s["producer"]={"path":str(Path(__file__).resolve()),"sha256":sha(__file__)}
   M._write_json(state,s)
  return result
 finally: ACTIVE=None
M.run_from_config=run_from_config_785

def control(config_path,output):
 global ACTIVE
 cp=Path(config_path).resolve();cfg=json.loads(cp.read_text());ref=cfg["surface_I"]
 parent=M.load_pinned_npz(ref,"785 actual-input I")
 points=np.asarray(parent["points_world_m_f32"],np.float32);faces=np.asarray(parent["faces"],np.int64)
 fp=np.arange(len(faces),dtype=np.int64)
 no=FIXED_TYPED(parent,[],points,faces,fp)
 offs=np.asarray(no["quality_star_patch_union_offsets"],np.int64)
 for j,key in enumerate(A.T):
  if np.asarray(no[key]).shape!=(len(faces),):raise AssertionError("bad scalar face lane "+key)
  for i in range(len(faces)):
   lo,hi=map(int,offs[i:i+2])
   if hi<=lo:raise AssertionError(f"empty typed union at row {i}")
   expected=tuple(int(no[k][lo]) for k in A.U)[j]
   if int(no[key][i])!=expected:raise AssertionError(f"representative mismatch {key} row {i}")
 row=None
 for i in range(len(faces)):
  if B.union_tuples(parent,[i]) and B.ancestry_union(parent,[i]):row=i;break
 if row is None:raise AssertionError("actual input has no typed face+ancestry probe")
 tri=faces[row]
 fake={"keep":int(tri[0]),"drop":int(tri[1]),"anchor_f32":points[int(tri[0])],
  "fit":{"move_cap_m":5e-4,"control_only":True},
  "surface_full_endpoint_face_rows":np.asarray([row],np.int64),
  "surface_retired_face_rows":np.empty(0,np.int64),
  "surface_changed_output_face_rows":np.asarray([row],np.int64),
  "surface_star_vertices":np.unique(tri),
  "volume_star_points":{r:np.empty(0,np.int64) for r in ("Skin","Fat")},
  "volume_star_cells":{r:np.empty(0,np.int64) for r in ("Skin","Fat")},
  "volumes":{r:{"new_tets":np.empty((0,4),np.int64),"materials":np.empty(0,np.int32),
    "parent_rows":np.empty(0,np.int64),"old_rows":np.empty(0,np.int64),"point_rows":np.empty(0,np.int64),
    "keep_volume_node":0,"drop_volume_node":1,"old_bad":0,"exact_delta":0,"delta_m3":0.0} for r in ("Skin","Fat")},
  "native_bad_gain_by_role":{"Skin":0,"Fat":0},"source_min_face_apex_bound":0.0,"exact_paired_delta":"0"}
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
 ACTIVE={"out":output,"config":{"path":str(cp),"sha256":sha(cp)},"parent_surface":ref,"journal":None}
 try: out=M.typed_source_output(parent,[fake],points,faces,fp)
 finally: ACTIVE=None
 if int(np.asarray(out["source_collapse_operation_rows"])[row])!=0:raise AssertionError("operation row marker lost")
 lo,hi=map(int,np.asarray(out["quality_star_patch_union_offsets"],np.int64)[row:row+2])
 got=sorted(set(tuple(int(out[k][x]) for k in A.U) for x in range(lo,hi)))
 if got!=B.union_tuples(parent,[row]):raise AssertionError("complete operation union changed")
 lo,hi=map(int,np.asarray(out["face_ancestry_offsets"],np.int64)[row:row+2])
 anc=sorted(set(tuple(int(out[k][x]) for k in A.A) for x in range(lo,hi)))
 if anc!=sorted(tuple(map(int,x)) for x in B.ancestry_union(parent,[row])):raise AssertionError("operation ancestry changed")
 journal_path=output/"accepted-operations-pre-serialization.json"
 journal=json.loads(journal_path.read_text())
 if journal.get("operation_count")!=1:raise AssertionError("pre-serialization operation journal missing")
 result={"schema":"h706.coupled-anchor-collapse-serialization-control.v1",
  "status":"PASS_ACTUAL_INPUT_SERIALIZER_REGRESSION",
  "input_config":{"path":str(cp),"sha256":sha(cp)},
  "parent_surface":{"path":str(Path(ref["path"]).resolve()),"sha256":ref["sha256"]},
  "face_count":int(len(faces)),"scalar_source_lanes_checked":list(A.T),
  "synthetic_operation_face_row":int(row),"operation_union_tuples":len(got),
  "operation_ancestry_tuples":len(anc),"journal":{"path":str(journal_path),"sha256":sha(journal_path)},
  "geometry_or_solver":"NOT_RUN_CONTROL_ONLY",
  "physical_admission":False,"native_steps":0}
 rp=output/"serialization-control.json";rp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 result["receipt"]={"path":str(rp),"sha256":sha(rp)}
 print(json.dumps(result,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=("control","run"),required=True)
 p.add_argument("--config",type=Path);p.add_argument("--state",type=Path);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--move-cap-m",type=float,default=5e-4);p.add_argument("--max-operations",type=int,default=32)
 p.add_argument("--seconds",type=float,default=90);p.add_argument("--max-edge-length-m",type=float,default=5e-3)
 p.add_argument("--max-edges",type=int,default=100000);a=p.parse_args()
 if a.mode=="control":
  if a.config is None:p.error("control requires --config")
  control(a.config,a.output);return
 if a.state is None:p.error("run requires --state")
 result=G.G.run_verified_state(a.state,a.output,move_cap_m=a.move_cap_m,max_operations=a.max_operations,
  time_limit_s=a.seconds,max_edge_length_m=a.max_edge_length_m,max_edges=a.max_edges)
 wp=a.output.resolve()/"state-wrapper.json"
 if wp.exists():
  wrapper=json.loads(wp.read_text());st=Path(result.get("candidate_run",{}).get("path",""))
  if st.exists():wrapper["pre_serialization_journal"]=json.loads(st.read_text()).get("serialization_fix_785",{}).get("journal")
  wrapper["serialization_fix_785"]={"path":str(Path(__file__).resolve()),"sha256":sha(__file__),
   "pre_serialization_journal":wrapper.get("pre_serialization_journal")}
  M._write_json(wp,wrapper);result=wrapper
 print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":main()
