from __future__ import annotations
import hashlib, importlib.util, json, math, struct
from pathlib import Path

E=Path("/Users/n/numi-human-resting-evidence-20261005")
V8=E/"native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8"
V8_REPORT=V8/"composition-report.json"
V8_REPORT_SHA="f2fd49d0486e20bdca5ea8638215466f4a59ff59d94f1dffa53c8caa5018b460"
V8_DMAP=V8/"final-exact-D311-to-lobes-map.jsonl"
V8_DMAP_SHA="726eaf407129577b18eacc7a794a7f8fd20bcb5bfa333820a0ee19a2001747d1"
V8_NHA=V8/"final/resting-thorax.nhanatomy"
V8_NHA_SHA="1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc"
PARENT_1078_NHA=E/"native-lung-conditioned-final-compose-1078/final/resting-thorax.nhanatomy"
PARENT_1078_NHA_SHA="7f6a8175e3cadb93a6dfc43535467b6896cf1b8bacfd0334df7b5bb03c414e92"
PARENT_1078_DMAP=E/"native-lung-conditioned-final-compose-1078/final-exact-D311-to-lobes-map.jsonl"
PARENT_1078_DMAP_SHA="05ab5b8f6ab02a0f3236495b385bc0002d750be58820117e7f149e64df1e398a"
V8_ADAPTER=Path(__file__).resolve().parent/"v8_dmap_adapter_1096.py"
V8_ADAPTER_SHA="9a55338d874ecc4695e1545829c21266b9b5734d8ca8d64acab817906091c7bd"
EXPECTED_COUNTS={305:19743,306:21600,307:5514,308:486,309:0}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()

def _must_hash(path,digest,label):
 p=Path(path).resolve()
 if not p.is_file() or sha(p)!=digest:raise ValueError(label+" hash mismatch: "+str(p))
 return p

def _tri_bits(row,face):
 return tuple(tuple(struct.pack("<f",float(c)).hex() for c in row["vertices6"][int(v),:3])
              for v in row["faces"][int(face)])

def _get_output(outputs,name):
 v=outputs.get(name)
 if not isinstance(v,dict) or not isinstance(v.get("path"),str) or not isinstance(v.get("sha256"),str):
  raise ValueError("child composition output lacks path/hash: "+name)
 p=Path(v["path"]).resolve()
 if not p.is_file() or sha(p)!=v["sha256"]:raise ValueError("child composition output hash mismatch: "+name)
 return p,v["sha256"]

def _normalize_input(item):
 if isinstance(item,str):return item
 if isinstance(item,dict) and isinstance(item.get("path"),str):return item.get("sha256")
 return None

def validate_registered_triangles(parent_doc,parent_rows,child_rows):
 if any(sid not in parent_rows or sid not in child_rows for sid in (305,306,307,308,309,311)):
  raise ValueError("parent or child NHA lacks a registered lung owner")
 for sid in (305,306,307,308,309,311):
  maps=(parent_doc["pairs"].values() if sid==311 else (parent_doc["pairs"][sid],))
  for pair_map in maps:
   items=pair_map["d2l"].items()
   for dface,lface in items:
    fi=int(dface) if sid==311 else int(lface)
    parent_face=tuple(int(x) for x in parent_rows[sid]["faces"][fi])
    child_face=tuple(int(x) for x in child_rows[sid]["faces"][fi])
    if child_face != parent_face:
     raise ValueError("child registered face index triple differs from immutable V8 parent: row %d face %d"%(sid,fi))
    if _tri_bits(child_rows[sid],fi)!=_tri_bits(parent_rows[sid],fi):
     raise ValueError("child registered triangle differs from immutable V8 parent: row %d face %d"%(sid,fi))
 return {"checked_d_rows":sum(len(m["d2l"]) for m in parent_doc["pairs"].values()),
         "checked_lobe_rows":sum(len(parent_doc["pairs"][sid]["l2d"]) for sid in parent_doc["pairs"]),
         "exact_face_index_triple_match":True,"exact_xyz_triangle_match":True}

def load_v8_current_dmap(*,base,composition_report_path,final_nha_path,final_nha_sha,final_rows):
 """Validate child metadata and exactly rebind unchanged registered V8 D faces."""
 report_path=Path(composition_report_path).resolve()
 if not report_path.is_file():raise ValueError("child D-map composition report missing")
 report=json.loads(report_path.read_text())
 if report.get("schema")!="numi.human.final-lung-selected-composition-dryrun-v1":
  raise ValueError("unsupported child D-map composition schema")
 if report_path==V8_REPORT.resolve():raise ValueError("candidate must carry a distinct child composition report")
 outputs=report.get("outputs",{})
 child_nha,child_sha=_get_output(outputs,"final_nha")
 child_dmap,child_dmap_sha=_get_output(outputs,"d_map")
 child_receipt,child_receipt_sha=_get_output(outputs,"final_receipt")
 child_manifest,child_manifest_sha=_get_output(outputs,"final_manifest")
 if child_nha!=Path(final_nha_path).resolve() or child_sha!=final_nha_sha or sha(child_nha)!=final_nha_sha:
  raise ValueError("child composition is not bound to the exact candidate NHA")
 if child_dmap!=V8_DMAP.resolve() or child_dmap_sha!=V8_DMAP_SHA:
  raise ValueError("child must reuse the exact immutable V8 current D map")
 if child_dmap_sha!=sha(V8_DMAP):raise ValueError("immutable V8 D map changed")

 v8_report=_must_hash(V8_REPORT,V8_REPORT_SHA,"V8 composition report")
 v8_nha=_must_hash(V8_NHA,V8_NHA_SHA,"V8 NHA")
 v8_dmap=_must_hash(V8_DMAP,V8_DMAP_SHA,"V8 current D map")
 parent_nha=_must_hash(PARENT_1078_NHA,PARENT_1078_NHA_SHA,"1078 parent NHA")
 parent_dmap=_must_hash(PARENT_1078_DMAP,PARENT_1078_DMAP_SHA,"1078 parent D map")
 raw_inputs=report.get("inputs")
 if not isinstance(raw_inputs,dict) or not raw_inputs:raise ValueError("child composition has no hashed inputs")
 inputs={}
 for raw_path,value in raw_inputs.items():
  p=Path(raw_path).resolve();digest=_normalize_input(value)
  if not digest or not p.is_file() or sha(p)!=digest:raise ValueError("child composition input hash mismatch: "+str(p))
  inputs[str(p)]=digest
 # The child pins the immutable V8 report/payload/map. The V8 report is
 # independently replayed below and hash-binds its 1078 inputs transitively.
 required={v8_report:V8_REPORT_SHA,v8_nha:V8_NHA_SHA,v8_dmap:V8_DMAP_SHA}
 for p,digest in required.items():
  if inputs.get(str(p))!=digest:raise ValueError("child composition does not hash-bind V8/1078 ancestry: "+str(p))
 for p,digest in ((child_nha,child_sha),(child_dmap,child_dmap_sha),
                  (child_receipt,child_receipt_sha),(child_manifest,child_manifest_sha)):
  if not Path(p).is_file() or sha(p)!=digest:raise ValueError("child output changed during map validation: "+str(p))

 def verify_payload_sidecar(path,label):
  doc=json.loads(path.read_text());payload=doc.get("payload",{})
  if Path(payload.get("path","")).resolve()!=child_nha or payload.get("sha256")!=child_sha or int(payload.get("surface_count",-1))!=524:
   raise ValueError("child "+label+" does not bind current 524-row NHA")
 verify_payload_sidecar(child_receipt,"receipt")
 verify_payload_sidecar(child_manifest,"manifest")

 if sha(V8_ADAPTER)!=V8_ADAPTER_SHA:raise ValueError("pinned V8 D-map reader changed")
 spec=importlib.util.spec_from_file_location("pinned_v8_dmap_parent",V8_ADAPTER)
 v8=importlib.util.module_from_spec(spec);spec.loader.exec_module(v8)
 parser=base.load(base.PARSER,"successor_dmap_parser")
 if sha(base.PARSER)!=base.PARSER_SHA:raise ValueError("pinned NHA parser changed")
 v8_rows=parser.parse_payload(v8_nha)[1]
 parent_doc=v8.load_v8_current_dmap(base=base,composition_report_path=v8_report,
                                     final_nha_path=v8_nha,final_nha_sha=V8_NHA_SHA,
                                     final_rows=v8_rows)
 if parent_doc["map_count"]!=47343 or parent_doc["zero_lobes"]!=[309]:
  raise ValueError("immutable V8 parent D map did not validate")
 parent_d_rows=parent_doc["source_rows"]
 current_rows={}
 validation=validate_registered_triangles(parent_doc,v8_rows,final_rows)
 current_rows={}
 for sid in (305,306,307,308,309,311):
  old=parent_d_rows[sid];cur=final_rows[sid]
  cf=base.np.asarray(cur["faces"],dtype=base.np.int64)
  cv=base.np.asarray(cur["vertices6"][:,:3],dtype="<f4")
  if cf.shape!=old["f"].shape or cv.shape!=old["v"].shape:raise ValueError("child row index-space size changed for mapped owner "+str(sid))
  current_rows[sid]={"v":cv.copy(),"f":cf.copy()}
  for k in ("source_face_ids","run_owner","run_flags","patch_kinds"):
   if k in old:
    if len(old[k])!=len(cf):raise ValueError("V8 D-map metadata row count mismatch")
    current_rows[sid][k]=old[k].copy()

 grouped={sid:[] for sid in EXPECTED_COUNTS}
 with child_dmap.open() as f:
  for line in f:
   if not line.strip():continue
   r=json.loads(line);sid=int(r["lobe_stable_id"])
   if sid not in grouped:raise ValueError("child D map includes undeclared lobe")
   grouped[sid].append({"lobe":sid,"dface":int(r["d_face_row"]),"dsrc":int(r["d_source_face_id"]),
    "owner":int(r["d_run_owner"]),"flag":int(r["d_run_flag"]),"lface":int(r["l_face_row"]),
    "lsrc":int(r["l_source_face_id"]),"kind":int(r["l_patch_kind"]),
    "role":str(r["interface_role"]),"orientation":str(r["orientation"])})
 pairs={}
 for sid,want in EXPECTED_COUNTS.items():
  if len(grouped[sid])!=want:raise ValueError("child D map per-owner count mismatch: "+str(sid))
  pairs[sid]=base.validate_pair_sources(sid,grouped[sid],current_rows,want)
 doc={"path":str(report_path),"sha256":sha(report_path),"map_path":str(child_dmap),
      "map_sha256":child_dmap_sha,"inputs":inputs,"source_rows":current_rows,"pairs":pairs,
      "declared_pairs":sorted(EXPECTED_COUNTS),"map_count":sum(EXPECTED_COUNTS.values()),
      "counts_by_lobe":EXPECTED_COUNTS.copy(),"zero_lobes":[309],
      "registered_area_m2":float(parent_doc["registered_area_m2"]),
      "registered_face_validation":validation,
      "child_D_registration_summary":"not emitted by this composition report; count, source identities, exact current triangles, winding, and reciprocal boundaries are recomputed from the immutable V8 map",
      "parent_binding":"Exact V8 D face rows and source/owner semantics are rebound to the child; all mapped child D/lobe triangles equal V8 Float32 triangles and V8 already proves their 1078 ancestry."}
 base.verify_map_nha(doc,final_rows)
 return doc
