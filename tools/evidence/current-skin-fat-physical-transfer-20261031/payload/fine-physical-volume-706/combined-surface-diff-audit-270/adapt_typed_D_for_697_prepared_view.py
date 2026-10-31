#!/usr/bin/env python3
"""Materialize a metadata-only current typed-D view accepted by actual 697.

The source candidate and coordinates remain untouched. The prepared-row domain
is current candidate face rows; exact-source rows stay invalid, while the
candidate's full typed ancestry CSR and quality-star union arrays are copied.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
RUNNER=H/"handoff/run_native_domain_217_typedD_697.py"
HANDOFF=H/"handoff/prepare_tetgen_handoff_697_typedD.py"
REGISTERED=H/"handoff/registered_surface_handoff.py"
DEFAULT_REGISTRY=H/"current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz"
SCHEMA="h706.current-typed-D-697-prepared-view.v1"
TYPED_GROUP_FIELDS=("face_source_layer","face_source_body","face_source_bone","face_source_original_id")
ANCESTRY_FIELDS=("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def array_payload_equal(a,b):
    x,y=np.asarray(a),np.asarray(b)
    return x.shape==y.shape and x.dtype==y.dtype and np.ascontiguousarray(x).tobytes()==np.ascontiguousarray(y).tobytes()

def build_prepared_arrays(source, source_sha, identity_schema):
    required={"points_world_m","faces","face_source_layer","face_source_body","face_source_bone",
              "face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1",
              "face_ancestry_offsets",*ANCESTRY_FIELDS}
    missing=required-set(source)
    if missing: raise ValueError(f"current typed D lacks fields: {sorted(missing)}")
    n=len(source["faces"])
    faces=np.asarray(source["faces"],np.int64)
    points=np.asarray(source["points_world_m"],np.float32)
    if faces.shape!=(n,3) or n==0 or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError("current typed D geometry is malformed")
    if not np.isfinite(points).all() or faces.min()<0 or faces.max()>=len(points):
        raise ValueError("current typed D geometry has invalid point references")
    signatures=np.stack([np.asarray(source[k],np.int64) for k in TYPED_GROUP_FIELDS],axis=1)
    if signatures.shape!=(n,len(TYPED_GROUP_FIELDS)):
        raise ValueError("typed group fields do not align with faces")
    unique,group=np.unique(signatures,axis=0,return_inverse=True)
    keys=[dict(zip(TYPED_GROUP_FIELDS,map(int,row))) for row in unique]
    offsets=np.r_[0,np.cumsum(np.bincount(group,minlength=len(keys)))].astype(np.int64)
    group_rows=np.argsort(group,kind="stable").astype(np.int64)
    used_ids=set(map(str,np.unique(np.asarray(source["face_source_original_id"],np.int64))))
    absent=sorted(used_ids-set(identity_schema))
    if absent: raise ValueError(f"current face original IDs absent from pinned source registry: {absent[:12]}")
    ao=np.asarray(source["face_ancestry_offsets"],np.int64)
    lanes=[np.asarray(source[k],np.int64) for k in ANCESTRY_FIELDS]
    if (ao.shape!=(n+1,) or ao[0]!=0 or ao[-1]<0 or np.any(np.diff(ao)<=0) or
        any(v.shape!=(int(ao[-1]),) or np.any(v<0) for v in lanes)):
        raise ValueError("current typed D full ancestry CSR malformed")
    out={k:np.asarray(v).copy() for k,v in source.items()}
    # This is a row-domain adapter, not a historical exact-face map: each
    # prepared source row is the current typed output row; exact-source stays -1.
    out["face_group_id"]=group.astype(np.int64)
    out["source_group_offsets"]=offsets
    out["source_group_face_rows"]=group_rows
    out["source_group_keys_json"]=np.asarray([json.dumps(keys,separators=(",",":"),sort_keys=True)])
    out["face_parent_source_face_rows"]=np.arange(n,dtype=np.int64)
    out["face_exact_source_row"]=np.full(n,-1,dtype=np.int64)
    out["face_source_identity_schema_by_original_id_json"]=np.asarray(
        [json.dumps(identity_schema,separators=(",",":"),sort_keys=True)])
    out["lineage_preparation_input_sha256"]=np.asarray([source_sha])
    out["face_ancestry_method"]=np.asarray([
        "metadata-only current typed-D 697 adapter: current face-row source-group partition by layer/body/bone/original_id; exact-source rows remain -1; copied complete face_ancestry CSR and quality_star_patch_union arrays remain authoritative"])
    return out,{"source_group_count":len(keys),"face_count":n,"ancestry_tuple_count":int(ao[-1]),
                "current_original_ids":len(used_ids),"current_ids_present_in_registry":len(used_ids)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    ap.add_argument("--source-sha256",required=True)
    ap.add_argument("--identity-registry",default=str(DEFAULT_REGISTRY))
    ap.add_argument("--identity-registry-sha256",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--receipt",required=True)
    a=ap.parse_args()
    t0=time.monotonic()
    source_path=Path(a.source).resolve()
    registry_path=Path(a.identity_registry).resolve()
    out=Path(a.out).resolve();receipt_path=Path(a.receipt).resolve()
    if sha(source_path)!=a.source_sha256: raise ValueError("source typed-D SHA mismatch")
    if sha(registry_path)!=a.identity_registry_sha256: raise ValueError("identity registry source SHA mismatch")
    if out.exists() or receipt_path.exists(): raise FileExistsError("refusing to overwrite adapter output")
    with np.load(source_path,allow_pickle=False) as z: source={k:np.asarray(z[k]) for k in z.files}
    with np.load(registry_path,allow_pickle=False) as z:
        if "face_source_identity_schema_by_original_id_json" not in z.files:
            raise ValueError("pinned identity registry lacks typed identity schema")
        identity_schema=json.loads(str(np.asarray(z["face_source_identity_schema_by_original_id_json"]).item()))
    prepared,summary=build_prepared_arrays(source,a.source_sha256,identity_schema)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**prepared)
    if sha(source_path)!=a.source_sha256 or sha(registry_path)!=a.identity_registry_sha256:
        raise ValueError("source changed while the adapter view was being written")

    # Exercise the actual 697 runner wrapper and its registered_surface_handoff
    # implementation. This loads a surface record only; it does not run 697 main.
    runner=import_path("current_typed_D_697_runner",RUNNER)
    owner=runner.load_typed_owner()
    loaded=owner.load_surface(str(out),"D")
    if not np.array_equal(np.asarray(loaded["points"],np.float32),np.asarray(source["points_world_m"],np.float32)):
        raise ValueError("697 loader point bytes differ from source")
    if not np.array_equal(np.asarray(loaded["faces"],np.int64),np.asarray(source["faces"],np.int64)):
        raise ValueError("697 loader faces differ from source")
    for key,loaded_key in (("face_source_layer","source_layers"),("face_source_body","source_bodies"),
                            ("face_source_bone","source_bones"),("face_source_face_id","source_face_ids"),
                            ("face_source_original_id","source_original_ids"),
                            ("face_source_identity0","source_identity0"),("face_source_identity1","source_identity1")):
        if not np.array_equal(np.asarray(loaded[loaded_key]),np.asarray(source[key])):
            raise ValueError(f"697 loader changed typed field {key}")
    if not np.all(np.asarray(loaded["exact_source_rows"],np.int64)==-1) or not np.all(np.asarray(loaded["ancestry_kind"],np.int8)==6):
        raise ValueError("697 loader did not preserve derived kind-6/no-exact-source semantics")
    loaded_ancestry=[loaded["ancestry"],loaded["ancestry_original_ids"],loaded["ancestry_parent_rows"]]
    ao=np.asarray(source["face_ancestry_offsets"],np.int64)
    expected=[[] for _ in range(3)]
    for i in range(len(source["faces"])):
        for j,key in enumerate(ANCESTRY_FIELDS):
            expected[j].append(np.asarray(source[key],np.int64)[ao[i]:ao[i+1]].tolist())
    if any(loaded_ancestry[i]!=expected[i] for i in range(3)):
        raise ValueError("697 loader changed or truncated full current face ancestry")
    with np.load(out,allow_pickle=False) as z: adapted={k:np.asarray(z[k]) for k in z.files}
    for k in source:
        if not array_payload_equal(adapted[k],source[k]):
            raise ValueError(f"metadata-only view changed source member bytes {k}")
    build_path=Path(runner.BUILD)
    receipt={
        "schema":SCHEMA,
        "status":"PASS_ACTUAL_697_TYPED_D_LOADER_METADATA_VIEW",
        "physical_admission":False,"native_steps":0,
        "source_candidate":{"path":str(source_path),"sha256":a.source_sha256},
        "prepared_view":{"path":str(out),"sha256":sha(out),"geometry_members_bitwise_equal":True,
                         "source_members_bitwise_preserved":len(source)},
        "identity_registry":{"path":str(registry_path),"sha256":a.identity_registry_sha256,
                              "registered_identity_schema_sha256":hashlib.sha256(
                                  json.dumps(identity_schema,separators=(",",":"),sort_keys=True).encode()).hexdigest()},
        "registered_surface_handoff":{"path":str(REGISTERED),"sha256":sha(REGISTERED),
                                      "prepare_adapter_path":str(HANDOFF),"prepare_adapter_sha256":sha(HANDOFF)},
        "native_697_owner":{"path":str(build_path),"sha256":sha(build_path),
                            "runner_path":str(RUNNER),"runner_sha256":sha(RUNNER)},
        "loaded_D":{"points":len(loaded["points"]),"faces":len(loaded["faces"]),
                    "ancestry_kind_counts":{"6":len(loaded["ancestry_kind"])},
                    "exact_source_row_count":0,"source_group_count":summary["source_group_count"],
                    "ancestry_tuple_count":summary["ancestry_tuple_count"],
                    "typed_fields_bitwise_equal":True,"full_face_ancestry_equal":True},
        "adapter_semantics":"source groups partition current typed D rows by layer/body/bone/original ID; they are row-domain buckets, not claims of a measured unique source face. Exact source rows remain -1; existing typed source fields, full face ancestry CSR, and full quality-star patch-union arrays remain unchanged.",
        "elapsed_s":time.monotonic()-t0,
    }
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\\n")
    print(json.dumps(receipt,indent=2,sort_keys=True))
    print("receipt_sha256",sha(receipt_path))

if __name__=="__main__":
    main()

