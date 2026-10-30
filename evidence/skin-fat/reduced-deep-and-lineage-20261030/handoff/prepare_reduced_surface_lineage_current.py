"""Compose reduced surface preparation lineage from an authenticated source.

Source-group membership is rebuilt from anatomical signatures, never temporary
construction run IDs. Face CSR records derivation parents, not exact overlap.
Coordinates, faces and original evidence remain unmodified.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
FIELDS=("face_source_layer","face_source_body","face_source_bone","face_source_original_id")
IDENTITY=FIELDS+("face_source_face_id","face_source_identity0","face_source_identity1")
CSR=("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def materialize(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def text(a):return str(np.asarray(a).reshape(-1)[0])
def check_csr(a,offset_key,members,count):
    off=np.asarray(a[offset_key],np.int64);values=[np.asarray(a[k],np.int64) for k in members]
    if (off.shape!=(count+1,) or off[0]!=0 or np.any(np.diff(off)<=0) or
        any(len(v)!=off[-1] or np.any(v<0) for v in values)):
        raise ValueError("invalid provenance CSR: "+offset_key)
    return off,values
def prepare(candidate,source,output,receipt):
    candidate,source,output,receipt=map(Path,(candidate,source,output,receipt))
    if output.exists() or receipt.exists():raise FileExistsError("refusing to overwrite lineage output")
    a=materialize(candidate);s=materialize(source)
    source_sha=sha(source)
    if text(a["source_candidate_sha256"])!=source_sha:raise ValueError("reduced source hash mismatch")
    faces=np.asarray(a["faces"],np.int64);points=np.asarray(a["points_world_m"],np.float32)
    sf=np.asarray(s["faces"],np.int64);n=len(faces);ns=len(sf)
    if len(points)==0 or faces.shape!=(n,3) or faces.min()<0 or faces.max()>=len(points):
        raise ValueError("invalid reduced geometry")
    parent=np.asarray(a["face_parent_source_face_rows"],np.int64).copy()
    if parent.shape!=(n,):raise ValueError("invalid primary source face")
    invalid_primary=(parent<0)|(parent>=ns)
    if np.any(invalid_primary):
        if "face_repair_parent_source_offsets" in a:
            roff,(rrows,)=check_csr(a,"face_repair_parent_source_offsets",("face_repair_parent_source_rows",),n)
        elif "topology_flip_parent_D624_face_offsets" in a:
            roff,(rrows,)=check_csr(a,"topology_flip_parent_D624_face_offsets",("topology_flip_parent_D624_face_rows",),n)
        else:
            raise ValueError("invalid primary source face without a declared repair-parent set")
        if rrows.max()>=ns:raise ValueError("repair parent outside source face domain")
        a["lineage_history_face_parent_source_face_rows"]=parent.copy()
        for row in np.flatnonzero(invalid_primary):
            parent[row]=int(np.min(rrows[roff[row]:roff[row+1]]))
        a["face_parent_source_face_rows"]=parent.copy()
    signatures=np.stack([np.asarray(s[k],np.int64) for k in FIELDS],axis=1)
    unique,source_group=np.unique(signatures,axis=0,return_inverse=True)
    oldkeys=json.loads(text(a["source_group_keys_json"]))
    keys=[dict(zip(FIELDS,map(int,row))) for row in unique]
    if keys!=oldkeys:raise ValueError("source anatomical group inventory differs from reduction")
    oldgroup=np.asarray(a["face_group_id"],np.int64)
    group=source_group[parent]
    order=np.argsort(source_group,kind="stable").astype(np.int64)
    offsets=np.r_[0,np.cumsum(np.bincount(source_group,minlength=len(keys)))].astype(np.int64)
    if offsets[-1]!=ns or len(offsets)!=len(keys)+1:raise AssertionError("reconstructed group domain")
    source_has_csr = "face_ancestry_offsets" in s or "union_face_ancestry_offsets" in s
    if "face_ancestry_offsets" in s:
        soff,lanes=check_csr(s,"face_ancestry_offsets",CSR,ns)
        source_semantics = "composed from the authenticated source's full declared typed CSR"
    elif "union_face_ancestry_offsets" in s:
        soff,lanes=check_csr(s,"union_face_ancestry_offsets",
                            ("union_face_ancestry_rows","union_face_ancestry_run_original_ids","union_face_ancestry_face_ids"),ns)
        source_semantics = "composed from the authenticated D source's full declared typed union CSR"
    else:
        if any(k in s for k in CSR+("union_face_ancestry_rows","union_face_ancestry_run_original_ids","union_face_ancestry_face_ids")):
            raise ValueError("source has a partial typed ancestry CSR")
        for key in IDENTITY:
            if key not in s or np.asarray(s[key]).shape != (ns,):
                raise ValueError("unreduced source lacks complete declared identity lanes")
        soff=np.arange(ns+1,dtype=np.int64)
        lanes=[np.arange(ns,dtype=np.int64),
               np.asarray(s["face_source_original_id"],np.int64),
               np.asarray(s["face_source_face_id"],np.int64)]
        if any(np.any(v<0) for v in lanes):
            raise ValueError("unreduced source declares a negative source face identity")
        source_semantics = "authenticated unreduced source has no full historical CSR; use its declared source row, original ID and face ID only"
    if "face_repair_parent_source_offsets" in a:
        poff,(prows,)=check_csr(a,"face_repair_parent_source_offsets",("face_repair_parent_source_rows",),n)
    elif "topology_flip_parent_D624_face_offsets" in a:
        poff,(prows,)=check_csr(a,"topology_flip_parent_D624_face_offsets",
                              ("topology_flip_parent_D624_face_rows",),n)
        a["face_repair_parent_source_offsets"]=poff.copy()
        a["face_repair_parent_source_rows"]=prows.copy()
    else:
        poff=np.arange(n+1,dtype=np.int64);prows=parent.copy()
    if prows.max()>=ns:raise ValueError("repair parent outside source face domain")
    flat=[[],[],[]];foff=[0]
    for i in range(n):
        parents=np.unique(np.r_[prows[poff[i]:poff[i+1]],parent[i]])
        tuples=set()
        for p in parents:
            lo,hi=soff[p:p+2]
            tuples.update(zip(*(v[lo:hi].tolist() for v in lanes)))
        if not tuples:raise ValueError("empty composed face lineage")
        for t in sorted(tuples):
            for j in range(3):flat[j].append(t[j])
        foff.append(len(flat[0]))
    a["source_group_offsets"]=offsets
    a["source_group_face_rows"]=order
    a["face_group_id"]=group
    a["face_ancestry_offsets"]=np.asarray(foff,np.int64)
    for k,values in zip(CSR,flat):a[k]=np.asarray(values,np.int64)
    changes={}
    for k in IDENTITY:
        prior=np.asarray(a[k])
        current=np.asarray(s[k])[parent].copy()
        changes[k]=int(np.count_nonzero(prior!=current))
        # Retain the former representative as historical diagnostic metadata.
        if changes[k]:a["lineage_history_"+k]=prior.copy()
        a[k]=current
    schema_key="face_source_identity_schema_by_original_id_json"
    if schema_key not in s:
        raise ValueError("authenticated source lacks its typed identity schema")
    schema=json.loads(text(s[schema_key]))
    if not isinstance(schema,dict) or set(map(str,np.unique(a["face_source_original_id"])))-set(schema):
        raise ValueError("authenticated source identity schema is incomplete")
    if schema_key in a and text(a[schema_key])!=text(s[schema_key]):
        a["lineage_history_"+schema_key]=a[schema_key].copy()
    a[schema_key]=np.asarray(s[schema_key]).copy()
    a["envelope_source_face_row"]=parent.copy()
    a["envelope_source_original_id"]=a["face_source_original_id"].copy()
    a["face_exact_source_row"]=np.full(n,-1,np.int64)
    a["lineage_preparation_input_sha256"]=np.asarray([sha(candidate)])
    a["face_ancestry_method"]=np.asarray(["typed derivation CSR of each declared representative/repair source-parent set; " + source_semantics + "; not exact triangle-overlap ancestry; source-group CSR retains the full source inventory"])
    a["face_ancestry_note"]=a["face_ancestry_method"].copy()
    np.savez_compressed(output,**a)
    check=materialize(output)
    for k in ("faces","points_world_m","points_world_m_f32","points_world_m_f64",
              "vertex_parent_source_candidate_offsets","vertex_parent_source_candidate_rows"):
        if k in a and not np.array_equal(check[k],a[k]):raise AssertionError("emission changed "+k)
    if not np.array_equal(check["faces"],faces) or not np.array_equal(np.asarray(check["points_world_m"],np.float32),points):
        raise AssertionError("lineage repair changed geometry")
    result={"status":"PASS_LINEAGE_PREPARATION_GEOMETRY_UNCHANGED","candidate":str(candidate),
            "candidate_sha256":sha(candidate),"source":str(source),"source_sha256":source_sha,
            "output":str(output),"output_sha256":sha(output),"face_count":n,
            "source_group_count":len(keys),"old_group_offset_count":len(materialize(candidate)["source_group_offsets"]),
            "new_group_offset_count":len(offsets),"repaired_face_group_ids":int(np.count_nonzero(oldgroup!=group)),
            "repaired_representative_identity_lanes":changes,"repair_union_representatives_selected":int(np.count_nonzero(invalid_primary)),"composed_typed_ancestry_tuples":len(flat[0]),
            "ancestry_semantics":"derivation parents; representative mapping is not complete geometric overlap",
            "source_had_full_declared_typed_CSR":source_has_csr,
            "source_ancestry_semantics":source_semantics,
            "source_inventory_faces":ns,"geometry_bitwise_unchanged":True,
            "F32_points_sha256":hashlib.sha256(points.astype("<f4").tobytes()).hexdigest(),
            "faces_sha256":hashlib.sha256(faces.astype("<i8").tobytes()).hexdigest(),
            "exact_parent_claims":0,"physical_admission":False,"runtime":0,"accepted_steps":0,
            "owner_sha256":sha(__file__)}
    receipt.write_text(json.dumps(result,indent=2)+"\n")
    return result
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("candidate","source","output","receipt"):ap.add_argument("--"+name,required=True,type=Path)
    args=ap.parse_args();print(json.dumps(prepare(args.candidate,args.source,args.output,args.receipt)))
