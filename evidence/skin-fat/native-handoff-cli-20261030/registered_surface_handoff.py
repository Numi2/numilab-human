"""Typed registered surface preparation for the existing native volume owner.

Preserves declared derivation/source identities and rejects malformed or
unqualified reduction metadata. This performs no physical or volume admission.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_prepared_reduced_surface_217(module, path, layer, arrays=None):
    """Preserve prepared typed derivation ancestry without exact-parent claims."""
    if arrays is None:
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: np.asarray(z[key]) for key in z.files}
    a = arrays
    required = {"points_world_m", "faces", "face_group_id", "source_group_offsets",
                "source_group_face_rows", "source_group_keys_json",
                "face_parent_source_face_rows", "face_exact_source_row",
                "face_ancestry_offsets", "face_ancestry_source_face_rows",
                "face_ancestry_source_original_ids", "face_ancestry_parent_face_rows",
                "face_source_layer", "face_source_body", "face_source_bone",
                "face_source_face_id", "face_source_original_id",
                "face_source_identity0", "face_source_identity1",
                "face_source_identity_schema_by_original_id_json",
                "lineage_preparation_input_sha256", "face_ancestry_method"}
    if required - set(a):
        raise ValueError("prepared reduced surface lacks typed provenance: " + str(sorted(required-set(a))))
    points = np.asarray(a["points_world_m"], np.float32)
    faces = np.asarray(a["faces"], np.int64)
    n = len(faces)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or
        faces.shape != (n,3) or not n or faces.min()<0 or faces.max()>=len(points)):
        raise ValueError("prepared reduced geometry malformed")
    tri=points[faces].astype(np.float64)
    if np.any(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0):
        raise ValueError("prepared reduced geometry has collapsed F32 triangles")
    keys=json.loads(str(a["source_group_keys_json"].reshape(-1)[0]))
    offsets=np.asarray(a["source_group_offsets"],np.int64)
    rows=np.asarray(a["source_group_face_rows"],np.int64)
    group=np.asarray(a["face_group_id"],np.int64)
    parent=np.asarray(a["face_parent_source_face_rows"],np.int64)
    if (offsets.shape!=(len(keys)+1,) or offsets[0]!=0 or offsets[-1]!=len(rows) or
        np.any(np.diff(offsets)<=0) or not np.array_equal(np.sort(rows),np.arange(len(rows))) or
        group.shape!=(n,) or group.min()<0 or group.max()>=len(keys) or
        parent.shape!=(n,) or parent.min()<0 or parent.max()>=len(rows)):
        raise ValueError("prepared reduced typed source-group CSR malformed")
    source_group=np.empty(len(rows),np.int64)
    for i in range(len(keys)): source_group[rows[offsets[i]:offsets[i+1]]]=i
    if not np.array_equal(source_group[parent],group):
        raise ValueError("prepared reduced primary source/group disagreement")
    for k in ("face_source_layer","face_source_body","face_source_bone","face_source_original_id"):
        values=np.asarray(a[k],np.int64)
        if values.shape!=(n,) or any(values[i]!=keys[int(group[i])][k] for i in range(n)):
            raise ValueError("prepared reduced anatomical group identity mismatch")
    ao=np.asarray(a["face_ancestry_offsets"],np.int64)
    lanes=[np.asarray(a[k],np.int64) for k in
           ("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows")]
    if (ao.shape!=(n+1,) or ao[0]!=0 or np.any(np.diff(ao)<=0) or
        any(len(v)!=ao[-1] or np.any(v<0) for v in lanes)):
        raise ValueError("prepared reduced face ancestry CSR malformed")
    if np.asarray(a["face_exact_source_row"]).shape!=(n,) or np.any(a["face_exact_source_row"]!=-1):
        raise ValueError("prepared reduction may not invent an exact source face")
    schema=json.loads(str(a["face_source_identity_schema_by_original_id_json"].reshape(-1)[0]))
    if set(map(str,np.unique(a["face_source_original_id"])))-set(schema):
        raise ValueError("prepared reduced identity schema incomplete")
    result={"path":str(path),"points":points,"faces":faces,"source_rows":parent,
            "exact_source_rows":np.full(n,-1,np.int64),
            "source_original_ids":np.asarray(a["face_source_original_id"],np.int64),
            "source_layers":np.asarray(a["face_source_layer"],np.int8),
            "source_bodies":np.asarray(a["face_source_body"],np.int32),
            "source_bones":np.asarray(a["face_source_bone"],np.int32),
            "source_run_original_ids":np.asarray(a["face_source_original_id"],np.int64),
            "source_face_ids":np.asarray(a["face_source_face_id"],np.int64),
            "source_identity0":np.asarray(a["face_source_identity0"],np.int64),
            "source_identity1":np.asarray(a["face_source_identity1"],np.int64),
            "ancestry_kind":np.full(n,6,np.int8),"typed_identity_schema":schema,
            "typed_source_npz_sha256":sha(path),"raw_npz_keys":sorted(a),
            "reduced_ancestry_semantics":str(a["face_ancestry_method"].reshape(-1)[0]),
            "reduced_source_group_offsets":offsets,"reduced_source_group_face_rows":rows}
    for name,values in zip(("ancestry","ancestry_original_ids","ancestry_parent_rows"),lanes):
        result[name]=[values[ao[i]:ao[i+1]].tolist() for i in range(n)]
    return result

def load_direct_typed_surface_217(module, path, layer):
    """Adapt an unreduced typed I source through its actual declared identities."""
    with np.load(path, allow_pickle=False) as z:
        a = {key: np.asarray(z[key]) for key in z.files}
    if any(k in a for k in ("source_candidate_sha256","face_parent_source_face_rows","face_group_id")):
        raise ValueError("reduced typed surface requires its prepared lineage route")
    n=len(a["faces"])
    fields=("face_source_layer","face_source_body","face_source_bone","face_source_original_id")
    signatures=np.stack([np.asarray(a[k],np.int64) for k in fields],axis=1)
    unique,group=np.unique(signatures,axis=0,return_inverse=True)
    keys=[dict(zip(fields,map(int,row))) for row in unique]
    a["face_group_id"]=group
    a["source_group_offsets"]=np.r_[0,np.cumsum(np.bincount(group,minlength=len(keys)))].astype(np.int64)
    a["source_group_face_rows"]=np.argsort(group,kind="stable").astype(np.int64)
    a["source_group_keys_json"]=np.asarray([json.dumps(keys,separators=(",",":"))])
    a["face_parent_source_face_rows"]=np.arange(n,dtype=np.int64)
    a["face_exact_source_row"]=np.full(n,-1,np.int64)
    if "face_ancestry_offsets" not in a:
        a["face_ancestry_offsets"]=np.arange(n+1,dtype=np.int64)
        a["face_ancestry_source_face_rows"]=np.arange(n,dtype=np.int64)
        a["face_ancestry_source_original_ids"]=np.asarray(a["face_source_original_id"],np.int64)
        a["face_ancestry_parent_face_rows"]=np.asarray(a["face_source_face_id"],np.int64)
    a["lineage_preparation_input_sha256"]=np.asarray([sha(path)])
    a["face_ancestry_method"]=np.asarray(["declared typed identities of this authenticated unreduced source asset; no exact original render-face or full historical overlap claim"])
    return load_prepared_reduced_surface_217(module,path,layer,arrays=a)
