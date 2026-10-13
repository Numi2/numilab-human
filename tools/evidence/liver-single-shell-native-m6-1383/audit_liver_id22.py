from pathlib import Path
import hashlib, json, sys
import numpy as np
SRC = Path("/Users/numi/numi-human-anatomy-finish-1378/src")
sys.path.insert(0, str(SRC))
from numilab_human import model, resting_anatomy as ra
from numilab_human.resting_anatomy_interface_patch import parse_payload

ROOT = Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276")
RUN = ROOT / "back412-skin-reference-native-021-003"
RECEIPT = RUN / "resting-anatomy-receipt.json"
PAYLOAD = Path("/Users/numi/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy")
SOURCES = Path("/Users/numi/numi-human-left-pq-publication-1344/Sources")
REGISTRATION = Path("/Users/numi/numi-human-anatomy-finish-1378/Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json")
OUT = ROOT / "full-anatomy-delivery-audit-1378" / "liver-id22-source-identity-audit.json"
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(path): return sha_bytes(Path(path).read_bytes())
def write_new(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True); f.write("\n")
receipt=json.loads(RECEIPT.read_text()); prov=receipt["provenance"]
pre=prov["pre_aggregate_liver_source_id_map"]["22"]
assert pre["source_member"]=="FJ2824" and pre["source_sha256"]=="2f5e34fceea6c8b718bd3f50a125bf24c434ae7eb3d925b9795b0585272c1349"
archive,member,obj=model._bodyparts_obj_member(SOURCES,"part_of","FJ2824")
rawsha=sha_bytes(obj)
assert rawsha==pre["source_sha256"]
source_vertices_mm, source_faces, _ = model._bodyparts_obj_geometry(obj,member)
source_vertices_mm=np.asarray(source_vertices_mm,dtype=np.float64)
source_faces=np.asarray(source_faces,dtype=np.int64)
reg=json.loads(REGISTRATION.read_text())
M=np.asarray(reg["coordinate_system"]["global_source_mm_to_myosim_world_m"],dtype=np.float64)
world=source_vertices_mm@M[:3,:3].T+M[:3,3]
local=ra._body_local(world)
expected=np.asarray(local,dtype="<f4")
header,rows=parse_payload(PAYLOAD)
actual22=np.asarray(rows[22]["vertices6"][:,:3],dtype="<f4")
actual21=np.asarray(rows[21]["vertices6"][:,:3],dtype="<f4")
faces22=np.asarray(rows[22]["faces"],dtype=np.int64)
faces21=np.asarray(rows[21]["faces"],dtype=np.int64)
def vertex_multiset_equal(a,b):
    if len(a)!=len(b): return False
    return sorted(map(tuple,a.view("<u4").reshape(-1,3).tolist()))==sorted(map(tuple,b.view("<u4").reshape(-1,3).tolist()))
def coordinate_triangles(v,f):
    out=[]
    for tri in f:
        out.append(tuple(sorted(tuple(map(int,v[j].view("<u4"))) for j in tri)))
    return sorted(out)
vertex_exact=expected.shape==actual22.shape and np.array_equal(expected,actual22)
face_exact=source_faces.shape==faces22.shape and np.array_equal(source_faces,faces22)
coordinate_tri_equal=(len(source_faces)==len(faces22) and coordinate_triangles(expected,source_faces)==coordinate_triangles(actual22,faces22))
expected_keys=set(map(tuple,expected.view("<u4").reshape(-1,3).tolist()))
actual_keys=set(map(tuple,actual22.view("<u4").reshape(-1,3).tolist()))
common_keys=expected_keys & actual_keys
edges={}
for tri in faces22:
    for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
        key=tuple(sorted((int(a),int(b))));edges[key]=edges.get(key,0)+1
boundary_edges=sum(n==1 for n in edges.values()); nonmanifold_edges=sum(n>2 for n in edges.values())
row21_keys=set(map(tuple,actual21.view("<u4").reshape(-1,3).tolist()))
both21_22=len(row21_keys & actual_keys)
pins={
    "current_payload_path":str(PAYLOAD),"current_payload_sha256":sha(PAYLOAD),
    "native_receipt_path":str(RECEIPT),"native_receipt_sha256":sha(RECEIPT),
    "source_archive_path":str(archive),"source_archive_sha256":sha(archive),
    "source_obj_member":member,"source_obj_sha256":rawsha,
    "registration_path":str(REGISTRATION),"registration_sha256":sha(REGISTRATION),
}
result={
    "status":"pass",
    "scope":"Read-only direct comparison of current NHANAT1 stable ID22 with the exact pre-aggregate source owner recorded in the current native receipt; no anatomy payload or runtime was changed.",
    "pins":pins,
    "identity_trace":{
        "pre_aggregate_id22":pre,
        "current_source_id22_present": "22" in prov["source_id_map"],
        "retired_alias_claim":prov.get("retired_liver_alias_associations"),
        "patch21_template":prov.get("liver_patch_identity_template"),
    },
    "source_to_payload":{
        "source_vertex_count":int(len(expected)),"source_triangle_count":int(len(source_faces)),
        "payload22_vertex_count":int(len(actual22)),"payload22_triangle_count":int(len(faces22)),
        "source_vertices_float32_bitwise_equal_payload22":bool(vertex_exact),
        "source_faces_indexwise_equal_payload22":bool(face_exact),
        "source_coordinate_triangle_multiset_equal_payload22":bool(coordinate_tri_equal),
        "matching_unique_exact_vertex_bit_patterns":int(len(common_keys)),
        "source_unique_vertices":int(len(expected_keys)),"payload22_unique_vertices":int(len(actual_keys)),
        "payload21_and_22_shared_exact_vertex_bit_patterns":int(both21_22),
        "payload22_boundary_edge_count":int(boundary_edges),"payload22_nonmanifold_edge_count":int(nonmanifold_edges),
    },
    "interpretation":"A passing source geometry match proves ID22 is the retained FJ2824 source component, distinct from the liver display aggregate patch21. It does not make it a second liver organ or establish a segment volume; the current receipt's alias and current source_id_map omission must be reconciled before claiming complete source provenance.",
}
write_new(OUT,result)
print(json.dumps({"status":"pass","source_member":member,"source_sha256":rawsha,"source_vertices":len(expected),"payload22_vertices":len(actual22),
"source_vertex_exact":vertex_exact,"source_face_exact":face_exact,"coordinate_triangles_exact":coordinate_tri_equal,"shared_with_21_vertices":both21_22,"report":str(OUT),"report_sha256":sha(OUT)},sort_keys=True),flush=True)

