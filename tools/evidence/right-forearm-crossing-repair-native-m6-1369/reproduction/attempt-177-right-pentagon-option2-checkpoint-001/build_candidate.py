from __future__ import annotations
import copy, hashlib, json, pathlib, pickle

BASE = pathlib.Path("/Users/n/numi-human-fds132-m6-1350")
SRC = BASE / "attempt-166-right-capB-ancestry-correction-001/result-001/candidate.pkl"
SRC_SHA = "25d759377d4861660f12c6458cd5bde4d48e152462048f15aea6410b70b2c17d"
OUT = BASE / "attempt-177-right-pentagon-option2-checkpoint-001/result-002"
OUT.mkdir(parents=True, exist_ok=False)
raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA
parent = pickle.loads(raw)
candidate = copy.deepcopy(parent)

patch_ids = [3304, 3323, 3599]
new_global = [
    (4632, 4668, 4675),
    (4632, 4661, 4668),
    (4668, 4671, 4675),
]
old_global = [tuple(map(int, candidate["surfaces"][133]["faces_global"][i])) for i in patch_ids]
assert old_global == [(4632,4661,4675),(4661,4668,4675),(4668,4671,4675)], old_global

# Map stable global IDs to each row's local vertex namespace explicitly.
for sid in (131,133):
    surf = candidate["surfaces"][sid]
    ids = surf["global_vertex_ids"]
    assert len(ids) == len(set(ids))
    assert len(surf["faces"]) == len(surf["faces_global"]) == len(surf["roles"])
    g2l = {int(g): i for i,g in enumerate(ids)}
    assert all(tuple(ids[int(j)] for j in face) == tuple(map(int,gface))
               for face,gface in zip(surf["faces"], surf["faces_global"]))
    if sid == 133:
        for i, tri in zip(patch_ids,new_global):
            assert all(g in g2l for g in tri)
            surf["faces_global"][i] = tuple(tri)
            surf["faces"][i] = tuple(g2l[g] for g in tri)

# Pool transitive source ancestry across all three predecessors.
pooled = [[133,2594],[133,2596],[133,2597],[133,2846]]
for i in patch_ids:
    assert int(candidate["surfaces"][133]["parent_ids"][i]) == 133
    assert [133,int(candidate["surfaces"][133]["parent_faces"][i])] in pooled

role_overrides = []
for sid,face_i in ((131,6543),(133,3323)):
    surf = candidate["surfaces"][sid]
    before = int(surf["roles"][face_i])
    assert before == 0
    surf["roles"][face_i] = 1
    role_overrides.append({"stable_id":sid,"face_index":face_i,"from_role":before,"to_role":1,
                           "reason":"opposite-winding exact facet introduced by synchronized pentagon retessellation; contiguous with retained paired-interface patch"})

# Preserve the oriented patch boundary exactly.
def directed_boundary(tris):
    counts = {}
    for tri in tris:
        a,b,c = tri
        for e in ((a,b),(b,c),(c,a)):
            rev=(e[1],e[0])
            if counts.get(rev,0):
                counts[rev]-=1
            else:
                counts[e]=counts.get(e,0)+1
    return sorted(e for e,n in counts.items() for _ in range(n) if n)
old_boundary=directed_boundary(old_global)
new_boundary=directed_boundary(new_global)
assert old_boundary == new_boundary == sorted([(4632,4661),(4661,4668),(4668,4671),(4671,4675),(4675,4632)])

left = tuple(map(int,candidate["surfaces"][131]["faces_global"][6543]))
right = tuple(map(int,candidate["surfaces"][133]["faces_global"][3323]))
assert left == (4632,4668,4661) and right == (4632,4661,4668)
assert set(((left[0],left[1]),(left[1],left[2]),(left[2],left[0]))) == set(((right[1],right[0]),(right[2],right[1]),(right[0],right[2])))

candidate["pentagon_retriangulation_177"] = {
  "schema":"numi.human.right-pentagon-retriangulation-checkpoint-177.v1",
  "parent_candidate_path":str(SRC),"parent_candidate_sha256":SRC_SHA,
  "patch_face_indices":patch_ids,"old_global_triangles":[list(t) for t in old_global],
  "new_global_triangles":[list(t) for t in new_global],
  "patch_boundary_global_directed_edges":[list(e) for e in new_boundary],
  "pooled_transitive_source_parent_ancestry_by_modified_133_face":{str(i):pooled for i in patch_ids},
  "role_overrides":role_overrides,
  "new_shared_interface_pair":{"stable_id_131_face_index":6543,"stable_id_131_triangle_global_ids":list(left),
    "stable_id_133_face_index":3323,"stable_id_133_triangle_global_ids":list(right),"opposite_winding":True,
    "classification":"inferred extension of retained source-partition interface; not a measured attachment"},
  "shared_interface_pair_count_before":2275,"shared_interface_pair_count_after":2276,
  "source_xyz_unchanged":True,"full21_weights_unchanged":True,"candidate_admission":False,
  "deviation_evidence":{"sampled_symmetric_patch_deviation_m":1.1287937033664741e-6,"conservative_bound":"not yet established"},
  "source_patch_area_delta_fraction":0.022621723322148846,
  "source_signed_volume_delta_m3":1.063369940109139e-14,
  "source_row_components":4,"source_union_components":12,
  "screen_report_path":str(BASE/"attempt-175-right-pentagon-variants-helper-001/report.json"),
  "screen_report_sha256":"8ac613512e8652ff231eb3d1dde7842d56197b5e6ccd044413c6de00d1a38c1d",
}

for sid in (131,133):
    assert candidate["surfaces"][sid]["global_vertex_ids"] == parent["surfaces"][sid]["global_vertex_ids"]
    assert candidate["surfaces"][sid]["vertices_exact_m"] == parent["surfaces"][sid]["vertices_exact_m"]
    assert candidate["surfaces"][sid]["vertices_float32_m"] == parent["surfaces"][sid]["vertices_float32_m"]
    assert candidate["surfaces"][sid]["core_weights_float64"] == parent["surfaces"][sid]["core_weights_float64"]

cand_path=OUT/"candidate.pkl"
cand_bytes=pickle.dumps(candidate,protocol=pickle.HIGHEST_PROTOCOL)
cand_path.write_bytes(cand_bytes)
cand_sha=hashlib.sha256(cand_bytes).hexdigest()
report={"schema":"numi.human.right-pentagon-retriangulation-checkpoint-177-report.v1",
  "candidate_path":str(cand_path),"candidate_sha256":cand_sha,"candidate_bytes":len(cand_bytes),
  "source_candidate_path":str(SRC),"source_candidate_sha256":SRC_SHA,
  "source_xyz_unchanged":True,"full21_weights_unchanged":True,
  "old_global_triangles":[list(t) for t in old_global],"new_global_triangles":[list(t) for t in new_global],
  "oriented_patch_boundary_preserved":True,"new_shared_opposite_winding_pair":True,
  "role_overrides":role_overrides,"transitive_ancestry_pool":pooled,
  "shared_interface_pair_count_before":2275,"shared_interface_pair_count_after":2276,
  "candidate_admission":False,"fullsource_and_fullpose_audits_pending":True,
  "deviation_bound":"not yet established; sampled 1.1287937033664741e-6 m is not a conservative bound"}
(OUT/"checkpoint-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,sort_keys=True))
