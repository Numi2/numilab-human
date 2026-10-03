"""Independently compare raw and split scan-001 lower-limb voxel meshes."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from numilab_human.healthy_total_body_ct_surface_audit import (
    _bad_vertex_links, _edge_incidence_histogram, _read_binary_ply_gzip, _sha256_file,
)
from numilab_human.physiology import canonical

ROOT = Path.cwd()
BASE = ROOT / "Docs/media/healthy-total-body-ct-surface-20261003"
CONTROL = BASE / "skeletal-scan-001-run-v1/receipt.json"
CANDIDATE = BASE / "lower-limb-contact-split-scan-001-v1/receipt.json"
CONTROL_AUDIT = BASE / "independent-skeletal-set-audit-v1.json"
CANDIDATE_AUDIT = BASE / "independent-lower-limb-contact-split-scan-001-audit-v1.json"
OUTPUT = BASE / "lower-limb-contact-split-controlled-comparison-v1.json"
LABEL_IDS = [15, 16, 20, 30]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def geom(vertices, faces):
    triangles = vertices[faces].astype(np.float64)
    area = float((0.5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]),axis=1)).sum())
    relative = triangles - vertices.astype(np.float64).mean(axis=0)
    volume = float(np.einsum("ij,ij->i", relative[:,0], np.cross(relative[:,1],relative[:,2])).sum(dtype=np.float64)/6.0)
    return area, volume

def require(ok, message):
    if not ok: raise ValueError("lower-limb controlled comparison: "+message)

raw = json.loads(CONTROL.read_text())
new = json.loads(CANDIDATE.read_text())
raw_audit = json.loads(CONTROL_AUDIT.read_text())
new_audit = json.loads(CANDIDATE_AUDIT.read_text())
require(raw_audit["compiler_receipt_sha256"] == sha(CONTROL), "raw compiler receipt hash mismatch")
require(raw_audit["all_mesh_files_and_source_identity_checks_pass"] is True, "raw independent audit failed")
require(new_audit["runs"][0]["compiler_receipt_sha256"] == sha(CANDIDATE), "candidate compiler receipt hash mismatch")
require(new_audit["runs"][0]["all_independent_source_and_mesh_geometry_checks_pass"] is True, "candidate independent audit failed")
require(raw["scan"] == new["scan"] and raw["source"] == new["source"], "source, scan, or affine identity differs")
require(new["topology_method"] == "voxel_boundary_contact_fan_split_candidate", "candidate method differs")
raw_meshes = {int(x["label_id"]): x for x in raw["meshes"]}
new_meshes = {int(x["label_id"]): x for x in new["meshes"]}
raw_checked = {int(x["label_id"]): x for x in raw_audit["selected_meshes"]}
new_run = new_audit["runs"][0]
new_checked = {int(x["label_id"]): x for x in new_run["meshes"]}
rows=[]
for label_id in LABEL_IDS:
    old, cur = raw_meshes[label_id], new_meshes[label_id]
    old_check, cur_check = raw_checked[label_id], new_checked[label_id]
    old_path, cur_path = CONTROL.parent/old["mesh_file"], CANDIDATE.parent/cur["mesh_file"]
    require(_sha256_file(old_path)==old["mesh_file_sha256"]==old_check["mesh_sha256"], f"raw mesh hash mismatch: {label_id}")
    require(_sha256_file(cur_path)==cur["mesh_file_sha256"]==cur_check["mesh_sha256"], f"candidate mesh hash mismatch: {label_id}")
    require(old_check["ply_gzip_index_envelope_and_edge_checks_pass"] and cur_check["compiler_topology_measurements_reproduced"], f"mesh audit failed: {label_id}")
    old_v, old_f, _ = _read_binary_ply_gzip(old_path, np)
    cur_v, cur_f, _ = _read_binary_ply_gzip(cur_path, np)
    same = bool(np.array_equal(old_v[old_f], cur_v[cur_f]))
    old_area, old_volume = geom(old_v, old_f)
    cur_area, cur_volume = geom(cur_v, cur_f)
    hist = _edge_incidence_histogram(cur_f, np)
    bad = _bad_vertex_links(cur_f, len(cur_v))
    closed = set(hist)=={"2"} and bad==0
    require(same and len(old_f)==len(cur_f), f"triangle geometry changed: {label_id}")
    require(closed == cur["mesh_metrics"]["closed_two_manifold"], f"candidate topology receipt mismatch: {label_id}")
    require(abs(cur_volume-cur["mesh_metrics"]["source_voxel_occupancy_volume_candidate_mm3"])/cur["mesh_metrics"]["source_voxel_occupancy_volume_candidate_mm3"] <= 1e-9, f"candidate occupancy mismatch: {label_id}")
    rows.append({
        "label_id":label_id,"label_name":cur["source_label_name"],
        "raw_ply_sha256":old["mesh_file_sha256"],"candidate_ply_sha256":cur["mesh_file_sha256"],
        "raw_closed_two_manifold":old_check["closed_two_manifold"],"candidate_closed_two_manifold":closed,
        "candidate_edge_incidence_histogram":hist,"candidate_nonmanifold_vertex_count":bad,
        "triangle_count_unchanged":len(old_f)==len(cur_f),"triangle_coordinate_sequence_identical":same,
        "surface_area_absolute_delta_mm2":abs(cur_area-old_area),
        "signed_volume_absolute_delta_mm3":abs(cur_volume-old_volume),
        "candidate_relative_occupancy_volume_error":abs(cur_volume-cur["mesh_metrics"]["source_voxel_occupancy_volume_candidate_mm3"])/cur["mesh_metrics"]["source_voxel_occupancy_volume_candidate_mm3"],
    })
result={
 "schema":"numi.healthy-total-body-ct-lower-limb-contact-split-controlled-comparison.v1",
 "status":"controlled_geometry_comparison_complete",
 "scan_id":"001","source_archive_sha256":new["source"]["archive_sha256"],
 "control_receipt_sha256":sha(CONTROL),"candidate_receipt_sha256":sha(CANDIDATE),
 "control_independent_audit_sha256":sha(CONTROL_AUDIT),"candidate_independent_audit_sha256":sha(CANDIDATE_AUDIT),
 "analysis_source_sha256":sha(Path(__file__)),
 "surface_count":len(rows),"raw_closed_surface_count":sum(r["raw_closed_two_manifold"] for r in rows),
 "candidate_closed_surface_count":sum(r["candidate_closed_two_manifold"] for r in rows),
 "defective_surfaces_closed_by_candidate":sum((not r["raw_closed_two_manifold"]) and r["candidate_closed_two_manifold"] for r in rows),
 "all_triangle_counts_unchanged":all(r["triangle_count_unchanged"] for r in rows),
 "all_triangle_coordinate_sequences_identical":all(r["triangle_coordinate_sequence_identical"] for r in rows),
 "maximum_surface_area_absolute_delta_mm2":max(r["surface_area_absolute_delta_mm2"] for r in rows),
 "maximum_absolute_signed_volume_delta_mm3":max(r["signed_volume_absolute_delta_mm3"] for r in rows),
 "maximum_candidate_relative_occupancy_volume_error":max(r["candidate_relative_occupancy_volume_error"] for r in rows),
 "rows":rows,
 "boundary":"Scan-001 automatic segmentation topology comparison only. Identical source triangle coordinates and occupancy arithmetic do not establish expert segmentation accuracy, clinical anatomy, cross-scan registration, Numi subject binding, physical tissue ownership, mechanics, or physiology."
}
require(result["candidate_closed_surface_count"]==4 and result["defective_surfaces_closed_by_candidate"]==2, "expected raw-to-candidate topology transition differs")
require(result["all_triangle_counts_unchanged"] and result["all_triangle_coordinate_sequences_identical"] and result["maximum_surface_area_absolute_delta_mm2"]==0.0, "geometry preservation gate failed")
require(result["maximum_candidate_relative_occupancy_volume_error"]<=1e-9, "occupancy volume gate failed")
require(not OUTPUT.exists(), "output exists; reports are immutable")
OUTPUT.write_bytes(canonical(result)+b"\n")
print(json.dumps({"output":str(OUTPUT),"sha256":sha(OUTPUT),"raw_closed":result["raw_closed_surface_count"],"candidate_closed":result["candidate_closed_surface_count"],"fixed":result["defective_surfaces_closed_by_candidate"],"max_volume_delta_mm3":result["maximum_absolute_signed_volume_delta_mm3"],"max_occupancy_error":result["maximum_candidate_relative_occupancy_volume_error"]},sort_keys=True))
