#!/usr/bin/env python3
"""Read-only owner/map/area/resource preflight for the 1096 geometry-only census."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_lung_cycle_1096 as audit

E = audit.E
V8 = E / "native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8"
RUN = E / "final-native-scene-preflight-936/skin-927-lung-1113-viewer-018-v015-attempt1/native-run"
cfg = {
    "base": audit.BASE,
    "run": RUN,
    "nha_path": V8 / "final/resting-thorax.nhanatomy",
    "nha_sha": "1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc",
    "map_reports": [],
    "d_map_composition_report": V8 / "composition-report.json",
    "lobe_lineage_report": V8 / "current-reciprocal-map-report-v2.json",
    "geometry_only_area_mismatch": True,
}
base, adapters, ctx = audit.prepare_worker_context(cfg)
if ctx["steps"] != [0, 4991, 5375, 5759, 6111, 6495, 7743, 10000]:
    raise SystemExit("accepted capture sequence differs from predeclared eight poses")
if ctx["requested_roots"] != 10000 or abs(ctx["dt_seconds"] - 0.0020000000949949026) > 1e-15:
    raise SystemExit("accepted horizon or Float32 clock differs")
if ctx["d_map_doc"]["map_count"] != 47343:
    raise SystemExit("D map face count mismatch")
if ctx["d_map_doc"]["counts_by_lobe"] != {305: 19743, 306: 21600, 307: 5514, 308: 486}:
    raise SystemExit("D map per-lobe count mismatch")
if ctx["d_map_doc"]["zero_lobes"] != [309] or len(ctx["lineage"]["pairs"]) != 10:
    raise SystemExit("explicit zero pair or ten-pair lobe lineage missing")
if ctx["area_binding"]["status"] != "FAIL_geometry_or_payload_binding":
    raise SystemExit("selected run must retain its actual failed respiration-area binding")
if ctx["area_binding"]["mismatch_fields"] != ["payload_sha256", "per_lobe_geometry_sha256"]:
    raise SystemExit("unexpected area binding mismatch classification")
probe = json.loads(audit.MEMORY_PROBE_REPORT.read_text())
triangles = sum(len(ctx["nha_rows"][sid]["faces"]) for sid in audit.EXPECTED_ROWS)
memory = audit.validate_memory_probe(
    probe, probe_path=audit.MEMORY_PROBE_REPORT,
    reference_triangle_count=int(probe["face_count_reference"]),
    worker_count=2, candidate_triangle_count=triangles,
)
result = {
    "schema": "numi.human.native-transformed-lung-audit-preflight.v1",
    "status": "READY_GEOMETRY_ONLY_AREA_BINDING_FAILED",
    "run": {"path": str(ctx["run_path"]), "accepted_steps": ctx["steps"],
            "requested_roots": ctx["requested_roots"], "dt_seconds": ctx["dt_seconds"]},
    "NHA": {"path": str(ctx["nha_path"]), "sha256": ctx["nha_sha"]},
    "D_lobe_map": {"composition_report": ctx["d_map_doc"]["path"],
                   "composition_report_sha256": ctx["d_map_doc"]["sha256"],
                   "map": ctx["d_map_doc"]["map_path"],
                   "map_sha256": ctx["d_map_doc"]["map_sha256"],
                   "map_count": ctx["d_map_doc"]["map_count"],
                   "counts_by_lobe": ctx["d_map_doc"]["counts_by_lobe"],
                   "zero_lobes": ctx["d_map_doc"]["zero_lobes"]},
    "lobe_lineage": {"report": ctx["lineage"]["path"], "sha256": ctx["lineage"]["sha256"],
                     "face_map": ctx["lineage"]["map_path"],
                     "face_map_sha256": ctx["lineage"]["map_sha256"],
                     "edge_map": ctx["lineage"]["edge_path"],
                     "edge_map_sha256": ctx["lineage"]["edge_sha256"],
                     "pairs": len(ctx["lineage"]["pairs"]),
                     "source_simplex_validation": ctx["lineage"]["source_simplex_validation"]},
    "respiration_area_binding": ctx["area_binding"],
    "resource_plan": memory,
    "tracked_input_count": len(ctx["tracked"]),
    "input_hashes": ctx["input_hashes"],
    "geometry_scan_started": False,
    "qualification": "Owner and source maps are bound; respiration geometry derivation is stale for selected payload. This preflight authorizes only explicitly geometry-only audit output.",
}
out = HERE / "preflight-native1113-001.json"
audit.write_json(out, result)
print(json.dumps({"status": result["status"], "report": str(out),
                  "tracked_inputs": len(ctx["tracked"]), "triangles": triangles,
                  "workers": 2, "estimated_total_bytes": memory["total_estimated_bytes"],
                  "area_mismatch_fields": ctx["area_binding"]["mismatch_fields"]}, sort_keys=True))
