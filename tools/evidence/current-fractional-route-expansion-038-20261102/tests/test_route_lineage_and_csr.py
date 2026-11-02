from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
FIXTURES = ROOT / "fixtures"


def read_json(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def test_generator_accepts_canonical_run_config_suffix_and_self_template_receipt():
    source = (ROOT / "src/prepare_run_manifest_038c.py").read_text(encoding="utf-8")
    predicate = 'Path(str(row.get("path", ""))).name in {"run-config.json", "run-config-038.json"}'
    second = 'Path(row["path"]).name in {"run-config.json", "run-config-038.json"}'
    assert source.count(predicate) == 1
    assert source.count(second) == 1

    manifest = read_json("out-007-selftest-run-manifest.json")
    preflight = read_json("out-007-selftest-preflight.json")
    config = read_json("out-007-selftest-run-config-038.json")
    assert Path(manifest["template_manifest"]["path"]).name == "run-manifest.json"
    assert Path(manifest["template_manifest"]["path"]).parent.name == "out-007"
    assert Path(manifest["template_run_config"]["path"]).name == "run-config-038.json"
    assert config["maximum_wall_seconds"] == 1800
    assert preflight["status"] == "PASS_LAUNCH_READY_NOT_RUN"
    assert preflight["native_steps"] == 0
    assert preflight["physical_admission"] is False


def test_out007_is_pinned_source_only_and_binds_route_receipt():
    manifest = read_json("out-007-run-manifest.json")
    preflight = read_json("out-007-preflight.json")
    append = read_json("whole-cut-route-skin-append-validation.json")
    digest = sha256(EVIDENCE / "out-007-run-manifest.json")

    assert preflight["status"] == "PASS_LAUNCH_READY_NOT_RUN"
    assert preflight["run_manifest"]["sha256"] == digest
    assert preflight["source_control_sha256"] == manifest["source_control_manifest"]["sha256"]
    assert preflight["source_closure_sha256"] == manifest["source_closure"]["sha256"]
    assert preflight["maximum_transport_distance_m"] == "0.187444"
    assert manifest["maximum_wall_seconds"] == 1800
    assert manifest["physical_admission"] is False
    assert manifest["native_steps"] == 0

    skin = manifest["named_input_pins"]["carrier_witness_skin"]
    fat = manifest["named_input_pins"]["carrier_witness_fat"]
    lineage = manifest["carrier_witness_skin_route_lineage"]
    assert skin["sha256"] == append["output_csv"]["sha256"]
    assert skin["bytes"] == append["output_csv"]["bytes"]
    assert fat["sha256"] == append["pins"]["fat_v15_unchanged"]["sha256"]
    assert lineage["route_pair_count"] == 300860
    assert lineage["physical_admission"] is False and lineage["native_steps"] == 0

    file_pins = {(row["path"], row["sha256"]) for row in manifest["file_pins"]}
    for pin in (lineage["route_census_report"], lineage["route_proposal_npz"],
                lineage["append_validation"], lineage["manifest_binding_source"]):
        assert (pin["path"], pin["sha256"]) in file_pins
    generator_refs = {row["sha256"] for row in manifest["file_pins"]
                      if Path(row["path"]).name in {"prepare_run_manifest_038b.py", "prepare_run_manifest_038c.py"}}
    assert generator_refs == {
        "9089b62122ccab5270a3d084b84c571ee127fc6a5a105c3c1c14c36e4d82a24e",
        "ea830f1c9cd4211642c80c32cd7fc532ad0fa64eb85fadd2d8b426cd43575887",
    }


def test_failed_source_variants_are_pinned_with_their_failure_receipts():
    history = read_json("attempt-history.json")
    assert "one full allocator diagnostic" in history["scope"]
    by_owner = {row["owner"]: row for row in history["attempts"]}
    for filename in ("prepare_run_manifest_038.py",
                     "append_skin_witnesses_from_whole_cut_routes_038.py",
                     "append_skin_witnesses_from_whole_cut_routes_038b.py"):
        source = ROOT / "failed-attempts" / filename
        assert sha256(source) == by_owner[filename]["source_sha256"]
    assert by_owner["append_skin_witnesses_from_whole_cut_routes_038c.py"]["status"] == "CSV_WRITTEN_RECEIPT_ASSEMBLY_FAILED"
    assert by_owner["verify_and_finalize_skin_witness_append_038d.py"]["status"] == "PASS_EXISTING_OUTPUT_RECOVERY"


def test_recovered_whole_cut_census_remains_nonadmitted():
    report = read_json("whole-cut-route-census-report.json")
    append = read_json("whole-cut-route-skin-append-validation.json")
    assert report["status"] == "WHOLE_T_CROSSING_ROUTE_CENSUS_PASS_REPORT_RECOVERED"
    assert report["physical_admission"] is False
    assert report["proposal_artifact"]["rows"] == 300860
    assert report["allowlist_reconstruction"]["status"] == "PASS_EXACT_PARITY_BEFORE_PROPOSAL_WRITE"
    assert report["allowlist_reconstruction"]["candidate_pair_already_present"] == 0
    assert report["whole_T_routes"]["T_children_covered_by_any_new_route"] == 72675
    assert report["whole_T_routes"]["T_children_without_any_new_route"] == 282518
    assert append["status"] == "PASS_ALL_ROUTE_SLOTS_REPLAYED_AND_APPENDED"
    assert append["validation"]["capacity_interpretation"].startswith("cut-arc capacity sum only")
    assert append["physical_admission"] is False and append["native_steps"] == 0


def test_expanded_032d_carrier_csr_maps_every_route_to_its_parent():
    route_path = FIXTURES / "whole-cut-crossing-routes.npz"
    packet_path = FIXTURES / "skin-adaptive-source-quadrature-032d.npz"
    assert sha256(route_path) == "6ba416a2f06261e2e002b0727b1e0497ce447ddca6db7f7466092111ef5a4009"
    assert sha256(packet_path) == "7f648262440539761f5cc6f14c9a2a7c5bf73c5cf75b310ec747872d83c0efb3"

    with np.load(route_path, allow_pickle=False) as routes, np.load(packet_path, allow_pickle=False) as packet:
        parent_rows = packet["source_parent_tet_rows"].astype(np.int64)
        global_rows = packet["global_source330_tet_rows"].astype(np.int64)
        offsets = packet["parent_offsets"].astype(np.int64)
        weights = packet["carrier_weights"]
        bary = packet["carrier_barycentric"]
        assert offsets.shape == (len(parent_rows) + 1,)
        assert offsets[0] == 0 and offsets[-1] == len(weights)
        assert np.all(np.diff(offsets) >= 0)
        point_parent = np.repeat(np.arange(len(parent_rows), dtype=np.int64), np.diff(offsets))
        assert point_parent.shape == weights.shape

        carrier = routes["packet_carrier_row"].astype(np.int64)
        assert len(carrier) == 300860
        assert np.all((carrier >= 0) & (carrier < len(weights)))
        assert np.array_equal(routes["virtual_parent_row"], carrier)
        owners = point_parent[carrier]
        assert np.array_equal(routes["source_local_parent_row"], parent_rows[owners])
        assert np.array_equal(routes["source_global330_tet_row"], global_rows[owners])
        assert np.array_equal(routes["source_barycentric"], bary[carrier])
        assert np.array_equal(routes["source_weight"], weights[carrier])

        pairs = np.rec.fromarrays([routes["group_id"], routes["child_global_row"]])
        assert len(np.unique(pairs)) == 300860
        reached = routes["reached_source_virtual_rows"].astype(np.int64)
        assert np.all((reached >= 0) & (reached < len(weights)))


def test_failed_attempts_are_explicit_in_recovery_receipt():
    append = read_json("whole-cut-route-skin-append-validation.json")
    recovery = append["prior_append_attempt_recovery"]
    assert recovery["existing_output_reused_without_reappend"] is True
    assert recovery["appended_suffix_replayed_byte_exactly"] is True
    assert recovery["prior_exception"].startswith("KeyError p1_skin")


def test_actual_out007_cut_is_retained_as_failed_not_admitted():
    result = read_json("actual-out-007-run-result.json")
    execution = read_json("out-007-execution.json")
    cut_manifest = read_json("out-007-full-cut-manifest.json")
    header = EVIDENCE / "out-007-full-cut-witness.header.txt"

    assert result["status"] == "INFEASIBLE_FULL_GRAPH_CUT"
    assert result["stage"] == "minimum_flow"
    assert result["full_graph_exhausted"] is True
    assert result["no_augmenting_path"] is True
    assert result["timed_out"] is False
    assert result["exit_code"] == 6
    assert result["minimum_demand_shortfall_kg"] == "0.019380173772159"
    assert result["cut_residual_child_rows"] == 60604
    assert result["cut_witness"]["status"] == "INFEASIBLE_FULL_GRAPH_CUT"
    assert result["cut_witness"]["compressed_bytes"] == 376396775
    assert result["cut_witness"]["large_compressed_witness_copied"] is False
    assert result["cut_witness"]["compressed_sha256"] == cut_manifest["witness_sha256"]
    assert result["cut_witness"]["header"]["sha256"] == sha256(header)
    assert result["run_pins"]["execution"]["sha256"] == sha256(EVIDENCE / "out-007-execution.json")
    assert execution["status"] == "DIAGNOSTIC_RUN_EXIT_NONZERO"
    assert result["mass_admission"] == "NOT_PASSED"
    assert result["physical_admission"] is False
    assert result["native_steps"] == 0
