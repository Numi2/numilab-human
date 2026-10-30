#!/usr/bin/env python3
"""Register the existing combined 217 node map against authenticated role bundle 143.

This is a source-only preparation route. It validates the inherited-plus-delta
geometry proof (713), then delegates node-map construction to the existing 217
source-005 owner. It does not run mass transport or physical/native execution.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BASE = H.parent
SOURCE_005 = H / "downstream-003-current-typed-228/native-domain-217-source-005/register_combined_volume_nodes_217.py"
SOURCE_005_SHA = "14452cc73ee20c50fad46ab369b65b799e80f5f00577cb6b21a8c51192ec676c"
SOURCE_008 = H / "downstream-004-same-material-current-reference-229/native-domain-217-source-008/register_combined_volume_nodes_217.py"
SOURCE_008_SHA = "2cd3e2ec2d9715b326f5d0c585527fa459d6e2231a18688b56f923b79aa84db9"
SOURCE_008_LABEL_EXPORTER = SOURCE_005.parent / "export_child_semantic_labels_217.py"
SOURCE_008_LABEL_EXPORTER_SHA = "9cf388fd7a9a18e72c478b9e5d1f6601884bdcf46be6dacb2d04e13fa2b66115"
BUNDLE = H / "registered-role-bundle-143/bundle-receipt.json"
BUNDLE_SHA = "f0a342f5ecc2c070ea4cb4bb8f7271e3750250b52e7e471d280ca382dc972898"
GEOMETRY_SCOPE = BASE / "native-launch-route-preparation-713-geometry-scope-derivative/geometry-scope-run-001/geometry-scope.json"
GEOMETRY_SCOPE_SHA = "35d6a46efac809166d2ef0ea6d5860391c78f2e2ecff2466ac8b2b794ed1ed34"
HANDOFF = H / "combined-child-handoff-142"
LABELS = HANDOFF / "semantic-labels-171-vectorized-full-export"
PARENT = BASE / "native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
POSES = BASE / "native-physical-parent-mass-330/native-source-identity-control-003/registered-skin-poses-100.json"
OWNER = "numi-human-prep-venv-20261005"
EXPECTED_CHAIN = {
    "010-self-and-interlayer-only": "810a2f07d589eb3646155c277840ce0e98455b965cfe1d935e01f36d7e29acbc",
    "014-all185-f32-f64": "48f5e8d76cb123f4ac890061163641331403ed342f1adf01821a0b8ff73c26bc",
    "complete-geometry-038": "af18d3e4d3f02284acc4a27740c5c9256464c17fb99ca443a7186716a836cdf7",
    "D-delta-128": "12457244fe94dc502b6c2c6cb1f505091b7684ca0cad883887f2273baaf32c23",
    "D-delta-137": "f9481aa1fead9ffa22e71ff55badc34ec2a12ce495b73f85e4b87bdeb61f0b4e",
}
EXPECTED_HANDOFF_SHA = "8f0daa820b2392309939db3bec35e6f7e1d7d65e9289a9a93f2f9a529573b49d"
EXPECTED_LABEL_RECEIPT_SHA = "25cbf2565d1e41a7be7cfb97762033f057cde18dac6923609bdae67be4a7cf79"
EXPECTED_ROUNDTRIP_SHA = "359e3cbbbe3499ccbe8a087c61a2dbe234eceab9a2060b77b14935cb23b89120"
EXPECTED_LABEL_ROWS = 1_596_903
MAX_PARENT_DISTANCE_M = 0.02
AMBIGUITY_TOLERANCE_M = 1e-8
CAPACITY_PROFILE = "full-reference-500k-nodes-2m-cells"
FACE_FIELDS = ("face_source_layer", "face_source_body", "face_source_bone",
               "face_source_face_id", "face_source_original_id",
               "face_source_identity0", "face_source_identity1")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def pin(path: Path, expected: str, context: str) -> None:
    require(path.is_file(), f"missing {context}: {path}")
    actual = sha(path)
    require(actual == expected, f"{context} SHA mismatch: expected {expected}, got {actual}")


def verify_nonadmitting(r: dict, context: str) -> None:
    require(r.get("physical_admission") is False, f"{context} claims physical admission")
    require(r.get("runtime", r.get("native_steps", 0)) == 0, f"{context} has runtime/native steps")


def role_map(bundle: dict) -> dict[str, dict]:
    roles = bundle.get("candidate_roles")
    require(isinstance(roles, dict) and set(roles) == {"O", "I", "D"}, "bundle 143 lacks exact O/I/D role coverage")
    return roles


def verify_geometry_scope(scope_path: Path) -> dict:
    pin(scope_path, GEOMETRY_SCOPE_SHA, "713 geometry-scope receipt")
    scope = load_json(scope_path)
    require(scope.get("schema") == "numi.human.h706.geometry-scope-receipt.v1", "unexpected 713 geometry-scope schema")
    require(scope.get("status") == "PASS_GEOMETRY_SCOPE_INHERITED_PLUS_EXACT_DELTAS", "713 geometry-scope status mismatch")
    verify_nonadmitting(scope, "713 geometry scope")
    require(scope.get("mass_mapping") in (None, "NOT_RUN") and scope.get("accepted_steps") == 0,
            "713 scope must remain source-only with mass/runtime not run")
    chain = scope.get("coverage_chain")
    require(isinstance(chain, list), "713 coverage chain missing")
    by_name = {x.get("name"): x for x in chain}
    require(set(by_name) == set(EXPECTED_CHAIN), "713 coverage chain membership changed")
    for name, expected_sha in EXPECTED_CHAIN.items():
        entry = by_name[name]
        require(entry.get("sha256") == expected_sha, f"713 {name} pin mismatch")
        pin(Path(entry["path"]), expected_sha, f"713 {name}")
    # Keep the known mixed-units report explicitly excluded from current geometry claims.
    require(by_name["010-self-and-interlayer-only"].get("all185_subsection") == "EXCLUDED_INVALID_MIXED_UNITS",
            "invalid mixed-units 010 report was not explicitly excluded")
    report014 = load_json(Path(by_name["014-all185-f32-f64"]["path"]))
    require(report014.get("status") == "PASS_WHOLE_CURRENT_OID_VS_ALL185_F32_F64" and
            report014.get("physical_admission") is False and report014.get("all185_total_pairs") == 0 and
            report014.get("all185_records", {}).get("count") == 938272,
            "014 inherited all185 evidence does not match the scope receipt")
    report038 = load_json(Path(by_name["complete-geometry-038"]["path"]))
    require(report038.get("status") == "PASS_FIXED_COORDINATE_GEOMETRY_DELTA" and
            report038.get("physical_admission") is False and
            report038.get("changed_vs_all185_both_domains", {}).get("count") == 0 and
            report038.get("changed_vs_complete_current_surfaces", {}).get("count") == 0,
            "038 complete-geometry delta did not pass its recorded exact gates")
    report128 = load_json(Path(by_name["D-delta-128"]["path"]))
    require(report128.get("status") == "PASS_EXACT_CURRENT_D_ONE_ULP_DELTA" and
            report128.get("physical_admission") is False and
            report128.get("changed_vs_all185_both_domains", {}).get("count") == 0 and
            report128.get("changed_vs_complete_current_surfaces", {}).get("count") == 0,
            "128 D delta did not pass its recorded exact gates")
    report137 = load_json(Path(by_name["D-delta-137"]["path"]))
    require(report137.get("status") == "PASS_EXACT_CURRENT_D_BOUNDARY_REFINEMENT_DELTA" and
            report137.get("physical_admission") is False and
            report137.get("changed_vs_all185_both_domains", {}).get("count") == 0 and
            report137.get("changed_vs_complete_current_surfaces", {}).get("count") == 0,
            "137 D delta did not pass its recorded exact gates")
    return {"path": str(scope_path), "sha256": GEOMETRY_SCOPE_SHA,
            "status": scope["status"], "coverage_names": list(EXPECTED_CHAIN),
            "all185_baseline_records": 938272, "exact_delta_checks": 3,
            "physical_admission": False, "mass_mapping": "NOT_RUN"}


def verify_sidecar_bundle(bundle_path: Path, geometry_scope_path: Path, registered_dir: Path,
                          pins: dict[str, tuple[Path, str]], maximum_parent_distance: float) -> dict:
    bundle_sha = sha(bundle_path)
    b = load_json(bundle_path)
    require(b.get("schema") == "numi.human.registered-role-bundle-143.v1" and
            b.get("status") == "PREPARED_OI088_D140_217_REGISTRATION_INPUTS_NO_ADMISSION",
            "unexpected/non-prepared 143 bundle")
    verify_nonadmitting(b, "143 bundle")
    require(b.get("mass_mapping") == "NOT_RUN", "143 bundle must not claim mass mapping")
    require(maximum_parent_distance == MAX_PARENT_DISTANCE_M, "historical parent support bound must remain 20 mm")
    require(set(pins) == {"O", "I", "D"}, "current candidate pins must cover exactly O/I/D")
    roles = role_map(b)
    require(set(b.get("files", {})) >= {
        "registered-O.csv", "registered-I.csv", "registered-D.csv",
        "registered-O-faces.csv", "registered-I-faces.csv", "registered-D-faces.csv",
        "node-parent-registration.csv", "donor-frames.csv", "source-domain-registry.json",
        "parent-tet-full-weights-O.csv", "parent-tet-full-weights-I.csv", "parent-tet-full-weights-D.csv"},
        "143 bundle is missing required sidecar pins")
    sidecar_sha = {}
    for name, entry in b["files"].items():
        p = bundle_path.parent / name
        pin(p, entry["sha256"], f"143 sidecar {name}")
        require(p.stat().st_size == entry.get("size_bytes"), f"143 sidecar size mismatch: {name}")
        sidecar_sha[name] = entry["sha256"]
    require(b.get("donor_frames_sha256") == sidecar_sha["donor-frames.csv"], "143 donor frame SHA mismatch")
    require(b.get("node_parent_registration_rows") == {"O": roles["O"]["vertices"],
            "I": roles["I"]["vertices"], "D": roles["D"]["vertices"]},
            "143 role row counts mismatch")

    # Authenticate the source receipts that produced the combined sidecar bundle.
    sources = b.get("source_receipts", {})
    require(set(sources) == {"OI088", "D137", "D140"}, "143 source receipt set changed")
    verified_source_receipts = {}
    for name, rec in sources.items():
        p = Path(rec["path"])
        pin(p, rec["sha256"], f"143 source receipt {name}")
        verified_source_receipts[name] = {"path": str(p), "sha256": rec["sha256"]}
    oi = load_json(Path(sources["OI088"]["path"]))
    d140 = load_json(Path(sources["D140"]["path"]))
    d137 = load_json(Path(sources["D137"]["path"]))
    require(oi.get("schema") == "registered-authoring-surface-sidecars-receipt.v1" and
            oi.get("status") == "reference_registration_ready_geometry_not_admitted",
            "OI088 owner receipt is not the expected source-only sidecar export")
    verify_nonadmitting(oi, "OI088")
    require(oi.get("limits", {}).get("maximumParentRegistrationDistanceM") == MAX_PARENT_DISTANCE_M,
            "OI088 historical registration limit mismatch")
    require(d140.get("schema") == "registered-authoring-surface-sidecars-receipt.v1" and
            d140.get("status") == "reference_registration_ready_geometry_not_admitted",
            "D140 owner receipt is not the expected source-only sidecar export")
    verify_nonadmitting(d140, "D140")
    require(d140.get("limits", {}).get("maximumParentRegistrationDistanceM") == MAX_PARENT_DISTANCE_M,
            "D140 historical registration limit mismatch")
    require(d140.get("sourceGeometryReceipt", {}).get("path") == sources["D137"]["path"] and
            d140.get("sourceGeometryReceipt", {}).get("sha256") == sources["D137"]["sha256"],
            "D140 did not pin the same D137 exact geometry delta")
    expected_roles = {"O": ("outerSurface", "1"), "I": ("innerSurface", "2"), "D": ("deepSurface", "3")}
    for role, (semantic_role, role_index) in expected_roles.items():
        spec = roles[role]
        candidate = Path(spec["candidate_path"]).resolve()
        expected_sha = spec["candidate_sha256"]
        p_pin, sha_pin = pins[role]
        require(candidate == p_pin.resolve() and expected_sha == sha_pin,
                f"explicit {role} CLI pin differs from the 143 bundle")
        pin(candidate, expected_sha, f"143 {role} current typed surface")
        oi_spec = oi.get("candidateSurfaces", {}).get(role)
        d_spec = d140.get("candidateSurfaces", {}).get(role)
        selected_spec = d_spec if role == "D" else oi_spec
        require(isinstance(selected_spec, dict) and Path(selected_spec.get("path", "")).resolve() == candidate and
                selected_spec.get("sha256") == expected_sha and selected_spec.get("points") == spec["vertices"] and
                selected_spec.get("faces") == spec["faces"], f"{role} owner candidate pin/count mismatch")
        # Check role-specific emitted sidecar pins against the actual source owner receipt.
        source_receipt = d140 if role == "D" else oi
        out_hash = source_receipt.get("outputSHA256", {})
        for suffix in (f"registered-{role}.csv", f"registered-{role}-faces.csv"):
            require(out_hash.get(suffix) == sidecar_sha[suffix], f"{role} output sidecar hash differs from its owner receipt")
        with np.load(candidate, allow_pickle=False) as z:
            for key in ("points_world_m", "faces", *FACE_FIELDS):
                require(key in z.files, f"{role} current typed candidate lacks {key}")
            points = np.asarray(z["points_world_m"])
            faces = np.asarray(z["faces"], dtype=np.int64)
            typed = {key: np.asarray(z[key], dtype=np.int64) for key in FACE_FIELDS}
        require(points.shape == (spec["vertices"], 3) and np.isfinite(points).all() and
                np.array_equal(points, points.astype(np.float32).astype(points.dtype)), f"{role} candidate point geometry invalid")
        require(faces.shape == (spec["faces"], 3) and np.all(faces >= 0) and np.all(faces < len(points)),
                f"{role} candidate face geometry invalid")
        # Validate the exact direct point-row identity and typed face sidecar in the registered bundle.
        node_path = bundle_path.parent / f"registered-{role}.csv"
        with node_path.open("r", encoding="utf-8", newline="") as stream:
            rows = csv.DictReader(stream)
            require(rows.fieldnames is not None and len(rows.fieldnames) > 0, f"{role} node CSV missing header")
            n = 0
            for n, row in enumerate(rows, 1):
                i = n - 1
                require(int(row["vertex"]) == i, f"{role} node row order mismatch at {i}")
                actual = np.asarray([float(row[f"w{a}"]) for a in "xyz"], dtype=np.float64)
                require(np.array_equal(actual, points[i].astype(np.float64)), f"{role} node sidecar coordinate mismatch at {i}")
                require(row["source_domain"] == "derived_registered_skin_authoring" and
                        row["source_mapping_kind"] == "vertex" and int(row["source_vertex_row"]) == i and
                        int(row["source_face_row"]) == -1 and
                        all(float(row[f"source_bary{k}"]) == 0.0 for k in range(3)) and
                        all(float(row[f"source_residual_{a}_m"]) == 0.0 for a in "xyz"),
                        f"{role} node sidecar does not express exact direct row identity at {i}")
            require(n == len(points), f"{role} node sidecar row count mismatch")
        face_path = bundle_path.parent / f"registered-{role}-faces.csv"
        with face_path.open("r", encoding="utf-8", newline="") as stream:
            rows = csv.DictReader(stream)
            require(rows.fieldnames is not None, f"{role} face CSV missing header")
            for n, row in enumerate(rows):
                require(int(row["face"]) == n and
                        [int(row[f"v{k}"]) for k in range(3)] == list(map(int, faces[n])),
                        f"{role} face connectivity mismatch at {n}")
                for key in FACE_FIELDS:
                    require(int(row[key]) == int(typed[key][n]), f"{role} typed face identity mismatch at {n}: {key}")
                if n % 100_000 == 0:
                    pass
            require(n + 1 == len(faces), f"{role} face sidecar row count mismatch")
        registry = load_json(bundle_path.parent / "source-domain-registry.json")
        registry_role = next((x for x in registry if str(x.get("roleIndex")) == role_index), None)
        require(registry_role is not None and registry_role.get("role") == semantic_role and
                registry_role.get("sourceSHA256") == expected_sha and
                registry_role.get("vertexCount") == spec["vertices"] and
                registry_role.get("faceCount") == spec["faces"], f"{role} source-domain registry mismatch")

    scope_auth = verify_geometry_scope(geometry_scope_path)
    scope = load_json(geometry_scope_path)
    # D137 is the last changed-face proof and directly names the final O/I/D candidate hashes.
    for role, spec in role_map(b).items():
        src = d137.get("sources", {}).get(role, {})
        require(Path(src.get("path", "")).resolve() == Path(spec["candidate_path"]).resolve() and
                src.get("sha256") == spec["candidate_sha256"], f"D137 delta does not pin final {role} candidate")
    # Tie the root geometry-scope receipt to the exact 142 mesh that labels 171 consumed.
    handoff_path = HANDOFF / "adapter-receipt.json"
    pin(handoff_path, EXPECTED_HANDOFF_SHA, "142 combined handoff")
    handoff = load_json(handoff_path)
    require(handoff.get("status") == "PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION" and
            handoff.get("mass_mapping") == "NOT_RUN" and handoff.get("physical_admission") is False,
            "142 child handoff must remain source-only")
    require(scope.get("combined_child_handoff", {}).get("receipt_path") == str(handoff_path) and
            scope.get("combined_child_handoff", {}).get("receipt_sha256") == EXPECTED_HANDOFF_SHA,
            "713 scope does not bind the 142 handoff")
    return {"schema": "numi.current-typed-217.role-bundle-143-geometry-scope-authorization.v1",
            "role_bundle_143": {"path": str(bundle_path), "sha256": bundle_sha},
            "geometry_scope_713": scope_auth,
            "source_receipts": verified_source_receipts,
            "roles": {r: {"path": roles[r]["candidate_path"], "sha256": roles[r]["candidate_sha256"],
                           "vertices": roles[r]["vertices"], "faces": roles[r]["faces"]} for r in ("O", "I", "D")},
            "sidecar_files": sidecar_sha,
            "oi088_complete_oid_flag": oi.get("completeOID"),
            "oi088_complete_oid_note": "false in the original receipt; current 038 plus D128/137 exact evidence is consumed through 713, no boolean was rewritten",
            "historical_parent_support_maximum_distance_m": MAX_PARENT_DISTANCE_M,
            "current_operational_map": "separate; created only from current child self tets by the 433 adapter",
            "source330_to_original_source_bound_changed": False,
            "conservative_mass_transport_bound_changed": False,
            "physical_admission": False, "mass_mapping": "NOT_RUN", "runtime": 0}


def verify_label_handoff() -> dict:
    handoff = load_json(HANDOFF / "adapter-receipt.json")
    pin(HANDOFF / "adapter-receipt.json", EXPECTED_HANDOFF_SHA, "142 combined handoff")
    require(handoff.get("combined_skin_fat_child_mesh", {}).get("status") == "COMBINED_POSITIVE_F32_CHILD_MESH_NOT_PHYSICAL_ADMISSION" and
            handoff.get("combined_skin_fat_child_mesh", {}).get("positive_float32_cells") == EXPECTED_LABEL_ROWS and
            handoff.get("mass_mapping") == "NOT_RUN" and handoff.get("physical_admission") is False,
            "142 combined child count/status mismatch")
    receipt_path = LABELS / "receipt.json"
    pin(receipt_path, EXPECTED_LABEL_RECEIPT_SHA, "171 P1 label export receipt")
    r = load_json(receipt_path)
    require(r.get("full_child_domain_export") is True and r.get("counts", {}).get("exported_child_rows") == EXPECTED_LABEL_ROWS and
            r.get("mass_transport") == "NOT_RUN" and r.get("physical_admission") is False and
            r.get("native_steps") == 0, "171 P1 export is not a full-domain source-only export")
    outputs = r.get("outputs", {})
    expected_files = {
        "labels": (LABELS / "child.labels.csv", "27b7620e328578b6521f341cdda9962412134bf6e8a826e7bc805c0063617685"),
        "parent_row_map": (LABELS / "child.labels.csv.source.csv", "222e9964d186e57a346bdf06a4226e1ea680be39b24f35e499177070374ce5de"),
        "p1_anchor_csr": (LABELS / "p1-anchor.npz", "4fe982548d0cfc4662e29f244b9f616daa21b39c05b5e4a5ef81f0ac6a80d985"),
        "p1_anchor_projection_csv": (LABELS / "p1-anchor-projection.csv", "b0b09a9bbd46b27a6763b82fa96c87c16e5c9a0649bcece769bfd3f4a5737870")}
    for name, (path, expected_sha) in expected_files.items():
        pin(path, expected_sha, f"171 {name}")
        spec = outputs.get(name, {})
        require(Path(spec.get("path", "")).resolve() == path.resolve() and spec.get("sha256") == expected_sha,
                f"171 receipt output pin mismatch for {name}")
    roundtrip_path = LABELS / "roundtrip-full-002.json"
    pin(roundtrip_path, EXPECTED_ROUNDTRIP_SHA, "full 671/433 roundtrip")
    q = load_json(roundtrip_path)
    require(q.get("status") == "FULL_DOMAIN_671_CENTROID_AND_433_P1_VALIDATORS_PASS_NOT_PHYSICAL_ADMISSION" and
            q.get("full_child_domain_rows") == EXPECTED_LABEL_ROWS and
            q.get("671_centroid_witness_rows_validated") == EXPECTED_LABEL_ROWS and
            q.get("433_p1_anchor_csr_rows_validated") == EXPECTED_LABEL_ROWS and
            q.get("mass_transport") == "NOT_RUN" and q.get("physical_admission") is False,
            "full 671/433 roundtrip is incomplete or non-source-only")
    return {"label_export_receipt": {"path": str(receipt_path), "sha256": EXPECTED_LABEL_RECEIPT_SHA},
            "full_roundtrip": {"path": str(roundtrip_path), "sha256": EXPECTED_ROUNDTRIP_SHA,
                               "rows": EXPECTED_LABEL_ROWS},
            "labels": expected_files["labels"][1], "parent_row_map": expected_files["parent_row_map"][1],
            "p1_anchor_csr": expected_files["p1_anchor_csr"][1],
            "p1_anchor_projection_csv": expected_files["p1_anchor_projection_csv"][1],
            "mass_transport": "NOT_RUN", "physical_admission": False}


def validate_bundle_for_owner(context: dict, directory: Path, role_pins: dict,
                              maximum_parent_distance: float) -> dict:
    require(maximum_parent_distance == MAX_PARENT_DISTANCE_M, "node registrar parent bound differs from bundle")
    require(set(role_pins) == {"O", "I", "D"}, "owner role pins are incomplete")
    for role, (path, expected) in role_pins.items():
        spec = context["bundle"]["candidate_roles"][role]
        require(Path(spec["candidate_path"]).resolve() == Path(path).resolve() and
                spec["candidate_sha256"] == expected, f"owner {role} candidate differs from 143")
    # The complete 143+713 composition was verified before dispatch; keep it in the returned owner receipt.
    return context["authorization"]


def build_bridge(bridge: Path, bundle_path: Path, geometry_scope_path: Path, bundle: dict,
                 authorization: dict) -> None:
    require(not bridge.exists(), f"refusing to overwrite existing bridge directory: {bridge}")
    bridge.mkdir(parents=True)
    for name in bundle["files"]:
        source = bundle_path.parent / name
        (bridge / name).symlink_to(source)
    (bridge / "bundle-receipt.json").symlink_to(bundle_path)
    (bridge / "geometry-scope.json").symlink_to(geometry_scope_path)
    bridge_receipt = {"schema": "numi.current-typed-217.role-bundle-143-geometry-scope-bridge.v1",
                      "role_bundle": authorization["role_bundle_143"],
                      "geometry_scope": authorization["geometry_scope_713"],
                      "authorization": authorization,
                      "physical_admission": False, "mass_mapping": "NOT_RUN", "runtime": 0}
    (bridge / "receipt.json").write_text(json.dumps(bridge_receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_owner(path: Path):
    path = path.resolve()
    if path == SOURCE_005.resolve():
        expected = SOURCE_005_SHA
        module_name = "registered_node_owner_217_source005"
    elif path == SOURCE_008.resolve():
        expected = SOURCE_008_SHA
        module_name = "registered_node_owner_217_source008"
        require(sha(SOURCE_005) == SOURCE_005_SHA, "217 source-005 parent registrar SHA mismatch")
    else:
        raise ValueError("unapproved 217 registrar path")
    require(sha(path) == expected, f"217 registrar SHA mismatch: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "cannot import pinned 217 source-005 registrar")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



POSE_MATRIX_F32_TOLERANCE = 1.0e-6
POSE_ORIGIN_F32_TOLERANCE_M = 0.0


def _matrix_from_serialized_pose(row: dict, phase: str, body_index: int, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    pose = row.get(phase, {})
    origin64 = np.asarray(pose.get("position_world_m"), dtype=np.float64)
    quat = np.asarray(pose.get("orientation_world_xyzw"), dtype=np.float64)
    serialized = np.asarray(pose.get("local_to_world_row_major_f32"), dtype=np.float32)
    if origin64.shape != (3,) or quat.shape != (4,) or serialized.shape != (9,):
        raise ValueError(f"all_bodies {phase} pose {body_index} lacks native F32 transform fields")
    if not np.isfinite(origin64).all() or not np.isfinite(quat).all() or not np.isfinite(serialized).all():
        raise ValueError(f"all_bodies {phase} pose {body_index} has nonfinite transform data")
    qnorm = float(np.linalg.norm(quat))
    if not np.isfinite(qnorm) or abs(qnorm - 1.0) > 1.0e-6:
        raise ValueError(f"all_bodies {phase} quaternion {body_index} is not normalized")
    rotation = serialized.astype(np.float64).reshape(3, 3)
    quat_rotation = Rotation.from_quat(quat).as_matrix()
    quaternion_delta = float(np.max(np.abs(rotation - quat_rotation)))
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if orthogonality_error > POSE_MATRIX_F32_TOLERANCE or determinant_error > POSE_MATRIX_F32_TOLERANCE:
        raise ValueError(f"all_bodies {phase} serialized F32 matrix is not a proper rotation at body {body_index}")
    if quaternion_delta > POSE_MATRIX_F32_TOLERANCE:
        raise ValueError(f"all_bodies {phase} serialized matrix/quaternion mismatch at body {body_index}: {quaternion_delta:.9g}")
    origin = origin64.astype(np.float32).astype(np.float64)
    stats["max_quaternion_matrix_delta"] = max(stats.get("max_quaternion_matrix_delta", 0.0), quaternion_delta)
    stats["max_orthogonality_error"] = max(stats.get("max_orthogonality_error", 0.0), orthogonality_error)
    stats["max_determinant_error"] = max(stats.get("max_determinant_error", 0.0), determinant_error)
    return origin, rotation


def read_poses_from_serialized_f32(owner, path: Path, frame_path: Path, stats: dict):
    """Use the authenticated native F32 matrices, not quat-reconstructed matrices."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("all_bodies")
    if not isinstance(raw, list) or len(raw) != 157:
        raise ValueError("expected exact native all_bodies 157-pose table")
    poses = []
    for i, row in enumerate(raw):
        if owner.u(row.get("body_index"), f"all_bodies body {i}") != i:
            raise ValueError("all_bodies is not exact body-index order")
        q = {}
        for phase in ("current", "rest"):
            origin, rotation = _matrix_from_serialized_pose(row, phase, i, stats)
            q[phase + "_origin"] = origin
            q[phase + "_R"] = rotation
        q["binding"] = owner.INVALID
        poses.append(q)
    bindings = {}
    used = set()
    frame_rows = owner.csvread(frame_path, owner.FRAME_HEADER)
    for i, r in enumerate(frame_rows):
        binding = owner.u(r["binding"], f"registered binding {i}")
        donor = owner.u(r["donor"], f"registered donor {i}")
        if donor >= 157 or donor in bindings or binding in used:
            raise ValueError("duplicate registered donor or binding")
        pose = poses[donor]
        bindings[donor] = binding
        used.add(binding)
        for phase in ("current", "rest"):
            origin = np.asarray([owner.f(r[f"{phase}_{a}"], f"{phase} origin {donor}") for a in "xyz"])
            rotation = np.asarray([owner.f(r[f"{phase}_R{i}{j}"], f"{phase} rotation {donor}")
                                   for i in range(3) for j in range(3)]).reshape(3, 3)
            if not np.array_equal(origin, pose[phase + "_origin"]):
                raise ValueError(f"registered donor frame {donor} differs from exact serialized all_bodies {phase} origin")
            if not np.array_equal(rotation, pose[phase + "_R"]):
                raise ValueError(f"registered donor frame {donor} differs from exact serialized all_bodies {phase} matrix")
        pose["binding"] = binding
    stats["schema"] = "numi.native-pose-serialized-f32-frame-validation.v1"
    stats["all_bodies_rows"] = len(raw)
    stats["registered_frame_rows"] = len(frame_rows)
    stats["frame_rows_exact"] = True
    stats["quaternion_consistency_tolerance"] = POSE_MATRIX_F32_TOLERANCE
    stats["pose_origin_source"] = "all_bodies.position_world_m rounded to exact F32 runtime representation"
    stats["pose_matrix_source"] = "all_bodies.local_to_world_row_major_f32"
    return poses, bindings


def normalize_surface_frame_matrices(surfaces: dict, poses: list, stats: dict) -> None:
    """Check sidecar quaternion-derived matrices against native F32 runtime matrices."""
    maximum = 0.0
    checked = 0
    for role, rows in surfaces.items():
        for surface in rows:
            donor = int(surface["donor"])
            expected = poses[donor]["current_R"].T
            actual = np.asarray(surface["matrix"], dtype=np.float64)
            if actual.shape != (3, 3) or not np.isfinite(actual).all():
                raise ValueError(f"{role} surface row has invalid pose matrix")
            delta = float(np.max(np.abs(actual - expected)))
            if delta > POSE_MATRIX_F32_TOLERANCE:
                raise ValueError(f"{role} surface row donor {donor} matrix exceeds F32 pose consistency tolerance: {delta:.9g}")
            maximum = max(maximum, delta)
            checked += 1
            # This is an in-memory validation normalization only. The source CSV and
            # point coordinates remain unchanged; build_rows then checks local→world.
            surface["matrix"] = expected
    stats["surface_matrix_rows_checked"] = checked
    stats["max_surface_matrix_delta"] = maximum
    stats["surface_matrix_tolerance"] = POSE_MATRIX_F32_TOLERANCE



P1_ANCHOR_CSR_SHA256 = "4fe982548d0cfc4662e29f244b9f616daa21b39c05b5e4a5ef81f0ac6a80d985"
P1_ANCHOR_SCHEMA = "numi.native-domain-217.p1-supported-reference-anchor-csr.v1"


def expand_child_with_p1_semantic_union(owner, child: dict, p1_path: Path,
                                        expected_sha256: str = P1_ANCHOR_CSR_SHA256):
    """Supply authenticated per-cell P1 semantic unions to the existing incidence check.

    The input scalar child labels remain unchanged and are checked against the
    selected P1 anchor row. The adapter only expands the in-memory incident tuple
    view consumed by source-005; no cell labels, node maps, or geometry are rewritten.
    """
    p1_path = Path(p1_path).resolve()
    if not p1_path.is_file() or sha(p1_path) != expected_sha256:
        raise ValueError("P1 semantic union CSR pin mismatch")
    with np.load(p1_path, allow_pickle=False) as z:
        schema = str(np.asarray(z["schema"]).item())
        cell_rows = np.asarray(z["child_tet_row"], dtype=np.int64)
        support_material = np.asarray(z["p1_support_role_material"], dtype=np.int64)
        support_cell_rows = np.asarray(z["p1_support_cell_row"], dtype=np.int64)
        offsets = np.asarray(z["p1_semantic_tuple_offsets"], dtype=np.int64)
        semantic = np.asarray(z["p1_semantic_tuples"], dtype=np.uint32)
        selected = np.asarray(z["reference_anchor_selected_semantic_tuple"], dtype=np.uint32)
        primary = np.asarray(z["p1_primary_group_tuple"], dtype=np.int64)
    count = len(child["tets"])
    if schema != P1_ANCHOR_SCHEMA:
        raise ValueError("unexpected P1 semantic union schema")
    if (cell_rows.shape != (count,) or not np.array_equal(cell_rows, np.arange(count, dtype=np.int64)) or
        support_cell_rows.shape != (count,) or support_material.shape != (count,) or
        offsets.shape != (count + 1,) or offsets[0] != 0 or
        offsets[-1] != len(semantic) or np.any(offsets[1:] < offsets[:-1]) or
        semantic.ndim != 2 or semantic.shape[1] != 4 or selected.shape != (count, 4) or
        primary.shape != (count, 3)):
        raise ValueError("malformed P1 semantic union row/CSR domains")
    support_rows_by_material = {}
    if not set(int(x) for x in np.unique(support_material)).issubset({15, 16}):
        raise ValueError("P1 support contains an unsupported material domain")
    for material in np.unique(support_material):
        role_rows = np.flatnonzero(support_material == material)
        if not np.array_equal(support_cell_rows[role_rows], np.arange(len(role_rows), dtype=np.int64)):
            raise ValueError(f"P1 role-local cell row domain is malformed for material {int(material)}")
        support_rows_by_material[str(int(material))] = int(len(role_rows))
    labels = np.asarray(child["labels"], dtype=np.uint32)
    materials = np.asarray(child["materials"], dtype=np.int64)
    if labels.shape != (count, 4) or materials.shape != (count,):
        raise ValueError("source-005 child label arrays do not match P1 child cell domain")
    if not np.array_equal(labels, selected):
        bad = int(np.flatnonzero(np.any(labels != selected, axis=1))[0])
        raise ValueError(f"scalar child label differs from authenticated selected P1 anchor at cell {bad}")
    if not np.array_equal(support_material, materials) or not np.array_equal(selected[:, :3], primary):
        raise ValueError("P1 material or primary semantic tuple differs from current child scalar domain")
    widths = offsets[1:] - offsets[:-1]
    if np.any(widths <= 0):
        raise ValueError("P1 semantic tuple union contains an empty child cell")
    expanded_materials = np.repeat(support_material.astype(np.uint32), widths)
    if not np.array_equal(semantic[:, 0], expanded_materials):
        raise ValueError("P1 semantic union crosses a child material boundary")
    # Prove the selected scalar anchor is present in the complete four-field union.
    cell_index = np.repeat(np.arange(count, dtype=np.int64), widths)
    matches = np.all(semantic == selected[cell_index], axis=1)
    selected_present = np.zeros(count, dtype=np.bool_)
    selected_present[cell_index[matches]] = True
    if not np.all(selected_present):
        bad = int(np.flatnonzero(~selected_present)[0])
        raise ValueError(f"selected P1 anchor is absent from the complete semantic union at cell {bad}")
    # Cache the small vocabulary of typed tuples and stream repeated cell rows to the
    # source-005 owner, avoiding a second materialized full-cell label table.
    unique_semantics = np.unique(semantic, axis=0)
    tuple_cache = {tuple(int(v) for v in row): tuple(int(v) for v in row) for row in unique_semantics}
    base_tets = child["tets"]
    def repeated_tets():
        for cell, tet in enumerate(base_tets):
            for _ in range(int(widths[cell])):
                yield tet
    def repeated_labels():
        for cell in range(count):
            for row in range(int(offsets[cell]), int(offsets[cell + 1])):
                yield tuple_cache[tuple(int(v) for v in semantic[row])]
    expanded = dict(child)
    expanded["tets"] = repeated_tets()
    expanded["labels"] = repeated_labels()
    stats = {
        "schema": "numi.217.current-child-p1-semantic-incidence-union.v1",
        "status": "PASS_AUTHENTICATED_COMPLETE_P1_UNION",
        "p1_csr_path": str(p1_path),
        "p1_csr_sha256": expected_sha256,
        "p1_schema": schema,
        "child_cells": count,
        "p1_semantic_tuple_rows": int(len(semantic)),
        "support_rows_by_material": support_rows_by_material,
        "unique_typed_semantic_tuples": int(len(unique_semantics)),
        "selected_scalar_anchor_rows_exact": count,
        "selected_anchor_rows_present_in_union": count,
        "empty_cell_unions": 0,
        "cross_material_union_rows": 0,
        "only_in_memory_incidence_expansion": True,
        "source_scalar_labels_unchanged": True,
        "geometry_changed": False,
        "mass_mapping": "NOT_RUN",
        "physical_admission": False,
        "runtime": 0,
    }
    return expanded, stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-map", type=Path, required=True)
    p.add_argument("--node", type=Path, required=True)
    p.add_argument("--ele", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--parent-base", type=Path, required=True)
    p.add_argument("--bundle-receipt", type=Path, default=BUNDLE)
    p.add_argument("--geometry-scope", type=Path, default=GEOMETRY_SCOPE)
    p.add_argument("--registered-source-dir", type=Path, default=BUNDLE.parent)
    p.add_argument("--poses", type=Path, default=POSES)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--maximum-parent-distance-m", type=float, default=MAX_PARENT_DISTANCE_M)
    p.add_argument("--ambiguity-tolerance-m", type=float, default=AMBIGUITY_TOLERANCE_M)
    p.add_argument("--capacity-profile", choices=(CAPACITY_PROFILE,), default=CAPACITY_PROFILE)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    try:
        require(args.maximum_parent_distance_m == MAX_PARENT_DISTANCE_M and
                args.ambiguity_tolerance_m == AMBIGUITY_TOLERANCE_M,
                "explicit reference bounds differ from the authenticated current route")
        scope_auth = verify_geometry_scope(args.geometry_scope)
        label_auth = verify_label_handoff()
        require(args.parent_base.resolve() == PARENT.resolve(), "parent base differs from source330 current packet")
        require(args.poses.resolve() == POSES.resolve(), "pose packet differs from the pinned source330 86-body pose")
        pin(args.bundle_receipt, BUNDLE_SHA, "143 role bundle receipt")
        bundle_sha = BUNDLE_SHA
        bundle = load_json(args.bundle_receipt)
        parent_spec = bundle.get("source330_parent_current", {})
        for suffix, key in (("", "massSHA256"), (".nodes.csv", "nodeSHA256"),
                            (".source.csv", "sourceSHA256"), (".tets.csv", "tetsSHA256"),
                            (".labels.csv", "labelsSHA256")):
            path = Path(str(args.parent_base.resolve()) + suffix)
            pin(path, parent_spec[key], f"330 parent {suffix or 'mass'}")
        require(bundle.get("source330_full86_weights", {}).get("sha256") == sha(Path(str(args.parent_base.resolve()) + ".weights.csv")),
                "330 full86 weight owner does not match role bundle")
        require(bundle.get("pose_snapshot", {}).get("sha256") == sha(args.poses), "330 pose snapshot SHA mismatch")
        authorization = verify_sidecar_bundle(args.bundle_receipt.resolve(), args.geometry_scope.resolve(),
                                              args.registered_source_dir.resolve(),
                                              {role: (Path(v["candidate_path"]), v["candidate_sha256"])
                                               for role, v in bundle["candidate_roles"].items()},
                                              args.maximum_parent_distance_m)
        context = {"bundle": bundle, "authorization": authorization}
        if args.preflight_only:
            print(json.dumps({"status": "PASS_143_713_217_SOURCE_PREFLIGHT_ONLY_NOT_REGISTRATION",
                              "authorization": authorization, "label_inputs": label_auth,
                              "physical_admission": False, "mass_mapping": "NOT_RUN", "runtime": 0},
                             sort_keys=True, separators=(",", ":")))
            return 0
        require(not args.output_dir.exists(), f"refusing to overwrite output directory {args.output_dir}")
        args.output_dir.mkdir(parents=True)
        bridge = args.output_dir / "registered-sidecars-143-713-bridge"
        build_bridge(bridge, args.bundle_receipt.resolve(), args.geometry_scope.resolve(), bundle, authorization)
        output = args.output_dir / "registered-node-map.csv"
        receipt_path = args.output_dir / "registration-receipt.json"
        pin(SOURCE_008_LABEL_EXPORTER, SOURCE_008_LABEL_EXPORTER_SHA, "source-005 label-owner dependency")
        owner = load_owner(SOURCE_008)
        owner.configure_capacity_profile(args.capacity_profile)
        pose_validation = {}
        p1_incidence_proof = {}
        owner.read_poses = lambda path, frame_path: read_poses_from_serialized_f32(owner, path, frame_path, pose_validation)
        original_build_rows = owner.build_rows
        def build_rows_with_serialized_pose_checks(maprows, child, surfaces, registrations, parent, owner_module, poses, maxdist, tol):
            normalize_surface_frame_matrices(surfaces, poses, pose_validation)
            child_with_p1, p1_stats = expand_child_with_p1_semantic_union(
                owner_module, child, LABELS / "p1-anchor.npz", P1_ANCHOR_CSR_SHA256)
            p1_incidence_proof.update(p1_stats)
            return original_build_rows(maprows, child_with_p1, surfaces, registrations, parent, owner_module, poses, maxdist, tol)
        owner.build_rows = build_rows_with_serialized_pose_checks
        owner.validate_current_typed_sidecar_bundle = lambda directory, role_pins, maxdist: validate_bundle_for_owner(context, directory, role_pins, maxdist)
        pins = [[role, bundle["candidate_roles"][role]["candidate_path"],
                 bundle["candidate_roles"][role]["candidate_sha256"]] for role in ("O", "I", "D")]
        owner_args = SimpleNamespace(
            maximum_parent_distance_m=args.maximum_parent_distance_m,
            ambiguity_tolerance_m=args.ambiguity_tolerance_m,
            node=args.node.resolve(), ele=args.ele.resolve(), labels=args.labels.resolve(),
            node_map=args.node_map.resolve(), parent_base=args.parent_base.resolve(),
            registered_dir=bridge.resolve(), poses=args.poses.resolve(), output=output,
            receipt=None, current_typed_sidecars=True, current_typed_role=pins)
        owner_receipt = owner.emit(owner_args)
        owner_receipt["inputs"].pop("current_typed_228_receipt", None)
        owner_receipt["inputs"]["role_bundle_143_receipt"] = {"path": str(args.bundle_receipt.resolve()), "sha256": bundle_sha}
        owner_receipt["inputs"]["geometry_scope_713"] = {"path": str(args.geometry_scope.resolve()), "sha256": GEOMETRY_SCOPE_SHA}
        owner_receipt["inputs"]["full_P1_label_export_receipt"] = {"path": str(LABELS / "receipt.json"), "sha256": EXPECTED_LABEL_RECEIPT_SHA}
        owner_receipt["inputs"]["full_P1_671_433_roundtrip"] = {"path": str(LABELS / "roundtrip-full-002.json"), "sha256": EXPECTED_ROUNDTRIP_SHA}
        owner_receipt["current_typed_sidecars"] = authorization
        owner_receipt["schema"] = "numi.current-typed-217.registered-node-map-143-713.v1"
        owner_receipt["status"] = "REGISTERED_NODE_MAP_PREPARED_SOURCE_ONLY_NOT_PHYSICAL_ADMISSION"
        owner_receipt["mass_mapping"] = "NOT_RUN"
        owner_receipt["physical_admission"] = False
        owner_receipt["runtime"] = 0
        owner_receipt["accepted_steps"] = 0
        owner_receipt["historical_parent_registration_maximum_distance_m"] = MAX_PARENT_DISTANCE_M
        owner_receipt["ambiguity_tolerance_m"] = AMBIGUITY_TOLERANCE_M
        owner_receipt["p1_label_route"] = label_auth
        owner_receipt["serialized_pose_matrix_validation"] = pose_validation
        owner_receipt["authenticated_p1_semantic_incidence_union"] = p1_incidence_proof
        owner_receipt["pose_frame_owner_source"] = {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())}
        owner_receipt["owner_source"] = {"path": str(SOURCE_008), "sha256": SOURCE_008_SHA}
        owner_receipt["owner_source_parent"] = {"path": str(SOURCE_005), "sha256": SOURCE_005_SHA}
        owner_receipt["owner_source_dependencies"] = {"source005_label_exporter": {"path": str(SOURCE_008_LABEL_EXPORTER), "sha256": SOURCE_008_LABEL_EXPORTER_SHA}}
        owner_receipt["output"] = {"path": str(output.resolve()), "sha256": sha(output)}
        receipt_path.write_text(json.dumps(owner_receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(owner_receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, ValueError, KeyError, csv.Error, json.JSONDecodeError) as exc:
        print(f"FAIL current 143/713 registration preparation: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
