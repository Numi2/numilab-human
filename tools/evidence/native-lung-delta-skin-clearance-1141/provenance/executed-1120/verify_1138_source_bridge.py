#!/usr/bin/env python3
"""Source-only D/lobe map bridge check for the selected 1138 composition."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import time
from pathlib import Path

E = Path("/Users/n/numi-human-resting-evidence-20261005")
HERE = Path(__file__).resolve().parent
OUT = HERE / "attempt-001-1138-source-map-bridge"
BASE_PATH = E / "native-lung-transformed-pose-audit-1079/audit_lung_interfaces_1042_pinned.py"
BASE_SHA = "791e2acdfdf917cd8554d83887f78682121a181216a07ff96c7ca2b985214345"
FEATURE_PATH = E / "native-lung-exact-classifier-index-1084/native_feature_index.py"
FEATURE_SHA = "2d9eab88a802a0223cad34f8b9d42066e55214d53dab5b7e0fc0f9b6aac9cd5e"
D_ADAPTER_PATH = E / "native-lung-d-lobe-boundary-parity-1086/native_feature_index_1085.py"
D_ADAPTER_SHA = "a982d97188fc8c207380afc2d3f8fe0e5934228cb816c2bea827115c84050928"
SOURCE_RULE_PATH = Path("/Users/n/numi-human-lung-contact-classifier-001/src/numilab_human/lung_contact_classification.py")
SOURCE_RULE_SHA = "fd95fa6cca92706faa9c7dcf88fbe2243d1a7d865b391d2f632cc0230cf42710"
CHILD_ADAPTER_PATH = E / "native-lung-audit-exact-delta-1120/successor_dmap_adapter_1120.py"
CHILD_ADAPTER_SHA = "8b20fd5b39fc8736d318d2be8cf379de2d92184a7a097ff3e8a427fd878bd085"
READER_PATH = E / "native-lung-audit-exact-delta-1120/audit_delta_1120.py"
READER_SHA = "f0ffe436fdf0698c970c0df4f94139c74bf808b729276d734a2f9ce6f4dd02ec"
COMPOSITION = E / "native-lung-free-apex-composition-1138-threeops-attempt1/composition-report.json"
COMPOSITION_SHA = "c1e90688d6eeb7a51207e70d59878b834009d8f54bf0d44df21512941e72d0c6"
CANDIDATE = E / "native-lung-free-apex-composition-1138-threeops-attempt1/final/resting-thorax.nhanatomy"
CANDIDATE_SHA = "c1fdc74087f7f28b7c75f57d3ed30eaf0586bbfcb0115007e324f1c176a6a424"
EXPECTED_COUNTS = {305: 19743, 306: 21600, 307: 5514, 308: 486, 309: 0}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def require(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha(path) != expected:
        raise ValueError(label + " hash mismatch: " + str(path))

def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    if OUT.exists():
        raise FileExistsError("refusing to overwrite prior bridge attempt: " + str(OUT))
    pins = ((BASE_PATH, BASE_SHA, "1042 base"), (FEATURE_PATH, FEATURE_SHA, "1084 feature"),
            (D_ADAPTER_PATH, D_ADAPTER_SHA, "1085 D adapter"),
            (SOURCE_RULE_PATH, SOURCE_RULE_SHA, "source contact rule"),
            (CHILD_ADAPTER_PATH, CHILD_ADAPTER_SHA, "successor D-map adapter"),
            (READER_PATH, READER_SHA, "delta reader"),
            (COMPOSITION, COMPOSITION_SHA, "1138 composition report"),
            (CANDIDATE, CANDIDATE_SHA, "1138 candidate NHA"))
    for path, digest, label in pins:
        require(path, digest, label)
    base = load(BASE_PATH, "source_bridge_base_1042")
    feature = load(FEATURE_PATH, "source_bridge_feature_1084")
    d_adapter = load(D_ADAPTER_PATH, "source_bridge_d_adapter_1085")
    source_rule = load(SOURCE_RULE_PATH, "source_bridge_contact_rule")
    d_adapter.install(base, feature_1084=feature, source_rule=source_rule)
    child = load(CHILD_ADAPTER_PATH, "source_bridge_child_adapter_1120")
    parser = base.load(base.PARSER, "source_bridge_nha_parser")
    if sha(base.PARSER) != base.PARSER_SHA:
        raise ValueError("pinned NHA parser changed")
    rows = parser.parse_payload(CANDIDATE)[1]
    start = time.monotonic()
    doc = child.load_v8_current_dmap(base=base, composition_report_path=COMPOSITION,
                                     final_nha_path=CANDIDATE, final_nha_sha=CANDIDATE_SHA,
                                     final_rows=rows)
    elapsed = time.monotonic() - start
    if doc["counts_by_lobe"] != EXPECTED_COUNTS or doc["zero_lobes"] != [309]:
        raise ValueError("unexpected D/lobe map coverage")
    validation = doc["registered_face_validation"]
    if (validation.get("checked_d_rows") != 47343 or validation.get("checked_lobe_rows") != 47343
            or validation.get("exact_face_index_triple_match") is not True
            or validation.get("exact_xyz_triangle_match") is not True):
        raise ValueError("registered face identity/geometry validation incomplete")
    composition_doc = json.loads(COMPOSITION.read_text())
    area_path = Path(composition_doc["outputs"]["final_area_derivation"]["path"])
    area_sha = composition_doc["outputs"]["final_area_derivation"]["sha256"]
    require(area_path, area_sha, "1138 effective-area derivation")
    OUT.mkdir()
    result = {
        "schema": "numi.human.lung-source-map-bridge-review.v1",
        "status": "PASS_source_map_bridge_only_not_native_geometry_acceptance",
        "source_composition": {"path": str(COMPOSITION), "sha256": COMPOSITION_SHA},
        "candidate_nha": {"path": str(CANDIDATE), "sha256": CANDIDATE_SHA},
        "current_d_map": {"path": doc["map_path"], "sha256": doc["map_sha256"],
                          "mapped_face_count": doc["map_count"], "counts_by_lobe": doc["counts_by_lobe"],
                          "explicit_zero_lobes": doc["zero_lobes"]},
        "validated_ancestry": doc["parent_binding"],
        "registered_face_checks": validation,
        "registered_area_m2": doc["registered_area_m2"],
        "effective_area_derivation": {"path": str(area_path), "sha256": area_sha},
        "source_inputs": {str(path): digest for path, digest, _ in pins},
        "parser": {"path": str(base.PARSER), "sha256": base.PARSER_SHA},
        "elapsed_seconds": elapsed,
        "scope_limit": "This is source and map lineage validation only. It does not establish transformed-pose collision clearance, native geometry acceptance, respiratory physiology, or endurance."
    }
    (OUT / "source-map-bridge-report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))

if __name__ == "__main__":
    main()
