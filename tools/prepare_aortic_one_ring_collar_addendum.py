from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
PARENT_PREFLIGHT = ROOT / "aortic7-connected-network-rebase-028-joint-quotient-gate-001" / "cold-preflight.json"
OUT = ROOT / "aortic7-source-one-ring-collar-addendum-001"
REPO = Path("/Users/n/numi-human-aortic-one-ring-20261016")
VERIFIER = Path("/Users/n/numi-human-anatomy-verifier-c73e6cc0")

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)

if OUT.exists():
    raise SystemExit(f"preserving existing immutable output: {OUT}")
if not PARENT_PREFLIGHT.is_file():
    raise SystemExit(f"missing pinned 028 preflight: {PARENT_PREFLIGHT}")
parent = json.loads(PARENT_PREFLIGHT.read_text())
bundle = parent["actual_invoked_capture_bundle"]
assert parent["028_candidate_nha_sha256"] == "5575f2f312b0f76837dea4daecf9df92c734461c9a45c61e34eb179811d010af"
assert parent["actual_invoked_source_nha_sha256"] == "e197a6a73ea1344a4bc540c7d9118ae290ff643ae96a54839b1de705fc3437b6"
assert sha(PARENT_PREFLIGHT) == "df49e1d0c9226c62373a46bd59ef5f98aacd1d881766ad330880de5effac763d"

# Load owner code from the isolated current worktree while keeping the shared
# verifier implementation pinned to the published c73e6cc0 checkout.
sys.path.insert(0, str(REPO / "src"))
import numilab_human

verifier_path = VERIFIER / "src" / "numilab_human" / "resting_anatomy_verify.py"
spec = importlib.util.spec_from_file_location(
    "numilab_human.resting_anatomy_verify", verifier_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load the pinned shared verifier")
verifier_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = verifier_module
numilab_human.resting_anatomy_verify = verifier_module
spec.loader.exec_module(verifier_module)

owner = importlib.import_module(
    "numilab_human.resting_brachiocephalic_3d_junction")
preparation = importlib.import_module(
    "numilab_human.resting_brachiocephalic_current_capture_network_preparation")
runner = importlib.import_module(
    "numilab_human.resting_brachiocephalic_current_capture_network_runner")

preflight, runtime = preparation.build_preflight(bundle, return_runtime_inputs=True)
assert preflight["actual_accepted_capture"]["invoked_anatomy_receipt_sha256"] == bundle["anatomy_receipt_sha256"]
assert preflight["actual_accepted_capture"]["pack_sha256"] == bundle["capture_pack_sha256"]
candidates, baselines, link_inputs, inventory = runner._bind_runtime_inputs(runtime)
edge = owner.AORTIC_ONE_RING_COLLAR_LINK
item = next(row for row in link_inputs
            if (row["bridge_key"], row["target_key"]) == edge)
declaration = owner.build_aortic_one_ring_collar_addendum(
    bridge_key=edge[0], target_key=edge[1],
    source_link_evidence=item["source_link_evidence"],
    source_rows=runtime["source_rows"], source_map=runtime["source_map"])
json.dumps(declaration, sort_keys=True, allow_nan=False)
report = {
    "schema": "numi.human.aortic-one-ring-source-collar-preparation.v1",
    "status": "SOURCE_ONLY_ONE_RING_DECLARATION_FROZEN_NO_CANDIDATE_PREDICATE",
    "parent_028_preflight_path": str(PARENT_PREFLIGHT.resolve()),
    "parent_028_preflight_sha256": sha(PARENT_PREFLIGHT),
    "candidate_nha_path": parent["028_source_candidate_path"],
    "candidate_nha_sha256": parent["028_candidate_nha_sha256"],
    "invoked_source_nha_sha256": parent["actual_invoked_source_nha_sha256"],
    "capture_pack_path": bundle["capture_pack_path"],
    "capture_pack_sha256": bundle["capture_pack_sha256"],
    "capture_receipt_path": bundle["capture_receipt_path"],
    "capture_receipt_sha256": bundle["capture_receipt_sha256"],
    "anatomy_receipt_path": bundle["anatomy_receipt_path"],
    "anatomy_receipt_sha256": bundle["anatomy_receipt_sha256"],
    "owner_code_path": str(Path(owner.__file__).resolve()),
    "owner_code_sha256": sha(Path(owner.__file__).resolve()),
    "preparation_code_path": str(Path(preparation.__file__).resolve()),
    "preparation_code_sha256": sha(Path(preparation.__file__).resolve()),
    "runner_code_path": str(Path(runner.__file__).resolve()),
    "runner_code_sha256": sha(Path(runner.__file__).resolve()),
    "shared_verifier_path": str(verifier_path.resolve()),
    "shared_verifier_sha256": sha(verifier_path),
    "source_link_input": {
        "typed_link": [list(edge[0]), list(edge[1])],
        "source_evidence_path": item["source_link_evidence"]["path"],
        "source_evidence_sha256": item["source_link_evidence"]["sha256"],
        "source_face_pair_ancestry_sha256": item["source_link_evidence"]["pairs_sha256"],
    },
    "source_one_ring_addendum": declaration,
    "candidate_geometry_predicate_run": False,
    "inside_both_or_connected_lumen_proven": False,
    "material_station_calibre_proven": False,
    "native_or_gpu_work_performed": False,
    "anatomical_admission": False,
}
OUT.mkdir(parents=True)
write_json(OUT / "source-one-ring-collar-addendum.json", report)
print(json.dumps({
    "status": report["status"],
    "output": str((OUT / "source-one-ring-collar-addendum.json").resolve()),
    "sha256": sha(OUT / "source-one-ring-collar-addendum.json"),
    "bridge_one_ring_faces": declaration["bounded_source_region"]["additional_bridge_one_ring_faces"],
    "additional_pair_count": len(declaration["bounded_source_region"]["additional_allowed_face_pairs"]),
}, sort_keys=True))
