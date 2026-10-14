#!/usr/bin/env python3
"""Prepare the scoped CSF wall repair in the existing integrated resting receipt."""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/Users/n/numi-human-final-integration-publish-1397")
D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
HEAD = D / "pons-fourth-ventricle-reconstruction-001"
BASE = D / "forearm492-head016-vessel-seam-integrated-source-003/resting-anatomy-receipt.json"
CANDIDATE = HEAD / "cord-local-clearance-candidate-compile-001/resting-thorax-candidate.nhanatomy"
COMPILE = HEAD / "cord-local-clearance-candidate-compile-001/compile-report.json"
MOTION = HEAD / "cord-local-clearance-motion-006/report.json"
OUT = D / "forearm492-head-cord-vessel-seam-integrated-source-004"
PINS = {
    BASE: "b2aa336c783b5fe8c75a89f3a9b27c5922d58dd71d9f230e5e57b021d61437f6",
    CANDIDATE: "9fb3ffb10e458589b2c69779bd1336bfc4dc8e704079a1e68e4b22a2b5b7ba09",
    COMPILE: "efc7efef96eba68da58cc8f8c7cc94ede25255a1a25a579f4ac11038a1f040c8",
    MOTION: "8d85e0346301924e43feee94ee030550f0cce149ba27e4b58208ecf5cb68bede",
    ROOT / "src/numilab_human/resting_common_field_passive_rebind.py": "d8566e09d27c1b2827f03d10d8e4786bce8ececd91c709646154ea47e97e0221",
    ROOT / "src/numilab_human/resting_brachiocephalic_common_field.py": "2f903282334baa57fbc1f620e42b112cd3d8027ff8fb1e088074594105506940",
}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok, message):
    if not ok:
        raise ValueError(message)
def main():
    require(not OUT.exists(), "preserve previous preparation")
    require(all(sha(path) == digest for path, digest in PINS.items()), "pinned input changed")
    sys.path.insert(0, str(ROOT / "src"))
    from numilab_human.resting_common_field_passive_rebind import rebind_passive_rows
    from numilab_human.resting_brachiocephalic_common_field import _resolve_sidecar
    from numilab_human.resting_run import _validate_parent_partitions
    base = json.loads(BASE.read_text())
    compiled = json.loads(COMPILE.read_text())
    motion = json.loads(MOTION.read_text())
    require(compiled["abi5_encode_proof"]["unmodified_rows_byte_exact"] == 524
            and compiled["floor_cap_vertices_unchanged"]
            and compiled["ordered_faces_unchanged"], "compiled scope changed")
    captures = motion["captures"]
    require([row["step"] for row in captures] == [0,3807,4767,5023,5599,6207,7423,8000,14975],
            "retained pose coverage changed")
    require(all(row["pair"]["triangle_pair_count"] == 0 and row["passed"] for row in captures),
            "scoped retained pose gate failed")
    raw = Path(base["payload"]["path"]).read_bytes()
    require(sha(base["payload"]["path"]) == base["payload"]["sha256"], "actual base changed")
    target = OUT / "resting-thorax.nhanatomy"
    result, audit = rebind_passive_rows(base, raw, CANDIDATE.read_bytes(),
                                      changed_ids=(323,), output_path=str(target))
    _validate_parent_partitions(result["passive_tissue_parent_bindings"])
    before_parents = copy.deepcopy(base["passive_tissue_parent_bindings"])
    after_parents = copy.deepcopy(result["passive_tissue_parent_bindings"])
    before_parents.pop("anatomy_payload_sha256")
    after_parents.pop("anatomy_payload_sha256")
    require(before_parents == after_parents, "parent ownership or maps changed")
    before_functions = copy.deepcopy(base["functional_bindings"])
    after_functions = copy.deepcopy(result["functional_bindings"])
    before_functions.pop("anatomy_payload_sha256")
    after_functions.pop("anatomy_payload_sha256")
    require(before_functions == after_functions, "functional owner changed")
    require(result["mass_geometry_accounting"] == base["mass_geometry_accounting"],
            "physical mass accounting changed")
    common = result["provenance"]["cardiac_geometry_binding"]["common_field"]
    sidecars = {key: _resolve_sidecar(BASE, BASE, common[key])
                for key in ("map", "polynomials", "domain_boxes")}
    require(len({p.name for p in sidecars.values()}) == 3, "sidecar name collision")
    report = {
        "status": "SOURCE_PREPARATION_ONLY_NOT_ADMISSION",
        "inputs": {str(path): digest for path, digest in PINS.items()},
        "passive_head_rebind": audit,
        "retained_scope": "nine counterfactual row323 versus row12 pair and containment checks only",
        "physical_mass_accounting_unchanged": result["mass_geometry_accounting"] == base["mass_geometry_accounting"],
        "functional_bindings_unchanged_except_anatomy_sha256": True,
        "child_and_all_parent_maps_byte_preserved": result["passive_tissue_parent_bindings"]["surfaces"] == base["passive_tissue_parent_bindings"]["surfaces"],
        "native_binary64_parent_partitions_validated": True,
        "native_admission": False, "anatomical_admission": False,
        "remaining_gates": ["all-nearby reference and retained motion", "joint regional geometry",
                            "native 16 and 60 seconds", "paired 330-second qualification"],
        "preparation_source_sha256": sha(__file__),
    }
    OUT.mkdir()
    shutil.copyfile(CANDIDATE, target)
    for key, source in sidecars.items():
        destination = OUT / source.name
        shutil.copyfile(source, destination)
        require(sha(destination) == common[key]["sha256"], "sidecar bytes changed")
        common[key]["path"] = destination.name
    report_path = OUT / "source-composition.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    result["provenance"]["resting_head_cord_clearance_preparation"] = {
        "report_path": str(report_path), "report_sha256": sha(report_path),
        "source_preparation_only": True,
    }
    receipt = OUT / "resting-anatomy-receipt.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt_path":str(receipt),"receipt_sha256":sha(receipt),
                      "report_sha256":sha(report_path),"anatomy_sha256":sha(target)}))
if __name__ == "__main__":
    main()
