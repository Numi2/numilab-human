#!/usr/bin/env python3
"""Compose reviewed regional preparations in the existing resting receipt.

Source-only operation. No anatomy, captured-motion, or native admission.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

from numilab_human.resting_brachiocephalic_common_field import _resolve_sidecar
from numilab_human.resting_brachiocephalic_joint_parent_composition import (
    extend_joint_source_inventory,
    merge_vascular_parent_maps,
)
from numilab_human.resting_common_field_passive_rebind import rebind_passive_rows
from numilab_human.resting_pleura_proxy import _parse_payload, _record_content_bytes

PINS = {
    "forearm_receipt": "1b32f01e90b821f8b04dd15b2951c6a28613db0b90385f9b0510ea1090e942f4",
    "vascular_receipt": "5ed44e1cfa912f8bf291347aff19c16cf2c55c4b9a36c3ab04fb6aaf17b70474",
    "head_payload": "2f8af1031b3c5b8cc26866f32935c9b695127a2f357732e566be3f14fdc2666d",
    "head_evidence": "93b2c9bd375295ca9fa7bd7de6b3a02a6f5daa4603cdec4ee67e5ecb49599e2c",
}
FDS_PROVENANCE = (
    "forearm492_source_preparation", "native_muscle_surfaces",
    "reference_surface_composition_binding", "reference_surface_composition_binding_history",
    "row157_parent_binding_composition",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose(forearm_path, vascular_path, head_path, evidence_path, output):
    inputs = {
        "forearm_receipt": Path(forearm_path).resolve(strict=True),
        "vascular_receipt": Path(vascular_path).resolve(strict=True),
        "head_payload": Path(head_path).resolve(strict=True),
        "head_evidence": Path(evidence_path).resolve(strict=True),
    }
    for key, path in inputs.items():
        require(sha(path) == PINS[key], "reviewed input changed: " + key)
    output = Path(output).resolve()
    require(not output.exists(), "preserve earlier attempts")
    f = json.loads(inputs["forearm_receipt"].read_text())
    v = json.loads(inputs["vascular_receipt"].read_text())
    require(f["mass_geometry_accounting"] == v["mass_geometry_accounting"], "physical mass mismatch")
    ff, vf = copy.deepcopy(f["functional_bindings"]), copy.deepcopy(v["functional_bindings"])
    ff.pop("anatomy_payload_sha256")
    vf.pop("anatomy_payload_sha256")
    require(ff == vf, "functional owner changed between passive candidates")
    for record in (f["payload"], v["payload"]):
        require(sha(record["path"]) == record["sha256"], "actual anatomy bytes changed")
    _fh, _fr, frows = _parse_payload(Path(f["payload"]["path"]).read_bytes())
    _vh, _vr, vrows = _parse_payload(Path(v["payload"]["path"]).read_bytes())
    require(list(frows) == list(vrows), "ordered anatomy identities differ")
    require(all(_record_content_bytes(frows[sid]) == _record_content_bytes(vrows[sid])
                and all(frows[sid][k] == vrows[sid][k] for k in ("body_index", "layer", "flags"))
                for sid in frows if sid != 477),
            "vascular successor changed unrelated anatomy")
    fp, vp = f["provenance"], v["provenance"]
    permitted = set(FDS_PROVENANCE) | {
        "source_id_map", "passive_brachiocephalic_vein_extension",
        "axillary_477_cap_reinstatement_successor", "cardiac_geometry_binding",
        "passive_brachiocephalic_exact_coordinate_owner_composition",
    }
    require(all(fp.get(k) == vp.get(k) for k in set(fp) | set(vp) if k not in permitted),
            "unexpected provenance conflict")
    native = fp["native_muscle_surfaces"]
    for path_key, hash_key in (("payload_path", "sha256"), ("manifest_path", "manifest_sha256")):
        require(sha(native[path_key]) == native[hash_key], "forearm source identity changed")
    fext, vext = f["passive_tissue_parent_bindings"], v["passive_tissue_parent_bindings"]
    require(fext["child_payload_sha256"] == native["sha256"], "forearm child binding changed")
    for extension in (fext, vext):
        for source in extension["source_payloads"]:
            require(sha(source["path"]) == source["sha256"], "parent source changed")
    extended, index_map, source_audit = extend_joint_source_inventory(
        vext, fext, append_paths_by_common_index={
            index: source["path"] for index, source in enumerate(vext["source_payloads"])
        },
    )
    merged, parent_audit = merge_vascular_parent_maps(
        vext, extended, final_anatomy_sha256=v["payload"]["sha256"],
        final_child_sha256=native["sha256"], common_source_to_joint=index_map,
    )
    require(merged["source_payloads"][:7] == fext["source_payloads"]
            and len(merged["source_payloads"]) == 8
            and merged["surfaces"] == fext["surfaces"], "joint source ownership changed")
    result = copy.deepcopy(v)
    for key in FDS_PROVENANCE:
        result["provenance"][key] = copy.deepcopy(fp[key])
    result["passive_tissue_parent_bindings"] = merged
    output_payload = output / "resting-thorax.nhanatomy"
    result, head_audit = rebind_passive_rows(
        result, Path(v["payload"]["path"]).read_bytes(), inputs["head_payload"].read_bytes(),
        changed_ids=(323, 345), output_path=str(output_payload),
    )
    common = result["provenance"]["cardiac_geometry_binding"]["common_field"]
    sidecars = {key: _resolve_sidecar(inputs["vascular_receipt"],
                                     inputs["vascular_receipt"], common[key])
                for key in ("map", "polynomials", "domain_boxes")}
    names = [path.name for path in sidecars.values()]
    require(len(set(names)) == 3, "sidecar filename collision")
    report = {
        "status": "SOURCE_PREPARATION_ONLY_NOT_ADMISSION",
        "inputs": {key: {"path": str(path), "sha256": PINS[key]} for key, path in inputs.items()},
        "source_inventory": source_audit, "parent_composition": parent_audit,
        "passive_head_rebind": head_audit,
        "physical_mass_accounting_unchanged": True,
        "functional_bindings_unchanged_except_anatomy_sha256": True,
        "back157158_and_fdp_parent_maps_preserved": True,
        "common_sidecar_sha256": {key: sha(path) for key, path in sidecars.items()},
        "code_sha256": {
            str(Path(path).resolve()): sha(path) for path in (
                __file__,
                *(sys.modules[name].__file__ for name in (
                    "numilab_human.resting_common_field_passive_rebind",
                    "numilab_human.resting_brachiocephalic_common_field",
                    "numilab_human.resting_brachiocephalic_joint_parent_composition",
                    "numilab_human.resting_pleura_proxy",
                )),
            )
        },
        "remaining_gates": ["all-nearby source geometry", "retained motion", "joint geometry",
                            "native 16 and 60 seconds", "paired 330-second qualification"],
        "native_admission": False, "anatomical_admission": False,
    }
    output.mkdir(parents=True)
    shutil.copyfile(inputs["head_payload"], output_payload)
    for key, source in sidecars.items():
        destination = output / source.name
        shutil.copyfile(source, destination)
        require(sha(destination) == common[key]["sha256"], "sidecar copy changed")
        common[key]["path"] = destination.name
    report_path = output / "source-composition.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    result["provenance"]["resting_regional_source_composition"] = {
        "report_path": str(report_path), "report_sha256": sha(report_path),
        "source_preparation_only": True,
    }
    receipt_path = output / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path),
            "report_sha256": sha(report_path), "anatomy_sha256": sha(output_payload),
            "child_sha256": native["sha256"], "parent_source_count": len(merged["source_payloads"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("forearm-receipt", "vascular-receipt", "head-payload", "head-evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.forearm_receipt, args.vascular_receipt,
                             args.head_payload, args.head_evidence, args.output), sort_keys=True))
