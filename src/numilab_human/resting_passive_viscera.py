"""Append registered passive viscera to the existing resting NHANAT payload.

Uses the existing organ-family compiler and atlas registration. No simulation,
new payload ABI, inferred organ shape, or additional physical mass is introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import resting_anatomy as anatomy
from . import model as human
from .organ_family_geometry import compiled_surface
from .physiology import load_anatomy


FAMILIES = (("FMA7200", "small intestine", "Abdomen", 56),
            ("FMA7201", "large intestine", "Abdomen", 8),
            ("FMA7202", "gallbladder", "Abdomen", 1),
            ("FMA15900", "urinary bladder", "pelvis", 1),
            ("FMA9600", "prostate", "pelvis", 1))


def vascular_bindings(source_map: dict) -> list[dict]:
    """Name the anatomical regions represented by the existing CVSim states.

    A displayed vessel is a registered anatomical identity, not a second blood
    volume. Unresolved vascular beds remain explicit aggregates of their region.
    """
    names = ("ascending_aorta", "brachiocephalic_arteries", "upper_body_arteries", "upper_body_veins",
             "superior_vena_cava", "descending_thoracic_aorta", "abdominal_aorta", "renal_arteries",
             "renal_veins", "splanchnic_arteries", "splanchnic_veins", "lower_body_arteries",
             "lower_body_veins", "abdominal_veins", "inferior_vena_cava", "right_atrium",
             "right_ventricle", "pulmonary_arteries", "pulmonary_veins", "left_atrium", "left_ventricle")
    direct = {1: [6], 5: [10], 6: [8], 7: [9], 15: [11], 16: [318], 17: [319], 20: [320], 21: [321]}
    direct[18] = sorted(int(k) for k, v in source_map.items() if v["layer"] == 5)
    direct[19] = sorted(int(k) for k, v in source_map.items() if v["layer"] == 6)
    organs = lambda labels: sorted(int(k) for k, v in source_map.items() if v["name"] in labels)
    splanchnic = organs({"stomach", "pancreas", "spleen", "liver", "small intestine", "large intestine", "gallbladder"})
    representatives = {2: [7], 3: organs({"brain"}), 4: organs({"brain"}),
                       8: [4, 5], 9: [4, 5], 10: splanchnic, 11: splanchnic,
                       12: [], 13: [], 14: splanchnic}
    rows = []
    for identifier, name in enumerate(names, 1):
        ids = sorted(set(direct.get(identifier, representatives.get(identifier, []))) |
                     {int(k) for k, value in source_map.items()
                      if value.get("vascular_compartment_id") == identifier})
        if any(str(i) not in source_map for i in ids):
            raise ValueError("vascular region refers to an absent anatomical identity")
        rows.append({"cvsim_compartment_id": identifier, "name": name,
                     "anatomical_region_id": "source_aggregate:CVSim21:" + name,
                     "stable_ids": ids,
                     "binding_kind": "named_vessel_or_chamber" if identifier in direct else "aggregate_region",
                     "geometry_role": "registered anatomical identity; existing CVSim compartment solely owns blood volume, pressure and flow",
                     "unresolved_geometry": "distal vascular bed is reduced, without individually rendered microvessels" if identifier not in direct else None})
    return rows


def append_viscera(base_receipt: Path, output: Path, *, families=FAMILIES,
                   hierarchy="part_of", layer=1, extension_key="passive_viscera_extension",
                   compartment_by_name=None) -> dict:
    if output.exists():
        raise ValueError("retain previous anatomy evidence; choose a new output directory")
    receipt = json.loads(base_receipt.read_text())
    base = Path(receipt["payload"]["path"])
    if human.sha256(base) != receipt["payload"]["sha256"]:
        raise ValueError("base anatomy receipt and payload differ")
    raw = base.read_bytes()
    magic, abi, count, nv, ni, registration_fp, source_sha = anatomy.HEADER.unpack_from(raw)
    vo = anatomy.HEADER.size + count * anatomy.RECORD.size
    io = vo + nv * 24
    if magic != b"NHANAT1\0" or abi != 5 or len(raw) != io + ni * 4:
        raise ValueError("expected existing exact NHANAT ABI5 payload")
    registration = json.loads(anatomy.REGISTRATION_PATH.read_text())
    registration_sha = human.sha256(anatomy.REGISTRATION_PATH)
    if registration_sha != receipt["provenance"]["bodyparts_registration_sha256"] or int(registration_sha[:8], 16) != registration_fp:
        raise ValueError("passive anatomy registration differs from the existing body")
    for entry in receipt["provenance"]["bodyparts3d_archives"]:
        if human.sha256(anatomy.SOURCES / entry["file"]) != entry["sha256"]:
            raise ValueError("pinned BodyParts3D archive changed")
    if human.sha256(anatomy.RIGID_PATH) != receipt["provenance"]["rigid_payload_sha256"]:
        raise ValueError("passive anatomy rigid source identity differs")
    bodies, _, _ = human._myosim_surface_route_context(anatomy.RIGID_PATH.parent,
        registration["source"]["myosim"]["source"]["archive_sha256"])
    atlas = load_anatomy(anatomy.SOURCES)
    source_map = receipt["provenance"]["source_id_map"]
    existing_members = {v["source_member"]: int(k) for k, v in source_map.items()}
    records, vertices, indices = [raw[anatomy.HEADER.size:vo]], [raw[vo:io]], [raw[io:]]
    next_id = max(map(int, source_map)) + 1
    additions = []
    added_members = {}
    family_rows = []
    for concept, label, body, expected_count in families:
        members = atlas["tables"][hierarchy].get((concept, label), set())
        if len(members) != expected_count:
            raise ValueError("passive source membership changed: " + label)
        family_ids = []
        for member_id in sorted(members):
            if member_id in existing_members:
                identifier = existing_members[member_id]
                if compartment_by_name is None or source_map[str(identifier)]["layer"] != layer:
                    raise ValueError("passive source would duplicate or reclassify base anatomy: " + member_id)
                source_map[str(identifier)]["vascular_compartment_id"] = compartment_by_name[label]
                family_ids.append(identifier)
                continue
            # The source explicitly assigns FJ2599 to both bowel families.
            # Keep one geometry identity and both family incidences.
            if member_id in added_members:
                family_ids.append(added_members[member_id])
                continue
            _, member, obj = human._bodyparts_obj_member(anatomy.SOURCES, hierarchy, member_id)
            digest = hashlib.sha256(obj).hexdigest()
            # Rectum follows the existing pelvic rigid frame; the remaining
            # large intestine and small intestine follow the abdominal frame.
            target = "pelvis" if member_id == "FJ2571" else body
            spec = {"hierarchy": hierarchy, "member_id": member_id,
                    "source_member_sha256": digest, "myosim_body": target}
            owner, source_body, local, normals, faces = compiled_surface(anatomy.SOURCES, spec, registration, bodies)
            packed = np.column_stack((local, normals)).astype("<f4")
            if not np.isfinite(packed).all() or faces.min() < 0 or faces.max() >= len(local):
                raise ValueError("invalid registered passive surface")
            records.append(anatomy.RECORD.pack(owner, nv, len(local), ni, faces.size, next_id, layer, 0))
            vertices.append(packed.tobytes()); indices.append((faces.ravel() + nv).astype("<u4").tobytes())
            item = {"stable_id": next_id, "name": label, "concept_id": concept,
                    "source_member": member_id, "source_member_path": member,
                    "source_sha256": digest, "hierarchy": hierarchy, "body_index": owner,
                    "source_body_id": source_body, "myosim_body": target,
                    "vertex_count": len(local), "triangle_count": len(faces),
                    "local_bounds_m": [local.min(axis=0).tolist(), local.max(axis=0).tolist()]}
            additions.append(item)
            source_map[str(next_id)] = {"name": "ileocecal junction" if member_id == "FJ2599" else label,
                "provider": "BodyParts3D v4.0 existing organ-family compiler",
                "source_member": member_id, "source_sha256": digest, "layer": layer, "body_index": owner,
                "repair": None, "head_family": None,
                "source_owner_metadata": {"concept_id": concept, "hierarchy": hierarchy, "label": label,
                                          "member_id": member_id, "member_sha256": digest,
                                          "myosim_body": target, "core_body_index": owner,
                                          "layer": "vessel" if layer == 2 else "organ"}}
            if compartment_by_name is not None:
                source_map[str(next_id)]["vascular_compartment_id"] = compartment_by_name[label]
            added_members[member_id] = next_id; family_ids.append(next_id)
            next_id += 1; nv += len(local); ni += faces.size
        family_rows.append({"concept_id": concept, "name": label, "source_members": sorted(members), "stable_ids": family_ids})
    payload = anatomy.HEADER.pack(magic, abi, count + len(additions), nv, ni, registration_fp, source_sha)
    payload += b"".join(records) + b"".join(vertices) + b"".join(indices)
    output.mkdir(parents=True)
    path = output / "resting-thorax.nhanatomy"
    path.write_bytes(payload)
    digest = human.sha256(path)
    receipt["payload"].update(path=str(path), sha256=digest, surface_count=count+len(additions), vertex_count=nv, index_count=ni)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = digest
    receipt["functional_bindings"]["vascular_compartment_bindings"] = vascular_bindings(source_map)
    receipt["provenance"][extension_key] = {
        "base_receipt": str(base_receipt), "base_receipt_sha256": human.sha256(base_receipt),
        "base_payload_sha256": hashlib.sha256(raw).hexdigest(), "added_surfaces": additions,
        "complete_source_families": family_rows,
        "preserved_base_record_bytes_sha256": hashlib.sha256(raw[anatomy.HEADER.size:vo]).hexdigest(),
        "preserved_base_vertex_bytes_sha256": hashlib.sha256(raw[vo:io]).hexdigest(),
        "preserved_base_index_bytes_sha256": hashlib.sha256(raw[io:]).hexdigest(),
        "geometry_repair_applied": False, "additional_physical_mass_kg": 0,
        "license": "CC-BY-4.0", "scope": "passive atlas inspection with existing body-frame registration; no added constitutive tissue or vessel-wall mechanics; source interfaces not yet qualified under loaded motion"}
    # Existing derivation receipts stay bound to their original intermediate
    # payload. The append-only lineage above binds them to this new composite.
    receipt_path = output / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    manifest = {"schema": "numi.human.resting-anatomy-manifest.v1", "payload": receipt["payload"],
                "receipt": {"path": str(receipt_path), "sha256": human.sha256(receipt_path)},
                "functional_bindings": receipt["functional_bindings"], "qualification": receipt["qualification"],
                "source_surfaces": source_map, "mass_geometry_accounting": receipt["mass_geometry_accounting"],
                extension_key: receipt["provenance"][extension_key]}
    (output / "resting-anatomy-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for source_path in (anatomy.NHTISS_PATH, anatomy.NHTISS_MANIFEST_PATH):
        (output / source_path.name).symlink_to(source_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(append_viscera(args.base_receipt, args.output)["payload"], sort_keys=True))
