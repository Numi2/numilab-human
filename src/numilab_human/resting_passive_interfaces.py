"""Apply audited passive organ reference repairs to existing NHANAT1.

This is asset preparation, not a tissue solver. Original source identities and
all other surfaces remain intact. Native breathing-cycle admission is separate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .resting_anatomy_interface_patch import HEADER, RECORD, normals, signed_volume, topology_report
from .resting_pleura_proxy import _parse_payload, _record_content_bytes

REPAIRED = {3: ("pancreas", "FJ1895"), 13: ("spleen", "FJ2561")}
SOURCE_DEPENDENCIES = (2, 3, 5, 13, 398)
CARDIAC_IDS = (1, 23, 24, 318, 319, 320, 321)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("passive interface candidate: " + message)


def _surface(vertices, faces):
    v = np.asarray(vertices, dtype="<f4")
    f = np.asarray(faces)
    _require(v.ndim == 2 and v.shape[1] == 3 and len(v) > 3 and np.isfinite(v).all(),
             "invalid candidate positions")
    _require(f.ndim == 2 and f.shape[1] == 3 and len(f) > 3 and
             np.issubdtype(f.dtype, np.integer) and f.min() >= 0 and f.max() < len(v),
             "invalid candidate topology")
    f = f.astype(np.int64)
    t = topology_report(f)
    t.pop("boundary_edges")
    _require(not any(t[k] for k in ("boundary_edge_count", "nonmanifold_edge_count",
                                   "orientation_error_edge_count")), "candidate is not a closed oriented manifold")
    p = v[f].astype(np.float64)
    area2 = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)
    edge = np.maximum.reduce([np.linalg.norm(p[:, 1]-p[:, 0], axis=1),
                              np.linalg.norm(p[:, 2]-p[:, 1], axis=1),
                              np.linalg.norm(p[:, 0]-p[:, 2], axis=1)])
    _require(np.all(edge > 0) and np.all(area2 / np.maximum(edge, 1e-30) >= 1.2e-6),
             "candidate retains a triangle below the declared numerical margin")
    volume = signed_volume(v.astype(np.float64), f)
    _require(volume > 0, "candidate has nonpositive oriented volume")
    n = normals(v.astype(np.float64), f).astype("<f4")
    _require(np.isfinite(n).all(), "candidate normals are nonfinite")
    return np.column_stack((v, n)).astype("<f4"), f, {
        "topology": t, "volume_m3": float(volume),
        "minimum_triangle_altitude_m": float(np.min(area2 / edge)),
    }


def build_candidate(base_payload: Path, base_receipt: Path, reference_payload: Path,
                    candidate_dir: Path, audit_path: Path, derivation_path: Path,
                    output: Path, organ_set: str = "pancreas_spleen",
                    interface_path: Path | None = None) -> dict:
    _require(organ_set in ("pancreas_spleen", "bladder"), "unsupported organ repair scope")
    bladder = organ_set == "bladder"
    repaired = {462: ("urinary bladder", "FJ3149")} if bladder else REPAIRED
    dependencies = (415, 416, 455, 459, 462, 463) if bladder else SOURCE_DEPENDENCIES
    provenance_key = "passive_bladder_reference_interfaces" if bladder else "passive_reference_interfaces"
    _require(not output.exists(), "retain existing output; choose a new directory")
    raw = base_payload.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(base_receipt.read_text())
    reference_sha = _sha(reference_payload)
    header, records, rows = _parse_payload(raw)
    _, _, reference = _parse_payload(reference_payload.read_bytes())
    _require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1" and
             receipt.get("payload", {}).get("sha256") == source_sha and
             receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == source_sha,
             "receipt does not bind the current payload")
    binding = receipt["functional_bindings"].get("passive_viscera_geometry_binding", {})
    passive_ids = set(binding.get("stable_ids", []))
    anchors = "pelvic_anchor_stable_ids" if bladder else "upper_anchor_stable_ids"
    _require(set(repaired).issubset(passive_ids) and
             set(repaired).issubset(binding.get(anchors, [])),
             "repaired organs lack their declared passive anatomical attachment")
    _require(all(sid in rows and sid in reference and
                 _record_content_bytes(rows[sid]) == _record_content_bytes(reference[sid])
                 for sid in dependencies), "registered repair source or neighboring organs changed")
    source_map = receipt.get("provenance", {}).get("source_id_map", {})
    for sid, (name, member) in repaired.items():
        _require(source_map.get(str(sid), {}).get("name") == name and
                 source_map[str(sid)].get("source_member") == member and
                 rows[sid]["body_index"] == 20 and rows[sid]["layer"] == 1,
                 "candidate would replace a different anatomical identity or body owner")
    cardiac = receipt["provenance"].get("cardiac_geometry_binding", {})
    wall = cardiac.get("ventricular_wall_binding", {})
    _require(cardiac.get("output_anatomy_payload_sha256") == source_sha and
             wall.get("output_anatomy_payload_sha256") == source_sha,
             "existing cardiac bindings are stale")
    # A correction descriptor has its own source-payload digest. It must be
    # attached/rebound by its existing cardiac owner after this asset step.
    _require("local_coefficient_refinement" not in wall.get("wall_map", {}),
             "attach source-bound cardiac map refinement after passive geometry preparation")
    audit = json.loads(audit_path.read_text())
    derivation = json.loads(derivation_path.read_text())
    _require(audit.get("source_payload_sha256") == (source_sha if bladder else reference_sha) and
             derivation.get("input_payload_sha256") == reference_sha,
             "geometry audit or derivation is bound to another reference source")
    _require(derivation.get("separation_margin_m") in (0.000025, 0.00005, 0.0001),
             "undeclared reference interface separation")
    unresolved = []
    interface = None
    if bladder:
        _require(interface_path is not None, "bladder repair requires a source neck-interface proof")
        interface = json.loads(interface_path.read_text())
        ledger = {str(s): hashlib.sha256(_record_content_bytes(rows[s])).hexdigest() for s in passive_ids}
        _require(audit.get("source_record_sha256") == ledger, "bladder neighbor audit sources changed")
        _require(interface.get("exact_audit_sha256") == audit.get("exact_audit_sha256")
                 and interface.get("predicate_sha256") == audit.get("predicate_sha256")
                 and interface.get("candidate_sha256") == audit.get("candidate_sha256", {}).get("462")
                 and interface.get("source_members") == ["FJ3149", "FJ3139"]
                 and source_map.get("463", {}).get("source_member") == "FJ3139",
                 "bladder neck proof does not bind the source identities and exact audit")
        for key in ("distance_from_bladder_inferior_extent_m", "distance_from_prostate_superior_extent_m"):
            span = interface.get(key, [])
            _require(len(span) == 2 and 0 <= span[0] <= span[1] <= .005,
                     "bladder/prostate witnesses are outside the localized source neck")
    expected_pairs = {tuple(sorted((a, b))) for a in repaired for b in passive_ids if a != b}
    audited_pairs = {}
    for pair in audit.get("pairs", []):
        key = tuple(sorted(pair.get("ids", [])))
        _require(key not in audited_pairs, "duplicate neighboring-organ audit pair")
        audited_pairs[key] = pair
    _require(set(audited_pairs) == expected_pairs, "neighbor audit does not cover every declared passive organ")
    for key, pair in audited_pairs.items():
        if pair.get("invalid_geometry"):
            _require(not bladder, "bladder audit has an invalid neighbor")
            unresolved.append({"ids": list(key), "reason": pair["invalid_geometry"]})
        elif bladder and key == (462, 463):
            _require(type(pair.get("count")) is int and pair["count"] > 0
                     and pair["count"] == interface.get("source_interface_count")
                     and pair.get("triangle_pairs") == interface.get("candidate_triangle_pairs")
                     and pair.get("audit_complete", True) and not pair.get("count_is_lower_bound", False),
                     "bladder/prostate contact differs from the exact source interface")
        else:
            _require(type(pair.get("count")) is int and pair["count"] == 0 and
                     pair.get("audit_complete", True) and not pair.get("count_is_lower_bound", False),
                     "candidate neighbor audit found a crossing or is incomplete")
    updated = copy.deepcopy(rows)
    repairs = []
    for sid in repaired:
        path = candidate_dir / f"surface-{sid}.npz"
        digest = _sha(path)
        self_check = audit.get("self", {}).get(str(sid), {})
        _require(audit.get("candidate_sha256", {}).get(str(sid)) == digest and
                 type(self_check.get("count")) is int and self_check["count"] == 0 and
                 self_check.get("audit_complete", True) and not self_check.get("count_is_lower_bound", False),
                 "candidate bytes lack a zero-crossing exact self audit")
        with np.load(path, allow_pickle=False) as candidate:
            v, f, detail = _surface(candidate["vertices"], candidate["faces"])
        if bladder:
            # Independently check each claimed preserved source triangle; a
            # boolean flag in an external receipt cannot authorize a new contact.
            def face_key(points):
                p = tuple(tuple(map(float, x)) for x in points)
                return min(p, p[1:] + p[:1], p[2:] + p[:2])
            mapped = interface.get("candidate_to_original_interface_pairs", [])
            pairs = interface.get("candidate_triangle_pairs", [])
            original_pairs = interface.get("source_triangle_pairs", [])
            _require(len(mapped) == len(pairs) and len(pairs) == interface["source_interface_count"]
                     and sorted(mapped) == sorted(original_pairs), "incomplete source neck face correspondence")
            for (new_face, prostate_face), (old_face, source_prostate_face) in zip(pairs, mapped):
                _require(0 <= new_face < len(f) and 0 <= old_face < len(rows[462]["faces"])
                         and 0 <= prostate_face < len(rows[463]["faces"])
                         and prostate_face == source_prostate_face
                         and face_key(v[f[new_face], :3]) == face_key(rows[462]["vertices6"][rows[462]["faces"][old_face], :3]),
                         "a protected source bladder neck triangle changed")
        original_volume = signed_volume(rows[sid]["vertices6"][:, :3].astype(float), rows[sid]["faces"])
        _require(0 < detail["volume_m3"] <= original_volume and
                 detail["volume_m3"] >= 0.95 * original_volume,
                 "localized interface repair changes more than five percent of a passive organ")
        updated[sid]["vertices6"], updated[sid]["faces"] = v, f
        repairs.append({"stable_id": sid, "anatomical_name": repaired[sid][0],
                        "original_record_geometry_sha256": hashlib.sha256(_record_content_bytes(rows[sid])).hexdigest(),
                        "candidate_npz_sha256": digest, "source_volume_m3": float(original_volume), **detail})
    chunks, vertices, indices, nv, ni = [], [], [], 0, 0
    for record in records:
        sid = record[5]
        row = updated[sid]
        v, f = row["vertices6"], row["faces"]
        chunks.append(RECORD.pack(row["body_index"], nv, len(v), ni, f.size, sid, row["layer"], row["flags"]))
        vertices.append(np.asarray(v, dtype="<f4").tobytes())
        indices.append((f.reshape(-1) + nv).astype("<u4").tobytes())
        nv += len(v)
        ni += f.size
    result = HEADER.pack(header[0], header[1], len(records), nv, ni, header[5], header[6])
    result += b"".join(chunks + vertices + indices)
    _, _, result_rows = _parse_payload(result)
    _require(all(_record_content_bytes(rows[s]) == _record_content_bytes(result_rows[s])
                 for s in rows if s not in repaired), "unrelated anatomical surface changed")
    _require(all(s in result_rows for s in CARDIAC_IDS), "cardiac structures are incomplete")
    output_sha = hashlib.sha256(result).hexdigest()
    detail = {"interpretation": "inferred_reference_interface_repair_not_measured_subject_geometry",
              "compiler_source_sha256": _sha(Path(__file__)),
              "base_payload_sha256": source_sha, "reference_payload_sha256": reference_sha,
              "base_receipt_sha256": _sha(base_receipt), "exact_source_audit_sha256": _sha(audit_path),
              "derivation_receipt_sha256": _sha(derivation_path), "derivation": derivation,
              "repaired_surfaces": repairs, "unresolved_neighbor_checks": unresolved,
              "all_other_record_geometry_bytes_preserved": True,
              "additional_physical_mass_kg": 0, "additional_physiological_state": False,
              "qualification": "source self and valid passive-neighbor checks passed; unresolved source neighbors and accepted native breathing cycle remain required"}
    if bladder:
        detail["source_neck_interface"] = interface
        detail["source_neck_interface_sha256"] = _sha(interface_path)
        detail["qualification"] = "Exact source self and non-neck neighbor checks passed; the localized source bladder/prostate neck triangles are unchanged. Accepted native breathing-cycle checks remain required."
    receipt["provenance"][provenance_key] = detail
    for sid in repaired:
        source_map[str(sid)]["reference_geometry_status"] = detail["interpretation"]
    receipt["payload"].update(path=str(output / "resting-thorax.nhanatomy"), sha256=output_sha,
                              surface_count=len(records), vertex_count=nv, index_count=ni)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    cardiac["output_anatomy_payload_sha256"] = output_sha
    wall["output_anatomy_payload_sha256"] = output_sha
    receipt["qualification"][provenance_key] = detail["qualification"]
    output.mkdir(parents=True)
    (output / "resting-thorax.nhanatomy").write_bytes(result)
    receipt_path = output / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    manifest = {"schema": "numi.human.resting-anatomy-manifest.v1", "payload": receipt["payload"],
                "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)},
                "functional_bindings": receipt["functional_bindings"], "qualification": receipt["qualification"],
                "source_surfaces": source_map, "mass_geometry_accounting": receipt["mass_geometry_accounting"]}
    (output / "resting-anatomy-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base-payload", "base-receipt", "reference-payload", "candidate-dir", "audit-path", "derivation-path", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--organ-set", choices=("pancreas_spleen", "bladder"), default="pancreas_spleen")
    parser.add_argument("--interface-path", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_candidate(**vars(args))["payload"], sort_keys=True))


if __name__ == "__main__":
    main()
