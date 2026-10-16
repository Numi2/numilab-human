"""Source-bound reciprocal interfaces for the existing resting verifier.

A full exact two-solid certificate precedes any omission of that pair from
the ordinary nearby audit. Source row IDs, every ordered source facet, sparse
material supports, and exact native Float32 interface coordinates are checked.
"""
from pathlib import Path
import struct
import json
import numpy as np
from . import passive_attachment_composition as pac
from . import resting_anatomy_verify as verify
from . import cardiac_partition_certificate as cert



def _parent_carrier_material(key, child_sha256, context, vertex_count):
    """Authenticate every existing one-body carrier term; retain full doubles/order."""
    path = Path(context["path"]).resolve()
    digest = context["sha256"]
    pac.require(pac._surface_sha256(path) == digest, "parent receipt hash differs")
    receipt = json.loads(path.read_text())
    extension = receipt["passive_tissue_parent_bindings"]
    pac.require(extension["child_payload_sha256"] == child_sha256,
                "parent receipt child payload differs")
    rows = [row for row in extension["surfaces"] if row["stable_id"] == key[1]]
    pac.require(len(rows) == 1 and len(rows[0]["vertices"]) == vertex_count,
                "parent child surface allocation differs")
    sources = extension["source_payloads"]
    pac.require(0 < len(sources) <= 8, "parent source inventory invalid")
    cache, source_pins, result = {}, {}, []
    for terms in rows[0]["vertices"]:
        pac.require(bool(terms), "parent child material has no terms")
        position = binding_table = None
        active = []
        seen = set()
        total = 0.
        for term in terms:
            pac.require(isinstance(term, list) and len(term) == 4, "invalid parent term")
            source_index, stable_id, slot, coefficient = term
            pac.require(all(type(value) is int and value >= 0
                            for value in (source_index, stable_id, slot))
                        and source_index < len(sources)
                        and type(coefficient) in (int, float)
                        and np.isfinite(coefficient) and coefficient > 0,
                        "invalid parent term identity or coefficient")
            if source_index not in cache:
                source = sources[source_index]
                source_path = Path(source["path"]).resolve()
                pac.require(pac._surface_sha256(source_path) == source["sha256"],
                            "parent carrier source hash differs")
                data = pac._read_nhtiss4(source_path)
                cache[source_index] = {
                    int(row[6]): pac._row_slices(data, row) for row in data["records"]}
                source_pins[source_index] = dict(path=str(source_path), sha256=source["sha256"])
            table = cache[source_index]
            pac.require(stable_id in table, "parent carrier stable ID absent")
            row = table[stable_id]
            pac.require(slot < row["vertex_count"], "parent carrier vertex absent")
            values = np.frombuffer(row["vertex_bytes"], dtype="<f4").reshape(-1, 14)[slot]
            integers = values.view("<u4")
            weights = values[10:14]
            lanes = np.flatnonzero(weights > 0)
            pac.require(np.isfinite(values[:6]).all() and np.isfinite(weights).all()
                        and (weights >= 0).all() and len(lanes) == 1
                        and weights[lanes[0]] == 1,
                        "parent partition requires an existing one-body unit carrier")
            ordinal = int(integers[6+lanes[0]])
            records = row["binding_bytes"]
            pac.require(ordinal*36+36 <= len(records), "parent carrier support absent")
            record = records[ordinal*36:ordinal*36+36]
            body, *rest = struct.unpack("<I8f", record)
            pac.require(0 <= body < 192 and np.isfinite(rest).all() and rest[-1] > 0
                        and abs(sum(x*x for x in rest[3:7])-1) < 1e-4,
                        "parent carrier rigid rest transform invalid")
            point = values[:3].tobytes()
            if position is None:
                pac.require(len(records) % 36 == 0 and 0 < len(records)//36 <= 24,
                            "parent carrier complete support inventory invalid")
                bodies = []
                for offset in range(0, len(records), 36):
                    core, *transform = struct.unpack_from("<I8f", records, offset)
                    pac.require(0 <= core < 192 and np.isfinite(transform).all()
                                and transform[-1] > 0
                                and abs(sum(x*x for x in transform[3:7])-1) < 1e-4,
                                "parent carrier complete rest inventory invalid")
                    bodies.append(core)
                pac.require(len(bodies) == len(set(bodies)),
                            "parent carrier complete core inventory has duplicates")
                position, binding_table = point, records
            pac.require(point == position and records == binding_table,
                        "parent carrier source replicas or ordered rest inventories differ")
            pac.require(ordinal not in seen, "duplicate parent material support")
            seen.add(ordinal)
            active.append((ordinal, struct.pack("<d", coefficient)))
            total += coefficient
        pac.require(abs(total-1.) <= 1e-12, "full parent material partition invalid")
        # Sparse omissions are structural zeros. Never normalize or truncate.
        result.append((position, binding_table, tuple(active)))
    return result, dict(parent_receipt_path=str(path), parent_receipt_sha256=digest,
                        parent_sources=source_pins, complete_positive_terms_retained=True,
                        coefficient_normalization_or_truncation=False,
                        existing_one_body_carriers_only=True)


def _shared_muscle_partition(surfaces, first, second, source: Path, expected_sha256: str,
                             *, coordinate_provenance="accepted_native_capture",
                             exact_certificate=True, parent_bindings_context=None):
    pac.require(coordinate_provenance in ("accepted_native_capture", "counterfactual_owner_replay"),
                "shared partition coordinate provenance invalid")
    first, second = verify.typed_key(first), verify.typed_key(second)
    pac.require(first != second and first[0] == second[0] == 51005,
                "shared muscle certificate requires two distinct typed muscle IDs")
    source = Path(source).resolve()
    pac.require(pac._surface_sha256(source) == expected_sha256, "shared partition source hash differs")
    data = pac._read_nhtiss4(source)
    prepared, tokens, bindings, local_arrays = {}, {}, {}, {}
    parent_proofs = {}
    for key in (first, second):
        rows = [r for r in data["records"] if int(r[6]) == key[1]]
        pac.require(len(rows) == 1 and int(rows[0][7]) == 1, "shared partition source ID/layer differs")
        row = rows[0]
        arrays = pac._nhtiss_row_arrays(data, row)
        local = pac._row_slices(data, row)
        observed = surfaces[key]
        if isinstance(observed, verify.PreparedSurface):
            observed = (observed.positions, observed.faces, observed.normals,
                        observed.source_vertex_ids)
        pac.require(observed[2] is not None,
                    "shared partition requires explicit Float32 coordinate-frame normals")
        captured, proof = verify.retain_bound_source_faces(
            key, observed, arrays["faces"], arrays["first_vertex"],
            np.arange(len(arrays["faces"]), dtype=np.int64))
        pac.require(proof["all_ordered_source_facets_match"], "shared partition ordered source facets differ")
        xyz, faces, normals, gids = captured
        sf = verify.prepare_surface(key, xyz, faces, normals=normals, source_vertex_ids=gids)
        pac.require(sf.closed and xyz.dtype == np.dtype("<f4") and normals.dtype == np.dtype("<f4"),
                    "shared partition requires closed exact Float32 surfaces")
        positions, source_normals = arrays["positions"], arrays["normals"]
        used = np.unique(arrays["faces"])
        pac.require(np.isfinite(source_normals).all()
                    and np.max(np.abs(np.linalg.norm(source_normals[used].astype(float), axis=1)-1)) < 1e-4,
                    "shared partition source normals invalid")
        material = []
        for i in range(len(positions)):
            support = []
            for lane in range(4):
                w = arrays["weights"][i, lane]
                if w > 0:
                    b = int(arrays["binding_indices"][i, lane])
                    record = local["binding_bytes"][b*36:(b+1)*36]
                    pac.require(len(record) == 36, "shared partition support index invalid")
                    fields = struct.unpack("<I8f", record)
                    pac.require(np.isfinite(fields[1:]).all() and fields[-1] > 0
                                and abs(sum(x*x for x in fields[4:8])-1) < 1e-4,
                                "shared partition registered rest transform invalid")
                    support.append((record, w.tobytes()))
            pac.require(support and abs(float(arrays["weights"][i].sum())-1) < 1e-5,
                        "shared partition source weights invalid")
            # Preserve arithmetic lane order. Equivalent unordered supports
            # are insufficient to prove byte-identical native emission.
            material.append((positions[i].tobytes(), tuple(support)))
        if parent_bindings_context is not None:
            material, parent_proofs[key] = _parent_carrier_material(
                key, expected_sha256, parent_bindings_context, len(positions))
        slot = gids.astype(np.int64)-proof["source_capture_vertex_namespace_offset"]-arrays["first_vertex"]
        pac.require((slot >= 0).all() and (slot < len(material)).all(), "source slot map invalid")
        tokens[key] = [material[int(i)] for i in slot]
        if key in parent_proofs:
            proof = dict(proof, full_parent_carrier_material=parent_proofs[key])
        prepared[key], bindings[key], local_arrays[key] = sf, proof, arrays

    return _finish_shared_partition(prepared, tokens, bindings, first, second, source,
                                    expected_sha256, coordinate_provenance, exact_certificate)


def _finish_shared_partition(prepared, tokens, bindings, first, second, source,
                             expected_sha256, coordinate_provenance, exact_certificate,
                             *, classify_connected_patches=False):
    pac.require(coordinate_provenance in ("accepted_native_capture", "counterfactual_owner_replay"),
                "shared partition coordinate provenance invalid")
    def triangle_tokens(key, face):
        return tuple(tokens[key][int(i)] for i in prepared[key].faces[face])
    first_faces = {}
    for i in range(len(prepared[first].faces)):
        token = triangle_tokens(first, i)
        first_faces.setdefault(tuple(sorted(token)), []).append(i)
    interface, pairs = [], []
    for j in range(len(prepared[second].faces)):
        token = triangle_tokens(second, j)
        matches = first_faces.get(tuple(sorted(token)), [])
        if not matches:
            continue
        pac.require(len(matches) == 1, "ambiguous source shared facet")
        i = matches[0]
        other = triangle_tokens(first, i)
        reverse = (token[0], token[2], token[1])
        pac.require(any(other == reverse[k:]+reverse[:k] for k in range(3)),
                    "source shared facet orientation is not reciprocal")
        a, b = prepared[first], prepared[second]
        xyz_a = {t: a.positions[int(v)].tobytes() for t, v in zip(other, a.faces[i])}
        xyz_b = {t: b.positions[int(v)].tobytes() for t, v in zip(token, b.faces[j])}
        pac.require(xyz_a == xyz_b, "shared material owner emitted different Float32 positions")
        interface.append(tuple(tuple(float(x) for x in p) for p in a.positions[a.faces[i]]))
        pairs.append([i, j])
    pac.require(bool(interface), "no reciprocal source/material shared interface")
    if not exact_certificate:
        return {"identities": [list(first), list(second)],
                "source_path": str(source), "source_sha256": expected_sha256,
                "shared_interface_face_count": len(interface),
                "ordered_source_bindings": {str(key): value for key, value in bindings.items()},
                "coordinate_provenance": coordinate_provenance,
                "exact_predicates_executed": False, "physical_state_advanced": False,
                "anatomical_admission": False}
    regions = {str(key): [tuple(tuple(float(x) for x in p) for p in sf.positions[f])
                          for f in sf.faces] for key, sf in prepared.items()}
    # Count each independently closed source component, including nested cavities.
    counts = {str(key): sf.quotient_topology["face_component_count"] for key, sf in prepared.items()}
    audit = cert.audit_shared_interface_partition(regions, interface,
                                                  expected_region_component_counts=counts,
                                                  classify_connected_patches=classify_connected_patches)
    pac.require(audit["interiors_disjoint"] and audit["shared_interface_opposite_chain"],
                "native shared partition certificate failed")
    proof = {"identities": [list(first), list(second)],
            "source_path": str(source), "source_sha256": expected_sha256,
            "ordered_source_bindings": {str(key): proof for key, proof in bindings.items()},
            "shared_face_pairs": pairs, "shared_interface_face_count": len(interface),
            "coordinate_provenance": coordinate_provenance,
            "exact_partition_certificate": audit,
            "physical_state_advanced": False, "anatomical_admission": False}
    if coordinate_provenance == "accepted_native_capture":
        proof["exact_native_partition_certificate"] = audit
    return proof


def _shared_rigid_head_partition(surfaces, first, second, source, expected_sha256,
                                 *, capture_receipt, capture_receipt_sha256,
                                 coordinate_provenance="accepted_native_capture",
                                 exact_certificate=True):
    from .resting_pleura_proxy import _parse_payload
    first, second = verify.typed_key(first), verify.typed_key(second)
    allowed = {frozenset(((51012 if a == 12 else 51010, a), (51010, b)))
               for a, b in ((12, 323), (12, 343), (323, 343), (323, 345), (343, 345))}
    pac.require(frozenset((first, second)) in allowed,
                "rigid head partition typed interface is unsupported")
    pac.require(isinstance(capture_receipt, dict) and
                capture_receipt.get("common_field_source_anatomy_payload_sha256") == expected_sha256,
                "rigid head partition source differs from accepted capture")
    poses = [r for r in capture_receipt.get("accepted_body_poses", [])
             if r.get("body_index") == 23]
    pac.require(len(poses) == 1, "rigid head partition requires unique accepted head pose")
    owner = {"body_index": 23, "capture_receipt_sha256": capture_receipt_sha256,
             "position_m": poses[0]["position_m"],
             "quaternion_xyzw": poses[0]["quaternion_xyzw"]}
    verify._normalize_rigid_owner_proof(owner)
    source = Path(source).resolve()
    pac.require(pac._surface_sha256(source) == expected_sha256,
                "shared partition source hash differs")
    _, _, rows = _parse_payload(source.read_bytes())
    prepared, tokens, bindings = {}, {}, {}
    for key in (first, second):
        row = rows[key[1]]
        pac.require(row["body_index"] == 23 and row["layer"] == (3 if key == (51012, 12) else 1),
                    "rigid head partition source owner or layer differs")
        observed = surfaces[key]
        if isinstance(observed, verify.PreparedSurface):
            observed = (observed.positions, observed.faces, observed.normals,
                        observed.source_vertex_ids)
        pac.require(observed[2] is not None,
                    "shared partition requires explicit Float32 coordinate-frame normals")
        captured, proof = verify.retain_bound_source_faces(
            key, observed, row["faces"], int(row["record"][1]),
            np.arange(len(row["faces"]), dtype=np.int64))
        pac.require(proof["all_ordered_source_facets_match"],
                    "shared partition ordered source facets differ")
        xyz, faces, normals, gids = captured
        sf = verify.prepare_surface(key, xyz, faces, normals=normals, source_vertex_ids=gids)
        pac.require(sf.closed and xyz.dtype == np.dtype("<f4")
                    and normals.dtype == np.dtype("<f4"),
                    "shared partition requires closed exact Float32 surfaces")
        local = row["vertices6"]
        used = np.unique(row["faces"])
        pac.require(np.isfinite(local).all()
                    and np.max(np.abs(np.linalg.norm(local[used, 3:].astype(float), axis=1)-1)) < 1e-4,
                    "shared partition source normals invalid")
        slot = gids.astype(np.int64)-proof["source_capture_vertex_namespace_offset"]-int(row["record"][1])
        pac.require((slot >= 0).all() and (slot < len(local)).all(), "source slot map invalid")
        # Same authenticated payload and rigid owner; local point bytes identify
        # reciprocal material boundaries. Current world equality is checked below.
        tokens[key] = [(local[int(i), :3].tobytes(), 23) for i in slot]
        prepared[key] = sf
        bindings[key] = dict(proof, accepted_rigid_owner=owner,
                             accepted_anatomy_payload_sha256=expected_sha256)
    # Existing exact mode: after all crossings are excluded, each uncut
    # boundary patch has one parity classification. No geometric tolerance.
    return _finish_shared_partition(prepared, tokens, bindings, first, second, source,
                                    expected_sha256, coordinate_provenance, exact_certificate,
                                    classify_connected_patches=True)


def _shared_rib_intercostal_partition(
        surfaces, first, second, source, expected_sha256, *,
        capture_receipt, compiled_bone_identity_inputs,
        coordinate_provenance="accepted_native_capture", exact_certificate=True, **_context):
    """Bind existing source attachment maps, then use the common exact oracle."""
    from .resting_pleura_proxy import _parse_payload
    from . import composed_bone_body_frame as bones
    first, second = verify.typed_key(first), verify.typed_key(second)
    keys = {first, second}
    allowed = ({(51004, 48), (51010, 313)}, {(51004, 49), (51010, 313)})
    pac.require(keys in allowed, "rib partition typed interface is unsupported")
    pac.require(coordinate_provenance == "accepted_native_capture",
                "rib partition requires an unchanged accepted native capture")
    source = Path(source).resolve()
    pac.require(pac._surface_sha256(source) == expected_sha256,
                "rib preparation report hash differs")
    report = json.loads(source.read_text())
    pac.require(report.get("status") == "PASS_EXISTING_SOURCE_OWNER_EXPORT_ONLY"
                and report.get("source_xyz_normals_allocations_and_attachment_maps_exact") is True
                and report.get("full_existing_185_bone_envelope_loader_passed") is True,
                "rib preparation report lacks complete source replay")
    nha = Path(report["output_nha_path"])
    bone_path = Path(report["output_bones_path"])
    bone_manifest = bone_path.with_suffix(".manifest.json")
    nha_sha = report["output_nha_sha256"]
    pac.require(pac._surface_sha256(nha) == nha_sha
                and capture_receipt.get("common_field_source_anatomy_payload_sha256") == nha_sha,
                "rib NHA differs from accepted capture")
    context = compiled_bone_identity_inputs
    pac.require(isinstance(context, dict) and "native_invocation" in context,
                "rib partition requires native-bound compiled bone identities")
    for key, path, expected in (
            ("bone_payload", bone_path, report["output_bones_sha256"]),
            ("bone_manifest", bone_manifest, report["output_bones_manifest_sha256"])):
        pac.require(Path(context[key]["path"]).resolve() == path.resolve()
                    and context[key]["sha256"] == expected
                    and pac._surface_sha256(path) == expected,
                    "rib compiled bone source pin differs")
    invocation = context["native_invocation"]
    pac.require(pac._surface_sha256(Path(invocation["path"])) == invocation["sha256"],
                "rib native invocation drifted")
    argv = json.loads(Path(invocation["path"]).read_text())["argv"]
    pac.require(pac._surface_sha256(Path(argv[3])) == report["output_bones_sha256"],
                "rib bone source differs from accepted native invocation")
    manifest = json.loads(bone_manifest.read_text())
    maps = {row["stable_id"]: row["interface_vertex_pairs"]
            for row in manifest["body_local_bakes"]["rows"]}
    _, _, rows = _parse_payload(nha.read_bytes())
    nr = rows[313]
    bones._runtime_attachment_expected_values(nr, maps, nha_stable_id=313)
    bone_raw = bone_path.read_bytes()
    _, records, vertex_start, index_start = bones._decode_bones(bone_raw)
    bone_key = next(key for key in keys if key[0] == 51004)
    sid = bone_key[1]
    br = next(row for row in records if row[5] == sid)
    pac.require(br[0] == 20, "rib compiled Core owner differs")
    positions = np.frombuffer(bone_raw, "<f4", count=br[2]*6,
                              offset=vertex_start+br[1]*24).reshape(-1, 6)
    faces = np.frombuffer(bone_raw, "<u4", count=br[4],
                         offset=index_start+br[3]*4).reshape(-1, 3).astype(np.int64)-br[1]
    local = {bone_key: positions, (51010, 313): np.asarray(nr["vertices6"], dtype="<f4")}
    source_faces = {bone_key: faces, (51010, 313): np.asarray(nr["faces"])}
    first_vertex = {bone_key: br[1], (51010, 313): int(nr["record"][1])}
    pair_map = dict(maps[sid])
    inverse_map = {value: key for key, value in maps[sid]}
    for a, b in maps[sid]:
        pac.require(0 <= a < len(positions)
                    and positions[a, :3].tobytes() == local[(51010, 313)][b, :3].tobytes(),
                    "rib source attachment coordinates differ")
    prepared, tokens, bindings = {}, {}, {}
    for key in (first, second):
        observed = surfaces[key]
        if isinstance(observed, verify.PreparedSurface):
            observed = (observed.positions, observed.faces, observed.normals,
                        observed.source_vertex_ids)
        pac.require(observed[2] is not None, "rib partition requires explicit native normals")
        captured, proof = verify.retain_bound_source_faces(
            key, observed, source_faces[key], first_vertex[key],
            np.arange(len(source_faces[key]), dtype=np.int64))
        xyz, fs, normals, gids = captured
        sf = verify.prepare_surface(key, xyz, fs, normals=normals, source_vertex_ids=gids)
        pac.require(sf.closed and xyz.dtype == np.dtype("<f4")
                    and normals.dtype == np.dtype("<f4"), "rib partition requires closed Float32 surfaces")
        slots = gids.astype(np.int64)-proof["source_capture_vertex_namespace_offset"]-first_vertex[key]
        pac.require((slots >= 0).all() and (slots < len(local[key])).all(),
                    "rib captured source slot is invalid")
        used = np.unique(source_faces[key])
        pac.require(np.isfinite(local[key]).all()
                    and np.max(np.abs(np.linalg.norm(
                        local[key][used, 3:].astype(float), axis=1)-1)) < 1e-4,
                    "rib source normals invalid")
        if key == bone_key:
            tokens[key] = [("shared", sid, int(i)) if int(i) in pair_map
                           else ("bone", sid, int(i)) for i in slots]
        else:
            tokens[key] = [("shared", sid, inverse_map[int(i)]) if int(i) in inverse_map
                           else ("intercostal", sid, int(i)) for i in slots]
        prepared[key] = sf
        bindings[key] = dict(proof, source_payload_sha256=(
            report["output_bones_sha256"] if key == bone_key else nha_sha),
            native_bound_bone_inputs=context)
    return _finish_shared_partition(
        prepared, tokens, bindings, first, second, source, expected_sha256,
        coordinate_provenance, exact_certificate, classify_connected_patches=True)


def _partition_dispatch(surfaces, first, second, source, expected_sha256,
                        *, capture_context=None, **kwargs):
    first, second = verify.typed_key(first), verify.typed_key(second)
    if {first, second} in ({(51004, 48), (51010, 313)}, {(51004, 49), (51010, 313)}):
        pac.require(capture_context is not None
                    and "compiled_bone_identity_inputs" in capture_context,
                    "rib partition requires native-bound compiled bone identities")
        return _shared_rib_intercostal_partition(
            surfaces, first, second, source, expected_sha256, **capture_context, **kwargs)
    if first[0] != 51005 or second[0] != 51005:
        pac.require(capture_context is not None, "rigid head partition requires accepted capture context")
        return _shared_rigid_head_partition(
            surfaces, first, second, source, expected_sha256,
            capture_receipt=capture_context["capture_receipt"],
            capture_receipt_sha256=capture_context["capture_receipt_sha256"], **kwargs)
    return _shared_muscle_partition(
        surfaces, first, second, source, expected_sha256,
        parent_bindings_context=(capture_context or {}).get("parent_bindings_context"), **kwargs)


def preflight_shared_muscle_partition(surfaces, first, second, source: Path,
                                     expected_sha256: str, *,
                                     coordinate_provenance="accepted_native_capture", capture_context=None):
    """Check owners, facets and reciprocal material presence without exact predicates."""
    return _partition_dispatch(
        surfaces, first, second, source, expected_sha256, capture_context=capture_context,
        coordinate_provenance=coordinate_provenance, exact_certificate=False)


def certify_shared_muscle_partition(surfaces, first, second, source: Path,
                                   expected_sha256: str, *,
                                   coordinate_provenance="accepted_native_capture", capture_context=None):
    """Require the existing exact two-solid certificate after source preparation."""
    return _partition_dispatch(
        surfaces, first, second, source, expected_sha256, capture_context=capture_context,
        coordinate_provenance=coordinate_provenance)


def audit_with_shared_partitions(surfaces, changed, *, partitions, targets=None,
                                required_closed=(), intended_enclosures=(),
                                source_patch_complex_interfaces=None, capture_context=None,
                                coordinate_provenance="accepted_native_capture", progress=None):
    """Use existing exact certificates and existing nearby predicates in order."""
    certificates, partners, seen = [], {}, set()
    for first, second, path, digest in partitions:
        first, second = verify.typed_key(first), verify.typed_key(second)
        pac.require(frozenset((first, second)) not in seen and first != second,
                    "duplicate shared partition declaration")
        seen.add(frozenset((first, second)))
        proof = certify_shared_muscle_partition(surfaces, first, second, path, digest,
                                                coordinate_provenance=coordinate_provenance,
                                                capture_context=capture_context)
        certificates.append(proof)
        partners.setdefault(first, set()).add(second)
        partners.setdefault(second, set()).add(first)
    targets = list(surfaces) if targets is None else list(targets)
    rows, checks = [], []
    for key in changed:
        key = verify.typed_key(key)
        selected = [t for t in targets if t not in partners.get(key, ())]
        result = verify.audit_changed_surfaces(surfaces, [key], targets=selected,
                    required_closed=[key] if key in required_closed else [],
                    intended_enclosures=intended_enclosures,
                    source_patch_complex_interfaces=source_patch_complex_interfaces, progress=progress)
        rows.extend(result.get("rows", []))
        checks.append({"typed_identity": list(key), "audit": result})
        if not result["passed"]:
            return dict(result, rows=rows, per_changed_surface=checks,
                        shared_partition_certificates=certificates)
    return {"passed": True, "rows": rows, "per_changed_surface": checks,
            "shared_partition_certificates": certificates,
            "scope": "Exact shared-solid certificates and all selected nearby surfaces; admission and motion remain separate.",
            "coordinate_provenance": coordinate_provenance}
