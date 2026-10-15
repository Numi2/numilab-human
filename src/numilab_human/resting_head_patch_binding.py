"""Bind an existing reciprocal head-patch report to the shared capture verifier.

This reconstructs ancestry and the accepted rigid owner only. The existing exact
patch validator and pair/solid predicates remain responsible for classification.
"""
from pathlib import Path
import copy
import gzip
import hashlib
import json
import zipfile
from fractions import Fraction

import numpy as np

from . import resting_anatomy_verify as verify
from .resting_pleura_proxy import _parse_payload


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _checked_inputs(pins):
    _need(isinstance(pins, dict) and pins, "head patch requires pinned source inputs")
    checked = {}
    for name, digest in pins.items():
        if name == "verifier_module":
            raw = Path(verify.__file__).read_bytes()
        elif digest == "member-only-verified-not-whole-archive-hash":
            _need(Path(name).is_file(), "head patch source archive is missing")
            continue
        elif "::" in name:
            archive, member = name.split("::", 1)
            with zipfile.ZipFile(archive) as source:
                raw = source.read(member)
        else:
            raw = Path(name).read_bytes()
        _need(isinstance(digest, str) and len(digest) == 64
              and _sha(raw) == digest, f"head patch source SHA differs: {name}")
        checked[name] = (digest, raw)
    return checked


def _one_input(checked, digest, suffix=None):
    matches = [(name, raw) for name, (sha, raw) in checked.items()
               if sha == digest and (suffix is None or name.endswith(suffix))]
    _need(len(matches) == 1, "head patch input pin is missing or ambiguous")
    return matches[0]


def bind_head_source_patch_report(surfaces, endpoint, report_path, expected_sha):
    """Return source-bound surfaces and a declaration from an existing report.

    Only the typed medulla source regions 343 and 374 patch is supported. Historical
    proof results grant no exception: all source pins, ordered facets, ancestry,
    generated point parents and current capture hashes are checked again.
    """
    raw = Path(report_path).read_bytes()
    _need(_sha(raw) == expected_sha, "head patch report SHA differs")
    report = json.loads(raw)
    _need(report.get("schema") ==
          "numi.human.head343-374-source-patch-fragment-complex-gate.v1",
          "head patch report schema differs")
    _need(report.get("status") in ("PASS_FOCUSED_SOURCE_PATCH_PAIR",
                                   "PREFLIGHT_ONLY_COMPLETE_NO_EXACT_AUDITS"),
          "head patch report is incomplete or failed")
    declaration = copy.deepcopy(report["source_patch_complex"])
    keys = ((51010, 343), (51010, 374))
    _need(declaration.get("identities") == [list(k) for k in keys],
          "head patch typed identities differ")
    pins = declaration["input_pins"]
    _need(pins["capture_pack_sha256"] == endpoint["pack_sha256"]
          and pins["capture_receipt_sha256"] == endpoint["capture_receipt_sha256"],
          "head patch belongs to a different accepted capture")
    checked = _checked_inputs(report["input_pins"])
    _one_input(checked, pins["capture_pack_sha256"], ".mrvpack")
    _, receipt_raw = _one_input(checked, pins["capture_receipt_sha256"], ".json")
    receipt = json.loads(receipt_raw)
    poses = [r for r in receipt.get("accepted_body_poses", [])
             if r.get("body_index") == 23]
    _need(len(poses) == 1, "head patch requires the unique accepted head rigid owner")
    owner = {"body_index": 23,
             "capture_receipt_sha256": endpoint["capture_receipt_sha256"],
             "position_m": poses[0]["position_m"],
             "quaternion_xyzw": poses[0]["quaternion_xyzw"]}
    _, nha_raw = _one_input(checked, pins["current_payload_sha256"], ".nhanatomy")
    _, _, source_rows = _parse_payload(nha_raw)
    # A seal authenticates bytes only; bind every declared parent to those bytes.
    for digest in pins.values():
        _one_input(checked, digest)
    _, parent_raw = _one_input(checked, pins["parent_payload_sha256"], ".nhanatomy")
    _, _, parent_rows = _parse_payload(parent_raw)
    _, arrangement_raw = _one_input(checked, pins["arrangement_sha256"], ".json")
    arrangement = json.loads(arrangement_raw)
    for source in declaration["source_faces"]:
        index = source["arrangement_record_index"]
        _need(0 <= index < len(arrangement) and
              source["arrangement_record"] == arrangement[index],
              "head patch arrangement record differs from source bytes")
        face = parent_rows[343]["faces"][source["source_row_face"]]
        _need(list(map(int, face)) == source["parent_ordered_source_vertex_ids"],
              "head patch parent face corners differ from source bytes")
        coordinates = parent_rows[343]["vertices6"][face, :3]
        encoded = [[[str(Fraction(float(x)).numerator), str(Fraction(float(x)).denominator)]
                    for x in point] for point in coordinates]
        _need(encoded == source["parent_vertices_exact"],
              "head patch parent coordinates differ from source bytes")
    members = report["source_member_hashes"]
    _need(members.get("FJ1769") == declaration["source_member_sha256"]
          and members.get("FJ1769") == pins["source_obj_sha256"],
          "head patch source identity differs")
    # Authenticate all named OBJ members, including the member-only archive pin.
    for member, digest in members.items():
        matches = [(name, value) for name, (sha, value) in checked.items()
                   if sha == digest and name.endswith("/" + member + ".obj")]
        _need(len(matches) == 1 and
              ("# File ID : " + member) in matches[0][1].decode("utf-8"),
              "head patch OBJ identity is missing or ambiguous")
    _, obj_raw = _one_input(checked, pins["source_obj_sha256"], ".obj")
    obj_faces = []
    for line in obj_raw.decode("utf-8").splitlines():
        if line.startswith("f "):
            corners = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            _need(len(corners) == 3 and min(corners) >= 0,
                  "head patch source OBJ is not positive-index triangular")
            obj_faces.append(corners)
    for source in declaration["source_faces"]:
        _need(obj_faces[source["source_face"]] == source["source_vertex_ids"],
              "head patch OBJ face corners differ from authenticated member")
    bound = {}
    for side, key, ancestry_name in zip(("first", "second"), keys,
                                       ("ancestry_first_sha256", "ancestry_second_sha256")):
        name, lineage_raw = _one_input(checked, pins[ancestry_name])
        if name.endswith(".gz"):
            lineage_raw = gzip.decompress(lineage_raw)
        lineage = copy.deepcopy(json.loads(lineage_raw)["rows"][str(key[1])])
        row = source_rows[key[1]]
        _need(row["body_index"] == 23 and row["layer"] == 1,
              "head patch NHA rigid owner or layer differs")
        _need(key in surfaces, "head patch surface is missing from the accepted capture")
        positions, faces, normals, gids = surfaces[key]
        _, proof = verify.retain_bound_source_faces(
            key, surfaces[key], row["faces"], int(row["record"][1]),
            np.arange(len(row["faces"]), dtype=np.int64))
        _need(proof["all_ordered_source_facets_match"] and len(lineage) == len(faces),
              "head patch ordered source facets or ancestry differ")
        for record in lineage:
            for frame in verify._source_face_lineage_chain(record):
                member = frame.get("source_member")
                if member in members:
                    _need(frame.get("source_member_sha256") in (None, members[member]),
                          "head patch ancestry source SHA differs")
                    frame["source_member_sha256"] = members[member]
        parents = {}
        for source in declaration["source_faces"]:
            for point_id, bary in zip(source["arrangement_point_ids"], source["barycentric_exact"]):
                parents.setdefault(point_id, []).append({
                    "source_member": declaration["source_member"],
                    "source_member_sha256": declaration["source_member_sha256"],
                    "source_face": source["source_face"],
                    "source_row_face": source["source_row_face"],
                    "source_row_payload_sha256": pins["parent_payload_sha256"],
                    "parent_ordered_source_vertex_ids": source["parent_ordered_source_vertex_ids"],
                    "barycentric_exact": bary})
        vertices = {}
        for point in declaration[side]["point_bindings"]:
            for gid in point["capture_gids"]:
                _need(gid not in vertices, "head patch repeats a material vertex")
                vertices[gid] = {"source_gid": gid,
                                 "source_member": declaration["source_member"],
                                 "source_member_sha256": declaration["source_member_sha256"],
                                 "source_parent_faces": parents[point["point_id"]]}
        bound[key] = verify.prepare_surface(
            key, positions, faces, normals=normals, source_vertex_ids=gids,
            source_face_lineage=lineage, source_vertex_lineage=vertices,
            rigid_owner_proof=owner)
    verify._prepare_source_patch_complex(bound[keys[0]], bound[keys[1]], declaration)
    return bound, declaration, {
        "path": str(Path(report_path).resolve()), "sha256": expected_sha,
        "source_patch_complex_sha256": declaration["source_patch_complex_sha256"],
        "scope": "Current capture source ancestry binding; exact predicates remain required."}
