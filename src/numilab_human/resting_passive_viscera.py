"""Append registered passive viscera to the existing resting NHANAT payload.

Uses the existing organ-family compiler and atlas registration. No simulation,
new payload ABI, inferred organ shape, or additional physical mass is introduced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import resting_anatomy as anatomy
from . import model as human
from .organ_family_geometry import compiled_surface
from .physiology import load_anatomy
from .resting_scene import load_rigid
from .resting_pleura_proxy import _parse_payload, _record_content_bytes


FAMILIES = (("FMA7200", "small intestine", "Abdomen", 56),
            ("FMA7201", "large intestine", "Abdomen", 8),
            ("FMA7202", "gallbladder", "Abdomen", 1),
            ("FMA15900", "urinary bladder", "pelvis", 1),
            ("FMA9600", "prostate", "pelvis", 1))

# These source components all belong to FMA7201, but are not seven distinct
# bowel segments. Keep family ownership for the reduced circulation while
# preserving the narrower source identities used for anatomical interfaces.
COLON_COMPONENTS = {
    "FJ2566": ("FMA14545", "ascending colon"),
    "FJ2567": ("FMA14547", "descending colon"),
    "FJ2568": ("FMA15044", "taenia libera"),
    "FJ2569": ("FMA15042", "taenia mesocolica"),
    "FJ2570": ("FMA15043", "taenia omentalis"),
    "FJ2571": ("FMA14544", "rectum"),
    "FJ2572": ("FMA14546", "transverse colon"),
}

# BodyParts3D's 56 small-intestine source members include one duodenum,
# one ileocecal junction and 54 named jejunal/ileal zones. Resolve their
# narrower identity from the locked source table, without inventing 56 organs.
SMALL_INTESTINE_MEMBERS = frozenset(f"FJ{i}" for i in range(2573, 2629))
SMALL_INTESTINE_COMPONENTS = (
    ("FMA7206", "duodenum"),
    ("FMA11338", "ileocecal junction"),
    ("FMA14964", "proximal part of ileum"),
    ("FMA14965", "middle part of ileum"),
    ("FMA14966", "distal part of ileum"),
    ("FMA16981", "proximal part of jejunum"),
    ("FMA16982", "middle part of jejunum"),
    ("FMA16983", "distal part of jejunum"),
)

PASSIVE_COMPONENT_IDENTITY_REFRESH = "passive_component_identity_refresh"


def anatomical_component_identity(atlas: dict, member_id: str) -> dict | None:
    """Resolve supported bowel components against the loaded source table.

    This records source anatomy, not an automatic exemption from collision or
    tissue-volume checks. A taenia is a muscular component of the colon wall;
    its actual interface still needs a geometry audit.
    """
    component = COLON_COMPONENTS.get(member_id)
    hierarchy, family = "part_of", "FMA7201"
    if component is None:
        if member_id not in SMALL_INTESTINE_MEMBERS:
            return None
        hierarchy, family = "is_a", "FMA7200"
        table = atlas.get("tables", {}).get(hierarchy, {})
        matches = [row for row in SMALL_INTESTINE_COMPONENTS
                   if member_id in table.get(row, set())]
        if len(matches) != 1:
            raise ValueError("source anatomical component membership changed or ambiguous: " + member_id)
        component = matches[0]
    concept, label = component
    members = atlas["tables"][hierarchy].get((concept, label), set())
    if member_id not in members:
        raise ValueError("source anatomical component membership changed: " + member_id)
    role = ("colon wall muscle component" if concept in {"FMA15042", "FMA15043", "FMA15044"}
            else "large intestine segment")
    if family == "FMA7200":
        role = ("intestinal junction" if concept == "FMA11338" else
                "small intestine segment" if concept == "FMA7206" else "small intestine zone")
    return {"concept_id": concept, "name": label, "hierarchy": hierarchy,
            "source_member": member_id, "family_concept_id": family,
            "geometry_role": role,
            "separate_physiological_compartment": False}


def refresh_passive_component_identities(receipt: dict, atlas: dict) -> dict:
    """Enrich an existing receipt's source map from pinned BP3D membership.

    This is metadata-only: it neither imports meshes nor changes stable IDs,
    surface records, payload hashes, mass, or the broad family names used by
    the existing reduced physiology bindings. The exact source component is
    added under each row's existing ``source_owner_metadata``. Completeness
    and parent-family checks prevent a partial or mismatched receipt from
    acquiring apparently authoritative names.
    """
    source = atlas.get("source") if isinstance(atlas, dict) else None
    if (not isinstance(source, dict) or source.get("id") != "bodyparts3d_4"
            or source.get("version") != "4.0"):
        raise ValueError("component identity refresh requires pinned BodyParts3D 4.0 tables")
    source_tables = source.get("tables")
    if not isinstance(source_tables, list):
        raise ValueError("component identity refresh lacks source table provenance")
    table_hashes = {}
    for table in source_tables:
        if not isinstance(table, dict):
            raise ValueError("malformed source table provenance")
        filename, digest = table.get("file"), table.get("sha256")
        if (not isinstance(filename, str) or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("malformed or duplicate source table provenance")
        if filename in table_hashes:
            raise ValueError("malformed or duplicate source table provenance")
        table_hashes[filename] = digest
    required_tables = {"isa_element_parts.txt", "partof_element_parts.txt"}
    if set(table_hashes) != required_tables or any(
            any(c not in "0123456789abcdef" for c in digest) for digest in table_hashes.values()):
        raise ValueError("component identity refresh requires both pinned source membership tables")

    expected_members = SMALL_INTESTINE_MEMBERS | frozenset(COLON_COMPONENTS)
    updated = deepcopy(receipt)
    provenance = updated.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("source_id_map"), dict):
        raise ValueError("component identity refresh requires an existing source_id_map")
    source_map = provenance["source_id_map"]
    locations: dict[str, tuple[str, dict]] = {}
    for stable_id, row in source_map.items():
        if not isinstance(stable_id, str) or not stable_id.isdecimal() or str(int(stable_id)) != stable_id:
            raise ValueError("source_id_map contains a noncanonical stable ID")
        if not isinstance(row, dict):
            raise ValueError("source_id_map contains a malformed row")
        member_id = row.get("source_member")
        if not isinstance(member_id, str) or member_id not in expected_members:
            continue
        if member_id in locations:
            raise ValueError("source member has duplicate rendered identities: " + member_id)
        owner = row.get("source_owner_metadata")
        if not isinstance(owner, dict):
            raise ValueError("bowel source row lacks source_owner_metadata: " + member_id)
        small = member_id in SMALL_INTESTINE_MEMBERS
        family_id = "FMA7200" if small else "FMA7201"
        family_label = "small intestine" if small else "large intestine"
        if (owner.get("concept_id") != family_id or owner.get("hierarchy") != "part_of"
                or owner.get("label") != family_label or owner.get("member_id") != member_id
                or not isinstance(row.get("source_sha256"), str)
                or owner.get("member_sha256") != row.get("source_sha256")):
            raise ValueError("bowel source identity differs from its registered family owner: " + member_id)
        locations[member_id] = (stable_id, row)

    if set(locations) != expected_members:
        missing = sorted(expected_members - set(locations))
        extra = sorted(set(locations) - expected_members)
        raise ValueError(f"source_id_map bowel membership is incomplete or changed: missing={missing}, extra={extra}")

    identities = []
    for member_id in sorted(expected_members):
        stable_id, row = locations[member_id]
        identity = anatomical_component_identity(atlas, member_id)
        if not isinstance(identity, dict):
            raise ValueError("source membership does not resolve to a supported bowel component: " + member_id)
        owner = row["source_owner_metadata"]
        prior = owner.get("anatomical_component")
        if prior is not None and prior != identity:
            raise ValueError("existing component identity conflicts with pinned source tables: " + member_id)
        owner["anatomical_component"] = identity
        row["name"] = identity["name"]
        identities.append({"stable_id": int(stable_id), **identity})

    try:
        payload_sha = updated["payload"]["sha256"]
        if (not isinstance(payload_sha, str) or len(payload_sha) != 64
                or any(c not in "0123456789abcdef" for c in payload_sha)):
            raise ValueError
        bindings = updated["functional_bindings"]
        if not isinstance(bindings, dict) or bindings.get("anatomy_payload_sha256") != payload_sha:
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("receipt payload identity is missing or inconsistent") from error

    identity_bytes = json.dumps(identities, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False).encode("utf-8")
    refresh = {
        "schema": "numi.human.passive_component_identity_refresh.v1",
        "owner": "numilab_human.resting_passive_viscera.anatomical_component_identity",
        "source": {key: source.get(key) for key in ("id", "version", "license", "attribution", "url")},
        "source_tables": [{"file": name, "sha256": table_hashes[name]} for name in sorted(table_hashes)],
        "stable_ids": sorted(int(locations[m][0]) for m in expected_members),
        "source_member_count": len(expected_members),
        "identity_rows_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "payload_sha256_unchanged": payload_sha,
        "geometry_changed": False,
        "physical_mass_changed_kg": 0,
        "source_map_component_display_names_changed": True,
        "source_owner_family_labels_changed": False,
        "vascular_region_memberships_changed": False,
    }
    previous = provenance.get(PASSIVE_COMPONENT_IDENTITY_REFRESH)
    if previous is not None and previous != refresh:
        raise ValueError("existing component identity refresh provenance conflicts with this source")
    provenance[PASSIVE_COMPONENT_IDENTITY_REFRESH] = refresh
    return updated


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
    def organs(labels):
        return sorted(int(k) for k, row in source_map.items()
                      if (row.get("source_owner_metadata") or {}).get("label", row.get("name")) in labels)
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


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("passive bowel composition: " + message)


def _area_weighted_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Recompute smooth normals from one complete aggregate surface."""
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    _require(xyz.ndim == 2 and xyz.shape[1] == 3 and tri.ndim == 2 and tri.shape[1] == 3,
             "malformed registered bowel mesh")
    _require(len(tri) > 0 and int(tri.min()) >= 0 and int(tri.max()) < len(xyz),
             "registered bowel mesh has invalid face indices")
    cross = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
    _require(np.isfinite(cross).all() and bool(np.all(np.linalg.norm(cross, axis=1) > 0.0)),
             "registered bowel mesh contains a zero-area or nonfinite face")
    normals = np.zeros_like(xyz)
    np.add.at(normals, tri[:, 0], cross)
    np.add.at(normals, tri[:, 1], cross)
    np.add.at(normals, tri[:, 2], cross)
    lengths = np.linalg.norm(normals, axis=1)
    _require(bool(np.all(np.isfinite(lengths) & (lengths > 0.0))),
             "registered bowel mesh has an undefined vertex normal")
    return (normals / lengths[:, None]).astype("<f4")


def _owned_surface_rows(vertices: np.ndarray, normals: np.ndarray,
                        faces: np.ndarray, owners: np.ndarray,
                        stable_ids: range) -> dict[int, dict]:
    """Create one local NHANAT row per source-owned face group.

    The global aggregate may share vertices across material owners. Each local
    row keeps the exact same packed positions and aggregate normal at those
    shared points, while indices remain local to the row as required by ABI5.
    """
    xyz = np.asarray(vertices, dtype="<f4")
    nrm = np.asarray(normals, dtype="<f4")
    tri = np.asarray(faces, dtype=np.int64)
    own = np.asarray(owners, dtype=np.int64)
    _require(xyz.ndim == 2 and xyz.shape[1] == 3 and nrm.shape == xyz.shape,
             "registered bowel positions or normals have invalid dimensions")
    _require(tri.ndim == 2 and tri.shape[1] == 3 and own.shape == (len(tri),),
             "registered bowel face-owner map has invalid dimensions")
    _require(np.isfinite(xyz).all() and np.isfinite(nrm).all(),
             "registered bowel mesh contains nonfinite coordinates")
    _require(len(tri) > 0 and int(tri.min()) >= 0 and int(tri.max()) < len(xyz),
             "registered bowel faces are out of bounds")
    allowed = set(stable_ids)
    _require(set(map(int, np.unique(own))) == allowed,
             "registered bowel source-owner IDs are incomplete or changed")
    result: dict[int, dict] = {}
    for stable_id in sorted(allowed):
        selected = tri[own == stable_id]
        _require(len(selected) > 0, f"registered bowel owner {stable_id} has no faces")
        used = np.unique(selected.reshape(-1))
        local = np.full(len(xyz), -1, dtype=np.int64)
        local[used] = np.arange(len(used), dtype=np.int64)
        local_faces = local[selected]
        _require(int(local_faces.min()) >= 0 and int(local_faces.max()) < len(used),
                 f"registered bowel owner {stable_id} has a nonlocal face")
        result[stable_id] = {
            "vertices6": np.column_stack((xyz[used], nrm[used])).astype("<f4"),
            "faces": local_faces.astype(np.int64),
        }
    return result


def _validate_registered_bowel_inputs(*, source_base_receipt: Path,
                                      bowel_source: Path, bowel_report: Path,
                                      colon_source: Path, registered_bowel: Path,
                                      registration_receipt: Path,
                                      protected_pair_audit: Path,
                                      whole_bowel_audit: Path,
                                      all_neighbor_audit: Path) -> tuple[dict, dict, dict, dict]:
    """Bind the exact B190 and 242 ancestry and their retained static audits."""
    source_receipt = json.loads(source_base_receipt.read_text())
    source_sha = _sha(Path(source_receipt["payload"]["path"]))
    _require(source_sha == source_receipt["payload"].get("sha256"),
             "B190 source receipt does not match its payload")
    source_receipt_sha = _sha(source_base_receipt)

    old_report = json.loads(bowel_report.read_text())
    _require(old_report.get("schema") == "numi.human.passive_descending_colon_exact_arrangement_candidate.v1",
             "unrecognized B190 geometry report")
    _require(old_report.get("input", {}).get("anatomy_sha256") == source_sha
             and old_report.get("input", {}).get("receipt_sha256") == source_receipt_sha,
             "B190 report is not derived from the supplied exact source receipt")
    bowel_sha = _sha(bowel_source)
    colon_sha = _sha(colon_source)
    _require(old_report.get("output_sha256", {}).get("bowel") == bowel_sha
             and old_report.get("output_sha256", {}).get("descending_colon") == colon_sha,
             "B190 report does not bind the supplied bowel/colon meshes")
    _require(old_report.get("cross_interface", {}).get("forbidden_outside_patch_pairs") == 0,
             "B190 bowel/colon interface report has forbidden outside-patch pairs")

    registration = json.loads(registration_receipt.read_text())
    registered_sha = _sha(registered_bowel)
    _require(registration.get("schema") == "numi.human.duodenum_liver_local_reference_registration.v1"
             and registration.get("status") == "static_source_registration_candidate_pending_native_respiratory_and_whole_scene_admission",
             "registered B242 candidate is missing its explicit static-only receipt")
    _require(registration.get("inputs", {}).get("source_bowel", {}).get("sha256") == bowel_sha
             and registration.get("inputs", {}).get("source_bowel", {}).get("derivation_report_sha256") == _sha(bowel_report),
             "B242 candidate does not bind the supplied B190 bowel and report")
    _require(registration.get("inputs", {}).get("candidate_positions", {}).get("sha256") == _sha(
        Path(registration["inputs"]["candidate_positions"]["path"])),
        "B242 position field input hash is stale")
    verification = registration.get("verification", {})
    _require(verification.get("exact_aggregate_self_forbidden_pairs") == 0
             and verification.get("exact_cross_pairs_against_closed_liver_ids18_19") == 0
             and verification.get("exact_protected_pair_sets_2_4_13_454_458_460_461") ==
             "identical to B190 source; zero added or removed pairs"
             and verification.get("all_metrics_are_static_source_geometry") is True,
             "B242 static geometry gates are missing or changed")
    metrics = registration.get("geometry", {})
    _require(metrics.get("faces_exactly_unchanged") is True
             and metrics.get("material_indices_unchanged") is True
             and metrics.get("stable_owner_ids_unchanged") is True
             and metrics.get("changed_shared_coordinate_groups") == 0
             and metrics.get("boundary_edges") == 0
             and metrics.get("nonmanifold_edges") == 0
             and metrics.get("zero_area_faces") == 0
             and metrics.get("orientation_inversions") == 0,
             "B242 candidate changed source topology or shared-owner coordinates")

    pair = json.loads(protected_pair_audit.read_text())
    _require(pair.get("status") == "exact_pair_set_comparison"
             and pair.get("inputs", {}).get("source_bowel_sha256") == bowel_sha
             and pair.get("inputs", {}).get("candidate_sha256") == registration["inputs"]["candidate_positions"]["sha256"],
             "protected-neighbor pair-set audit does not bind the B242 source-position candidate")
    _require(all(row.get("new_pairs") == 0 and row.get("removed_pairs") == 0
                 and row.get("source_pairs") == row.get("candidate_pairs")
                 for row in pair.get("pairs", []))
             and {row.get("stable_id") for row in pair.get("pairs", [])} == {2, 4, 13, 454, 458, 460, 461},
             "protected-neighbor pair sets are incomplete or changed")

    whole = json.loads(whole_bowel_audit.read_text())
    _require(whole.get("schema") == "numi.human.passive_bowel_registered_242_composite_audit.v1"
             and whole.get("inputs", {}).get("source_bowel", {}).get("sha256") == bowel_sha
             and whole.get("inputs", {}).get("registered_bowel", {}).get("sha256") == registered_sha
             and whole.get("inputs", {}).get("descending_colon", {}).get("sha256") == colon_sha,
             "whole-bowel source/registration interface audit is stale")
    _require(whole.get("topology", {}).get("registered", {}).get("closed_oriented_manifold_candidate") is True
             and whole.get("topology", {}).get("registered_forbidden_self_pairs") == 0
             and whole.get("topology", {}).get("source_forbidden_self_pairs") == 0
             and whole.get("descending_colon_455", {}).get("new_pairs") == 0
             and whole.get("descending_colon_455", {}).get("new398_pairs") == 0
             and whole.get("taenia_456", {}).get("new398_pairs") == 0,
             "whole-bowel/colon/taenia interface audit is incomplete or failed")

    neighbors = json.loads(all_neighbor_audit.read_text())
    _require(neighbors.get("schema") == "numi.human.duodenum_reference_registration_all_neighbor_pairset_audit.v1"
             and neighbors.get("inputs", {}).get("B190") == bowel_sha
             and neighbors.get("inputs", {}).get("candidate242") == registered_sha,
             "expanded all-neighbor pair-set audit is stale")
    pairsets = neighbors.get("pairsets", {})
    _require(set(pairsets) == {str(i) for i in neighbors.get("external_surface_ids", [])}
             and all(row.get("added") == 0 for row in pairsets.values()),
             "expanded all-neighbor audit is incomplete or introduces a pair")
    # The only intended changes in the exhaustive source-pair inventory are the
    # 590 liver-18/19 crossings removed by the 398 reference registration.
    _require(pairsets.get("18", {}).get("removed") == 407
             and pairsets.get("19", {}).get("removed") == 183
             and all(row.get("removed") == 0 for sid, row in pairsets.items() if sid not in {"18", "19"}),
             "expanded pair-set change does not match the retained exact witness result")
    return source_receipt, old_report, registration, {
        "protected": pair, "whole": whole, "neighbors": neighbors,
    }


def compose_bowel_reference_candidate(*, base_payload: Path, base_receipt: Path,
                                      source_base_receipt: Path, bowel_source: Path,
                                      bowel_report: Path, colon_source: Path,
                                      registered_bowel: Path, registration_receipt: Path,
                                      protected_pair_audit: Path, whole_bowel_audit: Path,
                                      all_neighbor_audit: Path, output: Path) -> dict:
    """Replace only existing passive bowel rows with source-bound B190/242 geometry.

    B190 provides the closed 56-zone aggregate and reciprocal descending-colon
    interface. B242 applies the small, inferred position-only correction to
    stable ID398 while retaining the exact B190 face ownership and all shared
    source-coordinate groups. This emits the existing NHANAT1 ABI5 payload;
    it adds no physiological state, force, mass, or organ compartment.
    """
    _require(not output.exists(), "retain existing output; choose a new directory")
    _require(base_payload.is_file() and base_receipt.is_file(), "current NHANAT payload or receipt is missing")
    source_receipt, source_report, registration, audits = _validate_registered_bowel_inputs(
        source_base_receipt=source_base_receipt, bowel_source=bowel_source,
        bowel_report=bowel_report, colon_source=colon_source,
        registered_bowel=registered_bowel, registration_receipt=registration_receipt,
        protected_pair_audit=protected_pair_audit, whole_bowel_audit=whole_bowel_audit,
        all_neighbor_audit=all_neighbor_audit)

    raw = base_payload.read_bytes()
    base_sha = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(base_receipt.read_text())
    _require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1"
             and receipt.get("payload", {}).get("sha256") == base_sha
             and receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == base_sha,
             "composition base receipt does not bind its exact NHANAT payload")
    header, records, rows = _parse_payload(raw)
    source_map = receipt.get("provenance", {}).get("source_id_map", {})
    passive = set(receipt.get("functional_bindings", {}).get("passive_viscera_geometry_binding", {}).get("stable_ids", []))
    target_ids = set(range(398, 454)) | {455}
    _require(target_ids.issubset(rows) and target_ids.issubset(passive)
             and all(str(sid) in source_map for sid in target_ids),
             "composition base lacks the existing source-owned bowel stable IDs")
    source_map_before = source_receipt.get("provenance", {}).get("source_id_map", {})
    for stable_id in range(398, 454):
        source = source_map_before.get(str(stable_id), {})
        current = source_map.get(str(stable_id), {})
        _require(current.get("source_member") == source.get("source_member")
                 and current.get("source_sha256") == source.get("source_sha256")
                 and current.get("source_owner_metadata", {}).get("member_id") == source.get("source_member"),
                 f"current base source identity changed for bowel owner {stable_id}")
        _require(rows[stable_id]["body_index"] == 20 and rows[stable_id]["layer"] == 1,
                 f"bowel owner {stable_id} is not in the shared passive torso frame")
    _require(rows[455]["body_index"] == 20 and rows[455]["layer"] == 1
             and source_map.get("455", {}).get("source_member") == "FJ2567",
             "descending-colon source owner 455 changed")

    with np.load(bowel_source, allow_pickle=False) as source, np.load(registered_bowel, allow_pickle=False) as candidate, np.load(colon_source, allow_pickle=False) as colon:
        required = {"vertices", "faces", "material_index", "material_owner_stable_id"}
        _require(required.issubset(source.files) and required.issubset(candidate.files),
                 "B190/B242 aggregate mesh arrays are incomplete")
        source_vertices = np.asarray(source["vertices"], dtype="<f4")
        source_faces = np.asarray(source["faces"], dtype=np.int32)
        source_materials = np.asarray(source["material_index"], dtype=np.int16)
        source_owners = np.asarray(source["material_owner_stable_id"], dtype=np.int32)
        candidate_vertices = np.asarray(candidate["vertices"], dtype="<f4")
        candidate_faces = np.asarray(candidate["faces"], dtype=np.int32)
        candidate_materials = np.asarray(candidate["material_index"], dtype=np.int16)
        candidate_owners = np.asarray(candidate["material_owner_stable_id"], dtype=np.int32)
        _require(source_faces.tobytes() == candidate_faces.tobytes()
                 and source_materials.tobytes() == candidate_materials.tobytes()
                 and source_owners.tobytes() == candidate_owners.tobytes(),
                 "B242 must preserve exact B190 faces, material indices, and owner identities")
        _require(source_vertices.shape == candidate_vertices.shape and source_vertices.ndim == 2
                 and source_vertices.shape[1] == 3 and np.isfinite(candidate_vertices).all(),
                 "B242 vertex buffer has invalid dimensions")
        _require(set(map(int, np.unique(source_owners))) == set(range(398, 454)),
                 "B190 mesh does not have exactly the 56 existing bowel owners")

        # Prove all cross-owner coordinate groups remain coincident after the
        # position-only update. This catches accidental per-ID field application.
        incident = [set() for _ in range(len(source_vertices))]
        for face, owner in zip(source_faces, source_owners):
            for vertex in face:
                incident[int(vertex)].add(int(owner))
        owner_by_coord: dict[bytes, set[int]] = {}
        candidate_by_coord: dict[bytes, bytes] = {}
        for index, point in enumerate(source_vertices):
            key = point.tobytes()
            owners_at_point = incident[index]
            owner_by_coord.setdefault(key, set()).update(owners_at_point)
            mapped = candidate_vertices[index].tobytes()
            prior = candidate_by_coord.setdefault(key, mapped)
            _require(prior == mapped,
                     "B242 moved exact shared source coordinates inconsistently across owner rows")

        if "vertex_normals" in candidate.files:
            reported_normals = np.asarray(candidate["vertex_normals"], dtype="<f4")
            _require(reported_normals.shape == candidate_vertices.shape and np.isfinite(reported_normals).all(),
                     "B242 aggregate vertex-normal field is malformed")
        aggregate_normals = _area_weighted_vertex_normals(candidate_vertices, candidate_faces)
        if "vertex_normals" in candidate.files:
            max_normal_delta = float(np.max(np.linalg.norm(
                reported_normals.astype(np.float64) - aggregate_normals.astype(np.float64), axis=1)))
            _require(max_normal_delta <= 1e-5,
                     "B242 normal field differs from independent aggregate area-weighted recomputation")
        else:
            max_normal_delta = None
        replacement_rows = _owned_surface_rows(candidate_vertices, aggregate_normals,
                                               candidate_faces, candidate_owners,
                                               range(398, 454))

        _require({"vertices", "faces"}.issubset(colon.files), "B190 colon455 mesh is incomplete")
        colon_vertices = np.asarray(colon["vertices"], dtype="<f4")
        colon_faces = np.asarray(colon["faces"], dtype=np.int64)
        colon_normals = _area_weighted_vertex_normals(colon_vertices, colon_faces)
        replacement_rows[455] = {
            "vertices6": np.column_stack((colon_vertices, colon_normals)).astype("<f4"),
            "faces": colon_faces,
        }

    updated = deepcopy(rows)
    for stable_id, replacement in replacement_rows.items():
        updated[stable_id]["vertices6"] = replacement["vertices6"]
        updated[stable_id]["faces"] = replacement["faces"]

    chunks: list[bytes] = []
    packed_vertices: list[bytes] = []
    packed_indices: list[bytes] = []
    nv = ni = 0
    for old_record in records:
        body_index, _vertex_start, _vertex_count, _index_start, _index_count, stable_id, layer, flags = map(int, old_record)
        row = updated[stable_id]
        local_vertices = np.asarray(row["vertices6"], dtype="<f4")
        local_faces = np.asarray(row["faces"], dtype=np.int64).reshape(-1, 3)
        _require(local_vertices.ndim == 2 and local_vertices.shape[1] == 6
                 and np.isfinite(local_vertices).all()
                 and (len(local_faces) == 0 or (int(local_faces.min()) >= 0 and int(local_faces.max()) < len(local_vertices))),
                 f"replacement surface {stable_id} is malformed")
        chunks.append(anatomy.RECORD.pack(body_index, nv, len(local_vertices), ni,
                                          local_faces.size, stable_id, layer, flags))
        packed_vertices.append(local_vertices.astype("<f4", copy=False).tobytes())
        packed_indices.append((local_faces.reshape(-1) + nv).astype("<u4").tobytes())
        nv += len(local_vertices)
        ni += local_faces.size
    result = anatomy.HEADER.pack(header[0], header[1], len(records), nv, ni,
                                 header[5], header[6]) + b"".join(chunks + packed_vertices + packed_indices)
    _, _, verified_rows = _parse_payload(result)
    for stable_id in rows:
        if stable_id not in replacement_rows:
            _require(_record_content_bytes(rows[stable_id]) == _record_content_bytes(verified_rows[stable_id]),
                     f"non-owned anatomical geometry changed at stable ID {stable_id}")
    for stable_id in replacement_rows:
        _require(verified_rows[stable_id]["body_index"] == rows[stable_id]["body_index"]
                 and verified_rows[stable_id]["layer"] == rows[stable_id]["layer"]
                 and verified_rows[stable_id]["flags"] == rows[stable_id]["flags"],
                 f"existing body/layer ownership changed at stable ID {stable_id}")

    output.mkdir(parents=True)
    payload_path = output / "resting-thorax.nhanatomy"
    payload_path.write_bytes(result)
    output_sha = hashlib.sha256(result).hexdigest()
    receipt = deepcopy(receipt)
    receipt["payload"].update(path=str(payload_path), sha256=output_sha,
                              surface_count=len(records), vertex_count=nv, index_count=ni)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    cardiac = receipt.get("provenance", {}).get("cardiac_geometry_binding", {})
    wall = cardiac.get("ventricular_wall_binding", {})
    wall_map = wall.get("wall_map", {})
    pending_refinement = wall_map.pop("local_coefficient_refinement", None)
    if pending_refinement is not None:
        # This source-bound v2 map must be re-emitted by its existing cardiac
        # owner after the final passive/respiratory composition. Keeping it in
        # the active wall map here would falsely bind it to an earlier complete
        # payload SHA and fail the native loader's source-identity gate.
        pending = {
            "schema": "numi.human.pending_cardiac_refinement_rebind.v1",
            "status": "inactive_pending_source_owner_rebind",
            "descriptor_sha256": hashlib.sha256(json.dumps(
                pending_refinement, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
            "source_payload_sha256": pending_refinement.get("source_payload_sha256"),
            "descriptor": deepcopy(pending_refinement),
            "deferred_by_owner": "numilab_human.resting_passive_viscera.compose_bowel_reference_candidate",
            "rebind_owner": "existing cardiac geometry binding owner after final anatomy composition",
            "rebind_required_for_native_admission": True,
        }
        prior_pending = receipt["provenance"].get("pending_cardiac_refinement_rebind")
        _require(prior_pending is None or prior_pending == pending,
                 "existing pending cardiac refinement provenance differs")
        receipt["provenance"]["pending_cardiac_refinement_rebind"] = pending
    if "output_anatomy_payload_sha256" in cardiac:
        cardiac["output_anatomy_payload_sha256"] = output_sha
    if "output_anatomy_payload_sha256" in wall:
        wall["output_anatomy_payload_sha256"] = output_sha
    prior_identity = receipt.get("provenance", {}).pop(PASSIVE_COMPONENT_IDENTITY_REFRESH, None)
    if prior_identity is not None and prior_identity.get("payload_sha256_unchanged") != output_sha:
        receipt["provenance"]["passive_component_identity_refresh_predecessor"] = prior_identity
    atlas = load_anatomy(anatomy.SOURCES)
    receipt = refresh_passive_component_identities(receipt, atlas)
    receipt.setdefault("provenance", {})["passive_bowel_reference_composition"] = {
        "schema": "numi.human.passive_bowel_registered_reference_composition.v1",
        "owner": "numilab_human.resting_passive_viscera.compose_bowel_reference_candidate",
        "source_payload_sha256": base_sha,
        "source_base_receipt_sha256": _sha(base_receipt),
        "B190_source_payload_sha256": _sha(Path(source_receipt["payload"]["path"])),
        "B190_source_receipt_sha256": _sha(source_base_receipt),
        "B190_report_sha256": _sha(bowel_report),
        "B190_bowel_sha256": _sha(bowel_source),
        "B190_descending_colon455_sha256": _sha(colon_source),
        "B242_registered_bowel_sha256": _sha(registered_bowel),
        "B242_registration_receipt_sha256": _sha(registration_receipt),
        "protected_pair_audit_sha256": _sha(protected_pair_audit),
        "whole_bowel_audit_sha256": _sha(whole_bowel_audit),
        "all_neighbor_pairset_audit_sha256": _sha(all_neighbor_audit),
        "replaced_stable_ids": sorted(replacement_rows),
        "small_bowel_representation": "one closed aggregate shell partitioned into 56 source-owned exterior surface patches; no independent zone volumes",
        "registration_interpretation": "inferred passive reference geometry correction for local duodenum-liver contact; not measured-subject geometry",
        "max_ID398_vertex_displacement_m": registration["geometry"]["touched_owner_metrics"]["398"]["max_displacement_m"],
        "registered_small_bowel_signed_volume_delta_ml": registration["geometry"]["signed_volume_delta_ml"],
        "max_recomputed_normal_delta_from_candidate": max_normal_delta,
        "shared_coordinate_groups_checked": len([1 for owners_at_point in owner_by_coord.values() if len(owners_at_point) > 1]),
        "shared_coordinate_groups_moved": registration["geometry"]["changed_shared_coordinate_groups"],
        "nonowned_surface_geometry_bytes_preserved": True,
        "additional_physical_mass_kg": 0,
        "additional_force_or_state": False,
        "source_license": "CC-BY-4.0",
        "qualification": "Static Float32 source geometry and recorded exact pair-set gates only; respiratory accepted-frame and native integrated-scene admission remain pending. The cardiac coefficient refinement is retained as inactive pending provenance and must be rebound to the final complete payload by its existing source owner.",
    }
    receipt["qualification"]["passive_bowel_reference_composition"] = receipt["provenance"]["passive_bowel_reference_composition"]["qualification"]
    if pending_refinement is not None:
        receipt["qualification"]["cardiac_refinement"] = "inactive pending re-emission by the existing cardiac geometry owner; this intermediate is not native-ready"
    receipt_path = output / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    manifest = {"schema": "numi.human.resting-anatomy-manifest.v1", "payload": receipt["payload"],
                "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)},
                "functional_bindings": receipt["functional_bindings"],
                "qualification": receipt["qualification"],
                "source_surfaces": receipt["provenance"]["source_id_map"],
                "mass_geometry_accounting": receipt["mass_geometry_accounting"],
                "passive_bowel_reference_composition": receipt["provenance"]["passive_bowel_reference_composition"]}
    (output / "resting-anatomy-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def compose_pylorus_stomach_arrangement(*, base_payload: Path, base_receipt: Path,
                                        arranged_stomach: Path, arrangement_report: Path,
                                        interface_audit: Path, neighbor_equivalence_audit: Path,
                                        neighbor_screen: Path, output: Path) -> dict:
    """Restore the source-bound arranged stomach required by the 398 interface.

    The previous bowel composition retained the raw 6,098-vertex stomach row,
    while its duodenum row already contained 639 reciprocal triangles from the
    exact pylorus arrangement.  This narrow follow-on installs the certified
    arranged stomach row and leaves every other NHANAT record byte-identical.
    The exact interface and neighboring-contact reports must bind the complete
    input payload and the exact arrangement asset; stale certificates fail.
    """
    _require(not output.exists(), "retain existing output; choose a new directory")
    for path in (base_payload, base_receipt, arranged_stomach, arrangement_report,
                 interface_audit, neighbor_equivalence_audit, neighbor_screen):
        _require(path.is_file(), f"pylorus arrangement dependency is missing: {path}")

    raw = base_payload.read_bytes()
    base_sha = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(base_receipt.read_text())
    _require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1"
             and receipt.get("payload", {}).get("sha256") == base_sha
             and receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == base_sha,
             "pylorus composition base receipt does not bind its exact NHANAT payload")
    header, records, rows = _parse_payload(raw)
    source_map = receipt.get("provenance", {}).get("source_id_map", {})
    stomach_owner = source_map.get("2", {})
    _require(2 in rows and 398 in rows
             and rows[2]["body_index"] == rows[398]["body_index"] == 20
             and rows[2]["layer"] == rows[398]["layer"] == 1
             and stomach_owner.get("source_member") == "FJ2564"
             and stomach_owner.get("source_owner_metadata", {}).get("concept_id") == "FMA7148",
             "pylorus arrangement requires the existing torso-frame stomach/duodenum source identities")

    arranged_sha = _sha(arranged_stomach)
    arrangement = json.loads(arrangement_report.read_text())
    _require(arrangement.get("schema") == "numi.human.pylorus_exact_pair_receipt.v1"
             and arrangement.get("status") == "static_source_candidate_not_native_qualified"
             and arrangement.get("outputs", {}).get("stomach", {}).get("sha256") == arranged_sha
             and arrangement.get("outputs", {}).get("stomach", {}).get("self_intersection_pair_count") == 0
             and arrangement.get("arrangement", {}).get("reciprocal_shared_patch_triangles") == 639,
             "pylorus arrangement receipt is stale or failed its source topology/interface gates")

    interface = json.loads(interface_audit.read_text())
    _require(interface.get("schema") == "numi.human.pylorus_current249_whole_patch_union_audit.v1"
             and interface.get("status") == "pass_static_exact_interface"
             and interface.get("payload_sha256") == base_sha
             and interface.get("stomach_row_sha256") == arranged_sha
             and interface.get("exact_cross_pairs") == 10539
             and interface.get("declared_patch_triangles") == 639
             and interface.get("current_zone398_retained_patch_triangles") == 639
             and interface.get("unexpected_count") == 0,
             "current payload has no complete exact arranged-stomach/duodenum patch certificate")
    neighbor_equivalence = json.loads(neighbor_equivalence_audit.read_text())
    _require(neighbor_equivalence.get("schema") == "numi.human.pylorus_stomach_neighbor_parent_contact_equivalence.v1"
             and neighbor_equivalence.get("payload_sha256") == base_sha
             and {row.get("neighbor_id") for row in neighbor_equivalence.get("neighbors", [])} == {15, 16, 311}
             and all(row.get("parent_pairs_added") == 0 and row.get("parent_pairs_removed") == 0
                     and row.get("unchanged_contact_geometry_parent_pairs") == row.get("parent_pairs_same")
                     for row in neighbor_equivalence["neighbors"]),
             "arranged stomach changes existing liver or diaphragm contact geometry")
    screen = json.loads(neighbor_screen.read_text())
    _require(screen.get("schema") == "numi.human.pylorus_arranged_stomach_neighbor_screen.v1"
             and screen.get("payload_sha256") == base_sha
             and {row.get("stable_id") for row in screen.get("pairs_with_crossings_or_degenerates", [])}
             == {15, 16, 311, 398},
             "arranged stomach neighbor screen found a changed/unclassified contact owner")

    with np.load(arranged_stomach, allow_pickle=False) as mesh:
        _require({"vertices", "faces", "source_face_index"}.issubset(mesh.files),
                 "arranged stomach is missing exact source-face lineage")
        vertices = np.asarray(mesh["vertices"], dtype="<f4")
        faces = np.asarray(mesh["faces"], dtype=np.int64)
        source_faces = np.asarray(mesh["source_face_index"], dtype=np.int64)
        _require(vertices.shape == (6258, 3) and faces.shape == (12512, 3)
                 and source_faces.shape == (len(faces),)
                 and np.isfinite(vertices).all()
                 and (faces.size == 0 or (int(faces.min()) >= 0 and int(faces.max()) < len(vertices)))
                 and (len(source_faces) == 0 or (int(source_faces.min()) >= 0
                                                  and int(source_faces.max()) < len(rows[2]["faces"]))),
                 "arranged stomach dimensions or parent-face lineage changed")
        normals = _area_weighted_vertex_normals(vertices, faces)
        replacement = np.column_stack((vertices, normals)).astype("<f4")
        _require(replacement.shape == (len(vertices), 6) and np.isfinite(replacement).all(),
                 "arranged stomach normal field is invalid")

    updated = deepcopy(rows)
    updated[2]["vertices6"] = replacement
    updated[2]["faces"] = faces
    chunks: list[bytes] = []
    packed_vertices: list[bytes] = []
    packed_indices: list[bytes] = []
    nv = ni = 0
    for old_record in records:
        body_index, _vertex_start, _vertex_count, _index_start, _index_count, stable_id, layer, flags = map(int, old_record)
        row = updated[stable_id]
        local_vertices = np.asarray(row["vertices6"], dtype="<f4")
        local_faces = np.asarray(row["faces"], dtype=np.int64).reshape(-1, 3)
        _require(local_vertices.ndim == 2 and local_vertices.shape[1] == 6
                 and np.isfinite(local_vertices).all()
                 and (len(local_faces) == 0 or (int(local_faces.min()) >= 0 and int(local_faces.max()) < len(local_vertices))),
                 f"replacement surface {stable_id} is malformed")
        chunks.append(anatomy.RECORD.pack(body_index, nv, len(local_vertices), ni,
                                          local_faces.size, stable_id, layer, flags))
        packed_vertices.append(local_vertices.astype("<f4", copy=False).tobytes())
        packed_indices.append((local_faces.reshape(-1) + nv).astype("<u4").tobytes())
        nv += len(local_vertices)
        ni += local_faces.size
    result = anatomy.HEADER.pack(header[0], header[1], len(records), nv, ni,
                                 header[5], header[6]) + b"".join(chunks + packed_vertices + packed_indices)
    _, _, verified_rows = _parse_payload(result)
    for stable_id in rows:
        if stable_id != 2:
            _require(_record_content_bytes(rows[stable_id]) == _record_content_bytes(verified_rows[stable_id]),
                     f"non-owned anatomical geometry changed at stable ID {stable_id}")
    _require(verified_rows[2]["body_index"] == rows[2]["body_index"]
             and verified_rows[2]["layer"] == rows[2]["layer"]
             and verified_rows[2]["flags"] == rows[2]["flags"],
             "existing stomach body/layer ownership changed")

    output.mkdir(parents=True)
    payload_path = output / "resting-thorax.nhanatomy"
    payload_path.write_bytes(result)
    output_sha = hashlib.sha256(result).hexdigest()
    receipt = deepcopy(receipt)
    receipt["payload"].update(path=str(payload_path), sha256=output_sha,
                              surface_count=len(records), vertex_count=nv, index_count=ni)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    cardiac = receipt.get("provenance", {}).get("cardiac_geometry_binding", {})
    wall = cardiac.get("ventricular_wall_binding", {})
    wall_map = wall.get("wall_map", {})
    pending_refinement = wall_map.pop("local_coefficient_refinement", None)
    if pending_refinement is not None:
        pending = {
            "schema": "numi.human.pending_cardiac_refinement_rebind.v1",
            "status": "inactive_pending_source_owner_rebind",
            "descriptor_sha256": hashlib.sha256(json.dumps(
                pending_refinement, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
            "source_payload_sha256": pending_refinement.get("source_payload_sha256"),
            "descriptor": deepcopy(pending_refinement),
            "deferred_by_owner": "numilab_human.resting_passive_viscera.compose_pylorus_stomach_arrangement",
            "rebind_owner": "existing cardiac geometry binding owner after final anatomy composition",
            "rebind_required_for_native_admission": True,
        }
        prior_pending = receipt["provenance"].get("pending_cardiac_refinement_rebind")
        _require(prior_pending is None or prior_pending == pending,
                 "existing pending cardiac refinement provenance differs")
        receipt["provenance"]["pending_cardiac_refinement_rebind"] = pending
    if "output_anatomy_payload_sha256" in cardiac:
        cardiac["output_anatomy_payload_sha256"] = output_sha
    if "output_anatomy_payload_sha256" in wall:
        wall["output_anatomy_payload_sha256"] = output_sha
    source_identity = receipt["provenance"]["source_id_map"]["2"]
    receipt.setdefault("provenance", {})["pylorus_stomach_arrangement"] = {
        "schema": "numi.human.pylorus_stomach_registered_arrangement.v1",
        "owner": "numilab_human.resting_passive_viscera.compose_pylorus_stomach_arrangement",
        "replaced_stable_ids": [2],
        "source_member": source_identity["source_member"],
        "source_member_sha256": source_identity["source_sha256"],
        "source_concept_id": "FMA7148",
        "arranged_stomach_sha256": arranged_sha,
        "arrangement_report_sha256": _sha(arrangement_report),
        "current_interface_audit_sha256": _sha(interface_audit),
        "neighbor_equivalence_audit_sha256": _sha(neighbor_equivalence_audit),
        "neighbor_screen_sha256": _sha(neighbor_screen),
        "reciprocal_pylorus_patch_triangles": 639,
        "current_exact_interface_pairs": 10539,
        "current_unexpected_patch_pairs": 0,
        "changed_neighbor_contact_geometry": False,
        "source_face_lineage_preserved": True,
        "interpretation": "inferred passive stomach-to-duodenum transition near pylorus; no measured pyloric plane or independent GI dynamics",
        "additional_physical_mass_kg": 0,
        "additional_force_or_state": False,
        "source_license": "CC-BY-4.0",
        "qualification": "static Float32 interface and neighboring contact certificates only; no accepted-breath or native integrated-scene qualification. Cardiac refinement requires source-owner rebind after final payload composition.",
    }
    receipt.setdefault("qualification", {})["pylorus_stomach_arrangement"] = receipt["provenance"]["pylorus_stomach_arrangement"]["qualification"]
    if pending_refinement is not None:
        receipt["qualification"]["cardiac_refinement"] = "inactive pending re-emission by the existing cardiac geometry owner; this intermediate is not native-ready"
    receipt_path = output / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    manifest = {"schema": "numi.human.resting-anatomy-manifest.v1", "payload": receipt["payload"],
                "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)},
                "functional_bindings": receipt["functional_bindings"],
                "qualification": receipt["qualification"],
                "source_surfaces": receipt["provenance"]["source_id_map"],
                "mass_geometry_accounting": receipt["mass_geometry_accounting"],
                "pylorus_stomach_arrangement": receipt["provenance"]["pylorus_stomach_arrangement"]}
    (output / "resting-anatomy-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest



def _validated_torso_rest_body(receipt: dict, registration: dict,
                               rigid_path: Path) -> dict[str, dict]:
    """Derive the one source body used by the vein append from pinned NHRIGID2.

    The full-body source manifest is intentionally not synthesized. This
    narrow path verifies the existing receipt-pinned rigid payload, its
    embedded MyoSim source identity, source-body 8 -> core-body 20 mapping,
    and the receipt's recorded torso rest transform.
    """
    rigid_path = Path(rigid_path)
    expected_rigid = receipt.get("provenance", {}).get("rigid_payload_sha256")
    if not isinstance(expected_rigid, str) or human.sha256(rigid_path) != expected_rigid:
        raise ValueError("brachiocephalic source: NHRIGID2 differs from the base receipt")
    rigid = load_rigid(rigid_path)
    provenance = receipt.get("provenance", {})
    registered_source = (registration.get("source", {}).get("myosim", {})
                         .get("source", {}).get("archive_sha256"))
    if (not isinstance(registered_source, str)
            or rigid["source_sha256"] != registered_source
            or provenance.get("myosim_source_archive_sha256") != registered_source):
        raise ValueError("brachiocephalic source: NHRIGID2 and registration source archive differ")
    source_map = rigid.get("source_map")
    if (not isinstance(source_map, list) or len(source_map) != rigid.get("source_count")
            or len(source_map) <= 8 or source_map[8] != 20 or source_map.count(20) != 1
            or 20 not in rigid.get("core_poses", {})):
        raise ValueError("brachiocephalic source: pinned source-body 8 to core-body 20 mapping changed")
    pose = np.asarray(rigid["core_poses"][20], dtype=np.float64)
    expected_position = np.asarray(provenance.get("torso_rest_position_world_m", []), dtype=np.float64)
    expected_orientation = np.asarray(provenance.get("torso_rest_orientation_xyzw", []), dtype=np.float64)
    if (pose.shape != (7,) or expected_position.shape != (3,) or expected_orientation.shape != (4,)
            or not np.isfinite(pose).all() or not np.isfinite(expected_position).all()
            or not np.isfinite(expected_orientation).all()
            or not np.allclose(pose[:3], expected_position, rtol=0.0, atol=1.0e-8)
            or not np.allclose(pose[3:], expected_orientation, rtol=0.0, atol=1.0e-8)):
        raise ValueError("brachiocephalic source: NHRIGID2 torso rest pose differs from receipt")
    base_row = provenance.get("source_id_map", {}).get("10", {})
    owner = base_row.get("source_owner_metadata", {})
    if (base_row.get("body_index") != 20 or owner.get("core_body_index") != 20
            or owner.get("myosim_body") != "torso"):
        raise ValueError("brachiocephalic source: existing torso-owned SVC row changed")
    return {"torso": {
        "core_body_index": 20,
        "source_body_id": 8,
        "default_com_position_world_m": pose[:3].tolist(),
        "default_inertial_quaternion_world_xyzw": pose[3:].tolist(),
    }}


def _exact_float32_seam_quotient(local_vertices: np.ndarray, faces: np.ndarray,
                                 source_vertices_mm: np.ndarray,
                                 member_id: str) -> dict[str, np.ndarray]:
    """Collapse exact serialized seam duplicates while retaining face lineage."""
    local = np.asarray(local_vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    source = np.asarray(source_vertices_mm, dtype=np.float64)
    if (local.ndim != 2 or local.shape[1] != 3 or source.shape != local.shape
            or tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0
            or not np.isfinite(local).all() or not np.isfinite(source).all()
            or int(tri.min()) < 0 or int(tri.max()) >= len(local)):
        raise ValueError("brachiocephalic source: malformed compiled mesh " + member_id)
    native_xyz = local.astype("<f4")
    _, first, inverse = np.unique(native_xyz, axis=0, return_index=True, return_inverse=True)
    first_order = np.argsort(first, kind="stable")
    first = first[first_order]
    remap = np.empty(len(first), dtype=np.int64)
    remap[first_order] = np.arange(len(first), dtype=np.int64)
    lift = remap[inverse]
    for quotient_id in range(len(first)):
        source_ids = np.flatnonzero(lift == quotient_id)
        if not np.all(source[source_ids] == source[source_ids[0]]):
            raise ValueError("brachiocephalic source: Float32 quotient would merge distinct source coordinates " + member_id)
    quotient_faces = lift[tri]
    if (len(first) < 3 or np.any(quotient_faces[:, 0] == quotient_faces[:, 1])
            or np.any(quotient_faces[:, 1] == quotient_faces[:, 2])
            or np.any(quotient_faces[:, 2] == quotient_faces[:, 0])):
        raise ValueError("brachiocephalic source: quotient collapsed a source face " + member_id)
    quotient_xyz = native_xyz[first]
    normals = _area_weighted_vertex_normals(quotient_xyz, quotient_faces)
    return {
        "vertices": np.column_stack((quotient_xyz, normals)).astype("<f4"),
        "faces": quotient_faces.astype("<u4"),
        "source_vertex_to_quotient": lift.astype("<u4"),
        "source_face_ids": np.arange(len(tri), dtype="<u4"),
    }


def _require_brachiocephalic_source_coverage(pins: dict, *, rigid_path: Path,
                                                sources_path: Path) -> None:
    """Require source-frame inputs, not the historical producer parent path."""
    pinned_paths = {Path(name).resolve() for name in pins}
    required_sources = {
        Path(rigid_path).resolve(),
        (Path(sources_path) / "partof_BP3D_4.0_obj_99.zip").resolve(),
    }
    if not required_sources.issubset(pinned_paths):
        raise ValueError("brachiocephalic source: explicit rigid/archive inputs differ from quotient report")


def _load_brachiocephalic_quotient_rows(*, quotient_npz_path: Path,
                                         quotient_report_path: Path,
                                         rigid_path: Path,
                                         sources_path: Path) -> tuple[dict[str, dict], dict]:
    """Load the pinned seam-quotient artifact, without treating it as admission."""
    report_path = Path(quotient_report_path)
    npz_path = Path(quotient_npz_path)
    expected_report_sha = "f189f18efcbfa2953374caf5c3b11af90b135f972764713953de3360bbe140da"
    expected_npz_sha = "02df2dba7d698d8e22ca7b6493d1b0b97d1afa00b06150b39d1e0a7990e3e385"
    if human.sha256(report_path) != expected_report_sha or human.sha256(npz_path) != expected_npz_sha:
        raise ValueError("brachiocephalic source: pinned quotient report or NPZ hash changed")
    report = json.loads(report_path.read_text())
    if (report.get("prepared_npz_sha256") != expected_npz_sha
            or report.get("native_admission") is not False
            or "junction_probe_not_nonintersection_gate" not in {
                row.get("classification") for row in report.get("default_source_junction_probes", [])
            }):
        raise ValueError("brachiocephalic source: quotient report scope or identity changed")
    pins = report.get("input_sha256")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("brachiocephalic source: quotient report lacks pinned input hashes")
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_file() or human.sha256(path) != expected:
            raise ValueError("brachiocephalic source: quotient input hash changed: " + str(path))
    # The report is permanently bound to its original N27 producer parent.
    # A later append parent is independently validated by append_viscera's
    # receipt/payload, registration, NHRIGID, source archive, and torso-owner
    # gates; its path is not expected in this historical quotient report.
    _require_brachiocephalic_source_coverage(
        pins, rigid_path=rigid_path, sources_path=sources_path)
    expected_rows = {
        "FJ3482": (377, 376, 748, [[4, 376]]),
        "FJ3583": (275, 275, 546, []),
    }
    rows = report.get("rows")
    if (not isinstance(rows, list) or len(rows) != len(expected_rows)
            or {row.get("member") for row in rows} != set(expected_rows)):
        raise ValueError("brachiocephalic source: quotient report does not contain both named veins")
    for row in rows:
        source_count, quotient_count, face_count, duplicate_groups = expected_rows[row["member"]]
        if (row.get("source_vertices"), row.get("quotient_vertices"), row.get("faces"),
                row.get("duplicate_source_groups")) != (
                source_count, quotient_count, face_count, duplicate_groups):
            raise ValueError("brachiocephalic source: quotient row dimensions changed for " + row["member"])
        if (row.get("source_displacement_m") != 0 or row.get("source_triangles_preserved_geometrically") is not True
                or row.get("closed_oriented") is not True or row.get("euler_characteristic") != 2
                or row.get("exact_self_pairs") != 0):
            raise ValueError("brachiocephalic source: quotient topology/source identity changed for " + row["member"])
    loaded = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        expected_names = {
            f"{member}_{suffix}"
            for member in expected_rows
            for suffix in ("vertices", "faces", "source_vertex_to_quotient", "source_face_ids")
        }
        if set(archive.files) != expected_names:
            raise ValueError("brachiocephalic source: quotient NPZ array inventory changed")
        for member in expected_rows:
            item = {suffix: np.asarray(archive[f"{member}_{suffix}"]).copy()
                    for suffix in ("vertices", "faces", "source_vertex_to_quotient", "source_face_ids")}
            if (item["vertices"].dtype != np.dtype("<f4") or item["vertices"].ndim != 2
                    or item["vertices"].shape[1] != 6 or not np.isfinite(item["vertices"]).all()
                    or item["faces"].dtype.kind != "u" or item["faces"].ndim != 2
                    or item["faces"].shape[1] != 3 or len(item["faces"]) == 0
                    or int(item["faces"].max()) >= len(item["vertices"])
                    or item["source_vertex_to_quotient"].dtype.kind != "u"
                    or item["source_face_ids"].dtype.kind != "u"):
                raise ValueError("brachiocephalic source: malformed quotient NPZ row " + member)
            loaded[member] = item
    return loaded, report


def append_viscera(base_receipt: Path, output: Path, *, families=FAMILIES,
                   hierarchy="part_of", layer=1, extension_key="passive_viscera_extension",
                   compartment_by_name=None, members_by_name=None,
                   sources_path: Path | None = None,
                   registration_path: Path | None = None, rigid_path: Path | None = None,
                   quotient_npz_path: Path | None = None,
                   quotient_report_path: Path | None = None,
                   local_intercostal_arrangement_binding: dict | None = None) -> dict:
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
    sources = Path(anatomy.SOURCES if sources_path is None else sources_path).resolve()
    registration_file = Path(anatomy.REGISTRATION_PATH if registration_path is None else registration_path)
    registration = json.loads(registration_file.read_text())
    registration_sha = human.sha256(registration_file)
    if registration_sha != receipt["provenance"]["bodyparts_registration_sha256"] or int(registration_sha[:8], 16) != registration_fp:
        raise ValueError("passive anatomy registration differs from the existing body")
    for entry in receipt["provenance"]["bodyparts3d_archives"]:
        archive = sources / entry["file"]
        # An explicit narrow source directory may contain only the archive for
        # the selected hierarchy. Both relationship tables are still checked
        # by load_anatomy against sources.lock.json below.
        if entry.get("hierarchy") == hierarchy or archive.is_file() or sources_path is None:
            if not archive.is_file() or human.sha256(archive) != entry["sha256"]:
                raise ValueError("pinned BodyParts3D archive changed: " + entry["file"])
    rigid_file = Path(anatomy.RIGID_PATH if rigid_path is None else rigid_path)
    if human.sha256(rigid_file) != receipt["provenance"]["rigid_payload_sha256"]:
        raise ValueError("passive anatomy rigid source identity differs")
    if rigid_path is None:
        bodies, _, _ = human._myosim_surface_route_context(
            rigid_file.parent, registration["source"]["myosim"]["source"]["archive_sha256"])
    else:
        bodies = _validated_torso_rest_body(receipt, registration, rigid_file)
    atlas = load_anatomy(sources)
    source_map = receipt["provenance"]["source_id_map"]
    existing_members = {v["source_member"]: int(k) for k, v in source_map.items()}
    base_header, base_records, base_rows = _parse_payload(raw)
    decoded_ids = set(base_rows)
    mapped_ids = {int(stable_id) for stable_id in source_map}
    if len(decoded_ids) != count or decoded_ids != mapped_ids:
        raise ValueError("passive anatomy decoded rows and source_id_map differ")
    next_id = max(decoded_ids) + 1
    arrangement_rows = {}
    arrangement_summaries = {}
    arrangement_row_records = []
    arrangement_context = None
    arrangement_module_sha = None
    append_owner_sha = None
    if local_intercostal_arrangement_binding is not None:
        if not (extension_key == "passive_brachiocephalic_vein_extension"
                and receipt["payload"]["sha256"]
                == "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"):
            raise ValueError("local intercostal arrangement is restricted to the direct pinned N27 vein append")
        from . import resting_intercostal_mesh_arrangement as intercostal
        arrangement_context = intercostal.load_pinned_fragment_repair(
            base_receipt_path=base_receipt,
            base_payload_sha256=receipt["payload"]["sha256"],
            repair_parent_receipt_path=Path(
                local_intercostal_arrangement_binding["repair_parent_receipt_path"]),
            repair_parent_receipt_sha256=local_intercostal_arrangement_binding[
                "repair_parent_receipt_sha256"],
            repair_parent_payload_path=Path(
                local_intercostal_arrangement_binding["repair_parent_payload_path"]),
            repair_report_path=Path(
                local_intercostal_arrangement_binding["repair_report_path"]),
            repair_report_sha256=local_intercostal_arrangement_binding[
                "repair_report_sha256"],
        )
        arrangement_descriptors = {
            int(stable_id): descriptor
            for stable_id, descriptor in local_intercostal_arrangement_binding["rows"].items()
        }
        arrangement_rows, arrangement_summaries = intercostal.compose_pinned_arrangement_set(
            base_rows=base_rows,
            repair_context=arrangement_context,
            arrangement_descriptors=arrangement_descriptors,
        )
        arrangement_module_sha = human.sha256(Path(intercostal.__file__))
        append_owner_sha = human.sha256(Path(__file__))
        for stable_id in sorted(arrangement_rows):
            descriptor = arrangement_descriptors[stable_id]
            summary = arrangement_summaries[str(stable_id)]
            row_binding = {
                "stable_id": stable_id,
                "method": "existing ABI5 append owner; exact 028 retained-face parent plus positive full_candidate parent maps",
                "full_candidate": {
                    key: str(value) if key.endswith("_path") else value
                    for key, value in descriptor.items()
                },
                "repair_parent_receipt_sha256": arrangement_context["repair_parent_receipt_sha256"],
                "repair_parent_payload_sha256": arrangement_context["repair_parent_payload_sha256"],
                "fragment_repair_report_sha256": arrangement_context["repair_report_sha256"],
                "compiled_row_sha256": summary["compiled_row_sha256"],
                "retained_n27_face_map_sha256": summary["retained_n27_face_map_sha256"],
                "retained_n27_source_face_count": summary["retained_n27_source_face_count"],
                "new_appended_vertex_count": summary["new_appended_vertex_count"],
                "full_candidate_patch_face_count": summary["full_candidate_patch_face_count"],
                "positive_exact_parent_map_for_each_appended_vertex": True,
                "append_owner_source_sha256": append_owner_sha,
                "arrangement_composer_source_sha256": arrangement_module_sha,
                "physical_mass_or_state_added": False,
            }
            arrangement_row_records.append(row_binding)
            map_row = source_map[str(stable_id)]
            prior_repair = arrangement_context["repair_parent_source_id_map"][
                str(stable_id)].get("repair")
            if prior_repair is None:
                prior_repair = {}
            if not isinstance(prior_repair, dict):
                raise ValueError("intercostal source map repair metadata is not an object")
            prior_repair = deepcopy(prior_repair)
            if "local_intersection_remesh" in prior_repair:
                raise ValueError("intercostal source map already has a local remesh repair")
            prior_repair["local_intersection_remesh"] = row_binding
            map_row["repair"] = prior_repair
        if arrangement_context["unrelated_n27_rows_byte_exact"] is not True:
            raise ValueError("pinned 028 repair did not preserve unrelated N27 rows")
    records, vertices, indices = (
        [raw[anatomy.HEADER.size:vo]], [raw[vo:io]], [raw[io:]])
    if rigid_path is not None and (quotient_npz_path is None or quotient_report_path is None):
        raise ValueError("brachiocephalic source requires both pinned quotient inputs")
    if (quotient_npz_path is None) != (quotient_report_path is None):
        raise ValueError("passive source quotient requires both NPZ and report inputs")
    quotient_rows = None
    quotient_report = None
    if quotient_npz_path is not None:
        expected_families = (
            ("FMA4761", "left brachiocephalic vein", "torso", 1),
            ("FMA4751", "right brachiocephalic vein", "torso", 1),
        )
        expected_members = {
            "left brachiocephalic vein": {"FJ3482"},
            "right brachiocephalic vein": {"FJ3583"},
        }
        expected_compartments = {
            "left brachiocephalic vein": 4,
            "right brachiocephalic vein": 4,
        }
        if (hierarchy != "part_of" or layer != 2 or tuple(families) != expected_families
                or members_by_name != expected_members or compartment_by_name != expected_compartments
                or extension_key != "passive_brachiocephalic_vein_extension"):
            raise ValueError("pinned quotient is restricted to the exact left/right torso vein source rows")
        quotient_rows, quotient_report = _load_brachiocephalic_quotient_rows(
            quotient_npz_path=quotient_npz_path, quotient_report_path=quotient_report_path,
            rigid_path=rigid_file, sources_path=sources)
    additions = []
    added_members = {}
    family_rows = []
    for concept, label, body, expected_count in families:
        concept_members = atlas["tables"][hierarchy].get((concept, label), set())
        members = (set(concept_members) if members_by_name is None
                   else set(members_by_name.get(label, ())))
        if (len(members) != expected_count or not members
                or not members.issubset(concept_members)):
            raise ValueError("passive source member selection changed: " + label)
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
            _, member, obj = human._bodyparts_obj_member(sources, hierarchy, member_id)
            digest = hashlib.sha256(obj).hexdigest()
            # Rectum follows the existing pelvic rigid frame; the remaining
            # large intestine and small intestine follow the abdominal frame.
            target = "pelvis" if member_id == "FJ2571" else body
            spec = {"hierarchy": hierarchy, "member_id": member_id,
                    "source_member_sha256": digest, "myosim_body": target}
            owner, source_body, local, normals, faces = compiled_surface(sources, spec, registration, bodies)
            packed = np.column_stack((local, normals)).astype("<f4")
            repair = None
            if quotient_rows is not None:
                prepared = quotient_rows.get(member_id)
                if prepared is None:
                    raise ValueError("brachiocephalic source quotient lacks " + member_id)
                source_vertices_mm, source_faces = human._bodyparts_obj_triangles(obj, member)
                source_vertices_mm = np.asarray(source_vertices_mm, dtype=np.float64)
                if not np.array_equal(np.asarray(source_faces, dtype=np.int64), np.asarray(faces, dtype=np.int64)):
                    raise ValueError("brachiocephalic source: compiled face order changed for " + member_id)
                recomputed = _exact_float32_seam_quotient(
                    local, faces, source_vertices_mm, member_id)
                for field in ("vertices", "faces", "source_vertex_to_quotient", "source_face_ids"):
                    if recomputed[field].tobytes() != prepared[field].tobytes():
                        raise ValueError("brachiocephalic source quotient does not match compiled member: "
                                         + member_id + "/" + field)
                packed = recomputed["vertices"]
                faces = recomputed["faces"]
                local = packed[:, :3]
                normals = packed[:, 3:]
                repair = {
                    "kind": "exact_float32_source_seam_quotient",
                    "source_vertex_count": int(len(source_vertices_mm)),
                    "quotient_vertex_count": int(len(local)),
                    "source_face_count": int(len(source_faces)),
                    "face_lineage": "all source faces retained in source order through source_vertex_to_quotient",
                    "moved_source_coordinates_m": 0,
                    "non_source_coordinate_added": False,
                }
            if not np.isfinite(packed).all() or faces.min() < 0 or faces.max() >= len(local):
                raise ValueError("invalid registered passive surface")
            records.append(anatomy.RECORD.pack(owner, nv, len(local), ni, faces.size, next_id, layer, 0))
            vertices.append(packed.tobytes()); indices.append((faces.ravel() + nv).astype("<u4").tobytes())
            component = anatomical_component_identity(atlas, member_id)
            item = {"stable_id": next_id, "name": component["name"] if component else label,
                    "family_label": label, "concept_id": concept,
                    "source_member": member_id, "source_member_path": member,
                    "source_sha256": digest, "hierarchy": hierarchy, "body_index": owner,
                    "myosim_body": target,
                    "vertex_count": len(local), "triangle_count": len(faces),
                    "local_bounds_m": [local.min(axis=0).tolist(), local.max(axis=0).tolist()]}
            if repair is not None:
                item["source_body_id"] = source_body
                item["geometry_repair"] = repair
            if component is not None:
                item["anatomical_component"] = component
            additions.append(item)
            source_map[str(next_id)] = {"name": component["name"] if component else label,
                "provider": "BodyParts3D v4.0 existing organ-family compiler",
                "source_member": member_id, "source_sha256": digest, "layer": layer, "body_index": owner,
                "repair": repair, "head_family": None,
                "source_owner_metadata": {"concept_id": concept, "hierarchy": hierarchy, "label": label,
                                          "member_id": member_id, "member_sha256": digest,
                                          "myosim_body": target, "core_body_index": owner,
                                          "layer": "vessel" if layer == 2 else "organ"}}
            if repair is not None:
                source_map[str(next_id)]["source_owner_metadata"]["source_body_id"] = source_body
            if component is not None:
                source_map[str(next_id)]["source_owner_metadata"]["anatomical_component"] = component
            if compartment_by_name is not None:
                source_map[str(next_id)]["vascular_compartment_id"] = compartment_by_name[label]
            added_members[member_id] = next_id; family_ids.append(next_id)
            next_id += 1; nv += len(local); ni += faces.size
        family_row = {"concept_id": concept, "name": label,
                      "source_members": sorted(members), "stable_ids": family_ids}
        if members_by_name is not None:
            family_row["concept_member_count"] = len(concept_members)
            family_row["member_selection"] = "explicit source member allowlist"
        family_rows.append(family_row)
    payload = anatomy.HEADER.pack(magic, abi, count + len(additions), nv, ni, registration_fp, source_sha)
    payload += b"".join(records) + b"".join(vertices) + b"".join(indices)
    if arrangement_rows:
        from .resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
        payload, reemit_proof = encode_payload_with_row_replacements(payload, arrangement_rows)
        if reemit_proof["replaced_stable_ids"] != sorted(arrangement_rows):
            raise ValueError("existing ABI5 row-replacement proof differs from requested IDs")
    _, _, appended_rows = _parse_payload(payload)
    for stable_id in base_rows:
        if stable_id not in arrangement_rows and _record_content_bytes(base_rows[stable_id]) != _record_content_bytes(appended_rows[stable_id]):
            raise ValueError("passive append changed an unmodified N27 anatomy row: " + str(stable_id))
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
        "source_component_table_sha256": human.sha256(sources / "partof_element_parts.txt"),
        "preserved_base_record_bytes_sha256": hashlib.sha256(raw[anatomy.HEADER.size:vo]).hexdigest(),
        "preserved_base_vertex_bytes_sha256": hashlib.sha256(raw[vo:io]).hexdigest(),
        "preserved_base_index_bytes_sha256": hashlib.sha256(raw[io:]).hexdigest(),
        "geometry_repair_applied": quotient_report is not None,
        "additional_physical_mass_kg": 0,
        "license": "CC-BY-4.0",
        "scope": "passive atlas inspection with existing body-frame registration; no added constitutive tissue or vessel-wall mechanics; source interfaces not yet qualified under loaded motion"}
    if arrangement_rows:
        receipt["provenance"]["local_intercostal_source_fragment_repair"] = deepcopy(
            arrangement_context["fragment_repair_declaration"])
        receipt["provenance"]["local_intercostal_mesh_repair"] = {
            "schema": "numi.human.local_intercostal_mesh_repair.v1",
            "method": "documented local arrangement remesh composed through the existing NHANAT ABI5 append owner",
            "base_receipt_path": str(Path(base_receipt).resolve()),
            "base_receipt_sha256": human.sha256(Path(base_receipt)),
            "base_payload_sha256": hashlib.sha256(raw).hexdigest(),
            "repair_parent_receipt_path": arrangement_context["repair_parent_receipt_path"],
            "repair_parent_receipt_sha256": arrangement_context["repair_parent_receipt_sha256"],
            "repair_parent_payload_path": arrangement_context["repair_parent_payload_path"],
            "repair_parent_payload_sha256": arrangement_context["repair_parent_payload_sha256"],
            "repair_report_path": arrangement_context["repair_report_path"],
            "repair_report_sha256": arrangement_context["repair_report_sha256"],
            "rows": arrangement_row_records,
            "abi5_payload_reemit_proof": reemit_proof,
            "unrelated_n27_rows_byte_exact": True,
            "unchanged_original_vertex_slots_preserved": True,
            "old_original_faces_preserved_or_explicitly_reported": True,
            "all_028_retained_faces_preserved": True,
            "no_physical_mass_or_state_added": True,
            "native_confirmation_required": True,
            "anatomical_admission": False,
        }
    if quotient_report is not None:
        receipt["provenance"][extension_key].update({
            "source_quotient_report": {
                "path": str(Path(quotient_report_path)),
                "sha256": human.sha256(Path(quotient_report_path)),
                "npz_path": str(Path(quotient_npz_path)),
                "npz_sha256": human.sha256(Path(quotient_npz_path)),
                "native_admission": False,
                "junction_probe_classification": "junction_probe_not_nonintersection_gate",
                "pinned_input_sha256": quotient_report.get("input_sha256"),
            },
            "verified_registration_sha256": registration_sha,
            "verified_registration_path": str(registration_file),
            "verified_rigid_payload_sha256": human.sha256(rigid_file),
            "verified_rigid_payload_path": str(rigid_file),
            "verified_rest_body_context": {
                "body_name": "torso",
                "source_body_id": int(bodies["torso"]["source_body_id"]),
                "core_body_index": int(bodies["torso"]["core_body_index"]),
                "default_com_position_world_m": bodies["torso"]["default_com_position_world_m"],
                "default_inertial_quaternion_world_xyzw": bodies["torso"]["default_inertial_quaternion_world_xyzw"],
                "validated_against_receipt_torso_rest": True,
            },
            "additional_force_or_state": False,
            "scope": "passive atlas vessel identities compiled by the existing NHANAT ABI5 append owner; no added CVSim state, blood volume, physical mass, constitutive tissue, or vessel-wall mechanics; source junction probes do not establish route collars, nonintersection, shared cardiac deformation, accepted poses, or native admission",
        })
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
