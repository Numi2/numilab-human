"""Materialize the cumulative Human source target union, without promoting physics.

This is an offline source inventory, not a HumanPack runtime or an evidence
validator. A declaration's presence says nothing about whether its equations,
material, control, sensing, or physical validation have been implemented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from .model import ImportError as HumanImportError


SCHEMA = "HumanPack.target-coverage.v1"
COMPILER_VERSION = "numilab-human.target-coverage.1"
ROOT = Path(__file__).resolve().parents[2]

# These are the mandatory rows of DEVELOPMENT_ROADMAP.md, expanded rather than
# interpreted as a runtime-dependent subset. Source adapters can only add leaves.
MANDATORY = {
    "skeletal_topology": (
        "axial_skeleton", "appendicular_skeleton", "cervical_system", "hyoid_system",
        "craniofacial_system", "articulated_hands", "intrinsic_fingers", "articulated_feet",
        "intrinsic_toes", "source_joint_constraints", "whole_human_collision_topology",
    ),
    "neuromuscular_system": (
        "all_source_composed_muscles", "motor_compartments", "semantic_stimulation_ids",
        "muscle_routes", "muscle_wraps", "activation", "muscle_fibres", "compliant_tendons",
        "aponeurosis_state", "entheses", "fatigue", "energetics", "volumetric_active_muscle_ownership",
    ),
    "load_bearing_connective_anatomy": (
        "tendons", "aponeuroses", "fascia", "ligaments", "cartilage", "menisci",
        "intervertebral_discs", "joint_capsules", "rigid_deformable_contact", "fluid_support_fields",
    ),
    "exterior_and_systemic_anatomy": (
        "physical_skin", "fat", "fascia", "organs", "vessels", "nerves", "common_topology_field_graph",
        "mechanics_status", "boundary_status", "sensor_status", "validation_status",
    ),
    "biological_sensing": (
        "muscle_spindle", "golgi_tendon", "joint", "plantar", "cutaneous", "vestibular",
        "visual", "physiological", "delay", "noise", "adaptation", "history",
        "separation_from_privileged_truth",
    ),
    "capability_distribution": (
        "equilibrium", "recovery", "locomotion", "running", "turning", "manipulation", "carrying",
        "sit_stand", "terrain_interaction", "contact_interaction", "physiological_tasks",
        "intervention_tasks", "subject_distributions", "parameter_distributions",
    ),
    "personalization_and_evidence": (
        "subject_atlas", "population_priors", "population_posteriors", "identifiability",
        "uncertainty_quantification", "held_out_physical_data", "competitive_comparisons",
        "exact_apple_execution_evidence",
    ),
}

# Every leaf carries these unresolved obligations. A shared contract reference
# avoids pretending source declarations already supply equations or tolerances.
OBLIGATIONS = {
    "required_representation": "source-traceable representation in the common Human topology and field graph",
    "required_fidelity": "unknown",
    "owning_equations": "unknown",
    "interfaces": "unknown",
    "oracle": "unknown",
    "observable": "unknown",
    "tolerance": "unknown",
    "qualification_distribution": "unknown",
    "evidence_status": "unknown",
    "qualification_requirement": "integrated qualification on the exact compiled stack",
}


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise HumanImportError("coverage records must contain finite JSON values") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _safe_relative(name: str) -> Path:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name or not value.parts:
        raise HumanImportError(f"unsafe source register path: {name}")
    return Path(*value.parts)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise HumanImportError(f"invalid JSON register {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise HumanImportError(f"{path.name} must contain an object")
    return value


class Inventory:
    def __init__(self) -> None:
        self.sources: dict[str, dict[str, Any]] = {}
        self.leaves: dict[str, dict[str, Any]] = {}
        self.registers: list[dict[str, Any]] = []

    def source(self, source_id: str, metadata: dict[str, Any]) -> str:
        record = {"source_id": source_id, "metadata": metadata,
                  "source_url": next((metadata[field] for field in
                                      ("url", "repository", "project_url", "base_url", "doi", "authority")
                                      if metadata.get(field)), "unknown"),
                  "license": metadata.get("license", "unknown"),
                  "allowed_use": metadata.get("redistribution", "unknown")}
        key = digest(record)
        self.sources[key] = {"record_sha256": key, **record}
        return key

    def leaf(self, source_key: str, local_id: str, name: str, kind: str,
             *, domain: str = "unknown", declaration: Any = None) -> None:
        source = self.sources[source_key]
        record = {
            "semantic_id": f"{source['source_id']}:{local_id}",
            "source_local_id": local_id,
            "name": name,
            "kind": kind,
            "domain": domain,
            "source_record_sha256": source_key,
            "declaration_sha256": digest(declaration),
            "obligations_sha256": digest(OBLIGATIONS),
            "evidence_status": "unknown",
        }
        key = digest(record)
        self.leaves[key] = {"leaf_sha256": key, **record}

    def register(self, source_key: str, path: str, expected: str | None,
                 actual: str | None, status: str, *, count: int = 0) -> None:
        self.registers.append({
            "source_record_sha256": source_key, "path": path,
            "expected_sha256": expected, "actual_sha256": actual,
            "status": status, "declaration_count": count,
        })
        if status != "materialized":
            self.leaf(source_key, f"unresolved-register/{path}", path,
                      "unresolved_source_register", declaration={"reason": status})


def _verified(path: Path, expected: str | None) -> str | None:
    if not path.is_file():
        return None
    if not isinstance(expected, str) or len(expected) != 64:
        return None
    actual = file_digest(path)
    if actual != expected:
        raise HumanImportError(f"source hash mismatch: {path.name}")
    return actual


def _xml_declarations(stream: Any, inventory: Inventory, source_key: str, path: str) -> int:
    """Retain every named XML object, plus unnamed physical route/contact rows.

    The full source file digest binds field values and topology. Element paths
    distinguish model variants, parent scopes, repeated names, and unnamed rows.
    No tag is discarded merely because its physical role is not understood.
    Streaming keeps FEBio volume meshes from becoming a host-memory requirement.
    """
    stack: list[tuple[str, Counter[str]]] = []
    count = 0
    physical_parents = {"actuator", "sensor", "tendon", "spatial", "fixed", "equality", "contact"}
    try:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            tag = str(element.tag).rsplit("}", 1)[-1]
            if event == "start":
                name = element.attrib.get("name")
                parent_tag = stack[-1][0].split("[", 1)[0] if stack else ""
                token = tag + (f"[name={quote(name, safe='')}]" if name else "")
                if stack:
                    stack[-1][1][token] += 1
                    ordinal = stack[-1][1][token]
                else:
                    ordinal = 1
                # Retaining ordinals also disambiguates invalid repeated names.
                token += f"[{ordinal}]"
                stack.append((token, Counter()))
                physical_row = (parent_tag in physical_parents or tag in {"PathWrap", "PathPoint"})
                if name is not None or physical_row:
                    local = path + "#/" + "/".join(item[0] for item in stack)
                    inventory.leaf(source_key, local, name or local, f"xml:{tag}",
                                   declaration={"tag": tag, "attributes": element.attrib})
                    count += 1
            else:
                stack.pop()
                element.clear()
    except ET.ParseError as error:
        raise HumanImportError(f"invalid pinned XML register {path}: {error}") from error
    return count


def _bodyparts(path: Path, inventory: Inventory, source_key: str, role: str) -> int:
    count = 0
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t")
        next(rows, None)
        seen: set[str] = set()
        for row in rows:
            if not row:
                continue
            if len(row) != 3:
                raise HumanImportError(f"invalid BodyParts register row: {path.name}")
            if role.endswith("labels"):
                local, name = f"{path.name}/{row[0]}/{row[1]}", row[2]
                kind = "anatomical_label"
            else:
                # These are element identities; membership edges remain bound by
                # the source digest and are not merged across the two hierarchies.
                local, name = f"{path.name}/element/{row[2]}", row[2]
                kind = "anatomical_element"
            if local not in seen:
                seen.add(local)
                inventory.leaf(source_key, local, name, kind, declaration={"identity": local})
                count += 1
    return count


def _xml_register(stream: Any, inventory: Inventory, source_key: str, path: str) -> tuple[int, str]:
    before = len(inventory.leaves)
    try:
        return _xml_declarations(stream, inventory, source_key, path), "materialized"
    except HumanImportError:
        # A malformed pinned legacy model remains a target gap. Keep its exact
        # digest and any declarations parsed before the error; do not omit the
        # register, rewrite the source, or call a partial parse complete.
        return len(inventory.leaves) - before, "invalid_source_xml"


def _source_files(inventory: Inventory, source_key: str, sources: Path,
                  metadata: dict[str, Any]) -> None:
    for filename, properties in sorted(metadata["files"].items()):
        path = sources / _safe_relative(filename)
        expected = properties.get("sha256")
        actual = _verified(path, expected)
        if actual is None:
            inventory.register(source_key, filename, expected, None,
                               "missing" if not path.is_file() else "unpinned")
            continue
        role = properties.get("role", "unknown")
        count = 0
        if role.endswith("labels") or role.endswith("compound_definitions"):
            count = _bodyparts(path, inventory, source_key, role)
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    if PurePosixPath(name).suffix.lower() == ".obj":
                        inventory.leaf(source_key, f"{filename}#/{name}",
                                       PurePosixPath(name).stem, "anatomical_mesh",
                                       declaration={"member": name})
                        count += 1
        else:
            # Relation files bind the graph even though they do not create a new
            # anatomical structure for every membership of that same structure.
            inventory.leaf(source_key, filename, filename, "source_topology_register",
                           declaration={"sha256": actual, "role": role})
            count = 1
        inventory.register(source_key, filename, expected, actual, "materialized", count=count)


def _archive(inventory: Inventory, source_key: str, sources: Path,
             metadata: dict[str, Any]) -> None:
    relative = str(_safe_relative(metadata["storage_dir"]) / _safe_relative(metadata["archive_file"]))
    path = sources / relative
    expected = metadata.get("archive_sha256")
    actual = _verified(path, expected)
    if actual is None:
        inventory.register(source_key, relative, expected, None,
                           "missing" if not path.is_file() else "unpinned")
        return
    count = 0
    with tarfile.open(path, "r:gz") as archive:
        members = sorted(archive.getmembers(), key=lambda member: member.name)
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        if len(roots) != 1:
            raise HumanImportError(f"source archive {path.name} must have one root")
        for member in members:
            _safe_relative(member.name)
            if not member.isfile() or Path(member.name).suffix.lower() not in {".xml", ".osim"}:
                continue
            local = str(PurePosixPath(*PurePosixPath(member.name).parts[1:]))
            stream = archive.extractfile(member)
            assert stream is not None
            data = stream.read()
            member_hash = hashlib.sha256(data).hexdigest()
            member_count, status = _xml_register(io.BytesIO(data), inventory, source_key, local)
            inventory.register(source_key, f"{relative}#/{local}", member_hash,
                               member_hash, status, count=member_count)
            count += member_count
    inventory.register(source_key, relative, expected, actual, "materialized", count=count)


def _nonmuscle_inventory_tables(value: dict[str, Any], model: dict[str, Any]) -> dict[str, list[Any]]:
    """Validate the optional source-only table and resolve its route dependencies."""
    tables = {}
    for category in ("sites", "wrap_geometries"):
        rows = value.get(category)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise HumanImportError(f"composed MyoSim IR has invalid {category} table")
        tables[category] = list(rows)
    records = value.get("nonmuscle_tendons")
    if records is None:
        return tables
    tendon_count = model.get("tendon_count")
    if not isinstance(records, list) or type(tendon_count) is not int or tendon_count < 0:
        raise HumanImportError("composed nonmuscle tendon inventory has invalid count or table")
    muscle_ids = {entry.get("tendon") for entry in value["muscles"]}
    if any(type(tendon) is not int or not 0 <= tendon < tendon_count for tendon in muscle_ids):
        raise HumanImportError("composed muscle tendon has an invalid source ID")
    seen = set(muscle_ids)
    required_parameters = {"tendon_stiffness", "tendon_damping", "tendon_lengthspring",
                           "tendon_limited", "tendon_range", "tendon_frictionloss",
                           "tendon_solref_lim", "tendon_solimp_lim", "tendon_solref_fri",
                           "tendon_solimp_fri", "tendon_actuatorid"}
    for entry in records:
        if not isinstance(entry, dict):
            raise HumanImportError("composed nonmuscle tendon must be an object")
        tendon = entry.get("id")
        if type(tendon) is not int or not 0 <= tendon < tendon_count or tendon in seen:
            raise HumanImportError("composed nonmuscle tendon has an invalid or repeated source ID")
        seen.add(tendon)
        parameters = entry.get("compiled_mujoco_parameters")
        if (entry.get("native_mechanics_status") != "not_lowered"
                or not isinstance(parameters, dict) or not required_parameters <= parameters.keys()
                or type(parameters.get("tendon_actuatorid")) is not int
                or parameters["tendon_actuatorid"] != -1):
            raise HumanImportError("composed nonmuscle tendon must retain source parameters without mechanics promotion")
        dependencies = {}
        for category in tables:
            rows = entry.get(category)
            if not isinstance(rows, list):
                raise HumanImportError(f"composed nonmuscle tendon has no {category} dependencies")
            dependencies[category] = set()
            existing = {row.get("id"): row for row in tables[category]}
            for row in rows:
                if not isinstance(row, dict):
                    raise HumanImportError(f"composed nonmuscle tendon has invalid {category} row")
                identifier = row.get("id")
                if type(identifier) is not int or identifier < 0 or identifier in dependencies[category]:
                    raise HumanImportError(f"composed nonmuscle tendon has invalid {category} identity")
                dependencies[category].add(identifier)
                if identifier in existing and canonical_bytes(existing[identifier]) != canonical_bytes(row):
                    raise HumanImportError(f"composed nonmuscle tendon conflicts with source {category}")
                if identifier not in existing:
                    tables[category].append(row)
                    existing[identifier] = row
        route = entry.get("route")
        if not isinstance(route, list) or len(route) < 2:
            raise HumanImportError("composed nonmuscle tendon has no spatial route")
        for node in route:
            if not isinstance(node, dict):
                raise HumanImportError("composed nonmuscle tendon has an invalid route node")
            kind, identifier, side = node.get("kind"), node.get("source_id"), node.get("side_site_source_id")
            category = "sites" if kind == "site" else "wrap_geometries"
            if (kind not in {"site", "sphere", "cylinder"}
                    or type(identifier) is not int or identifier not in dependencies[category]
                    or type(side) is not int or side < -1
                    or (side >= 0 and side not in dependencies["sites"])
                    or (kind == "site" and side != -1)):
                raise HumanImportError("composed nonmuscle tendon has an unresolved route dependency")
        if route[0]["kind"] != "site" or route[-1]["kind"] != "site":
            raise HumanImportError("composed nonmuscle tendon is not site bounded")
    if seen != set(range(tendon_count)):
        raise HumanImportError("composed nonmuscle tendon inventory omits declared source tendons")
    tables["nonmuscle_tendons"] = records
    return tables


def _myosim_ir(inventory: Inventory, source_key: str, path: Path | None,
               expected: str | None, metadata: dict[str, Any]) -> None:
    if path is None:
        inventory.register(source_key, "composed/myofullbody", None, None, "not_supplied")
        return
    actual = _verified(path, expected)
    if actual is None:
        raise HumanImportError("composed MyoSim IR requires an existing file and explicit SHA-256")
    value = _read_json(path)
    if (value.get("schema") != "numi.human.myosim-mujoco-export.v1"
            or value.get("source", {}).get("archive_sha256") != metadata.get("archive_sha256")
            or value.get("source", {}).get("revision") != metadata.get("revision")):
        raise HumanImportError("composed MyoSim IR differs from the pinned source identity")
    model = value.get("model", {})
    for category, declared, offset in (("bodies", "body_count_with_world", 1),
                                        ("joints", "joint_count", 0), ("muscles", "nu", 0)):
        entries = value.get(category)
        expected_count = model.get(declared)
        if (not isinstance(entries, list) or not isinstance(expected_count, int)
                or expected_count < offset or len(entries) != expected_count - offset):
            raise HumanImportError(f"composed MyoSim {category} count differs from the model declaration")
    source_tables = _nonmuscle_inventory_tables(value, model)
    count = 0
    categories = ["bodies", "joints", "joint_equalities", "sites", "wrap_geometries", "muscles"]
    if "nonmuscle_tendons" in source_tables:
        categories.append("nonmuscle_tendons")
    for category in categories:
        entries = source_tables.get(category, value.get(category))
        if not isinstance(entries, list):
            raise HumanImportError(f"composed MyoSim IR has no {category} table")
        seen: set[str] = set()
        for entry in entries:
            name = entry.get("name")
            identifier = str(name if name is not None else entry.get("id"))
            if identifier == "None" or identifier in seen:
                raise HumanImportError(f"composed MyoSim {category} has invalid or repeated identity")
            seen.add(identifier)
            local = f"composed/myofullbody/{category}/{quote(identifier, safe='')}"
            inventory.leaf(source_key, local, identifier, f"composed:{category}", declaration=entry)
            count += 1
            if category in {"muscles", "nonmuscle_tendons"}:
                for index, route in enumerate(entry.get("route", [])):
                    inventory.leaf(source_key, f"{local}/route/{index}", f"{identifier} route {index}",
                                   "composed:route_element", declaration=route)
                    count += 1
    contact = value.get("support_contact", {})
    for index, geometry in enumerate(contact.get("geometries", [])):
        identifier = str(geometry.get("name", index))
        inventory.leaf(source_key, f"composed/myofullbody/support/{quote(identifier, safe='')}",
                       identifier, "composed:source_support_contact", declaration=geometry)
        count += 1
    muscle_tendons = {entry.get("tendon") for entry in value["muscles"]}
    if "nonmuscle_tendons" in source_tables:
        inventory.register(source_key, "composed/myofullbody/nonmuscle_tendons", expected, actual,
                           "materialized", count=len(source_tables["nonmuscle_tendons"]))
    elif (None in muscle_tendons or model.get("tendon_count") != len(muscle_tendons)):
        inventory.register(source_key, "composed/myofullbody/nonmuscle_tendons", None, None,
                           "incomplete_composed_inventory")
    # Binding the derivative digest is essential: the archive pin alone does not
    # prove the compiler or composition transform that produced these live IDs.
    inventory.register(source_key, "composed/myofullbody", expected, actual, "materialized", count=count)


def _supplements(inventory: Inventory, sources: Path, root: Path) -> None:
    from .open_knee import EXPECTED_HASHES
    knee = {"dataset": "Open Knee(s) oks003", "doi": "10.18735/b0zv-n395",
            "license": "CC-BY-4.0", "files": EXPECTED_HASHES}
    key = inventory.source("open_knee_oks003", knee)
    for filename, expected in sorted(EXPECTED_HASHES.items()):
        relative = f"open-knee-oks003/{filename}"
        path = sources / relative
        actual = _verified(path, expected)
        if actual is None:
            inventory.register(key, relative, expected, None, "missing")
        else:
            count, status = _xml_register(path, inventory, key, filename) if path.suffix != ".txt" else (0, "materialized")
            inventory.register(key, relative, expected, actual, status, count=count)
    path = root / "config/zanatomy-calf-visual-supplement.v1.json"
    if path.is_file():
        value = _read_json(path)
        key = inventory.source("zanatomy_registered_visual_supplement", value["source"])
        for entry in value["objects"]:
            inventory.leaf(key, entry["id"], entry["id"], "registered_visual_object", declaration=entry)
        actual = file_digest(path)
        inventory.register(key, "config/zanatomy-calf-visual-supplement.v1.json", actual,
                           actual, "materialized", count=len(value["objects"]))


def validate_manifest(value: dict[str, Any]) -> None:
    """Verify immutable content, references, mandatory scope, and claim boundary."""
    required_fields = {"schema", "compiler", "source_lock_sha256", "previous_manifest_sha256",
                       "manifest_sha256", "obligations", "source_records", "registers",
                       "register_history", "leaves", "counts", "scope_status",
                       "integrated_qualification", "evidence_boundary"}
    if not isinstance(value, dict) or set(value) != required_fields:
        raise HumanImportError("target coverage manifest fields differ from the v1 schema")
    if value.get("schema") != SCHEMA:
        raise HumanImportError("unsupported target coverage schema")
    payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != digest(payload):
        raise HumanImportError("target coverage manifest digest mismatch")
    if value.get("obligations") != OBLIGATIONS:
        raise HumanImportError("target coverage obligations changed or qualification was fabricated")
    sources = value.get("source_records", [])
    if not all(isinstance(value[field], list) for field in ("source_records", "leaves", "registers", "register_history")):
        raise HumanImportError("target coverage record tables must be arrays")
    keys = set()
    for source in sources:
        if (not isinstance(source, dict) or set(source) != {"record_sha256", "source_id", "metadata", "source_url", "license", "allowed_use"}
                or not isinstance(source["metadata"], dict)
                or not all(isinstance(source[field], str) and source[field]
                           for field in ("record_sha256", "source_id", "source_url", "license", "allowed_use"))):
            raise HumanImportError("target coverage source fields differ from the v1 schema")
        key = source.get("record_sha256")
        if key in keys or key != digest({k: v for k, v in source.items() if k != "record_sha256"}):
            raise HumanImportError("target coverage source identity mismatch or duplicate")
        keys.add(key)
    leaves = value.get("leaves", [])
    leaf_keys = set()
    source_by_key = {source["record_sha256"]: source for source in sources}
    for leaf in leaves:
        if (not isinstance(leaf, dict) or set(leaf) != {"leaf_sha256", "semantic_id", "source_local_id", "name", "kind", "domain", "source_record_sha256", "declaration_sha256", "obligations_sha256", "evidence_status"}
                or not all(isinstance(item, str) for item in leaf.values())):
            raise HumanImportError("target coverage leaf fields differ from the v1 schema")
        key = leaf.get("leaf_sha256")
        if key in leaf_keys or key != digest({k: v for k, v in leaf.items() if k != "leaf_sha256"}):
            raise HumanImportError("target coverage leaf identity mismatch or duplicate")
        if (leaf.get("source_record_sha256") not in keys
                or leaf.get("obligations_sha256") != digest(OBLIGATIONS)
                or leaf.get("evidence_status") != "unknown"):
            raise HumanImportError("target coverage leaf reference or evidence status is invalid")
        source = source_by_key[leaf["source_record_sha256"]]
        if leaf.get("semantic_id") != f"{source['source_id']}:{leaf.get('source_local_id')}":
            raise HumanImportError("target coverage semantic identity differs from its source")
        leaf_keys.add(key)
    mandatory = Inventory()
    mandatory_key = mandatory.source("mandatory", {"authority": "Docs/DEVELOPMENT_ROADMAP.md",
                                                   "catalog_version": 1, "license": "Apache-2.0"})
    for domain, names in MANDATORY.items():
        for name in names:
            mandatory.leaf(mandatory_key, f"{domain}/{name}", name.replace("_", " "),
                           "mandatory_target", domain=domain, declaration={"domain": domain, "name": name})
    if not mandatory.leaves.keys() <= leaf_keys:
        raise HumanImportError("mandatory Human target leaves were removed")
    if value.get("integrated_qualification") != "unknown":
        raise HumanImportError("source coverage cannot promote integrated qualification")
    for register in [*value.get("registers", []), *value.get("register_history", [])]:
        if (not isinstance(register, dict) or set(register) != {"source_record_sha256", "path", "expected_sha256", "actual_sha256", "status", "declaration_count"}
                or not isinstance(register["path"], str) or not register["path"]
                or type(register["declaration_count"]) is not int or register["declaration_count"] < 0
                or register["status"] not in {"materialized", "missing", "unpinned", "not_supplied", "unavailable_source_inventory", "incomplete_composed_inventory", "invalid_source_xml", "retained_source_not_in_current_lock"}):
            raise HumanImportError("target coverage register fields differ from the v1 schema")
        if register.get("source_record_sha256") not in keys:
            raise HumanImportError("source register reference is invalid")
        for field in ("actual_sha256", "expected_sha256"):
            fingerprint = register[field]
            if fingerprint is not None and (not isinstance(fingerprint, str) or len(fingerprint) != 64
                                             or any(character not in "0123456789abcdef" for character in fingerprint)):
                raise HumanImportError("source register has an invalid SHA-256")
        if register["status"] == "materialized" and (register["actual_sha256"] is None
                                                      or register["actual_sha256"] != register["expected_sha256"]):
            raise HumanImportError("materialized source register must match an explicit content pin")
    registers = value.get("registers", [])
    unresolved = sum(item["status"] != "materialized" for item in registers)
    counts = {"leaves": len(leaves), "mandatory_leaves": sum(map(len, MANDATORY.values())),
              "by_kind": dict(sorted(Counter(item["kind"] for item in leaves).items())),
              "unresolved_current_registers": unresolved}
    if value.get("counts") != counts:
        raise HumanImportError("target coverage summary differs from the retained leaves")
    expected_scope = "blocked" if unresolved else "source_union_materialized"
    if value.get("scope_status") != expected_scope:
        raise HumanImportError("target coverage scope summary hides unresolved sources")


def validate_transition(previous: dict[str, Any], candidate: dict[str, Any]) -> None:
    validate_manifest(previous)
    validate_manifest(candidate)
    for field, identity in (("leaves", "leaf_sha256"), ("source_records", "record_sha256")):
        before = {item[identity]: item for item in previous[field]}
        after = {item[identity]: item for item in candidate[field]}
        if not before.keys() <= after.keys() or any(after[key] != item for key, item in before.items()):
            raise HumanImportError(f"target coverage transition deletes or changes prior {field}")
    before_registers = {digest(item) for item in [*previous["registers"], *previous.get("register_history", [])]}
    after_registers = {digest(item) for item in [*candidate["registers"], *candidate.get("register_history", [])]}
    if not before_registers <= after_registers:
        raise HumanImportError("target coverage transition deletes prior source register provenance")


def materialize(*, sources: Path, source_lock: Path, repository_root: Path = ROOT,
                myosim_ir: Path | None = None, myosim_ir_sha256: str | None = None,
                previous: dict[str, Any] | None = None, supplements: bool = True) -> dict[str, Any]:
    lock = _read_json(source_lock)
    if lock.get("schema") != "numi.human.source-lock.v1" or not isinstance(lock.get("sources"), dict):
        raise HumanImportError("unsupported Human source lock")
    if (myosim_ir is None) != (myosim_ir_sha256 is None):
        raise HumanImportError("composed MyoSim IR and its explicit SHA-256 must be supplied together")
    if myosim_ir is not None and "myosim_fullbody" not in lock["sources"]:
        raise HumanImportError("composed MyoSim IR requires a pinned MyoSim source register")
    inventory = Inventory()
    mandatory_key = inventory.source("mandatory", {"authority": "Docs/DEVELOPMENT_ROADMAP.md",
                                                   "catalog_version": 1, "license": "Apache-2.0"})
    for domain, names in MANDATORY.items():
        for name in names:
            inventory.leaf(mandatory_key, f"{domain}/{name}", name.replace("_", " "),
                           "mandatory_target", domain=domain, declaration={"domain": domain, "name": name})
    for source_id, metadata in sorted(lock["sources"].items()):
        key = inventory.source(source_id, metadata)
        if "files" in metadata:
            _source_files(inventory, key, sources, metadata)
        elif "archive_file" in metadata:
            _archive(inventory, key, sources, metadata)
        elif "sha256" in metadata:
            filename = metadata.get("model_file") or PurePosixPath(metadata.get("path", "")).name
            path = sources / _safe_relative(filename)
            expected = metadata.get("sha256")
            actual = _verified(path, expected)
            count, status = _xml_register(path, inventory, key, filename) if actual else (0, "missing")
            inventory.register(key, filename, expected, actual, status, count=count)
        else:
            inventory.register(key, "source_inventory", None, None, "unavailable_source_inventory")
        if source_id == "myosim_fullbody":
            _myosim_ir(inventory, key, myosim_ir, myosim_ir_sha256, metadata)
    if supplements:
        _supplements(inventory, sources, repository_root)
    if previous is not None:
        validate_manifest(previous)
        # Old versions remain byte-for-byte targets. New revisions add versions;
        # even a missing current source cannot erase a previously known leaf.
        current_source_ids = {source["source_id"] for source in inventory.sources.values()}
        for source in previous["source_records"]:
            inventory.sources.setdefault(source["record_sha256"], source)
            if source["source_id"] not in current_source_ids:
                inventory.register(source["record_sha256"], "retained_source_inventory", None, None,
                                   "retained_source_not_in_current_lock")
        for leaf in previous["leaves"]:
            inventory.leaves.setdefault(leaf["leaf_sha256"], leaf)
    registers = sorted(inventory.registers, key=lambda item: (item["source_record_sha256"], item["path"]))
    history = {}
    if previous:
        for item in [*previous["registers"], *previous.get("register_history", [])]:
            history[digest(item)] = item
    result = {
        "schema": SCHEMA, "compiler": COMPILER_VERSION,
        "source_lock_sha256": file_digest(source_lock),
        "previous_manifest_sha256": previous["manifest_sha256"] if previous else None,
        "obligations": OBLIGATIONS,
        "source_records": sorted(inventory.sources.values(), key=lambda item: item["record_sha256"]),
        "registers": registers,
        "register_history": [history[key] for key in sorted(history)],
        "leaves": sorted(inventory.leaves.values(), key=lambda item: item["leaf_sha256"]),
        "counts": {"leaves": len(inventory.leaves), "mandatory_leaves": sum(map(len, MANDATORY.values())),
                   "by_kind": dict(sorted(Counter(item["kind"] for item in inventory.leaves.values()).items())),
                   "unresolved_current_registers": sum(item["status"] != "materialized" for item in registers)},
        "scope_status": "blocked" if any(item["status"] != "materialized" for item in registers) else "source_union_materialized",
        "integrated_qualification": "unknown",
        "evidence_boundary": "Source declarations and mandatory targets only. Physical roles, fidelity, equations, interfaces, validation and source composition remain unresolved unless separately evidenced. No runtime admission or physical qualification is implied.",
    }
    result["manifest_sha256"] = digest(result)
    validate_manifest(result)
    if previous is not None:
        validate_transition(previous, result)
    return result


def command(arguments: argparse.Namespace) -> int:
    previous = _read_json(arguments.previous) if arguments.previous else None
    result = materialize(sources=arguments.sources, source_lock=arguments.source_lock,
                         myosim_ir=arguments.myosim_ir, myosim_ir_sha256=arguments.myosim_ir_sha256,
                         previous=previous)
    output = arguments.output
    encoded = canonical_bytes(result) + b"\n"
    if output.exists() and output.read_bytes() != encoded:
        raise HumanImportError("target coverage output is immutable; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with output.open("xb") as stream:
            stream.write(encoded)
    print(json.dumps({"manifest_sha256": result["manifest_sha256"], "scope_status": result["scope_status"],
                      "integrated_qualification": result["integrated_qualification"], **result["counts"]}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--myosim-ir", type=Path, help="composed MyoSim export; source declarations alone do not enumerate mirrored live IDs")
    parser.add_argument("--myosim-ir-sha256", help="explicit content pin for the composed MyoSim export")
    parser.add_argument("--previous", type=Path, help="previous immutable coverage manifest; every prior leaf is retained")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=command)
