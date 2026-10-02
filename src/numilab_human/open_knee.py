"""Compile the exact Open Knee(s) oks003 knee into a mechanics-ready payload.

Open Knee(s) remains the anatomical/tissue authority. MyoSim supplies only the
live body frames and joint axis used to place the specimen in Numi Human. The
compiler admits one proper uniform scale, one anatomically constructed proper
rotation, and one knee-origin translation; it never performs an anisotropic
warp or an unconstrained nearest-surface flip.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .myosim_bone_proximity import _compiled_meshes_by_body
from .myosim_export import export_fullbody
from .upper_limb_registration import _rotation_xyzw


SCHEMA = "numi.human.open-knee-oks003-payload.v3"
MAGIC = b"NHKNEE1\0"
ABI = 3
INVALID_INDEX = 0xFFFFFFFF

EXPECTED_HASHES = {
    "Geometry.feb": "3642bd368bbc867569f181fa76129f746470e807e3977585d3803f092dd11262",
    "ModelProperties.xml": "0ac446ce098b9a09505992eb4f4419c7b944cd57a4afbf6392b137f4806603c1",
    "FeBio_custom.feb": "00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a",
    "license.txt": "d72918838b4adf30979d2a26c23837f0ca05185ba799a3a4fe1fe1b4c05b20b8",
}
ARCHIVED_REFERENCE_SOLVER_VERSION = "2.9.1"
ARCHIVED_REFERENCE_LOG_SHA256 = (
    "d47631c09ce7fc93154c7d6c84caa299aab03709abe9ab7d2a41a60a1b78e426"
)
ARCHIVED_REFERENCE_OBSERVATIONS_SHA256 = (
    "9a99bbdcc94b4ba1aca22c9cc3d72919db9b1e08fec703732f8fcf635705d149"
)
ARCHIVED_REFERENCE_CONTACT_SHA256 = (
    "4ff55ec22a541d0207dcfe4d3e60cfd2ab426fd1c012d69b910c7439fb2da382"
)
ARCHIVED_REFERENCE_XPLT_SHA256 = (
    "c370ae9f94e9faee2d7060bf2a6819e03be1312e82ca79bd2e9cebf8b34398de"
)
ARCHIVED_REFERENCE_GEOMETRY_SHA256 = (
    "4155db1d0d7b87ffb2c668102d2495870e4461a539b18e6708f1f4817b5601bf"
)

EXPECTED_REGIONS = {
    "QAT": (14963, "tet4", 69410),
    "TBC-L": (40669, "tet4", 200079),
    "PCL": (3714, "tet4", 14379),
    "PTC": (26121, "tet4", 121105),
    "PTB": (8642, "tri3", 17280),
    "ACL": (15792, "tet4", 72552),
    "FBB": (3794, "tri3", 7584),
    "MCL": (15693, "tet4", 62712),
    "PTL": (9280, "tet4", 35616),
    "MNS-L": (10901, "tet4", 44953),
    "MNS-M": (11706, "tet4", 51009),
    "LCL": (2960, "tet4", 9773),
    "TBC-M": (18060, "tet4", 75627),
    "TBB": (20900, "tri3", 41796),
    "FMB": (20171, "tri3", 40338),
    "FMC": (24870, "tet4", 87072),
}

EXTENSOR_STACK_MINIMUMS_M = {
    "patellar_cartilage_posterior_to_bone_m": 0.005,
    "quadriceps_tendon_proximal_to_patella_m": 0.020,
    "patellar_tendon_distal_to_patella_m": 0.020,
}

REGION_KIND = {
    "FMB": 1, "TBB": 1, "FBB": 1, "PTB": 1,
    "FMC": 2, "TBC-L": 2, "TBC-M": 2, "PTC": 2,
    "MNS-L": 3, "MNS-M": 3,
    "ACL": 4, "PCL": 4, "MCL": 4, "LCL": 4,
    "PTL": 5, "QAT": 5,
}

VISUAL_BODY_ROLE = {
    "FMB": "femur", "FMC": "femur",
    "TBB": "tibia", "FBB": "tibia",
    "TBC-L": "tibia", "TBC-M": "tibia",
    "MNS-L": "tibia", "MNS-M": "tibia",
    "PTB": "patella", "PTC": "patella",
    "ACL": "femur", "PCL": "femur",
    "MCL": "femur", "LCL": "femur",
    "PTL": "patella", "QAT": "patella",
}

RIGID_COUNTERPART_ROLE = {
    "FMB": "femur", "TBB": "tibia", "FBB": "tibia",
    "PTB": "patella",
}

REGION_STRUCT = struct.Struct("<16s8I11fI")
SURFACE_STRUCT = struct.Struct("<48s5I")
NODE_SET_STRUCT = struct.Struct("<48s5I")
SURFACE_PAIR_STRUCT = struct.Struct("<48sII")
NODE_STRUCT = struct.Struct("<3fI3fI3fI")
TETRAHEDRON_STRUCT = struct.Struct("<4I")
FACE_STRUCT = struct.Struct("<3I")
MEMBERSHIP_STRUCT = struct.Struct("<I")
HEADER_STRUCT = struct.Struct("<8s12I128s")

MATERIAL_HAS_HOMOGENEOUS_FIBER = 1 << 0
MATERIAL_HAS_ISOCHORIC_IN_SITU_STRETCH = 1 << 1
MATERIAL_HAS_ISOTROPIC_MOONEY_RIVLIN = 1 << 2


def _cartilage_material_values(name: str, material: dict) -> tuple[float, float, float]:
    """Admit the pinned source solid law; never substitute a calibration fit."""
    if name not in ("FMC", "PTC", "TBC-L", "TBC-M") or material.get("type") != "Mooney-Rivlin":
        raise RuntimeError("Open Knee cartilage requires its source Mooney-Rivlin material")
    try:
        values = tuple(float(material[key]) for key in ("c1", "c2", "k"))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Open Knee cartilage source material is incomplete") from error
    if not all(math.isfinite(v) for v in values) or not (values[0] > 0 and values[1] >= 0 and values[2] > 0):
        raise RuntimeError("Open Knee cartilage source material left its physical gate")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _orientation_preserving_connectivity(
    indices: tuple[int, ...], *, reflected: bool
) -> tuple[int, ...]:
    """Reverse an element's parity after a spatial reflection.

    A sagittal reflection changes the sign of tetrahedron volume and triangle
    normals.  Swapping the first two indices restores the source element's
    orientation without changing its topology or attachment membership.
    """
    if not reflected:
        return indices
    if len(indices) not in {3, 4}:
        raise ValueError("orientation correction requires tri3 or tet4 connectivity")
    return (indices[1], indices[0], *indices[2:])


def _vector(text: str | None, label: str) -> tuple[float, float, float]:
    if text is None:
        raise ValueError(f"Open Knee(s) {label} is absent")
    values = tuple(float(value.strip()) for value in text.replace(" ", "").split(","))
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Open Knee(s) {label} is not a finite 3-vector")
    return values


@dataclass
class Region:
    name: str
    node_ids: list[int] = field(default_factory=list)
    nodes_mm: list[tuple[float, float, float]] = field(default_factory=list)
    element_type: str = ""
    elements: list[tuple[int, ...]] = field(default_factory=list)


@dataclass
class Surface:
    name: str
    faces: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class Source:
    regions: dict[str, Region]
    node_sets: dict[str, list[int]]
    surfaces: dict[str, Surface]
    surface_pairs: list[tuple[str, str, str]]
    landmarks: dict[str, tuple[float, float, float] | int]
    materials: dict[str, dict[str, float | str]]
    fiber_directions: dict[str, tuple[float, float, float]]
    # Complete authored FEBio program, including nested material/prestrain laws,
    # per-element meniscus fibres, joints, contact enforcement, curves and steps.
    # The legacy NHKNEE1 payload does NOT execute this program. Admission and
    # source-frame freezing are owned by open_knee_reference.
    mechanical_program: ET.Element | None = None


def _parse_febio_fiber_directions(
    path: Path,
) -> dict[str, tuple[float, float, float]]:
    """Read source-authored homogeneous fibre axes from the solved FEBio deck."""
    root = ET.parse(path).getroot()
    material_root = root.find("Material")
    if material_root is None:
        raise ValueError("Open Knee(s) FeBio deck has no Material table")
    directions: dict[str, tuple[float, float, float]] = {}
    for material in material_root.findall("material"):
        name = material.attrib.get("name", "")
        fiber = material.find("fiber")
        if fiber is None:
            continue
        if fiber.attrib.get("type") != "vector" or name in directions:
            raise ValueError(f"Open Knee(s) material {name} fibre identity is invalid")
        direction = _vector(fiber.text, f"{name} fibre")
        length = math.sqrt(sum(value * value for value in direction))
        if not 0.999 <= length <= 1.001:
            raise ValueError(f"Open Knee(s) material {name} fibre is not unit length")
        directions[name] = tuple(value / length for value in direction)
    return directions


def _parse_model_properties(path: Path) -> tuple[
    dict[str, tuple[float, float, float] | int],
    dict[str, dict[str, float | str]],
]:
    root = ET.parse(path).getroot()
    landmarks_element = root.find("Landmarks")
    if landmarks_element is None:
        raise ValueError("Open Knee(s) ModelProperties has no Landmarks")
    landmarks: dict[str, tuple[float, float, float] | int] = {}
    for child in landmarks_element:
        if child.text is None:
            continue
        compact = child.text.strip()
        if "," in compact:
            landmarks[child.tag] = _vector(compact, child.tag)
        else:
            landmarks[child.tag] = int(compact)
    required = {"FMO", "Xf_axis", "Yf_axis", "Zf_axis", "TBO", "PTO"}
    if not required.issubset(landmarks):
        raise ValueError("Open Knee(s) anatomical coordinate landmarks are incomplete")
    materials: dict[str, dict[str, float | str]] = {}
    material_root = root.find("Material")
    if material_root is None:
        raise ValueError("Open Knee(s) ModelProperties has no Material table")
    for element in material_root.findall("material"):
        name = element.attrib.get("name")
        kind = element.attrib.get("type")
        if not name or not kind or name in materials:
            raise ValueError("Open Knee(s) material identity is invalid")
        values: dict[str, float | str] = {"type": kind}
        for child in element:
            text = (child.text or "").strip()
            try:
                values[child.tag] = float(text)
            except ValueError:
                values[child.tag] = text
        materials[name] = values
    return landmarks, materials


def parse_source(directory: Path, *, enforce_exact: bool = True) -> Source:
    directory = directory.resolve()
    if enforce_exact:
        for name, expected in EXPECTED_HASHES.items():
            path = directory / name
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"Open Knee(s) oks003 source identity drifted for {name}")
        license_text = (directory / "license.txt").read_text(
            encoding="utf-8", errors="strict"
        )
        if "Creative Commons Attribution 4.0 International" not in license_text:
            raise ValueError("Open Knee(s) CC BY 4.0 license text is absent")
    landmarks, materials = _parse_model_properties(directory / "ModelProperties.xml")
    fiber_directions = _parse_febio_fiber_directions(
        directory / "FeBio_custom.feb"
    )
    regions: dict[str, Region] = {}
    node_sets: dict[str, list[int]] = {}
    surfaces: dict[str, Surface] = {}
    surface_pairs: list[tuple[str, str, str]] = []
    current_nodes: Region | None = None
    current_elements: Region | None = None
    current_node_set: list[int] | None = None
    current_surface: Surface | None = None
    pair_name: str | None = None
    pair_master: str | None = None
    pair_slave: str | None = None
    geometry = directory / "Geometry.feb"
    for event, element in ET.iterparse(geometry, events=("start", "end")):
        tag = element.tag
        if event == "start":
            if tag == "Nodes":
                name = element.attrib.get("name", "")
                if not name or name in regions:
                    raise ValueError("Open Knee(s) node-region identity is invalid")
                current_nodes = regions.setdefault(name, Region(name))
            elif tag == "Elements":
                name = element.attrib.get("name", "")
                current_elements = regions.get(name)
                if current_elements is None or current_elements.element_type:
                    raise ValueError(f"Open Knee(s) element region {name} is invalid")
                current_elements.element_type = element.attrib.get("type", "")
            elif tag == "NodeSet":
                name = element.attrib.get("name", "")
                if not name or name in node_sets:
                    raise ValueError("Open Knee(s) node-set identity is invalid")
                current_node_set = node_sets.setdefault(name, [])
            elif tag == "Surface":
                name = element.attrib.get("name", "")
                if not name or name in surfaces:
                    raise ValueError("Open Knee(s) surface identity is invalid")
                current_surface = surfaces.setdefault(name, Surface(name))
            elif tag == "SurfacePair":
                pair_name = element.attrib.get("name", "")
                pair_master = None
                pair_slave = None
            continue
        if tag == "node":
            identifier = int(element.attrib["id"])
            if current_nodes is not None:
                current_nodes.node_ids.append(identifier)
                current_nodes.nodes_mm.append(_vector(element.text, "geometry node"))
            elif current_node_set is not None:
                current_node_set.append(identifier)
        elif tag == "elem" and current_elements is not None:
            values = tuple(int(value.strip()) for value in (element.text or "").split(","))
            current_elements.elements.append(values)
        elif tag == "tri3" and current_surface is not None:
            values = tuple(int(value.strip()) for value in (element.text or "").split(","))
            if len(values) != 3:
                raise ValueError(f"Open Knee(s) surface {current_surface.name} is not tri3")
            current_surface.faces.append(values)
        elif tag == "master" and pair_name is not None:
            pair_master = element.attrib.get("surface")
        elif tag == "slave" and pair_name is not None:
            pair_slave = element.attrib.get("surface")
        elif tag == "Nodes":
            current_nodes = None
        elif tag == "Elements":
            current_elements = None
        elif tag == "NodeSet":
            current_node_set = None
        elif tag == "Surface":
            current_surface = None
        elif tag == "SurfacePair":
            if not pair_name or not pair_master or not pair_slave:
                raise ValueError("Open Knee(s) surface pair is incomplete")
            surface_pairs.append((pair_name, pair_master, pair_slave))
            pair_name = None
        element.clear()
    all_node_ids: set[int] = set()
    node_owner: dict[int, str] = {}
    for name, region in regions.items():
        if len(region.node_ids) != len(region.nodes_mm):
            raise ValueError(f"Open Knee(s) region {name} node table is incomplete")
        for identifier in region.node_ids:
            if identifier in all_node_ids:
                raise ValueError(f"Open Knee(s) duplicate global node {identifier}")
            all_node_ids.add(identifier)
            node_owner[identifier] = name
        expected_width = 4 if region.element_type == "tet4" else 3
        if any(len(values) != expected_width for values in region.elements):
            raise ValueError(f"Open Knee(s) region {name} element width drifted")
        if any(node_owner.get(node) not in {None, name} for values in region.elements for node in values):
            raise ValueError(f"Open Knee(s) region {name} element crosses node ownership")
    # Node ownership is complete only after every Nodes block has been seen.
    for name, region in regions.items():
        if any(node_owner.get(node) != name for values in region.elements for node in values):
            raise ValueError(f"Open Knee(s) region {name} element references foreign nodes")
    for name, members in node_sets.items():
        if len(members) != len(set(members)) or any(node not in all_node_ids for node in members):
            raise ValueError(f"Open Knee(s) node set {name} is invalid")
    for name, surface in surfaces.items():
        if any(node not in all_node_ids for face in surface.faces for node in face):
            raise ValueError(f"Open Knee(s) surface {name} references an invalid node")
    if any(master not in surfaces or slave not in surfaces for _, master, slave in surface_pairs):
        raise ValueError("Open Knee(s) surface-pair reference is invalid")
    if enforce_exact:
        if set(regions) != set(EXPECTED_REGIONS):
            raise ValueError("Open Knee(s) exact region identity set drifted")
        for name, (nodes, kind, elements) in EXPECTED_REGIONS.items():
            region = regions[name]
            if (len(region.nodes_mm), region.element_type, len(region.elements)) != (
                nodes, kind, elements
            ):
                raise ValueError(f"Open Knee(s) exact region counts drifted for {name}")
        if len(node_sets) != 42 or len(surfaces) != 88 or len(surface_pairs) != 19:
            raise ValueError("Open Knee(s) exact attachment/contact topology drifted")
    return Source(
        regions, node_sets, surfaces, surface_pairs, landmarks, materials,
        fiber_directions, ET.parse(directory / "FeBio_custom.feb").getroot(),
    )


_SOURCE_SECTION_REASONS = {
    "Module": "the NHKNEE1 runtime does not execute FEBio module or solver settings",
    "Material": "selected scalar fields are copied to the reduced hybrid, not the complete FEBio material laws",
    "Geometry": "source coordinates and topology are compiled to an exact archive, but the native runtime does not execute the complete source geometry problem",
    "MeshData": "source element-wise fiber frames are compiled exactly but are not executed by NHKNEE1",
    "Boundary": "source rigid ties are reduced to hybrid attachment ownership rather than the source rigid-body graph",
    "Discrete": "source discrete springs and discrete ties are not executed by NHKNEE1",
    "LoadData": "source load curves and prestrain continuation are not executed by NHKNEE1",
    "Step": "source prescribed coordinates, contact enforcement, constraints and step controls are not executed by NHKNEE1",
    "Output": "source output requests are retained as provenance but have no native mechanics execution",
}
_EXPECTED_SOURCE_PROGRAM_COUNTS = {
    "materials": 21,
    "rigid_bodies": 9,
    "rigid_ties": 18,
    "cylindrical_joints": 6,
    "other_constraints": 1,
    "rigid_springs": 1,
    "prescribed_body_boundaries": 2,
    "contacts": 18,
    "load_curves": 9,
}


def _xml_digest(element: ET.Element) -> str:
    return hashlib.sha256(ET.tostring(element, encoding="utf-8")).hexdigest()


def _required_xml_text(element: ET.Element, tag: str, path: str) -> str:
    child = element.find(tag)
    if child is None or child.text is None or not child.text.strip():
        raise ValueError(f"Open Knee(s) source mechanics is missing {path}/{tag}")
    return child.text.strip()


def _required_xml_float(element: ET.Element, tag: str, path: str) -> float:
    text = _required_xml_text(element, tag, path)
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"Open Knee(s) source mechanics has invalid number at {path}/{tag}") from error
    if not math.isfinite(value):
        raise ValueError(f"Open Knee(s) source mechanics has non-finite number at {path}/{tag}")
    return value


def _required_xml_flag(element: ET.Element, tag: str, path: str) -> bool:
    value = _required_xml_float(element, tag, path)
    if value not in (0.0, 1.0):
        raise ValueError(f"Open Knee(s) source mechanics flag {path}/{tag} must be 0 or 1")
    return bool(value)


def _required_xml_vector(element: ET.Element, tag: str, path: str) -> tuple[float, float, float]:
    return _vector(_required_xml_text(element, tag, path), f"source mechanics {path}/{tag}")


def _source_curve_points(element: ET.Element) -> list[dict[str, Any]]:
    points = []
    previous_time: float | None = None
    for point in element.findall("point"):
        text = (point.text or "").strip()
        try:
            pair = tuple(float(value.strip()) for value in text.split(","))
        except ValueError as error:
            raise ValueError("Open Knee(s) source load curve has an invalid point") from error
        if len(pair) != 2 or not all(math.isfinite(value) for value in pair):
            raise ValueError("Open Knee(s) source load curve point must be a finite time/value pair")
        if previous_time is not None and pair[0] <= previous_time:
            raise ValueError("Open Knee(s) source load curve times must increase strictly")
        previous_time = pair[0]
        points.append({
            "source_text": text,
            "time": pair[0],
            "value": pair[1],
            "attributes": dict(sorted(point.attrib.items())),
        })
    if not points:
        raise ValueError("Open Knee(s) source load curve has no points")
    return points


def _geometry_archive_cross_references(
    path: Path,
    *,
    expected_sha256: str | None,
    contact_pair_names: set[str],
    rigid_tie_node_sets: set[str],
    mesh_data_element_counts: dict[str, int] | None = None,
    source_material_ids: set[int] | None = None,
    binary_output_path: Path | None = None,
    source_volume_mesh_output_path: Path | None = None,
) -> dict[str, Any]:
    """Stream archived geometry and resolve source contact/tie/material IDs.

    The archive is over 100 MB. Only node IDs, requested rigid-tie members,
    surface and element connectivity digests/counts, and named pair links are retained.
    This validates references without substituting the smaller registered
    Geometry.feb or materializing a second full mesh in memory.
    """
    if not path.is_file():
        raise ValueError("Open Knee(s) archived source geometry file is absent")
    geometry_sha256 = _sha256(path)
    if expected_sha256 is not None and geometry_sha256 != expected_sha256:
        raise ValueError("Open Knee(s) archived Geometry_custom identity drifted")

    store_binary = binary_output_path is not None
    geometry_payload = bytearray()
    geometry_payload_size = 0

    def append_geometry_record(payload: bytes) -> int:
        nonlocal geometry_payload_size
        offset = geometry_payload_size
        if store_binary:
            geometry_payload.extend(payload)
        geometry_payload_size += len(payload)
        return offset

    node_ids: set[int] = set()
    referenced_node_ids: set[int] = set()
    node_coordinate_groups: dict[str, dict[str, Any]] = {}
    node_sets: dict[str, dict[str, Any]] = {}
    surfaces: dict[str, dict[str, Any]] = {}
    surface_pairs: dict[str, dict[str, str]] = {}
    element_sets: dict[str, dict[str, Any]] = {}
    source_mesh_node_groups: dict[str, dict[str, Any]] = {}
    source_mesh_element_groups: dict[str, dict[str, Any]] = {}
    active: dict[int, dict[str, Any]] = {}
    stack: list[ET.Element] = []
    node_record_count = 0

    def local_tag(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    def record_number(text: str | None, label: str) -> int:
        try:
            value = int((text or "").strip())
        except ValueError as error:
            raise ValueError(f"Open Knee(s) archived geometry has invalid {label}") from error
        if value <= 0:
            raise ValueError(f"Open Knee(s) archived geometry has nonpositive {label}")
        return value

    try:
        events = ET.iterparse(path, events=("start", "end"))
        for event, element in events:
            tag = local_tag(element)
            if event == "start":
                parent = stack[-1] if stack else None
                parent_tag = local_tag(parent) if parent is not None else ""
                if parent_tag == "Geometry" and tag in {
                    "Nodes", "NodeSet", "Surface", "SurfacePair", "Elements"
                }:
                    name = element.attrib.get("name")
                    if not name:
                        raise ValueError(
                            f"Open Knee(s) archived geometry {tag} has no name"
                        )
                    if tag == "Nodes":
                        active[id(element)] = {
                            "kind": tag, "name": name, "count": 0,
                            "offset": geometry_payload_size,
                            "source_mesh_payload": bytearray()
                            if source_volume_mesh_output_path is not None else None,
                            "source_mesh_node_ids": set()
                            if source_volume_mesh_output_path is not None else None,
                        }
                    elif tag == "NodeSet":
                        active[id(element)] = {
                            "kind": tag, "name": name, "count": 0,
                            "ids": [] if name in rigid_tie_node_sets else None,
                            "offset": geometry_payload_size,
                            "digest": hashlib.sha256(),
                        }
                    elif tag == "Surface":
                        active[id(element)] = {
                            "kind": tag, "name": name, "count": 0,
                            "offset": geometry_payload_size,
                            "face_type": None,
                            "digest": hashlib.sha256(),
                        }
                    elif tag == "Elements":
                        element_type = element.attrib.get("type")
                        arity = {"tet4": 4, "tri3": 3}.get(element_type)
                        if arity is None:
                            raise ValueError(
                                f"Open Knee(s) archived geometry uses unsupported element type {element_type}"
                            )
                        active[id(element)] = {
                            "kind": tag, "name": name, "type": element_type,
                            "arity": arity, "offset": geometry_payload_size,
                            "material_id_source_text": element.attrib.get("mat"),
                            "attributes": dict(sorted(element.attrib.items())),
                            "count": 0, "ids": set(), "first_id": None, "last_id": None,
                            "ids_follow_source_order": True,
                            "source_mesh_payload": bytearray()
                            if source_volume_mesh_output_path is not None and
                            element_type == "tet4" else None,
                            "source_mesh_node_ids": set()
                            if source_volume_mesh_output_path is not None and
                            element_type == "tet4" else None,
                            "digest": hashlib.sha256(),
                        }
                    else:
                        active[id(element)] = {"kind": tag, "name": name}
                elif parent_tag == "SurfacePair" and tag in {"master", "slave"}:
                    group = active.get(id(parent))
                    if group is None or group["kind"] != "SurfacePair":
                        raise ValueError(
                            "Open Knee(s) archived geometry has malformed SurfacePair nesting"
                        )
                    target = element.attrib.get("surface")
                    if not target or tag in group:
                        raise ValueError(
                            "Open Knee(s) archived SurfacePair reference is missing or duplicated"
                        )
                    group[tag] = target
                stack.append(element)
                continue

            parent = stack[-2] if len(stack) > 1 else None
            parent_tag = local_tag(parent) if parent is not None else ""
            if parent_tag == "Nodes" and tag == "node":
                group = active.get(id(parent))
                node_id = record_number(element.attrib.get("id"), "node ID")
                try:
                    coordinates = tuple(
                        float(value.strip()) for value in (element.text or "").split(",")
                    )
                except ValueError as error:
                    raise ValueError(
                        "Open Knee(s) archived geometry has invalid node coordinates"
                    ) from error
                if len(coordinates) != 3 or not all(
                    math.isfinite(value) for value in coordinates
                ):
                    raise ValueError(
                        "Open Knee(s) archived geometry node must have three finite coordinates"
                    )
                node_ids.add(node_id)
                node_record_count += 1
                if group is not None:
                    group["count"] += 1
                    node_record = struct.pack("<I3d", node_id, *coordinates)
                    append_geometry_record(node_record)
                    if group["source_mesh_payload"] is not None:
                        group["source_mesh_payload"].extend(node_record)
                        group["source_mesh_node_ids"].add(node_id)
            elif parent_tag == "NodeSet":
                group = active.get(id(parent))
                if group is not None:
                    if tag != "node":
                        if group["ids"] is not None:
                            raise ValueError(
                                "Open Knee(s) archived rigid-tie node set uses an unsupported member"
                            )
                    else:
                        node_id = record_number(element.attrib.get("id"), "node-set node ID")
                        group["count"] += 1
                        referenced_node_ids.add(node_id)
                        group["digest"].update(f"{node_id}\n".encode("ascii"))
                        append_geometry_record(struct.pack("<I", node_id))
                        if group["ids"] is not None:
                            group["ids"].append(node_id)
            elif parent_tag == "Surface":
                group = active.get(id(parent))
                if group is not None:
                    text = (element.text or "").strip()
                    try:
                        face_ids = tuple(int(value.strip()) for value in text.split(","))
                    except ValueError as error:
                        raise ValueError(
                            "Open Knee(s) archived source surface has invalid connectivity"
                        ) from error
                    if not face_ids or any(value <= 0 for value in face_ids):
                        raise ValueError(
                            "Open Knee(s) archived source surface has invalid node IDs"
                        )
                    arity = {"tri3": 3, "quad4": 4}.get(tag)
                    if arity is None or len(face_ids) != arity:
                        raise ValueError(
                            f"Open Knee(s) archived source surface has unsupported face type {tag}"
                        )
                    if group["face_type"] is None:
                        group["face_type"] = tag
                    elif group["face_type"] != tag:
                        raise ValueError(
                            "Open Knee(s) archived source surface mixes face types"
                        )
                    group["count"] += 1
                    referenced_node_ids.update(face_ids)
                    append_geometry_record(struct.pack(
                        "<I" + ("I" * arity),
                        record_number(element.attrib.get("id"), "surface face ID"),
                        *face_ids,
                    ))
                    encoded = ",".join(str(value) for value in face_ids)
                    group["digest"].update(
                        f"{tag}:{element.attrib.get('id', '')}:{encoded}\n".encode("ascii")
                    )
            elif parent_tag == "Elements" and tag == "elem":
                group = active.get(id(parent))
                if group is not None:
                    element_id = record_number(
                        element.attrib.get("id"), "element ID"
                    )
                    if element_id in group["ids"]:
                        raise ValueError(
                            "Open Knee(s) archived geometry has duplicate element IDs in a set"
                        )
                    try:
                        node_references = tuple(
                            int(value.strip())
                            for value in (element.text or "").split(",")
                        )
                    except ValueError as error:
                        raise ValueError(
                            "Open Knee(s) archived element has invalid connectivity"
                        ) from error
                    if (
                        len(node_references) != group["arity"] or
                        any(value <= 0 for value in node_references)
                    ):
                        raise ValueError(
                            "Open Knee(s) archived element has invalid connectivity"
                        )
                    group["count"] += 1
                    if group["first_id"] is None:
                        group["first_id"] = element_id
                    elif element_id != group["last_id"] + 1:
                        group["ids_follow_source_order"] = False
                    group["last_id"] = element_id
                    group["ids"].add(element_id)
                    referenced_node_ids.update(node_references)
                    element_record = struct.pack(
                        "<I" + ("I" * group["arity"]), element_id,
                        *node_references,
                    )
                    append_geometry_record(element_record)
                    if group["source_mesh_payload"] is not None:
                        group["source_mesh_payload"].extend(element_record)
                        group["source_mesh_node_ids"].update(node_references)
                    encoded = ",".join(str(value) for value in node_references)
                    group["digest"].update(
                        f"{element_id}:{encoded}\n".encode("ascii")
                    )

            if tag in {"Nodes", "NodeSet", "Surface", "SurfacePair", "Elements"} and id(element) in active:
                group = active.pop(id(element))
                name = group["name"]
                if group["kind"] == "Nodes":
                    if name in node_coordinate_groups:
                        raise ValueError(
                            "Open Knee(s) archived geometry has duplicate Nodes names"
                        )
                    node_coordinate_groups[name] = {
                        "node_count": group["count"],
                        "binary_offset_bytes": group["offset"],
                        "binary_record_stride_bytes": 28,
                        "binary_bytes": group["count"] * 28,
                        "binary_record_layout": "little-endian u32 node_id, then float64[3] coordinates in source units",
                    }
                    if group["source_mesh_payload"] is not None:
                        source_mesh_node_groups[name] = {
                            "count": group["count"],
                            "payload": group["source_mesh_payload"],
                            "node_ids": group["source_mesh_node_ids"],
                        }
                elif group["kind"] == "NodeSet":
                    if name in node_sets:
                        raise ValueError(
                            "Open Knee(s) archived geometry has duplicate NodeSet names"
                        )
                    node_sets[name] = {
                        "node_count": group["count"],
                        "node_ids_sha256": group["digest"].hexdigest(),
                        "binary_offset_bytes": group["offset"],
                        "binary_record_stride_bytes": 4,
                        "binary_bytes": group["count"] * 4,
                        "binary_record_layout": "little-endian u32 node_id",
                    }
                    if group["ids"] is not None:
                        node_sets[name]["node_ids"] = group["ids"]
                elif group["kind"] == "Surface":
                    if name in surfaces:
                        raise ValueError(
                            "Open Knee(s) archived geometry has duplicate Surface names"
                        )
                    surfaces[name] = {
                        "face_count": group["count"],
                        "face_type": group["face_type"],
                        "binary_offset_bytes": group["offset"],
                        "binary_record_stride_bytes": (
                            16 if group["face_type"] == "tri3" else 20
                        ),
                        "binary_bytes": group["count"] * (
                            16 if group["face_type"] == "tri3" else 20
                        ),
                        "binary_record_layout": (
                            "little-endian u32 face_id and u32[3] node_ids"
                            if group["face_type"] == "tri3"
                            else "little-endian u32 face_id and u32[4] node_ids"
                        ),
                        "connectivity_sha256": group["digest"].hexdigest(),
                    }
                elif group["kind"] == "Elements":
                    if name in element_sets:
                        raise ValueError(
                            "Open Knee(s) archived geometry has duplicate Elements names"
                        )
                    material_id_source_text = group["material_id_source_text"]
                    material_id = (
                        record_number(material_id_source_text, "element material ID")
                        if material_id_source_text is not None else None
                    )
                    if expected_sha256 is not None and material_id is None:
                        raise ValueError(
                            f"Open Knee(s) archived element set {name} has no material assignment"
                        )
                    if (
                        material_id is not None and source_material_ids is not None and
                        material_id not in source_material_ids
                    ):
                        raise ValueError(
                            f"Open Knee(s) archived element set {name} references unknown source material {material_id}"
                        )
                    element_sets[name] = {
                        "element_type": group["type"],
                        "element_count": group["count"],
                        "binary_offset_bytes": group["offset"],
                        "binary_record_stride_bytes": 4 * (1 + group["arity"]),
                        "binary_bytes": group["count"] * 4 * (1 + group["arity"]),
                        "binary_record_layout": (
                            "little-endian u32 element_id and u32[4] node_ids"
                            if group["type"] == "tet4" else
                            "little-endian u32 element_id and u32[3] node_ids"
                        ),
                        "first_element_id_in_source_order": group["first_id"],
                        "last_element_id_in_source_order": group["last_id"],
                        "element_ids_contiguous_in_source_order": group[
                            "ids_follow_source_order"
                        ],
                        "material_id": material_id,
                        "material_id_source_text": material_id_source_text,
                        "source_attributes": group["attributes"],
                        "connectivity_sha256": group["digest"].hexdigest(),
                    }
                    if group["source_mesh_payload"] is not None:
                        source_mesh_element_groups[name] = {
                            "count": group["count"],
                            "payload": group["source_mesh_payload"],
                            "node_ids": group["source_mesh_node_ids"],
                        }
                else:
                    if name in surface_pairs:
                        raise ValueError(
                            "Open Knee(s) archived geometry has duplicate SurfacePair names"
                        )
                    if "master" not in group or "slave" not in group:
                        raise ValueError(
                            "Open Knee(s) archived SurfacePair lacks master or slave"
                        )
                    surface_pairs[name] = {
                        "master_surface": group["master"],
                        "slave_surface": group["slave"],
                    }

            if stack and stack[-1] is element:
                stack.pop()
            if parent is not None:
                parent.remove(element)
            element.clear()
    except ET.ParseError as error:
        raise ValueError("Open Knee(s) archived Geometry_custom XML is invalid") from error

    if not node_ids or node_record_count != len(node_ids):
        raise ValueError("Open Knee(s) archived geometry node IDs are empty or duplicated")
    if not referenced_node_ids.issubset(node_ids):
        raise ValueError("Open Knee(s) archived geometry references a missing mesh node")
    for name in rigid_tie_node_sets:
        record = node_sets.get(name)
        if record is None or record["node_count"] == 0 or "node_ids" not in record:
            raise ValueError(
                f"Open Knee(s) archived geometry is missing rigid-tie NodeSet {name}"
            )
        if not set(record["node_ids"]).issubset(node_ids):
            raise ValueError(
                f"Open Knee(s) archived rigid-tie NodeSet {name} refers to a missing node"
            )

    resolved_mesh_data_element_sets = {}
    for name, expected_count in sorted((mesh_data_element_counts or {}).items()):
        record = element_sets.get(name)
        if record is None or record["element_count"] != expected_count:
            raise ValueError(
                f"Open Knee(s) archived geometry element set {name} does not match MeshData"
            )
        if expected_sha256 is not None and not record[
            "element_ids_contiguous_in_source_order"
        ]:
            raise ValueError(
                f"Open Knee(s) archived geometry element set {name} does not preserve MeshData local-ID order"
            )
        resolved_mesh_data_element_sets[name] = record

    resolved_pairs: dict[str, dict[str, Any]] = {}
    for name in sorted(contact_pair_names):
        pair = surface_pairs.get(name)
        if pair is None:
            raise ValueError(
                f"Open Knee(s) archived geometry is missing contact SurfacePair {name}"
            )
        master, slave = pair["master_surface"], pair["slave_surface"]
        if master not in surfaces or slave not in surfaces:
            raise ValueError(
                f"Open Knee(s) archived contact pair {name} references an unknown Surface"
            )
        resolved_pairs[name] = {
            **pair,
            "master_face_count": surfaces[master]["face_count"],
            "master_connectivity_sha256": surfaces[master]["connectivity_sha256"],
            "slave_face_count": surfaces[slave]["face_count"],
            "slave_connectivity_sha256": surfaces[slave]["connectivity_sha256"],
        }
    binary_storage = None
    if binary_output_path is not None:
        if len(geometry_payload) != geometry_payload_size:
            raise ValueError("Open Knee(s) source geometry binary size accounting drifted")
        binary_output_path.parent.mkdir(parents=True, exist_ok=True)
        binary_output_path.write_bytes(geometry_payload)
        binary_storage = {
            "schema": "numi.human.open-knee-source-geometry-binary.v1",
            "file": binary_output_path.name,
            "bytes": geometry_payload_size,
            "sha256": hashlib.sha256(geometry_payload).hexdigest(),
            "record_order": "source XML Geometry child order; per-group records retain source order",
        }
    source_volume_mesh_storage = None
    if source_volume_mesh_output_path is not None:
        volume_payload = bytearray()
        volume_groups = {}
        volume_set_names = sorted(
            (
                name for name, record in element_sets.items()
                if record["element_type"] == "tet4"
            ),
            key=lambda key: (
                element_sets[key]["material_id"] is None,
                element_sets[key]["material_id"] or 0,
            ),
        )
        for name in volume_set_names:
            node_group = source_mesh_node_groups.get(name)
            element_group = source_mesh_element_groups.get(name)
            element_set = element_sets.get(name)
            if (
                node_group is None or element_group is None or element_set is None or
                element_set["element_type"] != "tet4" or
                element_group["count"] != element_set["element_count"] or
                node_group["count"] != node_coordinate_groups[name]["node_count"]
            ):
                raise ValueError(
                    f"Open Knee(s) source volume mesh {name} is incomplete or mismatched"
                )
            if not element_group["node_ids"].issubset(node_group["node_ids"]):
                raise ValueError(
                    f"Open Knee(s) source volume mesh {name} references nodes outside its named part"
                )
            material_id = element_set["material_id"]
            if material_id is None:
                raise ValueError(
                    f"Open Knee(s) source volume mesh {name} has no material ID"
                )
            header_offset = len(volume_payload)
            volume_payload.extend(struct.pack(
                "<4sIIII", b"NOKT", 1, material_id,
                node_group["count"], element_group["count"],
            ))
            node_offset = len(volume_payload)
            volume_payload.extend(node_group["payload"])
            element_offset = len(volume_payload)
            volume_payload.extend(element_group["payload"])
            volume_groups[name] = {
                "material_id": material_id,
                "node_count": node_group["count"],
                "tetrahedron_count": element_group["count"],
                "header_offset_bytes": header_offset,
                "node_records_offset_bytes": node_offset,
                "node_record_stride_bytes": 28,
                "tetrahedron_records_offset_bytes": element_offset,
                "tetrahedron_record_stride_bytes": 20,
                "binary_bytes": len(volume_payload) - header_offset,
                "source_coordinates": "preserved without scaling",
                "node_id_mapping": "source global node IDs preserved in both tables",
                "fiber_mapping": (
                    "source ElementData local lid maps to one-based tetrahedron order within the element set"
                    if name in (mesh_data_element_counts or {}) else
                    "source material fiber direction is compiled separately"
                ),
            }
        if expected_sha256 is not None and len(volume_groups) != 12:
            raise ValueError(
                "Open Knee(s) pinned source volume mesh must contain 12 tet4 element sets"
            )
        source_volume_mesh_output_path.parent.mkdir(parents=True, exist_ok=True)
        source_volume_mesh_output_path.write_bytes(volume_payload)
        source_volume_mesh_storage = {
            "schema": "numi.human.open-knee-source-volume-mesh.v1",
            "file": source_volume_mesh_output_path.name,
            "bytes": len(volume_payload),
            "sha256": hashlib.sha256(volume_payload).hexdigest(),
            "header_layout": "little-endian 4-byte NOKT magic, u32 version, u32 source material ID, u32 node count, u32 tetrahedron count",
            "groups": volume_groups,
        }
    return {
        "status": "resolved_against_pinned_Geometry_custom",
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "sha256": geometry_sha256,
        "coordinate_units": "not_declared_in_pinned_FEBio_source",
        "coordinate_binary_semantics": "source numeric coordinates preserved as float64 without scaling",
        "node_count": len(node_ids),
        "node_coordinate_groups": {
            name: node_coordinate_groups[name] for name in sorted(node_coordinate_groups)
        },
        "node_set_count": len(node_sets),
        "node_sets": {name: node_sets[name] for name in sorted(node_sets)},
        "surface_count": len(surfaces),
        "surfaces": {name: surfaces[name] for name in sorted(surfaces)},
        "surface_pair_count": len(surface_pairs),
        "element_set_count": len(element_sets),
        "rigid_tie_node_sets": {
            name: node_sets[name] for name in sorted(rigid_tie_node_sets)
        },
        "contact_surface_pairs": resolved_pairs,
        "element_sets": {name: element_sets[name] for name in sorted(element_sets)},
        "mesh_data_element_sets": resolved_mesh_data_element_sets,
        "binary_storage": binary_storage,
        "source_volume_mesh_storage": source_volume_mesh_storage,
        "unreferenced_surface_pair_names": sorted(
            set(surface_pairs) - contact_pair_names
        ),
    }


def _xml_parameter_records(element: ET.Element) -> list[dict[str, Any]]:
    return [
        {
            "name": child.tag,
            "source_text": (child.text or "").strip(),
            "attributes": dict(sorted(child.attrib.items())),
        }
        for child in element
    ]


def _typed_source_xml(element: ET.Element) -> dict[str, Any]:
    """Retain a source XML subtree while exposing finite scalar/vector leaves."""
    text = (element.text or "").strip()
    record: dict[str, Any] = {
        "tag": element.tag,
        "attributes": dict(sorted(element.attrib.items())),
        "source_text": text,
        "children": [_typed_source_xml(child) for child in element],
    }
    if text and not record["children"]:
        fields = [field.strip() for field in text.split(",")]
        try:
            numeric = [float(field) for field in fields]
        except ValueError:
            return record
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(
                f"Open Knee(s) source {element.tag} contains a non-finite number"
            )
        if len(numeric) == 1:
            record["numeric_value"] = numeric[0]
        else:
            record["numeric_vector"] = numeric
    return record


def _source_material_program(
    element: ET.Element,
    *,
    curve_ids: set[str],
    require_open_knee_tissue_schema: bool,
) -> dict[str, Any]:
    def scalar_parameters(parent: ET.Element | None) -> dict[str, float]:
        if parent is None:
            return {}
        result: dict[str, float] = {}
        for child in parent:
            if list(child):
                continue
            text = (child.text or "").strip()
            if not text or "," in text:
                continue
            try:
                value = float(text)
            except ValueError:
                continue
            if not math.isfinite(value):
                raise ValueError(
                    f"Open Knee(s) source material has non-finite {child.tag}"
                )
            result[child.tag] = value
        return result

    material_type = element.attrib.get("type", "")
    elastic = element.find("elastic")
    prestrain_element = element.find("prestrain")
    fiber = element.find("fiber")
    fiber_direction: list[float] | None = None
    if fiber is not None:
        try:
            fiber_direction = [
                float(value.strip()) for value in (fiber.text or "").split(",")
            ]
        except ValueError as error:
            raise ValueError("Open Knee(s) source material fiber vector is invalid") from error
        if len(fiber_direction) != 3 or not all(
            math.isfinite(value) for value in fiber_direction
        ):
            raise ValueError("Open Knee(s) source material fiber vector is invalid")
        length = math.sqrt(sum(value * value for value in fiber_direction))
        if length <= 1.0e-12:
            raise ValueError("Open Knee(s) source material fiber vector is degenerate")

    prestrain: dict[str, Any] | None = None
    if prestrain_element is not None:
        stretch = prestrain_element.find("stretch")
        if stretch is None:
            raise ValueError("Open Knee(s) source prestrain has no stretch definition")
        try:
            stretch_value = float((stretch.text or "").strip())
        except ValueError as error:
            raise ValueError("Open Knee(s) source prestrain stretch is invalid") from error
        if not math.isfinite(stretch_value) or stretch_value <= 0.0:
            raise ValueError("Open Knee(s) source prestrain stretch is nonpositive or non-finite")
        curve_id = stretch.attrib.get("lc")
        if curve_id is not None and curve_id not in curve_ids:
            raise ValueError("Open Knee(s) source prestrain references an unknown load curve")
        prestrain = {
            "type": prestrain_element.attrib.get("type"),
            "stretch": stretch_value,
            "load_curve_id": curve_id,
            "isochoric": _required_xml_flag(
                prestrain_element, "isochoric", "Material/material/prestrain"
            ),
        }

    outer_parameters = scalar_parameters(element)
    elastic_parameters = scalar_parameters(elastic)
    if require_open_knee_tissue_schema and material_type == "uncoupled prestrain elastic":
        if elastic is None or elastic.attrib.get("type") != "trans iso Mooney-Rivlin":
            raise ValueError(
                "Open Knee(s) ligament/tendon material has an unsupported elastic law"
            )
        required = {"density", "c1", "c2", "c3", "c4", "c5", "lam_max"}
        if required - set(elastic_parameters):
            raise ValueError(
                "Open Knee(s) ligament/tendon elastic parameters are incomplete"
            )
        if "k" not in outer_parameters or fiber_direction is None or prestrain is None:
            raise ValueError(
                "Open Knee(s) ligament/tendon bulk, fiber, or prestrain definition is missing"
            )
        if prestrain["type"] != "in-situ stretch":
            raise ValueError(
                "Open Knee(s) ligament/tendon has an unsupported prestrain type"
            )
        length = math.sqrt(sum(value * value for value in fiber_direction))
        if not 0.99999 <= length <= 1.00001:
            raise ValueError("Open Knee(s) source fiber direction must be unit length")

    return {
        "material_type": material_type,
        "elastic_type": None if elastic is None else elastic.attrib.get("type"),
        "outer_scalar_parameters": outer_parameters,
        "elastic_scalar_parameters": elastic_parameters,
        "fiber_direction_source_text": None if fiber is None else (fiber.text or "").strip(),
        "fiber_direction": fiber_direction,
        "prestrain": prestrain,
        "source_xml_tree": _typed_source_xml(element),
    }


def _source_mesh_element_data(
    root: ET.Element,
) -> tuple[list[dict[str, Any]], bytes, dict[str, int]]:
    records: list[dict[str, Any]] = []
    packed_records = bytearray()
    element_counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for element_data in root.findall("MeshData/ElementData"):
        variable = element_data.attrib.get("var")
        element_set = element_data.attrib.get("elem_set")
        if not variable or not element_set:
            raise ValueError("Open Knee(s) source ElementData lacks var or elem_set")
        key = (variable, element_set)
        if key in seen:
            raise ValueError("Open Knee(s) source ElementData has duplicate var/elem_set")
        seen.add(key)
        if variable != "fiber":
            records.append({
                "variable": variable,
                "element_set": element_set,
                "source_xml_sha256": _xml_digest(element_data),
                "native_execution_status": "unsupported_not_executed",
            })
            continue

        offset = len(packed_records)
        values_hash = hashlib.sha256()
        minimum_norm = math.inf
        maximum_norm = 0.0
        previous_id = 0
        for index, element in enumerate(element_data, 1):
            if element.tag != "elem":
                raise ValueError("Open Knee(s) source fiber ElementData uses unsupported rows")
            try:
                local_id = int(element.attrib.get("lid", ""))
                vector = tuple(
                    float(value.strip()) for value in (element.text or "").split(",")
                )
            except ValueError as error:
                raise ValueError("Open Knee(s) source element fiber is invalid") from error
            if local_id != index or local_id <= previous_id or len(vector) != 3 or not all(
                math.isfinite(value) for value in vector
            ):
                raise ValueError(
                    "Open Knee(s) source fiber rows must be finite, three-component, and consecutively indexed"
                )
            previous_id = local_id
            packed = struct.pack("<I3d", local_id, *vector)
            packed_records.extend(packed)
            values_hash.update(packed)
            vector_norm = math.hypot(*vector)
            if not math.isfinite(vector_norm) or vector_norm <= 1.0e-12:
                raise ValueError("Open Knee(s) source element fiber direction is degenerate")
            minimum_norm = min(minimum_norm, vector_norm)
            maximum_norm = max(maximum_norm, vector_norm)
        if previous_id == 0:
            raise ValueError("Open Knee(s) source fiber ElementData is empty")
        element_counts[element_set] = previous_id
        records.append({
            "variable": variable,
            "element_set": element_set,
            "record_count": previous_id,
            "local_element_ids": "one_based_contiguous",
            "raw_source_vectors_preserved": True,
            "vector_norm_range": [minimum_norm, maximum_norm],
            "binary_offset_bytes": offset,
            "binary_bytes": previous_id * 28,
            "binary_sha256": values_hash.hexdigest(),
            "record_layout": "little-endian u32 local_element_id, then float64[3] source vector",
            "source_xml_sha256": _xml_digest(element_data),
            "native_execution_status": "compiled_source_input_not_executed",
        })
    return records, bytes(packed_records), element_counts


def _isochoric_fiber_prestrain_tensor(
    stretch: float, fiber_direction: list[float]
) -> list[list[float]]:
    if not math.isfinite(stretch) or stretch <= 0.0:
        raise ValueError("Open Knee(s) source prestrain stretch is invalid")
    norm = math.sqrt(sum(value * value for value in fiber_direction))
    if not 0.99999 <= norm <= 1.00001:
        raise ValueError("Open Knee(s) source prestrain fiber direction is not unit length")
    transverse = stretch ** -0.5
    return [
        [
            transverse * (1.0 if row == column else 0.0) +
            (stretch - transverse) * fiber_direction[row] * fiber_direction[column]
            for column in range(3)
        ]
        for row in range(3)
    ]


def compile_source_mechanical_description(
    source: Source,
    *,
    source_deck: Path,
    expected_deck_sha256: str | None = None,
    source_geometry_archive_sha256: str | None = None,
    source_geometry_archive_path: Path | None = None,
    source_geometry_binary_output_path: Path | None = None,
    source_volume_mesh_output_path: Path | None = None,
    source_mesh_data_output_path: Path | None = None,
    reference_solver_version: str | None = None,
    reference_log_sha256: str | None = None,
) -> dict[str, Any]:
    """Compile an auditable inventory of the complete source mechanics program.

    The result deliberately records which constructs the current NHKNEE1 path
    rejects. Retaining source XML or selected scalar parameters is not execution.
    Geometry remains in the source coordinate system in this descriptor.
    """
    root = source.mechanical_program
    if root is None or root.tag != "febio_spec":
        raise ValueError("Open Knee(s) source mechanics requires a febio_spec root")
    deck_bytes = source_deck.read_bytes()
    deck_sha256 = hashlib.sha256(deck_bytes).hexdigest()
    if expected_deck_sha256 is not None and deck_sha256 != expected_deck_sha256:
        raise ValueError("Open Knee(s) FEBio mechanical program identity drifted")
    deck_root = ET.fromstring(deck_bytes)
    if ET.tostring(deck_root) != ET.tostring(root):
        raise ValueError("Open Knee(s) source program tree differs from the hashed FEBio deck")

    curve_ids = {
        item.attrib.get("id") for item in root.findall("LoadData/loadcurve")
    }

    sections: list[dict[str, Any]] = []
    unsupported: list[dict[str, str]] = []
    for section in root:
        reason = _SOURCE_SECTION_REASONS.get(
            section.tag,
            "source section has no exact NHKNEE1 implementation and is rejected",
        )
        status = (
            "nonmechanical_provenance_only"
            if section.tag == "Output"
            else "unsupported_not_executed"
        )
        tag_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for item in section.iter():
            tag_counts[item.tag] = tag_counts.get(item.tag, 0) + 1
            item_type = item.attrib.get("type")
            if item_type is not None:
                key = f"{item.tag}:{item_type}"
                type_counts[key] = type_counts.get(key, 0) + 1
        sections.append({
            "name": section.tag,
            "attributes": dict(sorted(section.attrib.items())),
            "element_count_including_section": sum(tag_counts.values()),
            "tag_counts": dict(sorted(tag_counts.items())),
            "typed_construct_counts": dict(sorted(type_counts.items())),
            "source_subtree_sha256": _xml_digest(section),
            "native_execution_status": status,
            "reason": reason,
        })
        if status == "unsupported_not_executed":
            unsupported.append({
                "source_path": section.tag,
                "status": status,
                "reason": reason,
            })

    materials = []
    for item in root.findall("Material/material"):
        materials.append({
            "id": item.attrib.get("id"),
            "name": item.attrib.get("name"),
            "type": item.attrib.get("type"),
            "source_xml_sha256": _xml_digest(item),
            "source_xml": ET.tostring(item, encoding="unicode"),
            "native_execution_status": "unsupported_not_executed",
        })
    source_material_ids = {int(material["id"]) for material in materials}
    if len(source_material_ids) != len(materials):
        raise ValueError("Open Knee(s) source material IDs are missing or duplicated")

    rigid_bodies = []
    for item in root.findall("Material/material"):
        if item.attrib.get("type") != "rigid body":
            continue
        rigid_bodies.append({
            "material_id": int(item.attrib["id"]),
            "name": item.attrib.get("name"),
            "center_of_mass_source_text": _required_xml_text(
                item, "center_of_mass", "Material/material"
            ),
            "center_of_mass": list(_required_xml_vector(
                item, "center_of_mass", "Material/material"
            )),
            "density_source_text": _required_xml_text(
                item, "density", "Material/material"
            ),
            "density": _required_xml_float(item, "density", "Material/material"),
            "source_xml_sha256": _xml_digest(item),
        })
    body_ids = [body["material_id"] for body in rigid_bodies]
    if len(body_ids) != len(set(body_ids)):
        raise ValueError("Open Knee(s) source rigid-body IDs are duplicated")
    cylindrical_joints = []
    rigid_springs = []
    other_constraints = []
    for item in root.findall("Step/Constraints/constraint"):
        kind = item.attrib.get("type", "")
        record: dict[str, Any] = {
            "name": item.attrib.get("name"),
            "type": kind,
            "source_xml_sha256": _xml_digest(item),
            "source_xml": ET.tostring(item, encoding="unicode"),
            "native_execution_status": "unsupported_not_executed",
        }
        if kind == "rigid cylindrical joint":
            axis = _required_xml_vector(item, "joint_axis", "Step/Constraints/constraint")
            axis_length = math.sqrt(sum(value * value for value in axis))
            if not 0.99999 <= axis_length <= 1.00001:
                raise ValueError("Open Knee(s) source cylindrical joint axis must be unit length")
            translation_element = item.find("translation")
            rotation_element = item.find("rotation")
            if translation_element is None or rotation_element is None:
                raise ValueError("Open Knee(s) source cylindrical joint coordinate is incomplete")
            translation_text = _required_xml_text(item, "translation", "Step/Constraints/constraint")
            rotation_text = _required_xml_text(item, "rotation", "Step/Constraints/constraint")
            translation_attributes = dict(sorted(translation_element.attrib.items()))
            rotation_attributes = dict(sorted(rotation_element.attrib.items()))
            record.update({
                "body_a": int(_required_xml_text(item, "body_a", "Step/Constraints/constraint")),
                "body_b": int(_required_xml_text(item, "body_b", "Step/Constraints/constraint")),
                "force_penalty_source_text": _required_xml_text(item, "force_penalty", "Step/Constraints/constraint"),
                "force_penalty": _required_xml_float(item, "force_penalty", "Step/Constraints/constraint"),
                "moment_penalty_source_text": _required_xml_text(item, "moment_penalty", "Step/Constraints/constraint"),
                "moment_penalty": _required_xml_float(item, "moment_penalty", "Step/Constraints/constraint"),
                "joint_origin_source_text": _required_xml_text(item, "joint_origin", "Step/Constraints/constraint"),
                "joint_origin": list(_required_xml_vector(item, "joint_origin", "Step/Constraints/constraint")),
                "joint_axis_source_text": _required_xml_text(item, "joint_axis", "Step/Constraints/constraint"),
                "joint_axis": list(axis),
                "prescribed_translation_source_text": _required_xml_text(item, "prescribed_translation", "Step/Constraints/constraint"),
                "prescribed_translation": _required_xml_flag(item, "prescribed_translation", "Step/Constraints/constraint"),
                "translation": {
                    "source_text": translation_text,
                    "value": _required_xml_float(item, "translation", "Step/Constraints/constraint"),
                    "load_curve_id": translation_attributes.get("lc"),
                    "attributes": translation_attributes,
                },
                "prescribed_rotation_source_text": _required_xml_text(item, "prescribed_rotation", "Step/Constraints/constraint"),
                "prescribed_rotation": _required_xml_flag(item, "prescribed_rotation", "Step/Constraints/constraint"),
                "rotation": {
                    "source_text": rotation_text,
                    "value": _required_xml_float(item, "rotation", "Step/Constraints/constraint"),
                    "load_curve_id": rotation_attributes.get("lc"),
                    "attributes": rotation_attributes,
                },
                "force": None if item.find("force") is None else {
                    "source_text": (item.findtext("force") or "").strip(),
                    "attributes": dict(sorted(item.find("force").attrib.items())),
                },
                "moment": None if item.find("moment") is None else {
                    "source_text": (item.findtext("moment") or "").strip(),
                    "attributes": dict(sorted(item.find("moment").attrib.items())),
                },
                "minaug_source_text": _required_xml_text(item, "minaug", "Step/Constraints/constraint"),
                "maxaug_source_text": _required_xml_text(item, "maxaug", "Step/Constraints/constraint"),
            })
            if record["body_a"] not in body_ids or record["body_b"] not in body_ids:
                raise ValueError("Open Knee(s) cylindrical joint references an unknown rigid body")
            if record["body_a"] == record["body_b"]:
                raise ValueError("Open Knee(s) cylindrical joint cannot join a body to itself")
            if record["force_penalty"] <= 0.0 or record["moment_penalty"] <= 0.0:
                raise ValueError("Open Knee(s) cylindrical joint penalties must be positive")
            for coordinate in (record["translation"], record["rotation"]):
                curve_id = coordinate["attributes"].get("lc")
                if curve_id is not None and curve_id not in curve_ids:
                    raise ValueError("Open Knee(s) cylindrical joint references an unknown load curve")
            cylindrical_joints.append(record)
        elif kind == "rigid spring":
            body_a = int(_required_xml_text(item, "body_a", "Step/Constraints/constraint"))
            body_b = int(_required_xml_text(item, "body_b", "Step/Constraints/constraint"))
            insertion_a = _required_xml_vector(
                item, "insertion_a", "Step/Constraints/constraint"
            )
            insertion_b = _required_xml_vector(
                item, "insertion_b", "Step/Constraints/constraint"
            )
            stiffness = _required_xml_float(item, "k", "Step/Constraints/constraint")
            free_length_element = item.find("free_length")
            free_length_source_text = (
                None if free_length_element is None or free_length_element.text is None
                else free_length_element.text.strip()
            )
            free_length = (
                0.0 if free_length_source_text is None
                else _required_xml_float(item, "free_length", "Step/Constraints/constraint")
            )
            if body_a not in body_ids or body_b not in body_ids or body_a == body_b:
                raise ValueError("Open Knee(s) rigid spring references invalid rigid bodies")
            if (not all(math.isfinite(value) for value in (*insertion_a, *insertion_b))
                    or not math.isfinite(stiffness) or stiffness <= 0.0
                    or not math.isfinite(free_length) or free_length < 0.0):
                raise ValueError("Open Knee(s) rigid spring parameters are invalid")
            record.update({
                "body_a": body_a,
                "body_b": body_b,
                "insertion_a_source_text": _required_xml_text(
                    item, "insertion_a", "Step/Constraints/constraint"
                ),
                "insertion_a": list(insertion_a),
                "insertion_b_source_text": _required_xml_text(
                    item, "insertion_b", "Step/Constraints/constraint"
                ),
                "insertion_b": list(insertion_b),
                "stiffness_source_text": _required_xml_text(
                    item, "k", "Step/Constraints/constraint"
                ),
                "stiffness": stiffness,
                "free_length_source_text": free_length_source_text,
                "free_length": free_length,
                "free_length_uses_initial_insertion_distance": free_length == 0.0,
            })
            rigid_springs.append(record)
        else:
            other_constraints.append(record)

    contacts = []
    for item in root.findall("Step/Contact/contact"):
        pair_name = item.attrib.get("surface_pair")
        contacts.append({
            "type": item.attrib.get("type"),
            "surface_pair": item.attrib.get("surface_pair"),
            "parameters": _xml_parameter_records(item),
            "source_xml_sha256": _xml_digest(item),
            "native_execution_status": "unsupported_not_executed",
        })

    prescribed_body_boundaries = []
    for item in root.findall("Step/Boundary/rigid_body"):
        material_id = int(item.attrib["mat"])
        if material_id not in body_ids:
            raise ValueError("Open Knee(s) prescribed boundary references an unknown rigid body")
        for coordinate in item.findall("prescribed"):
            curve_id = coordinate.attrib.get("lc")
            if curve_id is not None and curve_id not in curve_ids:
                raise ValueError("Open Knee(s) prescribed boundary references an unknown load curve")
        prescribed_body_boundaries.append({
            "material_id": material_id,
            "coordinates": _xml_parameter_records(item),
            "source_xml_sha256": _xml_digest(item),
            "native_execution_status": "unsupported_not_executed",
        })

    curves = []
    for item in root.findall("LoadData/loadcurve"):
        numeric_points = _source_curve_points(item)
        curves.append({
            "id": item.attrib.get("id"),
            "name": item.attrib.get("name"),
            "type": item.attrib.get("type"),
            "points": [
                {"source_text": (point.text or "").strip(),
                 "attributes": dict(sorted(point.attrib.items()))}
                for point in item.findall("point")
            ],
            "numeric_points": numeric_points,
            "source_xml_sha256": _xml_digest(item),
            "native_execution_status": "unsupported_not_executed",
        })

    curves_by_id = {curve["id"]: curve for curve in curves}
    for source_material, material in zip(
        root.findall("Material/material"), materials, strict=True
    ):
        source_program = _source_material_program(
            source_material,
            curve_ids={value for value in curve_ids if value is not None},
            require_open_knee_tissue_schema=expected_deck_sha256 is not None,
        )
        prestrain = source_program["prestrain"]
        if prestrain is not None:
            resolved_curve = (
                curves_by_id[prestrain["load_curve_id"]]["numeric_points"]
                if prestrain["load_curve_id"] is not None else
                [{"time": None, "value": 1.0}]
            )
            if prestrain["load_curve_id"] is not None:
                prestrain["resolved_load_curve"] = resolved_curve
            if prestrain["isochoric"]:
                fiber = source_program["fiber_direction"]
                if fiber is None:
                    raise ValueError(
                        "Open Knee(s) isochoric prestrain has no source fiber direction"
                    )
                target_states = []
                for point in resolved_curve:
                    stretch = prestrain["stretch"] * point["value"]
                    target_states.append({
                        "time": point["time"],
                        "load_curve_value": point["value"],
                        "fiber_stretch": stretch,
                        "fiber_direction": fiber,
                        "deformation_gradient": _isochoric_fiber_prestrain_tensor(
                            stretch, fiber
                        ),
                    })
                prestrain["target_states"] = target_states
                prestrain["target_state_status"] = (
                    "compiled_isochoric_target_tensor_not_applied_or_equilibrated"
                )
        material["source_program"] = source_program

    source_mesh_data, source_mesh_data_bytes, mesh_data_element_counts = (
        _source_mesh_element_data(root)
    )
    source_mesh_data_storage = None
    if source_mesh_data_output_path is not None:
        source_mesh_data_output_path.parent.mkdir(parents=True, exist_ok=True)
        source_mesh_data_output_path.write_bytes(source_mesh_data_bytes)
        for record in source_mesh_data:
            if record["variable"] == "fiber":
                record["storage_file"] = source_mesh_data_output_path.name
        source_mesh_data_storage = {
            "schema": "numi.human.open-knee-mesh-element-data-f64.v1",
            "file": source_mesh_data_output_path.name,
            "bytes": len(source_mesh_data_bytes),
            "sha256": hashlib.sha256(source_mesh_data_bytes).hexdigest(),
            "record_count": len(source_mesh_data_bytes) // 28,
            "record_stride_bytes": 28,
            "record_layout": "little-endian u32 local_element_id, then float64[3] raw source vector",
        }

    rigid_ties = []
    for item in root.findall("Boundary/rigid"):
        node_set = item.attrib.get("node_set")
        rigid_body = item.attrib.get("rb")
        if not node_set or rigid_body is None or int(rigid_body) not in body_ids:
            raise ValueError("Open Knee(s) source rigid tie has an invalid source reference")
        rigid_ties.append({
            "name": item.attrib.get("name"),
            "node_set_name": node_set,
            "rigid_body_material_id": int(rigid_body),
            "attributes": dict(sorted(item.attrib.items())),
            "source_xml_sha256": _xml_digest(item),
            "node_set_resolution": "not_checked_against_the_archived_Geometry_custom_file",
            "native_execution_status": "reduced_attachment_only",
        })
    discrete_materials = [
        {"id": item.attrib.get("id"), "type": item.attrib.get("type"),
         "parameters": _xml_parameter_records(item),
         "source_xml_sha256": _xml_digest(item),
         "native_execution_status": "unsupported_not_executed"}
        for item in root.findall("Discrete/discrete_material")
    ]
    discrete_interactions = [
        {"attributes": dict(sorted(item.attrib.items())),
         "source_xml_sha256": _xml_digest(item),
         "native_execution_status": "unsupported_not_executed"}
        for item in root.findall("Discrete/discrete")
    ]
    step_control = root.find("Step/Control")
    if step_control is None:
        raise ValueError("Open Knee(s) source Step has no Control program")
    if not sections:
        raise ValueError("Open Knee(s) source mechanical program has no sections")
    geometry_resolution = None
    if source_geometry_archive_path is not None:
        geometry_resolution = _geometry_archive_cross_references(
            source_geometry_archive_path,
            expected_sha256=source_geometry_archive_sha256,
            contact_pair_names={contact["surface_pair"] for contact in contacts},
            rigid_tie_node_sets={tie["attributes"]["node_set"] for tie in rigid_ties},
            mesh_data_element_counts=mesh_data_element_counts,
            source_material_ids=source_material_ids,
            binary_output_path=source_geometry_binary_output_path,
            source_volume_mesh_output_path=source_volume_mesh_output_path,
        )
        materials_by_id = {int(material["id"]): material for material in materials}
        for element_set in geometry_resolution["element_sets"].values():
            material_id = element_set["material_id"]
            if material_id is None:
                element_set["material_assignment_status"] = "source_geometry_material_not_authored"
                continue
            source_material = materials_by_id[material_id]
            element_set.update({
                "material_name": source_material["name"],
                "material_type": source_material["type"],
                "material_source_xml_sha256": source_material["source_xml_sha256"],
                "material_assignment_status": "resolved_to_pinned_source_deck_material",
            })
        if expected_deck_sha256 == EXPECTED_HASHES["FeBio_custom.feb"]:
            expected_mesh_data = {"MNS-L": 44953, "MNS-M": 51009}
            actual_mesh_data = {
                record["element_set"]: record["record_count"]
                for record in source_mesh_data
                if record["variable"] == "fiber"
            }
            if actual_mesh_data != expected_mesh_data:
                raise ValueError(
                    "Open Knee(s) pinned meniscus fiber element sets changed"
                )
            for name, material_id in (("MNS-L", 12), ("MNS-M", 13)):
                element_set = geometry_resolution["element_sets"].get(name)
                material = materials_by_id.get(material_id)
                if (
                    element_set is None or material is None or
                    element_set["material_id"] != material_id or
                    not element_set["element_ids_contiguous_in_source_order"] or
                    material["name"] != name or
                    material["type"] != "trans iso Mooney-Rivlin" or
                    material["source_program"]["outer_scalar_parameters"].get("c2") != 0.0
                ):
                    raise ValueError(
                        f"Open Knee(s) pinned source fiber set {name} no longer resolves to its c2=0 trans-iso material"
                    )
        for tie in rigid_ties:
            resolved = geometry_resolution["rigid_tie_node_sets"][
                tie["attributes"]["node_set"]
            ]
            tie["node_set_resolution"] = geometry_resolution["status"]
            tie["node_set_node_count"] = resolved["node_count"]
            tie["node_set_node_ids_sha256"] = resolved["node_ids_sha256"]
            tie["node_set_node_ids"] = resolved["node_ids"]
        for contact in contacts:
            contact["geometry_resolution"] = geometry_resolution[
                "contact_surface_pairs"
            ][contact["surface_pair"]]
    if expected_deck_sha256 is not None:
        observed_counts = {
            "materials": len(materials),
            "rigid_bodies": len(rigid_bodies),
            "rigid_ties": len(rigid_ties),
            "cylindrical_joints": len(cylindrical_joints),
            "rigid_springs": len(rigid_springs),
            "other_constraints": len(other_constraints),
            "prescribed_body_boundaries": len(prescribed_body_boundaries),
            "contacts": len(contacts),
            "load_curves": len(curves),
        }
        if observed_counts != _EXPECTED_SOURCE_PROGRAM_COUNTS:
            raise ValueError(
                "Open Knee(s) pinned source mechanical construct counts drifted: "
                f"{observed_counts}"
            )
    return {
        "schema": "numi.human.open-knee-source-mechanics.v1",
        "source_file": source_deck.name,
        "source_file_bytes": len(deck_bytes),
        "source_file_sha256": deck_sha256,
        "febio_spec_version": root.attrib.get("version"),
        "reference_solver_version_from_archived_log": reference_solver_version,
        "reference_log_sha256": reference_log_sha256,
        "source_geometry_archive_sha256": (
            geometry_resolution["sha256"]
            if geometry_resolution is not None else source_geometry_archive_sha256
        ),
        "source_geometry_resolution": geometry_resolution,
        "source_mesh_element_data": source_mesh_data,
        "source_mesh_element_data_storage": source_mesh_data_storage,
        "reference_state_contract": {
            "material_reference_coordinates": {
                "source_geometry_sha256": (
                    geometry_resolution["sha256"]
                    if geometry_resolution is not None else source_geometry_archive_sha256
                ),
                "status": "immutable_source_Geometry_custom_reference",
            },
            "initialized_current_coordinates": {
                "status": "not_materialized_by_source_program_compilation",
            },
            "prestrain_targets": {
                "status": "compiled_schedule_targets_not_applied_or_equilibrated",
            },
        },
        "source_geometry_reference": (
            None if root.find("Geometry") is None else
            dict(sorted(root.find("Geometry").attrib.items()))
        ),
        "contact_surface_pair_resolution": (
            geometry_resolution["status"] if geometry_resolution is not None else
            "not_checked_against_the_archived_Geometry_custom_file"
        ),
        "module": dict(root.find("Module").attrib) if root.find("Module") is not None else None,
        "coordinate_frame": "original FEBio source coordinates; no anatomical registration or scale applied",
        "source_equivalence_admission": "rejected_unsupported_source_mechanics",
        "source_sections": sections,
        "rigid_graph": {
            "bodies": rigid_bodies,
            "rigid_ties": rigid_ties,
            "cylindrical_joints": cylindrical_joints,
            "rigid_springs": rigid_springs,
            "other_constraints": other_constraints,
            "prescribed_body_boundaries": prescribed_body_boundaries,
        },
        "materials": materials,
        "discrete_materials": discrete_materials,
        "discrete_interactions": discrete_interactions,
        "contacts": contacts,
        "load_curves": curves,
        "step_control": {
            "step_name": root.find("Step").attrib.get("name"),
            "parameters": _xml_parameter_records(step_control),
            "source_xml_sha256": _xml_digest(step_control),
            "native_execution_status": "unsupported_not_executed",
        },
        "unsupported_source_sections": unsupported,
    }


def compile_source_mechanical_artifacts(
    *, open_knee: Path, geometry_archive: Path, reference_log: Path, output: Path,
) -> dict[str, Any]:
    """Write the pinned source program and element data without MyoSim imports.

    This compiles an auditable source inventory only. It does not run FEBio,
    execute a native mechanics solve, apply prestrain, or admit equivalence.
    """
    output.mkdir(parents=True, exist_ok=True)
    source = parse_source(open_knee, enforce_exact=True)
    if _sha256(reference_log) != ARCHIVED_REFERENCE_LOG_SHA256:
        raise ValueError("Open Knee(s) archived FEBio reference-log identity drifted")
    from .open_knee_febio_log import parse_febio_log

    reference_observations = parse_febio_log(
        reference_log.read_text(encoding="utf-8", errors="strict")
    )
    mesh_data_path = output / "source-meshdata.bin"
    geometry_binary_path = output / "source-geometry.bin"
    source_volume_mesh_path = output / "source-volume-mesh.bin"
    rigid_graph_path = output / "source-rigid-graph.bin"
    rigid_ties_path = output / "source-rigid-ties.bin"
    contact_path = output / "source-sliding-contact.bin"
    description = compile_source_mechanical_description(
        source,
        source_deck=open_knee / "FeBio_custom.feb",
        expected_deck_sha256=EXPECTED_HASHES["FeBio_custom.feb"],
        source_geometry_archive_sha256=ARCHIVED_REFERENCE_GEOMETRY_SHA256,
        source_geometry_archive_path=geometry_archive,
        source_geometry_binary_output_path=geometry_binary_path,
        source_volume_mesh_output_path=source_volume_mesh_path,
        source_mesh_data_output_path=mesh_data_path,
        reference_solver_version=reference_observations.get("version"),
        reference_log_sha256=ARCHIVED_REFERENCE_LOG_SHA256,
    )
    accepted_times = reference_observations.get("accepted_times", [])
    description["reference_run_observation"] = {
        "status": reference_observations.get("status"),
        "version": reference_observations.get("version"),
        "accepted_time_count": len(accepted_times),
        "first_accepted_time": accepted_times[0] if accepted_times else None,
        "last_accepted_time": accepted_times[-1] if accepted_times else None,
        "complete_observations": (
            reference_observations.get("status") ==
            "normal_termination_with_complete_observations"
        ),
        "binary_identity": "not_supplied_in_archive",
        "log_sha256": ARCHIVED_REFERENCE_LOG_SHA256,
    }
    rigid_graph_storage = _write_source_rigid_graph_program(
        description, rigid_graph_path
    )
    description["source_rigid_graph_program_storage"] = rigid_graph_storage
    rigid_ties_storage = _write_source_rigid_ties_program(
        description, source_volume_mesh_path, rigid_ties_path
    )
    description["source_rigid_ties_program_storage"] = rigid_ties_storage
    contact_storage = _write_source_sliding_contact_program(
        description, geometry_binary_path, source_volume_mesh_path, contact_path
    )
    description["source_sliding_contact_program_storage"] = contact_storage
    reference_baseline_path = output / "source-reference-baseline.json"
    reference_baseline_storage = _write_source_reference_baseline(
        description, reference_baseline_path
    )
    description["source_reference_baseline_storage"] = reference_baseline_storage
    description_path = output / "source-mechanics.json"
    description_path.write_text(
        json.dumps(description, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mesh_groups = description["source_mesh_element_data"]
    receipt = {
        "schema": "numi.human.open-knee-source-mechanics-receipt.v1",
        "status": "passed_source_inventory_compilation_not_native_admission",
        "scope": (
            "source program and exact elementwise fiber inputs compiled; no source "
            "mechanics are executed or admitted by this artifact"
        ),
        "source_equivalence_admission": description["source_equivalence_admission"],
        "validation": {
            "pinned_source_deck_and_geometry": "passed exact SHA-256 identity checks",
            "archived_reference_log": reference_observations.get("status"),
            "archived_reference_log_last_time": (
                accepted_times[-1] if accepted_times else None
            ),
            "resolved_contact_pairs": len(description["contacts"]),
            "resolved_rigid_ties": len(description["rigid_graph"]["rigid_ties"]),
            "meniscus_element_set_counts": "match exact Geometry_custom sets",
            "meniscus_vector_payload": "independently reproducible little-endian source values",
            "native_mechanics_execution": "not performed",
            "rigid_graph_program": (
                "compiled from source rigid bodies, cylindrical joints, and prescribed body coordinates; "
                "not executed by this compiler"
            ),
        },
        "resolved_geometry": {
            "node_count": description["source_geometry_resolution"]["node_count"],
            "surface_count": description["source_geometry_resolution"]["surface_count"],
            "surface_pair_count": description["source_geometry_resolution"][
                "surface_pair_count"
            ],
            "element_set_count": description["source_geometry_resolution"][
                "element_set_count"
            ],
            "resolved_contacts": len(description["source_geometry_resolution"][
                "contact_surface_pairs"
            ]),
            "resolved_rigid_ties": len(description["source_geometry_resolution"][
                "rigid_tie_node_sets"
            ]),
            "mesh_data_element_sets": description["source_geometry_resolution"][
                "mesh_data_element_sets"
            ],
            "unreferenced_surface_pair_names": description[
                "source_geometry_resolution"
            ]["unreferenced_surface_pair_names"],
        },
        "reference_inputs": {
            "FeBio_custom.feb": {
                "sha256": description["source_file_sha256"],
                "bytes": description["source_file_bytes"],
            },
            "Geometry_custom.feb": {
                "sha256": description["source_geometry_archive_sha256"],
                "bytes": description["source_geometry_resolution"]["file_bytes"],
            },
            "FeBio_custom.log": {
                "sha256": ARCHIVED_REFERENCE_LOG_SHA256,
                "bytes": reference_log.stat().st_size,
                "reported_solver_version": reference_observations.get("version"),
                "status": reference_observations.get("status"),
                "last_accepted_time": accepted_times[-1] if accepted_times else None,
                "original_binary_identity": "not_supplied_in_archive",
            },
        },
        "compiled_counts": {
            "materials": len(description["materials"]),
            "rigid_bodies": len(description["rigid_graph"]["bodies"]),
            "rigid_ties": len(description["rigid_graph"]["rigid_ties"]),
            "rigid_tie_nodes": rigid_ties_storage["record_count"],
            "cylindrical_joints": len(description["rigid_graph"]["cylindrical_joints"]),
            "contacts": len(description["contacts"]),
            "contact_surface_faces": contact_storage["face_count"],
            "load_curves": len(description["load_curves"]),
            "prestrain_target_states": sum(
                len((material["source_program"].get("prestrain") or {}).get(
                    "target_states", []
                ))
                for material in description["materials"]
            ),
            "source_mesh_element_data_groups": len(mesh_groups),
            "source_mesh_element_fiber_vectors": sum(
                group["record_count"] for group in mesh_groups
            ),
        },
        "mesh_element_data": {
            "file": mesh_data_path.name,
            "schema": description["source_mesh_element_data_storage"]["schema"],
            "bytes": mesh_data_path.stat().st_size,
            "sha256": _sha256(mesh_data_path),
            "record_count": description["source_mesh_element_data_storage"]["record_count"],
            "record_stride_bytes": description["source_mesh_element_data_storage"][
                "record_stride_bytes"
            ],
            "groups": [
                {
                    "element_set": group["element_set"],
                    "record_count": group["record_count"],
                    "binary_sha256": group["binary_sha256"],
                    "source_xml_sha256": group["source_xml_sha256"],
                    "geometry_element_count": description[
                        "source_geometry_resolution"
                    ]["mesh_data_element_sets"][group["element_set"]][
                        "element_count"
                    ],
                    "native_execution_status": group["native_execution_status"],
                }
                for group in mesh_groups
            ],
        },
        "geometry_binary": description["source_geometry_resolution"][
            "binary_storage"
        ],
        "source_volume_mesh_binary": description["source_geometry_resolution"][
            "source_volume_mesh_storage"
        ],
        "artifacts": {
            description_path.name: {
                "bytes": description_path.stat().st_size,
                "sha256": _sha256(description_path),
            },
            mesh_data_path.name: {
                "bytes": mesh_data_path.stat().st_size,
                "sha256": _sha256(mesh_data_path),
            },
            geometry_binary_path.name: {
                "bytes": geometry_binary_path.stat().st_size,
                "sha256": _sha256(geometry_binary_path),
            },
            source_volume_mesh_path.name: {
                "bytes": source_volume_mesh_path.stat().st_size,
                "sha256": _sha256(source_volume_mesh_path),
            },
            rigid_graph_path.name: rigid_graph_storage,
            rigid_ties_path.name: rigid_ties_storage,
            contact_path.name: contact_storage,
            reference_baseline_path.name: reference_baseline_storage,
        },
        "compiler_sources": {
            Path(__file__).name: _sha256(Path(__file__)),
            "cli.py": _sha256(Path(__file__).resolve().parents[2]
                               / "src/numilab_human/cli.py"),
        },
        "not_qualified": [
            "exact re-execution from the unidentified archived FEBio binary",
            "native source joint, material, prestrain, contact, and load execution",
            "source tissue equilibrium, extensor loading, or whole-body integration",
        ],
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


_SOURCE_RIGID_TIES_MAGIC = b"NHTIES1\0"
_SOURCE_RIGID_TIES_SCHEMA = "numi.human.open-knee-rigid-ties-program.v1"


def _write_source_rigid_ties_program(
    description: dict[str, Any], source_volume_mesh_path: Path, output_path: Path,
) -> dict[str, Any]:
    """Bind every source rigid-tie node to one preserved volume node and body.

    Coordinates remain in the volume sidecar. Matter derives each body-local
    point from those immutable coordinates and the source body's reference COM.
    No moving attachment or contact law is executed by this writer.
    """
    volume = source_volume_mesh_path.read_bytes()
    storage = description["source_geometry_resolution"]["source_volume_mesh_storage"]
    if hashlib.sha256(volume).hexdigest() != storage["sha256"]:
        raise ValueError("Open Knee(s) source volume sidecar identity drifted")
    node_material: dict[int, int] = {}
    offset = 0
    while offset < len(volume):
        if len(volume) - offset < 20:
            raise ValueError("Open Knee(s) source volume header is truncated")
        magic, version, material_id, node_count, tet_count = struct.unpack_from(
            "<4s4I", volume, offset
        )
        if magic != b"NOKT" or version != 1 or material_id not in range(5, 17):
            raise ValueError("Open Knee(s) source volume group is unsupported")
        offset += 20
        group_bytes = 28 * node_count + 20 * tet_count
        if len(volume) - offset < group_bytes:
            raise ValueError("Open Knee(s) source volume group is truncated")
        for local in range(node_count):
            node_id = struct.unpack_from("<I", volume, offset + 28 * local)[0]
            if not node_id or node_id in node_material:
                raise ValueError("Open Knee(s) source volume node ID is duplicated")
            node_material[node_id] = material_id
        offset += group_bytes
    ties = description["rigid_graph"]["rigid_ties"]
    body_ids = {body["material_id"] for body in description["rigid_graph"]["bodies"]}
    rows: list[tuple[int, int, int, int]] = []
    claimed: set[int] = set()
    for tie_index, tie in enumerate(ties):
        body_id = int(tie["rigid_body_material_id"])
        ids = tie["node_set_node_ids"]
        if body_id not in body_ids or len(ids) != tie["node_set_node_count"]:
            raise ValueError("Open Knee(s) source rigid tie has invalid ownership")
        for node_id in ids:
            if node_id not in node_material or node_id in claimed:
                raise ValueError("Open Knee(s) rigid tie node is missing or multiply owned")
            claimed.add(node_id)
            rows.append((node_id, node_material[node_id], body_id, tie_index))
    rows.sort()
    if len(ties) != 18 or len(rows) != 29427:
        raise ValueError("Open Knee(s) pinned rigid-tie topology drifted")
    deck_hash = bytes.fromhex(description["source_file_sha256"])
    geometry_hash = bytes.fromhex(description["source_geometry_archive_sha256"])
    if len(deck_hash) != 32 or len(geometry_hash) != 32:
        raise ValueError("Open Knee(s) source rigid-tie hashes are invalid")
    payload = bytearray(struct.pack(
        "<8s4I32s32s", _SOURCE_RIGID_TIES_MAGIC, 1, len(ties), len(rows), 0,
        deck_hash, geometry_hash,
    ))
    for row in rows:
        payload.extend(struct.pack("<4I", *row))
    output_path.write_bytes(payload)
    return {
        "schema": _SOURCE_RIGID_TIES_SCHEMA,
        "file": output_path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "record_count": len(rows),
        "tie_set_count": len(ties),
        "header_bytes": 88,
        "record_stride_bytes": 16,
        "record_layout": "little-endian source node ID, source tissue material ID, rigid body material ID, source tie-set index",
        "native_execution_status": "compiled_for_Matter_binding_not_executed",
    }


_SOURCE_SLIDING_CONTACT_MAGIC = b"NHCNTP1\0"
_SOURCE_SLIDING_CONTACT_SCHEMA = "numi.human.open-knee-sliding-contact-program.v1"
_SOURCE_SLIDING_CONTACT_PARAMETERS = (
    "laugon", "tolerance", "gaptol", "penalty", "two_pass",
    "auto_penalty", "fric_coeff", "search_tol", "search_radius",
    "minaug", "maxaug", "seg_up",
)


def _write_source_sliding_contact_program(
    description: dict[str, Any], geometry_binary_path: Path,
    volume_mesh_path: Path, output_path: Path,
) -> dict[str, Any]:
    """Pack the complete authored contact graph for the whole Matter case.

    This preserves source faces and enforcement parameters, including rigid
    bone vertices that have no tetrahedra. It is an execution input, not a
    replacement contact law or an accepted contact state.
    """
    geometry = description["source_geometry_resolution"]
    binary = geometry_binary_path.read_bytes()
    volume = volume_mesh_path.read_bytes()
    if hashlib.sha256(binary).hexdigest() != geometry["binary_storage"]["sha256"]:
        raise ValueError("Open Knee(s) contact geometry binary identity drifted")
    if hashlib.sha256(volume).hexdigest() != geometry[
        "source_volume_mesh_storage"
    ]["sha256"]:
        raise ValueError("Open Knee(s) contact volume binary identity drifted")
    contacts = description["contacts"]
    material_ids = {
        material["name"]: int(material["id"])
        for material in description["materials"]
    }
    node_owner: dict[int, int] = {}
    offset = 0
    while offset < len(volume):
        if len(volume) - offset < 20:
            raise ValueError("Open Knee(s) contact volume header is truncated")
        magic, version, owner, nodes, tetrahedra = struct.unpack_from(
            "<4s4I", volume, offset
        )
        if magic != b"NOKT" or version != 1 or owner not in material_ids.values():
            raise ValueError("Open Knee(s) contact volume group is invalid")
        offset += 20
        if len(volume) - offset < nodes * 28 + tetrahedra * 20:
            raise ValueError("Open Knee(s) contact volume group is truncated")
        for local in range(nodes):
            node_id = struct.unpack_from("<I", volume, offset + 28 * local)[0]
            if not node_id or node_id in node_owner:
                raise ValueError("Open Knee(s) contact volume node is duplicated")
            node_owner[node_id] = owner
        offset += nodes * 28 + tetrahedra * 20

    referenced_names = sorted({
        contact["geometry_resolution"][side]
        for contact in contacts
        for side in ("master_surface", "slave_surface")
    })
    source_surfaces = geometry["surfaces"]
    rigid_owners = {int(body["material_id"]) for body in
                    description["rigid_graph"]["bodies"]}
    needed_rigid: dict[int, set[int]] = {}
    surfaces = []
    face_bytes = bytearray()
    for name in referenced_names:
        record = source_surfaces.get(name)
        owner_name = name.split("_@_", 1)[0]
        owner = material_ids.get(owner_name)
        if (record is None or owner is None or
                record["face_type"] != "tri3" or len(name.encode("ascii")) >= 32):
            raise ValueError("Open Knee(s) source contact surface is unsupported")
        start = int(record["binary_offset_bytes"])
        count = int(record["face_count"])
        end = start + count * 16
        if (count <= 0 or end > len(binary) or
                record["binary_bytes"] != count * 16):
            raise ValueError("Open Knee(s) source contact face range is invalid")
        raw = binary[start:end]
        source_digest = hashlib.sha256()
        for local in range(count):
            face_id, a, b, c = struct.unpack_from("<4I", raw, local * 16)
            if not face_id or not a or len({a, b, c}) != 3:
                raise ValueError("Open Knee(s) source contact face is degenerate")
            source_digest.update(f"tri3:{face_id}:{a},{b},{c}\n".encode("ascii"))
            if owner in rigid_owners:
                needed_rigid.setdefault(owner, set()).update((a, b, c))
            elif any(node_owner.get(node) != owner for node in (a, b, c)):
                raise ValueError("Open Knee(s) tissue contact face has wrong owner")
        if source_digest.hexdigest() != record["connectivity_sha256"]:
            raise ValueError("Open Knee(s) source contact face hash drifted")
        surfaces.append((name, owner, len(face_bytes) // 16, count,
                         bytes.fromhex(record["connectivity_sha256"])))
        face_bytes.extend(raw)

    rigid_node_coordinates: dict[int, tuple[int, tuple[float, float, float]]] = {}
    for material_name, owner in material_ids.items():
        wanted = needed_rigid.get(owner)
        if not wanted:
            continue
        group = geometry["node_coordinate_groups"].get(material_name)
        if group is None or group["binary_record_stride_bytes"] != 28:
            raise ValueError("Open Knee(s) rigid contact node group is missing")
        start = int(group["binary_offset_bytes"])
        count = int(group["node_count"])
        if start + count * 28 > len(binary):
            raise ValueError("Open Knee(s) rigid contact node group is truncated")
        for local in range(count):
            node_id, x, y, z = struct.unpack_from(
                "<I3d", binary, start + local * 28
            )
            if node_id not in wanted:
                continue
            if node_id in rigid_node_coordinates or not all(
                math.isfinite(value) for value in (x, y, z)
            ):
                raise ValueError("Open Knee(s) rigid contact node is invalid")
            rigid_node_coordinates[node_id] = (owner, (x, y, z))
        if not wanted.issubset(rigid_node_coordinates):
            raise ValueError("Open Knee(s) rigid contact surface has a missing node")
    if len(rigid_node_coordinates) != len(set().union(*needed_rigid.values())):
        raise ValueError("Open Knee(s) rigid contact vertex coverage is incomplete")

    deck_hash = bytes.fromhex(description["source_file_sha256"])
    archive_hash = bytes.fromhex(description["source_geometry_archive_sha256"])
    if len(deck_hash) != 32 or len(archive_hash) != 32:
        raise ValueError("Open Knee(s) source contact identities are invalid")
    payload = bytearray(struct.pack(
        "<8s5I32s32s32s32s", _SOURCE_SLIDING_CONTACT_MAGIC, 1,
        len(contacts), len(surfaces), len(face_bytes) // 16,
        len(rigid_node_coordinates), deck_hash, archive_hash,
        bytes.fromhex(geometry["binary_storage"]["sha256"]),
        bytes.fromhex(geometry["source_volume_mesh_storage"]["sha256"]),
    ))
    for name, owner, first, count, digest in surfaces:
        payload.extend(struct.pack(
            "<32sIII32s", name.encode("ascii"), owner, first, count, digest
        ))
    surface_index = {record[0]: index for index, record in enumerate(surfaces)}
    for contact in contacts:
        if contact["type"] != "sliding-elastic":
            raise ValueError("Open Knee(s) source contact has an unsupported law")
        parameters = {
            parameter["name"]: float(parameter["source_text"])
            for parameter in contact["parameters"]
        }
        if set(parameters) != set(_SOURCE_SLIDING_CONTACT_PARAMETERS) or not all(
            math.isfinite(value) for value in parameters.values()
        ):
            raise ValueError("Open Knee(s) source contact parameters are incomplete")
        pair = contact["geometry_resolution"]
        name = contact["surface_pair"]
        if len(name.encode("ascii")) >= 32:
            raise ValueError("Open Knee(s) source contact pair name is too long")
        payload.extend(struct.pack(
            "<32sII12d32s", name.encode("ascii"),
            surface_index[pair["master_surface"]],
            surface_index[pair["slave_surface"]],
            *(parameters[key] for key in _SOURCE_SLIDING_CONTACT_PARAMETERS),
            bytes.fromhex(contact["source_xml_sha256"]),
        ))
    payload.extend(face_bytes)
    for node_id, (owner, (x, y, z)) in sorted(rigid_node_coordinates.items()):
        payload.extend(struct.pack("<II3d", node_id, owner, x, y, z))
    if description["source_file_sha256"] == EXPECTED_HASHES["FeBio_custom.feb"] and (
        len(contacts) != 18 or len(surfaces) != 36 or len(face_bytes) // 16 != 345070
    ):
        raise ValueError("Open Knee(s) pinned contact graph drifted")
    output_path.write_bytes(payload)
    return {
        "schema": _SOURCE_SLIDING_CONTACT_SCHEMA,
        "file": output_path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pair_count": len(contacts),
        "surface_count": len(surfaces),
        "face_count": len(face_bytes) // 16,
        "rigid_node_count": len(rigid_node_coordinates),
        "header_bytes": 156,
        "surface_record_bytes": 76,
        "pair_record_bytes": 168,
        "face_record_bytes": 16,
        "rigid_node_record_bytes": 32,
        "parameter_order": list(_SOURCE_SLIDING_CONTACT_PARAMETERS),
        "native_execution_status": "compiled_for_whole_Matter_contact_not_executed",
    }


_SOURCE_RIGID_GRAPH_MAGIC = b"NHRGPH3\0"
_SOURCE_RIGID_GRAPH_SCHEMA = "numi.human.open-knee-rigid-graph-program-f64.v3"
_SOURCE_RIGID_COORDINATES = ("x", "y", "z", "Rx", "Ry", "Rz")


def _source_program_load_curve_id(value: Any) -> int:
    if value is None or value == "":
        return -1
    parsed = int(value)
    if parsed < 0 or parsed > 0x7FFFFFFF:
        raise ValueError("Open Knee(s) rigid graph load curve ID is out of range")
    return parsed


def _source_rigid_graph_program_bytes(description: dict[str, Any]) -> bytes:
    """Encode the exact rigid graph records consumed by Matter's source check.

    The binary is a narrow execution input, not a claim that all FEBio source
    mechanics have been compiled. It contains the rigid bodies, cylindrical
    joints, rigid springs, prescribed boundaries, and referenced load curves.
    The JSON description remains authoritative for names and full XML records;
    hashes bind these numeric records to it.
    """
    graph = description["rigid_graph"]
    bodies = graph["bodies"]
    joints = graph["cylindrical_joints"]
    springs = graph["rigid_springs"]
    boundaries = graph["prescribed_body_boundaries"]
    referenced_curve_ids = {
        _source_program_load_curve_id(joint[key].get("load_curve_id"))
        for joint in joints for key in ("translation", "rotation")
    }
    referenced_curve_ids.update(
        _source_program_load_curve_id(value["attributes"].get("lc"))
        for boundary in boundaries for value in boundary["coordinates"]
    )
    referenced_curve_ids.discard(-1)
    curves_by_id = {int(curve["id"]): curve for curve in description["load_curves"]}
    if not referenced_curve_ids.issubset(curves_by_id):
        raise ValueError("Open Knee(s) rigid graph references a missing source load curve")
    referenced_curves = [curves_by_id[curve_id] for curve_id in sorted(referenced_curve_ids)]
    for curve in referenced_curves:
        if curve["type"] != "linear" or len(curve["numeric_points"]) < 2:
            raise ValueError(
                "Open Knee(s) rigid graph only compiles linear source load curves"
            )
        times = [float(point["time"]) for point in curve["numeric_points"]]
        values = [float(point["value"]) for point in curve["numeric_points"]]
        if (not all(math.isfinite(value) for value in (*times, *values)) or
                any(right <= left for left, right in zip(times, times[1:]))):
            raise ValueError("Open Knee(s) rigid graph load curve points are invalid")
    deck_digest = bytes.fromhex(description["source_file_sha256"])
    geometry_digest = bytes.fromhex(description["source_geometry_archive_sha256"])
    if len(deck_digest) != 32 or len(geometry_digest) != 32:
        raise ValueError("Open Knee(s) rigid graph requires SHA-256 source identities")
    output = bytearray(_SOURCE_RIGID_GRAPH_MAGIC)
    output.extend(struct.pack(
        "<IIIIII", 3, len(bodies), len(joints), len(springs),
        len(boundaries), len(referenced_curves)
    ))
    output.extend(deck_digest)
    output.extend(geometry_digest)

    body_ids: set[int] = set()
    for body in bodies:
        body_id = int(body["material_id"])
        com = tuple(float(value) for value in body["center_of_mass"])
        digest = bytes.fromhex(body["source_xml_sha256"])
        if body_id <= 0 or body_id in body_ids or len(com) != 3 or len(digest) != 32:
            raise ValueError("Open Knee(s) source rigid body record is invalid")
        if not all(math.isfinite(value) for value in com):
            raise ValueError("Open Knee(s) source rigid body center is non-finite")
        body_ids.add(body_id)
        output.extend(struct.pack("<I3d32s", body_id, *com, digest))

    for joint in joints:
        body_a, body_b = int(joint["body_a"]), int(joint["body_b"])
        origin = tuple(float(value) for value in joint["joint_origin"])
        axis = tuple(float(value) for value in joint["joint_axis"])
        penalties_and_targets = (
            float(joint["force_penalty"]), float(joint["moment_penalty"]),
            float(joint["translation"]["value"]), float(joint["rotation"]["value"]),
        )
        digest = bytes.fromhex(joint["source_xml_sha256"])
        if (body_a not in body_ids or body_b not in body_ids or body_a == body_b
                or len(origin) != 3 or len(axis) != 3 or len(digest) != 32
                or not all(math.isfinite(v) for v in (*origin, *axis, *penalties_and_targets))
                or penalties_and_targets[0] <= 0 or penalties_and_targets[1] <= 0):
            raise ValueError("Open Knee(s) source cylindrical joint record is invalid")
        flags = (int(bool(joint["prescribed_translation"])),
                 int(bool(joint["prescribed_rotation"])))
        curves = (
            _source_program_load_curve_id(joint["translation"].get("load_curve_id")),
            _source_program_load_curve_id(joint["rotation"].get("load_curve_id")),
        )
        output.extend(struct.pack(
            "<II10dIIii32s", body_a, body_b, *origin, *axis,
            *penalties_and_targets, *flags, *curves, digest,
        ))

    for spring in springs:
        body_a, body_b = int(spring["body_a"]), int(spring["body_b"])
        insertion_a = tuple(float(value) for value in spring["insertion_a"])
        insertion_b = tuple(float(value) for value in spring["insertion_b"])
        stiffness = float(spring["stiffness"])
        free_length = float(spring["free_length"])
        digest = bytes.fromhex(spring["source_xml_sha256"])
        if (body_a not in body_ids or body_b not in body_ids or body_a == body_b
                or len(insertion_a) != 3 or len(insertion_b) != 3
                or len(digest) != 32
                or not all(math.isfinite(v) for v in (*insertion_a, *insertion_b,
                                                       stiffness, free_length))
                or stiffness <= 0.0 or free_length < 0.0):
            raise ValueError("Open Knee(s) source rigid spring record is invalid")
        output.extend(struct.pack(
            "<II8d32s", body_a, body_b, *insertion_a, *insertion_b,
            stiffness, free_length, digest,
        ))

    for boundary in boundaries:
        body_id = int(boundary["material_id"])
        if body_id not in body_ids:
            raise ValueError("Open Knee(s) prescribed rigid body boundary references no body")
        coordinates = {
            str(value["attributes"].get("bc")): value
            for value in boundary["coordinates"]
        }
        if len(coordinates) != len(boundary["coordinates"]):
            raise ValueError("Open Knee(s) prescribed rigid body coordinates are duplicated")
        unsupported_coordinates = set(coordinates) - set(_SOURCE_RIGID_COORDINATES)
        if unsupported_coordinates:
            raise ValueError(
                "Open Knee(s) rigid graph has unsupported prescribed coordinates: "
                f"{sorted(unsupported_coordinates)}"
            )
        mask = 0
        values = [0.0] * 6
        curves = [-1] * 6
        for index, coordinate in enumerate(_SOURCE_RIGID_COORDINATES):
            record = coordinates.get(coordinate)
            if record is None:
                continue
            mask |= 1 << index
            values[index] = float(record["source_text"])
            curves[index] = _source_program_load_curve_id(
                record["attributes"].get("lc")
            )
        digest = bytes.fromhex(boundary["source_xml_sha256"])
        if len(digest) != 32 or not all(math.isfinite(value) for value in values):
            raise ValueError("Open Knee(s) prescribed rigid body boundary is invalid")
        output.extend(struct.pack("<IB3x6d6i32s", body_id, mask, *values, *curves, digest))
    for curve in referenced_curves:
        output.extend(struct.pack("<III", int(curve["id"]), 1, len(curve["numeric_points"])))
        for point in curve["numeric_points"]:
            output.extend(struct.pack("<2d", float(point["time"]), float(point["value"])))
        output.extend(bytes.fromhex(curve["source_xml_sha256"]))
    return bytes(output)


def _write_source_rigid_graph_program(
    description: dict[str, Any], output_path: Path,
) -> dict[str, Any]:
    payload = _source_rigid_graph_program_bytes(description)
    output_path.write_bytes(payload)
    return {
        "schema": _SOURCE_RIGID_GRAPH_SCHEMA,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "endianness": "little",
        "coordinate_unit": "source_deck_coordinate_unit_unresolved",
        "record_layout": {
            "header_bytes": 96,
            "body_bytes": 60,
            "cylindrical_joint_bytes": 136,
            "rigid_spring_bytes": 104,
            "prescribed_body_boundary_bytes": 112,
            "load_curve_header_bytes": 12,
            "load_curve_point_bytes": 16,
            "load_curve_sha256_bytes": 32,
        },
        "record_counts": {
            "bodies": len(description["rigid_graph"]["bodies"]),
            "cylindrical_joints": len(description["rigid_graph"]["cylindrical_joints"]),
            "rigid_springs": len(description["rigid_graph"]["rigid_springs"]),
            "prescribed_body_boundaries": len(
                description["rigid_graph"]["prescribed_body_boundaries"]
            ),
            "referenced_load_curves": len({
                int(curve["id"]) for curve in description["load_curves"]
                if int(curve["id"]) in {
                    _source_program_load_curve_id(joint[key].get("load_curve_id"))
                    for joint in description["rigid_graph"]["cylindrical_joints"]
                    for key in ("translation", "rotation")
                } | {
                    _source_program_load_curve_id(value["attributes"].get("lc"))
                    for boundary in description["rigid_graph"]["prescribed_body_boundaries"]
                    for value in boundary["coordinates"]
                }
            }),
        },
        "native_execution_status": "compiled_source_program_not_executed",
    }


def _write_source_reference_baseline(
    description: dict[str, Any], output_path: Path,
) -> dict[str, Any]:
    """Bind retained FEBio observations/contact outputs without rerunning FEBio."""
    root = Path(__file__).resolve().parents[2]
    archive_root = root / "Docs/media/open-knee-reference-20261001"
    observation_path = archive_root / "archived-observations.json.gz"
    contact_path = archive_root / "archived-contact.json.gz"
    observation_sha = _sha256(observation_path)
    contact_sha = _sha256(contact_path)
    if observation_sha != ARCHIVED_REFERENCE_OBSERVATIONS_SHA256:
        raise ValueError("Open Knee(s) archived observation identity drifted")
    if contact_sha != ARCHIVED_REFERENCE_CONTACT_SHA256:
        raise ValueError("Open Knee(s) archived contact identity drifted")
    observations = json.loads(gzip.decompress(observation_path.read_bytes()))
    contact = json.loads(gzip.decompress(contact_path.read_bytes()))
    if (
        observations.get("status") != "normal_termination_with_complete_observations" or
        observations.get("version") != ARCHIVED_REFERENCE_SOLVER_VERSION or
        not observations.get("observations_complete") or
        not observations.get("normal_termination_reported") or
        observations.get("error_termination_reported") or
        contact.get("archive_complete") is not True or
        contact.get("archive_identity_verified") is not True or
        contact.get("complete_chunk_prefix_sha256") != ARCHIVED_REFERENCE_XPLT_SHA256 or
        contact.get("available_bytes") != contact.get("expected_archive_bytes")
    ):
        raise ValueError("Open Knee(s) archived reference run is incomplete or mismatched")

    accepted_times = [float(value) for value in observations["accepted_times"]]
    if len(accepted_times) != 140 or not accepted_times or accepted_times[-1] != 2.0:
        raise ValueError("Open Knee(s) archived reference continuation changed")
    bodies = {
        str(body["material_id"]): body["name"]
        for body in description["rigid_graph"]["bodies"]
    }
    records_by_time: dict[float, dict[str, Any]] = defaultdict(dict)
    for record in observations["records"]:
        records_by_time[float(record["continuation_time"])][record["field"]] = record
    expected_fields = {
        ("center_of_mass", "mm"),
        ("rotation_quaternion", "dimensionless; xyzw"),
        ("Reaction_Forces", "N"),
        ("Reaction_Torques", "N mm"),
        ("Rigid_Connector_Force", "N"),
        ("Rigid_Connector_Moment", "N mm"),
    }
    observed_fields = {(record["field"], record["units"]) for record in observations["records"]}
    if observed_fields != expected_fields or len(observations["records"]) != 840:
        raise ValueError("Open Knee(s) archived rigid observation fields changed")

    observation_snapshots = []
    for target_time in (accepted_times[0], 1.0, accepted_times[-1]):
        actual_time = min(accepted_times, key=lambda value: abs(value - target_time))
        if abs(actual_time - target_time) > 1.0e-9:
            raise ValueError("Open Knee(s) reference checkpoint is absent")
        by_field = records_by_time[actual_time]
        if set(by_field) != {field for field, _ in expected_fields}:
            raise ValueError("Open Knee(s) archived rigid checkpoint is incomplete")
        observation_snapshots.append({
            "continuation_time": actual_time,
            "units": {"position": "mm", "rotation": "xyzw", "force": "N", "moment": "N mm"},
            "rigid_bodies": {
                bodies[body_id]: {
                    "material_id": int(body_id),
                    "center_of_mass": by_field["center_of_mass"]["values"][body_id],
                    "rotation_quaternion_xyzw": by_field["rotation_quaternion"]["values"][body_id],
                    "reaction_force": by_field["Reaction_Forces"]["values"][body_id],
                    "reaction_torque": by_field["Reaction_Torques"]["values"][body_id],
                }
                for body_id in sorted(bodies, key=int)
            },
            "rigid_connector_force": by_field["Rigid_Connector_Force"]["values"],
            "rigid_connector_moment": by_field["Rigid_Connector_Moment"]["values"],
        })

    contact_snapshots = []
    peak_pressure: dict[str, dict[str, Any]] = {}
    for state in contact["states"]:
        time = float(state["continuation_time"])
        fields = state["contact_fields"]
        if len(fields) != 72:
            raise ValueError("Open Knee(s) archived contact state is incomplete")
        for field in fields:
            if field["field"] != "contact pressure":
                continue
            item = peak_pressure.setdefault(field["name"], {
                "units": field["units"], "maximum": -math.inf,
                "continuation_time": None, "positive_faces_at_peak": 0,
            })
            if float(field["maximum"]) > item["maximum"]:
                item.update({
                    "maximum": float(field["maximum"]),
                    "continuation_time": time,
                    "positive_faces_at_peak": int(field["positive_faces"]),
                })
        if any(abs(time - target) < 1.0e-5 for target in (0.0, 1.0, 2.0)):
            contact_snapshots.append({
                "continuation_time": time,
                "surfaces": [{
                    "name": field["name"],
                    "field": field["field"],
                    "faces": int(field["faces"]),
                    "minimum": float(field["minimum"]),
                    "mean": float(field["mean"]),
                    "maximum": float(field["maximum"]),
                    "negative_faces": int(field["negative_faces"]),
                    "positive_faces": int(field["positive_faces"]),
                    "units": field["units"],
                } for field in fields],
            })
    if len(contact["states"]) != 141 or len(peak_pressure) != 36 or len(contact_snapshots) != 3:
        raise ValueError("Open Knee(s) archived contact continuation changed")

    baseline = {
        "schema": "numi.human.open-knee-source-reference-baseline.v1",
        "status": "passed_retained_reference_archive_identity_and_summary",
        "source_solver_rerun": "not_performed_by_native_compiler",
        "source_deck_sha256": description["source_file_sha256"],
        "source_geometry_sha256": description["source_geometry_archive_sha256"],
        "reference_run": {
            "solver": f"FEBio {observations['version']}",
            "status": observations["status"],
            "source_log_sha256": description["reference_log_sha256"],
            "solver_binary_identity": description["reference_run_observation"][
                "binary_identity"
            ],
            "step_control": description["step_control"],
            "time_semantics": observations["time_semantics"],
            "accepted_increment_count": int(observations["accepted_increment_count"]),
            "first_accepted_continuation_time": accepted_times[0],
            "last_accepted_continuation_time": accepted_times[-1],
            "observation_record_count": len(observations["records"]),
            "fields": [
                {"name": name, "units": units}
                for name, units in sorted(expected_fields)
            ],
            "checkpoints": observation_snapshots,
        },
        "contact_archive": {
            "format": contact["format"],
            "status": "complete_archive_identity_verified",
            "source_xplt_sha256": ARCHIVED_REFERENCE_XPLT_SHA256,
            "source_xplt_bytes": int(contact["expected_archive_bytes"]),
            "state_count": len(contact["states"]),
            "surface_count": int(contact["surface_count"]),
            "time_semantics": contact["time_semantics"],
            "checkpoints": contact_snapshots,
            "peak_pressure_by_surface": [
                {"name": name, **values}
                for name, values in sorted(peak_pressure.items())
            ],
        },
        "retained_archive_files": {
            observation_path.name: {
                "repository_path": str(observation_path.relative_to(root)),
                "bytes": observation_path.stat().st_size,
                "sha256": observation_sha,
            },
            contact_path.name: {
                "repository_path": str(contact_path.relative_to(root)),
                "bytes": contact_path.stat().st_size,
                "sha256": contact_sha,
            },
        },
        "unavailable_reference_outputs": [
            "elementwise tissue stress/strain and stored-energy fields are not present in the retained observation/contact summaries",
            "source native solver executable binary identity is not retained",
        ],
        "qualification": {
            "archived_source_solver_run": "normal_termination_with_complete_observations",
            "native_source_reproduction": "not_performed",
            "tissue_equilibrium_reproduced_by_matter": False,
        },
    }
    payload = json.dumps(baseline, indent=2, sort_keys=True) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    return {
        "schema": baseline["schema"],
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
        "retained_observation_archive_sha256": observation_sha,
        "retained_contact_archive_sha256": contact_sha,
        "source_xplt_sha256": ARCHIVED_REFERENCE_XPLT_SHA256,
        "native_execution_status": "archived_reference_outputs_summarized_not_rerun",
    }


def _unit(vector: Any, np: Any) -> Any:
    result = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(result))
    if result.shape != (3,) or not math.isfinite(length) or length <= 1.0e-12:
        raise RuntimeError("Open Knee(s) registration encountered a degenerate axis")
    return result / length


def _dot3(values: Any, axis: Any, np: Any) -> Any:
    """Three-component dot product without the platform BLAS matmul path."""
    points = np.asarray(values, dtype=float)
    direction = np.asarray(axis, dtype=float)
    if points.shape[-1:] != (3,) or direction.shape != (3,):
        raise RuntimeError("Open Knee(s) dot product received an invalid shape")
    return (
        points[..., 0] * direction[0]
        + points[..., 1] * direction[1]
        + points[..., 2] * direction[2]
    )


def _extensor_stack_metrics(region_node_world: dict[str, Any], anterior: Any,
                            proximal: Any, np: Any) -> dict[str, float]:
    """Source-specific layer/order preflight for the patellar extensor stack."""
    centers = {name: np.mean(np.asarray(region_node_world[name], dtype=float), axis=0)
               for name in ("PTB", "PTC", "QAT", "PTL")}
    metrics = {
        "patellar_cartilage_posterior_to_bone_m": float(_dot3(
            centers["PTB"] - centers["PTC"], anterior, np)),
        "quadriceps_tendon_proximal_to_patella_m": float(_dot3(
            centers["QAT"] - centers["PTB"], proximal, np)),
        "patellar_tendon_distal_to_patella_m": float(_dot3(
            centers["PTB"] - centers["PTL"], proximal, np)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Open Knee(s) patellar extensor stack is nonfinite")
    # These loose source-regression margins detect reversed or collapsed
    # layers. They are not clinical spacing or cartilage-contact criteria.
    for name, value in metrics.items():
        minimum = EXTENSOR_STACK_MINIMUMS_M[name]
        if value < minimum:
            raise RuntimeError(
                f"Open Knee(s) patellar extensor stack reversed/too close: "
                f"{name}={value:.6g} m minimum={minimum:.6g} m")
    return metrics


def _patellofemoral_surface_winding(
    source: Source, region_node_world: dict[str, Any], *, reflected: bool,
    np: Any,
) -> dict[str, dict[str, float | int]]:
    """Reject an inward or posteriorly facing patellofemoral contact layer.

    This is a static source-specific winding gate. It checks actual triangle
    faces against their tetrahedral interior after registration and, on the
    mirrored side, after the connectivity parity correction. It does not infer
    contact pressure or a physical cartilage gap.
    """
    groups = {
        "PTC": ("PTC_@_FMC_ContactFaces", "PTC_@_PTB_TiesFaces"),
        "FMC": ("FMC_@_PTC_ContactFaces", "FMC_@_FMB_TiesFaces"),
    }
    anterior = np.asarray([0.0, -1.0, 0.0], dtype=float)
    result: dict[str, dict[str, float | int]] = {}
    for name, surfaces in groups.items():
        region = source.regions[name]
        coordinates = np.asarray(region_node_world[name], dtype=float)
        if coordinates.shape != (len(region.node_ids), 3) or not bool(
            np.isfinite(coordinates).all()
        ):
            raise RuntimeError("Open Knee(s) patellofemoral surface has invalid nodes")
        local = {identifier: index for index, identifier in enumerate(region.node_ids)}
        wanted = {
            tuple(sorted(face))
            for surface_name in surfaces
            for face in source.surfaces[surface_name].faces
        }
        incidence: dict[tuple[int, int, int], list[int]] = {}
        for tetrahedron in region.elements:
            for opposite in range(4):
                key = tuple(sorted(tetrahedron[corner] for corner in range(4)
                                   if corner != opposite))
                if key in wanted:
                    incidence.setdefault(key, []).append(tetrahedron[opposite])
        for surface_name in surfaces:
            area_normal = np.zeros(3, dtype=float)
            total_area = 0.0
            faces = source.surfaces[surface_name].faces
            for face in faces:
                owners = incidence.get(tuple(sorted(face)), [])
                if len(owners) != 1:
                    raise RuntimeError(
                        f"Open Knee(s) patellofemoral surface {surface_name} "
                        "is not an exterior one-owner tetrahedron face")
                oriented = _orientation_preserving_connectivity(
                    face, reflected=reflected)
                points = coordinates[[local[identifier] for identifier in oriented]]
                inward = coordinates[local[owners[0]]] - points[0]
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                normal_length = float(np.linalg.norm(normal))
                if not (math.isfinite(normal_length) and normal_length > 0.0 and
                        float(_dot3(normal, inward, np)) < 0.0):
                    raise RuntimeError(
                        f"Open Knee(s) patellofemoral surface {surface_name} "
                        "has a degenerate or inward face")
                area_normal += normal
                total_area += normal_length
            if not (len(faces) > 0 and total_area > 0.0):
                raise RuntimeError(
                    f"Open Knee(s) patellofemoral surface {surface_name} is empty")
            cosine = float(_dot3(area_normal, anterior, np) / total_area)
            result[surface_name] = {
                "faces": len(faces),
                "all_exterior_outward": 1,
                "area_weighted_anterior_cosine": cosine,
            }
    if (result["PTC_@_FMC_ContactFaces"]["area_weighted_anterior_cosine"] > -0.7 or
        result["PTC_@_PTB_TiesFaces"]["area_weighted_anterior_cosine"] < 0.7):
        raise RuntimeError(
            "Open Knee(s) patellofemoral cartilage contact/tie faces point "
            "toward the wrong anterior layer")
    return result


def _anatomical_femoral_basis(
    knee_axis_line_body: Any,
    proximal_body: Any,
    femur_body_world_rotation: Any,
    np: Any,
) -> tuple[Any, Any, Any, float]:
    """Resolve the flexion-axis sign from the Human's anterior direction.

    A flexion axis is an unoriented line.  Using its arbitrary source sign to
    build the remaining femoral basis admits two proper rotations separated by
    180 degrees about the long axis.  The rejected choice placed the patella
    posteriorly and swapped the medial/lateral condyles while still passing a
    symmetric distal-femur surface fit.  BodyParts3D and the native Human
    cameras establish anterior as negative world Y, so choose the unique
    proper basis whose anterior axis points there.
    """
    proximal = _unit(proximal_body, np)
    axis = _unit(knee_axis_line_body, np)
    axis = _unit(axis - proximal * float(_dot3(axis, proximal, np)), np)
    world_rotation = np.asarray(femur_body_world_rotation, dtype=float)
    if world_rotation.shape != (3, 3) or not bool(np.all(np.isfinite(world_rotation))):
        raise RuntimeError("Open Knee(s) femur world rotation is invalid")
    human_anterior_world = np.asarray([0.0, -1.0, 0.0], dtype=float)
    candidates = []
    for signed_axis in (axis, -axis):
        anterior = _unit(np.cross(proximal, signed_axis), np)
        basis = np.column_stack((signed_axis, anterior, proximal))
        determinant = float(np.linalg.det(basis))
        if abs(determinant - 1.0) > 2.0e-5:
            raise RuntimeError("Open Knee(s) target femoral basis is not proper")
        world_anterior = np.einsum("ij,j->i", world_rotation, anterior)
        alignment = float(_dot3(world_anterior, human_anterior_world, np))
        candidates.append((alignment, signed_axis, anterior, basis))
    alignment, signed_axis, anterior, basis = max(candidates, key=lambda item: item[0])
    if alignment < 0.999:
        raise RuntimeError("Open Knee(s) cannot resolve an anatomically anterior femoral basis")
    return signed_axis, anterior, basis, alignment


def _quantile_width(points: Any, axis: Any, np: Any) -> float:
    projection = _dot3(points, _unit(axis, np), np)
    return float(np.quantile(projection, 0.98) - np.quantile(projection, 0.02))


def _sample(points: Any, maximum: int, np: Any) -> Any:
    points = np.asarray(points, dtype=float)
    if len(points) <= maximum:
        return points.copy()
    return points[np.linspace(0, len(points) - 1, maximum, dtype=int)]


def _nearest_metrics(first: Any, second: Any, np: Any) -> dict[str, float]:
    first = _sample(first, 400, np)
    second = _sample(second, 400, np)
    distances: list[Any] = []
    for source, target in ((first, second), (second, first)):
        values = []
        for address in range(0, len(source), 64):
            block = source[address : address + 64]
            squared = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
            values.append(np.min(squared, axis=1))
        distances.append(np.sqrt(np.concatenate(values)))
    combined = np.concatenate(distances)
    return {
        "mean_m": float(np.mean(combined)),
        "median_m": float(np.median(combined)),
        "p90_m": float(np.quantile(combined, 0.90)),
        "maximum_m": float(np.max(combined)),
    }


def _nearest_indices(source: Any, target: Any, np: Any) -> tuple[Any, Any]:
    indices = []
    squared_distances = []
    for address in range(0, len(source), 64):
        block = source[address : address + 64]
        squared = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        indices.append(nearest)
        squared_distances.append(squared[np.arange(len(nearest)), nearest])
    return np.concatenate(indices), np.concatenate(squared_distances)


def _bounded_translation_refinement(
    moving: Any, target: Any, maximum_translation_m: float, np: Any,
) -> tuple[Any, int, dict[str, float]]:
    """Fit only a robust translation while reserving every fifth point.

    FMO is a documented distal-posterior landmark, whereas the MyoSim body
    origin is a mechanics joint frame. Their offset is not assumed to be zero.
    Rotation, scale, and all relative source tissue geometry remain unchanged.
    """
    moving_sample = _sample(moving, 400, np)
    target_sample = _sample(target, 400, np)
    moving_addresses = np.arange(len(moving_sample))
    target_addresses = np.arange(len(target_sample))
    moving_training = moving_sample[moving_addresses % 5 != 0].copy()
    target_training = target_sample[target_addresses % 5 != 0]
    total = np.zeros(3)
    iterations = 0
    for iterations in range(1, 31):
        forward_indices, forward_squared = _nearest_indices(
            moving_training, target_training, np
        )
        reverse_indices, reverse_squared = _nearest_indices(
            target_training, moving_training, np
        )
        residuals = np.concatenate((
            target_training[forward_indices] - moving_training,
            target_training - moving_training[reverse_indices],
        ))
        distances = np.sqrt(np.concatenate((forward_squared, reverse_squared)))
        retained = residuals[distances <= float(np.quantile(distances, 0.80))]
        if len(retained) < 24:
            raise RuntimeError("Open Knee(s) translation refinement retained too few pairs")
        delta = np.median(retained, axis=0)
        proposed = total + delta
        length = float(np.linalg.norm(proposed))
        if length > maximum_translation_m:
            proposed *= maximum_translation_m / length
            delta = proposed - total
        total = proposed
        moving_training += delta
        if float(np.linalg.norm(delta)) <= 1.0e-7:
            break
    transformed = moving_sample + total
    moving_held = transformed[moving_addresses % 5 == 0]
    target_held = target_sample[target_addresses % 5 == 0]
    forward = _nearest_indices(moving_held, target_sample, np)[1]
    reverse = _nearest_indices(target_held, transformed, np)[1]
    distances = np.sqrt(np.concatenate((forward, reverse)))
    metrics = {
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.quantile(distances, 0.90)),
        "maximum_m": float(np.max(distances)),
    }
    return total, iterations, metrics


def _world_to_core(points: Any, target: dict[str, Any], np: Any) -> Any:
    rotation = _rotation_xyzw(target["default_inertial_quaternion_world_xyzw"], np)
    position = np.asarray(target["default_com_position_world_m"], dtype=float)
    return np.einsum("ki,ij->kj", np.asarray(points, dtype=float) - position, rotation)


def _core_to_world(points: Any, target: dict[str, Any], np: Any) -> Any:
    rotation = _rotation_xyzw(target["default_inertial_quaternion_world_xyzw"], np)
    position = np.asarray(target["default_com_position_world_m"], dtype=float)
    return np.einsum("ki,ji->kj", np.asarray(points, dtype=float), rotation) + position


def _fixed_name(value: str, width: int) -> bytes:
    encoded = value.encode("ascii")
    if not encoded or len(encoded) >= width:
        raise RuntimeError(f"Open Knee(s) payload name is invalid: {value}")
    return encoded + b"\0" * (width - len(encoded))


def compile_payload(
    *, sources: Path, open_knee: Path, registration_path: Path, output: Path,
    side: str = "left", projected_visual_frame: bool = False,
    source_geometry_archive_path: Path | None = None,
    source_mesh_data_output_path: Path | None = None,
) -> dict[str, Any]:
    if side not in {"left", "right"}:
        raise ValueError("Open Knee(s) payload side must be left or right")
    side_suffix = "l" if side == "left" else "r"

    def body_name(role: str) -> str:
        return f"{role}_{side_suffix}"
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "Open Knee(s) compilation requires the pinned MyoSim/MuJoCo environment"
        ) from error
    source = parse_source(open_knee, enforce_exact=True)
    expected_fiber_regions = {"ACL", "PCL", "MCL", "LCL", "PTL", "QAT"}
    if set(source.fiber_directions) != expected_fiber_regions:
        raise RuntimeError(
            "Open Knee(s) homogeneous ligament/tendon fibre table drifted"
        )
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema") != (
        "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
    ):
        raise RuntimeError("Open Knee(s) compilation requires registration candidate v2")
    targets: dict[str, dict[str, Any]] = {}
    for anchor in registration.get("anchors", []):
        name = anchor.get("target", {}).get("name")
        if name in {
            "femur_r", "tibia_r", "patella_r",
            "femur_l", "tibia_l", "patella_l",
        }:
            target = anchor["target"]
            if name in targets and targets[name] != target:
                raise RuntimeError(f"Open Knee(s) target frame drifted for {name}")
            targets[name] = target
    expected_indices = {
        "femur_r": 131, "tibia_r": 136, "patella_r": 142,
        "femur_l": 145, "tibia_l": 150, "patella_l": 156,
    }
    if set(targets) != set(expected_indices):
        raise RuntimeError("Open Knee(s) live bilateral knee body frames are incomplete")
    if any(
        int(targets[name]["core_body_index"]) != index
        for name, index in expected_indices.items()
    ):
        raise RuntimeError("Open Knee(s) pinned bilateral knee body indices drifted")

    exported = export_fullbody(sources)
    source_bodies = {body["name"]: body for body in exported["bodies"]}
    femur_body = source_bodies["femur_l"]
    model = build_model("myofullbody")
    visual_targets = targets
    visual_reference_pose = "source_default_qpos0"
    projected_visual_maximum_body_shift_m = 0.0
    if projected_visual_frame:
        from .myosim_export import _matrix_to_quaternion
        from .myosim_visual import _visual_qpos

        projected_qpos, projection = _visual_qpos(model, mujoco, False)
        if projection["pose_state"] != "source_equality_projected_neutral":
            raise RuntimeError("Open Knee(s) projected visual frame was not equality projected")
        projected_data = mujoco.MjData(model)
        projected_data.qpos[:] = projected_qpos
        mujoco.mj_forward(model, projected_data)
        visual_targets = {name: target.copy() for name, target in targets.items()}
        for name, target in visual_targets.items():
            body_index = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_index < 0 or body_index != int(target["source_body_id"]):
                raise RuntimeError("Open Knee(s) projected visual body binding drifted")
            projected_position = np.asarray(projected_data.xipos[body_index], dtype=float)
            default_position = np.asarray(target["default_com_position_world_m"], dtype=float)
            projected_visual_maximum_body_shift_m = max(
                projected_visual_maximum_body_shift_m,
                float(np.linalg.norm(projected_position - default_position)),
            )
            target["default_com_position_world_m"] = projected_position.tolist()
            target["default_inertial_quaternion_world_xyzw"] = (
                _matrix_to_quaternion(projected_data.ximat[body_index], mujoco)
            )
        visual_reference_pose = "source_equality_projected_neutral"
    meshes = _compiled_meshes_by_body(model, mujoco, np)
    femur_meshes = meshes.get(int(femur_body["id"]), [])
    if not femur_meshes:
        raise RuntimeError("Open Knee(s) registration has no MyoSim femur mesh")
    myosim_femur_body = np.concatenate([
        np.asarray(mesh["vertices"], dtype=float) for mesh in femur_meshes
    ])
    femur_body_world_rotation = _rotation_xyzw(
        femur_body["default_body_quaternion_world_xyzw"], np
    )
    femur_body_world_position = np.asarray(
        femur_body["default_body_position_world_m"], dtype=float
    )
    knee_origin_body = np.asarray([-4.6e-07, -0.404425, 0.00126526], dtype=float)
    knee_axis_line_body = _unit([3.98373e-10, 0.0707131, -0.997497], np)
    proximal_body = _unit([0.0, 1.0, 0.0], np)
    knee_axis_body, anterior_body, target_basis, anterior_alignment = (
        _anatomical_femoral_basis(
            knee_axis_line_body,
            proximal_body,
            femur_body_world_rotation,
            np,
        )
    )
    xf = _unit(source.landmarks["Xf_axis"], np)
    yf = _unit(source.landmarks["Yf_axis"], np)
    zf = _unit(source.landmarks["Zf_axis"], np)
    source_basis = np.column_stack((xf, yf, zf))
    if abs(float(np.linalg.det(source_basis)) - 1.0) > 2.0e-5:
        raise RuntimeError("Open Knee(s) femoral anatomical basis is not proper orthonormal")
    rotation = np.einsum("ij,kj->ik", target_basis, source_basis)
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2.0e-5:
        raise RuntimeError("Open Knee(s) registration rotation is not proper")
    fmo_m = 0.001 * np.asarray(source.landmarks["FMO"], dtype=float)
    source_fmb_m = 0.001 * np.asarray(source.regions["FMB"].nodes_mm, dtype=float)
    source_distal = source_fmb_m[
        (_dot3(source_fmb_m - fmo_m, zf, np) >= -0.040)
        & (_dot3(source_fmb_m - fmo_m, zf, np) <= 0.035)
    ]
    target_distal = myosim_femur_body[
        (_dot3(myosim_femur_body - knee_origin_body, proximal_body, np) >= -0.040)
        & (_dot3(myosim_femur_body - knee_origin_body, proximal_body, np) <= 0.065)
    ]
    if min(len(source_distal), len(target_distal)) < 40:
        raise RuntimeError("Open Knee(s) distal-femur width samples are incomplete")
    source_width = _quantile_width(source_distal, xf, np)
    target_width = _quantile_width(target_distal, knee_axis_body, np)
    uniform_scale = target_width / source_width
    if not 0.90 <= uniform_scale <= 1.10:
        raise RuntimeError("Open Knee(s) uniform anthropometric scale left its 0.90-1.10 gate")
    translation = knee_origin_body - uniform_scale * np.einsum(
        "ij,j->i", rotation, fmo_m
    )
    transformed_fmb_body = (
        uniform_scale * np.einsum("ki,ji->kj", source_fmb_m, rotation)
        + translation
    )
    transformed_distal = transformed_fmb_body[
        (_dot3(transformed_fmb_body - knee_origin_body, proximal_body, np) >= -0.040)
        & (_dot3(transformed_fmb_body - knee_origin_body, proximal_body, np) <= 0.065)
    ]
    refinement, refinement_iterations, femur_metrics = (
        _bounded_translation_refinement(
            transformed_distal, target_distal, 0.035, np
        )
    )
    translation += refinement
    transformed_fmb_body += refinement
    if femur_metrics["p90_m"] > 0.020:
        raise RuntimeError(
            "Open Knee(s) held-out distal-femur placement exceeded 20 mm: "
            f"scale={uniform_scale:.9f} source_width={source_width:.9f} "
            f"target_width={target_width:.9f} refinement={refinement.tolist()} "
            f"metrics={femur_metrics}"
        )
    sagittal_mirror_x = None
    bilateral_frame_symmetry_maximum_m = 0.0
    if side == "right":
        left_femur_position = np.asarray(
            source_bodies["femur_l"]["default_body_position_world_m"], dtype=float
        )
        right_femur_position = np.asarray(
            source_bodies["femur_r"]["default_body_position_world_m"], dtype=float
        )
        sagittal_mirror_x = 0.5 * (
            left_femur_position[0] + right_femur_position[0]
        )
        for role in ("femur", "tibia", "patella"):
            left = np.asarray(
                source_bodies[f"{role}_l"]["default_body_position_world_m"],
                dtype=float,
            )
            right = np.asarray(
                source_bodies[f"{role}_r"]["default_body_position_world_m"],
                dtype=float,
            )
            mirrored = left.copy()
            mirrored[0] = 2.0 * sagittal_mirror_x - mirrored[0]
            bilateral_frame_symmetry_maximum_m = max(
                bilateral_frame_symmetry_maximum_m,
                float(np.linalg.norm(mirrored - right)),
            )
        if bilateral_frame_symmetry_maximum_m > 0.0005:
            raise RuntimeError(
                "Open Knee(s) right-knee mirror exceeds bilateral frame symmetry gate"
            )

    ordered_names = list(EXPECTED_REGIONS)
    region_index = {name: index for index, name in enumerate(ordered_names)}
    global_node_index: dict[int, int] = {}
    region_node_world: dict[str, Any] = {}
    region_node_visual: dict[str, Any] = {}
    node_anchor_body: dict[int, int] = {}
    node_anchor_local: dict[int, tuple[float, float, float]] = {}
    node_count = 0
    for name in ordered_names:
        region = source.regions[name]
        source_m = 0.001 * np.asarray(region.nodes_mm, dtype=float)
        femur_body_local = (
            uniform_scale * np.einsum("ki,ji->kj", source_m, rotation)
            + translation
        )
        world = np.einsum(
            "ki,ji->kj", femur_body_local, femur_body_world_rotation
        ) + femur_body_world_position
        if sagittal_mirror_x is not None:
            world = world.copy()
            world[:, 0] = 2.0 * sagittal_mirror_x - world[:, 0]
        visual_body_name = body_name(VISUAL_BODY_ROLE[name])
        visual = _world_to_core(world, visual_targets[visual_body_name], np)
        region_node_world[name] = world
        region_node_visual[name] = visual
        for local, identifier in enumerate(region.node_ids):
            global_node_index[identifier] = node_count + local
        node_count += len(region.node_ids)

    human_anterior_world = np.asarray([0.0, -1.0, 0.0], dtype=float)
    output_lateral_world = np.asarray(
        [1.0, 0.0, 0.0] if side == "left" else [-1.0, 0.0, 0.0],
        dtype=float,
    )
    knee_origin_world = np.einsum(
        "ij,j->i", femur_body_world_rotation, knee_origin_body
    ) + femur_body_world_position
    if sagittal_mirror_x is not None:
        knee_origin_world = knee_origin_world.copy()
        knee_origin_world[0] = 2.0 * sagittal_mirror_x - knee_origin_world[0]
    patella_anterior_offset_m = float(_dot3(
        np.mean(region_node_world["PTB"], axis=0) - knee_origin_world,
        human_anterior_world,
        np,
    ))
    fibula_lateral_offset_m = float(_dot3(
        np.mean(region_node_world["FBB"], axis=0)
        - np.mean(region_node_world["TBB"], axis=0),
        output_lateral_world,
        np,
    ))
    if patella_anterior_offset_m < 0.025:
        raise RuntimeError(
            "Open Knee(s) anatomical orientation gate placed the patella posteriorly"
        )
    if fibula_lateral_offset_m < 0.020:
        raise RuntimeError(
            "Open Knee(s) anatomical orientation gate placed the fibula medially"
        )
    proximal_world = _unit(np.einsum(
        "ij,j->i", femur_body_world_rotation, proximal_body), np)
    if sagittal_mirror_x is not None:
        proximal_world = proximal_world.copy()
        proximal_world[0] *= -1.0
    extensor_stack_metrics = _extensor_stack_metrics(
        region_node_world, human_anterior_world, proximal_world, np)
    _patellofemoral_surface_winding(
        source, region_node_world,
        reflected=sagittal_mirror_x is not None, np=np)

    node_sets_order = sorted(source.node_sets)
    for set_name in node_sets_order:
        if "_@_" not in set_name or not set_name.endswith("_TiesNodes"):
            continue
        owner_name, counterpart = set_name.removesuffix("_TiesNodes").split("_@_", 1)
        counterpart_role = RIGID_COUNTERPART_ROLE.get(counterpart)
        if counterpart_role is None or owner_name in RIGID_COUNTERPART_ROLE:
            continue
        target = visual_targets[body_name(counterpart_role)]
        target_body = int(target["core_body_index"])
        owner = source.regions.get(owner_name)
        if owner is None:
            raise RuntimeError(f"Open Knee(s) anchor owner {owner_name} is absent")
        owner_ids = {identifier: index for index, identifier in enumerate(owner.node_ids)}
        for identifier in source.node_sets[set_name]:
            local = owner_ids.get(identifier)
            if local is None:
                raise RuntimeError(f"Open Knee(s) rigid tie {set_name} crosses region ownership")
            global_index = global_node_index[identifier]
            previous = node_anchor_body.get(global_index)
            if previous is not None and previous != target_body:
                raise RuntimeError("Open Knee(s) node has conflicting rigid attachment owners")
            node_anchor_body[global_index] = target_body
            local_point = _world_to_core(
                region_node_world[owner_name][local : local + 1], target, np
            )[0]
            node_anchor_local[global_index] = tuple(float(value) for value in local_point)

    surfaces_by_region: dict[str, list[str]] = defaultdict(list)
    for name in source.surfaces:
        owner = name.split("_", 1)[0]
        if owner not in source.regions:
            raise RuntimeError(f"Open Knee(s) surface owner is unknown: {name}")
        surfaces_by_region[owner].append(name)
    # RegionDisk owns a contiguous surface range. Keep that physical layout in
    # the same source-authoritative region order as the region table; a global
    # alphabetical sort would make first_surface address another structure.
    for names in surfaces_by_region.values():
        names.sort()
    surfaces_order = [
        surface_name
        for region_name in ordered_names
        for surface_name in surfaces_by_region[region_name]
    ]
    if len(surfaces_order) != len(source.surfaces):
        raise RuntimeError("Open Knee(s) surface partition is incomplete")
    surface_index = {name: index for index, name in enumerate(surfaces_order)}
    surface_pair_records = []
    for name, master, slave in source.surface_pairs:
        surface_pair_records.append((name, surface_index[master], surface_index[slave]))

    payload = bytearray()
    header_bytes = HEADER_STRUCT.size
    payload.extend(b"\0" * header_bytes)
    first_node = 0
    first_tet = 0
    first_surface = 0
    region_records = []
    for name in ordered_names:
        region = source.regions[name]
        tet_count = len(region.elements) if region.element_type == "tet4" else 0
        material = source.materials.get(name, {})
        material_flags = 0
        c1 = c2 = c3 = c4 = c5 = lam_max = bulk = initial_stretch = 0.0
        fiber_world = (0.0, 0.0, 0.0)
        if REGION_KIND[name] == 2:
            c1, c2, bulk = _cartilage_material_values(name, material)
            material_flags = MATERIAL_HAS_ISOTROPIC_MOONEY_RIVLIN
        if name in source.fiber_directions:
            required = {"c1", "c2", "c3", "c4", "c5", "lam_max", "k", "initial_stretch"}
            if not required.issubset(material):
                raise RuntimeError(
                    f"Open Knee(s) source material {name} is incomplete"
                )
            c1, c2, c3, c4, c5, lam_max, bulk, initial_stretch = (
                float(material[key]) for key in
                ("c1", "c2", "c3", "c4", "c5", "lam_max", "k", "initial_stretch")
            )
            if not (
                c1 > 0.0 and c2 >= 0.0 and c3 > 0.0 and c4 > 0.0 and
                c5 > 0.0 and lam_max > 1.0 and bulk > 0.0 and
                initial_stretch >= 1.0
            ):
                raise RuntimeError(
                    f"Open Knee(s) source material {name} left its physical gate"
                )
            source_fiber = np.asarray(source.fiber_directions[name], dtype=float)
            fiber_body = np.einsum("ij,j->i", rotation, source_fiber)
            fiber_world_array = np.einsum(
                "ij,j->i", femur_body_world_rotation, fiber_body
            )
            if sagittal_mirror_x is not None:
                fiber_world_array = fiber_world_array.copy()
                fiber_world_array[0] *= -1.0
            fiber_world_array = _unit(fiber_world_array, np)
            fiber_world = tuple(float(value) for value in fiber_world_array)
            material_flags = (
                MATERIAL_HAS_HOMOGENEOUS_FIBER |
                MATERIAL_HAS_ISOCHORIC_IN_SITU_STRETCH
            )
        region_records.append(REGION_STRUCT.pack(
            _fixed_name(name, 16), REGION_KIND[name],
            int(targets[body_name(VISUAL_BODY_ROLE[name])]["core_body_index"]),
            first_node, len(region.node_ids), first_tet, tet_count,
            first_surface, len(surfaces_by_region[name]),
            c1, c2, c3, c4, c5, lam_max, bulk, initial_stretch,
            *fiber_world, material_flags,
        ))
        first_node += len(region.node_ids)
        first_tet += tet_count
        first_surface += len(surfaces_by_region[name])
    for record in region_records:
        payload.extend(record)

    first_face = 0
    surface_records = []
    for name in surfaces_order:
        surface = source.surfaces[name]
        owner = name.split("_", 1)[0]
        surface_records.append(SURFACE_STRUCT.pack(
            _fixed_name(name, 48), region_index[owner], first_face,
            len(surface.faces), 1 if name.endswith("_All_Faces") else 0, 0,
        ))
        first_face += len(surface.faces)
    for record in surface_records:
        payload.extend(record)

    first_membership = 0
    node_set_records = []
    for name in node_sets_order:
        owner = name.split("_", 1)[0]
        anchor_body = INVALID_INDEX
        if "_@_" in name and name.endswith("_TiesNodes"):
            counterpart = name.removesuffix("_TiesNodes").split("_@_", 1)[1]
            counterpart_role = RIGID_COUNTERPART_ROLE.get(counterpart)
            if counterpart_role is not None and owner not in RIGID_COUNTERPART_ROLE:
                anchor_body = int(
                    targets[body_name(counterpart_role)]["core_body_index"]
                )
        node_set_records.append(NODE_SET_STRUCT.pack(
            _fixed_name(name, 48), region_index[owner], first_membership,
            len(source.node_sets[name]), anchor_body, 0,
        ))
        first_membership += len(source.node_sets[name])
    for record in node_set_records:
        payload.extend(record)
    for name, master, slave in surface_pair_records:
        payload.extend(SURFACE_PAIR_STRUCT.pack(_fixed_name(name, 48), master, slave))

    node_index = 0
    for name in ordered_names:
        world = region_node_world[name]
        visual = region_node_visual[name]
        for local in range(len(world)):
            anchor_body = node_anchor_body.get(node_index, INVALID_INDEX)
            anchor_local = node_anchor_local.get(node_index, (0.0, 0.0, 0.0))
            flags = 1 if anchor_body != INVALID_INDEX else 0
            payload.extend(NODE_STRUCT.pack(
                *(float(value) for value in world[local]), anchor_body,
                *(float(value) for value in visual[local]), 0,
                *anchor_local, flags,
            ))
            node_index += 1
    for name in ordered_names:
        region = source.regions[name]
        if region.element_type != "tet4":
            continue
        for element in region.elements:
            oriented = _orientation_preserving_connectivity(
                element, reflected=sagittal_mirror_x is not None
            )
            payload.extend(TETRAHEDRON_STRUCT.pack(
                *(global_node_index[identifier] for identifier in oriented)
            ))
    for name in surfaces_order:
        for face in source.surfaces[name].faces:
            oriented = _orientation_preserving_connectivity(
                face, reflected=sagittal_mirror_x is not None
            )
            payload.extend(FACE_STRUCT.pack(
                *(global_node_index[identifier] for identifier in oriented)
            ))
    for name in node_sets_order:
        for identifier in source.node_sets[name]:
            payload.extend(MEMBERSHIP_STRUCT.pack(global_node_index[identifier]))

    hashes = b"".join(bytes.fromhex(EXPECTED_HASHES[name]) for name in (
        "Geometry.feb", "ModelProperties.xml", "FeBio_custom.feb", "license.txt"
    ))
    payload[:header_bytes] = HEADER_STRUCT.pack(
        MAGIC, ABI, header_bytes, len(ordered_names), node_count, first_tet,
        len(surfaces_order), first_face, len(node_sets_order), first_membership,
        len(surface_pair_records), 1 if side == "right" else 0, 0, hashes,
    )
    output.mkdir(parents=True, exist_ok=True)
    payload_stem = (
        "open-knee-oks003-left" if side == "left"
        else "open-knee-oks003-right-mirrored"
    )
    payload_path = output / f"{payload_stem}.nhknee"
    payload_path.write_bytes(payload)

    attachment_counts = defaultdict(int)
    for body in node_anchor_body.values():
        attachment_counts[body] += 1
    manifest = {
        "schema": SCHEMA,
        "status": (
            "equality_projected_visual_frame_candidate"
            if projected_visual_frame else
            "exact_source_payload_registered_to_live_left_knee_candidate"
            if side == "left" else
            "exact_left_source_topology_mirrored_to_live_right_knee_candidate"
        ),
        "visual_reference_frame": {
            "pose": visual_reference_pose,
            "maximum_body_origin_shift_from_default_m":
                projected_visual_maximum_body_shift_m,
            "changes_rest_world_positions": False,
            "changes_source_tetrahedra": False,
        },
        "source": {
            "dataset": "Open Knee(s) oks003",
            "doi": "10.18735/b0zv-n395",
            "license": "CC BY 4.0",
            "files": {name: {"sha256": digest} for name, digest in EXPECTED_HASHES.items()},
            "subject": {"side": "left", "sex": "female", "age_years": 25,
                        "height_m": 1.73, "mass_kg": 68.0, "bmi": 22.8},
        },
        "source_mechanical_program": compile_source_mechanical_description(
            source,
            source_deck=open_knee / "FeBio_custom.feb",
            expected_deck_sha256=EXPECTED_HASHES["FeBio_custom.feb"],
            source_geometry_archive_sha256=ARCHIVED_REFERENCE_GEOMETRY_SHA256,
            source_geometry_archive_path=source_geometry_archive_path,
            source_mesh_data_output_path=source_mesh_data_output_path,
            reference_solver_version=ARCHIVED_REFERENCE_SOLVER_VERSION,
            reference_log_sha256=ARCHIVED_REFERENCE_LOG_SHA256,
        ),
        "registration": {
            "method": (
                "FMO_to_live_left_knee_origin_Xf_to_flexion_axis_Zf_to_proximal_axis_uniform_condylar_width_scale"
                if side == "left"
                else "qualified_left_world_registration_then_sagittal_mirror_into_live_right_knee_frames"
            ),
            "output_side": side,
            "sagittal_mirror_world_x_m": sagittal_mirror_x,
            "bilateral_frame_symmetry_maximum_m": bilateral_frame_symmetry_maximum_m,
            "source_axes": {name: list(source.landmarks[name]) for name in (
                "Xf_axis", "Yf_axis", "Zf_axis"
            )},
            "source_origin_mm": list(source.landmarks["FMO"]),
            "target_knee_origin_femur_body_m": [float(value) for value in knee_origin_body],
            "target_flexion_axis_femur_body": [float(value) for value in knee_axis_body],
            "target_anterior_axis_femur_body": [float(value) for value in anterior_body],
            "target_proximal_axis_femur_body": [float(value) for value in proximal_body],
            "target_anterior_world": [0.0, -1.0, 0.0],
            "target_anterior_alignment": anterior_alignment,
            "patella_anterior_offset_m": patella_anterior_offset_m,
            "fibula_lateral_offset_m": fibula_lateral_offset_m,
            "patellar_extensor_stack": extensor_stack_metrics,
            "proper_rotation_source_to_femur_body": rotation.tolist(),
            "translation_femur_body_m": [float(value) for value in translation],
            "FMO_to_mechanics_origin_initial_translation_femur_body_m": [
                float(value) for value in (
                    knee_origin_body - uniform_scale * np.einsum(
                        "ij,j->i", rotation, fmo_m
                    )
                )
            ],
            "bounded_surface_translation_refinement_femur_body_m": [
                float(value) for value in refinement
            ],
            "bounded_surface_translation_refinement_norm_m": float(
                np.linalg.norm(refinement)
            ),
            "bounded_surface_translation_refinement_maximum_m": 0.035,
            "bounded_surface_translation_refinement_iterations": refinement_iterations,
            "uniform_scale": uniform_scale,
            "source_condylar_width_m": source_width,
            "target_condylar_width_m": target_width,
            "distal_femur_surface_metrics": femur_metrics,
            "gates": {
                "proper_rotation_determinant": float(np.linalg.det(rotation)),
                "anterior_alignment_minimum": 0.999,
                "patella_anterior_offset_minimum_m": 0.025,
                "fibula_lateral_offset_minimum_m": 0.020,
                "patellar_cartilage_posterior_to_bone_minimum_m": EXTENSOR_STACK_MINIMUMS_M[
                    "patellar_cartilage_posterior_to_bone_m"],
                "quadriceps_tendon_proximal_to_patella_minimum_m": EXTENSOR_STACK_MINIMUMS_M[
                    "quadriceps_tendon_proximal_to_patella_m"],
                "patellar_tendon_distal_to_patella_minimum_m": EXTENSOR_STACK_MINIMUMS_M[
                    "patellar_tendon_distal_to_patella_m"],
                "uniform_scale_minimum": 0.90, "uniform_scale_maximum": 1.10,
                "held_out_p90_maximum_m": 0.020,
                "reflection": side == "right", "anisotropic_warp": False,
                "reflection_scope": (
                    "none" if side == "left"
                    else "one_world_sagittal_mirror_of_the_qualified_left_specimen"
                ),
                "connectivity_parity_correction": (
                    "none" if side == "left"
                    else "swap_first_two_indices_of_each_tet4_and_tri3"
                ),
                "extra_joint": False,
            },
        },
        "runtime_binding": {
            body_name(role): {
                "core_body_index": int(targets[body_name(role)]["core_body_index"])
            }
            for role in ("femur", "patella", "tibia")
        },
        "topology": {
            "region_count": len(ordered_names),
            "node_count": node_count,
            "tetrahedron_count": first_tet,
            "surface_count": len(surfaces_order),
            "surface_face_count": first_face,
            "node_set_count": len(node_sets_order),
            "node_set_membership_count": first_membership,
            "surface_pair_count": len(surface_pair_records),
            "rigid_attachment_node_count_by_body": {
                str(body): count for body, count in sorted(attachment_counts.items())
            },
            "regions": [
                {"name": name, "kind": REGION_KIND[name],
                 "visual_body": body_name(VISUAL_BODY_ROLE[name]),
                 "nodes": len(source.regions[name].node_ids),
                 "element_type": source.regions[name].element_type,
                 "elements": len(source.regions[name].elements),
                 "all_surface_faces": len(source.surfaces[f"{name}_All_Faces"].faces)}
                for name in ordered_names
            ],
            "node_sets": {name: len(source.node_sets[name]) for name in node_sets_order},
            "surface_pairs": [
                {"name": name, "master": master, "slave": slave}
                for name, master, slave in source.surface_pairs
            ],
        },
        "materials": source.materials,
        "homogeneous_fiber_directions": {
            name: {
                "source": list(source.fiber_directions[name]),
                "registered_world": list(struct.unpack(
                    "<3f", region_records[region_index[name]][80:92]
                )),
            }
            for name in sorted(source.fiber_directions)
        },
        "payload": {
            "file": payload_path.name, "magic": "NHKNEE1", "abi": ABI,
            "bytes": len(payload), "sha256": _sha256(payload_path),
        },
        "evidence_boundary": (
            "This is the registered reduced-hybrid payload, not a complete executable "
            "FEBio mechanical problem or source-equivalent knee. It preserves selected "
            "oks003 geometry, topology, material parameters, attachments and surface pairs; "
            "it does not execute the source rigid-joint, prestrain, contact and load program. "
            "The left output has a bounded anatomical registration; the "
            "right output is its explicitly labelled sagittal mirror in the measured live "
            "bilateral frames, not an independently segmented right specimen. Neither is "
            "subject-matched, a coarsened Apple FEM solve, or admitted production contact."
        ),
    }
    manifest_path = output / f"{payload_stem}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--open-knee", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--projected-visual-frame", action="store_true")
    arguments = parser.parse_args(argv)
    compile_payload(
        sources=arguments.sources.resolve(),
        open_knee=arguments.open_knee.resolve(),
        registration_path=arguments.registration.resolve(),
        output=arguments.output.resolve(),
        side=arguments.side,
        projected_visual_frame=arguments.projected_visual_frame,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
