"""Compile the exact Open Knee(s) oks003 knee into a mechanics-ready payload.

Open Knee(s) remains the anatomical/tissue authority. MyoSim supplies only the
live body frames and joint axis used to place the specimen in Numi Human. The
compiler admits one proper uniform scale, one anatomically constructed proper
rotation, and one knee-origin translation; it never performs an anisotropic
warp or an unconstrained nearest-surface flip.
"""

from __future__ import annotations

import argparse
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


SCHEMA = "numi.human.open-knee-oks003-payload.v2"
MAGIC = b"NHKNEE1\0"
ABI = 2
INVALID_INDEX = 0xFFFFFFFF

EXPECTED_HASHES = {
    "Geometry.feb": "3642bd368bbc867569f181fa76129f746470e807e3977585d3803f092dd11262",
    "ModelProperties.xml": "0ac446ce098b9a09505992eb4f4419c7b944cd57a4afbf6392b137f4806603c1",
    "FeBio_custom.feb": "00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a",
    "license.txt": "d72918838b4adf30979d2a26c23837f0ca05185ba799a3a4fe1fe1b4c05b20b8",
}

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
        fiber_directions,
    )


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
    side: str = "left",
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
        visual = _world_to_core(world, targets[visual_body_name], np)
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

    node_sets_order = sorted(source.node_sets)
    for set_name in node_sets_order:
        if "_@_" not in set_name or not set_name.endswith("_TiesNodes"):
            continue
        owner_name, counterpart = set_name.removesuffix("_TiesNodes").split("_@_", 1)
        counterpart_role = RIGID_COUNTERPART_ROLE.get(counterpart)
        if counterpart_role is None or owner_name in RIGID_COUNTERPART_ROLE:
            continue
        target = targets[body_name(counterpart_role)]
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
            "exact_source_payload_registered_to_live_left_knee_candidate"
            if side == "left"
            else "exact_left_source_topology_mirrored_to_live_right_knee_candidate"
        ),
        "source": {
            "dataset": "Open Knee(s) oks003",
            "doi": "10.18735/b0zv-n395",
            "license": "CC BY 4.0",
            "files": {name: {"sha256": digest} for name, digest in EXPECTED_HASHES.items()},
            "subject": {"side": "left", "sex": "female", "age_years": 25,
                        "height_m": 1.73, "mass_kg": 68.0, "bmi": 22.8},
        },
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
            "This preserves exact oks003 specimen geometry, topology, material, attachment, "
            "and contact data. The left output has a bounded anatomical registration; the "
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
    arguments = parser.parse_args(argv)
    compile_payload(
        sources=arguments.sources.resolve(),
        open_knee=arguments.open_knee.resolve(),
        registration_path=arguments.registration.resolve(),
        output=arguments.output.resolve(),
        side=arguments.side,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
