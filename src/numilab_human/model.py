from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ImportError(ValueError):
    """Raised when a source cannot make a source-faithful Human v1 artifact."""


# Rajagopal's serialized Millard muscles omit these optional properties.  They
# are the documented OpenSim Millard2012EquilibriumMuscle class defaults, kept
# explicit here so a learned-excitation path never silently assumes a cadence.
_MILLARD_ACTIVATION_DEFAULTS = {
    "activation_time_constant_seconds": 0.01,
    "deactivation_time_constant_seconds": 0.04,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ImportError(f"required file is absent: {path}") from error
    except json.JSONDecodeError as error:
        raise ImportError(f"invalid JSON in {path}: {error}") from error


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _text(element: ET.Element, name: str) -> str | None:
    child = next((item for item in element if _local_name(item) == name), None)
    if child is None or child.text is None:
        return None
    result = " ".join(child.text.split())
    return result or None


def _number_or_text(value: str | None) -> float | list[float] | str | None:
    if value is None:
        return None
    tokens = value.split()
    try:
        numbers = [float(token) for token in tokens]
    except ValueError:
        return value
    return numbers[0] if len(numbers) == 1 else numbers


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [item for item in element.iter() if _local_name(item) == name]


def _opensim_direct_properties(element: ET.Element, names: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        value = _number_or_text(_text(element, name))
        if value is not None:
            result[name] = value
    return result


def _opensim_direct_leaf_properties(
    element: ET.Element, excluded: Iterable[str] = ()
) -> dict[str, Any]:
    """Preserve every direct scalar OpenSim property without flattening trees."""
    ignored = set(excluded)
    result: dict[str, Any] = {}
    for child in element:
        name = _local_name(child)
        if name in ignored or any(isinstance(item.tag, str) for item in child):
            continue
        value = _number_or_text(child.text)
        if value is not None:
            result[name] = value
    return result


def _source_xml(element: ET.Element) -> str:
    """Keep a source subtree available to a future lowerer that needs more data."""
    return ET.tostring(element, encoding="unicode")


def _joint_frames(joint: ET.Element) -> list[dict[str, Any]]:
    frames = next((item for item in joint if _local_name(item) == "frames"), None)
    if frames is None:
        return []
    result: list[dict[str, Any]] = []
    for frame in frames:
        identifier = frame.get("name")
        if not identifier:
            continue
        result.append(
            {
                "id": identifier,
                "kind": _local_name(frame),
                "parent_frame": _text(frame, "socket_parent") or _text(frame, "socket_parent_frame"),
                "translation_m": _number_or_text(_text(frame, "translation")),
                "orientation_rad": _number_or_text(_text(frame, "orientation")),
            }
        )
    return result


def _body_inertia(body: ET.Element) -> dict[str, float]:
    """Normalize either OpenSim inertia spelling into one explicit tensor map."""
    compact = _number_or_text(_text(body, "inertia"))
    names = ("xx", "yy", "zz", "xy", "xz", "yz")
    if compact is not None:
        if (
            not isinstance(compact, list)
            or len(compact) != len(names)
            or not all(isinstance(value, float) for value in compact)
        ):
            raise ImportError(
                f"Body {body.get('name', '<unnamed>')} has malformed OpenSim inertia"
            )
        return dict(zip(names, compact, strict=True))
    direct = _opensim_direct_properties(
        body,
        tuple(f"inertia_{name}" for name in names),
    )
    if not direct:
        return {}
    if not {"inertia_xx", "inertia_yy", "inertia_zz"}.issubset(direct):
        raise ImportError(
            f"Body {body.get('name', '<unnamed>')} has incomplete principal inertia"
        )
    return {
        name: float(direct.get(f"inertia_{name}", 0.0))
        for name in names
    }


def _joint_motion_axes(joint: ET.Element) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for axis in joint.iter():
        if _local_name(axis) != "TransformAxis":
            continue
        function = next(
            (
                item
                for item in axis
                if _local_name(item) not in {"coordinates", "axis"}
            ),
            None,
        )
        function_kind = None
        function_element = function
        if function is not None:
            if _local_name(function) == "function":
                function_element = next(
                    (item for item in function if isinstance(item.tag, str)),
                    None,
                )
                function_kind = (
                    _local_name(function_element)
                    if function_element is not None
                    else None
                )
            else:
                function_kind = _local_name(function)
        result.append(
            {
                "id": axis.get("name"),
                "coordinates": _text(axis, "coordinates"),
                "axis": _number_or_text(_text(axis, "axis")),
                "function_kind": function_kind,
                "function_parameters": (
                    _opensim_direct_leaf_properties(function_element)
                    if function_element is not None
                    else {}
                ),
                "function_source_xml": (
                    _source_xml(function_element)
                    if function_element is not None
                    else None
                ),
                # A lowerer must preserve this transform's source semantics;
                # carrying its XML prevents a lossy host-side approximation.
                "source_xml": _source_xml(axis),
            }
        )
    return result


_SIMM_TINY = 1.0e-7
_SIMM_ROUNDOFF = 2.0e-13


def _finite_scalar(value: Any, context: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ImportError(f"{context} must be a finite scalar")
    return value


def _finite_scalars(value: Any, context: str, minimum: int = 1) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) < minimum
        or not all(isinstance(item, float) and math.isfinite(item) for item in value)
    ):
        raise ImportError(f"{context} must be {minimum} or more finite scalars")
    return value


def _simm_spline_coefficients(
    abscissae: list[float], ordinates: list[float]
) -> tuple[list[float], list[float], list[float]]:
    """Match OpenSim's serialized SimmSpline coefficient construction."""
    if len(abscissae) != len(ordinates) or len(abscissae) < 2:
        raise ImportError("SimmSpline requires equally sized x/y arrays with at least two points")
    if any(right <= left for left, right in zip(abscissae, abscissae[1:])):
        raise ImportError("SimmSpline x values must be strictly increasing")
    count = len(abscissae)
    slope = [0.0] * count
    quadratic = [0.0] * count
    cubic = [0.0] * count
    if count == 2:
        value = (ordinates[1] - ordinates[0]) / max(_SIMM_TINY, abscissae[1] - abscissae[0])
        return [value, value], quadratic, cubic

    final = count - 1
    penultimate = count - 2
    cubic[0] = max(_SIMM_TINY, abscissae[1] - abscissae[0])
    quadratic[1] = (ordinates[1] - ordinates[0]) / cubic[0]
    for index in range(1, final):
        cubic[index] = max(_SIMM_TINY, abscissae[index + 1] - abscissae[index])
        slope[index] = 2.0 * (cubic[index - 1] + cubic[index])
        quadratic[index + 1] = (ordinates[index + 1] - ordinates[index]) / cubic[index]
        quadratic[index] = quadratic[index + 1] - quadratic[index]

    slope[0] = -cubic[0]
    slope[final] = -cubic[penultimate]
    quadratic[0] = 0.0
    quadratic[final] = 0.0
    if count > 3:
        d31 = max(_SIMM_TINY, abscissae[3] - abscissae[1])
        d20 = max(_SIMM_TINY, abscissae[2] - abscissae[0])
        d1 = max(_SIMM_TINY, abscissae[final] - abscissae[count - 3])
        d2 = max(_SIMM_TINY, abscissae[penultimate] - abscissae[count - 4])
        d30 = max(_SIMM_TINY, abscissae[3] - abscissae[0])
        d3 = max(_SIMM_TINY, abscissae[final] - abscissae[count - 4])
        quadratic[0] = (quadratic[2] / d31 - quadratic[1] / d20) * cubic[0] * cubic[0] / d30
        quadratic[final] = -(
            quadratic[penultimate] / d1 - quadratic[count - 3] / d2
        ) * cubic[penultimate] * cubic[penultimate] / d3

    for index in range(1, count):
        scale = cubic[index - 1] / slope[index - 1]
        slope[index] -= scale * cubic[index - 1]
        quadratic[index] -= scale * quadratic[index - 1]
    quadratic[final] /= slope[final]
    for offset in range(final):
        index = penultimate - offset
        quadratic[index] = (quadratic[index] - cubic[index] * quadratic[index + 1]) / slope[index]

    slope[final] = (
        (ordinates[final] - ordinates[penultimate]) / cubic[penultimate]
        + cubic[penultimate] * (quadratic[penultimate] + 2.0 * quadratic[final])
    )
    for index in range(final):
        slope[index] = (
            (ordinates[index + 1] - ordinates[index]) / cubic[index]
            - cubic[index] * (quadratic[index + 1] + 2.0 * quadratic[index])
        )
        cubic[index] = (quadratic[index + 1] - quadratic[index]) / cubic[index]
        quadratic[index] *= 3.0
    quadratic[final] *= 3.0
    cubic[final] = cubic[penultimate]
    return slope, quadratic, cubic


def evaluate_opensim_axis_function(
    axis: dict[str, Any], coordinate_values: dict[str, float]
) -> tuple[float, float]:
    """Return an OpenSim TransformAxis displacement and first derivative."""
    value, derivative, _ = _evaluate_opensim_axis_function_with_second_derivative(
        axis, coordinate_values
    )
    return value, derivative


def _evaluate_opensim_axis_function_with_second_derivative(
    axis: dict[str, Any], coordinate_values: dict[str, float]
) -> tuple[float, float, float]:
    """Return source displacement plus its first two scalar derivatives.

    The supported function set is exactly the one in the pinned Rajagopal
    source. The second derivative is retained for the FunctionBased Hdot
    term; it does not claim that the host evaluator is the articulated solver.
    """
    kind = axis.get("function_kind")
    parameters = axis.get("function_parameters")
    if not isinstance(parameters, dict):
        raise ImportError(f"OpenSim TransformAxis {axis.get('id')} has no function parameters")
    coordinate = axis.get("coordinates")
    coordinate = coordinate.strip() if isinstance(coordinate, str) else ""
    if coordinate:
        if any(character.isspace() for character in coordinate):
            raise ImportError(
                f"OpenSim TransformAxis {axis.get('id')} has a multi-coordinate function; "
                "the source evaluator requires an explicit multi-coordinate extension"
            )
        try:
            argument = _finite_scalar(
                coordinate_values[coordinate], f"coordinate {coordinate}"
            )
        except KeyError as error:
            raise ImportError(f"missing OpenSim coordinate value {coordinate}") from error
    elif kind != "Constant":
        raise ImportError(f"OpenSim TransformAxis {axis.get('id')} has no independent coordinate")
    else:
        argument = 0.0

    if kind == "Constant":
        return _finite_scalar(parameters.get("value"), "OpenSim Constant value"), 0.0, 0.0
    if kind == "LinearFunction":
        coefficients = _finite_scalars(
            parameters.get("coefficients"), "OpenSim LinearFunction coefficients", 2
        )
        if len(coefficients) != 2:
            raise ImportError("OpenSim LinearFunction must have slope and intercept")
        return coefficients[0] * argument + coefficients[1], coefficients[0], 0.0
    if kind == "PolynomialFunction":
        coefficients = _finite_scalars(
            parameters.get("coefficients"), "OpenSim PolynomialFunction coefficients"
        )
        value = 0.0
        derivative = 0.0
        second_derivative = 0.0
        for coefficient in coefficients:
            second_derivative = second_derivative * argument + 2.0 * derivative
            derivative = derivative * argument + value
            value = value * argument + coefficient
        return value, derivative, second_derivative
    if kind == "SimmSpline":
        abscissae = _finite_scalars(parameters.get("x"), "OpenSim SimmSpline x", 2)
        ordinates = _finite_scalars(parameters.get("y"), "OpenSim SimmSpline y", 2)
        slope, quadratic, cubic = _simm_spline_coefficients(abscissae, ordinates)
        final = len(abscissae) - 1
        if argument < abscissae[0]:
            return ordinates[0] + (argument - abscissae[0]) * slope[0], slope[0], 0.0
        if argument > abscissae[final]:
            return ordinates[final] + (argument - abscissae[final]) * slope[final], slope[final], 0.0
        if abs(argument - abscissae[0]) <= _SIMM_ROUNDOFF:
            return ordinates[0], slope[0], 2.0 * quadratic[0]
        if abs(argument - abscissae[final]) <= _SIMM_ROUNDOFF:
            return ordinates[final], slope[final], 2.0 * quadratic[final]
        low, high = 0, final
        while True:
            index = (low + high) // 2
            if argument < abscissae[index]:
                high = index
            elif argument > abscissae[index + 1]:
                low = index
            else:
                break
        delta = argument - abscissae[index]
        return (
            ordinates[index] + delta * (slope[index] + delta * (quadratic[index] + delta * cubic[index])),
            slope[index] + delta * (2.0 * quadratic[index] + 3.0 * delta * cubic[index]),
            2.0 * quadratic[index] + 6.0 * delta * cubic[index],
        )
    raise ImportError(
        f"OpenSim TransformAxis {axis.get('id')} has unsupported function {kind!r}"
    )


def _normalize_spatial_axis(value: list[float], label: str) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(magnitude) or magnitude <= 1.0e-10:
        raise ImportError(f"{label} must have a finite non-zero axis")
    return [component / magnitude for component in value]


def _cross3(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _add3(left: list[float], right: list[float]) -> list[float]:
    return [left[index] + right[index] for index in range(3)]


def _scaled3(value: list[float], scalar: float) -> list[float]:
    return [component * scalar for component in value]


def _matrix_multiply3(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][cursor] * right[cursor][column] for cursor in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _matrix_apply3(matrix: list[list[float]], value: list[float]) -> list[float]:
    return [sum(matrix[row][column] * value[column] for column in range(3)) for row in range(3)]


def _axis_angle_matrix3(axis: list[float], angle: float) -> list[list[float]]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    remainder = 1.0 - cosine
    x, y, z = axis
    return [
        [cosine + x * x * remainder, x * y * remainder - z * sine, x * z * remainder + y * sine],
        [y * x * remainder + z * sine, cosine + y * y * remainder, y * z * remainder - x * sine],
        [z * x * remainder - y * sine, z * y * remainder + x * sine, cosine + z * z * remainder],
    ]


def _evaluate_opensim_spatial_transform(
    axes: list[dict[str, Any]], coordinate_values: dict[str, float], coordinate_velocities: dict[str, float]
) -> dict[str, Any]:
    """Evaluate Simbody FunctionBased pose, H, and Hdot in source axis order."""
    normalized_axes = [
        _normalize_spatial_axis(axis["axis"], f"OpenSim TransformAxis {axis.get('id')} axis")
        for axis in axes
    ]
    for first, second in ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)):
        cross = _cross3(normalized_axes[first], normalized_axes[second])
        if sum(component * component for component in cross) <= 1.0e-10:
            raise ImportError("OpenSim FunctionBased SpatialTransform has colinear source axes")

    values: list[tuple[float, float, float]] = [
        _evaluate_opensim_axis_function_with_second_derivative(axis, coordinate_values)
        for axis in axes
    ]
    rotation0 = _axis_angle_matrix3(normalized_axes[0], values[0][0])
    rotation01 = _matrix_multiply3(
        rotation0, _axis_angle_matrix3(normalized_axes[1], values[1][0])
    )
    rotation = _matrix_multiply3(
        rotation01, _axis_angle_matrix3(normalized_axes[2], values[2][0])
    )
    translation = [0.0, 0.0, 0.0]
    for index in range(3, 6):
        translation = _add3(translation, _scaled3(normalized_axes[index], values[index][0]))

    angular_axes = [
        normalized_axes[0],
        _matrix_apply3(rotation0, normalized_axes[1]),
        _matrix_apply3(rotation01, normalized_axes[2]),
    ]
    def velocity_for(index: int) -> float:
        coordinate = axes[index].get("coordinates")
        if not isinstance(coordinate, str) or not coordinate.strip():
            return 0.0
        return _finite_scalar(
            coordinate_velocities[coordinate.strip()], f"OpenSim coordinate velocity {coordinate.strip()}"
        )

    theta_dot0 = values[0][1] * velocity_for(0)
    theta_dot1 = values[1][1] * velocity_for(1)
    angular_axis_dots = [
        [0.0, 0.0, 0.0],
        _cross3(_scaled3(angular_axes[0], theta_dot0), angular_axes[1]),
        _cross3(
            _add3(_scaled3(angular_axes[0], theta_dot0), _scaled3(angular_axes[1], theta_dot1)),
            angular_axes[2],
        ),
    ]
    motion = {
        coordinate: {"coordinate": coordinate, "angular": [0.0, 0.0, 0.0], "linear": [0.0, 0.0, 0.0]}
        for coordinate in coordinate_values
    }
    motion_dot = {
        coordinate: {"coordinate": coordinate, "angular": [0.0, 0.0, 0.0], "linear": [0.0, 0.0, 0.0]}
        for coordinate in coordinate_values
    }
    for index, axis in enumerate(axes):
        coordinate = axis.get("coordinates")
        coordinate = coordinate.strip() if isinstance(coordinate, str) else ""
        if not coordinate:
            continue
        derivative = values[index][1]
        derivative_dot = values[index][2] * velocity_for(index)
        if index < 3:
            motion[coordinate]["angular"] = _add3(
                motion[coordinate]["angular"], _scaled3(angular_axes[index], derivative)
            )
            motion_dot[coordinate]["angular"] = _add3(
                motion_dot[coordinate]["angular"],
                _add3(
                    _scaled3(angular_axis_dots[index], derivative),
                    _scaled3(angular_axes[index], derivative_dot),
                ),
            )
        else:
            motion[coordinate]["linear"] = _add3(
                motion[coordinate]["linear"], _scaled3(normalized_axes[index], derivative)
            )
            motion_dot[coordinate]["linear"] = _add3(
                motion_dot[coordinate]["linear"], _scaled3(normalized_axes[index], derivative_dot)
            )
    return {
        "composition": "R0(axis0)*R1(axis1)*R2(axis2); p=sum(translation_axis_i*function_i)",
        "rotation_parent_frame_row_major": [component for row in rotation for component in row],
        "translation_parent_frame_m": translation,
        "motion_subspace_parent_frame": [motion[coordinate] for coordinate in coordinate_values],
        "motion_subspace_dot_parent_frame": [motion_dot[coordinate] for coordinate in coordinate_values],
    }


def evaluate_opensim_custom_joint(
    joint: dict[str, Any],
    coordinate_values: dict[str, float] | None = None,
    coordinate_velocities: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate source FunctionBased axes, pose, H, and Hdot without lowering it."""
    if joint.get("kind") != "CustomJoint":
        raise ImportError(f"OpenSim joint {joint.get('id')} is not a CustomJoint")
    supplied = {} if coordinate_values is None else dict(coordinate_values)
    for coordinate in joint.get("coordinates", []):
        identifier = coordinate.get("id")
        if not isinstance(identifier, str):
            raise ImportError(f"OpenSim CustomJoint {joint.get('id')} has unnamed coordinates")
        if identifier not in supplied:
            supplied[identifier] = _finite_scalar(
                coordinate.get("default_value"), f"OpenSim coordinate {identifier} default value"
            )
    values = {
        identifier: _finite_scalar(value, f"OpenSim coordinate {identifier}")
        for identifier, value in supplied.items()
    }
    supplied_velocities = {} if coordinate_velocities is None else dict(coordinate_velocities)
    velocities = {
        coordinate["id"]: _finite_scalar(
            supplied_velocities.get(coordinate["id"], 0.0),
            f"OpenSim coordinate velocity {coordinate['id']}",
        )
        for coordinate in joint.get("coordinates", [])
        if isinstance(coordinate.get("id"), str)
    }
    axes = joint.get("motion_axes")
    if not isinstance(axes, list) or len(axes) != 6:
        raise ImportError(
            f"OpenSim CustomJoint {joint.get('id')} must retain six SpatialTransform axes"
        )
    evaluated = []
    for index, axis in enumerate(axes):
        displacement, derivative, second_derivative = (
            _evaluate_opensim_axis_function_with_second_derivative(axis, values)
        )
        direction = _vector3(axis.get("axis"), f"OpenSim TransformAxis {axis.get('id')} axis")
        evaluated.append(
            {
                "id": axis.get("id"),
                "spatial_kind": "rotation" if index < 3 else "translation",
                "axis": direction,
                "coordinate": axis.get("coordinates") or None,
                "function_kind": axis.get("function_kind"),
                "displacement": displacement,
                "derivative": derivative,
                "second_derivative": second_derivative,
            }
        )
    return {
        "source_joint": joint.get("id"),
        "coordinate_values": values,
        "coordinate_velocities": velocities,
        "axes": evaluated,
        "spatial_transform": _evaluate_opensim_spatial_transform(axes, values, velocities),
    }


def rajagopal_custom_joint_ir(model: dict[str, Any]) -> dict[str, Any]:
    """Emit source FunctionBased tables plus pose/H/Hdot test vectors."""
    joints = [joint for joint in model.get("joints", []) if joint.get("kind") == "CustomJoint"]
    compiled = []
    for joint in joints:
        default = evaluate_opensim_custom_joint(joint)
        compiled.append(
            {
                "id": joint["id"],
                "coordinates": joint["coordinates"],
                "spatial_transform": [
                    {
                        "id": axis["id"],
                        "coordinates": axis["coordinates"],
                        "axis": axis["axis"],
                        "function_kind": axis["function_kind"],
                        "function_parameters": axis["function_parameters"],
                    }
                    for axis in joint["motion_axes"]
                ],
                "default_value_test_vector": default,
                "unit_velocity_test_vectors": [
                    evaluate_opensim_custom_joint(
                        joint, coordinate_velocities={coordinate["id"]: 1.0}
                    )
                    for coordinate in joint["coordinates"]
                ],
            }
        )
    return {
        "schema": "numi.human.opensim-custom-joint-ir.v2",
        "source": {
            "id": model.get("source_id"),
            "file": model.get("source_file"),
            "sha256": model.get("source_sha256"),
            "model_id": model.get("model_id"),
        },
        "joint_count": len(compiled),
        "function_kinds": dict(
            sorted(
                Counter(
                    axis["function_kind"]
                    for joint in compiled
                    for axis in joint["spatial_transform"]
                ).items()
            )
        ),
        "runtime_requirement": (
            "Function-based articulated joints with source-order SpatialTransform composition, "
            "analytic H/Hdot, and per-coordinate generalized-force projection."
        ),
        "evidence_boundary": (
            "Source semantics and compiler test vectors only. The pinned core revision has a "
            "matching bounded Metal kinematic evaluator, but the articulated solver does not "
            "yet consume this IR; this is not a dynamics proof or a replacement for OpenSim."
        ),
        "joints": compiled,
    }


# These fields mirror MetalRobo's
# include/metalrobo/opensim_spatial_transform_gpu.h.  They are deliberately
# written here rather than inferred from a host compiler so a provenance-locked
# Human artifact can be transferred to the pinned Apple runtime unchanged.
_OPENSIM_SPATIAL_GPU_ABI_VERSION = 1
_OPENSIM_SPATIAL_GPU_AXIS_COUNT = 6
_OPENSIM_SPATIAL_GPU_MAX_COORDINATES = 6
_OPENSIM_SPATIAL_GPU_MAX_COEFFICIENTS = 16
_OPENSIM_SPATIAL_GPU_MAX_KNOTS = 16
_OPENSIM_SPATIAL_GPU_NO_COORDINATE = 0xFFFFFFFF
_OPENSIM_SPATIAL_GPU_PROGRAM_BYTES = 2512
_OPENSIM_SPATIAL_GPU_INPUT_BYTES = 64
_OPENSIM_SPATIAL_GPU_FUNCTION_KIND = {
    "Constant": 0,
    "LinearFunction": 1,
    "PolynomialFunction": 2,
    "SimmSpline": 3,
}


def _pack_opensim_float_block(
    values: list[float], capacity: int, context: str
) -> bytes:
    """Pack a fixed-capacity float stream in the Core ABI's float4 order."""
    if len(values) > capacity or not all(math.isfinite(value) for value in values):
        raise ImportError(f"{context} exceeds its finite GPU capacity")
    padded = values + [0.0] * (capacity - len(values))
    try:
        return struct.pack(f"<{capacity}f", *padded)
    except OverflowError as error:
        raise ImportError(f"{context} is outside the finite FP32 GPU range") from error


def _opensim_gpu_function_bytes(
    axis: dict[str, Any], coordinate_index: int
) -> bytes:
    """Compile one parsed TransformAxis to MetalRobo's fixed ABI record."""
    kind_name = axis.get("function_kind")
    if kind_name not in _OPENSIM_SPATIAL_GPU_FUNCTION_KIND:
        raise ImportError(
            f"OpenSim TransformAxis {axis.get('id')} has unsupported GPU function {kind_name!r}"
        )
    parameters = axis.get("function_parameters")
    if not isinstance(parameters, dict):
        raise ImportError(f"OpenSim TransformAxis {axis.get('id')} has no function parameters")
    kind = _OPENSIM_SPATIAL_GPU_FUNCTION_KIND[kind_name]
    coefficients: list[float] = []
    abscissae: list[float] = []
    ordinates: list[float] = []
    slope: list[float] = []
    quadratic: list[float] = []
    cubic: list[float] = []
    if kind_name == "Constant":
        coefficients = [_finite_scalar(parameters.get("value"), f"OpenSim {axis.get('id')} value")]
        if coordinate_index != _OPENSIM_SPATIAL_GPU_NO_COORDINATE:
            raise ImportError(f"OpenSim constant TransformAxis {axis.get('id')} selects a coordinate")
    elif kind_name == "LinearFunction":
        coefficients = _finite_scalars(
            parameters.get("coefficients"), f"OpenSim {axis.get('id')} coefficients"
        )
        if len(coefficients) != 2:
            raise ImportError(f"OpenSim LinearFunction {axis.get('id')} requires two coefficients")
    elif kind_name == "PolynomialFunction":
        coefficients = _finite_scalars(
            parameters.get("coefficients"), f"OpenSim {axis.get('id')} coefficients"
        )
    else:
        abscissae = _finite_scalars(parameters.get("x"), f"OpenSim {axis.get('id')} x", 2)
        ordinates = _finite_scalars(parameters.get("y"), f"OpenSim {axis.get('id')} y", 2)
        slope, quadratic, cubic = _simm_spline_coefficients(abscissae, ordinates)
    direction = _normalize_spatial_axis(
        _vector3(axis.get("axis"), f"OpenSim TransformAxis {axis.get('id')} axis"),
        f"OpenSim TransformAxis {axis.get('id')} axis",
    )
    try:
        header = struct.pack(
            "<4I", kind, coordinate_index, len(coefficients), len(abscissae)
        )
        direction_bytes = struct.pack("<4f", *direction, 0.0)
    except OverflowError as error:
        raise ImportError(f"OpenSim TransformAxis {axis.get('id')} is outside FP32 range") from error
    payload = b"".join(
        (
            header,
            direction_bytes,
            _pack_opensim_float_block(
                coefficients,
                _OPENSIM_SPATIAL_GPU_MAX_COEFFICIENTS,
                f"OpenSim {axis.get('id')} coefficients",
            ),
            _pack_opensim_float_block(
                abscissae,
                _OPENSIM_SPATIAL_GPU_MAX_KNOTS,
                f"OpenSim {axis.get('id')} abscissae",
            ),
            _pack_opensim_float_block(
                ordinates,
                _OPENSIM_SPATIAL_GPU_MAX_KNOTS,
                f"OpenSim {axis.get('id')} ordinates",
            ),
            _pack_opensim_float_block(
                slope,
                _OPENSIM_SPATIAL_GPU_MAX_KNOTS,
                f"OpenSim {axis.get('id')} spline slope",
            ),
            _pack_opensim_float_block(
                quadratic,
                _OPENSIM_SPATIAL_GPU_MAX_KNOTS,
                f"OpenSim {axis.get('id')} spline quadratic",
            ),
            _pack_opensim_float_block(
                cubic,
                _OPENSIM_SPATIAL_GPU_MAX_KNOTS,
                f"OpenSim {axis.get('id')} spline cubic",
            ),
        )
    )
    if len(payload) != 416:
        raise ImportError("internal OpenSim spatial-function GPU ABI size mismatch")
    return payload


def pack_opensim_spatial_transform_gpu(joint: dict[str, Any]) -> tuple[bytes, list[str]]:
    """Compile a parsed CustomJoint into the fixed MetalRobo program ABI.

    This returns a kinematic program only.  It deliberately does not lower a
    CustomJoint into a serial chain, a rigid-body dynamics joint, or a muscle
    actuator.
    """
    if joint.get("kind") != "CustomJoint":
        raise ImportError(f"OpenSim joint {joint.get('id')} is not a CustomJoint")
    coordinates = joint.get("coordinates")
    axes = joint.get("motion_axes")
    if not isinstance(coordinates, list) or not (1 <= len(coordinates) <= 6):
        raise ImportError(f"OpenSim CustomJoint {joint.get('id')} has invalid coordinate count")
    if not isinstance(axes, list) or len(axes) != _OPENSIM_SPATIAL_GPU_AXIS_COUNT:
        raise ImportError(f"OpenSim CustomJoint {joint.get('id')} must retain six SpatialTransform axes")
    coordinate_ids = [coordinate.get("id") for coordinate in coordinates]
    if (
        not all(isinstance(identifier, str) and identifier for identifier in coordinate_ids)
        or len(set(coordinate_ids)) != len(coordinate_ids)
    ):
        raise ImportError(f"OpenSim CustomJoint {joint.get('id')} has invalid coordinate identifiers")
    coordinate_index = {identifier: index for index, identifier in enumerate(coordinate_ids)}
    records: list[bytes] = []
    for axis in axes:
        selected = axis.get("coordinates")
        selected = selected.strip() if isinstance(selected, str) else ""
        if any(character.isspace() for character in selected):
            raise ImportError(
                f"OpenSim TransformAxis {axis.get('id')} has a multi-coordinate function"
            )
        if selected and selected not in coordinate_index:
            raise ImportError(
                f"OpenSim TransformAxis {axis.get('id')} selects unknown coordinate {selected}"
            )
        records.append(
            _opensim_gpu_function_bytes(
                axis,
                coordinate_index[selected]
                if selected
                else _OPENSIM_SPATIAL_GPU_NO_COORDINATE,
            )
        )
    payload = struct.pack(
        "<4I", _OPENSIM_SPATIAL_GPU_ABI_VERSION, len(coordinate_ids), 0, 0
    ) + b"".join(records)
    if len(payload) != _OPENSIM_SPATIAL_GPU_PROGRAM_BYTES:
        raise ImportError("internal OpenSim spatial-transform GPU ABI size mismatch")
    return payload, coordinate_ids


def pack_opensim_spatial_transform_input_gpu(
    coordinate_ids: list[str], coordinate_values: dict[str, float], coordinate_velocities: dict[str, float]
) -> bytes:
    """Pack one six-coordinate state record consumed by the Metal probe ABI."""
    if not (1 <= len(coordinate_ids) <= _OPENSIM_SPATIAL_GPU_MAX_COORDINATES):
        raise ImportError("OpenSim spatial-transform GPU input has invalid coordinate count")
    values = [
        _finite_scalar(coordinate_values[identifier], f"OpenSim coordinate {identifier}")
        for identifier in coordinate_ids
    ]
    velocities = [
        _finite_scalar(coordinate_velocities[identifier], f"OpenSim coordinate velocity {identifier}")
        for identifier in coordinate_ids
    ]
    try:
        payload = struct.pack(
            "<16f",
            *(values + [0.0] * (8 - len(values)) + velocities + [0.0] * (8 - len(velocities))),
        )
    except OverflowError as error:
        raise ImportError("OpenSim spatial-transform GPU input is outside finite FP32 range") from error
    if len(payload) != _OPENSIM_SPATIAL_GPU_INPUT_BYTES:
        raise ImportError("internal OpenSim spatial-transform GPU input ABI size mismatch")
    return payload


def _safe_gpu_artifact_component(value: str, context: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ImportError(f"{context} cannot be represented in a safe artifact path")
    return value


def rajagopal_custom_joint_gpu_artifacts(
    model: dict[str, Any],
) -> tuple[dict[str, Any], dict[Path, bytes]]:
    """Return source-derived Core program/input sidecars and their manifest."""
    joints = [joint for joint in model.get("joints", []) if joint.get("kind") == "CustomJoint"]
    artifacts: dict[Path, bytes] = {}
    programs: list[dict[str, Any]] = []
    for joint in joints:
        identifier = joint.get("id")
        if not isinstance(identifier, str):
            raise ImportError("OpenSim CustomJoint has no identifier")
        safe_joint = _safe_gpu_artifact_component(identifier, "OpenSim CustomJoint identifier")
        program, coordinate_ids = pack_opensim_spatial_transform_gpu(joint)
        program_path = Path("opensim-spatial-programs") / f"{safe_joint}.mrospatial"
        artifacts[program_path] = program
        default_values = {
            coordinate["id"]: _finite_scalar(
                coordinate.get("default_value"),
                f"OpenSim coordinate {coordinate['id']} default value",
            )
            for coordinate in joint["coordinates"]
        }
        input_specs: list[tuple[str, dict[str, float]]] = [("default", {})]
        input_specs.extend(
            (f"velocity-{_safe_gpu_artifact_component(coordinate, 'OpenSim coordinate identifier')}", {coordinate: 1.0})
            for coordinate in coordinate_ids
        )
        inputs: list[dict[str, Any]] = []
        for label, supplied_velocities in input_specs:
            velocities = {coordinate: supplied_velocities.get(coordinate, 0.0) for coordinate in coordinate_ids}
            input_path = Path("opensim-spatial-programs") / f"{safe_joint}.{label}.mrospatialinput"
            content = pack_opensim_spatial_transform_input_gpu(
                coordinate_ids, default_values, velocities
            )
            artifacts[input_path] = content
            inputs.append(
                {
                    "id": label,
                    "file": input_path.as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                    "coordinate_values": default_values,
                    "coordinate_velocities": velocities,
                }
            )
        programs.append(
            {
                "id": identifier,
                "coordinate_ids": coordinate_ids,
                "program_file": program_path.as_posix(),
                "program_sha256": hashlib.sha256(program).hexdigest(),
                "program_bytes": len(program),
                "inputs": inputs,
            }
        )
    return (
        {
            "schema": "numi.human.opensim-spatial-transform-artifacts.v1",
            "source": {
                "id": model.get("source_id"),
                "file": model.get("source_file"),
                "sha256": model.get("source_sha256"),
                "model_id": model.get("model_id"),
            },
            "core_program_abi": {
                "name": "MROpenSimSpatialTransformGPU",
                "version": _OPENSIM_SPATIAL_GPU_ABI_VERSION,
                "bytes": _OPENSIM_SPATIAL_GPU_PROGRAM_BYTES,
                "byte_order": "little-endian IEEE-754 binary32",
                "coordinate_capacity": _OPENSIM_SPATIAL_GPU_MAX_COORDINATES,
            },
            "core_input_abi": {
                "name": "MROpenSimSpatialTransformInputGPU",
                "version": _OPENSIM_SPATIAL_GPU_ABI_VERSION,
                "bytes": _OPENSIM_SPATIAL_GPU_INPUT_BYTES,
                "byte_order": "little-endian IEEE-754 binary32",
            },
            "evidence_boundary": (
                "These immutable source-derived sidecars are accepted by the bounded Core "
                "kinematic evaluator only. They are not an articulated-dynamics, force "
                "projection, contact, or muscle-actuation admission."
            ),
            "program_count": len(programs),
            "programs": programs,
        },
        artifacts,
    )


def rajagopal_millard_muscle_ir(model: dict[str, Any]) -> dict[str, Any]:
    """Validate and retain every source field needed by a Millard lowerer."""
    expected_parameters = {
        "max_isometric_force",
        "optimal_fiber_length",
        "tendon_slack_length",
        "pennation_angle_at_optimal",
        "ignore_tendon_compliance",
        "fiber_damping",
        "default_activation",
        "minimum_activation",
        "TendonForceLengthCurve",
    }
    muscles = [
        muscle
        for muscle in model.get("muscles", [])
        if muscle.get("kind") == "Millard2012EquilibriumMuscle"
    ]
    if len(muscles) != len(model.get("muscles", [])):
        unsupported = sorted(
            {
                str(muscle.get("kind"))
                for muscle in model.get("muscles", [])
                if muscle.get("kind") != "Millard2012EquilibriumMuscle"
            }
        )
        raise ImportError(
            "Rajagopal muscle IR requires only Millard2012EquilibriumMuscle; found "
            + ", ".join(unsupported)
        )
    body_ids = {body.get("id") for body in model.get("bodies", [])}
    wrap_ids = {wrap.get("id") for wrap in model.get("wrap_objects", [])}
    compiled: list[dict[str, Any]] = []
    for muscle in muscles:
        identifier = muscle.get("id")
        parameters = muscle.get("parameters")
        curves = muscle.get("curves")
        points = muscle.get("path_points")
        wraps = muscle.get("path_wraps")
        if not isinstance(identifier, str) or not identifier:
            raise ImportError("Millard muscle has no identifier")
        if not isinstance(parameters, dict) or set(parameters) != expected_parameters:
            raise ImportError(f"Millard muscle {identifier} has an incomplete parameter contract")
        for name in (
            "max_isometric_force",
            "optimal_fiber_length",
            "tendon_slack_length",
            "pennation_angle_at_optimal",
            "fiber_damping",
            "default_activation",
            "minimum_activation",
        ):
            _finite_scalar(parameters[name], f"Millard muscle {identifier} {name}")
        if parameters["ignore_tendon_compliance"] not in ("true", "false", True, False):
            raise ImportError(f"Millard muscle {identifier} has invalid tendon-compliance flag")
        if not isinstance(curves, dict) or set(curves) != {
            "ActiveForceLengthCurve",
            "FiberForceLengthCurve",
            "ForceVelocityCurve",
            "TendonForceLengthCurve",
        }:
            raise ImportError(f"Millard muscle {identifier} has incomplete curve records")
        if not isinstance(points, list) or len(points) < 2:
            raise ImportError(f"Millard muscle {identifier} requires two or more path points")
        if not isinstance(wraps, list):
            raise ImportError(f"Millard muscle {identifier} has invalid path wraps")
        for point in points:
            frame = point.get("parent_frame")
            if not isinstance(frame, str) or not frame.startswith("/bodyset/"):
                raise ImportError(f"Millard muscle {identifier} has unresolved path-point frame")
            body = frame.rsplit("/", 1)[-1]
            if body not in body_ids:
                raise ImportError(f"Millard muscle {identifier} references unknown body {body}")
            _vector3(point.get("location_m"), f"Millard muscle {identifier} path point")
        for wrap in wraps:
            wrap_id = wrap.get("wrap_object")
            if not isinstance(wrap_id, str) or wrap_id not in wrap_ids:
                raise ImportError(f"Millard muscle {identifier} references unknown wrap object {wrap_id}")
        compiled.append(
            {
                "id": identifier,
                "parameters": parameters,
                "curves": curves,
                "path_points": points,
                "path_wraps": wraps,
                "source_xml": muscle.get("source_xml"),
            }
        )
    return {
        "schema": "numi.human.opensim-millard-muscle-ir.v1",
        "source": {
            "id": model.get("source_id"),
            "file": model.get("source_file"),
            "sha256": model.get("source_sha256"),
            "model_id": model.get("model_id"),
        },
        "muscle_count": len(compiled),
        "path_point_count": sum(len(muscle["path_points"]) for muscle in compiled),
        "path_wrap_count": sum(len(muscle["path_wraps"]) for muscle in compiled),
        "wrap_objects": model.get("wrap_objects", []),
        "muscles": compiled,
        "runtime_requirement": (
            "The qualified bounded Core path evaluates source static fiber-tendon equilibrium, "
            "GeometryPath wrapping, body-frame moment-arm force scatter, and explicitly "
            "parameterized per-control activation on device. Dynamic fibre/tendon state "
            "advancement and held-out validation remain separate requirements."
        ),
        "evidence_boundary": (
            "Exact OpenSim muscle, curve, GeometryPath, and wrap source records only. "
            "This IR alone does not evaluate or apply a Hill-type force; that occurs in "
            "the qualified owner Core path."
        ),
    }


def mortensen_neck_source_ir(model: dict[str, Any]) -> dict[str, Any]:
    """Retain the selected cervical/hyoid model without inventing a body bridge.

    Mortensen 2018 is an OpenSim 3 model: its body-owned joints and path-point
    body references differ from the OpenSim 4 sockets used by Rajagopal.  This
    source IR deliberately preserves that representation.  It is the input to
    a later rest-pose registration against the active MyoSim thorax/neck, not
    a claim that two independently scaled skeletons can be joined by name.
    """
    if model.get("source_id") != "mortensen_2018_neck":
        raise ImportError("Mortensen neck IR requires the selected mortensen_2018_neck source")
    if model.get("model_id") != "HYOID_Scaled":
        raise ImportError("Mortensen neck model identity drifted from HYOID_Scaled")
    if model.get("opensim_document_version") != "30000":
        raise ImportError("Mortensen neck IR requires the OpenSim 3 source layout")

    expected_bodies = {
        "ground", "spine", "ribcage", "rscapula", "rclavicle", "lscapula", "lclavicle",
        "cerv7", "cerv6", "cerv5", "cerv4", "cerv3", "cerv2", "cerv1", "skull", "jaw",
    }
    bodies = model.get("bodies")
    joints = model.get("joints")
    muscles = model.get("muscles")
    if not isinstance(bodies, list) or {body.get("id") for body in bodies} != expected_bodies:
        raise ImportError("Mortensen neck body set drifted from the selected source")
    if not isinstance(joints, list) or len(joints) != 15 or not all(
        joint.get("legacy_opensim3") is True for joint in joints
    ):
        raise ImportError("Mortensen neck requires its 15 body-owned OpenSim 3 joints")
    if not isinstance(muscles, list) or len(muscles) != 72 or not all(
        muscle.get("kind") == "Millard2012EquilibriumMuscle" for muscle in muscles
    ):
        raise ImportError("Mortensen neck requires the complete 72-muscle Millard set")

    body_ids = {body["id"] for body in bodies}
    compiled_muscles: list[dict[str, Any]] = []
    for muscle in muscles:
        identifier = muscle.get("id")
        parameters = muscle.get("parameters")
        curves = muscle.get("curves")
        points = muscle.get("path_points")
        wraps = muscle.get("path_wraps")
        if not isinstance(identifier, str) or not identifier:
            raise ImportError("Mortensen muscle has no identifier")
        if not isinstance(parameters, dict) or not isinstance(curves, dict):
            raise ImportError(f"Mortensen muscle {identifier} has incomplete source properties")
        required_parameters = {
            "max_isometric_force", "optimal_fiber_length", "tendon_slack_length",
            "pennation_angle_at_optimal", "minimum_activation",
        }
        if not required_parameters.issubset(parameters):
            raise ImportError(f"Mortensen muscle {identifier} is missing a Hill-type source parameter")
        if set(curves) != {
            "ActiveForceLengthCurve", "FiberForceLengthCurve", "ForceVelocityCurve", "TendonForceLengthCurve",
        }:
            raise ImportError(f"Mortensen muscle {identifier} has an incomplete Millard curve set")
        if not isinstance(points, list) or len(points) < 2 or not isinstance(wraps, list):
            raise ImportError(f"Mortensen muscle {identifier} has an invalid GeometryPath")
        for point in points:
            frame = point.get("parent_frame")
            if not isinstance(frame, str) or frame not in body_ids:
                raise ImportError(f"Mortensen muscle {identifier} has a path point outside its body set")
            _vector3(point.get("location_m"), f"Mortensen muscle {identifier} path point")
        compiled_muscles.append(
            {
                "id": identifier,
                "parameters": parameters,
                "curves": curves,
                "path_points": points,
                "path_wraps": wraps,
                "source_xml": muscle.get("source_xml"),
            }
        )

    cervical_order = ["cerv7", "cerv6", "cerv5", "cerv4", "cerv3", "cerv2", "cerv1", "skull"]
    cervical_joints = [
        joint for joint in joints
        if joint.get("parent_frame") in set(cervical_order) | {"spine"}
        and joint.get("child_frame") in set(cervical_order)
    ]
    if len(cervical_joints) != 8:
        raise ImportError("Mortensen neck must retain its eight serial cervical/skull joints")
    return {
        "schema": "numi.human.mortensen-neck-source-ir.v1",
        "source": {
            "id": model.get("source_id"),
            "file": model.get("source_file"),
            "sha256": model.get("source_sha256"),
            "model_id": model.get("model_id"),
            "opensim_document_version": model.get("opensim_document_version"),
        },
        "model": {
            "body_count": len(bodies), "joint_count": len(joints), "muscle_count": len(compiled_muscles),
            "cervical_body_order": cervical_order,
            "cervical_joint_count": len(cervical_joints),
            "hyoid_and_jaw_support": True,
            "explicit_ignore_tendon_compliance_count": sum(
                1 for muscle in compiled_muscles
                if "ignore_tendon_compliance" in muscle["parameters"]
            ),
        },
        "bodies": bodies,
        "joints": joints,
        "muscles": compiled_muscles,
        "integration_contract": {
            "active_body": "MyoSim myofullbody",
            "source_root": "spine",
            "candidate_active_attachment": "cervical_spine",
            "required_before_force_application": [
                "source-to-source rest-pose registration from Mortensen spine to MyoSim cervical_spine",
                "explicit replacement or merge decision for MyoSim neck/head bodies",
                "mapped path-point and wrap geometry frames validated at the registered pose",
                "native Millard equilibrium and force-scatter oracle for the merged model",
            ],
        },
        "evidence_boundary": (
            "Complete selected Mortensen source records only. This artifact does not attach its "
            "separately scaled spine, skull, or 72 muscles to MyoSim, and therefore does not yet "
            "apply cervical muscle force in the active full-body Core runtime."
        ),
    }


def rajagopal_walking_contract(model: dict[str, Any]) -> dict[str, Any]:
    """Emit the source-backed contract required before a learned walk rollout.

    This is intentionally a *contract*, not a RobotPack or a gait claim.  It
    makes the mobile source root and the 80-dimensional excitation surface
    inspectable while refusing to manufacture the BodyParts3D registrations,
    foot colliders, or calibrated contact constants that the sources do not
    contain.
    """
    skeleton = rajagopal_rigid_skeleton_ir(model)
    millard = rajagopal_millard_muscle_ir(model)
    root = next((joint for joint in skeleton["joints"] if joint["id"] == "ground_pelvis"), None)
    if not isinstance(root, dict) or root.get("kind") != "CustomJoint":
        raise ImportError("walking contract requires the Rajagopal ground_pelvis CustomJoint")
    root_coordinates = root.get("coordinates")
    if not isinstance(root_coordinates, list) or len(root_coordinates) != 6:
        raise ImportError("walking contract requires six mobile ground_pelvis coordinates")
    root_ids = [coordinate.get("id") for coordinate in root_coordinates]
    expected_root_ids = [
        "pelvis_tilt", "pelvis_list", "pelvis_rotation",
        "pelvis_tx", "pelvis_ty", "pelvis_tz",
    ]
    if root_ids != expected_root_ids:
        raise ImportError("walking contract ground_pelvis coordinate order drifted from Rajagopal source")
    muscle_ids = [muscle["id"] for muscle in millard["muscles"]]
    if len(muscle_ids) != 80 or len(set(muscle_ids)) != len(muscle_ids):
        raise ImportError("walking contract requires the complete unique 80-muscle Rajagopal set")
    coordinates = [
        coordinate["id"]
        for joint in skeleton["joints"]
        for coordinate in joint.get("coordinates", [])
        if isinstance(coordinate.get("id"), str)
    ]
    return {
        "schema": "numi.human.rajagopal-walking-contract.v1",
        "source": skeleton["source"],
        "articulation": {
            "root_joint": "ground_pelvis",
            "root_mode": "source_function_based_mobile",
            "coordinates": root_ids,
            "coordinate_count": len(coordinates),
            "qualified_core": [
                "source-default-preserving 7-q/6-v pelvis mobile-root reduction",
                "reusable Core mobile-root reducer with canonical source-index maps",
                "FunctionBased dense free motion and streamed contact response",
                "complete ordered Millard native-task excitation surface",
            ],
            "requires_core": [
                "walking-task scenario admission after anatomical collider registration",
            ],
        },
        "policy": {
            "action_kind": "bounded_muscle_excitation",
            "action_count": len(muscle_ids),
            "action_order": muscle_ids,
            "observation": {
                "coordinates": coordinates,
                "velocity_coordinates": coordinates,
                "muscle_activation_order": muscle_ids,
                "contact_features": "requires registered foot colliders",
            },
            "state": {
                "persistent_activation": "source default/minimum activation bounded by excitation",
                "fiber_tendon": "equilibrium warm-start state; no OpenSim-equivalence claim",
                "activation_dynamics": {
                    "model": "first_order_excitation_to_activation",
                    "parameters": _MILLARD_ACTIVATION_DEFAULTS,
                    "provenance": "OpenSim Millard2012EquilibriumMuscle documented class defaults; absent from the pinned Rajagopal XML",
                },
            },
        },
        "contact": {
            "scenario": "flat_ground",
            "foot_bodies": ["calcn_r", "toes_r", "calcn_l", "toes_l"],
            "status": "blocked_by_bodyparts_registration_and_contact_calibration",
            "required_artifacts": [
                "per-foot source-to-body registration",
                "conservative collider proxy manifest",
                "friction/compliance parameter manifest",
                "collision exclusions and deterministic replay scenario",
            ],
        },
        "visual_layers": {
            "requested": ["skin", "bones", "muscles", "vessels", "nerves"],
            "status": "blocked_by_per-mesh_body_attachment_validation",
            "rule": "unregistered BodyParts3D geometry must remain static or hidden",
        },
        "evidence_boundary": (
            "This source-derived policy and mobile-root contract does not provide a trained policy, "
            "registered collider, contact calibration, gait validation, or deformable anatomy."
        ),
    }


def rajagopal_lower_body_pilot(model: dict[str, Any]) -> dict[str, Any]:
    """Build the small, runnable lower-body contact scaffold.

    This is deliberately an engineering prototype: it uses the real Rajagopal
    mobile pelvis and 80 muscle controls, but gives the four foot bodies simple
    pads so that standing and walking work can start before anatomical meshes
    are attached.  The pads are not BodyParts3D geometry and must never be
    presented as anatomical contact surfaces.
    """
    walking = rajagopal_walking_contract(model)
    core_manifest, _ = rajagopal_core_reference_artifact(model)
    body_order = core_manifest.get("body_order")
    if not isinstance(body_order, list) or body_order[:1] != ["__ground__"]:
        raise ImportError("lower-body pilot requires canonical Rajagopal Core body order")
    body_indices = {body: index - 1 for index, body in enumerate(body_order) if index > 0}
    foot_bodies = ["calcn_r", "toes_r", "calcn_l", "toes_l"]
    if any(body not in body_indices for body in foot_bodies):
        raise ImportError("lower-body pilot requires all four Rajagopal foot bodies")
    return {
        "schema": "numi.human.lower-body-pilot.v1",
        "source": walking["source"],
        "intent": "muscle-driven lower-body standing and flat-ground walking prototype",
        "articulation": walking["articulation"],
        "policy": {
            "action_kind": "bounded_muscle_excitation",
            "action_count": walking["policy"]["action_count"],
            "action_order": walking["policy"]["action_order"],
        },
        "contact": {
            "terrain": "flat_ground",
            "mode": "temporary_engineering_pads",
            "shape": "box",
            "full_dimensions_m": [0.06, 0.03, 0.06],
            "local_position_m": [0.0, 0.0, 0.0],
            "local_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "collision_scope": "pads_to_ground_only",
            "pads": [
                {"body": body, "mobile_body_index": body_indices[body]}
                for body in foot_bodies
            ],
        },
        "curriculum": [
            "deterministic contact",
            "standing",
            "forward commanded walking",
        ],
        "visual": {
            "layers": ["skin", "bones", "muscles", "organs", "vessels", "nerves"],
            "current_mode": "source_static_reference_layers",
            "next_step": "attach visual meshes to lower-body segments for runtime inspection",
        },
        "boundary": (
            "The foot pads are temporary non-anatomical engineering scaffolding. "
            "They enable native task/contact work but do not register, calibrate, or replace "
            "BodyParts3D foot geometry."
        ),
    }


def bodyparts_lower_body_attachment_worklist(
    anatomy: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Produce reviewable BodyParts3D-to-Rajagopal candidates, never bindings.

    Names are only useful for triage.  A candidate still needs a rest-frame
    transform and visual/collision review before it may move with a body.
    """
    body_ids = {body.get("id") for body in model.get("bodies", [])}
    target_terms = {
        "pelvis": "pelvis", "femur": "femur", "tibia": "tibia", "fibula": "fibula",
        "patella": "patella", "talus": "talus", "calcaneus": "calcn", "toe": "toes",
    }

    candidates: list[dict[str, Any]] = []
    layer_counts: Counter[str] = Counter()
    for component in anatomy.get("components", []):
        name = str(component.get("name", "")).lower()
        target = next((body for term, body in target_terms.items() if term in name), None)
        if target is None:
            continue
        for element in component.get("element_meshes", []):
            if not element.get("mesh_present"):
                continue
            layer = str(component.get("anatomy_class", "unclassified_surface"))
            candidates.append({
                "element_id": element["element_id"], "concept_id": component["concept_id"],
                "name": component["name"], "layer": layer, "candidate_body": target,
                "status": "candidate_requires_rest_frame_registration",
            })
            layer_counts[layer] += 1
    foot = [entry for entry in candidates if entry["candidate_body"] in {"calcn", "toes"}]
    if not foot:
        raise ImportError("BodyParts3D attachment worklist found no foot candidate surfaces")
    return {
        "schema": "numi.human.bodyparts-lower-body-attachment-worklist.v1",
        "source": {"id": anatomy.get("source_id"), "version": anatomy.get("version"), "archives": anatomy.get("archives")},
        "candidate_count": len(candidates), "layer_counts": dict(sorted(layer_counts.items())),
        "candidates": candidates,
        "foot_collider_work": {
            "source_bodies": ["calcn_r", "toes_r", "calcn_l", "toes_l"],
            "bodyparts_candidate_count": len(foot),
            "required": ["validated source-to-body transforms", "conservative proxy geometry", "pair exclusions", "contact parameter receipt"],
            "status": "blocked_by_registration_and_calibration",
        },
        "evidence_boundary": "String/name correspondence proposes review only; this file contains no transform, skinning weight, collider, or physical material parameter.",
    }


def bodyparts_foot_registration_template(
    anatomy: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Emit a fail-closed hand-off for the four walking-contact foot bodies.

    BodyParts3D labels can help a reviewer find a mesh, but do not establish
    coordinate frames, handedness, scale, or a collision proxy.  This template
    deliberately carries no matrix or physical constant; a later reviewed
    registration receipt must supply those values for each source body.
    """
    foot_targets = (
        ("calcn_r", "right", "calcaneus", ("calcaneus",)),
        ("toes_r", "right", "toes", ("toe",)),
        ("calcn_l", "left", "calcaneus", ("calcaneus",)),
        ("toes_l", "left", "toes", ("toe",)),
    )
    known_bodies = {body.get("id") for body in model.get("bodies", [])}
    missing = [body_id for body_id, _, _, _ in foot_targets if body_id not in known_bodies]
    if missing:
        raise ImportError(
            "foot registration template requires Rajagopal bodies: " + ", ".join(missing)
        )

    def laterality(name: str) -> str | None:
        lowered = name.lower()
        has_right = bool(re.search(r"\bright\b", lowered))
        has_left = bool(re.search(r"\bleft\b", lowered))
        if has_right == has_left:
            return None
        return "right" if has_right else "left"

    archive_by_hierarchy = {
        archive.get("hierarchy"): {
            "file": archive.get("file"), "sha256": archive.get("sha256"),
        }
        for archive in anatomy.get("archives", [])
    }
    registrations: list[dict[str, Any]] = []
    for body_id, side, landmark, terms in foot_targets:
        candidates: list[dict[str, Any]] = []
        for component in anatomy.get("components", []):
            component_name = str(component.get("name", ""))
            if not any(term in component_name.lower() for term in terms) or laterality(component_name) != side:
                continue
            hierarchy = component.get("hierarchy")
            for element in component.get("element_meshes", []):
                if not element.get("mesh_present"):
                    continue
                candidates.append({
                    "element_id": element["element_id"],
                    "concept_id": component["concept_id"],
                    "name": component["name"],
                    "anatomy_class": component.get("anatomy_class"),
                    "hierarchy": hierarchy,
                    "archive": archive_by_hierarchy.get(hierarchy),
                    "status": "candidate_requires_human_review",
                })
        registrations.append({
            "opensim_body": body_id,
            "laterality": side,
            "anatomical_landmark": landmark,
            "bodyparts_candidates": candidates,
            "bodyparts_candidate_count": len(candidates),
            "registration": {
                "status": "requires_explicit_reviewed_transform",
                "source_frame": "BodyParts3D OBJ coordinates (axis convention must be verified)",
                "target_frame": f"Rajagopal OpenSim body frame: {body_id}",
                "required_receipt_fields": [
                    "reviewed source member IDs and hashes",
                    "axis and unit conversion",
                    "4x4 source-to-body rest-frame transform",
                    "landmark/residual visual review from multiple angles",
                ],
            },
            "collision_proxy": {
                "status": "requires_separate_reviewed_proxy",
                "required_receipt_fields": [
                    "conservative proxy geometry",
                    "ground/self pair exclusions",
                    "friction and compliance calibration receipt",
                ],
            },
        })
    return {
        "schema": "numi.human.bodyparts-foot-registration-template.v1",
        "source": {
            "bodyparts": {
                "id": anatomy.get("source_id"),
                "version": anatomy.get("version"),
                "archives": anatomy.get("archives"),
            },
            "opensim": {
                "id": model.get("source_id"),
                "file": model.get("source_file"),
                "sha256": model.get("source_sha256"),
            },
        },
        "walking_contact_bodies": [body_id for body_id, _, _, _ in foot_targets],
        "registrations": registrations,
        "status": "blocked_by_explicit_transform_and_contact_calibration_receipts",
        "evidence_boundary": (
            "This is a provenance-pinned review template, not a registration manifest. "
            "It contains no source-to-body transform, collision geometry, pair exclusion, "
            "friction, compliance, or walking claim."
        ),
    }


def bodyparts_foot_collider_preflight(
    sources: Path, anatomy: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Derive exact source-local enclosing-box candidates for foot review.

    The boxes enclose the original OBJ triangles in BodyParts3D coordinates.
    They are not colliders until a reviewer supplies a source-to-Rajagopal
    rest-frame transform, chooses a contact representation, and calibrates its
    material response.
    """
    template = bodyparts_foot_registration_template(anatomy, model)
    per_foot: list[dict[str, Any]] = []
    total_meshes = 0
    for registration in template["registrations"]:
        unique_candidates: dict[tuple[str, str], dict[str, Any]] = {}
        for candidate in registration["bodyparts_candidates"]:
            hierarchy = candidate.get("hierarchy")
            element_id = candidate.get("element_id")
            if not isinstance(hierarchy, str) or not isinstance(element_id, str):
                raise ImportError("foot registration candidate lacks a source hierarchy or OBJ member")
            unique_candidates.setdefault((hierarchy, element_id), candidate)
        meshes: list[dict[str, Any]] = []
        for (hierarchy, element_id), candidate in sorted(unique_candidates.items()):
            archive_path, member, obj = _bodyparts_obj_member(sources, hierarchy, element_id)
            vertices, triangles = _bodyparts_obj_triangles(obj, member)
            lower = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
            upper = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
            center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
            half_extent = [(upper[axis] - lower[axis]) * 0.5 for axis in range(3)]
            meshes.append({
                "source": {
                    "archive": archive_path.name,
                    "archive_sha256": sha256(archive_path),
                    "member": member,
                    "member_id": element_id,
                    "member_sha256": hashlib.sha256(obj).hexdigest(),
                    "hierarchy": hierarchy,
                    "concept_id": candidate["concept_id"],
                    "name": candidate["name"],
                },
                "geometry": {
                    "vertex_count": len(vertices),
                    "triangle_count": len(triangles),
                    "bounds_mm": {"minimum": lower, "maximum": upper},
                },
                "source_local_proxy_candidate": {
                    "shape": "axis_aligned_box",
                    "center_mm": center,
                    "half_extents_mm": half_extent,
                    "enclosure": "each source triangle is enclosed by the source-coordinate AABB",
                    "status": "source_local_only_requires_registered_transform",
                },
            })
        total_meshes += len(meshes)
        per_foot.append({
            "opensim_body": registration["opensim_body"],
            "laterality": registration["laterality"],
            "anatomical_landmark": registration["anatomical_landmark"],
            "source_mesh_count": len(meshes),
            "source_meshes": meshes,
            "admission": {
                "status": "blocked_by_registered_transform_and_contact_calibration",
                "required_receipt_fields": [
                    "selected source members or a reviewed merged proxy",
                    "BodyParts3D-to-Rajagopal rest-frame transform",
                    "ground/self pair exclusions",
                    "friction, compliance, and restitution calibration receipt",
                    "multi-angle transformed-mesh and proxy residual review",
                ],
            },
        })
    return {
        "schema": "numi.human.bodyparts-foot-collider-preflight.v1",
        "source": template["source"],
        "walking_contact_bodies": template["walking_contact_bodies"],
        "source_coordinate_units": "BodyParts3D OBJ millimetres",
        "per_foot": per_foot,
        "source_mesh_count": total_meshes,
        "status": "source_local_proxy_candidates_not_admitted",
        "evidence_boundary": (
            "The emitted boxes exactly enclose source mesh triangles in BodyParts3D coordinates. "
            "They are not OpenSim-frame colliders, collision-pair settings, contact parameters, "
            "or walking evidence."
        ),
    }


def bodyparts_foot_registration_receipt_template(
    sources: Path, anatomy: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Compose one provenance-pinned, reviewer-completed foot hand-off.

    This is intentionally a blank receipt, not a registration or collider
    manifest.  It joins the source identities and source-local enclosure
    candidates that a reviewer must inspect before supplying transforms,
    visual evidence, and calibrated contact values.
    """
    preflight = bodyparts_foot_collider_preflight(sources, anatomy, model)
    canonical_preflight = json.dumps(
        preflight, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    receipts: list[dict[str, Any]] = []
    for foot in preflight["per_foot"]:
        meshes = foot["source_meshes"]
        receipts.append({
            "opensim_body": foot["opensim_body"],
            "laterality": foot["laterality"],
            "anatomical_landmark": foot["anatomical_landmark"],
            "source_meshes": [
                {
                    "source": mesh["source"],
                    "source_local_proxy_candidate": mesh[
                        "source_local_proxy_candidate"
                    ],
                }
                for mesh in meshes
            ],
            "reviewed_registration": {
                "status": "requires_reviewer_completion",
                "axis_and_unit_conversion": {
                    "status": "required",
                    "source_units": "BodyParts3D OBJ millimetres",
                    "target_frame": (
                        "Rajagopal OpenSim body frame: " + foot["opensim_body"]
                    ),
                },
                "source_to_body_rest_transform": {
                    "status": "required",
                    "format": "4x4 rigid affine matrix after reviewed unit conversion",
                },
                "multi_angle_visual_review": {
                    "status": "required",
                    "minimum_distinct_views": 3,
                    "required_fields": [
                        "camera/view identifiers",
                        "render artifact hashes",
                        "landmark and proxy residual measurements",
                        "reviewer identity and date",
                    ],
                },
            },
            "reviewed_contact": {
                "status": "requires_reviewer_completion",
                "required_fields": [
                    "conservative OpenSim-frame proxy geometry",
                    "ground and self-collision exclusions",
                    "friction, compliance, and restitution calibration receipt",
                ],
            },
            "admission": "blocked_by_reviewer_completed_registration_and_contact_receipts",
        })
    return {
        "schema": "numi.human.bodyparts-foot-registration-receipt-template.v1",
        "source": preflight["source"],
        "preflight_sha256": hashlib.sha256(canonical_preflight).hexdigest(),
        "walking_contact_bodies": preflight["walking_contact_bodies"],
        "receipts": receipts,
        "status": "not_a_registration_or_collider_manifest",
        "evidence_boundary": (
            "This template carries exact source mesh identities and local enclosure "
            "candidates only. It contains no reviewed transform, visual evidence, "
            "contact parameter, collider binding, task admission, or walking claim."
        ),
    }


def validate_bodyparts_foot_registration_receipt(
    receipt: dict[str, Any], sources: Path, anatomy: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed on a reviewer-completed registration/contact receipt.

    Structural completeness and provenance are necessary before a Core
    lowerer can consume a foot manifest. They are not evidence that the
    registration, collider, or calibration is physically valid.
    """
    expected = bodyparts_foot_registration_receipt_template(sources, anatomy, model)
    if receipt.get("schema") != expected["schema"]:
        raise ImportError("foot receipt has an unsupported schema")
    if receipt.get("source") != expected["source"]:
        raise ImportError("foot receipt source provenance does not match the pinned sources")
    if receipt.get("preflight_sha256") != expected["preflight_sha256"]:
        raise ImportError("foot receipt does not bind to the current source-local preflight")
    if receipt.get("walking_contact_bodies") != expected["walking_contact_bodies"]:
        raise ImportError("foot receipt walking-contact body order drifted")
    actual_receipts = receipt.get("receipts")
    if not isinstance(actual_receipts, list) or len(actual_receipts) != len(expected["receipts"]):
        raise ImportError("foot receipt must contain exactly one entry for every walking-contact body")

    def finite_number(value: Any, field: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ImportError(f"foot receipt {field} must be a finite number")
        return float(value)

    def text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ImportError(f"foot receipt {field} must be a nonempty string")
        return value

    def sha(value: Any, field: str) -> str:
        candidate = text(value, field)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", candidate):
            raise ImportError(f"foot receipt {field} must be a SHA-256 digest")
        return candidate.lower()

    def rigid_transform(value: Any, body: str) -> None:
        if not isinstance(value, dict):
            raise ImportError(f"foot receipt {body} rest transform must be an object")
        matrix = value.get("matrix")
        if not isinstance(matrix, list) or len(matrix) != 4 or any(
            not isinstance(row, list) or len(row) != 4 for row in matrix
        ):
            raise ImportError(f"foot receipt {body} rest transform must be a 4x4 matrix")
        m = [[finite_number(cell, f"{body} rest transform") for cell in row] for row in matrix]
        tolerance = 1.0e-5
        if any(abs(m[3][column] - (1.0 if column == 3 else 0.0)) > tolerance
               for column in range(4)):
            raise ImportError(f"foot receipt {body} rest transform is not affine")
        rotation = [[m[row][column] for column in range(3)] for row in range(3)]
        for row in rotation:
            if abs(sum(value * value for value in row) - 1.0) > tolerance:
                raise ImportError(f"foot receipt {body} rest transform rotation is not orthonormal")
        for first in range(3):
            for second in range(first + 1, 3):
                if abs(sum(rotation[first][axis] * rotation[second][axis] for axis in range(3))) > tolerance:
                    raise ImportError(f"foot receipt {body} rest transform rotation is not orthogonal")
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if abs(determinant - 1.0) > tolerance:
            raise ImportError(f"foot receipt {body} rest transform must have a proper rotation")

    verified_bodies: list[str] = []
    for expected_entry, entry in zip(expected["receipts"], actual_receipts, strict=True):
        body = expected_entry["opensim_body"]
        if not isinstance(entry, dict) or entry.get("opensim_body") != body or \
                entry.get("laterality") != expected_entry["laterality"] or \
                entry.get("anatomical_landmark") != expected_entry["anatomical_landmark"]:
            raise ImportError("foot receipt body identities do not match the reviewed template")
        if not expected_entry["source_meshes"]:
            raise ImportError(f"foot receipt {body} has no BodyParts3D source mesh to register")
        if entry.get("source_meshes") != expected_entry["source_meshes"]:
            raise ImportError(f"foot receipt {body} source mesh identities changed")
        registration = entry.get("reviewed_registration")
        if not isinstance(registration, dict):
            raise ImportError(f"foot receipt {body} is missing reviewed registration")
        conversion = registration.get("axis_and_unit_conversion")
        if not isinstance(conversion, dict):
            raise ImportError(f"foot receipt {body} is missing axis/unit conversion")
        if conversion.get("axis_permutation") not in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
            raise ImportError(f"foot receipt {body} must provide an axis permutation")
        signs = conversion.get("axis_signs")
        if not isinstance(signs, list) or len(signs) != 3 or any(sign not in (-1, 1) for sign in signs):
            raise ImportError(f"foot receipt {body} must provide three axis signs")
        if finite_number(conversion.get("scale_m_per_source_unit"), f"{body} unit scale") <= 0.0:
            raise ImportError(f"foot receipt {body} unit scale must be positive")
        text(conversion.get("reviewer"), f"{body} conversion reviewer")
        rigid_transform(registration.get("source_to_body_rest_transform"), body)
        visual = registration.get("multi_angle_visual_review")
        if not isinstance(visual, dict):
            raise ImportError(f"foot receipt {body} is missing multi-angle visual review")
        views = visual.get("views")
        if not isinstance(views, list) or len(views) < 3:
            raise ImportError(f"foot receipt {body} requires at least three visual-review views")
        view_ids: set[str] = set()
        for index, view in enumerate(views):
            if not isinstance(view, dict):
                raise ImportError(f"foot receipt {body} visual view {index} must be an object")
            view_id = text(view.get("id"), f"{body} visual view id")
            if view_id in view_ids:
                raise ImportError(f"foot receipt {body} visual view identifiers must be unique")
            view_ids.add(view_id)
            sha(view.get("artifact_sha256"), f"{body} visual artifact")
            if finite_number(view.get("maximum_landmark_residual_mm"), f"{body} visual residual") < 0.0:
                raise ImportError(f"foot receipt {body} visual residual cannot be negative")
        text(visual.get("reviewer"), f"{body} visual reviewer")
        contact = entry.get("reviewed_contact")
        if not isinstance(contact, dict):
            raise ImportError(f"foot receipt {body} is missing reviewed contact")
        proxy = contact.get("proxy_geometry")
        if not isinstance(proxy, dict) or proxy.get("body_frame") != body:
            raise ImportError(f"foot receipt {body} proxy must be authored in its OpenSim body frame")
        if proxy.get("shape") not in {"box", "convex_hull", "capsule"}:
            raise ImportError(f"foot receipt {body} proxy shape is not an admitted conservative primitive")
        if not isinstance(proxy.get("parameters"), dict) or not proxy["parameters"]:
            raise ImportError(f"foot receipt {body} proxy requires parameters")
        exclusions = contact.get("collision_exclusions")
        if not isinstance(exclusions, list):
            raise ImportError(f"foot receipt {body} collision exclusions must be an array")
        calibration = contact.get("calibration")
        if not isinstance(calibration, dict):
            raise ImportError(f"foot receipt {body} is missing contact calibration")
        if finite_number(calibration.get("friction"), f"{body} friction") <= 0.0 or \
                finite_number(calibration.get("normal_stiffness"), f"{body} normal stiffness") <= 0.0 or \
                finite_number(calibration.get("normal_damping"), f"{body} normal damping") < 0.0:
            raise ImportError(f"foot receipt {body} contact calibration must be physically signed")
        restitution = finite_number(calibration.get("restitution"), f"{body} restitution")
        if restitution < 0.0 or restitution > 1.0:
            raise ImportError(f"foot receipt {body} restitution must be in [0, 1]")
        sha(calibration.get("evidence_sha256"), f"{body} contact calibration evidence")
        text(calibration.get("reviewer"), f"{body} contact reviewer")
        verified_bodies.append(body)
    return {
        "schema": "numi.human.bodyparts-foot-registration-receipt-validation.v1",
        "source": expected["source"],
        "preflight_sha256": expected["preflight_sha256"],
        "reviewed_foot_bodies": verified_bodies,
        "receipt_sha256": hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "status": "structurally_complete_not_physics_or_walking_qualified",
        "evidence_boundary": (
            "This validator checks source provenance and reviewer-supplied structural "
            "fields. It does not establish transform accuracy, collider conservatism, "
            "contact calibration validity, runtime admission, or walking performance."
        ),
    }


def bodyparts_visual_layer_previews(sources: Path, output: Path, anatomy: dict[str, Any]) -> dict[str, Any]:
    """Export one exact, reviewable source mesh for each requested anatomy layer."""
    requested = ("skin_surface", "bone", "muscle_surface", "vessel_surface", "nerve_surface")
    selected: dict[str, tuple[str, str]] = {"skin_surface": ("FJ2810", "is_a")}
    archive_sizes: dict[tuple[str, str], int] = {}
    for archive_kind, filename in (("is_a", "isa_BP3D_4.0_obj_99.zip"), ("part_of", "partof_BP3D_4.0_obj_99.zip")):
        with zipfile.ZipFile(sources / filename) as archive:
            for info in archive.infolist():
                match = re.search(r"/(FJ\d+)\.obj$", info.filename)
                if match:
                    archive_sizes[(archive_kind, match.group(1))] = info.file_size
    for layer in requested[1:]:
        best: tuple[int, str, str] | None = None
        for component in anatomy.get("components", []):
            if component.get("anatomy_class") != layer:
                continue
            for element in component.get("element_meshes", []):
                member = element.get("element_id")
                hierarchy = element.get("hierarchy")
                if isinstance(member, str) and isinstance(hierarchy, str):
                    candidate = (archive_sizes.get((hierarchy, member), 0), member, hierarchy)
                    if best is None or candidate > best:
                        best = candidate
        if best is not None:
            selected[layer] = (best[1], best[2])
    missing = [layer for layer in requested if layer not in selected]
    if missing:
        raise ImportError("BodyParts3D is missing preview meshes for: " + ", ".join(missing))
    layers = {layer: bodyparts_visual_preview(sources, output / layer, member_id=member, archive_kind=archive_kind)
              for layer, (member, archive_kind) in selected.items()}
    return {"schema": "numi.human.bodyparts-visual-layers.v1", "layers": layers,
            "evidence_boundary": "Exact source-static mesh previews only; no registration, skinning, collision, contact, or tissue mechanics."}


def _opensim_joint_frame_body(
    joint: dict[str, Any], frame_reference: Any, context: str
) -> str | None:
    """Resolve an OpenSim joint socket through its local frame chain to a body."""
    if not isinstance(frame_reference, str) or not frame_reference:
        raise ImportError(f"OpenSim joint {joint.get('id')} has unresolved {context} frame")
    frames = {
        frame.get("id"): frame
        for frame in joint.get("frames", [])
        if isinstance(frame, dict) and isinstance(frame.get("id"), str)
    }
    current = frame_reference
    seen: set[str] = set()
    while True:
        if current in {"ground", "/ground"}:
            return None
        if current.startswith("/bodyset/"):
            body = current.rsplit("/", 1)[-1]
            if not body:
                raise ImportError(f"OpenSim joint {joint.get('id')} has invalid {context} body frame")
            return body
        local = current.rsplit("/", 1)[-1]
        if local in seen:
            raise ImportError(f"OpenSim joint {joint.get('id')} has cyclic {context} frame chain")
        seen.add(local)
        frame = frames.get(local)
        if frame is None:
            raise ImportError(
                f"OpenSim joint {joint.get('id')} cannot resolve {context} frame {frame_reference}"
            )
        parent = frame.get("parent_frame")
        if not isinstance(parent, str) or not parent:
            raise ImportError(f"OpenSim joint {joint.get('id')} has unresolved {context} frame parent")
        current = parent


def rajagopal_rigid_skeleton_ir(model: dict[str, Any]) -> dict[str, Any]:
    """Resolve source OpenSim joint sockets to an exact rigid-body tree IR."""
    body_ids = {body.get("id") for body in model.get("bodies", [])}
    if not body_ids or not all(isinstance(identifier, str) and identifier for identifier in body_ids):
        raise ImportError("Rajagopal skeleton has invalid body identifiers")
    inbound: dict[str, str] = {}
    joints: list[dict[str, Any]] = []
    for joint in model.get("joints", []):
        identifier = joint.get("id")
        kind = joint.get("kind")
        if not isinstance(identifier, str) or not isinstance(kind, str):
            raise ImportError("Rajagopal skeleton has unnamed joint")
        parent_body = _opensim_joint_frame_body(
            joint, joint.get("parent_frame"), "parent"
        )
        child_body = _opensim_joint_frame_body(
            joint, joint.get("child_frame"), "child"
        )
        if child_body is None or child_body not in body_ids:
            raise ImportError(f"OpenSim joint {identifier} has invalid child body")
        if parent_body is not None and parent_body not in body_ids:
            raise ImportError(f"OpenSim joint {identifier} has invalid parent body")
        if child_body in inbound:
            raise ImportError(
                f"OpenSim body {child_body} has multiple inbound joints: "
                f"{inbound[child_body]} and {identifier}"
            )
        inbound[child_body] = identifier
        lowering: dict[str, Any]
        if kind == "PinJoint":
            lowering = {"status": "core_scalar_supported", "numi_primitive": "revolute"}
        elif kind == "CustomJoint":
            lowering = {
                "status": "core_function_based_supported",
                "program_file": f"opensim-spatial-programs/{identifier}.mrospatial",
                "accelerated_runtime": "bounded_fixed_or_source_default_mobile_root",
            }
        elif kind == "UniversalJoint":
            coordinates = joint.get("coordinates", [])
            if (
                isinstance(coordinates, list)
                and coordinates
                and all(
                    coordinate.get("locked") in (True, "true", "True")
                    and coordinate.get("default_value") == 0.0
                    for coordinate in coordinates
                )
            ):
                lowering = {
                    "status": "exact_locked_lowering",
                    "numi_primitive": "fixed",
                    "condition": "all source coordinates locked at zero default",
                }
            else:
                lowering = {"status": "requires_core_extension"}
        else:
            lowering = {"status": "unsupported_source_kind"}
        joints.append(
            {
                "id": identifier,
                "kind": kind,
                "parent_body": parent_body,
                "child_body": child_body,
                "parent_frame": joint.get("parent_frame"),
                "child_frame": joint.get("child_frame"),
                "coordinates": joint.get("coordinates"),
                "frames": joint.get("frames"),
                "motion_axes": joint.get("motion_axes"),
                "lowering": lowering,
                "source_xml": joint.get("source_xml"),
            }
        )
    roots = sorted(
        joint["child_body"] for joint in joints if joint["parent_body"] is None
    )
    if len(roots) != 1:
        raise ImportError(
            "Rajagopal skeleton requires one ground-attached rigid-tree root; found "
            + ", ".join(roots)
        )
    if len(joints) != len(body_ids):
        raise ImportError("Rajagopal skeleton source does not form one body/joint tree")
    return {
        "schema": "numi.human.opensim-rigid-skeleton-ir.v1",
        "source": {
            "id": model.get("source_id"),
            "file": model.get("source_file"),
            "sha256": model.get("source_sha256"),
            "model_id": model.get("model_id"),
        },
        "root_body": roots[0],
        "body_count": len(body_ids),
        "joint_count": len(joints),
        "bodies": model.get("bodies", []),
        "joints": joints,
        "lowering_summary": dict(
            sorted(Counter(joint["lowering"]["status"] for joint in joints).items())
        ),
        "runtime_requirement": (
            "The qualified Core path assembles one source body/frame/inertia "
            "tree and FunctionBased programs into MetalWorld-resident direct-effort "
            "state, including fixed-root and source-default-preserving mobile-root "
            "synthetic streamed-contact response probes. "
            "BodyParts3D registration, anatomical colliders/materials, and broader "
            "model admission remain separate work."
        ),
        "evidence_boundary": (
            "Exact OpenSim body, frame, joint, and coordinate topology only. This IR does "
            "not itself register BodyParts3D geometry, create collision proxies, or prove "
            "an OpenSim-equivalent articulated solve."
        ),
    }


_RAJAGOPAL_CORE_REFERENCE_MAGIC = b"NHRIGID1"
_RAJAGOPAL_CORE_REFERENCE_ABI = 1
_RAJAGOPAL_MILLARD_REFERENCE_MAGIC = b"NHMUSC1\0"
_RAJAGOPAL_MILLARD_REFERENCE_ABI = 3
_MR_ENGINE_ABI_VERSION = 5
_MR_MOTION_DYNAMIC = 2
_MR_ROOT_FIXED = 0
_MR_JOINT_REVOLUTE = 0
_MR_JOINT_FIXED = 5
_MR_JOINT_FUNCTION_BASED = 7
_MR_DOF_POSITION_LIMIT = 1 << 2

# MyoSim's authored MuJoCo model uses an inertial-frame body convention.  The
# Core ABI is also COM centred, so the native lowerer retains each source
# inertia frame and uses exact zero-inertia transform carriers only where a
# MuJoCo body owns multiple serial joints.  Those carriers are not anatomy or
# added mass: they preserve source joint order without a fabricated inertia.
_MYOSIM_CORE_REFERENCE_MAGIC = b"NHRIGID2"
_MYOSIM_CORE_REFERENCE_ABI = 1
_MYOSIM_MUSCLE_REFERENCE_LEGACY_MAGIC = b"NHMYO1\0\0"
_MYOSIM_MUSCLE_REFERENCE_LEGACY_ABI = 1
_MYOSIM_MUSCLE_REFERENCE_MAGIC = b"NHMYO2\0\0"
_MYOSIM_MUSCLE_REFERENCE_ABI = 2
_MYOSIM_MUSCLE_ARCHITECTURE_FORMAT = "<8f"
_MYOSIM_MUSCLE_ARCHITECTURE_BYTES = struct.calcsize(_MYOSIM_MUSCLE_ARCHITECTURE_FORMAT)
# Source-authored full-body foot support witnesses.  These are compiled
# MuJoCo capsule/ellipsoid surface points against MyoSim's own ground plane,
# not BodyParts3D collision proxies.
_MYOSIM_SUPPORT_CONTACT_MAGIC = b"NHCNT1\0\0"
_MYOSIM_SUPPORT_CONTACT_ABI = 1
_MYOSIM_JOINT_EQUALITY_MAGIC = b"NHEQ1\0\0\0"
_MYOSIM_JOINT_EQUALITY_ABI = 1
_MYOSIM_JOINT_EQUALITY_RECORD_BYTES = 96
# Canonical Numi Human endpoint program.  The compact eight-byte magic is the
# binary spelling of the public ``NHTENDON1`` payload name.
_NUMI_HUMAN_TENDON_MAGIC = b"NHTEND1\0"
_NUMI_HUMAN_TENDON_ABI = 1
_NUMI_HUMAN_TENDON_POINT = 0
_NUMI_HUMAN_TENDON_TRIANGLE = 1
# NHTENDON2 never migrates an authored MyoSim endpoint.  A mechanically
# admitted record instead carries a source-point-preserving distributed wrench
# envelope on one exact NHBONES1 member.  The runtime keeps the source route
# and force law authoritative, then transfers its terminal force across the
# envelope while conserving both resultant force and moment.
_NUMI_HUMAN_TENDON_ENVELOPE_MAGIC = b"NHTEND2\0"
_NUMI_HUMAN_TENDON_ENVELOPE_ABI = 2
_NUMI_HUMAN_TENDON_ENVELOPE = 2
# NHTENDON3 retains the exact NHTENDON2 force-envelope layout, but explicitly
# admits a route-private endpoint at the named bone surface.  The last binding
# float carries the migration magnitude; NHTENDON2 keeps that field zero.
_NUMI_HUMAN_TENDON_MIGRATED_ENVELOPE_MAGIC = b"NHTEND3\0"
_NUMI_HUMAN_TENDON_MIGRATED_ENVELOPE_ABI = 3
_NUMI_HUMAN_TENDON_MIGRATED_ENVELOPE = 3
_MR_MOTION_STATIC = 0
_MR_ROOT_FLOATING = 1
_MR_JOINT_PRISMATIC = 1
_MR_DOF_ROOT = 1 << 0
_MYOSIM_ROUTE_SITE = 1
_MYOSIM_ROUTE_SPHERE = 2
_MYOSIM_ROUTE_CYLINDER = 3


def _myosim_muscle_payload_architecture(
    magic: bytes, abi: int, muscle_count: int, reserved0: int, reserved1: int,
) -> tuple[int, int]:
    """Return the optional appended architecture-table shape for NHMYO1/2."""
    if magic == _MYOSIM_MUSCLE_REFERENCE_LEGACY_MAGIC and abi == _MYOSIM_MUSCLE_REFERENCE_LEGACY_ABI:
        if reserved0 != 0 or reserved1 != 0:
            raise ImportError("legacy MyoSim muscle payload has nonzero reserved fields")
        return 0, 0
    if magic == _MYOSIM_MUSCLE_REFERENCE_MAGIC and abi == _MYOSIM_MUSCLE_REFERENCE_ABI:
        if reserved0 != muscle_count or reserved1 != _MYOSIM_MUSCLE_ARCHITECTURE_BYTES:
            raise ImportError("NHMYO2 architecture table shape is invalid")
        return reserved0, reserved1
    raise ImportError("MyoSim muscle payload has an unsupported ABI")


def _myosim_muscle_payload_bytes(
    site_count: int, wrap_count: int, route_count: int, muscle_count: int,
    architecture_count: int, architecture_bytes: int,
) -> int:
    return (
        struct.calcsize("<8s9I32s") + 16 * site_count + 64 * wrap_count
        + 16 * route_count + 164 * muscle_count
        + architecture_count * architecture_bytes
    )


def myosim_part_control_catalog(myosim_artifact: Path) -> dict[str, Any]:
    """Resolve exact source muscles incident to each compiled Human body.

    This is launch-time control metadata. It reads the source-pinned NHMYO
    route table and never invents a joint, torque, muscle, or anatomical
    grouping. Native Metal still owns force evaluation and pose execution.
    """
    artifact = myosim_artifact.resolve()
    manifest_path = artifact / "myosim-fullbody-reference.manifest.json"
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "numi.human.myosim-fullbody-reference.v1"
    ):
        raise ImportError("MyoSim part control requires a full-body reference manifest")
    source = manifest.get("source")
    source_sha = source.get("archive_sha256") if isinstance(source, dict) else None
    payloads = manifest.get("payloads")
    payload_descriptor = payloads.get("muscles") if isinstance(payloads, dict) else None
    if (
        not isinstance(source_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_sha)
        or not isinstance(payload_descriptor, dict)
    ):
        raise ImportError("MyoSim part control has incomplete source provenance")
    payload_path = artifact / str(payload_descriptor.get("file"))
    expected_payload_sha = payload_descriptor.get("sha256")
    if (
        not payload_path.is_file()
        or not isinstance(expected_payload_sha, str)
        or sha256(payload_path) != expected_payload_sha
    ):
        raise ImportError("MyoSim part control muscle payload is missing or drifted")
    payload = payload_path.read_bytes()
    header_format = "<8s9I32s"
    header_bytes = struct.calcsize(header_format)
    if len(payload) < header_bytes:
        raise ImportError("MyoSim part control muscle payload header is truncated")
    (
        magic, abi, body_count, muscle_count, site_count, wrap_count,
        route_count, _, reserved0, reserved1, embedded_sha,
    ) = struct.unpack_from(header_format, payload)
    architecture_count, architecture_bytes = _myosim_muscle_payload_architecture(
        magic, abi, muscle_count, reserved0, reserved1,
    )
    if (
        embedded_sha.hex() != source_sha
        or len(payload) != _myosim_muscle_payload_bytes(
            site_count, wrap_count, route_count, muscle_count,
            architecture_count, architecture_bytes,
        )
    ):
        raise ImportError("MyoSim part control muscle payload ABI is invalid")
    core_tree = manifest.get("core_tree")
    body_order = core_tree.get("body_order") if isinstance(core_tree, dict) else None
    muscle_metadata = manifest.get("muscles")
    if (
        not isinstance(body_order, list) or len(body_order) != body_count
        or any(not isinstance(name, str) or not name for name in body_order)
        or len(set(body_order)) != body_count
        or not isinstance(muscle_metadata, list) or len(muscle_metadata) != muscle_count
    ):
        raise ImportError("MyoSim part control identity tables are incomplete")
    offset = header_bytes
    sites = [
        struct.unpack_from("<I3f", payload, offset + 16 * index)
        for index in range(site_count)
    ]
    offset += 16 * site_count
    wraps = [
        struct.unpack_from("<2I14f", payload, offset + 64 * index)
        for index in range(wrap_count)
    ]
    offset += 64 * wrap_count
    routes = [
        struct.unpack_from("<4I", payload, offset + 16 * index)
        for index in range(route_count)
    ]
    offset += 16 * route_count
    muscle_records = [
        struct.unpack_from("<4I37f", payload, offset + 164 * index)
        for index in range(muscle_count)
    ]
    muscles: list[dict[str, Any]] = []
    muscles_by_part: dict[int, list[dict[str, Any]]] = {}
    seen_names: set[str] = set()
    for muscle_index, (record, metadata) in enumerate(
        zip(muscle_records, muscle_metadata, strict=True)
    ):
        if not isinstance(metadata, dict):
            raise ImportError("MyoSim part control has an invalid muscle identity")
        source_index = metadata.get("source_actuator_index")
        name = metadata.get("name")
        route_offset, muscle_route_count = record[1], record[2]
        if (
            source_index != muscle_index
            or not isinstance(name, str) or not name or name in seen_names
            or route_offset + muscle_route_count > len(routes)
            or muscle_route_count < 2
        ):
            raise ImportError("MyoSim part control muscle identity or route drifted")
        seen_names.add(name)
        route_body_indices: list[int] = []
        for route in routes[route_offset:route_offset + muscle_route_count]:
            kind, target = route[0], route[1]
            if kind == _MYOSIM_ROUTE_SITE:
                if target >= len(sites):
                    raise ImportError("MyoSim part control route references an absent site")
                body_index = sites[target][0]
            elif kind in {_MYOSIM_ROUTE_SPHERE, _MYOSIM_ROUTE_CYLINDER}:
                if target >= len(wraps):
                    raise ImportError("MyoSim part control route references an absent wrap")
                body_index = wraps[target][0]
            else:
                raise ImportError("MyoSim part control route has an unsupported node kind")
            if body_index >= body_count:
                raise ImportError("MyoSim part control route escapes the Core body table")
            if body_index not in route_body_indices:
                route_body_indices.append(body_index)
        muscle_record = {
            "source_actuator_index": source_index,
            "name": name,
            "route_core_body_indices": route_body_indices,
            "route_body_names": [body_order[index] for index in route_body_indices],
        }
        muscles.append(muscle_record)
        for body_index in route_body_indices:
            muscles_by_part.setdefault(body_index, []).append({
                "source_actuator_index": source_index,
                "name": name,
            })
    parts = [
        {
            "core_body_index": body_index,
            "body_name": body_order[body_index],
            "source_muscle_count": len(muscles_by_part[body_index]),
            "source_muscles": muscles_by_part[body_index],
        }
        for body_index in sorted(muscles_by_part)
    ]
    return {
        "schema": "numi.human.myosim-part-control-catalog.v1",
        "source": {
            "manifest_file": manifest_path.name,
            "manifest_sha256": sha256(manifest_path),
            "muscle_payload_file": payload_path.name,
            "muscle_payload_sha256": expected_payload_sha,
            "myosim_source_archive_sha256": source_sha,
        },
        "coverage": {
            "core_body_count": body_count,
            "controllable_part_count": len(parts),
            "source_muscle_count": len(muscles),
        },
        "parts": parts,
        "muscles": muscles,
        "control_semantics": (
            "selecting a part activates the union of exact source muscles whose compiled route "
            "contains that Core body; selecting a muscle activates only that exact source actuator"
        ),
        "evidence_boundary": (
            "This catalog exposes bounded diagnostic excitation, not a movement controller, "
            "synergy inference, direct torque path, clinical motor map, or independent digit articulation."
        ),
    }


def myosim_part_control_plan(
    myosim_artifact: Path, part_names: list[str], muscle_names: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve requested source body and muscle names to exact actuator rows."""
    catalog = myosim_part_control_catalog(myosim_artifact)
    requested_parts = list(dict.fromkeys(part_names))
    requested_muscles = list(dict.fromkeys(muscle_names or []))
    if not requested_parts and not requested_muscles:
        raise ImportError("MyoSim part control requires at least one part or muscle")
    parts_by_name = {part["body_name"]: part for part in catalog["parts"]}
    muscles_by_name = {muscle["name"]: muscle for muscle in catalog["muscles"]}
    unknown_parts = [name for name in requested_parts if name not in parts_by_name]
    unknown_muscles = [name for name in requested_muscles if name not in muscles_by_name]
    if unknown_parts:
        raise ImportError("unknown controllable MyoSim part: " + ", ".join(unknown_parts))
    if unknown_muscles:
        raise ImportError("unknown MyoSim source muscle: " + ", ".join(unknown_muscles))
    selected_indices = {
        muscle["source_actuator_index"]
        for name in requested_parts
        for muscle in parts_by_name[name]["source_muscles"]
    }
    selected_indices.update(
        muscles_by_name[name]["source_actuator_index"] for name in requested_muscles
    )
    selected_muscles = [
        muscle for muscle in catalog["muscles"]
        if muscle["source_actuator_index"] in selected_indices
    ]
    if not selected_muscles:
        raise ImportError("MyoSim part control selection resolved no source muscles")
    focus_body_index = (
        parts_by_name[requested_parts[0]]["core_body_index"]
        if len(requested_parts) == 1 else None
    )
    return {
        "schema": "numi.human.myosim-part-control-plan.v1",
        "source": catalog["source"],
        "requested_parts": requested_parts,
        "requested_muscles": requested_muscles,
        "focus_core_body_index": focus_body_index,
        "selected_source_muscle_count": len(selected_muscles),
        "selected_source_muscles": selected_muscles,
        "control_semantics": catalog["control_semantics"],
        "evidence_boundary": catalog["evidence_boundary"],
    }


def _myosim_active_force_length(normalized_length: float, lower: float, upper: float) -> float:
    if normalized_length < lower or normalized_length > upper:
        return 0.0
    lower_mid = 0.5 * (lower + 1.0)
    upper_mid = 0.5 * (1.0 + upper)
    if normalized_length <= lower_mid:
        x = (normalized_length - lower) / max(1.0e-12, lower_mid - lower)
        return 0.5 * x * x
    if normalized_length <= 1.0:
        x = (1.0 - normalized_length) / max(1.0e-12, 1.0 - lower_mid)
        return 1.0 - 0.5 * x * x
    if normalized_length <= upper_mid:
        x = (normalized_length - 1.0) / max(1.0e-12, upper_mid - 1.0)
        return 1.0 - 0.5 * x * x
    x = (upper - normalized_length) / max(1.0e-12, upper - upper_mid)
    return 0.5 * x * x


def _myosim_passive_force_length(normalized_length: float, upper: float, scale: float) -> float:
    if normalized_length <= 1.0:
        return 0.0
    upper_mid = 0.5 * (1.0 + upper)
    if normalized_length <= upper_mid:
        x = (normalized_length - 1.0) / max(1.0e-12, upper_mid - 1.0)
        return scale * 0.5 * x * x
    x = (normalized_length - upper_mid) / max(1.0e-12, upper_mid - 1.0)
    return scale * (0.5 + x)


def _numi_generic_tendon_force(normalized_length: float) -> float:
    """One normalized Rajagopal/OpenSim-derived tendon curve for every muscle."""
    strain_at_one = 0.049
    stiffness_at_one = 1.375 / strain_at_one
    force_at_toe = 2.0 / 3.0
    strain_at_toe = strain_at_one - (1.0 - force_at_toe) / stiffness_at_one
    strain = normalized_length - 1.0
    if strain <= 0.0:
        return 0.0
    if strain >= strain_at_toe:
        return force_at_toe + stiffness_at_one * (strain - strain_at_toe)
    # C1 Hermite toe: zero force/slope at slack and the source-default force
    # and slope at the linear transition. The 0.5 curviness is the canonical
    # OpenSim default retained in the ABI below; no per-muscle curve is fitted.
    t = strain / strain_at_toe
    return (-2.0 * t**3 + 3.0 * t**2) * force_at_toe + (t**3 - t**2) * (
        strain_at_toe * stiffness_at_one
    )


def _numi_static_compliant_force(
    path_length: float, activation: float, optimal_fiber_length: float,
    tendon_slack_length: float, gain: list[float], bias: list[float],
) -> float:
    """Solve zero-pennation static fibre/tendon equilibrium in normalized force."""
    lower = min(0.05 * optimal_fiber_length, 0.5 * path_length)
    upper = path_length
    if not lower < upper:
        return 0.0

    def residual(fiber_length: float) -> tuple[float, float]:
        fiber_normalized = fiber_length / optimal_fiber_length
        tendon_normalized = (path_length - fiber_length) / tendon_slack_length
        tendon_force = _numi_generic_tendon_force(tendon_normalized)
        fiber_force = activation * _myosim_active_force_length(
            fiber_normalized, gain[4], gain[5]
        ) + _myosim_passive_force_length(fiber_normalized, bias[5], bias[7])
        return tendon_force - fiber_force, tendon_force

    left_residual, left_force = residual(lower)
    right_residual, right_force = residual(upper)
    if left_residual * right_residual > 0.0:
        return left_force if abs(left_residual) < abs(right_residual) else right_force
    midpoint_force = 0.0
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        midpoint_residual, midpoint_force = residual(midpoint)
        if midpoint_residual > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return max(0.0, midpoint_force)


def _fit_myosim_compliant_architecture(
    length_range: list[float], acceleration_scale: float,
    gain: list[float], bias: list[float], oracle_length: float | None = None,
) -> tuple[float, float, float]:
    """Fit positive L0/LT to the retained static MuJoCo force surface.

    MyoSim does not identify pennation or elastic tendon architecture. We fit
    only two positive lengths offline and keep its authored active/passive
    curves and Fmax authoritative. A two-dimensional bounded search is needed:
    forcing L0 + 1.049 LT to the source optimum reproduces the nonphysical
    negative tendon offsets present in many rigid-tendon source actuators.
    """
    source_optimal_length = (length_range[1] - length_range[0]) / max(
        1.0e-12, gain[1] - gain[0]
    )
    source_optimal_path = length_range[0] + (1.0 - gain[0]) * source_optimal_length
    if not math.isfinite(source_optimal_path) or source_optimal_path <= 1.0e-6:
        raise ImportError("MyoSim muscle has no positive source-optimal path length")
    maximum_force = gain[2] if gain[2] >= 0.0 else gain[3] / max(1.0e-12, acceleration_scale)
    passive_force = bias[2] if bias[2] >= 0.0 else bias[3] / max(1.0e-12, acceleration_scale)
    if not math.isfinite(maximum_force) or maximum_force <= 0.0 or not math.isfinite(passive_force):
        raise ImportError("MyoSim muscle has no positive finite force scale")

    samples: list[tuple[float, float, float]] = []
    normalized_minimum = min(gain[0], gain[1])
    normalized_maximum = max(gain[0], gain[1])
    normalized_samples = [
        normalized_minimum + (normalized_maximum - normalized_minimum) * index / 8.0
        for index in range(9)
    ]
    if normalized_minimum <= 1.0 <= normalized_maximum:
        normalized_samples.append(1.0)
    if oracle_length is not None and math.isfinite(oracle_length) and oracle_length > 0.0:
        normalized_samples.append(
            gain[0] + (oracle_length - length_range[0]) / source_optimal_length
        )
    for normalized_length in sorted(set(normalized_samples)):
        path_length = length_range[0] + (normalized_length - gain[0]) * source_optimal_length
        if path_length <= 1.0e-6:
            continue
        active = _myosim_active_force_length(normalized_length, gain[4], gain[5])
        passive = _myosim_passive_force_length(normalized_length, bias[5], bias[7])
        for activation in (0.1, 0.5, 1.0):
            target = activation * active + (passive_force / maximum_force) * passive
            samples.append((path_length, activation, max(0.0, target)))
    if not samples:
        raise ImportError("MyoSim muscle architecture fit has no valid force samples")

    best: tuple[float, float, float] | None = None
    minimum_operating_path = min(
        value for value in length_range if math.isfinite(value) and value > 1.0e-6
    )
    for fiber_index in range(17):
        optimal_fiber = source_optimal_length * (
            0.35 + 1.30 * fiber_index / 16.0
        )
        for tendon_index in range(17):
            tendon_slack = minimum_operating_path * (
                0.005 + 0.745 * tendon_index / 16.0
            )
            squared_error = 0.0
            squared_target = 0.0
            for path_length, activation, target in samples:
                predicted = _numi_static_compliant_force(
                    path_length, activation, optimal_fiber, tendon_slack, gain, bias
                )
                squared_error += (predicted - target) ** 2
                squared_target += target * target
            normalized_rmse = math.sqrt(squared_error / len(samples)) / max(
                1.0e-6, math.sqrt(squared_target / len(samples))
            )
            if best is None or normalized_rmse < best[2]:
                best = (optimal_fiber, tendon_slack, normalized_rmse)
    if best is None or not all(math.isfinite(value) and value > 0.0 for value in best):
        raise ImportError("MyoSim muscle compliant architecture fit failed")
    return best


def _myosim_matrix_from_quaternion_xyzw(quaternion: list[float]) -> list[list[float]]:
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise ImportError("MyoSim quaternion must be a finite xyzw tuple")
    x, y, z, w = quaternion
    squared = x * x + y * y + z * z + w * w
    if squared <= 1.0e-14:
        raise ImportError("MyoSim quaternion must be nonzero")
    scale = 1.0 / math.sqrt(squared)
    x, y, z, w = (value * scale for value in (x, y, z, w))
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _myosim_vector(value: Any, context: str) -> list[float]:
    return _vector3(value, context)


def _myosim_matrix_vector(matrix: list[list[float]], value: list[float]) -> list[float]:
    return [sum(matrix[row][column] * value[column] for column in range(3)) for row in range(3)]


def _myosim_subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def _myosim_add(left: list[float], right: list[float]) -> list[float]:
    return [left[index] + right[index] for index in range(3)]


def _myosim_identity() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _myosim_body_local_from_body_frame(
    body: dict[str, Any], point_body: list[float], context: str
) -> list[float]:
    """Convert a MuJoCo body-frame point into Core's COM/inertia frame."""
    inertial_position = _myosim_vector(body.get("inertial_position_body_m"), context + " inertial position")
    inertial_rotation = _myosim_matrix_from_quaternion_xyzw(
        list(body.get("inertial_quaternion_body_xyzw", []))
    )
    return _myosim_matrix_vector(
        _matrix_transpose(inertial_rotation),
        _myosim_subtract(point_body, inertial_position),
    )


def _myosim_world_joint_anchor(
    body: dict[str, Any], joint_position_body: list[float], context: str
) -> list[float]:
    body_position = _myosim_vector(body.get("default_body_position_world_m"), context + " body position")
    body_rotation = _myosim_matrix_from_quaternion_xyzw(
        list(body.get("default_body_quaternion_world_xyzw", []))
    )
    return _myosim_add(body_position, _myosim_matrix_vector(body_rotation, joint_position_body))


def _myosim_pack_body_record(
    *,
    parent_body: int,
    inbound_joint: int,
    source_body: dict[str, Any] | None,
    virtual: bool,
    context: str,
) -> bytes:
    if virtual:
        motion = _MR_MOTION_STATIC
        mass = 0.0
        inverse_mass = 0.0
        inertia = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        inverse = inertia
    else:
        if source_body is None:
            raise ImportError(f"{context} has no source body")
        mass = _finite_scalar(source_body.get("mass_kg"), context + " mass")
        raw_inertia = source_body.get("inertia_kg_m2")
        if mass == 0.0:
            motion = _MR_MOTION_STATIC
            inverse_mass = 0.0
            inertia = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            inverse = inertia
        else:
            if mass < 0.0:
                raise ImportError(f"{context} has negative source mass")
            if not isinstance(raw_inertia, list) or len(raw_inertia) != 3:
                raise ImportError(f"{context} has no diagonal source inertia")
            principal = [_finite_scalar(value, context + " principal inertia") for value in raw_inertia]
            if not all(value > 0.0 for value in principal):
                raise ImportError(f"{context} has non-positive source inertia")
            motion = _MR_MOTION_DYNAMIC
            inverse_mass = 1.0 / mass
            inertia = [
                [principal[0], 0.0, 0.0],
                [0.0, principal[1], 0.0],
                [0.0, 0.0, principal[2]],
            ]
            inverse = [
                [1.0 / principal[0], 0.0, 0.0],
                [0.0, 1.0 / principal[1], 0.0],
                [0.0, 0.0, 1.0 / principal[2]],
            ]
    return (
        struct.pack("<4I", 0, parent_body, inbound_joint, motion)
        + _pack_float4([mass, inverse_mass, 0.0, 0.0], context + " mass")
        + _pack_float4([0.0, 0.0, 0.0, 0.0], context + " COM")
        + b"".join(_pack_float4([*row, 0.0], context + " inertia") for row in inertia)
        + b"".join(_pack_float4([*row, 0.0], context + " inverse inertia") for row in inverse)
        + _pack_float4([0.0, 0.0, 0.0, 0.0], context + " damping")
    )


def _myosim_pack_joint_record(
    *,
    parent_body: int,
    child_body: int,
    joint_type: int,
    q_offset: int,
    nq: int,
    v_offset: int,
    nv: int,
    axis: list[float],
    parent_anchor: list[float],
    child_anchor: list[float],
    parent_rotation: list[list[float]],
    child_rotation: list[list[float]],
    context: str,
) -> bytes:
    return (
        struct.pack("<8I", parent_body, child_body, joint_type, 0, q_offset, nq, v_offset, nv)
        + _pack_float4([*axis, 0.0], context + " axis")
        + _pack_float4([0.0, 0.0, 0.0, 0.0], context + " axis 1")
        + _pack_float4([0.0, 0.0, 0.0, 0.0], context + " axis 2")
        + _pack_float4([*parent_anchor, 0.0], context + " parent anchor")
        + _pack_float4([*child_anchor, 0.0], context + " child anchor")
        + _pack_float4(_quaternion_xyzw_from_matrix(parent_rotation), context + " parent rotation")
        + _pack_float4(_quaternion_xyzw_from_matrix(child_rotation), context + " child rotation")
    )


def _myosim_pack_dof_record(
    *,
    joint_index: int,
    q_index: int,
    v_index: int,
    local_dof: int,
    flags: int,
    limits: list[float],
    armature: float,
    damping: float,
    frictionloss: float,
    context: str,
) -> bytes:
    return (
        struct.pack("<8I", 0, joint_index, q_index, v_index, local_dof, flags, 0, 0)
        + _pack_float4(limits, context + " limits")
        # Core's drive tuple is [stiffness, viscous damping, armature,
        # dry friction].  MyoSim authors non-zero generalized damping on 122
        # of the scalar DoFs, including the wrist and finger chains.  Dropping
        # it made those very small inertias visually explode even when the
        # muscle-force residual was modest.
        + _pack_float4([0.0, damping, armature, frictionloss], context + " drive")
    )


def myosim_fullbody_reference_artifacts(
    exported: dict[str, Any],
) -> tuple[dict[str, Any], bytes, bytes, bytes, bytes]:
    """Lower MyoSim's compiled full body into Core rigid and muscle payloads.

    The source uses a MuJoCo free root and several multiple-joint bodies.  A
    one-joint-per-body Core tree therefore receives zero-inertia transform
    carriers between serial source joints.  They carry no mass, inertia, or
    anatomy and are admitted only by the CPU reference solver.
    """
    if exported.get("schema") != "numi.human.myosim-mujoco-export.v1":
        raise ImportError("MyoSim build requires numi.human.myosim-mujoco-export.v1")
    source = exported.get("source")
    model = exported.get("model")
    bodies = exported.get("bodies")
    joints = exported.get("joints")
    joint_equalities = exported.get("joint_equalities")
    sites = exported.get("sites")
    geometries = exported.get("wrap_geometries")
    muscles = exported.get("muscles")
    if not all(isinstance(value, list) for value in (
        bodies, joints, joint_equalities, sites, geometries, muscles
    )):
        raise ImportError("MyoSim export has incomplete source arrays")
    if not isinstance(source, dict) or not isinstance(model, dict):
        raise ImportError("MyoSim export has no source or model records")
    source_hash = source.get("archive_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ImportError("MyoSim export has no source archive SHA-256")
    if source.get("revision") != "33c89c2bde282553dde3f526768eb3bdcfaa7649":
        raise ImportError("MyoSim export is not the selected full-body source revision")

    body_by_id = {body.get("id"): body for body in bodies if isinstance(body, dict)}
    joint_by_id = {joint.get("id"): joint for joint in joints if isinstance(joint, dict)}
    if len(body_by_id) != len(bodies) or len(joint_by_id) != len(joints):
        raise ImportError("MyoSim export has duplicate or unnamed source identities")
    root_body_id = model.get("root_body")
    root_joint_id = model.get("root_joint")
    if root_body_id not in body_by_id or root_joint_id not in joint_by_id:
        raise ImportError("MyoSim export has an unresolved free root")
    root_joint = joint_by_id[root_joint_id]
    if root_joint.get("body") != root_body_id or root_joint.get("type") != 0:
        raise ImportError("MyoSim export free root is not a MuJoCo free joint")
    source_qpos = model.get("default_qpos")
    if not isinstance(source_qpos, list):
        raise ImportError("MyoSim export omits default qpos")
    source_qpos = [_finite_scalar(value, "MyoSim default qpos") for value in source_qpos]
    if len(source_qpos) != model.get("nq"):
        raise ImportError("MyoSim export default qpos count disagrees with nq")

    joints_for_body: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for joint in joints:
        body_id = joint.get("body")
        if body_id not in body_by_id:
            raise ImportError("MyoSim joint references an unknown source body")
        joints_for_body[body_id].append(joint)
    for body_joints in joints_for_body.values():
        body_joints.sort(key=lambda entry: int(entry.get("id", -1)))
    if joints_for_body[root_body_id] != [root_joint]:
        raise ImportError("MyoSim free root body owns additional unsupported source joints")

    # Core body 0 is the source's free-root body, represented at its true
    # source inertial/COM pose rather than a synthetic fixed anchor.
    root_source = body_by_id[root_body_id]
    root_position = _myosim_vector(root_source.get("default_com_position_world_m"), "MyoSim root COM")
    root_quaternion = list(root_source.get("default_inertial_quaternion_world_xyzw", []))
    _myosim_matrix_from_quaternion_xyzw(root_quaternion)
    body_nodes: list[dict[str, Any]] = [
        {"name": str(root_source.get("name")), "source_body": root_source, "parent": 0xFFFFFFFF,
         "inbound": 0xFFFFFFFF, "virtual": False}
    ]
    source_body_to_core: dict[int, int] = {int(root_body_id): 0}
    joint_records: list[bytes] = []
    dof_records: list[bytes] = []
    default_q = [*root_position, *[_finite_scalar(value, "MyoSim root quaternion") for value in root_quaternion]]
    default_v = [0.0] * 6
    root_armature = _finite_scalar(root_joint.get("armature"), "MyoSim free-root armature")
    if root_armature != 0.0:
        raise ImportError("MyoSim free-root armature is unsupported by the Core root ABI")
    for local_dof in range(6):
        dof_records.append(
            _myosim_pack_dof_record(
                joint_index=0xFFFFFFFF,
                q_index=local_dof if local_dof < 3 else 0xFFFFFFFF,
                v_index=local_dof,
                local_dof=local_dof,
                flags=_MR_DOF_ROOT,
                limits=[0.0, 0.0, 0.0, 0.0],
                armature=0.0,
                damping=0.0,
                frictionloss=0.0,
                context="MyoSim free root",
            )
        )

    unresolved = {int(body_id) for body_id in body_by_id if int(body_id) != int(root_body_id)}
    source_joint_map: list[dict[str, Any]] = []
    while unresolved:
        progressed = False
        for source_body_id in sorted(tuple(unresolved)):
            source_body = body_by_id[source_body_id]
            parent_source_id = source_body.get("parent")
            if parent_source_id not in source_body_to_core:
                continue
            parent_core = source_body_to_core[parent_source_id]
            body_joints = joints_for_body.get(source_body_id, [])
            source_body_rotation = _myosim_matrix_from_quaternion_xyzw(
                list(source_body.get("default_body_quaternion_world_xyzw", []))
            )
            source_body_position = _myosim_vector(
                source_body.get("default_body_position_world_m"),
                f"MyoSim body {source_body_id} default position",
            )
            parent_source = body_by_id[parent_source_id]
            parent_inertial_rotation = _myosim_matrix_from_quaternion_xyzw(
                list(parent_source.get("default_inertial_quaternion_world_xyzw", []))
            )
            parent_com = _myosim_vector(parent_source.get("default_com_position_world_m"), "MyoSim parent COM")
            child_inertial_rotation = _myosim_matrix_from_quaternion_xyzw(
                list(source_body.get("default_inertial_quaternion_world_xyzw", []))
            )
            if not body_joints:
                child_core = len(body_nodes)
                parent_rotation = _matrix_product(_matrix_transpose(parent_inertial_rotation), source_body_rotation)
                child_rotation = _matrix_product(_matrix_transpose(child_inertial_rotation), source_body_rotation)
                parent_anchor = _myosim_matrix_vector(
                    _matrix_transpose(parent_inertial_rotation),
                    _myosim_subtract(source_body_position, parent_com),
                )
                child_anchor = _myosim_body_local_from_body_frame(
                    source_body, [0.0, 0.0, 0.0], f"MyoSim fixed body {source_body_id}"
                )
                joint_index = len(joint_records)
                joint_records.append(
                    _myosim_pack_joint_record(
                        parent_body=parent_core, child_body=child_core, joint_type=_MR_JOINT_FIXED,
                        q_offset=len(default_q), nq=0, v_offset=len(default_v), nv=0,
                        axis=[0.0, 0.0, 0.0], parent_anchor=parent_anchor, child_anchor=child_anchor,
                        parent_rotation=parent_rotation, child_rotation=child_rotation,
                        context=f"MyoSim fixed body {source_body_id}",
                    )
                )
                body_nodes.append(
                    {"name": str(source_body.get("name")), "source_body": source_body,
                     "parent": parent_core, "inbound": joint_index, "virtual": False}
                )
                source_body_to_core[source_body_id] = child_core
            else:
                if any(joint.get("type") not in (2, 3) for joint in body_joints):
                    kinds = ", ".join(str(joint.get("type")) for joint in body_joints)
                    raise ImportError(f"MyoSim body {source_body_id} has unsupported non-free joint types: {kinds}")
                prior_core = parent_core
                for local_index, source_joint in enumerate(body_joints):
                    terminal = local_index == len(body_joints) - 1
                    child_core = len(body_nodes)
                    source_position = _myosim_vector(
                        source_joint.get("position_body_m"), f"MyoSim joint {source_joint.get('id')} position"
                    )
                    axis = _myosim_vector(source_joint.get("axis_body"), f"MyoSim joint {source_joint.get('id')} axis")
                    if sum(value * value for value in axis) <= 1.0e-14:
                        raise ImportError(f"MyoSim joint {source_joint.get('id')} has a zero source axis")
                    if local_index == 0:
                        parent_rotation = _matrix_product(
                            _matrix_transpose(parent_inertial_rotation), source_body_rotation
                        )
                        anchor_world = _myosim_world_joint_anchor(
                            source_body, source_position, f"MyoSim joint {source_joint.get('id')}"
                        )
                        parent_anchor = _myosim_matrix_vector(
                            _matrix_transpose(parent_inertial_rotation),
                            _myosim_subtract(anchor_world, parent_com),
                        )
                    else:
                        parent_rotation = _myosim_identity()
                        parent_anchor = source_position
                    if terminal:
                        child_rotation = _matrix_product(
                            _matrix_transpose(child_inertial_rotation), source_body_rotation
                        )
                        child_anchor = _myosim_body_local_from_body_frame(
                            source_body, source_position, f"MyoSim joint {source_joint.get('id')}"
                        )
                    else:
                        child_rotation = _myosim_identity()
                        child_anchor = source_position
                    source_type = source_joint.get("type")
                    joint_type = _MR_JOINT_PRISMATIC if source_type == 2 else _MR_JOINT_REVOLUTE
                    q_offset = len(default_q)
                    v_offset = len(default_v)
                    source_q_index = source_joint.get("qpos_address")
                    if not isinstance(source_q_index, int) or not 0 <= source_q_index < len(source_qpos):
                        raise ImportError(f"MyoSim joint {source_joint.get('id')} has invalid qpos address")
                    default_q.append(source_qpos[source_q_index])
                    default_v.append(0.0)
                    source_range = source_joint.get("range")
                    limited = bool(source_joint.get("limited"))
                    core_limit_status = "source_unlimited"
                    if limited:
                        if not isinstance(source_range, list) or len(source_range) != 2:
                            raise ImportError(f"MyoSim joint {source_joint.get('id')} has invalid range")
                        limits = [_finite_scalar(source_range[0], "MyoSim joint lower range"),
                                  _finite_scalar(source_range[1], "MyoSim joint upper range"), 0.0, 0.0]
                        if limits[0] > limits[1]:
                            raise ImportError(f"MyoSim joint {source_joint.get('id')} has inverted source range")
                        if limits[0] <= default_q[-1] <= limits[1]:
                            flags = _MR_DOF_POSITION_LIMIT
                            core_limit_status = "enforced"
                        else:
                            # MuJoCo admits this authored reset even when it sits
                            # beyond a hard range; Core's state validator does not.
                            # Keep the exact source range in the manifest and do
                            # not silently shift the source default or invent a
                            # different clamp.
                            limits = [0.0, 0.0, 0.0, 0.0]
                            flags = 0
                            core_limit_status = "retained_in_manifest_not_enforced_at_source_default"
                    else:
                        limits = [0.0, 0.0, 0.0, 0.0]
                        flags = 0
                    joint_index = len(joint_records)
                    joint_records.append(
                        _myosim_pack_joint_record(
                            parent_body=prior_core, child_body=child_core, joint_type=joint_type,
                            q_offset=q_offset, nq=1, v_offset=v_offset, nv=1,
                            axis=axis, parent_anchor=parent_anchor, child_anchor=child_anchor,
                            parent_rotation=parent_rotation, child_rotation=child_rotation,
                            context=f"MyoSim joint {source_joint.get('id')}",
                        )
                    )
                    dof_records.append(
                        _myosim_pack_dof_record(
                            joint_index=joint_index, q_index=q_offset, v_index=v_offset, local_dof=0,
                            flags=flags, limits=limits,
                            armature=_finite_scalar(source_joint.get("armature"), "MyoSim joint armature"),
                            damping=_finite_scalar(source_joint.get("damping"), "MyoSim joint damping"),
                            frictionloss=_finite_scalar(
                                source_joint.get("frictionloss"), "MyoSim joint friction loss"
                            ),
                            context=f"MyoSim joint {source_joint.get('id')}",
                        )
                    )
                    body_nodes.append(
                        {"name": (str(source_body.get("name")) if terminal else
                                  f"__myosim_serial_{source_body.get('name')}_{local_index}"),
                         "source_body": source_body if terminal else None,
                         "parent": prior_core, "inbound": joint_index, "virtual": not terminal}
                    )
                    source_joint_map.append(
                        {"source_joint_id": source_joint.get("id"), "source_name": source_joint.get("name"),
                         "core_joint_index": joint_index, "core_q_index": q_offset,
                         "core_v_index": v_offset, "source_type": source_type,
                         "source_range": source_range, "source_limited": limited,
                         "source_damping": _finite_scalar(
                             source_joint.get("damping"), "MyoSim joint damping"
                         ),
                         "source_frictionloss": _finite_scalar(
                             source_joint.get("frictionloss"), "MyoSim joint friction loss"
                         ),
                         "core_limit_status": core_limit_status}
                    )
                    prior_core = child_core
                source_body_to_core[source_body_id] = prior_core
            unresolved.remove(source_body_id)
            progressed = True
        if not progressed:
            raise ImportError("MyoSim source body graph is disconnected or cyclic")

    if len(joint_records) + 1 != len(body_nodes):
        raise ImportError("MyoSim Core tree did not assign one inbound joint to every non-root node")
    body_records = [
        _myosim_pack_body_record(
            parent_body=node["parent"], inbound_joint=node["inbound"], source_body=node["source_body"],
            virtual=bool(node["virtual"]), context=f"MyoSim body {node['name']}"
        )
        for node in body_nodes
    ]
    nq, nv = len(default_q), len(default_v)
    if nq != nv + 1 or nq != model.get("nq"):
        raise ImportError("MyoSim Core lowerer did not retain the source floating configuration dimensions")
    if nv != model.get("nv"):
        raise ImportError("MyoSim Core lowerer did not retain the source velocity dimensions")

    source_joint_to_core = {
        int(record["source_joint_id"]): record for record in source_joint_map
    }
    equality_records: list[bytes] = []
    equality_manifest: list[dict[str, Any]] = []
    for equality in joint_equalities:
        if not isinstance(equality, dict):
            raise ImportError("MyoSim joint equality record is malformed")
        source_dependent = equality.get("dependent_joint")
        source_master = equality.get("master_joint")
        dependent = source_joint_to_core.get(source_dependent)
        master = source_joint_to_core.get(source_master) if source_master != -1 else None
        if dependent is None or (source_master != -1 and master is None):
            raise ImportError("MyoSim joint equality has an unresolved Core coordinate")
        coefficients = equality.get("polycoef")
        solref = equality.get("solref")
        solimp = equality.get("solimp")
        if not isinstance(coefficients, list) or len(coefficients) != 5:
            raise ImportError("MyoSim joint equality polynomial is malformed")
        if not isinstance(solref, list) or len(solref) != 2:
            raise ImportError("MyoSim joint equality solref is malformed")
        if not isinstance(solimp, list) or len(solimp) != 5:
            raise ImportError("MyoSim joint equality solimp is malformed")
        dependent_q = int(dependent["core_q_index"])
        dependent_v = int(dependent["core_v_index"])
        master_q = 0xFFFFFFFF if master is None else int(master["core_q_index"])
        master_v = 0xFFFFFFFF if master is None else int(master["core_v_index"])
        references_and_coefficients0 = [
            _finite_scalar(equality.get("dependent_reference"), "MyoSim equality dependent reference"),
            _finite_scalar(equality.get("master_reference"), "MyoSim equality master reference"),
            *[_finite_scalar(value, "MyoSim equality coefficient") for value in coefficients[:2]],
        ]
        coefficients1 = [
            *[_finite_scalar(value, "MyoSim equality coefficient") for value in coefficients[2:]],
            0.0,
        ]
        equality_records.append(struct.pack(
            "<4I20f",
            dependent_q, dependent_v, master_q, master_v,
            *references_and_coefficients0,
            *coefficients1,
            *[_finite_scalar(value, "MyoSim equality solref") for value in solref], 0.0, 0.0,
            *[_finite_scalar(value, "MyoSim equality solimp") for value in solimp[:4]],
            _finite_scalar(solimp[4], "MyoSim equality solimp"), 0.0, 0.0, 0.0,
        ))
        equality_manifest.append({
            "source_equality_id": equality.get("id"),
            "name": equality.get("name"),
            "dependent_source_joint": source_dependent,
            "dependent_name": dependent["source_name"],
            "dependent_core_q": dependent_q,
            "dependent_core_v": dependent_v,
            "master_source_joint": source_master,
            "master_name": None if master is None else master["source_name"],
            "master_core_q": None if master is None else master_q,
            "master_core_v": None if master is None else master_v,
            "dependent_reference": references_and_coefficients0[0],
            "master_reference": references_and_coefficients0[1],
            "polycoef": [float(value) for value in coefficients],
            "solref": [float(value) for value in solref],
            "solimp": [float(value) for value in solimp],
        })
    if len(equality_records) != len(joint_equalities):
        raise ImportError("MyoSim joint equality lowering is incomplete")
    equality_header = struct.pack(
        "<8s10I32s",
        _MYOSIM_JOINT_EQUALITY_MAGIC, _MYOSIM_JOINT_EQUALITY_ABI,
        nq, nv, len(equality_records), _MYOSIM_JOINT_EQUALITY_RECORD_BYTES,
        len(joint_equalities), 0, 0, 0, 0, bytes.fromhex(source_hash),
    )
    equality_payload = b"".join([equality_header, *equality_records])
    if len(equality_payload) != 80 + _MYOSIM_JOINT_EQUALITY_RECORD_BYTES * len(equality_records):
        raise ImportError("internal MyoSim joint equality payload ABI size mismatch")
    world_gravity = _myosim_vector(model.get("gravity_m_s2"), "MyoSim gravity")
    timestep = _finite_scalar(model.get("timestep_seconds"), "MyoSim timestep")
    if timestep <= 0.0:
        raise ImportError("MyoSim source timestep must be positive")
    world = struct.pack(
        "<16I8f",
        _MR_ENGINE_ABI_VERSION, len(body_records), 1, len(joint_records),
        0, 0, nq, nv,
        1, 1, 1, 1,
        0, 0, 0, 0,
        *world_gravity, timestep,
        1.0e-8, 1.0e-9, 2.0, 1.0e-4,
    )
    articulation = struct.pack(
        "<12I", 0, _MR_ROOT_FLOATING, 0, len(body_records), 0, len(joint_records),
        0, nq, 0, nv, 0, 0
    )
    source_body_ids = sorted(int(identifier) for identifier in body_by_id)
    source_to_core = [source_body_to_core[identifier] for identifier in source_body_ids]
    source_body_records = [
        {
            "source_body_id": identifier,
            "name": body_by_id[identifier].get("name"),
            "core_body_index": source_body_to_core[identifier],
            "default_com_position_world_m": _myosim_vector(
                body_by_id[identifier].get("default_com_position_world_m"),
                "MyoSim source body COM",
            ),
            "default_inertial_quaternion_world_xyzw": list(
                body_by_id[identifier].get("default_inertial_quaternion_world_xyzw", [])
            ),
        }
        for identifier in source_body_ids
    ]
    expected_poses = []
    for record in source_body_records:
        expected_poses.extend(record["default_com_position_world_m"])
        quaternion = record["default_inertial_quaternion_world_xyzw"]
        _myosim_matrix_from_quaternion_xyzw(quaternion)
        expected_poses.extend(_finite_scalar(value, "MyoSim source pose quaternion") for value in quaternion)
    header = struct.pack(
        "<8s10I32s",
        _MYOSIM_CORE_REFERENCE_MAGIC, _MYOSIM_CORE_REFERENCE_ABI, _MR_ENGINE_ABI_VERSION,
        len(source_body_ids), len(body_records), len(joint_records), nq, nv, 0,
        sum(1 for node in body_nodes if node["virtual"]), 0, bytes.fromhex(source_hash),
    )
    rigid_payload = b"".join([
        header, world, articulation, *body_records, *joint_records, *dof_records,
        struct.pack(f"<{nq}f", *default_q), struct.pack(f"<{nv}f", *default_v),
        struct.pack(f"<{len(source_to_core)}I", *source_to_core),
        struct.pack(f"<{len(expected_poses)}f", *expected_poses),
    ])
    expected_rigid_bytes = (
        80 + 96 + 48 + 160 * len(body_records) + 144 * len(joint_records) +
        64 * nv + 4 * (nq + nv) + 4 * len(source_to_core) + 28 * len(source_to_core)
    )
    if len(rigid_payload) != expected_rigid_bytes:
        raise ImportError(
            "internal MyoSim Core rigid payload ABI size mismatch "
            f"({len(rigid_payload)} != {expected_rigid_bytes})"
        )

    site_by_id = {site.get("id"): site for site in sites if isinstance(site, dict)}
    geom_by_id = {geom.get("id"): geom for geom in geometries if isinstance(geom, dict)}
    used_site_ids = sorted({
        int(node.get("source_id"))
        for muscle in muscles for node in muscle.get("route", [])
        if node.get("kind") == "site"
    } | {
        int(node.get("side_site_source_id"))
        for muscle in muscles for node in muscle.get("route", [])
        if isinstance(node.get("side_site_source_id"), int) and int(node.get("side_site_source_id")) >= 0
    })
    used_geom_ids = sorted({
        int(node.get("source_id"))
        for muscle in muscles for node in muscle.get("route", [])
        if node.get("kind") in {"sphere", "cylinder"}
    })
    if any(identifier not in site_by_id for identifier in used_site_ids):
        raise ImportError("MyoSim muscle route references an absent source site")
    if any(identifier not in geom_by_id for identifier in used_geom_ids):
        raise ImportError("MyoSim muscle route references an absent wrap geometry")
    site_index = {identifier: index for index, identifier in enumerate(used_site_ids)}
    geom_index = {identifier: index for index, identifier in enumerate(used_geom_ids)}
    site_records: list[bytes] = []
    for identifier in used_site_ids:
        site = site_by_id[identifier]
        source_body_id = site.get("body")
        if source_body_id not in source_body_to_core:
            raise ImportError("MyoSim site has unresolved source body")
        local = _myosim_body_local_from_body_frame(
            body_by_id[source_body_id], _myosim_vector(site.get("position_body_m"), "MyoSim site position"),
            f"MyoSim site {identifier}",
        )
        site_records.append(struct.pack("<I3f", source_body_to_core[source_body_id], *local))
    geom_records: list[bytes] = []
    for identifier in used_geom_ids:
        geometry = geom_by_id[identifier]
        source_body_id = geometry.get("body")
        if source_body_id not in source_body_to_core:
            raise ImportError("MyoSim wrap geometry has unresolved source body")
        body = body_by_id[source_body_id]
        local = _myosim_body_local_from_body_frame(
            body, _myosim_vector(geometry.get("position_body_m"), "MyoSim wrap position"),
            f"MyoSim wrap {identifier}",
        )
        body_rotation = _myosim_matrix_from_quaternion_xyzw(
            list(body.get("inertial_quaternion_body_xyzw", []))
        )
        geometry_rotation = _myosim_matrix_from_quaternion_xyzw(
            list(geometry.get("quaternion_body_xyzw", []))
        )
        core_rotation = _matrix_product(_matrix_transpose(body_rotation), geometry_rotation)
        route_type = _MYOSIM_ROUTE_SPHERE if geometry.get("type") == 2 else _MYOSIM_ROUTE_CYLINDER
        if route_type == _MYOSIM_ROUTE_CYLINDER and geometry.get("type") != 5:
            raise ImportError(f"MyoSim wrap {identifier} has unsupported source geometry type")
        geom_records.append(
            struct.pack(
                "<2I14f", source_body_to_core[source_body_id], route_type,
                _finite_scalar(geometry.get("radius_m"), "MyoSim wrap radius"), 0.0,
                *local, *[value for row in core_rotation for value in row],
            )
        )
    route_records: list[bytes] = []
    muscle_records: list[bytes] = []
    architecture_records: list[bytes] = []
    muscle_manifest: list[dict[str, Any]] = []
    for muscle_index, muscle in enumerate(muscles):
        route = muscle.get("route")
        if not isinstance(route, list) or len(route) < 2:
            raise ImportError(f"MyoSim muscle {muscle.get('name')} has no spatial route")
        route_offset = len(route_records)
        for node in route:
            kind = node.get("kind")
            if kind == "site":
                route_type, target = _MYOSIM_ROUTE_SITE, site_index.get(node.get("source_id"))
            elif kind == "sphere":
                route_type, target = _MYOSIM_ROUTE_SPHERE, geom_index.get(node.get("source_id"))
            elif kind == "cylinder":
                route_type, target = _MYOSIM_ROUTE_CYLINDER, geom_index.get(node.get("source_id"))
            else:
                raise ImportError(f"MyoSim muscle {muscle.get('name')} has unsupported route node")
            if target is None:
                raise ImportError(f"MyoSim muscle {muscle.get('name')} route target was not lowered")
            side = node.get("side_site_source_id", -1)
            side_index = 0xFFFFFFFF if side == -1 else site_index.get(side)
            if side_index is None:
                raise ImportError(f"MyoSim muscle {muscle.get('name')} side site was not lowered")
            route_records.append(struct.pack("<4I", route_type, target, side_index, 0))
        length_range = muscle.get("length_range_m")
        control_range = muscle.get("control_range")
        gain = muscle.get("gain_parameters")
        bias = muscle.get("bias_parameters")
        dynamics = muscle.get("dynamic_parameters")
        if not all(isinstance(values, list) for values in (length_range, control_range, gain, bias, dynamics)):
            raise ImportError(f"MyoSim muscle {muscle.get('name')} has incomplete source parameters")
        if len(length_range) != 2 or len(control_range) != 2 or any(len(values) != 10 for values in (gain, bias, dynamics)):
            raise ImportError(f"MyoSim muscle {muscle.get('name')} has invalid source parameter dimensions")
        floats = [
            *[_finite_scalar(value, "MyoSim muscle length range") for value in length_range],
            _finite_scalar(muscle.get("acceleration_scale"), "MyoSim muscle acceleration scale"),
            *[_finite_scalar(value, "MyoSim muscle control range") for value in control_range],
            *[_finite_scalar(value, "MyoSim muscle gain") for value in gain],
            *[_finite_scalar(value, "MyoSim muscle bias") for value in bias],
            *[_finite_scalar(value, "MyoSim muscle dynamics") for value in dynamics],
            _finite_scalar(muscle.get("oracle_length_m"), "MyoSim muscle source length oracle"),
            _finite_scalar(muscle.get("oracle_force_n_at_activation_0_5"), "MyoSim muscle source force oracle"),
        ]
        if len(floats) != 37:
            raise ImportError("internal MyoSim muscle record field count mismatch")
        muscle_records.append(
            struct.pack("<4I37f", int(muscle.get("tendon")), route_offset, len(route), 0, *floats)
        )
        optimal_fiber_length, tendon_slack_length, fit_normalized_rmse = (
            _fit_myosim_compliant_architecture(
                [float(value) for value in length_range],
                float(muscle.get("acceleration_scale")),
                [float(value) for value in gain],
                [float(value) for value in bias],
                float(muscle.get("oracle_length_m")),
            )
        )
        architecture_records.append(struct.pack(
            _MYOSIM_MUSCLE_ARCHITECTURE_FORMAT,
            optimal_fiber_length,
            tendon_slack_length,
            0.049,
            1.375 / 0.049,
            2.0 / 3.0,
            0.5,
            0.1,
            fit_normalized_rmse,
        ))
        muscle_manifest.append({
            "source_actuator_index": muscle.get("id"), "name": muscle.get("name"),
            "source_tendon_index": muscle.get("tendon"), "route_nodes": len(route),
            "oracle_length_m": floats[-2], "oracle_force_n_at_activation_0_5": floats[-1],
            "compliant_architecture": {
                "optimal_fiber_length_m": optimal_fiber_length,
                "tendon_slack_length_m": tendon_slack_length,
                "pennation_angle_rad": 0.0,
                "fit_normalized_rmse": fit_normalized_rmse,
            },
        })
    muscle_header = struct.pack(
        "<8s9I32s", _MYOSIM_MUSCLE_REFERENCE_MAGIC, _MYOSIM_MUSCLE_REFERENCE_ABI,
        len(body_records), len(muscle_records), len(site_records), len(geom_records), len(route_records),
        int(model.get("tendon_count")), len(architecture_records),
        _MYOSIM_MUSCLE_ARCHITECTURE_BYTES, bytes.fromhex(source_hash)
    )
    muscle_payload = b"".join([
        muscle_header, *site_records, *geom_records, *route_records,
        *muscle_records, *architecture_records,
    ])
    expected_muscle_bytes = _myosim_muscle_payload_bytes(
        len(site_records), len(geom_records), len(route_records), len(muscle_records),
        len(architecture_records), _MYOSIM_MUSCLE_ARCHITECTURE_BYTES,
    )
    if len(muscle_payload) != expected_muscle_bytes:
        raise ImportError("internal MyoSim muscle payload ABI size mismatch")
    support_source = exported.get("support_contact")
    if not isinstance(support_source, dict):
        raise ImportError("MyoSim export has no authored support-contact records")
    ground = support_source.get("ground")
    support_geometries = support_source.get("geometries")
    if not isinstance(ground, dict) or not isinstance(support_geometries, list):
        raise ImportError("MyoSim support-contact export is malformed")
    ground_point = _myosim_vector(ground.get("point_world_m"), "MyoSim support ground point")
    ground_normal = _myosim_vector(ground.get("normal_world"), "MyoSim support ground normal")
    normal_length = math.sqrt(sum(component * component for component in ground_normal))
    if not 0.999 <= normal_length <= 1.001:
        raise ImportError("MyoSim support ground normal is not unit length")
    ground_normal = [component / normal_length for component in ground_normal]
    ground_friction = _finite_scalar(
        ground.get("friction_tangential"), "MyoSim support ground friction"
    )
    if ground_friction < 0.0:
        raise ImportError("MyoSim support ground friction is negative")
    support_records: list[bytes] = []
    support_manifest: list[dict[str, Any]] = []
    expected_support_names = {
        f"foot_col{ordinal}_{side}"
        for ordinal in (1, 3, 4)
        for side in ("r", "l")
    } | {
        f"bofoot_col{ordinal}_{side}"
        for ordinal in (1, 2)
        for side in ("r", "l")
    }
    observed_support_names: set[str] = set()
    for geometry in support_geometries:
        if not isinstance(geometry, dict):
            raise ImportError("MyoSim support geometry record is malformed")
        name = geometry.get("name")
        source_body_id = geometry.get("body")
        source_geom_id = geometry.get("id")
        if not isinstance(name, str) or not isinstance(source_body_id, int) or not isinstance(source_geom_id, int):
            raise ImportError("MyoSim support geometry lacks source identity")
        if name in observed_support_names:
            raise ImportError(f"MyoSim support geometry {name} is duplicated")
        observed_support_names.add(name)
        if source_body_id not in source_body_to_core:
            raise ImportError(f"MyoSim support geometry {name} has unresolved source body")
        local_point = _myosim_vector(
            geometry.get("support_point_local_com_m"), f"MyoSim support geometry {name} local point"
        )
        world_point = _myosim_vector(
            geometry.get("support_point_world_m"), f"MyoSim support geometry {name} world point"
        )
        signed_distance = _finite_scalar(
            geometry.get("default_signed_plane_distance_m"),
            f"MyoSim support geometry {name} signed distance",
        )
        derived_distance = sum(
            (world_point[axis] - ground_point[axis]) * ground_normal[axis]
            for axis in range(3)
        )
        if abs(derived_distance - signed_distance) > 1.0e-5:
            raise ImportError(f"MyoSim support geometry {name} plane-distance record drifted")
        # The static-world witness is the exact projection onto the compiled
        # source plane.  The signed separation remains in the manifest for a
        # caller to decide whether a support row should be admitted.
        plane_witness = [
            world_point[axis] - signed_distance * ground_normal[axis]
            for axis in range(3)
        ]
        friction = _finite_scalar(
            geometry.get("friction_tangential"), f"MyoSim support geometry {name} friction"
        )
        if friction < 0.0:
            raise ImportError(f"MyoSim support geometry {name} friction is negative")
        support_records.append(
            struct.pack(
                "<2I10f",
                source_body_to_core[source_body_id], source_geom_id,
                *local_point, *plane_witness,
                friction, signed_distance, 0.0, 0.0,
            )
        )
        support_manifest.append({
            "source_geom_id": source_geom_id,
            "name": name,
            "source_body_id": source_body_id,
            "source_body_name": geometry.get("body_name"),
            "core_body_index": source_body_to_core[source_body_id],
            "shape": geometry.get("type_name"),
            "source_size_m": geometry.get("size_m"),
            "default_signed_plane_distance_m": signed_distance,
            "friction_tangential": friction,
        })
    if observed_support_names != expected_support_names:
        raise ImportError(
            "MyoSim support geometry set drifted: expected " +
            ", ".join(sorted(expected_support_names)) + "; found " +
            ", ".join(sorted(observed_support_names))
        )
    support_manifest.sort(key=lambda record: str(record["name"]))
    # Preserve the same deterministic source-name order in the binary.
    order = {record["name"]: index for index, record in enumerate(support_manifest)}
    support_records = [
        record for _, record in sorted(
            zip((str(geometry.get("name")) for geometry in support_geometries), support_records),
            key=lambda item: order[item[0]],
        )
    ]
    support_header = struct.pack(
        "<8s4I32s7f",
        _MYOSIM_SUPPORT_CONTACT_MAGIC, _MYOSIM_SUPPORT_CONTACT_ABI,
        len(body_records), len(support_records), 0, bytes.fromhex(source_hash),
        *ground_point, *ground_normal, ground_friction,
    )
    support_payload = b"".join([support_header, *support_records])
    expected_support_bytes = 84 + 48 * len(support_records)
    if len(support_payload) != expected_support_bytes:
        raise ImportError("internal MyoSim support-contact payload ABI size mismatch")

    manifest = {
        "schema": "numi.human.myosim-fullbody-reference.v1",
        "source": source,
        "model": {
            "name": model.get("name"), "source_body_count": len(source_body_ids),
            "source_joint_count": len(joints), "source_nq": model.get("nq"), "source_nv": model.get("nv"),
            "source_muscle_count": len(muscles), "source_tendon_count": model.get("tendon_count"),
        },
        "core_tree": {
            "root": "floating_source_com", "engine_body_count": len(body_records),
            "zero_inertia_serial_transform_carrier_count": sum(1 for node in body_nodes if node["virtual"]),
            "joint_count": len(joint_records), "nq": nq, "nv": nv,
            "source_joint_map": source_joint_map,
            "body_order": [node["name"] for node in body_nodes],
            "source_body_records": source_body_records,
        },
        "payloads": {
            "rigid": {"file": "myosim-fullbody-core-reference.nhrigid", "bytes": len(rigid_payload),
                      "sha256": hashlib.sha256(rigid_payload).hexdigest(), "payload_abi": _MYOSIM_CORE_REFERENCE_ABI},
            "muscles": {"file": "myosim-fullbody-muscle-reference.nhmyo", "bytes": len(muscle_payload),
                        "sha256": hashlib.sha256(muscle_payload).hexdigest(), "payload_abi": _MYOSIM_MUSCLE_REFERENCE_ABI},
            "support_contact": {
                "file": "myosim-fullbody-support-contact.nhcnt",
                "bytes": len(support_payload),
                "sha256": hashlib.sha256(support_payload).hexdigest(),
                "payload_abi": _MYOSIM_SUPPORT_CONTACT_ABI,
            },
            "joint_equalities": {
                "file": "myosim-fullbody-joint-equalities.nheq",
                "bytes": len(equality_payload),
                "sha256": hashlib.sha256(equality_payload).hexdigest(),
                "payload_abi": _MYOSIM_JOINT_EQUALITY_ABI,
            },
        },
        "joint_equalities": {
            "count": len(equality_manifest),
            "records": equality_manifest,
            "semantics": (
                "Exact active MuJoCo scalar joint equalities. Each dependent coordinate is a "
                "quartic polynomial of its optional master about the source qpos0 references."
            ),
        },
        "support_contact": {
            "source_ground": {
                "source_geom_id": ground.get("source_geom_id"),
                "name": ground.get("name"),
                "point_world_m": ground_point,
                "normal_world": ground_normal,
                "friction_tangential": ground_friction,
            },
            "support_geometry_count": len(support_manifest),
            "support_geometries": support_manifest,
            "scope": "source-default stance support witnesses only",
            "boundary": (
                "Each record is an exact compiled MyoSim foot capsule/ellipsoid surface witness "
                "against MyoSim's compiled plane. It is admissible to a bounded unilateral support "
                "contact solve, but is not BodyParts3D collider registration, general collision detection, "
                "compliance calibration, or gait validation."
            ),
        },
        "route_coverage": {
            "muscle_count": len(muscle_records), "route_site_count": len(site_records),
            "wrap_geometry_count": len(geom_records),
            "wrap_geometry_kinds": {"sphere": sum(1 for record in geom_records if struct.unpack("<2I", record[:8])[1] == _MYOSIM_ROUTE_SPHERE),
                                    "cylinder": sum(1 for record in geom_records if struct.unpack("<2I", record[:8])[1] == _MYOSIM_ROUTE_CYLINDER)},
        },
        "compliant_muscle_architecture": {
            "record_count": len(architecture_records),
            "record_bytes": _MYOSIM_MUSCLE_ARCHITECTURE_BYTES,
            "law": "one normalized damped equilibrium tendon law; per-muscle positive L0/LT fit",
            "tendon_strain_at_one_normalized_force": 0.049,
            "tendon_stiffness_at_one_normalized_force": 1.375 / 0.049,
            "tendon_normalized_force_at_toe_end": 2.0 / 3.0,
            "tendon_curviness": 0.5,
            "normalized_fiber_damping": 0.1,
            "pennation_angle_rad": 0.0,
            "fit_normalized_rmse": {
                "maximum": max(record["compliant_architecture"]["fit_normalized_rmse"] for record in muscle_manifest),
                "mean": sum(record["compliant_architecture"]["fit_normalized_rmse"] for record in muscle_manifest) / len(muscle_manifest),
            },
            "boundary": (
                "MyoSim does not identify pennation or elastic tendon architecture. L0/LT are bounded offline "
                "fits to its retained static force surface; the normalized tendon curve and damping are shared "
                "Rajagopal/OpenSim-derived defaults, not per-muscle anatomical measurements."
            ),
        },
        "muscles": muscle_manifest,
        "runtime_requirement": (
            "Load both payloads with metalrobo_numilab_human_myosim_reference_probe. "
            "The native CPU reference preserves MuJoCo site/sphere/cylinder route geometry and "
            "the source muscle activation, active-force, and passive-force parameter records."
        ),
        "evidence_boundary": (
            "This is a source-complete native CPU-reference import. The optional support payload carries "
            "authored MyoSim default-stance foot witnesses, not a general collision world. Skin/organ "
            "mechanics, Mortensen neck registration, and a complete device-resident rollout remain separate deliverables."
        ),
    }
    return manifest, rigid_payload, muscle_payload, support_payload, equality_payload


# The fitted landmarks remain deliberately conservative: a source mesh vertex
# centroid is not an inertial COM.  The remaining entries below are exact,
# source-named meshes bound to an existing MyoSim articulated body through that
# common frame.  They expand the *visual* skeleton without changing the fitted
# transform or claiming collision, skin, or soft-tissue mechanics.
_BODYPARTS_MYOSIM_FIT_BONE_ANCHORS = (
    {"myosim_body": "sacrum", "bodyparts_name": "sacrum", "hierarchy": "is_a", "member_id": "FJ3393"},
    {"myosim_body": "femur_r", "bodyparts_name": "right femur", "hierarchy": "is_a", "member_id": "FJ3365"},
    {"myosim_body": "femur_l", "bodyparts_name": "left femur", "hierarchy": "is_a", "member_id": "FJ3259"},
    {"myosim_body": "tibia_r", "bodyparts_name": "right tibia", "hierarchy": "is_a", "member_id": "FJ3387"},
    {"myosim_body": "tibia_l", "bodyparts_name": "left tibia", "hierarchy": "is_a", "member_id": "FJ3282"},
    {"myosim_body": "calcn_r", "bodyparts_name": "right calcaneus", "hierarchy": "is_a", "member_id": "FJ3360"},
    {"myosim_body": "calcn_l", "bodyparts_name": "left calcaneus", "hierarchy": "is_a", "member_id": "FJ3256"},
    {"myosim_body": "clavicle_r", "bodyparts_name": "right clavicle", "hierarchy": "is_a", "member_id": "FJ3362"},
    {"myosim_body": "clavicle_l", "bodyparts_name": "left clavicle", "hierarchy": "is_a", "member_id": "FJ3237"},
    {"myosim_body": "scapula_r", "bodyparts_name": "right scapula", "hierarchy": "is_a", "member_id": "FJ3384"},
    {"myosim_body": "scapula_l", "bodyparts_name": "left scapula", "hierarchy": "is_a", "member_id": "FJ3279"},
    {"myosim_body": "humerus_r", "bodyparts_name": "right humerus", "hierarchy": "is_a", "member_id": "FJ3368"},
    {"myosim_body": "humerus_l", "bodyparts_name": "left humerus", "hierarchy": "is_a", "member_id": "FJ3262"},
    {"myosim_body": "ulna_r", "bodyparts_name": "right ulna", "hierarchy": "is_a", "member_id": "FJ3391"},
    {"myosim_body": "ulna_l", "bodyparts_name": "left ulna", "hierarchy": "is_a", "member_id": "FJ3286"},
    {"myosim_body": "radius_r", "bodyparts_name": "right radius", "hierarchy": "is_a", "member_id": "FJ3349"},
    {"myosim_body": "radius_l", "bodyparts_name": "left radius", "hierarchy": "is_a", "member_id": "FJ3277"},
)


def _bodyparts_visual_only_bone(
    myosim_body: str, bodyparts_name: str, member_id: str,
) -> dict[str, str | bool]:
    return {
        "myosim_body": myosim_body,
        "bodyparts_name": bodyparts_name,
        "hierarchy": "is_a",
        "member_id": member_id,
        "registration_anchor": False,
    }


_BODYPARTS_MYOSIM_MAJOR_BONE_EXTENSIONS = (
    # These exact meshes have unambiguous named source members and a sound
    # MyoSim rigid-link parent, but they remain out of the fitted landmark set:
    # their OBJ vertex centroids are especially poor proxies for the target
    # inertial COMs.  They inherit the established 17-anchor common frame.
    {"myosim_body": "pelvis", "bodyparts_name": "right hip bone", "hierarchy": "is_a", "member_id": "FJ3152", "registration_anchor": False},
    {"myosim_body": "pelvis", "bodyparts_name": "left hip bone", "hierarchy": "is_a", "member_id": "FJ3288", "registration_anchor": False},
    {"myosim_body": "tibia_r", "bodyparts_name": "right fibula", "hierarchy": "is_a", "member_id": "FJ3366", "registration_anchor": False},
    {"myosim_body": "tibia_l", "bodyparts_name": "left fibula", "hierarchy": "is_a", "member_id": "FJ3260", "registration_anchor": False},
    {"myosim_body": "talus_r", "bodyparts_name": "right talus", "hierarchy": "is_a", "member_id": "FJ3385", "registration_anchor": False},
    {"myosim_body": "talus_l", "bodyparts_name": "left talus", "hierarchy": "is_a", "member_id": "FJ3280", "registration_anchor": False},
    {"myosim_body": "patella_r", "bodyparts_name": "right patella", "hierarchy": "is_a", "member_id": "FJ3381", "registration_anchor": False},
    {"myosim_body": "patella_l", "bodyparts_name": "left patella", "hierarchy": "is_a", "member_id": "FJ3275", "registration_anchor": False},
    {"myosim_body": "torso", "bodyparts_name": "body of sternum", "hierarchy": "is_a", "member_id": "FJ3178", "registration_anchor": False},
)


_BODYPARTS_MYOSIM_CRANIAL_EXTENSIONS = tuple(
    _bodyparts_visual_only_bone("head", name, member)
    for name, member in (
        # FJ1282, used by the retired first visual binding, is an ocular
        # component incidentally listed under the broad ``skull`` part_of
        # hierarchy.  Bind only source meshes whose is_a terms identify the
        # actual cranial or mandibular bones.
        ("right parietal bone", "FJ3380"),
        ("left parietal bone", "FJ3274"),
        ("right temporal bone", "FJ3386"),
        ("left temporal bone", "FJ3281"),
        ("frontal bone", "FJ3200"),
        ("occipital bone", "FJ3309"),
        ("sphenoid bone", "FJ3394"),
        ("mandible", "FJ3289"),
    )
)


_BODYPARTS_MYOSIM_THORACIC_FOOT_EXTENSIONS = tuple(
    _bodyparts_visual_only_bone(body, name, member)
    for body, name, member in (
        # MyoSim uses one torso body for the thorax.  These remain individual
        # BodyParts3D source meshes, each carried by that live torso parent.
        ("torso", "right first rib", "FJ3334"),
        ("torso", "right second rib", "FJ3336"),
        ("torso", "right third rib", "FJ3338"),
        ("torso", "right fourth rib", "FJ3340"),
        ("torso", "right fifth rib", "FJ3342"),
        ("torso", "right sixth rib", "FJ3344"),
        ("torso", "right seventh rib", "FJ3346"),
        ("torso", "right eighth rib", "FJ3347"),
        ("torso", "right ninth rib", "FJ3348"),
        ("torso", "right tenth rib", "FJ3330"),
        ("torso", "right eleventh rib", "FJ3331"),
        ("torso", "right twelfth rib", "FJ3332"),
        ("torso", "left first rib", "FJ3228"),
        ("torso", "left second rib", "FJ3229"),
        ("torso", "left third rib", "FJ3230"),
        ("torso", "left fourth rib", "FJ3231"),
        ("torso", "left fifth rib", "FJ3232"),
        ("torso", "left sixth rib", "FJ3233"),
        ("torso", "left seventh rib", "FJ3234"),
        ("torso", "left eighth rib", "FJ3235"),
        ("torso", "left ninth rib", "FJ3236"),
        ("torso", "left tenth rib", "FJ3225"),
        ("torso", "left eleventh rib", "FJ3226"),
        ("torso", "left twelfth rib", "FJ3227"),
        # Rajagopal's ``calcn`` segment is the rigid foot, not the calcaneus
        # alone. Its MTP joint separates that segment from the collective toe
        # segment, so the midfoot must not spuriously follow toe flexion.
        ("calcn_r", "navicular bone of right foot", "FJ3308"),
        ("calcn_r", "right medial cuneiform bone", "FJ3377"),
        ("calcn_r", "right intermediate cuneiform bone", "FJ3370"),
        ("calcn_r", "right lateral cuneiform bone", "FJ3373"),
        ("calcn_r", "right cuboid bone", "FJ3364"),
        ("calcn_l", "navicular bone of left foot", "FJ3307"),
        ("calcn_l", "left medial cuneiform bone", "FJ3271"),
        ("calcn_l", "left intermediate cuneiform bone", "FJ3264"),
        ("calcn_l", "left lateral cuneiform bone", "FJ3267"),
        ("calcn_l", "left cuboid bone", "FJ3258"),
    )
)


_BODYPARTS_MYOSIM_REMAINING_SOURCE_EXTENSIONS = tuple(
    _bodyparts_visual_only_bone(body, name, member)
    for body, name, member in (
        ("cervical_spine", "atlas", "FJ3176"),
        ("cervical_spine", "axis", "FJ3177"),
        ("triquetrum_r", "right triquetral", "FJ3390"),
        ("triquetrum_l", "left triquetral", "FJ3285"),
    )
)


_BODYPARTS_MYOSIM_WRIST_HAND_EXTENSIONS = tuple(
    _bodyparts_visual_only_bone(body, name, member)
    for body, name, member in (
        # The BodyParts3D v4.0 archive has no separate triquetrum OBJ.  The
        # other seven named wrist bones are present and bind one-to-one with
        # MyoSim hand segments.
        ("scaphoid_r", "right scaphoid", "FJ3383"),
        ("lunate_r", "right lunate", "FJ3374"),
        ("pisiform_r", "right pisiform", "FJ3382"),
        ("capitate_r", "right capitate", "FJ3361"),
        ("trapezium_r", "right trapezium", "FJ3388"),
        ("trapezoid_r", "right trapezoid", "FJ3389"),
        ("hamate_r", "right hamate", "FJ3367"),
        ("scaphoid_l", "left scaphoid", "FJ3278"),
        ("lunate_l", "left lunate", "FJ3268"),
        ("pisiform_l", "left pisiform", "FJ3276"),
        ("capitate_l", "left capitate", "FJ3257"),
        ("trapezium_l", "left trapezium", "FJ3283"),
        ("trapezoid_l", "left trapezoid", "FJ3284"),
        ("hamate_l", "left hamate", "FJ3261"),
        ("firstmc_r", "right first metacarpal bone", "FJ3350"),
        ("secondmc_r", "right second metacarpal bone", "FJ3352"),
        ("thirdmc_r", "right third metacarpal bone", "FJ3354"),
        ("fourthmc_r", "right fourth metacarpal bone", "FJ3356"),
        ("fifthmc_r", "right fifth metacarpal bone", "FJ3358"),
        ("proximal_thumb_r", "proximal phalanx of right thumb", "FJ3327"),
        ("distal_thumb_r", "distal phalanx of right thumb", "FJ3198"),
        ("2proxph_r", "proximal phalanx of right index finger", "FJ3322"),
        ("midph2_r", "middle phalanx of right index finger", "FJ3303"),
        ("distph2_r", "distal phalanx of right index finger", "FJ3193"),
        ("3proxph_r", "proximal phalanx of right middle finger", "FJ3325"),
        ("midph3_r", "middle phalanx of right middle finger", "FJ3306"),
        ("distph3_r", "distal phalanx of right middle finger", "FJ3196"),
        ("4proxph_r", "proximal phalanx of right ring finger", "FJ3326"),
        ("midph4_r", "middle phalanx of right ring finger", "FJ3292"),
        ("distph4_r", "distal phalanx of right ring finger", "FJ3197"),
        ("5proxph_r", "proximal phalanx of right little finger", "FJ3323"),
        ("midph5_r", "middle phalanx of right little finger", "FJ3304"),
        ("distph5_r", "distal phalanx of right little finger", "FJ3194"),
        ("firstmc_l", "left first metacarpal bone", "FJ3240"),
        ("secondmc_l", "left second metacarpal bone", "FJ3243"),
        ("thirdmc_l", "left third metacarpal bone", "FJ3246"),
        ("fourthmc_l", "left fourth metacarpal bone", "FJ3249"),
        ("fifthmc_l", "left fifth metacarpal bone", "FJ3252"),
        ("proximal_thumb_l", "proximal phalanx of left thumb", "FJ3318"),
        ("distal_thumb_l", "distal phalanx of left thumb", "FJ3188"),
        ("2proxph_l", "proximal phalanx of left index finger", "FJ3313"),
        ("midph2_l", "middle phalanx of left index finger", "FJ3296"),
        ("distph2_l", "distal phalanx of left index finger", "FJ3183"),
        ("3proxph_l", "proximal phalanx of left middle finger", "FJ3316"),
        ("midph3_l", "middle phalanx of left middle finger", "FJ3299"),
        ("distph3_l", "distal phalanx of left middle finger", "FJ3186"),
        ("4proxph_l", "proximal phalanx of left ring finger", "FJ3317"),
        ("midph4_l", "middle phalanx of left ring finger", "FJ3291"),
        ("distph4_l", "distal phalanx of left ring finger", "FJ3187"),
        ("5proxph_l", "proximal phalanx of left little finger", "FJ3314"),
        ("midph5_l", "middle phalanx of left little finger", "FJ3297"),
        ("distph5_l", "distal phalanx of left little finger", "FJ3184"),
    )
)


_BODYPARTS_MYOSIM_TOE_EXTENSIONS = tuple(
    _bodyparts_visual_only_bone(body, name, member)
    for body, name, member in (
        # Metatarsals stay on the rigid foot across the one authored MTP joint.
        # All five phalangeal chains share the collective toes body; no
        # independent digital articulation is introduced.
        ("calcn_r", "right first metatarsal bone", "FJ3351"),
        ("calcn_r", "right second metatarsal bone", "FJ3353"),
        ("calcn_r", "right third metatarsal bone", "FJ3355"),
        ("calcn_r", "right fourth metatarsal bone", "FJ3357"),
        ("calcn_r", "right fifth metatarsal bone", "FJ3359"),
        ("toes_r", "proximal phalanx of right big toe", "FJ3310"),
        ("toes_r", "distal phalanx of right big toe", "FJ3192"),
        ("toes_r", "proximal phalanx of right second toe", "FJ3319"),
        ("toes_r", "middle phalanx of right second toe", "FJ3300"),
        ("toes_r", "distal phalanx of right second toe", "FJ3189"),
        ("toes_r", "proximal phalanx of right third toe", "FJ3320"),
        ("toes_r", "middle phalanx of right third toe", "FJ3301"),
        ("toes_r", "distal phalanx of right third toe", "FJ3190"),
        ("toes_r", "proximal phalanx of right fourth toe", "FJ3321"),
        ("toes_r", "middle phalanx of right fourth toe", "FJ3302"),
        ("toes_r", "distal phalanx of right fourth toe", "FJ3191"),
        ("toes_r", "proximal phalanx of right little toe", "FJ3324"),
        ("toes_r", "middle phalanx of right little toe", "FJ3305"),
        ("toes_r", "distal phalanx of right little toe", "FJ3195"),
        ("calcn_l", "left first metatarsal bone", "FJ3241"),
        ("calcn_l", "left second metatarsal bone", "FJ3244"),
        ("calcn_l", "left third metatarsal bone", "FJ3247"),
        ("calcn_l", "left fourth metatarsal bone", "FJ3250"),
        ("calcn_l", "left fifth metatarsal bone", "FJ3253"),
        ("toes_l", "proximal phalanx of left big toe", "FJ3329"),
        ("toes_l", "distal phalanx of left big toe", "FJ3182"),
        ("toes_l", "proximal phalanx of left second toe", "FJ3328"),
        ("toes_l", "middle phalanx of left second toe", "FJ3293"),
        ("toes_l", "distal phalanx of left second toe", "FJ3179"),
        ("toes_l", "proximal phalanx of left third toe", "FJ3311"),
        ("toes_l", "middle phalanx of left third toe", "FJ3294"),
        ("toes_l", "distal phalanx of left third toe", "FJ3180"),
        ("toes_l", "proximal phalanx of left fourth toe", "FJ3312"),
        ("toes_l", "middle phalanx of left fourth toe", "FJ3295"),
        ("toes_l", "distal phalanx of left fourth toe", "FJ3181"),
        ("toes_l", "proximal phalanx of left little toe", "FJ3315"),
        ("toes_l", "middle phalanx of left little toe", "FJ3298"),
        ("toes_l", "distal phalanx of left little toe", "FJ3185"),
    )
)


# Separate digital joints are not required to preserve the visible source toe
# chains and their insertions. Ten exact BodyParts3D phalangeal compounds, five
# per side, are carried by the two existing MyoSim toes bodies. Metatarsals stay
# on the rigid foot. Compilation rejects a missing member, a one-toe identity
# shift, a split phalangeal owner, a mismatched local transform, or a
# disconnected source chain.
_NUMI_HUMAN_TOE_RIGID_CHAINS = {
    "toes_r": (
        ("FJ3310", "FJ3192"),
        ("FJ3319", "FJ3300", "FJ3189"),
        ("FJ3320", "FJ3301", "FJ3190"),
        ("FJ3321", "FJ3302", "FJ3191"),
        ("FJ3324", "FJ3305", "FJ3195"),
    ),
    "toes_l": (
        ("FJ3329", "FJ3182"),
        ("FJ3328", "FJ3293", "FJ3179"),
        ("FJ3311", "FJ3294", "FJ3180"),
        ("FJ3312", "FJ3295", "FJ3181"),
        ("FJ3315", "FJ3298", "FJ3185"),
    ),
}
_NUMI_HUMAN_HALLUX_RIGID_COMPOUNDS = {
    body: chains[0] for body, chains in _NUMI_HUMAN_TOE_RIGID_CHAINS.items()
}
_NUMI_HUMAN_HALLUX_RIGID_COMPOUND_MAXIMUM_GAP_M = 0.001


# Rajagopal/MyoSim authors one EDL and one FDL route per side even though each
# anatomical muscle terminates as four lesser-toe slips.  Keep that source
# route and its single force law authoritative, but make the missing semantic
# correspondence explicit: its terminal wrench is distributed over the exact
# BodyParts3D distal phalanges of toes 2..5.  Hallux routes remain one-to-one.
# Every member on one side shares the same articulated MyoSim toes body, so the
# distribution changes neither the source endpoint nor the rigid-body wrench.
_NUMI_HUMAN_TOE_ENTHESIS_MEMBERS = {
    ("edl_r", 1): ("FJ3189", "FJ3190", "FJ3191", "FJ3195"),
    ("fdl_r", 1): ("FJ3189", "FJ3190", "FJ3191", "FJ3195"),
    ("ehl_r", 1): ("FJ3192",),
    ("fhl_r", 1): ("FJ3192",),
    ("edl_l", 1): ("FJ3179", "FJ3180", "FJ3181", "FJ3185"),
    ("fdl_l", 1): ("FJ3179", "FJ3180", "FJ3181", "FJ3185"),
    ("ehl_l", 1): ("FJ3182",),
    ("fhl_l", 1): ("FJ3182",),
}
_NUMI_HUMAN_TOE_ENTHESIS_MAXIMUM_SPREAD_M = 0.040


# The source mechanics owns a single pelvis rigid body carrying both exact
# BodyParts3D hip members, and one shank body per side carrying exact tibia and
# fibula members.  Refusing every endpoint on those multi-member bodies leaves
# major hip and knee/ankle actuators as point-only laws even when the route
# name, endpoint ordinal, laterality, and source geometry agree.  These tables
# make only that missing member identity explicit. They never move a MyoSim
# endpoint and remain subject to the ordinary 12 mm distance, connected-patch,
# force-amplification, and wrench-conservation gates.
_NUMI_HUMAN_PELVIS_ENTHESIS_BASE_NAMES = (
    "addbrev", "addlong", "addmagDist", "addmagIsch", "addmagMid",
    "addmagProx", "bflh", "glmax1", "glmax2", "glmax3", "glmed1",
    "glmed2", "glmed3", "glmin1", "glmin2", "glmin3", "grac",
    "iliacus", "piri", "psoas", "recfem", "sart", "semimem",
    "semiten", "tfl",
)
_NUMI_HUMAN_SHANK_ENTHESIS_MEMBER_CLASS = {
    # Fibular head/shaft endpoints.
    ("bflh", 1): "fibula",
    ("bfsh", 1): "fibula",
    ("edl", 0): "fibula",
    ("ehl", 0): "fibula",
    ("fhl", 0): "fibula",
    ("perbrev", 0): "fibula",
    ("perlong", 0): "fibula",
    # Tibial plateau, tuberosity, pes anserinus, and shaft endpoints. The
    # source model lumps multi-area soleus/tibialis-posterior origins into one
    # point; the exact authored point is nearer the named tibia member and is
    # retained unchanged.
    ("fdl", 0): "tibia",
    ("grac", 1): "tibia",
    ("recfem", 1): "tibia",
    ("sart", 1): "tibia",
    ("semimem", 1): "tibia",
    ("semiten", 1): "tibia",
    ("soleus", 0): "tibia",
    ("tfl", 1): "tibia",
    ("tibant", 0): "tibia",
    ("tibpost", 0): "tibia",
    ("vasint", 1): "tibia",
    ("vaslat", 1): "tibia",
    ("vasmed", 1): "tibia",
}
_NUMI_HUMAN_RIGID_FOOT_ENTHESIS_MEMBER_IDS = {
    "r": {
        "calcaneus": "FJ3360", "navicular": "FJ3308",
        "first_metatarsal": "FJ3351", "fifth_metatarsal": "FJ3359",
    },
    "l": {
        "calcaneus": "FJ3256", "navicular": "FJ3307",
        "first_metatarsal": "FJ3241", "fifth_metatarsal": "FJ3253",
    },
}
_NUMI_HUMAN_RIGID_FOOT_ENTHESIS_MEMBER_CLASS = {
    # Triceps surae joins the calcaneal tuberosity through the Achilles tendon.
    ("gaslat", 1): "calcaneus",
    ("gasmed", 1): "calcaneus",
    ("soleus", 1): "calcaneus",
    # The source mechanics lumps the rigid foot into ``calcn``; retain exact
    # anatomical insertion identity within that body.
    ("perbrev", 1): "fifth_metatarsal",
    ("perlong", 1): "first_metatarsal",
    ("tibant", 1): "first_metatarsal",
    ("tibpost", 1): "navicular",
}
_NUMI_HUMAN_LIMB_ENTHESIS_MEMBERS = {
    **{
        (f"{base}_{side}", 0): (member_id,)
        for side, member_id in (("r", "FJ3152"), ("l", "FJ3288"))
        for base in _NUMI_HUMAN_PELVIS_ENTHESIS_BASE_NAMES
    },
    **{
        (f"{base}_{side}", endpoint): (
            member_ids[member_class],
        )
        for side, member_ids in (
            ("r", {"tibia": "FJ3387", "fibula": "FJ3366"}),
            ("l", {"tibia": "FJ3282", "fibula": "FJ3260"}),
        )
        for (base, endpoint), member_class in (
            _NUMI_HUMAN_SHANK_ENTHESIS_MEMBER_CLASS.items()
        )
    },
    **{
        (f"{base}_{side}", endpoint): (
            _NUMI_HUMAN_RIGID_FOOT_ENTHESIS_MEMBER_IDS[side][member_class],
        )
        for side in ("r", "l")
        for (base, endpoint), member_class in (
            _NUMI_HUMAN_RIGID_FOOT_ENTHESIS_MEMBER_CLASS.items()
        )
    },
}


# MyoSim's torso collapses the thoracic cage into one rigid body, while its
# source route identifiers retain the exact thoracic vertebra or rib level.
# BodyParts3D keeps those bones as separate named members on that same body.
# Resolve only identities that are explicit in both pinned sources: ``Tn`` is
# thoracic vertebra n, ``Rn`` is the same-side rib n, QL ``12.1``--``12.3``
# labels terminate on the same-side twelfth rib, and QL ``T12`` terminates on
# the twelfth thoracic vertebra. Abdominal-wall route names such as EO/IO are
# not member authority by themselves. Only a validated pinned-source connected-
# component receipt may map them to a rib; all other termini remain point-owned.
_NUMI_HUMAN_THORACIC_VERTEBRA_ENTHESIS_MEMBER_IDS = {
    1: "FJ3158", 2: "FJ3160", 3: "FJ3163", 4: "FJ3166",
    5: "FJ3169", 6: "FJ3171", 7: "FJ3173", 8: "FJ3174",
    9: "FJ3175", 10: "FJ3154", 11: "FJ3155", 12: "FJ3156",
}
_NUMI_HUMAN_RIB_ENTHESIS_MEMBER_IDS = {
    "r": {
        1: "FJ3334", 2: "FJ3336", 3: "FJ3338", 4: "FJ3340",
        5: "FJ3342", 6: "FJ3344", 7: "FJ3346", 8: "FJ3347",
        9: "FJ3348", 10: "FJ3330", 11: "FJ3331", 12: "FJ3332",
    },
    "l": {
        1: "FJ3228", 2: "FJ3229", 3: "FJ3230", 4: "FJ3231",
        5: "FJ3232", 6: "FJ3233", 7: "FJ3234", 8: "FJ3235",
        9: "FJ3236", 10: "FJ3225", 11: "FJ3226", 12: "FJ3227",
    },
}
_NUMI_HUMAN_TWELFTH_RIB_ENTHESIS_ENDPOINTS = {
    "QL_ant_I.2-12.1": {"r": 1, "l": 1},
    "QL_ant_I.3-12.1": {"r": 1, "l": 1},
    "QL_ant_I.3-12.2": {"r": 1, "l": 1},
    "QL_ant_I.3-12.3": {"r": 1, "l": 1},
    # The mirrored source lists one L2-to-12.1 route in reverse endpoint
    # order; keep the compiled endpoint ordinal rather than normalizing it.
    "QL_mid_L2-12.1": {"r": 0, "l": 1},
    "QL_mid_L3-12.1": {"r": 0, "l": 0},
    "QL_mid_L3-12.2": {"r": 0, "l": 0},
    "QL_mid_L3-12.3": {"r": 0, "l": 0},
    "QL_mid_L4-12.3": {"r": 0, "l": 0},
}
_NUMI_HUMAN_AXIAL_ENTHESIS_MEMBERS = {
    **{
        (f"LTpT_T{level}_{side}", 1): (member_id,)
        for level, member_id in (
            _NUMI_HUMAN_THORACIC_VERTEBRA_ENTHESIS_MEMBER_IDS.items()
        )
        for side in ("r", "l")
    },
    **{
        (f"{family}_R{level}_{side}", 1): (
            _NUMI_HUMAN_RIB_ENTHESIS_MEMBER_IDS[side][level],
        )
        for family, levels in (("IL", range(5, 13)), ("LTpT", range(4, 13)))
        for level in levels
        for side in ("r", "l")
    },
    **{
        (f"{base}_{side}", endpoints[side]): (
            _NUMI_HUMAN_RIB_ENTHESIS_MEMBER_IDS[side][12],
        )
        for base, endpoints in (
            _NUMI_HUMAN_TWELFTH_RIB_ENTHESIS_ENDPOINTS.items()
        )
        for side in ("r", "l")
    },
    **{
        (f"{base}_{side}", 1): (
            _NUMI_HUMAN_THORACIC_VERTEBRA_ENTHESIS_MEMBER_IDS[12],
        )
        for base in ("QL_ant_I.2-T12", "QL_ant_I.3-T12")
        for side in ("r", "l")
    },
}
_NUMI_HUMAN_SOURCE_COMPONENT_ENTHESIS_SCHEMA = (
    "numi.human.myosim-abdominal-source-component-enthesis.v1"
)
_NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION = (
    "source_topology_resolved_rib_member"
)
_NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION = (
    "source_thorax_non_rib_component_endpoint"
)
_NUMI_HUMAN_SOURCE_COMPONENT_NON_BONE_DISPOSITION = (
    "source_model_non_bone_endpoint"
)
_NUMI_HUMAN_SOURCE_COMPONENT_EXPECTED_RECORDS = {
    ("rect_abd_r", 1): (22, _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION, None),
    ("rect_abd_l", 1): (23, _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION, None),
    ("EO1_r", 1): (186, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 7),
    ("EO2_r", 0): (187, _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION, None),
    ("EO3_r", 0): (188, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 9),
    ("EO4_r", 0): (189, _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION, None),
    ("EO5_r", 0): (190, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 10),
    ("EO6_r", 0): (191, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 11),
    ("IO4_r", 0): (195, _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION, None),
    ("IO5_r", 0): (196, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 10),
    ("IO6_r", 0): (197, _NUMI_HUMAN_SOURCE_COMPONENT_NON_BONE_DISPOSITION, None),
    ("EO1_l", 1): (198, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 7),
    ("EO2_l", 0): (199, _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION, None),
    ("EO3_l", 0): (200, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 9),
    ("EO4_l", 0): (201, _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION, None),
    ("EO5_l", 0): (202, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 10),
    ("EO6_l", 0): (203, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 11),
    ("IO4_l", 0): (207, _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION, None),
    ("IO5_l", 0): (208, _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION, 10),
    ("IO6_l", 0): (209, _NUMI_HUMAN_SOURCE_COMPONENT_NON_BONE_DISPOSITION, None),
}


def _numi_human_source_component_enthesis_receipt(
    receipt: Any, member_body_indices: dict[str, int], source_sha: str,
) -> dict[str, Any] | None:
    """Validate the optional source-topology endpoint-ownership receipt."""
    if receipt is None:
        return None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != _NUMI_HUMAN_SOURCE_COMPONENT_ENTHESIS_SCHEMA
        or receipt.get("status")
        != "candidate_passed_exact_component_identity_and_bilateral_gates"
        or receipt.get("inputs", {}).get("myosim_archive_sha256") != source_sha
        or receipt.get("endpoint_count") != 20
        or receipt.get("bilateral_pair_count") != 10
        or receipt.get("endpoint_migration_m") != 0.0
        or receipt.get("new_joint_count") != 0
        or receipt.get("source_mesh_name") != "torso_geom_13_ribcage_s"
        or receipt.get("source_connected_component_count") != 36
        or receipt.get("source_rib_component_count") != 24
        or receipt.get("source_component_surface_count") != 8
    ):
        raise ImportError("Numi Human source-component enthesis receipt is invalid")
    records = receipt.get("endpoint_records")
    pairs = receipt.get("bilateral_pairs")
    source_surfaces = receipt.get("source_component_surfaces")
    if (
        not isinstance(records, list) or len(records) != 20
        or not isinstance(pairs, list) or len(pairs) != 10
        or not isinstance(source_surfaces, list) or len(source_surfaces) != 8
        or any(not isinstance(pair, dict) or pair.get("passed") is not True for pair in pairs)
    ):
        raise ImportError("Numi Human source-component enthesis receipt is incomplete")
    allowed_dispositions = {
        _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION,
        _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION,
        _NUMI_HUMAN_SOURCE_COMPONENT_NON_BONE_DISPOSITION,
    }
    counts: Counter[str] = Counter()
    keys: set[tuple[str, int]] = set()
    rib_members = {
        member
        for side_members in _NUMI_HUMAN_RIB_ENTHESIS_MEMBER_IDS.values()
        for member in side_members.values()
    }
    source_surface_by_component: dict[int, dict[str, Any]] = {}
    for surface in source_surfaces:
        if not isinstance(surface, dict):
            raise ImportError("Numi Human source-component surface is malformed")
        component_index = surface.get("source_component_index")
        vertices = surface.get("vertices_core_m")
        triangles = surface.get("triangles")
        signature = surface.get("source_component_vertex_index_sha256")
        content_sha = surface.get("surface_content_sha256")
        if (
            not isinstance(component_index, int)
            or component_index in source_surface_by_component
            or not isinstance(vertices, list) or len(vertices) < 4
            or not isinstance(triangles, list) or len(triangles) < 4
            or surface.get("source_vertex_count") != len(vertices)
            or surface.get("source_triangle_count") != len(triangles)
            or not isinstance(signature, str)
            or re.fullmatch(r"[0-9a-f]{64}", signature) is None
            or not isinstance(content_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", content_sha) is None
            or surface.get("mechanics_role")
            != "exact_pinned_source_surface_fallback_after_bodyparts_rejection"
            or any(
                not isinstance(vertex, list) or len(vertex) != 3
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vertex)
                for vertex in vertices
            )
            or any(
                not isinstance(triangle, list) or len(triangle) != 3
                or any(not isinstance(index, int) or not 0 <= index < len(vertices) for index in triangle)
                for triangle in triangles
            )
        ):
            raise ImportError("Numi Human source-component surface identity is invalid")
        encoded = json.dumps(
            {"triangles": triangles, "vertices_core_m": vertices},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != content_sha:
            raise ImportError("Numi Human source-component surface content drifted")
        source_surface_by_component[component_index] = surface
    for record in records:
        if not isinstance(record, dict):
            raise ImportError("Numi Human source-component enthesis record is malformed")
        muscle = record.get("muscle")
        endpoint = record.get("endpoint")
        ordinal = record.get("endpoint_ordinal")
        disposition = record.get("disposition")
        members = record.get("bone_member_ids")
        expected = _NUMI_HUMAN_SOURCE_COMPONENT_EXPECTED_RECORDS.get(
            (muscle, ordinal)
        )
        if (
            not isinstance(muscle, str)
            or endpoint not in {"origin", "insertion"}
            or ordinal != (0 if endpoint == "origin" else 1)
            or disposition not in allowed_dispositions
            or not isinstance(members, list)
            or record.get("source_body_name") != "torso"
            or record.get("endpoint_migration_m") != 0.0
            or (muscle, ordinal) in keys
            or expected is None
            or record.get("source_actuator_index") != expected[0]
            or disposition != expected[1]
        ):
            raise ImportError("Numi Human source-component enthesis identity is invalid")
        keys.add((muscle, ordinal))
        counts[disposition] += 1
        if disposition == _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION:
            side = muscle.rsplit("_", 1)[-1]
            level = expected[2]
            expected_member = _NUMI_HUMAN_RIB_ENTHESIS_MEMBER_IDS[side][level]
            if (
                len(members) != 1 or members[0] != expected_member
                or members[0] not in rib_members
                or members[0] not in member_body_indices
                or member_body_indices[members[0]] != 20
                or record.get("side") != ("right" if side == "r" else "left")
                or record.get("thoracic_level") != level
                or not isinstance(record.get("source_component_index"), int)
                or not isinstance(record.get("source_triangle_index"), int)
                or not isinstance(record.get("source_component_vertex_index_sha256"), str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}", record["source_component_vertex_index_sha256"]
                )
                or record.get("source_mechanics_surface_id")
                != f"MYSRC{record['source_component_index']:02d}"
                or record.get("source_mechanics_surface_policy")
                != "fallback_only_after_bodyparts_member_rejection"
                or record["source_component_index"] not in source_surface_by_component
                or source_surface_by_component[record["source_component_index"]].get(
                    "source_component_vertex_index_sha256"
                ) != record["source_component_vertex_index_sha256"]
            ):
                raise ImportError(
                    "Numi Human source-component rib ownership is invalid"
                )
        elif (
            members or record.get("side") is not None
            or record.get("thoracic_level") is not None
        ):
            raise ImportError("Numi Human non-rib source-component endpoint names a bone")
    expected_counts = {
        _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION: 10,
        _NUMI_HUMAN_SOURCE_COMPONENT_NON_RIB_DISPOSITION: 8,
        _NUMI_HUMAN_SOURCE_COMPONENT_NON_BONE_DISPOSITION: 2,
    }
    if (
        keys != set(_NUMI_HUMAN_SOURCE_COMPONENT_EXPECTED_RECORDS)
        or dict(counts) != expected_counts
        or receipt.get("disposition_counts") != expected_counts
    ):
        raise ImportError("Numi Human source-component enthesis counts drifted")
    return json.loads(json.dumps(receipt))


_NUMI_HUMAN_SEMANTIC_ENTHESIS_MEMBERS = {
    **_NUMI_HUMAN_TOE_ENTHESIS_MEMBERS,
    **_NUMI_HUMAN_LIMB_ENTHESIS_MEMBERS,
    **_NUMI_HUMAN_AXIAL_ENTHESIS_MEMBERS,
}
_NUMI_HUMAN_RIGID_FOOT_MIGRATABLE_ENTHESES = frozenset({
    key for key, members in _NUMI_HUMAN_TOE_ENTHESIS_MEMBERS.items()
    if len(members) == 1
}) | frozenset({
    (f"{base}_{side}", endpoint)
    for side in ("r", "l")
    for (base, endpoint) in _NUMI_HUMAN_RIGID_FOOT_ENTHESIS_MEMBER_CLASS
})


def _numi_human_semantic_enthesis_kind(
    key: tuple[str, int], member_count: int,
) -> str:
    if key in _NUMI_HUMAN_TOE_ENTHESIS_MEMBERS:
        return (
            "lumped_digitorum_route_to_four_lesser_toe_distal_phalanges"
            if member_count == 4 else "single_named_distal_phalanx"
        )
    if key in _NUMI_HUMAN_LIMB_ENTHESIS_MEMBERS:
        member = _NUMI_HUMAN_LIMB_ENTHESIS_MEMBERS[key][0]
        if member in {"FJ3152", "FJ3288"}:
            return "single_named_bilateral_hip_member"
        if member in {
            value
            for members in _NUMI_HUMAN_RIGID_FOOT_ENTHESIS_MEMBER_IDS.values()
            for value in members.values()
        }:
            return "single_named_rigid_foot_enthesis_member"
        return "single_named_tibia_or_fibula_member"
    if key in _NUMI_HUMAN_AXIAL_ENTHESIS_MEMBERS:
        member = _NUMI_HUMAN_AXIAL_ENTHESIS_MEMBERS[key][0]
        return (
            "single_named_thoracic_vertebra_member"
            if member in _NUMI_HUMAN_THORACIC_VERTEBRA_ENTHESIS_MEMBER_IDS.values()
            else "single_named_lateralized_rib_member"
        )
    raise ImportError(f"Numi Human semantic enthesis key is not declared: {key}")


_NUMI_HUMAN_TOE_VISUAL_LOCK_RADIUS_M = 0.008
_NUMI_HUMAN_TOE_VISUAL_FEATHER_RADIUS_M = 0.018
_NUMI_HUMAN_TOE_VISUAL_DISTAL_LOCK_FRACTION = 0.20
_NUMI_HUMAN_TOE_VISUAL_DISTAL_FEATHER_FRACTION = 0.45
# These four BodyParts3D hallucis members contain one complete anatomical
# sheet plus 45--85 disconnected export shards.  The shards have no route or
# bone correspondence and stretch into false digital fragments after sparse
# three-body posing.  Retain the dominant exact source sheet, as is already
# done for the compound calcaneal-tendon members.
_NUMI_HUMAN_HALLUX_DOMINANT_SOURCE_SURFACE_MEMBERS = frozenset({
    "FJ1408", "FJ1408M", "FJ1415", "FJ1415M",
})
_NUMI_HUMAN_HALLUX_VISUAL_ENTHESIS_MINIMUM_GAP_M = 0.001
_NUMI_HUMAN_HALLUX_VISUAL_ENTHESIS_INSET_M = 0.00035


_BODYPARTS_MYOSIM_AXIAL_EXTENSIONS = tuple(
    _bodyparts_visual_only_bone(body, name, member)
    for body, name, member in (
        ("lumbar1", "first lumbar vertebra", "FJ3157"),
        ("lumbar2", "second lumbar vertebra", "FJ3159"),
        ("lumbar3", "third lumbar vertebra", "FJ3162"),
        ("lumbar4", "fourth lumbar vertebra", "FJ3165"),
        ("lumbar5", "fifth lumbar vertebra", "FJ3168"),
        # The source and MyoSim models both expose the thorax as a whole-body
        # segment.  Retain the individual source vertebrae as a linked visual
        # layer until a segment-level axial mechanics transfer is qualified.
        ("torso", "first thoracic vertebra", "FJ3158"),
        ("torso", "second thoracic vertebra", "FJ3160"),
        ("torso", "third thoracic vertebra", "FJ3163"),
        ("torso", "fourth thoracic vertebra", "FJ3166"),
        ("torso", "fifth thoracic vertebra", "FJ3169"),
        ("torso", "sixth thoracic vertebra", "FJ3171"),
        ("torso", "seventh thoracic vertebra", "FJ3173"),
        ("torso", "eighth thoracic vertebra", "FJ3174"),
        ("torso", "ninth thoracic vertebra", "FJ3175"),
        ("torso", "tenth thoracic vertebra", "FJ3154"),
        ("torso", "eleventh thoracic vertebra", "FJ3155"),
        ("torso", "twelfth thoracic vertebra", "FJ3156"),
        ("cervical_spine", "third cervical vertebra", "FJ3161"),
        ("cervical_spine", "fourth cervical vertebra", "FJ3164"),
        ("cervical_spine", "fifth cervical vertebra", "FJ3167"),
        ("cervical_spine", "sixth cervical vertebra", "FJ3170"),
        ("cervical_spine", "seventh cervical vertebra", "FJ3172"),
    )
)


# These are the source-mesh transitions that cross a MyoSim rigid-body
# boundary in the axial skeleton.  Vertebrae carried by one common body cannot
# separate under articulation, while these boundaries can be pulled apart by
# an otherwise useful per-body attachment refinement.  Keep the gate at 8 mm:
# it admits an intervertebral joint space, but rejects the 16.8 mm L4/L5 visual
# discontinuity found during the pectoral-fascia multi-angle review.
_NUMI_HUMAN_AXIAL_CONTINUITY_TRANSITIONS = (
    ("occiput_to_atlas", "FJ3309", "FJ3176"),
    ("cervical7_to_thoracic1", "FJ3172", "FJ3158"),
    ("thoracic1_to_thoracic2", "FJ3158", "FJ3160"),
    ("thoracic2_to_thoracic3", "FJ3160", "FJ3163"),
    ("thoracic3_to_thoracic4", "FJ3163", "FJ3166"),
    ("thoracic4_to_thoracic5", "FJ3166", "FJ3169"),
    ("thoracic5_to_thoracic6", "FJ3169", "FJ3171"),
    ("thoracic6_to_thoracic7", "FJ3171", "FJ3173"),
    ("thoracic7_to_thoracic8", "FJ3173", "FJ3174"),
    ("thoracic8_to_thoracic9", "FJ3174", "FJ3175"),
    ("thoracic9_to_thoracic10", "FJ3175", "FJ3154"),
    ("thoracic10_to_thoracic11", "FJ3154", "FJ3155"),
    ("thoracic11_to_thoracic12", "FJ3155", "FJ3156"),
    ("thoracic12_to_lumbar1", "FJ3156", "FJ3157"),
    ("lumbar1_to_lumbar2", "FJ3157", "FJ3159"),
    ("lumbar2_to_lumbar3", "FJ3159", "FJ3162"),
    ("lumbar3_to_lumbar4", "FJ3162", "FJ3165"),
    ("lumbar4_to_lumbar5", "FJ3165", "FJ3168"),
    ("lumbar5_to_sacrum", "FJ3168", "FJ3393"),
    ("sacrum_to_right_hip", "FJ3393", "FJ3152"),
    ("sacrum_to_left_hip", "FJ3393", "FJ3288"),
)
_NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M = 0.008


# Default-pose source-surface transitions spanning the shoulder, elbow, and
# wrist rigid-body boundaries.  The glenohumeral gates are deliberately tight
# because the transformed BodyParts3D humeral heads already sit in their
# glenoids.  The acromioclavicular and radial-carpal gates admit normal joint
# space in a bone-only source while rejecting the 16--30 mm gaps found in the
# 2026-08-29 multi-angle audit. The ulna does not directly contact the
# triquetrum: the TFCC/articular disc and meniscus homologue occupy that named
# ulnocarpal interval, and the pinned MyoSim source meshes measure 9.371 mm
# bilaterally. Its 12 mm gate is therefore source-calibrated rather than a
# relaxed radial-carpal threshold. These remain visual proximity gates, not
# cartilage, TFCC, ligament, or contact certificates.
_NUMI_HUMAN_UPPER_LIMB_CONTINUITY_TRANSITIONS = (
    ("right_clavicle_to_scapula", "FJ3362", "FJ3384", 0.012),
    ("right_scapula_to_humerus", "FJ3384", "FJ3368", 0.008),
    ("right_humerus_to_ulna", "FJ3368", "FJ3391", 0.006),
    ("right_humerus_to_radius", "FJ3368", "FJ3349", 0.006),
    ("right_radius_to_scaphoid", "FJ3349", "FJ3383", 0.007),
    ("right_radius_to_lunate", "FJ3349", "FJ3374", 0.007),
    ("right_ulna_to_triquetrum", "FJ3391", "FJ3390", 0.012),
    ("left_clavicle_to_scapula", "FJ3237", "FJ3279", 0.012),
    ("left_scapula_to_humerus", "FJ3279", "FJ3262", 0.008),
    ("left_humerus_to_ulna", "FJ3262", "FJ3286", 0.006),
    ("left_humerus_to_radius", "FJ3262", "FJ3277", 0.006),
    ("left_radius_to_scaphoid", "FJ3277", "FJ3278", 0.007),
    ("left_radius_to_lunate", "FJ3277", "FJ3268", 0.007),
    ("left_ulna_to_triquetrum", "FJ3286", "FJ3285", 0.012),
)


# Every distal upper-limb source mesh is restored to its exact BodyParts3D
# rest-frame displacement from the already-site-refined humerus on that side.
# Unlike independently tuned gap patches, this preserves the complete source
# arrangement through the elbow, wrist, metacarpals, and phalanges while the
# existing MyoSim bodies retain their authored independent articulation.
_NUMI_HUMAN_UPPER_LIMB_COHERENT_ROOTS = {"r": "humerus_r", "l": "humerus_l"}


_NUMI_HUMAN_HAND_CONTINUITY_TRANSITIONS = (
    ("right_trapezium_to_first_metacarpal", "FJ3388", "FJ3350"),
    ("right_trapezoid_to_second_metacarpal", "FJ3389", "FJ3352"),
    ("right_capitate_to_third_metacarpal", "FJ3361", "FJ3354"),
    ("right_hamate_to_fourth_metacarpal", "FJ3367", "FJ3356"),
    ("right_hamate_to_fifth_metacarpal", "FJ3367", "FJ3358"),
    ("right_first_metacarpal_to_thumb_proximal", "FJ3350", "FJ3327"),
    ("right_thumb_proximal_to_distal", "FJ3327", "FJ3198"),
    ("right_second_metacarpal_to_index_proximal", "FJ3352", "FJ3322"),
    ("right_index_proximal_to_middle", "FJ3322", "FJ3303"),
    ("right_index_middle_to_distal", "FJ3303", "FJ3193"),
    ("right_third_metacarpal_to_middle_proximal", "FJ3354", "FJ3325"),
    ("right_middle_proximal_to_middle", "FJ3325", "FJ3306"),
    ("right_middle_middle_to_distal", "FJ3306", "FJ3196"),
    ("right_fourth_metacarpal_to_ring_proximal", "FJ3356", "FJ3326"),
    ("right_ring_proximal_to_middle", "FJ3326", "FJ3292"),
    ("right_ring_middle_to_distal", "FJ3292", "FJ3197"),
    ("right_fifth_metacarpal_to_little_proximal", "FJ3358", "FJ3323"),
    ("right_little_proximal_to_middle", "FJ3323", "FJ3304"),
    ("right_little_middle_to_distal", "FJ3304", "FJ3194"),
    ("left_trapezium_to_first_metacarpal", "FJ3283", "FJ3240"),
    ("left_trapezoid_to_second_metacarpal", "FJ3284", "FJ3243"),
    ("left_capitate_to_third_metacarpal", "FJ3257", "FJ3246"),
    ("left_hamate_to_fourth_metacarpal", "FJ3261", "FJ3249"),
    ("left_hamate_to_fifth_metacarpal", "FJ3261", "FJ3252"),
    ("left_first_metacarpal_to_thumb_proximal", "FJ3240", "FJ3318"),
    ("left_thumb_proximal_to_distal", "FJ3318", "FJ3188"),
    ("left_second_metacarpal_to_index_proximal", "FJ3243", "FJ3313"),
    ("left_index_proximal_to_middle", "FJ3313", "FJ3296"),
    ("left_index_middle_to_distal", "FJ3296", "FJ3183"),
    ("left_third_metacarpal_to_middle_proximal", "FJ3246", "FJ3316"),
    ("left_middle_proximal_to_middle", "FJ3316", "FJ3299"),
    ("left_middle_middle_to_distal", "FJ3299", "FJ3186"),
    ("left_fourth_metacarpal_to_ring_proximal", "FJ3249", "FJ3317"),
    ("left_ring_proximal_to_middle", "FJ3317", "FJ3291"),
    ("left_ring_middle_to_distal", "FJ3291", "FJ3187"),
    ("left_fifth_metacarpal_to_little_proximal", "FJ3252", "FJ3314"),
    ("left_little_proximal_to_middle", "FJ3314", "FJ3297"),
    ("left_little_middle_to_distal", "FJ3297", "FJ3184"),
)
_NUMI_HUMAN_HAND_CONTINUITY_MAXIMUM_GAP_M = 0.004


_NUMI_HUMAN_KNEE_CONTINUITY_TRANSITIONS = (
    ("right_femur_to_tibia", "FJ3365", "FJ3387"),
    ("right_femur_to_patella", "FJ3365", "FJ3381"),
    ("left_femur_to_tibia", "FJ3259", "FJ3282"),
    ("left_femur_to_patella", "FJ3259", "FJ3275"),
)
_NUMI_HUMAN_KNEE_CONTINUITY_MAXIMUM_GAP_M = 0.004


_NUMI_HUMAN_LOWER_LIMB_COHERENT_ROOTS = {"r": "femur_r", "l": "femur_l"}
_NUMI_HUMAN_FOOT_CONTINUITY_TRANSITIONS = (
    ("right_tibia_to_talus", "FJ3387", "FJ3385"),
    ("right_fibula_to_talus", "FJ3366", "FJ3385"),
    ("right_talus_to_calcaneus", "FJ3385", "FJ3360"),
    ("right_talus_to_navicular", "FJ3385", "FJ3308"),
    ("right_calcaneus_to_cuboid", "FJ3360", "FJ3364"),
    ("right_navicular_to_medial_cuneiform", "FJ3308", "FJ3377"),
    ("right_navicular_to_intermediate_cuneiform", "FJ3308", "FJ3370"),
    ("right_navicular_to_lateral_cuneiform", "FJ3308", "FJ3373"),
    ("right_medial_cuneiform_to_first_metatarsal", "FJ3377", "FJ3351"),
    ("right_intermediate_cuneiform_to_second_metatarsal", "FJ3370", "FJ3353"),
    ("right_lateral_cuneiform_to_third_metatarsal", "FJ3373", "FJ3355"),
    ("right_cuboid_to_fourth_metatarsal", "FJ3364", "FJ3357"),
    ("right_cuboid_to_fifth_metatarsal", "FJ3364", "FJ3359"),
    ("right_first_metatarsal_to_hallux", "FJ3351", "FJ3310"),
    ("right_second_metatarsal_to_second_toe", "FJ3353", "FJ3319"),
    ("right_third_metatarsal_to_third_toe", "FJ3355", "FJ3320"),
    ("right_fourth_metatarsal_to_fourth_toe", "FJ3357", "FJ3321"),
    ("right_fifth_metatarsal_to_fifth_toe", "FJ3359", "FJ3324"),
    ("left_tibia_to_talus", "FJ3282", "FJ3280"),
    ("left_fibula_to_talus", "FJ3260", "FJ3280"),
    ("left_talus_to_calcaneus", "FJ3280", "FJ3256"),
    ("left_talus_to_navicular", "FJ3280", "FJ3307"),
    ("left_calcaneus_to_cuboid", "FJ3256", "FJ3258"),
    ("left_navicular_to_medial_cuneiform", "FJ3307", "FJ3271"),
    ("left_navicular_to_intermediate_cuneiform", "FJ3307", "FJ3264"),
    ("left_navicular_to_lateral_cuneiform", "FJ3307", "FJ3267"),
    ("left_medial_cuneiform_to_first_metatarsal", "FJ3271", "FJ3241"),
    ("left_intermediate_cuneiform_to_second_metatarsal", "FJ3264", "FJ3244"),
    ("left_lateral_cuneiform_to_third_metatarsal", "FJ3267", "FJ3247"),
    ("left_cuboid_to_fourth_metatarsal", "FJ3258", "FJ3250"),
    ("left_cuboid_to_fifth_metatarsal", "FJ3258", "FJ3253"),
    ("left_first_metatarsal_to_hallux", "FJ3241", "FJ3329"),
    ("left_second_metatarsal_to_second_toe", "FJ3244", "FJ3328"),
    ("left_third_metatarsal_to_third_toe", "FJ3247", "FJ3311"),
    ("left_fourth_metatarsal_to_fourth_toe", "FJ3250", "FJ3312"),
    ("left_fifth_metatarsal_to_fifth_toe", "FJ3253", "FJ3315"),
)
_NUMI_HUMAN_FOOT_CONTINUITY_MAXIMUM_GAP_M = 0.004


_BODYPARTS_MYOSIM_BONE_ANCHORS = (
    _BODYPARTS_MYOSIM_FIT_BONE_ANCHORS
    + _BODYPARTS_MYOSIM_MAJOR_BONE_EXTENSIONS
    + _BODYPARTS_MYOSIM_CRANIAL_EXTENSIONS
    + _BODYPARTS_MYOSIM_THORACIC_FOOT_EXTENSIONS
    + _BODYPARTS_MYOSIM_REMAINING_SOURCE_EXTENSIONS
    + _BODYPARTS_MYOSIM_WRIST_HAND_EXTENSIONS
    + _BODYPARTS_MYOSIM_TOE_EXTENSIONS
    + _BODYPARTS_MYOSIM_AXIAL_EXTENSIONS
)


def _matrix3_determinant(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _bodyparts_similarity_fit(
    source_points_m: list[list[float]], target_points_m: list[list[float]],
) -> dict[str, Any]:
    """Find the best proper signed-axis similarity, excluding reflections."""
    if len(source_points_m) != len(target_points_m) or len(source_points_m) < 3:
        raise ImportError("BodyParts3D registration requires at least three matched landmarks")
    source = [_vector3(value, "BodyParts3D registration source point") for value in source_points_m]
    target = [_vector3(value, "MyoSim registration target point") for value in target_points_m]
    target_mean = [sum(point[axis] for point in target) / len(target) for axis in range(3)]
    best: tuple[float, list[list[float]], tuple[int, int, int], tuple[int, int, int], float, list[float], list[float]] | None = None
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = [[0.0, 0.0, 0.0] for _ in range(3)]
            for row, source_axis in enumerate(axes):
                rotation[row][source_axis] = float(signs[row])
            if _matrix3_determinant(rotation) < 0.5:
                continue
            mapped = [_myosim_matrix_vector(rotation, point) for point in source]
            mapped_mean = [sum(point[axis] for point in mapped) / len(mapped) for axis in range(3)]
            denominator = sum((point[axis] - mapped_mean[axis]) ** 2 for point in mapped for axis in range(3))
            if denominator <= 1.0e-16:
                continue
            scale = sum(
                (point[axis] - mapped_mean[axis]) * (target[index][axis] - target_mean[axis])
                for index, point in enumerate(mapped) for axis in range(3)
            ) / denominator
            if not math.isfinite(scale) or scale <= 0.0:
                continue
            translation = [target_mean[axis] - scale * mapped_mean[axis] for axis in range(3)]
            residuals = [
                math.sqrt(sum(
                    (scale * mapped[index][axis] + translation[axis] - target[index][axis]) ** 2
                    for axis in range(3)
                ))
                for index in range(len(source))
            ]
            rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
            candidate = (rms, rotation, tuple(axes), tuple(signs), scale, translation, residuals)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ImportError("BodyParts3D registration could not find a proper positive-scale similarity")
    rms, rotation, axes, signs, scale, translation, residuals = best
    return {
        "rotation": rotation, "axis_permutation": list(axes), "axis_signs": list(signs),
        "scale_after_mm_to_m": scale, "translation_world_m": translation,
        "residuals_m": residuals, "rms_residual_m": rms,
    }


def _bodyparts_registration_matrix(
    rotation: list[list[float]], scale_after_mm_to_m: float, translation_world_m: list[float],
) -> list[list[float]]:
    linear_scale_m_per_mm = 0.001 * scale_after_mm_to_m
    return [
        [*(linear_scale_m_per_mm * value for value in row), translation_world_m[index]]
        for index, row in enumerate(rotation)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _bodyparts_local_registration_matrix(
    global_matrix: list[list[float]], body_com_world_m: list[float], body_quaternion_world_xyzw: list[float],
) -> list[list[float]]:
    world_to_body = _matrix_transpose(_myosim_matrix_from_quaternion_xyzw(body_quaternion_world_xyzw))
    local_linear = _matrix_product(world_to_body, [row[:3] for row in global_matrix[:3]])
    local_translation = _myosim_matrix_vector(
        world_to_body, _myosim_subtract([row[3] for row in global_matrix[:3]], body_com_world_m),
    )
    return [[*local_linear[row], local_translation[row]] for row in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def bodyparts_myosim_registration_candidate(
    sources: Path, anatomy: dict[str, Any], myosim_artifact: Path,
) -> dict[str, Any]:
    """Build a source-pinned, visual-only BodyParts3D/MyoSim rest-frame fit.

    Each local matrix maps exact OBJ millimetres to a Core inertial-body frame.
    This is sufficient for a native articulated bone visual, but it is not a
    collider, skinning, material, or physiology admission.
    """
    artifact = myosim_artifact.resolve()
    manifest_path = artifact / "myosim-fullbody-reference.manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "numi.human.myosim-fullbody-reference.v1":
        raise ImportError("BodyParts3D registration requires a MyoSim full-body reference artifact")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, dict):
        raise ImportError("MyoSim registration artifact has no payload records")
    payload_provenance: dict[str, dict[str, Any]] = {}
    for key in ("rigid", "muscles"):
        descriptor = payloads.get(key)
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), str):
            raise ImportError(f"MyoSim registration artifact has no {key} payload descriptor")
        payload = artifact / descriptor["file"]
        expected_hash = descriptor.get("sha256")
        if not payload.is_file() or not isinstance(expected_hash, str) or sha256(payload) != expected_hash:
            raise ImportError(f"MyoSim registration artifact {key} payload is missing or has drifted")
        payload_provenance[key] = {"file": descriptor["file"], "sha256": expected_hash, "bytes": payload.stat().st_size}
    core_tree = manifest.get("core_tree")
    records = core_tree.get("source_body_records") if isinstance(core_tree, dict) else None
    if not isinstance(records, list):
        raise ImportError("MyoSim registration artifact does not expose source rest-pose body records")
    body_by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ImportError("MyoSim registration artifact has an unnamed body record")
        name = record["name"]
        if name in body_by_name or not isinstance(record.get("source_body_id"), int) or not isinstance(record.get("core_body_index"), int):
            raise ImportError("MyoSim registration artifact has duplicate or invalid body identities")
        record["default_com_position_world_m"] = _myosim_vector(record.get("default_com_position_world_m"), f"MyoSim registration body {name} COM")
        record["default_inertial_quaternion_world_xyzw"] = list(record.get("default_inertial_quaternion_world_xyzw", []))
        _myosim_matrix_from_quaternion_xyzw(record["default_inertial_quaternion_world_xyzw"])
        body_by_name[name] = record
    archive_by_hierarchy = {
        descriptor.get("hierarchy"): descriptor
        for descriptor in anatomy.get("archives", []) if isinstance(descriptor, dict)
    }
    components_by_name: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for component in anatomy.get("components", []):
        if isinstance(component, dict) and isinstance(component.get("hierarchy"), str) and isinstance(component.get("name"), str):
            components_by_name[(component["hierarchy"], component["name"])].append(component)

    anchors: list[dict[str, Any]] = []
    source_points_m: list[list[float]] = []
    target_points_m: list[list[float]] = []
    fit_anchor_indices: list[int] = []
    for specification in _BODYPARTS_MYOSIM_BONE_ANCHORS:
        target_name = specification["myosim_body"]
        target = body_by_name.get(target_name)
        if target is None:
            raise ImportError(f"MyoSim registration artifact has no required body {target_name}")
        hierarchy = specification["hierarchy"]
        components = components_by_name.get((hierarchy, specification["bodyparts_name"]), [])
        if len(components) != 1:
            raise ImportError(f"BodyParts3D registration requires one component for {hierarchy}:{specification['bodyparts_name']}")
        component = components[0]
        member_id = specification["member_id"]
        if not any(entry.get("element_id") == member_id and entry.get("mesh_present") for entry in component.get("element_meshes", []) if isinstance(entry, dict)):
            raise ImportError(f"BodyParts3D registration component {component['name']} has no expected mesh {member_id}")
        archive_path, member, obj = _bodyparts_obj_member(sources, hierarchy, member_id)
        archive_descriptor = archive_by_hierarchy.get(hierarchy)
        if not isinstance(archive_descriptor, dict) or archive_descriptor.get("sha256") != sha256(archive_path):
            raise ImportError(f"BodyParts3D registration archive provenance drifted for {hierarchy}")
        vertices_mm, triangles = _bodyparts_obj_triangles(obj, member)
        centroid_mm = [sum(vertex[axis] for vertex in vertices_mm) / len(vertices_mm) for axis in range(3)]
        anchors.append({
            "source": {
                "archive": archive_path.name, "archive_sha256": sha256(archive_path), "hierarchy": hierarchy,
                "member": member, "member_id": member_id, "member_sha256": hashlib.sha256(obj).hexdigest(),
                "concept_id": component.get("concept_id"), "name": component["name"],
                "vertex_count": len(vertices_mm), "triangle_count": len(triangles), "vertex_centroid_mm": centroid_mm,
            },
            "target": {
                "source_body_id": target["source_body_id"], "core_body_index": target["core_body_index"], "name": target_name,
                "default_com_position_world_m": target["default_com_position_world_m"],
                "default_inertial_quaternion_world_xyzw": target["default_inertial_quaternion_world_xyzw"],
            },
        })
        if specification.get("registration_anchor", True):
            source_points_m.append([value * 0.001 for value in centroid_mm])
            target_points_m.append(target["default_com_position_world_m"])
            fit_anchor_indices.append(len(anchors) - 1)
    fit = _bodyparts_similarity_fit(source_points_m, target_points_m)
    global_matrix = _bodyparts_registration_matrix(fit["rotation"], fit["scale_after_mm_to_m"], fit["translation_world_m"])
    residual_by_anchor = dict(zip(fit_anchor_indices, fit["residuals_m"], strict=True))
    for index, anchor in enumerate(anchors):
        target = anchor["target"]
        centroid_world_m = [
            sum(global_matrix[row][column] * anchor["source"]["vertex_centroid_mm"][column] for column in range(3)) + global_matrix[row][3]
            for row in range(3)
        ]
        anchor["registration"] = {
            "source_obj_mm_to_core_inertial_body_m": _bodyparts_local_registration_matrix(
                global_matrix, target["default_com_position_world_m"], target["default_inertial_quaternion_world_xyzw"],
            ),
            "default_pose_vertex_centroid_world_m": centroid_world_m,
            "vertex_centroid_to_source_com_residual_m": residual_by_anchor.get(index),
            "status": (
                "provisional_visual_fit_anchor"
                if index in residual_by_anchor
                else "provisional_visual_binding_from_fitted_common_frame"
            ),
        }
    return {
        "schema": "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2",
        "source": {
            "bodyparts": {"id": anatomy.get("source_id"), "version": anatomy.get("version"), "archives": anatomy.get("archives")},
            "myosim": {"artifact_manifest": manifest_path.name, "artifact_manifest_sha256": sha256(manifest_path), "source": manifest.get("source"), "payloads": payload_provenance},
        },
        "coordinate_system": {
            "source": "BodyParts3D OBJ millimetres", "target": "MyoSim default world metres and Core inertial-body frames",
            "global_source_mm_to_myosim_world_m": global_matrix, "proper_axis_permutation": fit["axis_permutation"],
            "proper_axis_signs": fit["axis_signs"], "uniform_scale_after_mm_to_m": fit["scale_after_mm_to_m"], "translation_world_m": fit["translation_world_m"],
        },
        "fit": {
            "method": "equal-weight vertex-centroid to source inertial-COM similarity over 24 proper signed-axis maps",
            "anchor_count": len(fit_anchor_indices), "rms_vertex_centroid_to_com_residual_m": fit["rms_residual_m"], "max_vertex_centroid_to_com_residual_m": max(fit["residuals_m"]),
            "interpretation": "A mesh vertex centroid and rigid-body inertial COM are not homologous landmarks. These residuals diagnose common-frame plausibility only, not surface registration accuracy.",
        },
        "anchors": anchors,
        "coverage": {
            "registered_visual_bone_mesh_count": len(anchors),
            "similarity_fit_anchor_count": len(fit_anchor_indices),
            "linked_myo_body_count": len({anchor["target"]["name"] for anchor in anchors}),
            "source_mesh_groups": {
                "fitted_major_bones": len(_BODYPARTS_MYOSIM_FIT_BONE_ANCHORS),
                "major_bone_extensions": len(_BODYPARTS_MYOSIM_MAJOR_BONE_EXTENSIONS),
                "cranial_mandibular_bones": len(_BODYPARTS_MYOSIM_CRANIAL_EXTENSIONS),
                "ribs_and_midfoot_tarsals": len(_BODYPARTS_MYOSIM_THORACIC_FOOT_EXTENSIONS),
                "atlas_axis_and_triquetra": len(_BODYPARTS_MYOSIM_REMAINING_SOURCE_EXTENSIONS),
                "wrists_hands_digits": len(_BODYPARTS_MYOSIM_WRIST_HAND_EXTENSIONS),
                "feet_toes": len(_BODYPARTS_MYOSIM_TOE_EXTENSIONS),
                "axial_vertebrae": len(_BODYPARTS_MYOSIM_AXIAL_EXTENSIONS),
            },
            "not_yet_represented": [
                "skin, muscles, tendons, organs, vessels, nerves, and other soft-tissue layers",
            ],
        },
        "status": "provisional_visual_registration_not_admitted_to_collision_or_physics",
        "next_visual_validation": [
            "bind every emitted local matrix to the corresponding Metal articulated inertial-body pose",
            "inspect default-pose front, side, rear, and oblique full-skeleton frames at native resolution",
            "review the feet, wrists/hands, and axial transitions before adding the remaining source meshes",
        ],
        "evidence_boundary": "This inferred source-geometry/MyoSim-pose candidate is not collision/contact geometry, skinning, soft-tissue mechanics, joint-limit transfer, muscle-attachment transfer, or a medically validated anatomical registration.",
    }


def _myosim_attachment_sites_from_payload(
    payload: Path, expected_payload_sha256: str, expected_source_sha256: str,
) -> dict[int, list[list[float]]]:
    """Read the source-preserving NHRMYO1 site records for visual registration.

    This is deliberately an offline import operation.  It makes no change to
    the runtime muscle path: it only supplies anatomical surface observations
    for the explicitly inferred BodyParts3D visual correspondence below.
    """
    if not payload.is_file() or sha256(payload) != expected_payload_sha256:
        raise ImportError("BodyParts3D attachment registration muscle payload is missing or has drifted")
    raw = payload.read_bytes()
    header_format = "<8s9I32s"
    header_bytes = struct.calcsize(header_format)
    if len(raw) < header_bytes:
        raise ImportError("BodyParts3D attachment registration muscle payload is truncated")
    (
        magic, abi, body_count, muscle_count, site_count, wrap_count,
        route_count, tendon_count, reserved0, reserved1, source_sha,
    ) = struct.unpack_from(header_format, raw)
    try:
        architecture_count, architecture_bytes = _myosim_muscle_payload_architecture(
            magic, abi, muscle_count, reserved0, reserved1,
        )
    except ImportError as error:
        raise ImportError("BodyParts3D attachment registration muscle payload ABI disagreement") from error
    expected_bytes = _myosim_muscle_payload_bytes(
        site_count, wrap_count, route_count, muscle_count,
        architecture_count, architecture_bytes,
    )
    if (
        body_count == 0 or tendon_count == 0
        or source_sha.hex() != expected_source_sha256 or len(raw) != expected_bytes
    ):
        raise ImportError("BodyParts3D attachment registration muscle payload ABI/provenance disagreement")
    result: dict[int, list[list[float]]] = defaultdict(list)
    offset = header_bytes
    for index in range(site_count):
        body_index, x, y, z = struct.unpack_from("<I3f", raw, offset + 16 * index)
        if body_index >= body_count or not all(math.isfinite(value) for value in (x, y, z)):
            raise ImportError("BodyParts3D attachment registration site record is malformed")
        result[body_index].append([x, y, z])
    return dict(result)


def _bodyparts_attachment_quantile(values: list[float], percentile: float) -> float:
    if not values or not 0.0 <= percentile <= 1.0:
        raise ImportError("BodyParts3D attachment residual quantile is undefined")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(percentile * len(ordered))) - 1)]


def bodyparts_myosim_attachment_surface_registration_candidate(
    sources: Path, anatomy: dict[str, Any], myosim_artifact: Path,
) -> dict[str, Any]:
    """Infer a constrained per-articulated-body translation from source sites.

    The base common-frame fit aligns only mesh centroids to inertial COMs,
    which is insufficient evidence for a muscle insertion.  This importer
    therefore uses the exact source site records on each already-bound link to
    refine *translation only* against the union of that link's BodyParts3D
    bone surfaces.  Every mesh on one link receives the same correction, so
    the result remains a coherent rigid visual binding.  It never changes
    source paths, forces, body binding, scale, or orientation, and remains
    visual-only until independently reviewed.
    """
    candidate = json.loads(json.dumps(
        bodyparts_myosim_registration_candidate(sources, anatomy, myosim_artifact)
    ))
    muscle_descriptor = candidate["source"]["myosim"]["payloads"].get("muscles")
    if not isinstance(muscle_descriptor, dict):
        raise ImportError("BodyParts3D attachment registration has no MyoSim muscle payload descriptor")
    muscle_file = muscle_descriptor.get("file")
    muscle_sha = muscle_descriptor.get("sha256")
    if not isinstance(muscle_file, str) or not isinstance(muscle_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", muscle_sha):
        raise ImportError("BodyParts3D attachment registration muscle payload descriptor is invalid")
    source_sha = candidate["source"]["myosim"]["source"].get("archive_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ImportError("BodyParts3D attachment registration has no MyoSim source SHA-256")
    sites_by_body = _myosim_attachment_sites_from_payload(
        myosim_artifact.resolve() / muscle_file, muscle_sha, source_sha,
    )
    max_correspondence_distance_m = 0.12
    maximum_iterations = 8
    # Every BodyParts3D member carried by one articulated link must retain one
    # shared link-local transform.  Refining each mesh independently can lower
    # a local site residual while pulling the tibia away from its fibula (or a
    # skull bone away from its neighbour).  That produces a superficially
    # better point match but an incoherent rigid body and prevents downstream
    # skin/tissue payloads from using the same registration.  Refine the union
    # of source surfaces for each source body instead.
    anchors_by_body: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for anchor in candidate["anchors"]:
        target = anchor["target"]
        body_index = target.get("core_body_index") if isinstance(target, dict) else None
        if not isinstance(body_index, int) or body_index < 0:
            raise ImportError("BodyParts3D attachment registration has an invalid body index")
        anchors_by_body[body_index].append(anchor)

    # Retain the coherent BodyParts3D common-frame centroid relationships
    # before any per-body attachment refinement. They provide an anatomical
    # chain constraint for unsupported distal segments (especially thumbs)
    # after a neighbouring bone has been anchored by exact source sites.
    common_frame_centroid_by_name = {
        body_anchors[0]["target"]["name"]: list(
            body_anchors[0]["registration"]["default_pose_vertex_centroid_world_m"]
        )
        for body_anchors in anchors_by_body.values()
    }

    summary: list[dict[str, Any]] = []
    for body_index, body_anchors in sorted(anchors_by_body.items()):
        target = body_anchors[0]["target"]
        if any(anchor.get("target", {}).get("name") != target.get("name") for anchor in body_anchors):
            raise ImportError("BodyParts3D attachment registration has conflicting source-body identities")
        sites = sites_by_body.get(body_index, [])
        common_translation: list[float] | None = None
        common_quaternion: list[float] | None = None
        common_scale: float | None = None
        vertices: list[list[float]] = []
        source_member_ids: list[str] = []
        for anchor in body_anchors:
            source = anchor["source"]
            registration = anchor["registration"]
            archive_path, member, obj = _bodyparts_obj_member(
                sources, source["hierarchy"], source["member_id"]
            )
            if archive_path.name != source["archive"] or hashlib.sha256(obj).hexdigest() != source["member_sha256"]:
                raise ImportError("BodyParts3D attachment registration source mesh provenance drifted")
            translation, quaternion, scale = _bodyparts_visual_local_pose(
                registration["source_obj_mm_to_core_inertial_body_m"],
                f"BodyParts3D attachment local transform for {source['member_id']}",
            )
            if common_translation is None:
                common_translation, common_quaternion, common_scale = translation, quaternion, scale
            elif any(
                abs(current - reference) > 2.0e-5
                for current, reference in zip(
                    (*translation, *quaternion, scale),
                    (*common_translation, *common_quaternion, common_scale),
                    strict=True,
                )
            ):
                raise ImportError(
                    "BodyParts3D attachment registration has inconsistent source-to-body transforms"
                )
            rotation = _myosim_matrix_from_quaternion_xyzw(quaternion)
            vertices_mm, _ = _bodyparts_obj_triangles(obj, member)
            vertices.extend(
                _myosim_add(translation, [
                    scale * value for value in _myosim_matrix_vector(
                        rotation, [coordinate * 0.001 for coordinate in vertex]
                    )
                ])
                for vertex in vertices_mm
            )
            source_member_ids.append(source["member_id"])
        if common_translation is None or common_quaternion is None or common_scale is None or not vertices:
            raise ImportError("BodyParts3D attachment registration has no source surface vertices")

        def nearest(point: list[float]) -> tuple[list[float], float]:
            vertex = min(vertices, key=lambda candidate_vertex: sum(
                (point[axis] - candidate_vertex[axis]) ** 2 for axis in range(3)
            ))
            return vertex, math.sqrt(sum((point[axis] - vertex[axis]) ** 2 for axis in range(3)))

        initial_pairs = [(*nearest(point), point) for point in sites]
        initial_distances = [
            distance for _, distance, _ in initial_pairs
            if distance <= max_correspondence_distance_m
        ]
        delta = [0.0, 0.0, 0.0]
        if len(initial_distances) >= 3:
            for _ in range(maximum_iterations):
                pairs = [(*nearest(point), point) for point in sites]
                accepted = [
                    (vertex, point) for vertex, distance, point in pairs
                    if distance <= max_correspondence_distance_m
                ]
                if len(accepted) < 3:
                    break
                correction = [
                    sum(point[axis] - vertex[axis] for vertex, point in accepted) / len(accepted)
                    for axis in range(3)
                ]
                if math.sqrt(sum(value * value for value in correction)) <= 1.0e-5:
                    break
                for axis in range(3):
                    delta[axis] += correction[axis]
                vertices = [
                    [vertex[axis] + correction[axis] for axis in range(3)]
                    for vertex in vertices
                ]
        final_pairs = [(*nearest(point), point) for point in sites]
        final_distances = [
            distance for _, distance, _ in final_pairs
            if distance <= max_correspondence_distance_m
        ]
        initial_median = _bodyparts_attachment_quantile(initial_distances, 0.5) if initial_distances else None
        final_median = _bodyparts_attachment_quantile(final_distances, 0.5) if final_distances else None
        apply = (
            initial_median is not None and final_median is not None
            and final_median + 0.001 < initial_median
        )
        if apply:
            body_rotation = _myosim_matrix_from_quaternion_xyzw(
                target["default_inertial_quaternion_world_xyzw"]
            )
            world_delta = _myosim_matrix_vector(body_rotation, delta)
            for anchor in body_anchors:
                registration = anchor["registration"]
                local_matrix = registration["source_obj_mm_to_core_inertial_body_m"]
                for axis in range(3):
                    local_matrix[axis][3] += delta[axis]
                registration["default_pose_vertex_centroid_world_m"] = _myosim_add(
                    registration["default_pose_vertex_centroid_world_m"], world_delta
                )
                registration["status"] = "inferred_visual_attachment_surface_refinement"
        diagnostics = {
            "method": "iterative_nearest_bodyparts3d_body_surface_union_to_exact_myosim_site_translation_only",
            "maximum_correspondence_distance_m": max_correspondence_distance_m,
            "maximum_iterations": maximum_iterations,
            "source_site_count": len(sites),
            "source_member_count": len(body_anchors),
            "source_member_ids": source_member_ids,
            "accepted_site_count_before": len(initial_distances),
            "accepted_site_count_after": len(final_distances),
            "median_distance_before_m": initial_median,
            "p90_distance_before_m": _bodyparts_attachment_quantile(initial_distances, 0.9) if initial_distances else None,
            "median_distance_after_m": final_median,
            "p90_distance_after_m": _bodyparts_attachment_quantile(final_distances, 0.9) if final_distances else None,
            "translation_delta_core_body_m": delta,
            "applied": apply,
        }
        for anchor in body_anchors:
            anchor["registration"]["attachment_surface_refinement"] = diagnostics
        summary.append({
            "myosim_body": target["name"], "core_body_index": body_index,
            "source_member_count": len(body_anchors), "applied": apply,
            "median_distance_before_m": initial_median,
            "median_distance_after_m": final_median,
        })

    # Compact wrist and distal digit bodies often have no muscle site on the
    # bone itself. Leaving those meshes in the torso-fitted common frame puts
    # otherwise valid carpals, thumbs, and distal phalanges 7--16 cm away from
    # their articulated hand. Use a same-side, same-class body whose
    # attachment refinement succeeded to transfer only its centroid offset in
    # the target inertial-body frame. The exact source mesh, scale, and
    # orientation are preserved; only the unsupported translation is inferred.
    # This is deliberately narrower than a global nearest-body fallback.
    hand_centroid_donors = {
        "scaphoid_r": ("lunate_r", "body_local_centroid_offset"),
        "triquetrum_r": ("lunate_r", "body_local_centroid_offset"),
        "pisiform_r": ("lunate_r", "body_local_centroid_offset"),
        "trapezium_r": ("capitate_r", "body_local_centroid_offset"),
        "trapezoid_r": ("capitate_r", "body_local_centroid_offset"),
        "hamate_r": ("capitate_r", "body_local_centroid_offset"),
        "proximal_thumb_r": ("firstmc_r", "source_chain_displacement"),
        "distal_thumb_r": ("proximal_thumb_r", "source_chain_displacement"),
        "distph3_r": ("midph3_r", "source_chain_displacement"),
        "distph4_r": ("midph4_r", "source_chain_displacement"),
        "distph5_r": ("midph5_r", "source_chain_displacement"),
        "scaphoid_l": ("lunate_l", "body_local_centroid_offset"),
        "triquetrum_l": ("lunate_l", "body_local_centroid_offset"),
        "pisiform_l": ("lunate_l", "body_local_centroid_offset"),
        "trapezium_l": ("capitate_l", "body_local_centroid_offset"),
        "trapezoid_l": ("capitate_l", "body_local_centroid_offset"),
        "hamate_l": ("capitate_l", "body_local_centroid_offset"),
        "proximal_thumb_l": ("firstmc_l", "source_chain_displacement"),
        "distal_thumb_l": ("proximal_thumb_l", "source_chain_displacement"),
        "distph3_l": ("midph3_l", "source_chain_displacement"),
        "distph4_l": ("midph4_l", "source_chain_displacement"),
        "distph5_l": ("midph5_l", "source_chain_displacement"),
    }
    anchors_by_name = {
        body_anchors[0]["target"]["name"]: body_anchors
        for body_anchors in anchors_by_body.values()
    }
    summary_by_name = {record["myosim_body"]: record for record in summary}
    for target_name, (donor_name, fallback_method) in hand_centroid_donors.items():
        target_anchors = anchors_by_name.get(target_name)
        donor_anchors = anchors_by_name.get(donor_name)
        if not target_anchors or not donor_anchors:
            raise ImportError(
                f"BodyParts3D hand centroid fallback is missing {target_name} or {donor_name}"
            )
        target_diagnostics = target_anchors[0]["registration"][
            "attachment_surface_refinement"
        ]
        donor_diagnostics = donor_anchors[0]["registration"][
            "attachment_surface_refinement"
        ]
        if target_diagnostics["applied"]:
            continue
        donor_fallback = donor_anchors[0]["registration"].get(
            "kinematic_neighbor_centroid_fallback"
        )
        donor_supported = donor_diagnostics["applied"] or (
            fallback_method == "source_chain_displacement"
            and isinstance(donor_fallback, dict)
            and donor_fallback.get("applied") is True
        )
        if not donor_supported:
            raise ImportError(
                f"BodyParts3D hand centroid donor {donor_name} was not geometrically anchored"
            )
        donor = donor_anchors[0]
        donor_target = donor["target"]
        donor_registration = donor["registration"]
        target = target_anchors[0]["target"]
        target_rotation = _myosim_matrix_from_quaternion_xyzw(
            target["default_inertial_quaternion_world_xyzw"]
        )
        if fallback_method == "source_chain_displacement":
            desired_world_centroid = _myosim_add(
                donor_registration["default_pose_vertex_centroid_world_m"],
                _myosim_subtract(
                    common_frame_centroid_by_name[target_name],
                    common_frame_centroid_by_name[donor_name],
                ),
            )
            method_description = (
                "same_side_attachment_refined_parent_plus_exact_"
                "bodyparts3d_common_frame_chain_displacement"
            )
        else:
            donor_rotation = _myosim_matrix_from_quaternion_xyzw(
                donor_target["default_inertial_quaternion_world_xyzw"]
            )
            donor_world_offset = _myosim_subtract(
                donor_registration["default_pose_vertex_centroid_world_m"],
                donor_target["default_com_position_world_m"],
            )
            donor_local_offset = _myosim_matrix_vector(
                _matrix_transpose(donor_rotation), donor_world_offset
            )
            desired_world_centroid = _myosim_add(
                target["default_com_position_world_m"],
                _myosim_matrix_vector(target_rotation, donor_local_offset),
            )
            method_description = (
                "same_side_same_class_attachment_refined_body_local_centroid_offset"
            )
        current_world_centroid = target_anchors[0]["registration"][
            "default_pose_vertex_centroid_world_m"
        ]
        world_delta = _myosim_subtract(
            desired_world_centroid, current_world_centroid
        )
        local_delta = _myosim_matrix_vector(
            _matrix_transpose(target_rotation), world_delta
        )
        for anchor in target_anchors:
            registration = anchor["registration"]
            local_matrix = registration["source_obj_mm_to_core_inertial_body_m"]
            for axis in range(3):
                local_matrix[axis][3] += local_delta[axis]
            registration["default_pose_vertex_centroid_world_m"] = _myosim_add(
                registration["default_pose_vertex_centroid_world_m"], world_delta
            )
            registration["status"] = (
                "inferred_visual_kinematic_neighbor_centroid_fallback"
            )
            registration["kinematic_neighbor_centroid_fallback"] = {
                "method": method_description,
                "donor_myosim_body": donor_name,
                "donor_core_body_index": donor_target["core_body_index"],
                "translation_delta_core_body_m": local_delta,
                "applied": True,
            }
        summary_by_name[target_name]["kinematic_neighbor_centroid_fallback"] = {
            "donor_myosim_body": donor_name,
            "applied": True,
        }

    # L5 has valid source sites, but their unconstrained translation does not
    # improve the attachment residual and is therefore (correctly) rejected
    # above.  Leaving it at the original common-frame position after both L4
    # and the sacrum have accepted refinements opens a 16.8 mm visual break.
    # Translate L5 by the mean *world-space* correction of those two immediate
    # anatomical neighbours.  This preserves the exact L5 mesh, orientation,
    # scale, Core owner, and source sites; it adds no joint or articulation.
    # The payload compiler below independently rejects the result unless every
    # cross-body axial surface transition is within the bounded 8 mm gate.
    axial_target_name = "lumbar5"
    axial_donor_names = ("lumbar4", "sacrum")
    axial_target_anchors = anchors_by_name.get(axial_target_name)
    axial_donor_anchors = [anchors_by_name.get(name) for name in axial_donor_names]
    if not axial_target_anchors or any(not anchors for anchors in axial_donor_anchors):
        raise ImportError("BodyParts3D axial fallback is missing L5, L4, or sacrum")
    axial_target_diagnostics = axial_target_anchors[0]["registration"][
        "attachment_surface_refinement"
    ]
    if not axial_target_diagnostics["applied"]:
        donor_world_deltas: list[list[float]] = []
        donor_body_indices: list[int] = []
        for donor_anchors in axial_donor_anchors:
            assert donor_anchors is not None
            donor = donor_anchors[0]
            diagnostics = donor["registration"]["attachment_surface_refinement"]
            if not diagnostics["applied"]:
                raise ImportError("BodyParts3D axial fallback donor was not geometrically anchored")
            donor_rotation = _myosim_matrix_from_quaternion_xyzw(
                donor["target"]["default_inertial_quaternion_world_xyzw"]
            )
            donor_world_deltas.append(_myosim_matrix_vector(
                donor_rotation, diagnostics["translation_delta_core_body_m"]
            ))
            donor_body_indices.append(donor["target"]["core_body_index"])
        world_delta = [
            sum(delta[axis] for delta in donor_world_deltas) / len(donor_world_deltas)
            for axis in range(3)
        ]
        target_rotation = _myosim_matrix_from_quaternion_xyzw(
            axial_target_anchors[0]["target"]["default_inertial_quaternion_world_xyzw"]
        )
        local_delta = _myosim_matrix_vector(
            _matrix_transpose(target_rotation), world_delta
        )
        receipt = {
            "method": "mean_world_translation_of_immediate_axial_neighbors",
            "donor_myosim_bodies": list(axial_donor_names),
            "donor_core_body_indices": donor_body_indices,
            "translation_delta_world_m": world_delta,
            "translation_delta_core_body_m": local_delta,
            "independent_articulation_count": 0,
            "applied": True,
        }
        for anchor in axial_target_anchors:
            registration = anchor["registration"]
            local_matrix = registration["source_obj_mm_to_core_inertial_body_m"]
            for axis in range(3):
                local_matrix[axis][3] += local_delta[axis]
            registration["default_pose_vertex_centroid_world_m"] = _myosim_add(
                registration["default_pose_vertex_centroid_world_m"], world_delta
            )
            registration["status"] = "inferred_visual_axial_neighbor_translation_fallback"
            registration["axial_neighbor_translation_fallback"] = receipt
        summary_by_name[axial_target_name]["axial_neighbor_translation_fallback"] = {
            "donor_myosim_bodies": list(axial_donor_names),
            "applied": True,
        }

    # Restore the distal upper limbs to one coherent BodyParts3D rest frame.
    # Per-body site refinement is useful for mechanics correspondence, but it
    # translated neighbouring visual bones by different amounts and created
    # visible elbow, wrist, metacarpal, and phalanx breaks.  Keep the refined
    # humerus as the side's proximal mechanics anchor, then place every distal
    # body at its exact pre-refinement source-centroid displacement from that
    # humerus.  This is a rest-registration correction: all authored MyoSim
    # joints and the existing hand topology remain unchanged; no independent
    # digit articulation is introduced.
    wrist_hand_target_names = {
        side: {
            specification["myosim_body"]
            for specification in _BODYPARTS_MYOSIM_WRIST_HAND_EXTENSIONS
            if specification["myosim_body"].endswith(f"_{side}")
        } | {f"triquetrum_{side}"}
        for side in ("r", "l")
    }
    for side, root_name in _NUMI_HUMAN_UPPER_LIMB_COHERENT_ROOTS.items():
        forearm_names = {f"radius_{side}", f"ulna_{side}"}
        hand_names = wrist_hand_target_names[side]
        all_names = forearm_names | hand_names
        required_names = all_names | {root_name}
        if any(
            name not in anchors_by_name or name not in common_frame_centroid_by_name
            for name in required_names
        ):
            missing = sorted(
                name for name in required_names
                if name not in anchors_by_name or name not in common_frame_centroid_by_name
            )
            raise ImportError(
                f"BodyParts3D coherent upper-limb correction is missing {', '.join(missing)}"
            )
        root_world_centroid = anchors_by_name[root_name][0]["registration"][
            "default_pose_vertex_centroid_world_m"
        ]
        for target_name in sorted(all_names):
            target_anchors = anchors_by_name[target_name]
            desired_world_centroid = _myosim_add(
                root_world_centroid,
                _myosim_subtract(
                    common_frame_centroid_by_name[target_name],
                    common_frame_centroid_by_name[root_name],
                ),
            )
            applied_world_delta = _myosim_subtract(
                desired_world_centroid,
                target_anchors[0]["registration"][
                    "default_pose_vertex_centroid_world_m"
                ],
            )
            target_rotation = _myosim_matrix_from_quaternion_xyzw(
                target_anchors[0]["target"]["default_inertial_quaternion_world_xyzw"]
            )
            local_delta = _myosim_matrix_vector(
                _matrix_transpose(target_rotation), applied_world_delta
            )
            receipt = {
                "method": "exact_bodyparts3d_common_rest_frame_displacement_from_refined_humerus",
                "side": "right" if side == "r" else "left",
                "root_myosim_body": root_name,
                "translation_delta_world_m": applied_world_delta,
                "translation_delta_core_body_m": local_delta,
                "preserved_group": "complete humerus-to-distal-phalanx BodyParts3D rest arrangement",
                "independent_articulation_count": 0,
                "existing_myosim_articulation_preserved": True,
                "applied": True,
            }
            for anchor in target_anchors:
                registration = anchor["registration"]
                local_matrix = registration["source_obj_mm_to_core_inertial_body_m"]
                for axis in range(3):
                    local_matrix[axis][3] += local_delta[axis]
                registration["default_pose_vertex_centroid_world_m"] = _myosim_add(
                    registration["default_pose_vertex_centroid_world_m"],
                    applied_world_delta,
                )
                registration["status"] = (
                    "inferred_visual_upper_limb_chain_translation_fallback"
                )
                registration["upper_limb_chain_translation_fallback"] = receipt
            summary_by_name[target_name]["upper_limb_chain_translation_fallback"] = {
                "side": receipt["side"],
                "root_myosim_body": root_name,
                "independent_articulation_count": 0,
                "existing_myosim_articulation_preserved": True,
                "applied": True,
            }

    # Apply the same source-coherent rest registration to each lower limb.
    # The femur remains the site-refined mechanics anchor; patella, paired
    # tibia/fibula, hindfoot, midfoot, metatarsals, and toes recover their exact
    # BodyParts3D displacement from it. Existing knee, ankle, subtalar, and MTP
    # articulation remains owned by MyoSim.
    for side, root_name in _NUMI_HUMAN_LOWER_LIMB_COHERENT_ROOTS.items():
        distal_names = {
            f"patella_{side}", f"tibia_{side}", f"talus_{side}",
            f"calcn_{side}", f"toes_{side}",
        }
        required_names = distal_names | {root_name}
        if any(
            name not in anchors_by_name or name not in common_frame_centroid_by_name
            for name in required_names
        ):
            missing = sorted(
                name for name in required_names
                if name not in anchors_by_name or name not in common_frame_centroid_by_name
            )
            raise ImportError(
                f"BodyParts3D coherent lower-limb correction is missing {', '.join(missing)}"
            )
        root_world_centroid = anchors_by_name[root_name][0]["registration"][
            "default_pose_vertex_centroid_world_m"
        ]
        for target_name in sorted(distal_names):
            target_anchors = anchors_by_name[target_name]
            desired_world_centroid = _myosim_add(
                root_world_centroid,
                _myosim_subtract(
                    common_frame_centroid_by_name[target_name],
                    common_frame_centroid_by_name[root_name],
                ),
            )
            applied_world_delta = _myosim_subtract(
                desired_world_centroid,
                target_anchors[0]["registration"][
                    "default_pose_vertex_centroid_world_m"
                ],
            )
            target_rotation = _myosim_matrix_from_quaternion_xyzw(
                target_anchors[0]["target"]["default_inertial_quaternion_world_xyzw"]
            )
            local_delta = _myosim_matrix_vector(
                _matrix_transpose(target_rotation), applied_world_delta
            )
            receipt = {
                "method": "exact_bodyparts3d_common_rest_frame_displacement_from_refined_femur",
                "side": "right" if side == "r" else "left",
                "root_myosim_body": root_name,
                "translation_delta_world_m": applied_world_delta,
                "translation_delta_core_body_m": local_delta,
                "preserved_group": "complete femur-to-distal-toe BodyParts3D rest arrangement",
                "independent_articulation_count": 0,
                "existing_myosim_articulation_preserved": True,
                "applied": True,
            }
            for anchor in target_anchors:
                registration = anchor["registration"]
                local_matrix = registration["source_obj_mm_to_core_inertial_body_m"]
                for axis in range(3):
                    local_matrix[axis][3] += local_delta[axis]
                registration["default_pose_vertex_centroid_world_m"] = _myosim_add(
                    registration["default_pose_vertex_centroid_world_m"],
                    applied_world_delta,
                )
                registration["status"] = (
                    "inferred_visual_lower_limb_coherent_rest_registration"
                )
                registration["lower_limb_coherent_rest_registration"] = receipt
            summary_by_name[target_name]["lower_limb_coherent_rest_registration"] = {
                "side": receipt["side"],
                "root_myosim_body": root_name,
                "existing_myosim_articulation_preserved": True,
                "applied": True,
            }
    candidate["schema"] = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
    candidate["status"] = "inferred_attachment_surface_visual_registration_not_admitted_to_collision_or_physics"
    candidate["attachment_surface_refinement"] = {
        "source_muscle_payload": {"file": muscle_file, "sha256": muscle_sha},
        "records": summary,
    }
    candidate["evidence_boundary"] = (
        "This inferred BodyParts3D/MyoSim surface correspondence is visual-only. "
        "It does not alter source muscle sites or paths, create a source attachment certificate, "
        "or admit collision, contact, skinning, soft-tissue mechanics, or medical validation."
    )
    return candidate


_BODYPARTS_MYOSIM_BONE_VISUAL_MAGIC = b"NHBONES1"
_BODYPARTS_MYOSIM_BONE_VISUAL_ABI = 2
_BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_MAGIC = b"NHTISS2\0"
_BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_ABI = 3
# ``NHTISS2`` remains the compact two-body payload used by the focused legacy
# posterior-chain command.  ``NHTISS3`` adds one explicit source-body binding
# and per-vertex three-way weights for shared tendons.  In particular, the
# calcaneal tendon cannot faithfully be reduced to tibia-to-calcaneus: its
# named gastrocnemius contributors originate on the femur while soleus
# originates on the tibia.  The native renderer accepts both versions so
# existing audited payloads remain inspectable.
_BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_MAGIC = b"NHTISS3\0"
_BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_ABI = 4
# ``NHTISS4`` replaces the record-wide three-body ceiling with a variable
# source-route body table and four sparse influences per vertex.  This matters
# for the hand: one BodyParts3D flexor/extensor surface contains multiple
# digital slips, while MyoSim authors a distinct multi-body path for each
# finger.  Collapsing that surface onto the middle-finger endpoint visibly
# stretched the other slips and was anatomically wrong.
_BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_VISUAL_MAGIC = b"NHTISS4\0"
_BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_VISUAL_ABI = 5
_BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_MAX_BINDINGS = 24
_BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_MAX_INFLUENCES = 4
_BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE = 1
_BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON = 2
_BODYPARTS_MYOSIM_SKIN_VISUAL_MAGIC = b"NHSKIN1\0"
# ABI 4 retains the compact vertex/binding layout and source-surface-local
# weights from ABI 3, but stores the rest normal in the registered world frame.
# Native rendering can then apply each body-relative rotation from that common
# rest frame rather than blending slightly different fitted source-to-body
# normal transforms across every skin triangle.
_BODYPARTS_MYOSIM_SKIN_VISUAL_ABI = 4
_BODYPARTS_MYOSIM_TORSO_ANATOMY_VISUAL_MAGIC = b"NHANAT1\0"
_BODYPARTS_MYOSIM_TORSO_ANATOMY_VISUAL_ABI = 1
_BODYPARTS_MYOSIM_TORSO_ANATOMY_LAYER_ORGAN = 1
_BODYPARTS_MYOSIM_TORSO_ANATOMY_LAYER_VESSEL = 2
_BODYPARTS_MYOSIM_TORSO_ANATOMY_LAYER_NERVE = 3


def _bodyparts_visual_registration_fingerprint(registration_file: Path) -> int:
    """Return a compact compatibility discriminator for paired visual payloads.

    Bone and soft-tissue payloads are separately importable, but their local
    transforms only compose correctly when both were produced from the same
    visual-registration receipt.  This is deliberately not a provenance gate:
    both complete receipt SHA-256 values stay in the adjacent manifests.  The
    compact value only prevents the native renderer from silently combining
    transform sets that have different rest frames.
    """
    return int(sha256(registration_file)[:8], 16)

# This is deliberately a small, exact source-surface bundle for the first
# articulated anatomy presentation.  Every selected surface spans a named
# proximal and distal source body.  The native visual path uses the two body
# rest transforms and a source-mesh longitudinal blend to keep both ends
# continuously posed with the active skeleton.  It is an articulated surface
# binding, not a continuum constitutive model.
_BODYPARTS_MYOSIM_RIGHT_POSTERIOR_CHAIN_TISSUES = (
    ("femur_r", "calcn_r", "right lateral head of gastrocnemius", "FJ1394", _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE),
    ("femur_r", "calcn_r", "right medial head of gastrocnemius", "FJ1397", _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE),
    ("tibia_r", "calcn_r", "right soleus", "FJ1437", _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE),
    ("tibia_r", "calcn_r", "right calcaneal tendon", "FJ1405", _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON),
)


def _bodyparts_vertex_normals(
    vertices: list[tuple[float, float, float]], triangles: list[tuple[int, int, int]], source_name: str,
) -> list[tuple[float, float, float]]:
    """Build unit vertex normals from exact source triangles without smoothing geometry."""
    accum = [[0.0, 0.0, 0.0] for _ in vertices]
    for first, second, third in triangles:
        a, b, c = vertices[first], vertices[second], vertices[third]
        left = [b[index] - a[index] for index in range(3)]
        right = [c[index] - a[index] for index in range(3)]
        normal = [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
        for vertex_index in (first, second, third):
            for axis in range(3):
                accum[vertex_index][axis] += normal[axis]
    center = [sum(vertex[axis] for vertex in vertices) / len(vertices) for axis in range(3)]
    normals: list[tuple[float, float, float]] = []
    for index, normal in enumerate(accum):
        squared = sum(value * value for value in normal)
        if squared <= 1.0e-24:
            normal = [vertices[index][axis] - center[axis] for axis in range(3)]
            squared = sum(value * value for value in normal)
        if squared <= 1.0e-24:
            raise ImportError(f"BodyParts3D mesh has no usable vertex normal: {source_name}")
        magnitude = math.sqrt(squared)
        normals.append(tuple(value / magnitude for value in normal))
    return normals


def _bodyparts_unit_tangent(normal: tuple[float, float, float]) -> tuple[float, float, float]:
    reference = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (0.0, 1.0, 0.0)
    tangent = (
        reference[1] * normal[2] - reference[2] * normal[1],
        reference[2] * normal[0] - reference[0] * normal[2],
        reference[0] * normal[1] - reference[1] * normal[0],
    )
    magnitude = math.sqrt(sum(value * value for value in tangent))
    if magnitude <= 1.0e-12:
        raise ImportError("BodyParts3D vertex normal cannot form a tangent")
    return tuple(value / magnitude for value in tangent)


def _bodyparts_visual_local_pose(matrix: Any, context: str) -> tuple[list[float], list[float], float]:
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise ImportError(f"{context} is not a 4x4 transform")
    rows = [[_finite_scalar(value, context) for value in row] for row in matrix]
    if any(abs(rows[3][axis] - (1.0 if axis == 3 else 0.0)) > 1.0e-6 for axis in range(4)):
        raise ImportError(f"{context} is not affine")
    linear = [row[:3] for row in rows[:3]]
    scales = [math.sqrt(sum(value * value for value in row)) for row in linear]
    scale_m_per_mm = sum(scales) / len(scales)
    if not math.isfinite(scale_m_per_mm) or scale_m_per_mm <= 0.0 or any(abs(value - scale_m_per_mm) > 1.0e-6 * scale_m_per_mm for value in scales):
        raise ImportError(f"{context} has non-uniform scale")
    rotation = [[value / scale_m_per_mm for value in row] for row in linear]
    if abs(_matrix3_determinant(rotation) - 1.0) > 1.0e-5:
        raise ImportError(f"{context} is not a proper rotation and uniform scale")
    quaternion = _quaternion_xyzw_from_matrix(rotation)
    # The source vertices below are authored in metres, not the OBJ's mm.
    return [row[3] for row in rows[:3]], quaternion, scale_m_per_mm / 0.001


def _bodyparts_bounded_vertex_gap(
    first: list[list[float]], second: list[list[float]], maximum_gap_m: float,
    context: str, gate_name: str = "axial continuity",
) -> float:
    """Return an exact vertex witness within a fail-closed distance gate.

    A regular grid avoids the quadratic scan across the larger sacrum/hip
    meshes.  Searching the 27 neighbouring cells is complete for the stated
    gate: any point closer than one cell width must be in one of those cells.
    This is a visual source-surface continuity witness, not a cartilage/contact
    or signed-distance certificate.
    """
    if not first or not second or not math.isfinite(maximum_gap_m) or maximum_gap_m <= 0.0:
        raise ImportError(f"{context} has no bounded source vertices")
    inverse_cell = 1.0 / maximum_gap_m
    buckets: dict[tuple[int, int, int], list[list[float]]] = defaultdict(list)
    for point in second:
        buckets[tuple(math.floor(value * inverse_cell) for value in point)].append(point)
    maximum_squared = maximum_gap_m * maximum_gap_m
    best_squared = math.inf
    for point in first:
        cell = tuple(math.floor(value * inverse_cell) for value in point)
        for offset in product((-1, 0, 1), repeat=3):
            for candidate in buckets.get(tuple(
                cell[axis] + offset[axis] for axis in range(3)
            ), ()):
                squared = sum(
                    (point[axis] - candidate[axis]) ** 2 for axis in range(3)
                )
                if squared < best_squared:
                    best_squared = squared
    if not math.isfinite(best_squared) or best_squared > maximum_squared:
        raise ImportError(
            f"{context} exceeds the {maximum_gap_m * 1000.0:.1f} mm {gate_name} gate"
        )
    return math.sqrt(best_squared)


def bodyparts_myosim_bone_visual_payload(
    sources: Path, anatomy: dict[str, Any], registration_path: Path, output: Path,
) -> dict[str, Any]:
    """Prepare exact visual-skeleton triangles for the native articulated renderer.

    This is an offline source importer.  Its ``.nhbones`` payload is consumed
    by a C++/Metal visual executable, which receives only compact geometry and
    link-local transforms; no Python is involved in rendering or simulation.
    """
    registration_file = registration_path.resolve()
    registration = read_json(registration_file)
    if registration.get("schema") not in {
        "numi.human.bodyparts3d-myosim-bone-registration-candidate.v1",
        "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2",
    }:
        raise ImportError("BodyParts3D visual payload requires a visual-skeleton registration candidate")
    if registration.get("status") not in {
        "provisional_visual_registration_not_admitted_to_collision_or_physics",
        "inferred_attachment_surface_visual_registration_not_admitted_to_collision_or_physics",
    }:
        raise ImportError("BodyParts3D visual payload requires an unmodified visual-only registration candidate")
    source = registration.get("source")
    expected_bodyparts = {
        "id": anatomy.get("source_id"), "version": anatomy.get("version"),
        "archives": anatomy.get("archives"),
    }
    if not isinstance(source, dict) or source.get("bodyparts") != expected_bodyparts:
        raise ImportError("BodyParts3D visual payload registration does not match parsed source provenance")
    myosim = source.get("myosim")
    if not isinstance(myosim, dict) or not isinstance(myosim.get("source"), dict):
        raise ImportError("BodyParts3D visual payload registration has no MyoSim source provenance")
    source_sha = myosim["source"].get("archive_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ImportError("BodyParts3D visual payload registration has no MyoSim source SHA-256")
    anchors = registration.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != len(_BODYPARTS_MYOSIM_BONE_ANCHORS):
        raise ImportError("BodyParts3D visual payload requires the complete visual-skeleton anchor set")
    member_body_indices = {
        str(anchor.get("source", {}).get("member_id")): int(
            anchor.get("target", {}).get("core_body_index", -1)
        )
        for anchor in anchors if isinstance(anchor, dict)
    }
    source_component_enthesis_receipt = _numi_human_source_component_enthesis_receipt(
        registration.get("abdominal_source_component_enthesis_registration"),
        member_body_indices,
        source_sha,
    )
    vertices_payload: list[tuple[float, float, float, float, float, float]] = []
    indices_payload: list[int] = []
    records_payload: list[bytes] = []
    provenance_anchors: list[dict[str, Any]] = []
    toe_member_ids = frozenset(
        member
        for chains in _NUMI_HUMAN_TOE_RIGID_CHAINS.values()
        for chain in chains
        for member in chain
    )
    toe_geometry: dict[str, dict[str, Any]] = {}
    axial_member_ids = frozenset(
        member
        for _, first, second in _NUMI_HUMAN_AXIAL_CONTINUITY_TRANSITIONS
        for member in (first, second)
    )
    axial_world_vertices: dict[str, list[list[float]]] = {}
    upper_limb_member_ids = frozenset(
        member
        for _, first, second, _ in _NUMI_HUMAN_UPPER_LIMB_CONTINUITY_TRANSITIONS
        for member in (first, second)
    )
    hand_member_ids = frozenset(
        member
        for _, first, second in _NUMI_HUMAN_HAND_CONTINUITY_TRANSITIONS
        for member in (first, second)
    )
    knee_member_ids = frozenset(
        member
        for _, first, second in _NUMI_HUMAN_KNEE_CONTINUITY_TRANSITIONS
        for member in (first, second)
    )
    foot_member_ids = frozenset(
        member
        for _, first, second in _NUMI_HUMAN_FOOT_CONTINUITY_TRANSITIONS
        for member in (first, second)
    )
    upper_limb_world_vertices: dict[str, list[list[float]]] = {}
    for stable_id, (specification, anchor) in enumerate(zip(_BODYPARTS_MYOSIM_BONE_ANCHORS, anchors, strict=True), start=1):
        if not isinstance(anchor, dict):
            raise ImportError("BodyParts3D visual payload has an invalid anchor")
        source_record = anchor.get("source")
        target_record = anchor.get("target")
        registration_record = anchor.get("registration")
        if not isinstance(source_record, dict) or not isinstance(target_record, dict) or not isinstance(registration_record, dict):
            raise ImportError("BodyParts3D visual payload anchor is incomplete")
        if source_record.get("hierarchy") != specification["hierarchy"] or source_record.get("member_id") != specification["member_id"] or source_record.get("name") != specification["bodyparts_name"] or target_record.get("name") != specification["myosim_body"]:
            raise ImportError("BodyParts3D visual payload anchor identity drifted")
        core_body_index = target_record.get("core_body_index")
        if not isinstance(core_body_index, int) or core_body_index < 0:
            raise ImportError("BodyParts3D visual payload anchor has an invalid Core body index")
        archive_path, member, obj = _bodyparts_obj_member(sources, specification["hierarchy"], specification["member_id"])
        if source_record.get("archive_sha256") != sha256(archive_path) or source_record.get("member") != member or source_record.get("member_sha256") != hashlib.sha256(obj).hexdigest():
            raise ImportError("BodyParts3D visual payload source mesh provenance drifted")
        vertices_mm, triangles = _bodyparts_obj_triangles(obj, member)
        if source_record.get("vertex_count") != len(vertices_mm) or source_record.get("triangle_count") != len(triangles):
            raise ImportError("BodyParts3D visual payload source topology drifted")
        translation, quaternion, scale = _bodyparts_visual_local_pose(
            registration_record.get("source_obj_mm_to_core_inertial_body_m"),
            f"BodyParts3D visual payload {specification['member_id']} local transform",
        )
        source_member_id = specification["member_id"]
        if (
            source_member_id in toe_member_ids
            or source_member_id in axial_member_ids
            or source_member_id in upper_limb_member_ids
            or source_member_id in hand_member_ids
            or source_member_id in knee_member_ids
            or source_member_id in foot_member_ids
        ):
            rotation = _myosim_matrix_from_quaternion_xyzw(quaternion)
            local_vertices = [
                _myosim_add(translation, [
                    scale * value for value in _myosim_matrix_vector(
                        rotation, [coordinate * 0.001 for coordinate in vertex],
                    )
                ])
                for vertex in vertices_mm
            ]
        if source_member_id in toe_member_ids:
            if source_member_id in toe_geometry:
                raise ImportError("BodyParts3D toe rigid chain duplicates a source member")
            toe_geometry[source_member_id] = {
                "myosim_body": specification["myosim_body"],
                "core_body_index": core_body_index,
                "local_pose": (*translation, *quaternion, scale),
                "vertices_core_body_m": local_vertices,
            }
        if source_member_id in axial_member_ids:
            if source_member_id in axial_world_vertices:
                raise ImportError("BodyParts3D axial continuity duplicates a source member")
            body_rotation = _myosim_matrix_from_quaternion_xyzw(
                target_record["default_inertial_quaternion_world_xyzw"]
            )
            body_position = target_record["default_com_position_world_m"]
            axial_world_vertices[source_member_id] = [
                _myosim_add(
                    body_position, _myosim_matrix_vector(body_rotation, point)
                )
                for point in local_vertices
            ]
        if (
            source_member_id in upper_limb_member_ids
            or source_member_id in hand_member_ids
            or source_member_id in knee_member_ids
            or source_member_id in foot_member_ids
        ):
            if source_member_id in upper_limb_world_vertices:
                raise ImportError("BodyParts3D upper-limb continuity duplicates a source member")
            body_rotation = _myosim_matrix_from_quaternion_xyzw(
                target_record["default_inertial_quaternion_world_xyzw"]
            )
            body_position = target_record["default_com_position_world_m"]
            upper_limb_world_vertices[source_member_id] = [
                _myosim_add(
                    body_position, _myosim_matrix_vector(body_rotation, point)
                )
                for point in local_vertices
            ]
        normals = _bodyparts_vertex_normals(vertices_mm, triangles, member)
        first_vertex = len(vertices_payload)
        first_index = len(indices_payload)
        for vertex, normal in zip(vertices_mm, normals, strict=True):
            vertices_payload.append((
                vertex[0] * 0.001, vertex[1] * 0.001, vertex[2] * 0.001,
                normal[0], normal[1], normal[2],
            ))
        indices_payload.extend(first_vertex + index for triangle in triangles for index in triangle)
        records_payload.append(struct.pack(
            "<6I8f", core_body_index, first_vertex, len(vertices_mm), first_index,
            len(triangles) * 3, stable_id, *translation, *quaternion, scale,
        ))
        provenance_anchors.append({
            "member_id": specification["member_id"], "member_sha256": source_record["member_sha256"],
            "core_body_index": core_body_index, "myosim_body": specification["myosim_body"],
            "vertex_count": len(vertices_mm), "triangle_count": len(triangles),
        })
    axial_transitions: list[dict[str, Any]] = []
    for name, first_member_id, second_member_id in _NUMI_HUMAN_AXIAL_CONTINUITY_TRANSITIONS:
        first = axial_world_vertices.get(first_member_id)
        second = axial_world_vertices.get(second_member_id)
        if first is None or second is None:
            raise ImportError(f"BodyParts3D axial continuity transition {name} is incomplete")
        gap_m = _bodyparts_bounded_vertex_gap(
            first, second, _NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
            f"BodyParts3D axial continuity transition {name}",
        )
        axial_transitions.append({
            "name": name,
            "source_member_ids": [first_member_id, second_member_id],
            "minimum_vertex_gap_m": gap_m,
            "maximum_allowed_gap_m": _NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
            "status": "bounded_default_pose_visual_continuity_witness",
        })
    upper_limb_transitions: list[dict[str, Any]] = []
    for (
        name, first_member_id, second_member_id, maximum_gap_m
    ) in _NUMI_HUMAN_UPPER_LIMB_CONTINUITY_TRANSITIONS:
        first = upper_limb_world_vertices.get(first_member_id)
        second = upper_limb_world_vertices.get(second_member_id)
        if first is None or second is None:
            raise ImportError(
                f"BodyParts3D upper-limb continuity transition {name} is incomplete"
            )
        gap_m = _bodyparts_bounded_vertex_gap(
            first, second, maximum_gap_m,
            f"BodyParts3D upper-limb continuity transition {name}",
            "upper-limb continuity",
        )
        upper_limb_transitions.append({
            "name": name,
            "source_member_ids": [first_member_id, second_member_id],
            "minimum_vertex_gap_m": gap_m,
            "maximum_allowed_gap_m": maximum_gap_m,
            "status": "bounded_default_pose_visual_continuity_witness",
        })
    hand_transitions: list[dict[str, Any]] = []
    for name, first_member_id, second_member_id in _NUMI_HUMAN_HAND_CONTINUITY_TRANSITIONS:
        first = upper_limb_world_vertices.get(first_member_id)
        second = upper_limb_world_vertices.get(second_member_id)
        if first is None or second is None:
            raise ImportError(f"BodyParts3D hand continuity transition {name} is incomplete")
        gap_m = _bodyparts_bounded_vertex_gap(
            first, second, _NUMI_HUMAN_HAND_CONTINUITY_MAXIMUM_GAP_M,
            f"BodyParts3D hand continuity transition {name}", "hand continuity",
        )
        hand_transitions.append({
            "name": name,
            "source_member_ids": [first_member_id, second_member_id],
            "minimum_vertex_gap_m": gap_m,
            "maximum_allowed_gap_m": _NUMI_HUMAN_HAND_CONTINUITY_MAXIMUM_GAP_M,
            "status": "bounded_default_pose_visual_continuity_witness",
        })
    knee_transitions: list[dict[str, Any]] = []
    for name, first_member_id, second_member_id in _NUMI_HUMAN_KNEE_CONTINUITY_TRANSITIONS:
        first = upper_limb_world_vertices.get(first_member_id)
        second = upper_limb_world_vertices.get(second_member_id)
        if first is None or second is None:
            raise ImportError(f"BodyParts3D knee continuity transition {name} is incomplete")
        gap_m = _bodyparts_bounded_vertex_gap(
            first, second, _NUMI_HUMAN_KNEE_CONTINUITY_MAXIMUM_GAP_M,
            f"BodyParts3D knee continuity transition {name}", "knee continuity",
        )
        knee_transitions.append({
            "name": name,
            "source_member_ids": [first_member_id, second_member_id],
            "minimum_vertex_gap_m": gap_m,
            "maximum_allowed_gap_m": _NUMI_HUMAN_KNEE_CONTINUITY_MAXIMUM_GAP_M,
            "status": "bounded_default_pose_visual_continuity_witness",
        })
    foot_transitions: list[dict[str, Any]] = []
    for name, first_member_id, second_member_id in _NUMI_HUMAN_FOOT_CONTINUITY_TRANSITIONS:
        first = upper_limb_world_vertices.get(first_member_id)
        second = upper_limb_world_vertices.get(second_member_id)
        if first is None or second is None:
            raise ImportError(f"BodyParts3D foot continuity transition {name} is incomplete")
        gap_m = _bodyparts_bounded_vertex_gap(
            first, second, _NUMI_HUMAN_FOOT_CONTINUITY_MAXIMUM_GAP_M,
            f"BodyParts3D foot continuity transition {name}", "foot continuity",
        )
        foot_transitions.append({
            "name": name,
            "source_member_ids": [first_member_id, second_member_id],
            "minimum_vertex_gap_m": gap_m,
            "maximum_allowed_gap_m": _NUMI_HUMAN_FOOT_CONTINUITY_MAXIMUM_GAP_M,
            "status": "bounded_default_pose_visual_continuity_witness",
        })
    toe_rigid_compounds: list[dict[str, Any]] = []
    for myosim_body, chains in _NUMI_HUMAN_TOE_RIGID_CHAINS.items():
        for digit, member_ids in enumerate(chains, start=1):
            records = [toe_geometry.get(member_id) for member_id in member_ids]
            if any(record is None for record in records):
                raise ImportError(
                    f"BodyParts3D {myosim_body} digit {digit} rigid chain is incomplete"
                )
            typed_records = [record for record in records if record is not None]
            core_body_indices = {record["core_body_index"] for record in typed_records}
            if (
                {record["myosim_body"] for record in typed_records} != {myosim_body}
                or len(core_body_indices) != 1
            ):
                raise ImportError(
                    f"BodyParts3D {myosim_body} digit {digit} rigid chain has split body ownership"
                )
            reference_pose = typed_records[0]["local_pose"]
            if any(
                any(abs(value - reference) > 1.0e-9 for value, reference in zip(
                    record["local_pose"], reference_pose, strict=True,
                ))
                for record in typed_records[1:]
            ):
                raise ImportError(
                    f"BodyParts3D {myosim_body} digit {digit} rigid chain has inconsistent local transforms"
                )
            adjacent_surface_gaps_m = [
                min(
                    math.dist(first, second)
                    for first in typed_records[index]["vertices_core_body_m"]
                    for second in typed_records[index + 1]["vertices_core_body_m"]
                )
                for index in range(len(typed_records) - 1)
            ]
            maximum_gap_m = max(adjacent_surface_gaps_m)
            if maximum_gap_m > _NUMI_HUMAN_HALLUX_RIGID_COMPOUND_MAXIMUM_GAP_M:
                raise ImportError(
                    f"BodyParts3D {myosim_body} digit {digit} rigid source chain is disconnected"
                )
            toe_rigid_compounds.append({
                "myosim_body": myosim_body,
                "core_body_index": next(iter(core_body_indices)),
                "digit": digit,
                "source_member_ids": list(member_ids),
                "distal_phalanx_member_id": member_ids[-1],
                "adjacent_surface_gaps_m": adjacent_surface_gaps_m,
                "maximum_adjacent_surface_gap_m": maximum_gap_m,
                "maximum_allowed_adjacent_surface_gap_m": (
                    _NUMI_HUMAN_HALLUX_RIGID_COMPOUND_MAXIMUM_GAP_M
                ),
                "independent_articulation_count": 0,
                "binding": "one shared existing MyoSim toes rigid-body transform",
            })
    if len(vertices_payload) > 0xFFFFFFFF or len(indices_payload) > 0xFFFFFFFF:
        raise ImportError("BodyParts3D visual payload exceeds the uint32 native renderer capacity")
    registration_fingerprint = _bodyparts_visual_registration_fingerprint(registration_file)
    header = struct.pack(
        "<8s5I32s", _BODYPARTS_MYOSIM_BONE_VISUAL_MAGIC, _BODYPARTS_MYOSIM_BONE_VISUAL_ABI,
        len(records_payload), len(vertices_payload), len(indices_payload), registration_fingerprint,
        bytes.fromhex(source_sha),
    )
    payload = b"".join([
        header, *records_payload,
        b"".join(struct.pack("<6f", *vertex) for vertex in vertices_payload),
        struct.pack(f"<{len(indices_payload)}I", *indices_payload),
    ])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "bodyparts3d-myosim-major-bones.nhbones"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-major-bone-visual-payload.v1",
        "payload": {
            "file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
            "magic": _BODYPARTS_MYOSIM_BONE_VISUAL_MAGIC.decode("ascii"), "payload_abi": _BODYPARTS_MYOSIM_BONE_VISUAL_ABI,
            "registration_fingerprint32": f"{registration_fingerprint:08x}",
            "bone_count": len(records_payload), "vertex_count": len(vertices_payload), "index_count": len(indices_payload),
        },
        "source": {
            "registration": {"file": registration_file.name, "sha256": sha256(registration_file)},
            "bodyparts": expected_bodyparts, "myosim_source_archive_sha256": source_sha,
            "anchors": provenance_anchors,
        },
        "runtime_binding": "one source-local bone instance per Core articulated inertial body; local translation, rotation, and uniform scale are carried in the native payload",
        **({
            "source_component_enthesis_registration": source_component_enthesis_receipt,
        } if source_component_enthesis_receipt is not None else {}),
        "toe_rigid_compounds": toe_rigid_compounds,
        "hallux_rigid_compounds": [
            record for record in toe_rigid_compounds if record["digit"] == 1
        ],
        "axial_continuity": {
            "transitions": axial_transitions,
            "maximum_transition_gap_m": max(
                record["minimum_vertex_gap_m"] for record in axial_transitions
            ),
            "maximum_allowed_gap_m": _NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
            "independent_articulation_count": 0,
            "evidence_boundary": (
                "Default-pose transformed source-vertex proximity catches gross visual separation; "
                "it is not an intervertebral-disc, cartilage, ligament, contact, or clinical certificate."
            ),
        },
        "upper_limb_continuity": {
            "transitions": upper_limb_transitions,
            "maximum_transition_gap_m": max(
                record["minimum_vertex_gap_m"] for record in upper_limb_transitions
            ),
            "independent_articulation_count": 0,
            "group_correction": (
                "exact BodyParts3D common-rest centroid displacement from each side's "
                "site-refined humerus; existing MyoSim articulations are preserved"
            ),
            "evidence_boundary": (
                "Default-pose transformed source-vertex proximity catches gross shoulder, "
                "elbow, and wrist separation. It does not establish cartilage, ligament, "
                "tendon material, contact, load transfer, or clinical registration."
            ),
        },
        "hand_digit_continuity": {
            "transitions": hand_transitions,
            "maximum_transition_gap_m": max(
                record["minimum_vertex_gap_m"] for record in hand_transitions
            ),
            "maximum_allowed_gap_m": _NUMI_HUMAN_HAND_CONTINUITY_MAXIMUM_GAP_M,
            "existing_myosim_articulation_preserved": True,
            "evidence_boundary": (
                "Default-pose source-vertex proximity verifies carpal-to-metacarpal and "
                "metacarpal-to-distal-phalanx visual continuity. It is not cartilage, "
                "ligament, contact, tendon-material, or clinical evidence."
            ),
        },
        "knee_continuity": {
            "transitions": knee_transitions,
            "maximum_transition_gap_m": max(
                record["minimum_vertex_gap_m"] for record in knee_transitions
            ),
            "maximum_allowed_gap_m": _NUMI_HUMAN_KNEE_CONTINUITY_MAXIMUM_GAP_M,
            "evidence_boundary": (
                "Default-pose femur/tibia and femur/patella surface proximity catches gross "
                "visual separation. It does not qualify meniscus, cartilage, ligament, "
                "patellofemoral contact, or dynamic knee loading."
            ),
        },
        "foot_continuity": {
            "transitions": foot_transitions,
            "maximum_transition_gap_m": max(
                record["minimum_vertex_gap_m"] for record in foot_transitions
            ),
            "maximum_allowed_gap_m": _NUMI_HUMAN_FOOT_CONTINUITY_MAXIMUM_GAP_M,
            "existing_myosim_articulation_preserved": True,
            "evidence_boundary": (
                "Default-pose source-vertex proximity verifies ankle mortise, hindfoot, "
                "midfoot, and tarsometatarsal visual continuity. It is not cartilage, "
                "ligament, plantar fascia, contact, or clinical evidence."
            ),
        },
        "status": "native_visual_binding_input_not_collision_or_physics",
        "evidence_boundary": "The payload contains triangle surfaces for a provisional bone visual only. It does not create colliders, skinning weights, soft-tissue mechanics, muscle attachments, or a medical registration claim.",
    }
    write_json(output / "bodyparts3d-myosim-major-bones.manifest.json", manifest)
    return manifest


def bodyparts_myosim_right_posterior_chain_visual_payload(
    sources: Path, anatomy: dict[str, Any], registration_path: Path, output: Path,
) -> dict[str, Any]:
    """Package exact posterior-calf source surfaces for native two-body posing.

    The compact payload is deliberately independent of ``NHBONES1``: it has
    explicit muscle/tendon layer identities while preserving the same fitted
    BodyParts3D-to-MyoSim rest frame as the visual skeleton.  Its per-vertex
    two-body blend is kinematic presentation data, never a substitute for a
    muscle/tendon continuum solve or an attachment-force transfer.
    """
    registration_file = registration_path.resolve()
    registration = read_json(registration_file)
    if registration.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2":
        raise ImportError("BodyParts3D posterior-chain visual payload requires a v2 visual-skeleton registration")
    if registration.get("status") not in {
        "provisional_visual_registration_not_admitted_to_collision_or_physics",
        "inferred_attachment_surface_visual_registration_not_admitted_to_collision_or_physics",
    }:
        raise ImportError("BodyParts3D posterior-chain visual payload requires a supported visual-only registration")
    source = registration.get("source")
    expected_bodyparts = {
        "id": anatomy.get("source_id"), "version": anatomy.get("version"),
        "archives": anatomy.get("archives"),
    }
    if not isinstance(source, dict) or source.get("bodyparts") != expected_bodyparts:
        raise ImportError("BodyParts3D posterior-chain payload registration does not match parsed source provenance")
    myosim = source.get("myosim")
    if not isinstance(myosim, dict) or not isinstance(myosim.get("source"), dict):
        raise ImportError("BodyParts3D posterior-chain payload registration has no MyoSim source provenance")
    source_sha = myosim["source"].get("archive_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ImportError("BodyParts3D posterior-chain payload has no MyoSim source SHA-256")
    coordinates = registration.get("coordinate_system")
    global_matrix = coordinates.get("global_source_mm_to_myosim_world_m") if isinstance(coordinates, dict) else None
    if not isinstance(global_matrix, list) or len(global_matrix) != 4:
        raise ImportError("BodyParts3D posterior-chain payload has no global rest-frame registration")
    anchors = registration.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != len(_BODYPARTS_MYOSIM_BONE_ANCHORS):
        raise ImportError("BodyParts3D posterior-chain payload requires the complete visual-skeleton anchor set")
    targets: dict[str, dict[str, Any]] = {}
    secondary_bone_sources: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        if not isinstance(anchor, dict) or not isinstance(anchor.get("target"), dict) or \
                not isinstance(anchor.get("source"), dict):
            raise ImportError("BodyParts3D posterior-chain payload has an incomplete registration target")
        target = anchor["target"]
        source_record = anchor["source"]
        name = target.get("name")
        if isinstance(name, str) and name not in targets:
            targets[name] = target
        if isinstance(name, str):
            secondary_bone_sources.setdefault(name, source_record)
    vertices_payload: list[tuple[float, float, float, float, float, float, float]] = []
    indices_payload: list[int] = []
    records_payload: list[bytes] = []
    provenance: list[dict[str, Any]] = []
    for stable_id, (primary_target_name, secondary_target_name, label, member_id, layer) in enumerate(
        _BODYPARTS_MYOSIM_RIGHT_POSTERIOR_CHAIN_TISSUES, start=1
    ):
        primary_target = targets.get(primary_target_name)
        secondary_target = targets.get(secondary_target_name)
        if not isinstance(primary_target, dict) or not isinstance(secondary_target, dict):
            raise ImportError(
                "BodyParts3D posterior-chain payload has no two-body target for " + label
            )
        primary_body_index = primary_target.get("core_body_index")
        secondary_body_index = secondary_target.get("core_body_index")
        if not isinstance(primary_body_index, int) or primary_body_index < 0 or \
                not isinstance(secondary_body_index, int) or secondary_body_index < 0 or \
                primary_body_index == secondary_body_index:
            raise ImportError(f"BodyParts3D posterior-chain payload {label} has invalid two-body targets")
        primary_body_position = _myosim_vector(
            primary_target.get("default_com_position_world_m"),
            f"BodyParts3D posterior-chain primary target {primary_target_name} position",
        )
        secondary_body_position = _myosim_vector(
            secondary_target.get("default_com_position_world_m"),
            f"BodyParts3D posterior-chain secondary target {secondary_target_name} position",
        )
        primary_body_quaternion = list(primary_target.get("default_inertial_quaternion_world_xyzw", []))
        secondary_body_quaternion = list(secondary_target.get("default_inertial_quaternion_world_xyzw", []))
        _myosim_matrix_from_quaternion_xyzw(primary_body_quaternion)
        _myosim_matrix_from_quaternion_xyzw(secondary_body_quaternion)
        archive_path, member, obj = _bodyparts_obj_member(sources, "is_a", member_id)
        vertices_mm, triangles = _bodyparts_obj_triangles(obj, member)
        normals = _bodyparts_vertex_normals(vertices_mm, triangles, member)
        primary_local_matrix = _bodyparts_local_registration_matrix(
            global_matrix, primary_body_position, primary_body_quaternion,
        )
        secondary_local_matrix = _bodyparts_local_registration_matrix(
            global_matrix, secondary_body_position, secondary_body_quaternion,
        )
        primary_translation, primary_quaternion, primary_scale = _bodyparts_visual_local_pose(
            primary_local_matrix, f"BodyParts3D posterior-chain payload {member_id} primary local transform",
        )
        secondary_translation, secondary_quaternion, secondary_scale = _bodyparts_visual_local_pose(
            secondary_local_matrix, f"BodyParts3D posterior-chain payload {member_id} secondary local transform",
        )
        body_axis = _myosim_subtract(secondary_body_position, primary_body_position)
        body_axis_squared = sum(value * value for value in body_axis)
        if body_axis_squared <= 1.0e-10:
            raise ImportError(f"BodyParts3D posterior-chain payload {label} has coincident body centres")
        global_vertices_m = [
            [
                sum(global_matrix[row][column] * vertex[column] for column in range(3)) + global_matrix[row][3]
                for row in range(3)
            ]
            for vertex in vertices_mm
        ]
        projections = [
            sum((vertex[axis] - primary_body_position[axis]) * body_axis[axis] for axis in range(3))
            for vertex in global_vertices_m
        ]
        projection_minimum = min(projections)
        projection_maximum = max(projections)
        if projection_maximum - projection_minimum <= 1.0e-6:
            raise ImportError(f"BodyParts3D posterior-chain payload {label} has no two-body blend extent")
        primary_weights = [
            max(0.0, min(1.0, (projection_maximum - projection) /
                            (projection_maximum - projection_minimum)))
            for projection in projections
        ]
        first_vertex = len(vertices_payload)
        first_index = len(indices_payload)
        attachment_weight_lock: dict[str, Any] | None = None
        if layer == _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON:
            bone_source = secondary_bone_sources.get(secondary_target_name)
            if not isinstance(bone_source, dict):
                raise ImportError(
                    f"BodyParts3D posterior-chain tendon {member_id} has no named secondary-bone source mesh"
                )
            bone_member_id, bone_hierarchy = bone_source.get("member_id"), bone_source.get("hierarchy")
            if not isinstance(bone_member_id, str) or not isinstance(bone_hierarchy, str):
                raise ImportError(
                    f"BodyParts3D posterior-chain tendon {member_id} has an invalid secondary-bone source mesh"
                )
            _, bone_member, bone_obj = _bodyparts_obj_member(sources, bone_hierarchy, bone_member_id)
            bone_vertices_mm, bone_triangles = _bodyparts_obj_triangles(bone_obj, bone_member)
            bone_vertices_world_m = [
                [
                    sum(global_matrix[row][column] * vertex[column] for column in range(3)) +
                    global_matrix[row][3]
                    for row in range(3)
                ]
                for vertex in bone_vertices_mm
            ]
            primary_weights, attachment_weight_lock = _bodyparts_secondary_attachment_weight_lock(
                global_vertices_m, primary_weights, bone_vertices_world_m, bone_triangles,
            )
            attachment_weight_lock.update({
                "secondary_body": secondary_target_name,
                "secondary_bone_member_id": bone_member_id,
                "secondary_bone_member": bone_member,
                "secondary_bone_member_sha256": hashlib.sha256(bone_obj).hexdigest(),
            })
        for vertex, normal, primary_weight in zip(vertices_mm, normals, primary_weights, strict=True):
            vertices_payload.append((
                vertex[0] * 0.001, vertex[1] * 0.001, vertex[2] * 0.001,
                normal[0], normal[1], normal[2], primary_weight,
            ))
        indices_payload.extend(first_vertex + index for triangle in triangles for index in triangle)
        records_payload.append(struct.pack(
            "<8I16f", primary_body_index, secondary_body_index, first_vertex, len(vertices_mm), first_index,
            len(triangles) * 3, stable_id, layer,
            *primary_translation, *primary_quaternion, primary_scale,
            *secondary_translation, *secondary_quaternion, secondary_scale,
        ))
        provenance.append({
            "stable_id": stable_id, "member_id": member_id, "member": member,
            "member_sha256": hashlib.sha256(obj).hexdigest(),
            "label": label, "layer": "muscle" if layer == _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE else "tendon",
            "primary_myosim_body": primary_target_name, "primary_core_body_index": primary_body_index,
            "secondary_myosim_body": secondary_target_name, "secondary_core_body_index": secondary_body_index,
            "primary_weight_range": [0.0, 1.0],
            "vertex_count": len(vertices_mm), "triangle_count": len(triangles),
        })
        if attachment_weight_lock is not None:
            provenance[-1]["secondary_attachment_weight_lock"] = attachment_weight_lock
    if len(vertices_payload) > 0xFFFFFFFF or len(indices_payload) > 0xFFFFFFFF:
        raise ImportError("BodyParts3D posterior-chain payload exceeds the uint32 native renderer capacity")
    registration_fingerprint = _bodyparts_visual_registration_fingerprint(registration_file)
    header = struct.pack(
        "<8s5I32s", _BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_MAGIC,
        _BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_ABI, len(records_payload),
        len(vertices_payload), len(indices_payload), registration_fingerprint,
        bytes.fromhex(source_sha),
    )
    payload = b"".join([
        header, *records_payload,
        b"".join(struct.pack("<7f", *vertex) for vertex in vertices_payload),
        struct.pack(f"<{len(indices_payload)}I", *indices_payload),
    ])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "bodyparts3d-myosim-right-posterior-chain.nhtissue"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-right-posterior-chain-visual-payload.v2",
        "payload": {
            "file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
            "magic": _BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_MAGIC.rstrip(b"\0").decode("ascii"),
            "payload_abi": _BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_ABI,
            "registration_fingerprint32": f"{registration_fingerprint:08x}",
            "surface_count": len(records_payload), "vertex_count": len(vertices_payload),
            "index_count": len(indices_payload),
        },
        "source": {
            "registration": {"file": registration_file.name, "sha256": sha256(registration_file)},
            "bodyparts": expected_bodyparts, "myosim_source_archive_sha256": source_sha,
            "surfaces": provenance,
        },
        "runtime_binding": (
            "exact BodyParts3D source surfaces use the visual-skeleton common rest frame and a "
            "per-vertex two-body linear blend in named Core articulated body frames; gastrocnemius "
            "spans femur-to-calcaneus while soleus and the calcaneal tendon span tibia-to-calcaneus; "
            "the calcaneal-tendon insertion uses an exact source-triangle proximity secondary-calcaneus weight lock"
        ),
        "status": "native_two_body_kinematic_surface_binding_input_not_collision_or_physics",
        "evidence_boundary": (
            "This payload exposes exact posterior-calf muscle and calcaneal-tendon triangles at the "
            "registered source default pose. Its two-body kinematic surface binding preserves render "
            "continuity as the named skeleton moves, but does not provide a muscle/tendon continuum "
            "solve, surface-attachment force transfer, collision/contact, or medical registration."
        ),
    }
    write_json(output / "bodyparts3d-myosim-right-posterior-chain.manifest.json", manifest)
    return manifest


def _bodyparts_mirror_surface_specification(specification: dict[str, Any]) -> dict[str, Any]:
    """Mirror one right-side audited surface-map row without guessing anatomy."""
    mirrored = json.loads(json.dumps(specification))
    member_id = mirrored.get("member_id")
    if not isinstance(member_id, str) or not re.fullmatch(r"FJ[0-9]+", member_id):
        raise ImportError("BodyParts3D surface-map mirror has an invalid right-side member")
    mirrored["member_id"] = member_id + "M"
    source_name = mirrored.get("source_name")
    if not isinstance(source_name, str) or "right" not in source_name:
        raise ImportError("BodyParts3D surface-map mirror has no right-side source name")
    mirrored["source_name"] = source_name.replace("right", "left")

    def mirror_name(value: Any, context: str) -> str:
        if not isinstance(value, str) or not value:
            raise ImportError("BodyParts3D surface-map mirror has an invalid " + context)
        return value[:-2] + "_l" if value.endswith("_r") else value + "_l"

    muscles = mirrored.get("myosim_muscles")
    if not isinstance(muscles, list):
        raise ImportError("BodyParts3D surface-map mirror has no MyoSim muscle list")
    mirrored["myosim_muscles"] = [mirror_name(value, "MyoSim muscle") for value in muscles]
    for key in ("primary_body", "secondary_body"):
        if key in mirrored:
            mirrored[key] = mirror_name(mirrored[key], key)
    mirrored.pop("mirror", None)
    return mirrored


def _bodyparts_myosim_surface_specifications() -> list[dict[str, Any]]:
    """Read the versioned semantic table used for full-body surface binding."""
    mapping = read_json(REPOSITORY_ROOT / "config/bodyparts3d-myosim-surface-map.v1.json")
    if mapping.get("schema") != "numi.human.bodyparts3d-myosim-surface-map.v1":
        raise ImportError("BodyParts3D/MyoSim surface map schema is unsupported")
    entries = mapping.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ImportError("BodyParts3D/MyoSim surface map has no entries")
    result: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise ImportError("BodyParts3D/MyoSim surface map has an invalid entry")
        result.append(dict(raw))
        if raw.get("mirror") is True:
            result.append(_bodyparts_mirror_surface_specification(raw))
        elif raw.get("mirror") not in (None, False):
            raise ImportError("BodyParts3D/MyoSim surface map mirror must be boolean")
    members = [entry.get("member_id") for entry in result]
    if any(not isinstance(member, str) or not re.fullmatch(r"FJ[0-9]+M?", member) for member in members):
        raise ImportError("BodyParts3D/MyoSim surface map has an invalid member identity")
    if len(set(members)) != len(members):
        raise ImportError("BodyParts3D/MyoSim surface map duplicates a source surface")
    return result


def _bodyparts_secondary_attachment_weight_lock(
    tissue_vertices_world_m: list[list[float]],
    primary_weights: list[float],
    secondary_bone_vertices_world_m: list[list[float]],
    secondary_bone_triangles: list[tuple[int, int, int]] | None = None,
    lock_radius_m: float = 0.003,
    feather_radius_m: float = 0.015,
) -> tuple[list[float], dict[str, Any]]:
    """Lock a source tendon insertion to its named source bone mesh.

    The two-body interpolation used by the broad surface package is a useful
    presentation approximation for a crossing muscle belly, but endpoint
    weights inferred from body-centre projection can leave a visible tendon
    insertion moving partly with the wrong body.  For a named tendon surface,
    retain the exact source mesh and use close rest-frame proximity to the
    named *secondary* bone *surface* to force that endpoint's weight to zero.
    The transition is feathered over a short source-space band, so the result
    does not introduce a hard seam in the rendered surface.

    This is still visual kinematic binding.  It neither alters the MyoSim
    muscle path nor establishes a mechanical or medical attachment.
    """
    if len(tissue_vertices_world_m) != len(primary_weights) or not tissue_vertices_world_m:
        raise ImportError("BodyParts3D tendon attachment lock has inconsistent source vertices")
    if not secondary_bone_vertices_world_m:
        raise ImportError("BodyParts3D tendon attachment lock has no secondary bone vertices")
    if (
        not math.isfinite(lock_radius_m) or not math.isfinite(feather_radius_m)
        or lock_radius_m <= 0.0 or feather_radius_m <= lock_radius_m
    ):
        raise ImportError("BodyParts3D tendon attachment lock radii are invalid")
    cell_size_m = feather_radius_m
    grid: dict[tuple[int, int, int], list[list[float]]] = {}
    for vertex in secondary_bone_vertices_world_m:
        if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex):
            raise ImportError("BodyParts3D tendon attachment lock has a non-finite bone vertex")
        key = tuple(math.floor(value / cell_size_m) for value in vertex)
        grid.setdefault(key, []).append(vertex)
    surface_triangles: list[tuple[list[float], list[float], list[float]]] = []
    if secondary_bone_triangles is not None:
        for triangle in secondary_bone_triangles:
            if len(triangle) != 3 or any(
                not isinstance(index, int) or not 0 <= index < len(secondary_bone_vertices_world_m)
                for index in triangle
            ):
                raise ImportError("BodyParts3D tendon attachment lock has an invalid bone triangle")
            surface_triangles.append(tuple(secondary_bone_vertices_world_m[index] for index in triangle))
        if not surface_triangles:
            raise ImportError("BodyParts3D tendon attachment lock has no secondary-bone triangles")

    def point_triangle_squared_distance(
        point: list[float], triangle: tuple[list[float], list[float], list[float]]
    ) -> float:
        # Christer Ericson, Real-Time Collision Detection: exact closest point
        # on a triangle, expressed directly in the source registration frame.
        first, second, third = triangle
        ab = [second[index] - first[index] for index in range(3)]
        ac = [third[index] - first[index] for index in range(3)]
        ap = [point[index] - first[index] for index in range(3)]
        dot = lambda left, right: sum(left[index] * right[index] for index in range(3))
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return dot(ap, ap)
        bp = [point[index] - second[index] for index in range(3)]
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return dot(bp, bp)
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            factor = d1 / (d1 - d3)
            difference = [ap[index] - factor * ab[index] for index in range(3)]
            return dot(difference, difference)
        cp = [point[index] - third[index] for index in range(3)]
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return dot(cp, cp)
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            factor = d2 / (d2 - d6)
            difference = [ap[index] - factor * ac[index] for index in range(3)]
            return dot(difference, difference)
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
            factor = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            bc = [third[index] - second[index] for index in range(3)]
            difference = [bp[index] - factor * bc[index] for index in range(3)]
            return dot(difference, difference)
        denominator = 1.0 / (va + vb + vc)
        first_weight, second_weight = vb * denominator, vc * denominator
        closest = [
            first[index] + first_weight * ab[index] + second_weight * ac[index]
            for index in range(3)
        ]
        difference = [point[index] - closest[index] for index in range(3)]
        return dot(difference, difference)

    result: list[float] = []
    locked = 0
    feathered = 0
    nearest_distance_m = math.inf
    for vertex, primary_weight in zip(tissue_vertices_world_m, primary_weights, strict=True):
        if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex) or \
                not math.isfinite(primary_weight) or not 0.0 <= primary_weight <= 1.0:
            raise ImportError("BodyParts3D tendon attachment lock has invalid tissue input")
        key = tuple(math.floor(value / cell_size_m) for value in vertex)
        nearest_squared = math.inf
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for candidate in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), []):
                        squared = sum((vertex[axis] - candidate[axis]) ** 2 for axis in range(3))
                        nearest_squared = min(nearest_squared, squared)
        if surface_triangles:
            # The local vertex grid is a cheap early candidate; the final
            # source-triangle query removes visible gaps caused by sparse
            # tessellation around a calcaneal insertion.
            nearest_squared = min(
                nearest_squared,
                min(
                    point_triangle_squared_distance(vertex, triangle)
                    for triangle in surface_triangles
                ),
            )
        if nearest_squared == math.inf:
            result.append(primary_weight)
            continue
        distance_m = math.sqrt(nearest_squared)
        nearest_distance_m = min(nearest_distance_m, distance_m)
        if distance_m <= lock_radius_m:
            result.append(0.0)
            locked += 1
        elif distance_m < feather_radius_m:
            result.append(primary_weight * (distance_m - lock_radius_m) /
                          (feather_radius_m - lock_radius_m))
            feathered += 1
        else:
            result.append(primary_weight)
    if locked == 0:
        raise ImportError(
            "BodyParts3D tendon surface has no source-mesh-proximate secondary-bone insertion"
        )
    return result, {
        "method": (
            "exact-source-triangle proximity to named secondary BodyParts3D bone mesh"
            if surface_triangles else
            "exact-source-vertex proximity to named secondary BodyParts3D bone mesh"
        ),
        "lock_radius_m": lock_radius_m,
        "feather_radius_m": feather_radius_m,
        "locked_vertex_count": locked,
        "feathered_vertex_count": feathered,
        "nearest_vertex_distance_m": nearest_distance_m,
        "boundary": (
            "This changes only visual articulated-body blend weights at a named tendon insertion; "
            "it is not a topological weld, a tendon constitutive model, force transfer, "
            "or medical attachment validation."
        ),
    }


def _bodyparts_primary_bone_attachment_weights(
    tissue_vertices_world_m: list[list[float]],
    primary_bone_vertices_world_m: list[list[float]],
    lock_radius_m: float,
    feather_radius_m: float,
) -> tuple[list[float], dict[str, Any]]:
    """Bind only a source-mesh-proximate insertion band to a primary body.

    Broad fan-shaped muscles such as pectoralis major cannot use a projection
    along the two body centres: that gives their wide thoracic origin partial
    humerus ownership and lifts the inferior edge as the shoulder moves.  The
    exact BodyParts3D humerus surface instead identifies the narrow insertion
    band.  Everything outside its feather radius remains on the authored
    secondary route body.

    This is visual kinematic binding only.  It does not change a MyoSim route,
    endpoint, force, or tendon transaction.
    """
    secondary_attenuation, evidence = _bodyparts_secondary_attachment_weight_lock(
        tissue_vertices_world_m,
        [1.0] * len(tissue_vertices_world_m),
        primary_bone_vertices_world_m,
        None,
        lock_radius_m,
        feather_radius_m,
    )
    primary_weights = [1.0 - value for value in secondary_attenuation]
    thoracic_owner_count = sum(weight <= 1.0e-8 for weight in primary_weights)
    if thoracic_owner_count == 0:
        raise ImportError(
            "BodyParts3D primary attachment band leaves no secondary-body-owned surface"
        )
    evidence.update({
        "method": (
            "exact BodyParts3D primary-bone source-vertex proximity insertion "
            "lock with secondary-body origin ownership"
        ),
        "primary_locked_vertex_count": sum(
            weight >= 1.0 - 1.0e-8 for weight in primary_weights
        ),
        "primary_feathered_vertex_count": sum(
            1.0e-8 < weight < 1.0 - 1.0e-8 for weight in primary_weights
        ),
        "secondary_owned_vertex_count": thoracic_owner_count,
        "boundary": (
            "This changes only BodyParts3D visual blend weights at a named "
            "source-bone insertion; it is not a muscle material solve, a "
            "topological weld, force transfer, or clinical attachment map."
        ),
    })
    return primary_weights, evidence


def _bodyparts_source_mm_to_body_world(
    vertices_mm: list[list[float]], body_position_world_m: list[float], body_quaternion_xyzw: list[float],
    local_translation_m: list[float], local_quaternion_xyzw: list[float], local_uniform_scale: float,
) -> list[list[float]]:
    """Map source OBJ millimetres through the exact native bone binding at rest."""
    body_position = _myosim_vector(body_position_world_m, "BodyParts3D visual body position")
    body_rotation = _myosim_matrix_from_quaternion_xyzw(body_quaternion_xyzw)
    local_rotation = _myosim_matrix_from_quaternion_xyzw(local_quaternion_xyzw)
    local_translation = _myosim_vector(local_translation_m, "BodyParts3D visual local translation")
    if not math.isfinite(local_uniform_scale) or local_uniform_scale <= 0.0:
        raise ImportError("BodyParts3D visual local scale is invalid")
    result: list[list[float]] = []
    for vertex in vertices_mm:
        if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex):
            raise ImportError("BodyParts3D visual source mesh has a non-finite vertex")
        stored_m = [coordinate * 0.001 for coordinate in vertex]
        local = [
            local_translation[row] + local_uniform_scale * sum(
                local_rotation[row][column] * stored_m[column] for column in range(3)
            )
            for row in range(3)
        ]
        result.append([
            body_position[row] + sum(body_rotation[row][column] * local[column] for column in range(3))
            for row in range(3)
        ])
    return result


def _bodyparts_world_to_body_stored_m(
    vertices_world_m: list[list[float]], body_position_world_m: list[float], body_quaternion_xyzw: list[float],
    local_translation_m: list[float], local_quaternion_xyzw: list[float], local_uniform_scale: float,
    context: str,
) -> list[list[float]]:
    """Invert one native bone binding so a visual insertion stays on that bone."""
    body_position = _myosim_vector(body_position_world_m, context + " body position")
    body_rotation = _myosim_matrix_from_quaternion_xyzw(body_quaternion_xyzw)
    local_rotation = _myosim_matrix_from_quaternion_xyzw(local_quaternion_xyzw)
    local_translation = _myosim_vector(local_translation_m, context + " local translation")
    if not math.isfinite(local_uniform_scale) or local_uniform_scale <= 0.0:
        raise ImportError(context + " has an invalid local scale")
    result: list[list[float]] = []
    for vertex in vertices_world_m:
        if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex):
            raise ImportError(context + " has a non-finite world vertex")
        body_local = [
            sum(body_rotation[column][row] * (vertex[column] - body_position[column]) for column in range(3))
            for row in range(3)
        ]
        local_offset = [body_local[row] - local_translation[row] for row in range(3)]
        result.append([
            sum(local_rotation[column][row] * local_offset[column] for column in range(3)) / local_uniform_scale
            for row in range(3)
        ])
    return result


def _bodyparts_project_tendon_attachment_band(
    vertices_world_m: list[list[float]], distal_attenuation: list[float],
    bone_vertices_world_m: list[list[float]], bone_triangles: list[tuple[int, int, int]],
    visual_enthesis_inset_m: float = 0.005,
) -> tuple[list[list[float]], dict[str, float | int | str]]:
    """Project an already locked tendon insertion band onto named bone triangles.

    The projection acts only in the visual source-registration layer.  The
    tendon retains its original topology, and MyoSim sites, paths, parameters,
    force, and tendon dynamics remain untouched.
    """
    if len(vertices_world_m) != len(distal_attenuation) or not bone_triangles:
        raise ImportError("BodyParts3D tendon surface projection input is incomplete")
    if not math.isfinite(visual_enthesis_inset_m):
        raise ImportError("BodyParts3D tendon surface projection has an invalid enthesis offset")
    if any(len(vertex) != 3 or not all(math.isfinite(value) for value in vertex) for vertex in bone_vertices_world_m):
        raise ImportError("BodyParts3D tendon surface projection has invalid bone vertices")

    def closest_point_and_normal(
        point: list[float], first: list[float], second: list[float], third: list[float],
    ) -> tuple[list[float], list[float]]:
        ab = [second[index] - first[index] for index in range(3)]
        ac = [third[index] - first[index] for index in range(3)]
        normal = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ]
        magnitude = math.sqrt(sum(value * value for value in normal))
        if magnitude <= 1.0e-16:
            raise ImportError("BodyParts3D tendon surface projection has a degenerate bone triangle")
        normal = [value / magnitude for value in normal]
        ap = [point[index] - first[index] for index in range(3)]
        dot = lambda left, right: sum(left[index] * right[index] for index in range(3))
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return first, normal
        bp = [point[index] - second[index] for index in range(3)]
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return second, normal
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            return [first[index] + ab[index] * d1 / (d1 - d3) for index in range(3)], normal
        cp = [point[index] - third[index] for index in range(3)]
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return third, normal
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= d2:
            return [first[index] + ac[index] * d2 / (d2 - d6) for index in range(3)], normal
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
            bc = [third[index] - second[index] for index in range(3)]
            factor = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return [second[index] + bc[index] * factor for index in range(3)], normal
        denominator = 1.0 / (va + vb + vc)
        return [
            first[index] + (vb * ab[index] + vc * ac[index]) * denominator
            for index in range(3)
        ], normal

    result: list[list[float]] = []
    corrections: list[float] = []
    fully_locked = feathered = 0
    for vertex, attenuation in zip(vertices_world_m, distal_attenuation, strict=True):
        if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex) or \
                not math.isfinite(attenuation) or not 0.0 <= attenuation <= 1.0:
            raise ImportError("BodyParts3D tendon surface projection has invalid tendon input")
        blend = 1.0 - attenuation
        if blend <= 1.0e-8:
            result.append(vertex)
            continue
        closest: list[float] | None = None
        closest_normal: list[float] | None = None
        nearest_squared = math.inf
        for triangle in bone_triangles:
            if len(triangle) != 3 or any(not 0 <= index < len(bone_vertices_world_m) for index in triangle):
                raise ImportError("BodyParts3D tendon surface projection has an invalid bone triangle index")
            candidate, normal = closest_point_and_normal(
                vertex, *(bone_vertices_world_m[index] for index in triangle),
            )
            squared = sum((vertex[index] - candidate[index]) ** 2 for index in range(3))
            if squared < nearest_squared:
                closest, closest_normal, nearest_squared = candidate, normal, squared
        if closest is None or closest_normal is None:
            raise ImportError("BodyParts3D tendon surface projection has no named bone target")
        difference = [vertex[index] - closest[index] for index in range(3)]
        side = 1.0 if sum(difference[index] * closest_normal[index] for index in range(3)) >= 0.0 else -1.0
        # Put only the locked source boundary slightly *inside* its named
        # calcaneus. The opaque bone then occludes the independently authored
        # terminal cap instead of presenting a floating strip or inventing a
        # connector/collar between two meshes. The feather band preserves a
        # continuous deformation into the unmodified source tendon.
        target = [
            closest[index] - side * closest_normal[index] * visual_enthesis_inset_m
            for index in range(3)
        ]
        corrected = [vertex[index] * attenuation + target[index] * blend for index in range(3)]
        result.append(corrected)
        corrections.append(math.sqrt(sum((vertex[index] - corrected[index]) ** 2 for index in range(3))))
        if attenuation <= 1.0e-8:
            fully_locked += 1
        else:
            feathered += 1
    if not corrections:
        raise ImportError("BodyParts3D tendon surface projection has no distal attachment band")
    return result, {
        "method": "exact named secondary BodyParts3D bone triangle interior enthesis inset over the existing lock/feather band",
        "visual_enthesis_inset_m": visual_enthesis_inset_m,
        "projected_vertex_count": len(corrections),
        "fully_locked_vertex_count": fully_locked,
        "feathered_vertex_count": feathered,
        "rms_correction_m": math.sqrt(sum(value * value for value in corrections) / len(corrections)),
        "max_correction_m": max(corrections),
        "boundary": "visual rest-surface registration only; not a tendon weld, force-transfer law, continuum, or clinical attachment certificate",
    }


def _bodyparts_stitch_tendon_enthesis_band(
    vertices_world_m: list[list[float]], triangles: list[tuple[int, int, int]], distal_attenuation: list[float],
    bone_vertices_world_m: list[list[float]], bone_triangles: list[tuple[int, int, int]], member: str,
) -> tuple[list[list[float]], list[tuple[int, int, int]], list[float], dict[str, Any]]:
    """Close the opened distal source cap with a narrow named-bone enthesis strip.

    BodyParts3D's tendon and calcaneus are independently authored closed
    surfaces.  Hiding a cap that has been inset into the bone prevents the
    cap from leaking through a coarse calcaneus, but it can leave the visible
    tendon apparently suspended above the bone.  This visual-only strip
    stitches the newly-opened *source boundary* to its nearest named
    calcaneal surface.  The result is a continuous display surface and not a
    source-geometry claim, weld, force-transfer model, or tendon continuum.
    """
    if len(vertices_world_m) != len(distal_attenuation) or not triangles:
        raise ImportError(f"BodyParts3D tendon {member} enthesis stitching input is incomplete")
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        if len(triangle) != 3 or any(not 0 <= index < len(vertices_world_m) for index in triangle):
            raise ImportError(f"BodyParts3D tendon {member} enthesis stitching has an invalid triangle")
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = (min(first, second), max(first, second))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    # The cap removal leaves an open loop made of fully calcaneus-locked
    # source edges.  Do not close an arbitrary proximal source opening.
    source_edges = [
        edge for edge, count in edge_counts.items()
        if count == 1 and all(distal_attenuation[index] <= 1.0e-8 for index in edge)
    ]
    if not source_edges:
        raise ImportError(f"BodyParts3D tendon {member} has no distal source-cap boundary to stitch")
    source_indices = sorted({index for edge in source_edges for index in edge})
    source_points = [vertices_world_m[index] for index in source_indices]
    # A 0.35 mm exterior lift defeats z-fighting while remaining a thin
    # insertion presentation, rather than a synthetic tendon extension.
    bone_points, projection = _bodyparts_project_tendon_attachment_band(
        source_points, [0.0] * len(source_points), bone_vertices_world_m, bone_triangles,
        visual_enthesis_inset_m=-0.00035,
    )
    target_indices = {
        source_index: len(vertices_world_m) + target_index
        for target_index, source_index in enumerate(source_indices)
    }
    stitched_vertices = [*vertices_world_m, *bone_points]
    stitched_attenuation = [*distal_attenuation, *([0.0] * len(bone_points))]
    stitched_triangles = list(triangles)
    for first, second in source_edges:
        first_target, second_target = target_indices[first], target_indices[second]
        stitched_triangles.extend(((first, second, second_target), (first, second_target, first_target)))
    return stitched_vertices, stitched_triangles, stitched_attenuation, {
        "method": "visual_only_named_bone_enthesis_strip_from_trimmed_source_cap_boundary",
        "source_boundary_edge_count": len(source_edges),
        "source_boundary_vertex_count": len(source_indices),
        "generated_triangle_count": len(source_edges) * 2,
        "surface_lift_m": 0.00035,
        "source_projection": projection,
        "boundary": (
            "inferred visual display strip from the exact tendon cap boundary to the named calcaneus; "
            "not BodyParts3D source geometry, a tissue weld, tendon continuum, force-transfer law, or clinical attachment"
        ),
    }


def _bodyparts_drop_interior_tendon_cap_triangles(
    triangles: list[tuple[int, int, int]], distal_attenuation: list[float], member: str,
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Hide only fully interior source terminal-cap faces at a named insertion.

    The locked vertices are inset into the calcaneus, so a triangle made only
    from them is an interior closure face rather than visible tendon anatomy.
    Keeping it can expose a small fan through openings in a coarse source bone
    mesh.  This preserves every non-interior source triangle and creates no
    connector, weld, or replacement mesh.
    """
    retained: list[tuple[int, int, int]] = []
    removed = 0
    for triangle in triangles:
        if len(triangle) != 3 or any(not 0 <= index < len(distal_attenuation) for index in triangle):
            raise ImportError(f"BodyParts3D tendon {member} has an invalid source triangle")
        if all(distal_attenuation[index] <= 1.0e-8 for index in triangle):
            removed += 1
        else:
            retained.append(triangle)
    if not retained:
        raise ImportError(f"BodyParts3D tendon {member} loses all source triangles at its insertion")
    return retained, {
        "method": "drop_exact_source_triangles_fully_interior_to_named_bone_enthesis",
        "source_triangle_count": len(triangles),
        "retained_triangle_count": len(retained),
        "dropped_fully_locked_interior_cap_triangle_count": removed,
        "boundary": (
            "removes only source closure faces whose three vertices are already inset inside the named bone; "
            "does not generate a bridge, weld, continuum, or force-transfer geometry"
        ),
    }


def _bodyparts_source_element_names(sources: Path) -> set[tuple[str, str]]:
    """Read source FMA labels only to validate explicit map rows, never infer one."""
    source = sources / "isa_element_parts.txt"
    if not source.is_file():
        raise ImportError("BodyParts3D element relation source is unavailable")
    names: set[tuple[str, str]] = set()
    for raw in source.read_text(encoding="utf-8").splitlines():
        columns = raw.split("\t")
        if len(columns) != 3:
            raise ImportError("BodyParts3D element relation source has an invalid row")
        _, name, member_id = columns
        names.add((member_id, name))
    return names


def _myosim_surface_route_context(artifact: Path, source_sha: str) -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any],
]:
    """Load exact MyoSim source route endpoints from the compact native payload."""
    artifact = artifact.resolve()
    manifest_path = artifact / "myosim-fullbody-reference.manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "numi.human.myosim-fullbody-reference.v1":
        raise ImportError("MyoSim surface binding requires the full-body reference manifest")
    manifest_source = manifest.get("source")
    if not isinstance(manifest_source, dict) or manifest_source.get("archive_sha256") != source_sha:
        raise ImportError("MyoSim surface binding source provenance does not match registration")
    payloads = manifest.get("payloads")
    descriptor = payloads.get("muscles") if isinstance(payloads, dict) else None
    if not isinstance(descriptor, dict):
        raise ImportError("MyoSim surface binding manifest has no muscle payload")
    filename, expected_hash = descriptor.get("file"), descriptor.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ImportError("MyoSim surface binding muscle payload descriptor is invalid")
    payload_path = artifact / filename
    if not payload_path.is_file() or sha256(payload_path) != expected_hash:
        raise ImportError("MyoSim surface binding muscle payload provenance drifted")
    payload = payload_path.read_bytes()
    header_size = struct.calcsize("<8s9I32s")
    if len(payload) < header_size:
        raise ImportError("MyoSim surface binding muscle payload is truncated")
    magic, abi, body_count, muscle_count, site_count, geometry_count, route_count, _, reserved0, reserved1, payload_sha = struct.unpack_from(
        "<8s9I32s", payload
    )
    try:
        architecture_count, architecture_bytes = _myosim_muscle_payload_architecture(
            magic, abi, muscle_count, reserved0, reserved1,
        )
    except ImportError as error:
        raise ImportError("MyoSim surface binding muscle payload ABI is invalid") from error
    if payload_sha.hex() != source_sha:
        raise ImportError("MyoSim surface binding muscle payload ABI or source provenance is invalid")
    expected_bytes = _myosim_muscle_payload_bytes(
        site_count, geometry_count, route_count, muscle_count,
        architecture_count, architecture_bytes,
    )
    if len(payload) != expected_bytes or body_count == 0 or muscle_count == 0 or site_count == 0:
        raise ImportError("MyoSim surface binding muscle payload length is invalid")
    offset = header_size
    sites = [struct.unpack_from("<I3f", payload, offset + index * 16) for index in range(site_count)]
    offset += 16 * site_count
    geometries = [
        struct.unpack_from("<2I14f", payload, offset + index * 64)
        for index in range(geometry_count)
    ]
    offset += 64 * geometry_count
    routes = [struct.unpack_from("<4I", payload, offset + index * 16) for index in range(route_count)]
    offset += 16 * route_count
    muscle_records = [struct.unpack_from("<4I37f", payload, offset + index * 164) for index in range(muscle_count)]
    source_tree = manifest.get("core_tree")
    source_bodies = source_tree.get("source_body_records") if isinstance(source_tree, dict) else None
    if not isinstance(source_bodies, list):
        raise ImportError("MyoSim surface binding manifest has no source-body records")
    bodies: dict[str, dict[str, Any]] = {}
    core_indices: set[int] = set()
    for body in source_bodies:
        if not isinstance(body, dict):
            raise ImportError("MyoSim surface binding has an invalid source-body record")
        name, index = body.get("name"), body.get("core_body_index")
        if not isinstance(name, str) or not isinstance(index, int) or not 0 <= index < body_count or name in bodies or index in core_indices:
            raise ImportError("MyoSim surface binding source-body identity is invalid")
        _myosim_vector(body.get("default_com_position_world_m"), "MyoSim surface body position")
        _myosim_matrix_from_quaternion_xyzw(list(body.get("default_inertial_quaternion_world_xyzw", [])))
        bodies[name] = body
        core_indices.add(index)
    # The Core lowerer intentionally interposes zero-inertia serial-joint
    # carriers.  They are not source bodies and cannot own a MyoSim site, so
    # the source-body table must be a strict subset when carriers are present.
    if not core_indices or len(core_indices) > body_count:
        raise ImportError("MyoSim surface binding source-body coverage is invalid")
    body_by_index = {body["core_body_index"]: body for body in bodies.values()}

    def route_point(body_index: int, local: tuple[float, float, float]) -> dict[str, Any]:
        body = body_by_index.get(body_index)
        if body is None:
            raise ImportError("MyoSim surface route point has no source-body owner")
        position = _myosim_vector(
            body.get("default_com_position_world_m"),
            "MyoSim surface route body position",
        )
        rotation = _myosim_matrix_from_quaternion_xyzw(
            list(body.get("default_inertial_quaternion_world_xyzw", []))
        )
        world = _myosim_add(position, _myosim_matrix_vector(rotation, list(local)))
        return {
            "body": body["name"],
            "core_body_index": body_index,
            "world_m": world,
        }
    muscles = manifest.get("muscles")
    if not isinstance(muscles, list) or len(muscles) != muscle_count:
        raise ImportError("MyoSim surface binding muscle manifest length is invalid")
    routes_by_muscle: dict[str, dict[str, Any]] = {}
    for index, metadata in enumerate(muscles):
        if not isinstance(metadata, dict):
            raise ImportError("MyoSim surface binding has an invalid muscle manifest record")
        name = metadata.get("name")
        record = muscle_records[index]
        route_offset, count = record[1], record[2]
        if not isinstance(name, str) or name in routes_by_muscle or metadata.get("source_actuator_index") != index or count < 2 or route_offset + count > len(routes):
            raise ImportError("MyoSim surface binding muscle route identity is invalid")
        muscle_route = routes[route_offset:route_offset + count]
        first, last = muscle_route[0], muscle_route[-1]
        if first[0] != _MYOSIM_ROUTE_SITE or last[0] != _MYOSIM_ROUTE_SITE or first[1] >= len(sites) or last[1] >= len(sites):
            raise ImportError("MyoSim surface binding route has no source-site endpoints")
        primary_index, secondary_index = sites[first[1]][0], sites[last[1]][0]
        if primary_index == secondary_index or primary_index not in body_by_index or secondary_index not in body_by_index:
            raise ImportError("MyoSim surface binding route endpoint bodies are invalid")
        points: list[dict[str, Any]] = []
        binding_bodies: list[str] = []
        for node in muscle_route:
            kind, target = node[0], node[1]
            if kind == _MYOSIM_ROUTE_SITE:
                if target >= len(sites):
                    raise ImportError("MyoSim surface route references an absent site")
                site = sites[target]
                point = route_point(site[0], (site[1], site[2], site[3]))
                point["kind"] = "site"
            elif kind in {_MYOSIM_ROUTE_SPHERE, _MYOSIM_ROUTE_CYLINDER}:
                if target >= len(geometries):
                    raise ImportError("MyoSim surface route references an absent wrap")
                geometry = geometries[target]
                point = route_point(
                    geometry[0], (geometry[4], geometry[5], geometry[6])
                )
                point["kind"] = "sphere" if kind == _MYOSIM_ROUTE_SPHERE else "cylinder"
            else:
                raise ImportError("MyoSim surface route has an unsupported node kind")
            points.append(point)
            if point["body"] not in binding_bodies:
                binding_bodies.append(point["body"])
        routes_by_muscle[name] = {
            "source_actuator_index": index,
            "source_route_node_count": count,
            "primary_body": body_by_index[primary_index]["name"],
            "secondary_body": body_by_index[secondary_index]["name"],
            "binding_bodies": binding_bodies,
            "route_points": points,
        }
    return bodies, routes_by_muscle, manifest


def numi_human_tendon_endpoint_payload(
    myosim_artifact: Path, output: Path, surface_receipt_path: Path | None = None,
    allow_unadmitted_surface: bool = False,
) -> dict[str, Any]:
    """Compile one authoritative origin/insertion binding for every route.

    Point bindings preserve the authored MyoSim site exactly.  An optional,
    explicit surface receipt may replace an endpoint with a named bone
    triangle and barycentric point.  The native runtime consumes the resolved
    point as the route site itself, so the existing J^T projection remains the
    only force scatter.
    """
    artifact = myosim_artifact.resolve()
    manifest_path = artifact / "myosim-fullbody-reference.manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "numi.human.myosim-fullbody-reference.v1":
        raise ImportError("Numi Human tendon payload requires the full-body MyoSim reference")
    source = manifest.get("source")
    source_sha = source.get("archive_sha256") if isinstance(source, dict) else None
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ImportError("Numi Human tendon payload has no source archive identity")
    payloads = manifest.get("payloads")
    muscle_descriptor = payloads.get("muscles") if isinstance(payloads, dict) else None
    rigid_descriptor = payloads.get("rigid") if isinstance(payloads, dict) else None
    if not isinstance(muscle_descriptor, dict) or not isinstance(rigid_descriptor, dict):
        raise ImportError("Numi Human tendon payload requires rigid and muscle descriptors")
    muscle_file, muscle_sha = muscle_descriptor.get("file"), muscle_descriptor.get("sha256")
    if not isinstance(muscle_file, str) or not isinstance(muscle_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", muscle_sha):
        raise ImportError("Numi Human tendon muscle descriptor is invalid")
    muscle_path = artifact / muscle_file
    if not muscle_path.is_file() or sha256(muscle_path) != muscle_sha:
        raise ImportError("Numi Human tendon muscle payload is missing or has drifted")
    raw = muscle_path.read_bytes()
    header_format = "<8s9I32s"
    header_size = struct.calcsize(header_format)
    if len(raw) < header_size:
        raise ImportError("Numi Human tendon muscle payload is truncated")
    (
        magic, abi, body_count, muscle_count, site_count, wrap_count,
        route_count, tendon_count, reserved0, reserved1, embedded_source_sha,
    ) = struct.unpack_from(header_format, raw)
    try:
        architecture_count, architecture_bytes = _myosim_muscle_payload_architecture(
            magic, abi, muscle_count, reserved0, reserved1,
        )
    except ImportError as error:
        raise ImportError("Numi Human tendon muscle payload ABI is invalid") from error
    expected_size = _myosim_muscle_payload_bytes(
        site_count, wrap_count, route_count, muscle_count,
        architecture_count, architecture_bytes,
    )
    if (
        body_count == 0 or muscle_count == 0 or site_count == 0 or tendon_count == 0
        or embedded_source_sha.hex() != source_sha
        or len(raw) != expected_size
    ):
        raise ImportError("Numi Human tendon muscle payload ABI disagrees with its manifest")
    offset = header_size
    sites = [struct.unpack_from("<I3f", raw, offset + 16 * index) for index in range(site_count)]
    offset += 16 * site_count + 64 * wrap_count
    routes = [struct.unpack_from("<4I", raw, offset + 16 * index) for index in range(route_count)]
    offset += 16 * route_count
    muscles = [struct.unpack_from("<4I37f", raw, offset + 164 * index) for index in range(muscle_count)]
    muscle_metadata = manifest.get("muscles")
    if not isinstance(muscle_metadata, list) or len(muscle_metadata) != muscle_count:
        raise ImportError("Numi Human tendon muscle identity table is incomplete")

    receipt_records: dict[tuple[str, int], dict[str, Any]] = {}
    receipt_descriptor: dict[str, Any] | None = None
    if surface_receipt_path is not None:
        receipt_path = surface_receipt_path.resolve()
        receipt = read_json(receipt_path)
        if receipt.get("schema") != "numi.human.tendon-surface-registration.v1":
            raise ImportError("Numi Human tendon surface receipt has an unsupported schema")
        admission = receipt.get("admission")
        mechanically_admitted = isinstance(admission, dict) and admission.get("mechanical") is True
        if not mechanically_admitted and not allow_unadmitted_surface:
            raise ImportError("Numi Human tendon surface receipt is a candidate, not a mechanically admitted registration")
        records = receipt.get("records")
        if not isinstance(records, list):
            raise ImportError("Numi Human tendon surface receipt has no records")
        for record in records:
            if not isinstance(record, dict):
                raise ImportError("Numi Human tendon surface receipt has a malformed record")
            name, endpoint = record.get("muscle"), record.get("endpoint")
            ordinal = 0 if endpoint == "origin" else 1 if endpoint == "insertion" else None
            key = (name, ordinal) if isinstance(name, str) and ordinal is not None else None
            if key is None or key in receipt_records:
                raise ImportError("Numi Human tendon surface receipt has a duplicate or invalid endpoint")
            receipt_records[key] = record
        receipt_descriptor = {"file": receipt_path.name, "sha256": sha256(receipt_path),
                              "mechanically_admitted": mechanically_admitted}

    endpoint_records: list[bytes] = []
    endpoint_manifest: list[dict[str, Any]] = []
    triangle_records: list[bytes] = []
    consumed_receipts: set[tuple[str, int]] = set()
    for muscle_index, (record, metadata) in enumerate(zip(muscles, muscle_metadata, strict=True)):
        route_offset, count = record[1], record[2]
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if not isinstance(name, str) or metadata.get("source_actuator_index") != muscle_index or count < 2 or route_offset + count > route_count:
            raise ImportError("Numi Human tendon route identity is invalid")
        endpoint_nodes = ((0, route_offset), (1, route_offset + count - 1))
        for endpoint_ordinal, route_node_index in endpoint_nodes:
            route = routes[route_node_index]
            if route[0] != _MYOSIM_ROUTE_SITE or route[1] >= site_count:
                raise ImportError(f"Numi Human tendon route {name} has no source-site endpoint")
            site_index = route[1]
            source_site = sites[site_index]
            body_index = source_site[0]
            if body_index >= body_count or not all(math.isfinite(value) for value in source_site[1:]):
                raise ImportError(f"Numi Human tendon route {name} has an invalid source site")
            mode = _NUMI_HUMAN_TENDON_POINT
            triangle_index = 0xFFFFFFFF
            bone_stable_id = 0
            local_point = [float(value) for value in source_site[1:]]
            barycentric = [0.0, 0.0, 0.0]
            migration = 0.0
            surface_identity: dict[str, Any] | None = None
            receipt_key = (name, endpoint_ordinal)
            surface = receipt_records.get(receipt_key)
            if surface is not None:
                consumed_receipts.add(receipt_key)
                if surface.get("body_index") != body_index:
                    raise ImportError(f"Numi Human tendon surface {name} changes the endpoint body")
                vertices = surface.get("triangle_local_m")
                barycentric_source = surface.get("barycentric")
                bone_member_id = surface.get("bone_member_id")
                source_triangle_index = surface.get("source_triangle_index")
                bone_stable_id = surface.get("bone_stable_id")
                if (
                    not isinstance(vertices, list) or len(vertices) != 3
                    or not isinstance(barycentric_source, list) or len(barycentric_source) != 3
                    or not isinstance(bone_member_id, str) or not re.fullmatch(r"FJ[0-9]+M?", bone_member_id)
                    or not isinstance(source_triangle_index, int) or source_triangle_index < 0
                    or not isinstance(bone_stable_id, int) or not 0 < bone_stable_id <= 0xFFFFFFFF
                ):
                    raise ImportError(f"Numi Human tendon surface {name} has invalid triangle identity")
                parsed_vertices = [_myosim_vector(vertex, f"Numi Human tendon surface {name} triangle") for vertex in vertices]
                barycentric = [_finite_scalar(value, f"Numi Human tendon surface {name} barycentric") for value in barycentric_source]
                if any(value < 0.0 or value > 1.0 for value in barycentric) or abs(sum(barycentric) - 1.0) > 1.0e-6:
                    raise ImportError(f"Numi Human tendon surface {name} has invalid barycentric weights")
                local_point = [
                    sum(barycentric[vertex] * parsed_vertices[vertex][axis] for vertex in range(3))
                    for axis in range(3)
                ]
                migration = math.sqrt(sum((local_point[axis] - source_site[axis + 1]) ** 2 for axis in range(3)))
                mode = _NUMI_HUMAN_TENDON_TRIANGLE
                triangle_index = len(triangle_records)
                triangle_records.append(struct.pack(
                    "<4I12f", body_index, bone_stable_id, source_triangle_index, 0,
                    *[component for vertex in parsed_vertices for component in (*vertex, 0.0)],
                ))
                surface_identity = {
                    "bone_member_id": bone_member_id,
                    "bone_stable_id": bone_stable_id,
                    "source_triangle_index": source_triangle_index,
                    "triangle_index": triangle_index,
                    "barycentric": barycentric,
                }
            endpoint_records.append(struct.pack(
                "<8I8f", muscle_index, endpoint_ordinal, route_node_index, site_index,
                body_index, mode, triangle_index, bone_stable_id,
                *local_point, *barycentric, migration, 0.0,
            ))
            endpoint_manifest.append({
                "muscle_index": muscle_index, "muscle": name,
                "endpoint": "origin" if endpoint_ordinal == 0 else "insertion",
                "route_node_index": route_node_index, "source_site_index": site_index,
                "body_index": body_index,
                "attachment_mode": "source_site_point" if mode == _NUMI_HUMAN_TENDON_POINT else "registered_bone_triangle",
                "resolved_local_point_m": local_point,
                "endpoint_migration_m": migration,
                "surface": surface_identity,
            })
    unused_receipts = set(receipt_records) - consumed_receipts
    if unused_receipts:
        raise ImportError("Numi Human tendon surface receipt names unknown route endpoints")
    if len(endpoint_records) != 2 * muscle_count:
        raise ImportError("Numi Human tendon endpoint coverage is incomplete")
    unadmitted_surface_candidate = (
        receipt_descriptor is not None
        and receipt_descriptor["mechanically_admitted"] is False
    )

    header = struct.pack(
        "<8s8I32s32s", _NUMI_HUMAN_TENDON_MAGIC, _NUMI_HUMAN_TENDON_ABI,
        body_count, muscle_count, site_count, len(endpoint_records), len(triangle_records), 0, 0,
        bytes.fromhex(source_sha), bytes.fromhex(muscle_sha),
    )
    payload = b"".join([header, *endpoint_records, *triangle_records])
    if len(payload) != 104 + 64 * len(endpoint_records) + 64 * len(triangle_records):
        raise ImportError("Numi Human tendon payload ABI size mismatch")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tendon_path = output / "numi-human-tendon-endpoints.nhtendon"
    tendon_path.write_bytes(payload)
    tendon_manifest = {
        "schema": "numi.human.tendon-endpoint-payload.v1",
        "payload": {"file": tendon_path.name, "sha256": sha256(tendon_path), "bytes": len(payload),
                    "magic": "NHTENDON1", "payload_abi": _NUMI_HUMAN_TENDON_ABI},
        "source": {"myosim_manifest": {"file": manifest_path.name, "sha256": sha256(manifest_path)},
                   "myosim_archive_sha256": source_sha, "myosim_muscle_payload_sha256": muscle_sha,
                   "surface_receipt": receipt_descriptor},
        "coverage": {"muscle_count": muscle_count, "mechanical_endpoint_count": len(endpoint_records),
                     "expected_endpoint_count": 2 * muscle_count,
                     "source_site_point_count": sum(1 for item in endpoint_manifest if item["attachment_mode"] == "source_site_point"),
                     "registered_bone_triangle_count": len(triangle_records)},
        "endpoints": endpoint_manifest,
        "runtime_contract": "resolve each route endpoint once before the existing route-length Jacobian projection; never add a second force scatter",
        "status": (
            "candidate_route_endpoint_program_not_mechanically_admitted"
            if unadmitted_surface_candidate
            else "complete_route_endpoint_mechanical_coverage"
        ),
        "evidence_boundary": (
            "All routes retain an explicit bone-owned mechanical endpoint. This candidate includes an explicitly unadmitted surface registration and must not replace the production point program."
            if unadmitted_surface_candidate
            else "All routes retain an explicit bone-owned mechanical endpoint. Triangle records represent admitted surface traction points; this payload is not a deformable tendon material or clinical registration."
        ),
    }
    tendon_manifest_path = output / "numi-human-tendon-endpoints.manifest.json"
    write_json(tendon_manifest_path, tendon_manifest)
    canonical_manifest = {
        "schema": "numi.human.pack.v1",
        "owner": "Numi Lab Human",
        "payloads": {
            "rigid": rigid_descriptor,
            "muscles": muscle_descriptor,
            "support_contact": payloads.get("support_contact"),
            "tendon_endpoints": tendon_manifest["payload"],
        },
        "coverage": tendon_manifest["coverage"],
        "status": tendon_manifest["status"],
        "source_authorities": {
            "geometry": "BodyParts3D 4.0",
            "active_full_body_seed": "compiled MyoSim full-body source program",
            "lower_body_comparison": "OpenSim RajagopalLaiUhlrich2023",
            "upper_extremity_comparison": "OpenSim Upper Extremity Dynamic Model",
        },
        "runtime_owner": "Numi Lab C++/Metal; offline import is not a runtime force path",
    }
    write_json(output / "numi-human-pack.manifest.json", canonical_manifest)
    return tendon_manifest


def _tendon_closest_point_on_triangle(
    point: list[float], triangle: list[list[float]],
) -> tuple[list[float], list[float]]:
    """Return the exact Euclidean closest point and barycentric coordinates."""
    a, b, c = triangle
    subtract = lambda left, right: [left[axis] - right[axis] for axis in range(3)]
    dot = lambda left, right: sum(left[axis] * right[axis] for axis in range(3))
    ab, ac, ap = subtract(b, a), subtract(c, a), subtract(point, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return list(a), [1.0, 0.0, 0.0]
    bp = subtract(point, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return list(b), [0.0, 1.0, 0.0]
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        value = d1 / (d1 - d3)
        return [a[axis] + value * ab[axis] for axis in range(3)], [1.0 - value, value, 0.0]
    cp = subtract(point, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return list(c), [0.0, 0.0, 1.0]
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        value = d2 / (d2 - d6)
        return [a[axis] + value * ac[axis] for axis in range(3)], [1.0 - value, 0.0, value]
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        value = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        edge = subtract(c, b)
        return [b[axis] + value * edge[axis] for axis in range(3)], [0.0, 1.0 - value, value]
    denominator = va + vb + vc
    if abs(denominator) <= 1.0e-20:
        raise ImportError("Numi Human tendon envelope encountered a degenerate bone triangle")
    inverse = 1.0 / denominator
    v, w = vb * inverse, vc * inverse
    barycentric = [1.0 - v - w, v, w]
    return [
        sum(barycentric[index] * triangle[index][axis] for index in range(3))
        for axis in range(3)
    ], barycentric


def _tendon_inverse_matrix(matrix: list[list[float]]) -> list[list[float]] | None:
    size = len(matrix)
    augmented = [
        list(row) + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1.0e-10 * scale:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse_pivot = 1.0 / augmented[column][column]
        augmented[column] = [value * inverse_pivot for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor != 0.0:
                augmented[row] = [
                    augmented[row][index] - factor * augmented[column][index]
                    for index in range(2 * size)
                ]
    return [row[size:] for row in augmented]


def _tendon_envelope_force_maps(
    source_point: list[float], nodes: list[list[float]], patch_radius: float,
) -> tuple[list[list[list[float]]], dict[str, float]] | None:
    """Minimum-L2 nodal-force map conserving source force and moment.

    For every source-local terminal force F the returned matrices M_i satisfy
    sum(M_i F)=F and sum((x_i-a) cross (M_i F))=0.  Moment rows are scaled by
    the patch radius only for conditioning; the physical zero-moment equation
    is unchanged.
    """
    if len(nodes) != 4 or not math.isfinite(patch_radius) or patch_radius <= 1.0e-6:
        return None
    constraint = [[0.0 for _ in range(12)] for _ in range(6)]
    for node_index, node in enumerate(nodes):
        base = 3 * node_index
        for axis in range(3):
            constraint[axis][base + axis] = 1.0
        rx, ry, rz = [
            (node[axis] - source_point[axis]) / patch_radius for axis in range(3)
        ]
        skew = ((0.0, -rz, ry), (rz, 0.0, -rx), (-ry, rx, 0.0))
        for row in range(3):
            for column in range(3):
                constraint[3 + row][base + column] = skew[row][column]
    normal = [[
        sum(constraint[row][column] * constraint[other][column] for column in range(12))
        for other in range(6)
    ] for row in range(6)]
    inverse = _tendon_inverse_matrix(normal)
    if inverse is None:
        return None
    flat = [[
        sum(constraint[row][output] * inverse[row][axis] for row in range(6))
        for axis in range(3)
    ] for output in range(12)]
    maps = [
        [[flat[3 * node + row][column] for column in range(3)] for row in range(3)]
        for node in range(4)
    ]
    maximum_force_residual = 0.0
    maximum_moment_residual = 0.0
    for axis in range(3):
        nodal = [[maps[node][row][axis] for row in range(3)] for node in range(4)]
        resultant = [sum(force[row] for force in nodal) for row in range(3)]
        maximum_force_residual = max(maximum_force_residual, math.sqrt(sum(
            (resultant[row] - (1.0 if row == axis else 0.0)) ** 2 for row in range(3)
        )))
        moment = [0.0, 0.0, 0.0]
        for node, force in zip(nodes, nodal, strict=True):
            rx, ry, rz = [node[row] - source_point[row] for row in range(3)]
            fx, fy, fz = force
            cross = (ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx)
            moment = [moment[row] + cross[row] for row in range(3)]
        maximum_moment_residual = max(
            maximum_moment_residual, math.sqrt(sum(value * value for value in moment)),
        )
    gram = [[
        sum(flat[row][left] * flat[row][right] for row in range(12))
        for right in range(3)
    ] for left in range(3)]
    direction = [1.0 / math.sqrt(3.0)] * 3
    for _ in range(24):
        next_direction = [sum(gram[row][column] * direction[column] for column in range(3)) for row in range(3)]
        magnitude = math.sqrt(sum(value * value for value in next_direction))
        if magnitude <= 1.0e-18:
            break
        direction = [value / magnitude for value in next_direction]
    eigenvalue = sum(
        direction[row] * gram[row][column] * direction[column]
        for row in range(3) for column in range(3)
    )
    sampled_total_amplification = 0.0
    for components in product((-1.0, 0.0, 1.0), repeat=3):
        magnitude = math.sqrt(sum(value * value for value in components))
        if magnitude == 0.0:
            continue
        unit = [value / magnitude for value in components]
        total = 0.0
        for matrix in maps:
            force = [sum(matrix[row][column] * unit[column] for column in range(3)) for row in range(3)]
            total += math.sqrt(sum(value * value for value in force))
        sampled_total_amplification = max(sampled_total_amplification, total)
    metrics = {
        "force_residual": maximum_force_residual,
        "moment_residual_m": maximum_moment_residual,
        "l2_force_amplification": math.sqrt(max(0.0, eigenvalue)),
        "sampled_total_force_amplification": sampled_total_amplification,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        return None
    return maps, metrics


def _numi_human_bone_envelope_surfaces(bone_artifact: Path, source_sha: str) -> tuple[
    dict[int, list[dict[str, Any]]], dict[str, Any], Path,
]:
    artifact = bone_artifact.resolve()
    manifest_path = artifact / "bodyparts3d-myosim-major-bones.manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "numi.human.bodyparts3d-myosim-major-bone-visual-payload.v1":
        raise ImportError("Numi Human tendon envelopes require an NHBONES1 manifest")
    descriptor = manifest.get("payload")
    if not isinstance(descriptor, dict):
        raise ImportError("Numi Human tendon envelope bone descriptor is missing")
    filename, expected_sha = descriptor.get("file"), descriptor.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ImportError("Numi Human tendon envelope bone descriptor is invalid")
    payload_path = artifact / filename
    if not payload_path.is_file() or sha256(payload_path) != expected_sha:
        raise ImportError("Numi Human tendon envelope bone payload is missing or drifted")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("myosim_source_archive_sha256") != source_sha:
        raise ImportError("Numi Human tendon envelope bone registration targets another MyoSim source")
    anchors = source.get("anchors")
    if not isinstance(anchors, list):
        raise ImportError("Numi Human tendon envelope bone member identities are absent")
    member_body_indices = {
        str(anchor.get("member_id")): int(anchor.get("core_body_index", -1))
        for anchor in anchors if isinstance(anchor, dict)
    }
    source_component_enthesis_receipt = _numi_human_source_component_enthesis_receipt(
        manifest.get("source_component_enthesis_registration"),
        member_body_indices,
        source_sha,
    )
    raw = payload_path.read_bytes()
    header_size = struct.calcsize("<8s5I32s")
    if len(raw) < header_size:
        raise ImportError("Numi Human tendon envelope bone payload is truncated")
    magic, abi, bone_count, vertex_count, index_count, fingerprint, embedded_source = struct.unpack_from(
        "<8s5I32s", raw,
    )
    if (
        magic != _BODYPARTS_MYOSIM_BONE_VISUAL_MAGIC
        or abi != _BODYPARTS_MYOSIM_BONE_VISUAL_ABI
        or bone_count == 0 or bone_count != len(anchors)
        or embedded_source.hex() != source_sha
        or descriptor.get("registration_fingerprint32") != f"{fingerprint:08x}"
    ):
        raise ImportError("Numi Human tendon envelope NHBONES1 identity is invalid")
    record_size = struct.calcsize("<6I8f")
    vertex_size = struct.calcsize("<6f")
    expected_size = header_size + record_size * bone_count + vertex_size * vertex_count + 4 * index_count
    if len(raw) != expected_size:
        raise ImportError("Numi Human tendon envelope NHBONES1 length is invalid")
    offset = header_size
    records = [struct.unpack_from("<6I8f", raw, offset + record_size * index) for index in range(bone_count)]
    offset += record_size * bone_count
    vertices = [struct.unpack_from("<6f", raw, offset + vertex_size * index) for index in range(vertex_count)]
    offset += vertex_size * vertex_count
    indices = struct.unpack_from(f"<{index_count}I", raw, offset)
    by_body: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for stable_id, (record, anchor) in enumerate(zip(records, anchors, strict=True), start=1):
        body_index, first_vertex, count, first_index, count_indices, record_stable_id, *pose = record
        if (
            record_stable_id != stable_id or count == 0 or count_indices == 0 or count_indices % 3 != 0
            or first_vertex + count > vertex_count or first_index + count_indices > index_count
            or not isinstance(anchor, dict) or anchor.get("core_body_index") != body_index
        ):
            raise ImportError("Numi Human tendon envelope NHBONES1 record is malformed")
        translation = pose[:3]
        rotation = _myosim_matrix_from_quaternion_xyzw(list(pose[3:7]))
        scale = pose[7]
        if not math.isfinite(scale) or scale <= 0.0:
            raise ImportError("Numi Human tendon envelope bone scale is invalid")
        local_vertices = []
        for vertex in vertices[first_vertex:first_vertex + count]:
            transformed = _myosim_matrix_vector(rotation, [scale * vertex[axis] for axis in range(3)])
            local_vertices.append([translation[axis] + transformed[axis] for axis in range(3)])
        local_indices = indices[first_index:first_index + count_indices]
        if any(index < first_vertex or index >= first_vertex + count for index in local_indices):
            raise ImportError("Numi Human tendon envelope bone index escapes its member")
        triangles = [
            tuple(local_indices[index + axis] - first_vertex for axis in range(3))
            for index in range(0, count_indices, 3)
        ]
        by_body[body_index].append({
            "stable_id": stable_id,
            "member_id": anchor.get("member_id"),
            "vertices": local_vertices,
            "triangles": triangles,
        })
    return dict(by_body), {
        "file": payload_path.name,
        "sha256": expected_sha,
        "bytes": len(raw),
        "bone_count": bone_count,
        "registration_fingerprint32": f"{fingerprint:08x}",
        "manifest": {"file": manifest_path.name, "sha256": sha256(manifest_path)},
        **({
            "source_component_enthesis_registration": source_component_enthesis_receipt,
        } if source_component_enthesis_receipt is not None else {}),
    }, payload_path


def _numi_human_tendon_surface_envelope(
    source_point: list[float], surface: dict[str, Any], maximum_distance_m: float,
    maximum_patch_radius_m: float, maximum_force_amplification: float,
    migrate_endpoint_to_surface: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    vertices: list[list[float]] = surface["vertices"]
    triangles: list[tuple[int, int, int]] = surface["triangles"]
    nearest: tuple[float, int, list[float], list[float]] | None = None
    for triangle_index, triangle_indices in enumerate(triangles):
        triangle = [vertices[index] for index in triangle_indices]
        closest, barycentric = _tendon_closest_point_on_triangle(source_point, triangle)
        squared = sum((closest[axis] - source_point[axis]) ** 2 for axis in range(3))
        tolerance = max(1.0e-18, 1.0e-12 * max(squared, nearest[0] if nearest is not None else 0.0))
        if (
            nearest is None or squared < nearest[0] - tolerance
            or (abs(squared - nearest[0]) <= tolerance and triangle_index < nearest[1])
        ):
            nearest = (squared, triangle_index, closest, barycentric)
    if nearest is None:
        return None, "no_surface_triangle"
    squared_distance, source_triangle_index, closest_point, barycentric = nearest
    surface_distance = math.sqrt(squared_distance)
    if surface_distance > maximum_distance_m:
        return None, "surface_distance_exceeds_gate"
    seed_triangle = triangles[source_triangle_index]
    a, b, c = [vertices[index] for index in seed_triangle]
    left = [b[axis] - a[axis] for axis in range(3)]
    right = [c[axis] - a[axis] for axis in range(3)]
    normal = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
    normal_length = math.sqrt(sum(value * value for value in normal))
    left_length = math.sqrt(sum(value * value for value in left))
    if normal_length <= 1.0e-12 or left_length <= 1.0e-12:
        return None, "degenerate_nearest_triangle"
    normal = [value / normal_length for value in normal]
    tangent0 = [value / left_length for value in left]
    tangent1 = [
        normal[1] * tangent0[2] - normal[2] * tangent0[1],
        normal[2] * tangent0[0] - normal[0] * tangent0[2],
        normal[0] * tangent0[1] - normal[1] * tangent0[0],
    ]
    adjacency = surface.get("adjacency")
    if adjacency is None:
        adjacency = [dict() for _ in vertices]
        for first, second, third in triangles:
            for start, end in ((first, second), (second, third), (third, first)):
                edge = math.sqrt(sum((vertices[start][axis] - vertices[end][axis]) ** 2 for axis in range(3)))
                previous = adjacency[start].get(end)
                if previous is None or edge < previous:
                    adjacency[start][end] = edge
                    adjacency[end][start] = edge
        adjacency = [tuple(neighbours.items()) for neighbours in adjacency]
        surface["adjacency"] = adjacency
    geodesic = [math.inf] * len(vertices)
    queue: list[tuple[float, int]] = []
    for index in seed_triangle:
        distance = math.sqrt(sum((vertices[index][axis] - closest_point[axis]) ** 2 for axis in range(3)))
        geodesic[index] = distance
        heapq.heappush(queue, (distance, index))
    while queue:
        distance, index = heapq.heappop(queue)
        if distance != geodesic[index] or distance > maximum_patch_radius_m:
            continue
        for neighbour, edge in adjacency[index]:
            candidate = distance + edge
            if candidate <= maximum_patch_radius_m and candidate < geodesic[neighbour]:
                geodesic[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    candidates = [index for index, distance in enumerate(geodesic) if math.isfinite(distance)]
    best: tuple[float, dict[str, Any]] | None = None

    def consider_patch(
        nodes: list[list[float]], node_vertex_indices: list[int],
        method: str, node_surface_sources: list[dict[str, Any]] | None = None,
    ) -> None:
        nonlocal best
        patch_radius = max(math.sqrt(sum(
            (node[axis] - closest_point[axis]) ** 2 for axis in range(3)
        )) for node in nodes)
        if patch_radius > maximum_patch_radius_m + 1.0e-12:
            return
        force_application_point = closest_point if migrate_endpoint_to_surface else source_point
        mapped = _tendon_envelope_force_maps(force_application_point, nodes, patch_radius)
        if mapped is None:
            return
        maps, metrics = mapped
        amplification = metrics["sampled_total_force_amplification"]
        if (
            metrics["force_residual"] > 2.0e-6
            or metrics["moment_residual_m"] > 2.0e-8
            or amplification > maximum_force_amplification
        ):
            return
        score = amplification + 0.05 * metrics["l2_force_amplification"]
        record = {
            "body_index": surface["body_index"],
            "bone_stable_id": surface["stable_id"],
            "bone_member_id": surface["member_id"],
            "source_triangle_index": source_triangle_index,
            "nearest_barycentric": barycentric,
            "nearest_local_point_m": closest_point,
            "resolved_local_point_m": force_application_point,
            "node_vertex_indices": node_vertex_indices,
            "node_local_points_m": nodes,
            "surface_patch_method": method,
            "force_maps": maps,
            "surface_distance_m": surface_distance,
            "patch_radius_m": patch_radius,
            **metrics,
        }
        if node_surface_sources is not None:
            record["node_surface_sources"] = node_surface_sources
        if best is None or score < best[0]:
            best = (score, record)

    radii = sorted(set(min(maximum_patch_radius_m, value) for value in (0.006, 0.009, 0.012, 0.016)))
    for radius in radii:
        if radius < 0.003:
            continue
        eligible = [index for index in candidates if geodesic[index] <= radius]
        if len(eligible) < 4:
            continue
        for phase in (0.0, math.pi / 8.0, math.pi / 4.0):
            selected: list[int] = []
            valid = True
            for direction_index in range(4):
                angle = phase + 0.5 * math.pi * direction_index
                target = (math.cos(angle), math.sin(angle))
                ranked: list[tuple[float, int]] = []
                for index in eligible:
                    delta = [vertices[index][axis] - closest_point[axis] for axis in range(3)]
                    x = sum(delta[axis] * tangent0[axis] for axis in range(3))
                    y = sum(delta[axis] * tangent1[axis] for axis in range(3))
                    radial = math.hypot(x, y)
                    if radial < 0.32 * radius:
                        continue
                    cosine = (x * target[0] + y * target[1]) / radial
                    if cosine < 0.45:
                        continue
                    normal_offset = abs(sum(delta[axis] * normal[axis] for axis in range(3)))
                    score = cosine * radial - 0.25 * abs(radial - 0.78 * radius) - 0.35 * normal_offset
                    ranked.append((score, index))
                ranked.sort(reverse=True)
                chosen = next((index for _, index in ranked if index not in selected), None)
                if chosen is None:
                    valid = False
                    break
                selected.append(chosen)
            if not valid:
                continue
            nodes = [vertices[index] for index in selected]
            consider_patch(
                nodes, selected, "connected_geodesic_compass_vertices",
            )

    # Some source meshes are locally coarse or terminate at a small bone tip,
    # so a valid surface neighborhood can contain fewer than four mesh
    # vertices or fail the compass heuristic.  Build a bounded deterministic
    # quadrature pool without changing the surface: virtual candidates are
    # exact barycentric points on the already-selected source triangle and all
    # other candidates are vertices in its connected geodesic neighborhood.
    # NHTENDON2 consumes positions and maps, not vertex indices, so this is an
    # offline force-transfer discretization rather than geometry mutation.
    topology_candidate_count = 0
    if best is None:
        topology_candidates: list[dict[str, Any]] = []

        def add_topology_candidate(
            point: list[float], source: dict[str, Any], vertex_index: int | None = None,
        ) -> None:
            radius = math.sqrt(sum(
                (point[axis] - closest_point[axis]) ** 2 for axis in range(3)
            ))
            if radius > maximum_patch_radius_m + 1.0e-12:
                return
            if any(sum(
                (point[axis] - candidate["point"][axis]) ** 2 for axis in range(3)
            ) <= 1.0e-18 for candidate in topology_candidates):
                return
            topology_candidates.append({
                "point": list(point),
                "source": source,
                "vertex_index": vertex_index,
            })

        add_topology_candidate(
            closest_point,
            {
                "kind": "seed_triangle_barycentric",
                "source_triangle_index": source_triangle_index,
                "barycentric": barycentric,
            },
        )
        seed_values = [a, b, c]
        for local_vertex, vertex_index in enumerate(seed_triangle):
            vertex = vertices[vertex_index]
            delta = [vertex[axis] - closest_point[axis] for axis in range(3)]
            length = math.sqrt(sum(value * value for value in delta))
            if length <= 1.0e-12:
                continue
            fraction = min(1.0, 0.82 * maximum_patch_radius_m / length)
            weights = [(1.0 - fraction) * value for value in barycentric]
            weights[local_vertex] += fraction
            point = [
                sum(weights[index] * seed_values[index][axis] for index in range(3))
                for axis in range(3)
            ]
            add_topology_candidate(
                point,
                {
                    "kind": "seed_triangle_barycentric",
                    "source_triangle_index": source_triangle_index,
                    "barycentric": weights,
                },
                vertex_index if fraction == 1.0 else None,
            )
        fixed_barycentric = (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5),
        )
        for weights in fixed_barycentric:
            point = [
                sum(weights[index] * seed_values[index][axis] for index in range(3))
                for axis in range(3)
            ]
            add_topology_candidate(
                point,
                {
                    "kind": "seed_triangle_barycentric",
                    "source_triangle_index": source_triangle_index,
                    "barycentric": list(weights),
                },
            )

        eligible_vertices = [
            index for index in candidates
            if geodesic[index] <= maximum_patch_radius_m
        ]
        directional_vertices: list[int] = []
        for direction_index in range(8):
            angle = 0.25 * math.pi * direction_index
            direction = (math.cos(angle), math.sin(angle))
            ranked = []
            for index in eligible_vertices:
                delta = [vertices[index][axis] - closest_point[axis] for axis in range(3)]
                x = sum(delta[axis] * tangent0[axis] for axis in range(3))
                y = sum(delta[axis] * tangent1[axis] for axis in range(3))
                ranked.append((x * direction[0] + y * direction[1], -geodesic[index], -index, index))
            if ranked:
                directional_vertices.append(max(ranked)[3])
        # A nearly planar tangent stencil can be rank-valid yet require large
        # opposing nodal forces when the source point sits off a curved bone
        # surface. Add deterministic 3-D support extrema from the same
        # connected geodesic patch. These are still exact existing vertices;
        # the extra directions improve the wrench basis without enlarging the
        # 12 mm patch or relaxing the amplification gate.
        for first in (-1.0, 0.0, 1.0):
            for second in (-1.0, 0.0, 1.0):
                for third in (-1.0, 0.0, 1.0):
                    if first == second == third == 0.0:
                        continue
                    magnitude = math.sqrt(first * first + second * second + third * third)
                    direction = (first / magnitude, second / magnitude, third / magnitude)
                    ranked = []
                    for index in eligible_vertices:
                        delta = [
                            vertices[index][axis] - closest_point[axis]
                            for axis in range(3)
                        ]
                        ranked.append((
                            sum(delta[axis] * direction[axis] for axis in range(3)),
                            -geodesic[index], -index, index,
                        ))
                    if ranked:
                        directional_vertices.append(max(ranked)[3])
        directional_vertices.extend(sorted(
            eligible_vertices,
            key=lambda index: (-geodesic[index], index),
        )[:4])
        for index in directional_vertices:
            add_topology_candidate(
                vertices[index], {"kind": "connected_bone_vertex", "vertex_index": index}, index,
            )

        # Retain the seed-triangle points plus tangent and 3-D surface extrema.
        # Thirty-two candidates bound the fallback to 35,960 four-point
        # combinations. This is paid only after the fast compass patch fails;
        # all candidates remain on the original connected surface and every
        # force, moment, radius, and amplification gate remains unchanged.
        topology_candidates = topology_candidates[:32]
        topology_candidate_count = len(topology_candidates)
        for selected_candidates in combinations(topology_candidates, 4):
            nodes = [candidate["point"] for candidate in selected_candidates]
            vertex_indices = [candidate["vertex_index"] for candidate in selected_candidates]
            consider_patch(
                nodes,
                [int(index) for index in vertex_indices]
                if all(index is not None for index in vertex_indices) else [],
                "connected_geodesic_topology_aware_exact_surface_points",
                [candidate["source"] for candidate in selected_candidates],
            )
    if best is not None:
        reason = (
            "admitted_topology_aware_exact_surface_patch"
            if best[1]["surface_patch_method"] == "connected_geodesic_topology_aware_exact_surface_points"
            else "admitted"
        )
        return best[1], reason
    return None, (
        "surface_patch_has_fewer_than_four_exact_surface_points"
        if topology_candidate_count < 4
        else "surface_patch_conditioning_failed_after_topology_aware_exact_surface_points"
    )


def _numi_human_semantic_enthesis_envelope(
    source_point: list[float], surfaces: list[dict[str, Any]],
    member_ids: tuple[str, ...], maximum_surface_distance_m: float,
    maximum_patch_radius_m: float, maximum_force_amplification: float,
    semantic_kind: str | None = None,
    migrate_endpoint_to_surface: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve an explicit same-body anatomical enthesis correspondence.

    A one-member correspondence uses the ordinary connected surface patch. A
    four-member correspondence is reserved for the source model's lumped
    EDL/FDL route: one exact closest point is selected on each named distal
    phalanx and the minimum-norm maps preserve the authored route endpoint's
    force and moment. This does not create four muscles or move that endpoint.
    """
    if not member_ids or len(surfaces) != len(member_ids):
        return None, "semantic_enthesis_map_member_missing"
    if tuple(surface.get("member_id") for surface in surfaces) != member_ids:
        return None, "semantic_enthesis_map_member_order_drifted"
    body_indices = {surface.get("body_index") for surface in surfaces}
    if len(body_indices) != 1 or not all(isinstance(value, int) for value in body_indices):
        return None, "semantic_enthesis_map_crosses_rigid_bodies"
    if len(surfaces) == 1:
        envelope, reason = _numi_human_tendon_surface_envelope(
            source_point, surfaces[0], maximum_surface_distance_m,
            maximum_patch_radius_m, maximum_force_amplification,
            migrate_endpoint_to_surface,
        )
        if envelope is None:
            return None, reason
        envelope["semantic_enthesis_map"] = {
            "kind": semantic_kind or "single_named_enthesis_member",
            "bone_member_ids": list(member_ids),
            "node_bone_member_ids": [member_ids[0]] * 4,
            "node_bone_stable_ids": [surfaces[0]["stable_id"]] * 4,
            "source_endpoint_migration_m": (
                envelope["surface_distance_m"] if migrate_endpoint_to_surface else 0.0
            ),
        }
        return envelope, "admitted_semantic_single_enthesis_map"
    if len(surfaces) != 4:
        return None, "semantic_enthesis_map_requires_one_or_four_members"

    nearest_records: list[dict[str, Any]] = []
    for surface in surfaces:
        nearest: tuple[float, int, list[float], list[float]] | None = None
        vertices = surface.get("vertices")
        triangles = surface.get("triangles")
        if not isinstance(vertices, list) or not isinstance(triangles, list):
            return None, "semantic_enthesis_map_surface_is_malformed"
        for triangle_index, triangle_indices in enumerate(triangles):
            triangle = [vertices[index] for index in triangle_indices]
            closest, barycentric = _tendon_closest_point_on_triangle(
                source_point, triangle,
            )
            squared = sum(
                (closest[axis] - source_point[axis]) ** 2 for axis in range(3)
            )
            tolerance = max(1.0e-18, 1.0e-12 * max(squared, nearest[0] if nearest is not None else 0.0))
            if (
                nearest is None or squared < nearest[0] - tolerance
                or (abs(squared - nearest[0]) <= tolerance and triangle_index < nearest[1])
            ):
                nearest = (squared, triangle_index, closest, barycentric)
        if nearest is None:
            return None, "semantic_enthesis_map_has_no_surface_triangle"
        squared, triangle_index, closest, barycentric = nearest
        nearest_records.append({
            "surface": surface,
            "distance_m": math.sqrt(squared),
            "source_triangle_index": triangle_index,
            "local_point_m": closest,
            "barycentric": barycentric,
        })
    representative = min(nearest_records, key=lambda record: record["distance_m"])
    if representative["distance_m"] > maximum_surface_distance_m:
        return None, "semantic_enthesis_representative_distance_exceeds_gate"
    nodes = [record["local_point_m"] for record in nearest_records]
    spread = max(
        math.sqrt(sum((node[axis] - source_point[axis]) ** 2 for axis in range(3)))
        for node in nodes
    )
    if spread > _NUMI_HUMAN_TOE_ENTHESIS_MAXIMUM_SPREAD_M:
        return None, "semantic_enthesis_spread_exceeds_gate"
    mapped = _tendon_envelope_force_maps(source_point, nodes, spread)
    if mapped is None:
        return None, "semantic_enthesis_force_map_is_singular"
    maps, metrics = mapped
    if (
        metrics["force_residual"] > 2.0e-6
        or metrics["moment_residual_m"] > 2.0e-8
        or metrics["sampled_total_force_amplification"] > maximum_force_amplification
    ):
        return None, "semantic_enthesis_force_map_conditioning_failed"
    representative_surface = representative["surface"]
    return {
        "body_index": representative_surface["body_index"],
        # NHTENDON2 retains one compact stable identity. The complete ordered
        # multi-bone identity stays in the manifest and each node position is
        # already expressed in the shared toes-body local frame.
        "bone_stable_id": representative_surface["stable_id"],
        "bone_member_id": representative_surface["member_id"],
        "source_triangle_index": representative["source_triangle_index"],
        "nearest_barycentric": representative["barycentric"],
        "nearest_local_point_m": representative["local_point_m"],
        "node_vertex_indices": [],
        "node_local_points_m": nodes,
        "force_maps": maps,
        "surface_distance_m": representative["distance_m"],
        "patch_radius_m": spread,
        **metrics,
        "semantic_enthesis_map": {
            "kind": "lumped_digitorum_route_to_four_lesser_toe_distal_phalanges",
            "bone_member_ids": list(member_ids),
            "node_bone_member_ids": list(member_ids),
            "node_bone_stable_ids": [
                record["surface"]["stable_id"] for record in nearest_records
            ],
            "node_surface_distances_from_source_point_m": [
                record["distance_m"] for record in nearest_records
            ],
            "maximum_semantic_spread_m": _NUMI_HUMAN_TOE_ENTHESIS_MAXIMUM_SPREAD_M,
            "source_endpoint_migration_m": 0.0,
            "source_force_law_count": 1,
            "inferred_independent_toe_actuator_count": 0,
        },
    }, "admitted_semantic_multi_enthesis_map"


def numi_human_tendon_attachment_envelope_payload(
    myosim_artifact: Path, bone_artifact: Path, output: Path,
    maximum_surface_distance_m: float = 0.012,
    maximum_patch_radius_m: float = 0.012,
    maximum_force_amplification: float = 4.0,
    migrate_semantic_rigid_foot_endpoints: bool = False,
    maximum_migrated_endpoint_distance_m: float = 0.025,
) -> dict[str, Any]:
    """Compile fail-closed BodyParts3D enthesis force-transfer laws.

    Automatic admission is deliberately limited to a body with exactly one
    registered NHBONES1 member. Multi-member exceptions require an exact
    source-pinned semantic table: hallux routes remain one-to-one, a lumped
    EDL/FDL terminal wrench spans four named lesser-toe distal phalanges,
    bilateral hip/tibia/fibula and rigid-foot routes select one declared same-body member,
    and source-named thoracic routes select their exact vertebra or rib.
    Every other multi-bone body, absent geometry, distant surface, or
    ill-conditioned patch remains an explicit source-site point law. No
    authored MyoSim site, route, path length, or force parameter is changed by
    the default compiler.  The explicit rigid-foot migration option emits
    NHTENDON3 and moves only 18 route-private, one-to-one named endpoints onto
    their exact registered surface; lumped EDL/FDL endpoints remain fixed.
    """
    for value, label in (
        (maximum_surface_distance_m, "maximum surface distance"),
        (maximum_patch_radius_m, "maximum patch radius"),
        (maximum_force_amplification, "maximum force amplification"),
        (maximum_migrated_endpoint_distance_m, "maximum migrated endpoint distance"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ImportError(f"Numi Human tendon envelope {label} must be finite and positive")
    artifact = myosim_artifact.resolve()
    manifest_path = artifact / "myosim-fullbody-reference.manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "numi.human.myosim-fullbody-reference.v1":
        raise ImportError("Numi Human tendon envelopes require the full-body MyoSim reference")
    source = manifest.get("source")
    source_sha = source.get("archive_sha256") if isinstance(source, dict) else None
    payloads = manifest.get("payloads")
    muscle_descriptor = payloads.get("muscles") if isinstance(payloads, dict) else None
    rigid_descriptor = payloads.get("rigid") if isinstance(payloads, dict) else None
    if (
        not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha)
        or not isinstance(muscle_descriptor, dict) or not isinstance(rigid_descriptor, dict)
    ):
        raise ImportError("Numi Human tendon envelope MyoSim provenance is incomplete")
    muscle_file, muscle_sha = muscle_descriptor.get("file"), muscle_descriptor.get("sha256")
    if not isinstance(muscle_file, str) or not isinstance(muscle_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", muscle_sha):
        raise ImportError("Numi Human tendon envelope muscle descriptor is invalid")
    muscle_path = artifact / muscle_file
    if not muscle_path.is_file() or sha256(muscle_path) != muscle_sha:
        raise ImportError("Numi Human tendon envelope muscle payload is missing or drifted")
    raw = muscle_path.read_bytes()
    header_format = "<8s9I32s"
    header_size = struct.calcsize(header_format)
    if len(raw) < header_size:
        raise ImportError("Numi Human tendon envelope muscle payload is truncated")
    magic, abi, body_count, muscle_count, site_count, wrap_count, route_count, tendon_count, reserved0, reserved1, embedded_sha = struct.unpack_from(
        header_format, raw,
    )
    try:
        architecture_count, architecture_bytes = _myosim_muscle_payload_architecture(
            magic, abi, muscle_count, reserved0, reserved1,
        )
    except ImportError as error:
        raise ImportError("Numi Human tendon envelope muscle payload ABI is invalid") from error
    expected_size = _myosim_muscle_payload_bytes(
        site_count, wrap_count, route_count, muscle_count,
        architecture_count, architecture_bytes,
    )
    if (
        body_count == 0 or muscle_count == 0 or site_count == 0 or tendon_count == 0
        or embedded_sha.hex() != source_sha
        or len(raw) != expected_size
    ):
        raise ImportError("Numi Human tendon envelope muscle payload ABI is invalid")
    offset = header_size
    sites = [struct.unpack_from("<I3f", raw, offset + 16 * index) for index in range(site_count)]
    offset += 16 * site_count + 64 * wrap_count
    routes = [struct.unpack_from("<4I", raw, offset + 16 * index) for index in range(route_count)]
    offset += 16 * route_count
    muscles = [struct.unpack_from("<4I37f", raw, offset + 164 * index) for index in range(muscle_count)]
    metadata = manifest.get("muscles")
    if not isinstance(metadata, list) or len(metadata) != muscle_count:
        raise ImportError("Numi Human tendon envelope muscle identity table is incomplete")
    surfaces_by_body, bone_descriptor, _ = _numi_human_bone_envelope_surfaces(
        bone_artifact, source_sha,
    )
    surfaces_by_member: dict[str, dict[str, Any]] = {}
    for body_index, surfaces in surfaces_by_body.items():
        for surface in surfaces:
            surface["body_index"] = body_index
            member_id = surface.get("member_id")
            if not isinstance(member_id, str) or member_id in surfaces_by_member:
                raise ImportError("Numi Human tendon envelope bone-member identity is invalid")
            surfaces_by_member[member_id] = surface
    source_component_members: dict[tuple[str, int], tuple[str, ...]] = {}
    source_component_point_reasons: dict[tuple[str, int], str] = {}
    source_component_fallback_surfaces: dict[
        tuple[str, int], dict[str, Any]
    ] = {}
    source_component_receipt = bone_descriptor.get(
        "source_component_enthesis_registration"
    )
    if source_component_receipt is not None:
        source_surfaces = {
            int(surface["source_component_index"]): surface
            for surface in source_component_receipt["source_component_surfaces"]
        }
        for record in source_component_receipt["endpoint_records"]:
            key = (str(record["muscle"]), int(record["endpoint_ordinal"]))
            if key in _NUMI_HUMAN_SEMANTIC_ENTHESIS_MEMBERS:
                raise ImportError(
                    f"Numi Human source-component enthesis duplicates semantic map {key}"
                )
            if record["disposition"] == _NUMI_HUMAN_SOURCE_COMPONENT_RIB_DISPOSITION:
                source_component_members[key] = tuple(record["bone_member_ids"])
                component_index = int(record["source_component_index"])
                source_surface = source_surfaces[component_index]
                source_component_fallback_surfaces[key] = {
                    "body_index": 20,
                    "stable_id": 0x80000000 | (component_index + 1),
                    "member_id": str(record["source_mechanics_surface_id"]),
                    "vertices": source_surface["vertices_core_m"],
                    "triangles": [
                        tuple(triangle) for triangle in source_surface["triangles"]
                    ],
                    "source_component_index": component_index,
                    "source_surface_content_sha256": source_surface[
                        "surface_content_sha256"
                    ],
                }
            else:
                source_component_point_reasons[key] = str(record["disposition"])
    endpoint_payload: list[bytes] = []
    envelope_payload: list[bytes] = []
    endpoint_manifest: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    admitted_distances: list[float] = []
    admitted_amplifications: list[float] = []
    semantic_toe_enthesis_count = 0
    semantic_limb_enthesis_count = 0
    semantic_axial_enthesis_count = 0
    source_component_enthesis_count = 0
    source_component_mechanics_surface_enthesis_count = 0
    compass_vertex_envelope_count = 0
    topology_aware_exact_surface_envelope_count = 0
    for muscle_index, (record, muscle_metadata) in enumerate(zip(muscles, metadata, strict=True)):
        route_offset, count = record[1], record[2]
        name = muscle_metadata.get("name") if isinstance(muscle_metadata, dict) else None
        if not isinstance(name, str) or muscle_metadata.get("source_actuator_index") != muscle_index or count < 2 or route_offset + count > route_count:
            raise ImportError("Numi Human tendon envelope route identity is invalid")
        for endpoint_ordinal, route_node_index in ((0, route_offset), (1, route_offset + count - 1)):
            route = routes[route_node_index]
            if route[0] != _MYOSIM_ROUTE_SITE or route[1] >= site_count:
                raise ImportError(f"Numi Human tendon envelope route {name} has no source-site endpoint")
            site_index = route[1]
            site = sites[site_index]
            body_index = site[0]
            source_point = [float(value) for value in site[1:]]
            if body_index >= body_count or not all(math.isfinite(value) for value in source_point):
                raise ImportError(f"Numi Human tendon envelope route {name} has an invalid source endpoint")
            surfaces = surfaces_by_body.get(body_index, [])
            envelope: dict[str, Any] | None = None
            semantic_key = (name, endpoint_ordinal)
            migrate_endpoint = (
                migrate_semantic_rigid_foot_endpoints
                and semantic_key in _NUMI_HUMAN_RIGID_FOOT_MIGRATABLE_ENTHESES
            )
            semantic_members = (
                _NUMI_HUMAN_SEMANTIC_ENTHESIS_MEMBERS.get(semantic_key)
                or source_component_members.get(semantic_key)
            )
            if semantic_members is not None:
                semantic_surfaces = [
                    surfaces_by_member.get(member_id) for member_id in semantic_members
                ]
                if any(surface is None for surface in semantic_surfaces):
                    reason = "semantic_enthesis_map_member_missing"
                elif any(surface["body_index"] != body_index for surface in semantic_surfaces):
                    reason = "semantic_enthesis_map_body_mismatch"
                else:
                    semantic_maximum_distance = (
                        maximum_migrated_endpoint_distance_m
                        if migrate_endpoint else maximum_surface_distance_m
                    )
                    semantic_kind = (
                        "source_topology_resolved_lateralized_rib_member"
                        if semantic_key in source_component_members
                        else _numi_human_semantic_enthesis_kind(
                            semantic_key, len(semantic_members),
                        )
                    )
                    envelope, reason = _numi_human_semantic_enthesis_envelope(
                        source_point, semantic_surfaces, semantic_members,
                        semantic_maximum_distance, maximum_patch_radius_m,
                        maximum_force_amplification,
                        semantic_kind,
                        migrate_endpoint,
                    )
                    if (
                        envelope is None
                        and not migrate_endpoint
                        and semantic_key in source_component_fallback_surfaces
                        and reason in {
                            "surface_distance_exceeds_gate",
                            "surface_patch_has_fewer_than_four_exact_surface_points",
                            "surface_patch_conditioning_failed_after_topology_aware_exact_surface_points",
                        }
                    ):
                        bodyparts_rejection_reason = reason
                        source_surface = source_component_fallback_surfaces[
                            semantic_key
                        ]
                        envelope, source_reason = _numi_human_tendon_surface_envelope(
                            source_point, source_surface,
                            maximum_surface_distance_m,
                            maximum_patch_radius_m,
                            maximum_force_amplification,
                        )
                        reason = source_reason
                        if envelope is not None:
                            reason = "admitted_exact_pinned_source_component_surface"
                            envelope["surface_kind"] = (
                                "exact_pinned_source_component_mechanics_fallback"
                            )
                            envelope["bodyparts_rejection_reason"] = (
                                bodyparts_rejection_reason
                            )
                            envelope["semantic_enthesis_map"] = {
                                "kind": (
                                    "source_topology_mechanics_surface_after_"
                                    "bodyparts_rejection"
                                ),
                                "bone_member_ids": list(semantic_members),
                                "source_mechanics_surface_id": source_surface[
                                    "member_id"
                                ],
                                "source_component_index": source_surface[
                                    "source_component_index"
                                ],
                                "source_surface_content_sha256": source_surface[
                                    "source_surface_content_sha256"
                                ],
                                "node_bone_member_ids": [
                                    source_surface["member_id"]
                                ] * 4,
                                "node_bone_stable_ids": [
                                    source_surface["stable_id"]
                                ] * 4,
                                "source_endpoint_migration_m": 0.0,
                                "bodyparts_rejection_reason": (
                                    bodyparts_rejection_reason
                                ),
                            }
            elif semantic_key in source_component_point_reasons:
                reason = source_component_point_reasons[semantic_key]
            elif not surfaces:
                reason = "body_has_no_registered_bone_surface"
            elif len(surfaces) != 1:
                reason = "body_has_multiple_bone_members_without_semantic_enthesis_map"
            else:
                envelope, reason = _numi_human_tendon_surface_envelope(
                    source_point, surfaces[0], maximum_surface_distance_m,
                    maximum_patch_radius_m, maximum_force_amplification,
                )
            if envelope is None:
                rejection_counts[reason] += 1
                mode = _NUMI_HUMAN_TENDON_POINT
                envelope_index = 0xFFFFFFFF
                stable_id = 0
                metrics = (0.0, 0.0, 0.0, 0.0)
                surface_manifest = None
            else:
                mode = (
                    _NUMI_HUMAN_TENDON_MIGRATED_ENVELOPE
                    if migrate_endpoint else _NUMI_HUMAN_TENDON_ENVELOPE
                )
                envelope_index = len(envelope_payload)
                stable_id = envelope["bone_stable_id"]
                metrics = (
                    envelope["surface_distance_m"],
                    envelope["sampled_total_force_amplification"],
                    envelope["patch_radius_m"],
                    envelope["moment_residual_m"],
                )
                node_values = [component for node in envelope["node_local_points_m"] for component in (*node, 0.0)]
                map_values = [
                    component
                    for matrix in envelope["force_maps"]
                    for row in matrix
                    for component in (*row, 0.0)
                ]
                envelope_payload.append(struct.pack(
                    "<4I68f", body_index, stable_id, envelope["source_triangle_index"], 4,
                    *node_values, *map_values,
                    envelope["surface_distance_m"], envelope["patch_radius_m"],
                    envelope["sampled_total_force_amplification"], envelope["l2_force_amplification"],
                ))
                admitted_distances.append(envelope["surface_distance_m"])
                admitted_amplifications.append(envelope["sampled_total_force_amplification"])
                if envelope.get("surface_patch_method") == "connected_geodesic_compass_vertices":
                    compass_vertex_envelope_count += 1
                elif envelope.get("surface_patch_method") == "connected_geodesic_topology_aware_exact_surface_points":
                    topology_aware_exact_surface_envelope_count += 1
                if "semantic_enthesis_map" in envelope:
                    if semantic_key in source_component_members:
                        semantic_axial_enthesis_count += 1
                        source_component_enthesis_count += 1
                        if envelope.get("surface_kind") == (
                            "exact_pinned_source_component_mechanics_fallback"
                        ):
                            source_component_mechanics_surface_enthesis_count += 1
                    elif semantic_key in _NUMI_HUMAN_TOE_ENTHESIS_MEMBERS:
                        semantic_toe_enthesis_count += 1
                    elif semantic_key in _NUMI_HUMAN_LIMB_ENTHESIS_MEMBERS:
                        semantic_limb_enthesis_count += 1
                    elif semantic_key in _NUMI_HUMAN_AXIAL_ENTHESIS_MEMBERS:
                        semantic_axial_enthesis_count += 1
                    else:
                        raise ImportError(
                            f"Numi Human admitted undeclared semantic enthesis {semantic_key}"
                        )
                surface_manifest = {
                    key: envelope[key] for key in (
                        "bone_member_id", "bone_stable_id", "source_triangle_index",
                        "nearest_barycentric", "nearest_local_point_m", "node_vertex_indices",
                        "node_local_points_m", "surface_distance_m", "patch_radius_m",
                        "force_residual", "moment_residual_m", "l2_force_amplification",
                        "sampled_total_force_amplification",
                    )
                }
                if "surface_patch_method" in envelope:
                    surface_manifest["surface_patch_method"] = envelope[
                        "surface_patch_method"
                    ]
                if "node_surface_sources" in envelope:
                    surface_manifest["node_surface_sources"] = envelope[
                        "node_surface_sources"
                    ]
                if "semantic_enthesis_map" in envelope:
                    surface_manifest["semantic_enthesis_map"] = envelope[
                        "semantic_enthesis_map"
                    ]
                if "surface_kind" in envelope:
                    surface_manifest["surface_kind"] = envelope["surface_kind"]
                    surface_manifest["bodyparts_rejection_reason"] = envelope[
                        "bodyparts_rejection_reason"
                    ]
            resolved_point = (
                envelope["resolved_local_point_m"]
                if envelope is not None and migrate_endpoint else source_point
            )
            endpoint_migration = (
                envelope["surface_distance_m"]
                if envelope is not None and migrate_endpoint else 0.0
            )
            endpoint_payload.append(struct.pack(
                "<8I8f", muscle_index, endpoint_ordinal, route_node_index, site_index,
                body_index, mode, envelope_index, stable_id,
                *resolved_point, *metrics, endpoint_migration,
            ))
            endpoint_manifest.append({
                "muscle": name,
                "muscle_index": muscle_index,
                "endpoint": "origin" if endpoint_ordinal == 0 else "insertion",
                "route_node_index": route_node_index,
                "source_site_index": site_index,
                "body_index": body_index,
                "source_local_point_m": source_point,
                "resolved_local_point_m": resolved_point,
                "endpoint_migration_m": endpoint_migration,
                "attachment_mode": (
                    "registered_bone_migrated_distributed_envelope"
                    if mode == _NUMI_HUMAN_TENDON_MIGRATED_ENVELOPE else
                    "registered_source_surface_distributed_envelope"
                    if envelope is not None and envelope.get("surface_kind") ==
                    "exact_pinned_source_component_mechanics_fallback" else
                    "registered_bone_distributed_envelope"
                    if envelope is not None else "source_site_point"
                ),
                "admission_reason": reason,
                "surface": surface_manifest,
            })
    if len(endpoint_payload) != 2 * muscle_count:
        raise ImportError("Numi Human tendon envelope endpoint coverage is incomplete")
    payload_magic = (
        _NUMI_HUMAN_TENDON_MIGRATED_ENVELOPE_MAGIC
        if migrate_semantic_rigid_foot_endpoints else _NUMI_HUMAN_TENDON_ENVELOPE_MAGIC
    )
    payload_abi = (
        _NUMI_HUMAN_TENDON_MIGRATED_ENVELOPE_ABI
        if migrate_semantic_rigid_foot_endpoints else _NUMI_HUMAN_TENDON_ENVELOPE_ABI
    )
    header = struct.pack(
        "<8s10I32s32s32s", payload_magic,
        payload_abi, body_count, muscle_count, site_count,
        len(endpoint_payload), len(envelope_payload), bone_descriptor["bone_count"],
        int(bone_descriptor["registration_fingerprint32"], 16), 0, 0,
        bytes.fromhex(source_sha), bytes.fromhex(muscle_sha), bytes.fromhex(bone_descriptor["sha256"]),
    )
    payload = b"".join([header, *endpoint_payload, *envelope_payload])
    if len(payload) != 144 + 64 * len(endpoint_payload) + 288 * len(envelope_payload):
        raise ImportError("Numi Human tendon envelope payload ABI size mismatch")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "numi-human-tendon-attachments.nhtendon"
    payload_path.write_bytes(payload)
    admitted_count = len(envelope_payload)
    point_count = len(endpoint_payload) - admitted_count
    manifest_value = {
        "schema": f"numi.human.tendon-attachment-envelope-payload.v{payload_abi}",
        "payload": {
            "file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
            "magic": f"NHTENDON{payload_abi}", "payload_abi": payload_abi,
        },
        "source": {
            "myosim_manifest": {"file": manifest_path.name, "sha256": sha256(manifest_path)},
            "myosim_archive_sha256": source_sha,
            "myosim_muscle_payload_sha256": muscle_sha,
            "bodyparts3d_bone_payload": bone_descriptor,
        },
        "admission": {
            "method": (
                "single_named_NHBONES1_member_exact_nearest_triangle_connected_surface_patch_"
                "with_deterministic_topology_aware_exact_triangle_quadrature_fallback_"
                "or_explicit_same_body_semantic_member_map_minimum_L2_wrench_distribution_"
                "or_exact_pinned_source_component_surface_only_after_BodyParts_rejection"
            ),
            "maximum_surface_distance_m": maximum_surface_distance_m,
            **({
                "maximum_migrated_endpoint_distance_m":
                    maximum_migrated_endpoint_distance_m,
            } if migrate_semantic_rigid_foot_endpoints else {}),
            "maximum_patch_radius_m": maximum_patch_radius_m,
            "maximum_sampled_total_force_amplification": maximum_force_amplification,
            "route_private_surface_migration": migrate_semantic_rigid_foot_endpoints,
            "migratable_scope": sorted(
                f"{muscle}:{endpoint}"
                for muscle, endpoint in _NUMI_HUMAN_RIGID_FOOT_MIGRATABLE_ENTHESES
            ) if migrate_semantic_rigid_foot_endpoints else [],
            "multiple_bone_members_fail_closed": True,
            "multiple_bone_exception": (
                "only exact source-pinned toe maps, declared bilateral hip/tibia/fibula/rigid-foot "
                "route-member maps, source-named thoracic maps, and a validated pinned-source "
                "component receipt are admitted; "
                "all other multi-bone bodies fail closed"
            ),
            "toe_semantic_enthesis_map": {
                f"{muscle}:{endpoint}": list(members)
                for (muscle, endpoint), members in sorted(
                    _NUMI_HUMAN_TOE_ENTHESIS_MEMBERS.items()
                )
            },
            "limb_semantic_enthesis_map": {
                f"{muscle}:{endpoint}": {
                    "bone_member_ids": list(members),
                    "kind": _numi_human_semantic_enthesis_kind(
                        (muscle, endpoint), len(members),
                    ),
                }
                for (muscle, endpoint), members in sorted(
                    _NUMI_HUMAN_LIMB_ENTHESIS_MEMBERS.items()
                )
            },
            "axial_semantic_enthesis_map": {
                f"{muscle}:{endpoint}": {
                    "bone_member_ids": list(members),
                    "kind": _numi_human_semantic_enthesis_kind(
                        (muscle, endpoint), len(members),
                    ),
                }
                for (muscle, endpoint), members in sorted(
                    _NUMI_HUMAN_AXIAL_ENTHESIS_MEMBERS.items()
                )
            },
            "source_component_enthesis_map": {
                f"{muscle}:{endpoint}": {
                    "bone_member_ids": list(members),
                    "kind": "source_topology_resolved_lateralized_rib_member",
                }
                for (muscle, endpoint), members in sorted(
                    source_component_members.items()
                )
            },
            "source_component_point_dispositions": {
                f"{muscle}:{endpoint}": reason
                for (muscle, endpoint), reason in sorted(
                    source_component_point_reasons.items()
                )
            },
            "maximum_toe_semantic_spread_m": _NUMI_HUMAN_TOE_ENTHESIS_MAXIMUM_SPREAD_M,
            "source_endpoint_migration_m": max(
                (item["endpoint_migration_m"] for item in endpoint_manifest),
                default=0.0,
            ),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "coverage": {
            "muscle_count": muscle_count,
            "mechanical_endpoint_count": len(endpoint_payload),
            "expected_endpoint_count": 2 * muscle_count,
            "distributed_surface_envelope_count": admitted_count,
            "registered_bone_distributed_envelope_count": (
                admitted_count - source_component_mechanics_surface_enthesis_count
            ),
            "registered_bone_migrated_distributed_envelope_count": sum(
                item["attachment_mode"] == "registered_bone_migrated_distributed_envelope"
                for item in endpoint_manifest
            ),
            "semantic_toe_enthesis_envelope_count": semantic_toe_enthesis_count,
            "semantic_limb_enthesis_envelope_count": semantic_limb_enthesis_count,
            "semantic_axial_enthesis_envelope_count": semantic_axial_enthesis_count,
            "source_component_enthesis_envelope_count": source_component_enthesis_count,
            "source_component_mechanics_surface_enthesis_envelope_count": (
                source_component_mechanics_surface_enthesis_count
            ),
            "compass_vertex_envelope_count": compass_vertex_envelope_count,
            "topology_aware_exact_surface_envelope_count": topology_aware_exact_surface_envelope_count,
            "source_site_point_fallback_count": point_count,
            "surface_coverage_fraction": admitted_count / len(endpoint_payload),
            "maximum_endpoint_migration_m": max(
                (item["endpoint_migration_m"] for item in endpoint_manifest),
                default=0.0,
            ),
            "maximum_admitted_surface_distance_m": max(admitted_distances, default=0.0),
            "maximum_admitted_sampled_total_force_amplification": max(admitted_amplifications, default=0.0),
        },
        "endpoints": endpoint_manifest,
        "runtime_contract": (
            (
                "replace only each admitted one-to-one named rigid-foot endpoint with a route-private exact bone-surface site; "
                "retain every other authored endpoint and the source force law; on Apple Metal distribute the resolved route's exact "
                "terminal force across four same-body registered attachment-surface nodes, conserve resultant force and moment, and derive the "
                "generalized contribution only through articulated point Jacobians"
            ) if migrate_semantic_rigid_foot_endpoints else (
                "retain each authored MyoSim route endpoint and force law; on Apple Metal distribute its exact terminal force "
                "across four same-body source-registered attachment-surface nodes with precompiled 3x3 maps, conserve resultant force and moment, and derive any "
                "generalized contribution only through articulated point Jacobians"
            )
        ),
        "status": "complete_endpoint_coverage_with_inferred_surface_envelopes_and_explicit_point_fallbacks",
        "evidence_boundary": (
            (
                "NHTENDON3 changes only the explicitly listed, one-to-one rigid-foot/hallux route endpoints to exact points on the "
                "fixed registered BodyParts3D bones. It does not move a bone, create toe articulation, split the lumped EDL/FDL laws, "
                "or constitute clinical validation. Native path, force, replay, and visual review are required before mechanical admission. "
            ) if migrate_semantic_rigid_foot_endpoints else ""
        ) + (
            "Admitted envelopes are simulation-inferred from the source-pinned BodyParts3D/MyoSim registration and strict "
            "distance/conditioning gates. They are not source-authored enthesis coordinates, a deformable tendon continuum, "
            "a clinical attachment certificate. The four-node "
            "topology-aware fallback uses exact points on the selected source triangle or vertices in its connected surface; "
            "it is a force-transfer discretization and does not warp, refine, or relabel anatomy. The four-node "
            "EDL/FDL map distributes one lumped source law; it does not claim four independently actuated toe muscles. "
            "The bilateral EO3 fallback is an exact pinned MyoSim thorax-component mechanics surface admitted only after "
            "the named BodyParts3D rib failed the unchanged distance gate. It is not a BodyParts3D bone, does not move a "
            "rib or endpoint, and is not a deformable cartilage or enthesis material law. "
            "The bilateral hip/tibia/fibula/rigid-foot member assignments resolve only which exact source bone on an already-owned "
            "rigid body receives the unchanged endpoint wrench. The thoracic assignments likewise resolve only explicit "
            "MyoSim Tn/Rn/QL-12 labels to exact same-body BodyParts3D vertebrae or ribs. These mappings are not "
            "source-authored or clinical enthesis areas."
        ),
    }
    write_json(output / "numi-human-tendon-attachments.manifest.json", manifest_value)
    write_json(output / "numi-human-pack.manifest.json", {
        "schema": "numi.human.pack.v2",
        "owner": "Numi Lab Human",
        "payloads": {
            "rigid": rigid_descriptor,
            "muscles": muscle_descriptor,
            "support_contact": payloads.get("support_contact"),
            "bone_surfaces": bone_descriptor,
            "tendon_attachments": manifest_value["payload"],
        },
        "coverage": manifest_value["coverage"],
        "status": manifest_value["status"],
        "source_authorities": {
            "geometry": "BodyParts3D 4.0",
            "active_full_body_seed": "compiled MyoSim full-body source program",
            "lower_body_comparison": "OpenSim RajagopalLaiUhlrich2023",
            "upper_extremity_comparison": "OpenSim Upper Extremity Dynamic Model",
        },
        "runtime_owner": "Numi Lab C++/Metal; Python is an offline compiler only",
    })
    return manifest_value


def numi_human_achilles_surface_receipt(
    sources: Path, registration_path: Path, myosim_artifact: Path, output: Path,
) -> dict[str, Any]:
    """Register the six bilateral triceps-surae insertions to calcaneus faces.

    This is an explicit endpoint migration receipt, not an automatic admission
    rule. It preserves the exact source triangle and barycentric point so the
    native probe can measure the resulting path/force change.
    """
    registration_path = registration_path.resolve()
    registration = read_json(registration_path)
    if registration.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2":
        raise ImportError("Achilles surface receipt requires the attachment-refined v2 bone registration")
    anchors = registration.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != len(_BODYPARTS_MYOSIM_BONE_ANCHORS):
        raise ImportError("Achilles surface receipt requires the complete BodyParts3D bone registration")
    artifact = myosim_artifact.resolve()
    manifest = read_json(artifact / "myosim-fullbody-reference.manifest.json")
    source = manifest.get("source")
    source_sha = source.get("archive_sha256") if isinstance(source, dict) else None
    if not isinstance(source_sha, str):
        raise ImportError("Achilles surface receipt has no MyoSim source identity")
    payload_descriptor = manifest.get("payloads", {}).get("muscles")
    if not isinstance(payload_descriptor, dict):
        raise ImportError("Achilles surface receipt has no MyoSim muscle payload")
    payload_path = artifact / str(payload_descriptor.get("file"))
    expected_sha = payload_descriptor.get("sha256")
    if not payload_path.is_file() or not isinstance(expected_sha, str) or sha256(payload_path) != expected_sha:
        raise ImportError("Achilles surface receipt muscle payload is missing or has drifted")
    payload = payload_path.read_bytes()
    header_size = struct.calcsize("<8s9I32s")
    magic, abi, body_count, muscle_count, site_count, wrap_count, route_count, _, reserved0, reserved1, embedded_sha = struct.unpack_from(
        "<8s9I32s", payload
    )
    try:
        architecture_count, architecture_bytes = _myosim_muscle_payload_architecture(
            magic, abi, muscle_count, reserved0, reserved1,
        )
    except ImportError as error:
        raise ImportError("Achilles surface receipt muscle payload ABI is invalid") from error
    expected_size = _myosim_muscle_payload_bytes(
        site_count, wrap_count, route_count, muscle_count,
        architecture_count, architecture_bytes,
    )
    if embedded_sha.hex() != source_sha or len(payload) != expected_size:
        raise ImportError("Achilles surface receipt muscle payload ABI is invalid")
    offset = header_size
    sites = [struct.unpack_from("<I3f", payload, offset + 16 * index) for index in range(site_count)]
    offset += 16 * site_count + 64 * wrap_count
    routes = [struct.unpack_from("<4I", payload, offset + 16 * index) for index in range(route_count)]
    offset += 16 * route_count
    muscle_records = [struct.unpack_from("<4I37f", payload, offset + 164 * index) for index in range(muscle_count)]
    metadata = manifest.get("muscles")
    if not isinstance(metadata, list) or len(metadata) != muscle_count:
        raise ImportError("Achilles surface receipt has no complete muscle identity table")
    muscle_index = {
        entry.get("name"): index for index, entry in enumerate(metadata) if isinstance(entry, dict)
    }

    def dot(left: list[float], right: list[float]) -> float:
        return sum(left[axis] * right[axis] for axis in range(3))

    def subtract(left: list[float], right: list[float]) -> list[float]:
        return [left[axis] - right[axis] for axis in range(3)]

    def closest_barycentric(point: list[float], triangle: list[list[float]]) -> tuple[list[float], list[float]]:
        a, b, c = triangle
        ab, ac, ap = subtract(b, a), subtract(c, a), subtract(point, a)
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return list(a), [1.0, 0.0, 0.0]
        bp = subtract(point, b)
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return list(b), [0.0, 1.0, 0.0]
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return [a[axis] + v * ab[axis] for axis in range(3)], [1.0 - v, v, 0.0]
        cp = subtract(point, c)
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return list(c), [0.0, 0.0, 1.0]
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return [a[axis] + w * ac[axis] for axis in range(3)], [1.0 - w, 0.0, w]
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            edge = subtract(c, b)
            return [b[axis] + w * edge[axis] for axis in range(3)], [0.0, 1.0 - w, w]
        denominator = va + vb + vc
        if abs(denominator) <= 1.0e-18:
            raise ImportError("Achilles surface receipt encountered a degenerate calcaneus triangle")
        inverse = 1.0 / denominator
        v, w = vb * inverse, vc * inverse
        barycentric = [1.0 - v - w, v, w]
        return [sum(barycentric[index] * triangle[index][axis] for index in range(3)) for axis in range(3)], barycentric

    records: list[dict[str, Any]] = []
    for side, body_name, names in (
        ("right", "calcn_r", ("gaslat_r", "gasmed_r", "soleus_r")),
        ("left", "calcn_l", ("gaslat_l", "gasmed_l", "soleus_l")),
    ):
        stable_index = next(
            index for index, specification in enumerate(_BODYPARTS_MYOSIM_BONE_ANCHORS, start=1)
            if specification["myosim_body"] == body_name
        )
        anchor = anchors[stable_index - 1]
        source_record, target, registration_record = anchor.get("source"), anchor.get("target"), anchor.get("registration")
        if not all(isinstance(value, dict) for value in (source_record, target, registration_record)):
            raise ImportError(f"Achilles surface receipt has no {side} calcaneus anchor")
        body_index = target.get("core_body_index")
        matrix = registration_record.get("source_obj_mm_to_core_inertial_body_m")
        if not isinstance(body_index, int) or not isinstance(matrix, list) or len(matrix) != 4:
            raise ImportError(f"Achilles surface receipt has an invalid {side} calcaneus transform")
        _, member, obj = _bodyparts_obj_member(
            sources.resolve(), source_record["hierarchy"], source_record["member_id"]
        )
        vertices_mm, triangles = _bodyparts_obj_triangles(obj, member)
        vertices_local = [
            [sum(matrix[row][column] * vertex[column] for column in range(3)) + matrix[row][3] for row in range(3)]
            for vertex in vertices_mm
        ]
        for name in names:
            index = muscle_index.get(name)
            if not isinstance(index, int):
                raise ImportError(f"Achilles surface receipt has no MyoSim muscle {name}")
            route_offset, route_nodes = muscle_records[index][1], muscle_records[index][2]
            terminal = routes[route_offset + route_nodes - 1]
            if terminal[0] != _MYOSIM_ROUTE_SITE or terminal[1] >= len(sites):
                raise ImportError(f"Achilles surface receipt muscle {name} has no terminal site")
            site = sites[terminal[1]]
            if site[0] != body_index:
                raise ImportError(f"Achilles surface receipt muscle {name} terminates on the wrong body")
            point = [float(value) for value in site[1:]]
            nearest: tuple[float, int, list[list[float]], list[float], list[float]] | None = None
            for triangle_index, triangle_indices in enumerate(triangles):
                triangle = [vertices_local[vertex] for vertex in triangle_indices]
                candidate, barycentric = closest_barycentric(point, triangle)
                squared = sum((candidate[axis] - point[axis]) ** 2 for axis in range(3))
                if nearest is None or squared < nearest[0]:
                    nearest = (squared, triangle_index, triangle, candidate, barycentric)
            if nearest is None:
                raise ImportError(f"Achilles surface receipt has no calcaneus triangle for {name}")
            squared, triangle_index, triangle, candidate, barycentric = nearest
            records.append({
                "muscle": name, "endpoint": "insertion", "body_index": body_index,
                "body": body_name, "bone_member_id": source_record["member_id"],
                "bone_stable_id": stable_index, "source_triangle_index": triangle_index,
                "triangle_local_m": triangle, "barycentric": barycentric,
                "source_site_index": terminal[1], "source_local_point_m": point,
                "resolved_local_point_m": candidate, "endpoint_migration_m": math.sqrt(squared),
            })
    receipt = {
        "schema": "numi.human.tendon-surface-registration.v1",
        "scope": "bilateral Achilles insertions for gaslat, gasmed, and soleus",
        "source": {"registration": {"file": registration_path.name, "sha256": sha256(registration_path)},
                   "myosim_archive_sha256": source_sha, "myosim_muscle_payload_sha256": expected_sha},
        "records": records,
        "summary": {"record_count": len(records),
                    "maximum_endpoint_migration_m": max(record["endpoint_migration_m"] for record in records),
                    "rms_endpoint_migration_m": math.sqrt(sum(record["endpoint_migration_m"] ** 2 for record in records) / len(records))},
        "status": "explicit_inferred_surface_registration_for_native_force_validation",
        "admission": {
            "mechanical": False,
            "reason": (
                f"maximum {max(record['endpoint_migration_m'] for record in records) * 1000.0:.3f} mm "
                "route-site migration requires native path/force impact review; retain source point mechanics"
            ),
        },
        "evidence_boundary": "The receipt makes each route migration explicit and auditable. It is a simulation attachment registration, not medical or clinical validation.",
    }
    write_json(output.resolve(), receipt)
    return receipt


def bodyparts_myosim_fullbody_soft_tissue_visual_payload(
    sources: Path, anatomy: dict[str, Any], registration_path: Path, myosim_artifact: Path, output: Path,
    stable_id_subset: set[int] | None = None,
) -> dict[str, Any]:
    """Package source-authored limb, shoulder, arm, hand and abdominal surfaces.

    Ordinary two-body surfaces bind to their first and final **authored MyoSim
    route sites**. Multi-slip or multi-body surfaces instead retain every
    named route body and store four sparse, route-proximity influences per
    vertex. This prevents a shared digital flexor/extensor atlas surface from
    being dragged only by the middle finger. The calcaneal tendon is different:
    the source routes for its two
    gastrocnemius heads start on the femur, while soleus starts on the tibia
    and all three terminate on the calcaneus.  Its visual surface therefore
    inherits a three-body blend from the nearest named source muscle surface
    and locks its distal insertion to the exact calcaneal source surface.
    This is kinematic presentation data, not a deformable tendon or a
    replacement for the MyoSim force path.
    """
    registration_file = registration_path.resolve()
    registration = read_json(registration_file)
    if registration.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2" or \
            registration.get("status") not in {
                "provisional_visual_registration_not_admitted_to_collision_or_physics",
                "inferred_attachment_surface_visual_registration_not_admitted_to_collision_or_physics",
            }:
        raise ImportError("BodyParts3D full-body tissue payload requires a supported v2 visual registration")
    expected_bodyparts = {"id": anatomy.get("source_id"), "version": anatomy.get("version"), "archives": anatomy.get("archives")}
    source = registration.get("source")
    if not isinstance(source, dict) or source.get("bodyparts") != expected_bodyparts:
        raise ImportError("BodyParts3D full-body tissue payload registration does not match parsed source provenance")
    myosim = source.get("myosim")
    source_sha = myosim.get("source", {}).get("archive_sha256") if isinstance(myosim, dict) else None
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ImportError("BodyParts3D full-body tissue payload has no MyoSim source SHA-256")
    coordinates = registration.get("coordinate_system")
    global_matrix = coordinates.get("global_source_mm_to_myosim_world_m") if isinstance(coordinates, dict) else None
    if not isinstance(global_matrix, list) or len(global_matrix) != 4:
        raise ImportError("BodyParts3D full-body tissue payload has no global rest-frame registration")
    # Validate it eagerly before using its rows to transform hundreds of exact OBJ vertices.
    _bodyparts_visual_local_pose(global_matrix, "BodyParts3D full-body tissue global transform")
    bodies, route_muscles, myosim_manifest = _myosim_surface_route_context(myosim_artifact, source_sha)
    element_names = _bodyparts_source_element_names(sources)
    specifications = _bodyparts_myosim_surface_specifications()
    if stable_id_subset is not None:
        if not stable_id_subset or any(
            not isinstance(stable_id, int) or not 1 <= stable_id <= len(specifications)
            for stable_id in stable_id_subset
        ):
            raise ImportError("BodyParts3D full-body tissue subset has an invalid stable surface ID")
        # Stable IDs are part of NHTISS3, not output-order counters. A focused
        # source subset must therefore retain its original IDs so downstream
        # visual supplements can unambiguously reuse the exact named records.
        # The shared right calcaneal tendon needs the three source muscle
        # surfaces that establish its femur/tibia inheritance; reject a subset
        # that would silently infer those weights from unrelated anatomy.
        if 7 in stable_id_subset and not {1, 3, 5}.issubset(stable_id_subset):
            raise ImportError(
                "BodyParts3D calcaneal-tendon subset requires stable IDs 1, 3, and 5 "
                "for its named gastrocnemius/soleus body-weight inheritance"
            )
    registration_anchors = registration.get("anchors")
    if not isinstance(registration_anchors, list):
        raise ImportError("BodyParts3D full-body tissue payload has no visual-skeleton anchors")
    secondary_bone_sources: dict[str, dict[str, Any]] = {}
    body_bone_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    body_local_registrations: dict[str, tuple[list[float], list[float], float]] = {}
    for anchor in registration_anchors:
        if not isinstance(anchor, dict):
            raise ImportError("BodyParts3D full-body tissue payload has an invalid visual-skeleton anchor")
        source_record, target_record = anchor.get("source"), anchor.get("target")
        if not isinstance(source_record, dict) or not isinstance(target_record, dict):
            raise ImportError("BodyParts3D full-body tissue payload has an incomplete visual-skeleton anchor")
        target_name = target_record.get("name")
        member_id, hierarchy = source_record.get("member_id"), source_record.get("hierarchy")
        if not isinstance(target_name, str) or not isinstance(member_id, str) or \
                not isinstance(hierarchy, str):
            raise ImportError("BodyParts3D full-body tissue payload has an invalid visual-skeleton anchor identity")
        secondary_bone_sources.setdefault(target_name, source_record)
        body_bone_sources[target_name].append(source_record)
        registration_record = anchor.get("registration")
        if not isinstance(registration_record, dict):
            raise ImportError("BodyParts3D full-body tissue payload has no source-bone local registration")
        local_matrix = registration_record.get("source_obj_mm_to_core_inertial_body_m")
        body_local_registrations.setdefault(
            target_name,
            tuple(_bodyparts_visual_local_pose(
                local_matrix, f"BodyParts3D full-body tissue {target_name} local registration",
            )),
        )

    vertices_payload: list[bytes] = []
    indices_payload: list[int] = []
    records_payload: list[bytes] = []
    bindings_payload: list[bytes] = []
    provenance: list[dict[str, Any]] = []
    # Earlier source muscle surfaces are the only anatomical correspondences
    # used to distribute a shared tendon across multiple source bodies.  They
    # retain exact OBJ vertices and their own source-route-derived binding
    # weights; this never guesses a new origin or insertion.
    source_surface_bindings: dict[str, dict[str, Any]] = {}
    for stable_id, specification in enumerate(specifications, start=1):
        if stable_id_subset is not None and stable_id not in stable_id_subset:
            continue
        member_id, label = specification.get("member_id"), specification.get("source_name")
        source_muscles = specification.get("myosim_muscles")
        if not isinstance(member_id, str) or not isinstance(label, str) or (member_id, label) not in element_names:
            raise ImportError("BodyParts3D full-body tissue surface-map source identity drifted")
        if not isinstance(source_muscles, list) or not source_muscles or any(not isinstance(value, str) for value in source_muscles):
            raise ImportError("BodyParts3D full-body tissue surface-map has no named MyoSim muscles")
        matched_routes = []
        for muscle_name in source_muscles:
            route = route_muscles.get(muscle_name)
            if route is None:
                raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has unknown MyoSim muscle {muscle_name}")
            matched_routes.append(route)
        archive_path, member, obj = _bodyparts_obj_member(sources, "is_a", member_id)
        vertices_mm, triangles = _bodyparts_obj_triangles(obj, member)
        layer_name = specification.get("layer", "muscle")
        layer = _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE if layer_name == "muscle" else _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON if layer_name == "tendon" else None
        if layer is None:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has an invalid layer")
        source_component_selection: dict[str, Any] | None = None
        if (
            layer == _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON
            or member_id in _NUMI_HUMAN_HALLUX_DOMINANT_SOURCE_SURFACE_MEMBERS
        ):
            # These named OBJ members are compound source meshes. Their
            # dominant sheet is the only complete connected anatomical
            # surface; small disconnected export shards become floating or
            # stretched fragments after articulated posing. Preserve the
            # exact dominant sheet, not a remeshed repair.
            vertices_mm, triangles, source_component_selection = \
                _bodyparts_largest_connected_surface_component(vertices_mm, triangles, member)
        normals = _bodyparts_vertex_normals(vertices_mm, triangles, member)
        global_vertices = [[sum(global_matrix[row][column] * vertex[column] for column in range(3)) + global_matrix[row][3] for row in range(3)] for vertex in vertices_mm]
        explicit_primary, explicit_secondary = specification.get("primary_body"), specification.get("secondary_body")
        if explicit_primary is None and explicit_secondary is None:
            route_pairs = {
                (route["primary_body"], route["secondary_body"])
                for route in matched_routes
            }
            binding_names = []
            for route in matched_routes:
                for name in route["binding_bodies"]:
                    if name not in binding_names:
                        binding_names.append(name)
            if not 2 <= len(binding_names) <= _BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_MAX_BINDINGS:
                raise ImportError(
                    f"BodyParts3D full-body tissue surface {member_id} resolves "
                    f"{len(binding_names)} route bodies; supported range is 2.."
                    f"{_BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_MAX_BINDINGS}"
                )
            if len(route_pairs) == 1 and len(binding_names) == 2:
                primary_name, secondary_name = next(iter(route_pairs))
                # Preserve source endpoint order for the compact two-body
                # case; the route-body discovery order is otherwise used.
                binding_names = [primary_name, secondary_name]
                endpoint_source = "all_named_authored_myosim_route_endpoints"
            else:
                primary_name = matched_routes[0]["primary_body"]
                secondary_name = matched_routes[0]["secondary_body"]
                endpoint_source = (
                    "all_named_authored_myosim_route_nodes_with_sparse_four_influence_binding"
                )
        elif layer == _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON and \
                isinstance(explicit_primary, str) and isinstance(explicit_secondary, str):
            primary_name, secondary_name = explicit_primary, explicit_secondary
            # Preserve the full source-route body set instead of falsely
            # collapsing a shared tendon to the explicitly named tibia-to-
            # calcaneus pair.  The distal body is intentionally last so the
            # source-surface lock has one unambiguous attachment target.
            proximal_names: list[str] = []
            for route in matched_routes:
                for name in (route["primary_body"], route["secondary_body"]):
                    if name != secondary_name and name not in proximal_names:
                        proximal_names.append(name)
            binding_names = [*proximal_names, secondary_name]
            if len(binding_names) != 3 or primary_name not in binding_names:
                raise ImportError(
                    f"BodyParts3D shared tendon {member_id} does not resolve exactly three source endpoint bodies"
                )
            endpoint_source = "all_named_authored_myosim_route_endpoints_with_shared_tendon_three_body_binding"
        else:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has incomplete explicit endpoint bodies")

        binding_targets: list[dict[str, Any]] = []
        binding_local_poses: list[tuple[list[float], list[float], float]] = []
        binding_transforms: list[float] = []
        for binding_name in binding_names:
            target = bodies.get(binding_name)
            if not isinstance(target, dict):
                raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has unresolved endpoint body {binding_name}")
            position = _myosim_vector(
                target.get("default_com_position_world_m"),
                f"BodyParts3D {member_id} {binding_name} position",
            )
            quaternion = list(target.get("default_inertial_quaternion_world_xyzw", []))
            _myosim_matrix_from_quaternion_xyzw(quaternion)
            local_pose = body_local_registrations.get(binding_name)
            if local_pose is None:
                local_pose = tuple(_bodyparts_visual_local_pose(
                    _bodyparts_local_registration_matrix(global_matrix, position, quaternion),
                    f"BodyParts3D {member_id} {binding_name} transform",
                ))
            translation, rotation, scale = local_pose
            binding_targets.append(target)
            binding_local_poses.append((translation, rotation, scale))
            binding_transforms.extend([*translation, *rotation, scale])
        if len(binding_targets) != len(binding_names) or \
                len(binding_transforms) != 8 * len(binding_names):
            raise ImportError(
                f"BodyParts3D full-body tissue surface {member_id} has invalid body binding arity"
            )

        primary_target, secondary_target = bodies.get(primary_name), bodies.get(secondary_name)
        if not isinstance(primary_target, dict) or not isinstance(secondary_target, dict) or primary_name == secondary_name:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has unresolved primary/secondary anatomy")
        primary_position = _myosim_vector(primary_target.get("default_com_position_world_m"), f"BodyParts3D {member_id} primary position")
        secondary_position = _myosim_vector(secondary_target.get("default_com_position_world_m"), f"BodyParts3D {member_id} secondary position")
        body_axis = _myosim_subtract(secondary_position, primary_position)
        squared_axis = sum(value * value for value in body_axis)
        if squared_axis <= 1.0e-10:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has coincident endpoint centres")
        projections = [sum((vertex[axis] - primary_position[axis]) * body_axis[axis] for axis in range(3)) for vertex in global_vertices]
        minimum, maximum = min(projections), max(projections)
        if maximum - minimum <= 1.0e-6:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has no two-body blend extent")
        base_primary_weights = [
            max(0.0, min(1.0, (maximum - projection) / (maximum - minimum)))
            for projection in projections
        ]

        attachment_weight_lock: dict[str, Any] | None = None
        primary_attachment_weight_lock: dict[str, Any] | None = None
        toe_enthesis_weight_lock: dict[str, Any] | None = None
        stored_vertices_m = [[coordinate * 0.001 for coordinate in vertex] for vertex in vertices_mm]
        stored_normals = normals
        if layer == _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON:
            secondary_binding_index = binding_names.index(secondary_name)
            secondary_local_pose = binding_local_poses[secondary_binding_index]
            # The bone payload uses its per-anchor local registration, which
            # may refine the initial global fit.  Put the tendon into that
            # exact calcaneus rest frame before finding or projecting its
            # insertion, otherwise matching source OBJ coordinates can still
            # land on a visibly different rendered bone surface.
            global_vertices = _bodyparts_source_mm_to_body_world(
                vertices_mm, secondary_position,
                list(secondary_target.get("default_inertial_quaternion_world_xyzw", [])),
                *secondary_local_pose,
            )
            bone_source = secondary_bone_sources.get(secondary_name)
            if not isinstance(bone_source, dict):
                raise ImportError(
                    f"BodyParts3D tendon surface {member_id} has no named secondary-bone source mesh"
                )
            bone_member_id, bone_hierarchy = bone_source.get("member_id"), bone_source.get("hierarchy")
            if not isinstance(bone_member_id, str) or not isinstance(bone_hierarchy, str):
                raise ImportError(
                    f"BodyParts3D tendon surface {member_id} has an invalid secondary-bone source mesh"
                )
            _, bone_member, bone_obj = _bodyparts_obj_member(sources, bone_hierarchy, bone_member_id)
            bone_vertices_mm, bone_triangles = _bodyparts_obj_triangles(bone_obj, bone_member)
            bone_vertices_world_m = _bodyparts_source_mm_to_body_world(
                bone_vertices_mm, secondary_position,
                list(secondary_target.get("default_inertial_quaternion_world_xyzw", [])),
                *secondary_local_pose,
            )
            # Passing unit primary weights yields a pure 1 -> 0 distal lock
            # factor, independent of the arbitrary tibia-to-calcaneus body-
            # centre projection used by the old two-body presentation path.
            distal_attenuation, attachment_weight_lock = _bodyparts_secondary_attachment_weight_lock(
                global_vertices, [1.0] * len(global_vertices), bone_vertices_world_m, bone_triangles,
            )
            triangles, interior_cap_trim = _bodyparts_drop_interior_tendon_cap_triangles(
                triangles, distal_attenuation, member,
            )
            # The named-body weight lock keeps the distal source vertices in
            # the calcaneus frame, but it cannot repair a rest-frame gap
            # between independently authored tendon and bone surfaces.  Move
            # only that already locked/feathered boundary onto the exact
            # named calcaneal triangles.  This replaces the coarse generated
            # render-time collar with a continuous source-topology surface.
            global_vertices, surface_projection = _bodyparts_project_tendon_attachment_band(
                global_vertices, distal_attenuation, bone_vertices_world_m, bone_triangles,
            )
            global_vertices, triangles, distal_attenuation, enthesis_stitch = _bodyparts_stitch_tendon_enthesis_band(
                global_vertices, triangles, distal_attenuation,
                bone_vertices_world_m, bone_triangles, member,
            )
            stored_vertices_m = _bodyparts_world_to_body_stored_m(
                global_vertices, secondary_position,
                list(secondary_target.get("default_inertial_quaternion_world_xyzw", [])),
                *secondary_local_pose,
                f"BodyParts3D tendon {member_id} attachment projection",
            )
            stored_normals = _bodyparts_vertex_normals(
                [[coordinate * 1000.0 for coordinate in vertex] for vertex in stored_vertices_m], triangles, member,
            )
            attachment_weight_lock.update({
                "secondary_body": secondary_name,
                "secondary_bone_member_id": bone_member_id,
                "secondary_bone_member": bone_member,
                "secondary_bone_member_sha256": hashlib.sha256(bone_obj).hexdigest(),
                "surface_projection": surface_projection,
                "interior_cap_triangle_trim": interior_cap_trim,
                "enthesis_stitch": enthesis_stitch,
            })
            contributor_bindings = [
                binding for binding in source_surface_bindings.values()
                if set(binding["myosim_muscles"]).intersection(source_muscles)
            ]
            if not contributor_bindings:
                raise ImportError(
                    f"BodyParts3D shared tendon {member_id} has no prior named source muscle surface to inherit"
                )
            vertex_weights: list[list[float]] = []
            for vertex, attenuation in zip(global_vertices, distal_attenuation, strict=True):
                nearest: tuple[float, dict[str, Any], int] | None = None
                for contributor in contributor_bindings:
                    for contributor_index, candidate in enumerate(contributor["global_vertices"]):
                        squared_distance = sum(
                            (vertex[axis] - candidate[axis]) ** 2 for axis in range(3)
                        )
                        if nearest is None or squared_distance < nearest[0]:
                            nearest = (squared_distance, contributor, contributor_index)
                if nearest is None:
                    raise ImportError(f"BodyParts3D shared tendon {member_id} has no source muscle vertex")
                _, contributor, contributor_index = nearest
                weights_by_name = {name: 0.0 for name in binding_names}
                for name, weight in zip(
                    contributor["binding_names"], contributor["vertex_weights"][contributor_index], strict=True
                ):
                    if name in weights_by_name:
                        weights_by_name[name] += weight
                inherited_total = sum(weights_by_name.values())
                if inherited_total <= 1.0e-8:
                    raise ImportError(f"BodyParts3D shared tendon {member_id} inherited no compatible muscle body weight")
                weights = [weights_by_name[name] / inherited_total * attenuation for name in binding_names]
                weights[-1] += 1.0 - attenuation
                total = sum(weights)
                if not math.isfinite(total) or abs(total - 1.0) > 1.0e-6:
                    raise ImportError(f"BodyParts3D shared tendon {member_id} has non-unit three-body weight")
                vertex_weights.append(weights)
            attachment_weight_lock.update({
                "method": "exact_source_triangle_distal_lock_plus_nearest_named_source_muscle_three_body_weight_inheritance",
                "body_bindings": binding_names,
                "contributing_source_surface_members": sorted(binding["member_id"] for binding in contributor_bindings),
            })
        else:
            visual_binding = specification.get("visual_binding")
            if visual_binding is not None and not isinstance(visual_binding, dict):
                raise ImportError(
                    f"BodyParts3D full-body tissue surface {member_id} has an invalid visual binding"
                )
            if visual_binding is not None:
                if (
                    visual_binding.get("method") !=
                        "primary_source_bone_attachment_band"
                    or len(binding_names) != 2
                    or len(route_pairs) != 1
                ):
                    raise ImportError(
                        f"BodyParts3D full-body tissue surface {member_id} has an unsupported visual binding"
                    )
                lock_radius_m = visual_binding.get("lock_radius_m")
                feather_radius_m = visual_binding.get("feather_radius_m")
                if (
                    not isinstance(lock_radius_m, (int, float))
                    or isinstance(lock_radius_m, bool)
                    or not isinstance(feather_radius_m, (int, float))
                    or isinstance(feather_radius_m, bool)
                ):
                    raise ImportError(
                        f"BodyParts3D full-body tissue surface {member_id} has invalid attachment-band radii"
                    )
                primary_bone_source = secondary_bone_sources.get(primary_name)
                if not isinstance(primary_bone_source, dict):
                    raise ImportError(
                        f"BodyParts3D full-body tissue surface {member_id} has no named primary-bone source mesh"
                    )
                primary_bone_member_id = primary_bone_source.get("member_id")
                primary_bone_hierarchy = primary_bone_source.get("hierarchy")
                if not isinstance(primary_bone_member_id, str) or not isinstance(
                    primary_bone_hierarchy, str
                ):
                    raise ImportError(
                        f"BodyParts3D full-body tissue surface {member_id} has an invalid primary-bone source mesh"
                    )
                _, primary_bone_member, primary_bone_obj = _bodyparts_obj_member(
                    sources, primary_bone_hierarchy, primary_bone_member_id,
                )
                primary_bone_vertices_mm, _ = _bodyparts_obj_triangles(
                    primary_bone_obj, primary_bone_member,
                )
                primary_bone_vertices_world_m = [
                    [
                        sum(
                            global_matrix[row][column] * vertex[column]
                            for column in range(3)
                        ) + global_matrix[row][3]
                        for row in range(3)
                    ]
                    for vertex in primary_bone_vertices_mm
                ]
                primary_weights, primary_attachment_weight_lock = (
                    _bodyparts_primary_bone_attachment_weights(
                        global_vertices,
                        primary_bone_vertices_world_m,
                        float(lock_radius_m),
                        float(feather_radius_m),
                    )
                )
                require_inferior_secondary_ownership = visual_binding.get(
                    "require_inferior_secondary_ownership", False
                )
                if not isinstance(require_inferior_secondary_ownership, bool):
                    raise ImportError(
                        f"BodyParts3D full-body tissue surface {member_id} has an invalid inferior-origin gate"
                    )
                if require_inferior_secondary_ownership:
                    vertical_values = sorted(vertex[2] for vertex in global_vertices)
                    inferior_threshold_m = vertical_values[
                        max(0, len(vertical_values) // 10 - 1)
                    ]
                    inferior_indices = [
                        index for index, vertex in enumerate(global_vertices)
                        if vertex[2] <= inferior_threshold_m
                    ]
                    inferior_maximum_primary_weight = max(
                        primary_weights[index] for index in inferior_indices
                    )
                    if inferior_maximum_primary_weight > 1.0e-8:
                        raise ImportError(
                            f"BodyParts3D pectoralis surface {member_id} leaves its inferior origin on the humerus"
                        )
                    primary_attachment_weight_lock.update({
                        "inferior_origin_gate": "lowest_source_world_z_decile_is_secondary_body_owned",
                        "inferior_origin_vertex_count": len(inferior_indices),
                        "inferior_origin_threshold_m": inferior_threshold_m,
                        "inferior_origin_maximum_primary_weight": (
                            inferior_maximum_primary_weight
                        ),
                    })
                vertex_weights = [
                    [primary_weight, 1.0 - primary_weight]
                    for primary_weight in primary_weights
                ]
                primary_attachment_weight_lock.update({
                    "primary_body": primary_name,
                    "secondary_body": secondary_name,
                    "primary_bone_member_id": primary_bone_member_id,
                    "primary_bone_member": primary_bone_member,
                    "primary_bone_member_sha256": hashlib.sha256(
                        primary_bone_obj
                    ).hexdigest(),
                    "source_endpoint_migration_m": 0.0,
                })
                route_binding_diagnostics = {
                    "method": "primary_source_bone_attachment_band",
                    "binding_body_count": 2,
                    "maximum_vertex_influences": 2,
                    "primary_locked_vertex_count": (
                        primary_attachment_weight_lock[
                            "primary_locked_vertex_count"
                        ]
                    ),
                    "primary_feathered_vertex_count": (
                        primary_attachment_weight_lock[
                            "primary_feathered_vertex_count"
                        ]
                    ),
                    "secondary_owned_vertex_count": (
                        primary_attachment_weight_lock[
                            "secondary_owned_vertex_count"
                        ]
                    ),
                }
            elif len(binding_names) == 2 and len(route_pairs) == 1:
                vertex_weights = [
                    [primary_weight, 1.0 - primary_weight]
                    for primary_weight in base_primary_weights
                ]
                route_binding_diagnostics = {
                    "method": "two_authored_route_endpoint_body_axis_linear_blend",
                    "binding_body_count": 2,
                    "maximum_vertex_influences": 2,
                }
            else:
                binding_index = {name: index for index, name in enumerate(binding_names)}
                route_points = [
                    point
                    for route in matched_routes
                    for point in route["route_points"]
                ]
                if not route_points:
                    raise ImportError(
                        f"BodyParts3D full-body tissue surface {member_id} has no route points"
                    )
                vertex_weights = []
                maximum_influences = 0
                maximum_nearest_route_distance = 0.0
                for vertex in global_vertices:
                    squared_by_binding = [math.inf] * len(binding_names)
                    for point in route_points:
                        index = binding_index.get(point["body"])
                        if index is None:
                            raise ImportError(
                                f"BodyParts3D tissue {member_id} route point escapes its binding table"
                            )
                        squared = sum(
                            (vertex[axis] - point["world_m"][axis]) ** 2
                            for axis in range(3)
                        )
                        squared_by_binding[index] = min(squared_by_binding[index], squared)
                    nearest = sorted(
                        (squared, index)
                        for index, squared in enumerate(squared_by_binding)
                        if math.isfinite(squared)
                    )[:_BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_MAX_INFLUENCES]
                    if not nearest:
                        raise ImportError(
                            f"BodyParts3D tissue {member_id} vertex has no route-body influence"
                        )
                    maximum_nearest_route_distance = max(
                        maximum_nearest_route_distance, math.sqrt(nearest[0][0])
                    )
                    weights = [0.0] * len(binding_names)
                    if nearest[0][0] <= 1.0e-12:
                        weights[nearest[0][1]] = 1.0
                        active_influences = 1
                    else:
                        # A 3 mm softening radius avoids singular weights at a
                        # route point while retaining local digital-slip
                        # ownership. Four influences provide smooth joint
                        # transitions without letting a distant finger drag a
                        # neighboring tendon sheet.
                        raw = [1.0 / (squared + 9.0e-6) for squared, _ in nearest]
                        total = sum(raw)
                        if not math.isfinite(total) or total <= 0.0:
                            raise ImportError(
                                f"BodyParts3D tissue {member_id} route weights are non-finite"
                            )
                        for value, (_, index) in zip(raw, nearest, strict=True):
                            weights[index] = value / total
                        active_influences = len(nearest)
                    maximum_influences = max(maximum_influences, active_influences)
                    vertex_weights.append(weights)
                route_binding_diagnostics = {
                    "method": "nearest_exact_myosim_route_nodes_inverse_squared_four_influence",
                    "binding_body_count": len(binding_names),
                    "maximum_vertex_influences": maximum_influences,
                    "maximum_nearest_route_distance_m": maximum_nearest_route_distance,
                    "matched_route_count": len(matched_routes),
                }

            # The lower-body source has one EDL/FDL route for four anatomical
            # slips. Route-node proximity alone therefore leaves the distal
            # BodyParts3D slips partially following calcaneus/tibia and can
            # make one terminal branch appear attached to the adjacent toe.
            # Lock only source vertices close to the explicitly named distal
            # phalanges into the toes-body frame. This changes presentation
            # weights, not the MyoSim path or force law; the matching
            # NHTENDON2 semantic envelope below preserves the exact source
            # endpoint wrench across the same named bones.
            semantic_members = (
                _NUMI_HUMAN_TOE_ENTHESIS_MEMBERS.get((source_muscles[0], 1))
                if len(source_muscles) == 1 else None
            )
            if semantic_members is not None:
                if secondary_name not in {"toes_r", "toes_l"} or secondary_name not in binding_names:
                    raise ImportError(
                        f"BodyParts3D toe enthesis surface {member_id} does not terminate on a toes body"
                    )
                sources_by_member = {
                    record.get("member_id"): record
                    for record in body_bone_sources.get(secondary_name, [])
                }
                if any(member not in sources_by_member for member in semantic_members):
                    raise ImportError(
                        f"BodyParts3D toe enthesis surface {member_id} is missing a named distal phalanx"
                    )
                secondary_binding_index = binding_names.index(secondary_name)
                secondary_target = binding_targets[secondary_binding_index]
                secondary_local_pose = binding_local_poses[secondary_binding_index]
                rigid_chains = _NUMI_HUMAN_TOE_RIGID_CHAINS.get(secondary_name)
                expected_semantic_members = (
                    (rigid_chains[0][-1],) if len(semantic_members) == 1
                    else tuple(chain[-1] for chain in rigid_chains[1:])
                ) if rigid_chains is not None else ()
                selected_rigid_chains = (
                    rigid_chains[:1] if len(semantic_members) == 1 else rigid_chains[1:]
                ) if rigid_chains is not None else ()
                if (
                    semantic_members != expected_semantic_members
                    or any(
                        member not in sources_by_member
                        for chain in selected_rigid_chains
                        for member in chain
                    )
                ):
                    raise ImportError(
                        f"BodyParts3D digital surface {member_id} escapes its single-body toe compounds"
                    )
                toe_rigid_compounds = {
                    "myosim_body": secondary_name,
                    "core_body_index": secondary_target["core_body_index"],
                    "digits": [1] if len(semantic_members) == 1 else [2, 3, 4, 5],
                    "source_member_chains": [list(chain) for chain in selected_rigid_chains],
                    "distal_enthesis_member_ids": list(semantic_members),
                    "independent_articulation_count": 0,
                    "binding": "bone chains and terminal visual patch share one existing toes-body transform",
                }
                hallux_rigid_compound: dict[str, Any] | None = None
                if len(semantic_members) == 1:
                    hallux_rigid_compound = {
                        "myosim_body": secondary_name,
                        "core_body_index": secondary_target["core_body_index"],
                        "source_member_ids": list(selected_rigid_chains[0]),
                        "distal_enthesis_member_id": semantic_members[0],
                        "independent_articulation_count": 0,
                        "binding": "bone chain and terminal visual patch share one existing toes-body transform",
                    }
                secondary_position = _myosim_vector(
                    secondary_target.get("default_com_position_world_m"),
                    f"BodyParts3D {member_id} toe enthesis position",
                )
                secondary_quaternion = list(
                    secondary_target.get("default_inertial_quaternion_world_xyzw", [])
                )
                tissue_in_secondary_world = _bodyparts_source_mm_to_body_world(
                    vertices_mm, secondary_position, secondary_quaternion,
                    *secondary_local_pose,
                )
                enthesis_vertices_world: list[list[float]] = []
                enthesis_triangles: list[tuple[int, int, int]] = []
                for semantic_member in semantic_members:
                    source_record = sources_by_member[semantic_member]
                    _, bone_member, bone_obj = _bodyparts_obj_member(
                        sources, source_record["hierarchy"], semantic_member,
                    )
                    bone_vertices_mm, bone_triangles = _bodyparts_obj_triangles(
                        bone_obj, bone_member,
                    )
                    bone_vertices_world = _bodyparts_source_mm_to_body_world(
                        bone_vertices_mm, secondary_position, secondary_quaternion,
                        *secondary_local_pose,
                    )
                    first_bone_vertex = len(enthesis_vertices_world)
                    enthesis_vertices_world.extend(bone_vertices_world)
                    enthesis_triangles.extend(
                        tuple(first_bone_vertex + index for index in triangle)
                        for triangle in bone_triangles
                    )
                proximity_attenuation, toe_enthesis_weight_lock = \
                    _bodyparts_secondary_attachment_weight_lock(
                        tissue_in_secondary_world,
                        [1.0] * len(tissue_in_secondary_world),
                        enthesis_vertices_world,
                        enthesis_triangles,
                        _NUMI_HUMAN_TOE_VISUAL_LOCK_RADIUS_M,
                        _NUMI_HUMAN_TOE_VISUAL_FEATHER_RADIUS_M,
                    )
                distal_attenuation = list(proximity_attenuation)
                source_longitudinal_mm = [vertex[1] for vertex in vertices_mm]
                source_longitudinal_minimum_mm = min(source_longitudinal_mm)
                source_longitudinal_extent_mm = (
                    max(source_longitudinal_mm) - source_longitudinal_minimum_mm
                )
                if source_longitudinal_extent_mm <= 1.0e-6:
                    raise ImportError(
                        f"BodyParts3D toe enthesis surface {member_id} has no longitudinal extent"
                    )
                longitudinal_lock_mm = source_longitudinal_minimum_mm + (
                    _NUMI_HUMAN_TOE_VISUAL_DISTAL_LOCK_FRACTION
                    * source_longitudinal_extent_mm
                )
                longitudinal_feather_mm = source_longitudinal_minimum_mm + (
                    _NUMI_HUMAN_TOE_VISUAL_DISTAL_FEATHER_FRACTION
                    * source_longitudinal_extent_mm
                )
                longitudinal_locked = 0
                longitudinal_feathered = 0
                for vertex_index, source_y_mm in enumerate(source_longitudinal_mm):
                    if source_y_mm <= longitudinal_lock_mm:
                        longitudinal_attenuation = 0.0
                        longitudinal_locked += 1
                    elif source_y_mm < longitudinal_feather_mm:
                        longitudinal_attenuation = (
                            (source_y_mm - longitudinal_lock_mm)
                            / (longitudinal_feather_mm - longitudinal_lock_mm)
                        )
                        longitudinal_feathered += 1
                    else:
                        longitudinal_attenuation = 1.0
                    distal_attenuation[vertex_index] = min(
                        distal_attenuation[vertex_index], longitudinal_attenuation,
                    )
                toe_binding_index = binding_names.index(secondary_name)
                for weights, attenuation in zip(
                    vertex_weights, distal_attenuation, strict=True,
                ):
                    for binding_index in range(len(weights)):
                        if binding_index != toe_binding_index:
                            weights[binding_index] *= attenuation
                    weights[toe_binding_index] = (
                        1.0 - attenuation + attenuation * weights[toe_binding_index]
                    )
                    if abs(sum(weights) - 1.0) > 1.0e-6:
                        raise ImportError(
                            f"BodyParts3D toe enthesis surface {member_id} has non-unit locked weights"
                        )
                visual_enthesis_registration: dict[str, Any] | None = None
                if len(semantic_members) == 1:
                    source_gap_m = toe_enthesis_weight_lock["nearest_vertex_distance_m"]
                    if source_gap_m > _NUMI_HUMAN_HALLUX_VISUAL_ENTHESIS_MINIMUM_GAP_M:
                        projected_world, projection = _bodyparts_project_tendon_attachment_band(
                            tissue_in_secondary_world,
                            proximity_attenuation,
                            enthesis_vertices_world,
                            enthesis_triangles,
                            _NUMI_HUMAN_HALLUX_VISUAL_ENTHESIS_INSET_M,
                        )
                        stored_vertices_m = _bodyparts_world_to_body_stored_m(
                            projected_world,
                            secondary_position,
                            secondary_quaternion,
                            *secondary_local_pose,
                            f"BodyParts3D hallucis {member_id} visual enthesis registration",
                        )
                        stored_normals = _bodyparts_vertex_normals(
                            [
                                [coordinate * 1000.0 for coordinate in vertex]
                                for vertex in stored_vertices_m
                            ],
                            triangles,
                            member,
                        )
                        visual_enthesis_registration = {
                            "status": "source_gap_closed_against_exact_named_distal_phalanx",
                            "source_gap_m": source_gap_m,
                            "minimum_gap_requiring_projection_m": (
                                _NUMI_HUMAN_HALLUX_VISUAL_ENTHESIS_MINIMUM_GAP_M
                            ),
                            "projection": projection,
                            "source_endpoint_migration_m": 0.0,
                        }
                    else:
                        visual_enthesis_registration = {
                            "status": "exact_source_surface_already_contacts_named_distal_phalanx",
                            "source_gap_m": source_gap_m,
                            "minimum_gap_requiring_projection_m": (
                                _NUMI_HUMAN_HALLUX_VISUAL_ENTHESIS_MINIMUM_GAP_M
                            ),
                            "source_endpoint_migration_m": 0.0,
                        }
                toe_enthesis_weight_lock.update({
                    "method": (
                        "exact_source_triangle_proximity_to_semantically_named_"
                        "distal_phalanx_union_plus_source_longitudinal_terminal_band"
                    ),
                    "myosim_muscle": source_muscles[0],
                    "secondary_body": secondary_name,
                    "distal_phalanx_member_ids": list(semantic_members),
                    "source_endpoint_migration_m": 0.0,
                    "source_longitudinal_band": {
                        "axis": "BodyParts3D source Y; decreasing is distal for these exact bilateral surfaces",
                        "distal_lock_fraction": _NUMI_HUMAN_TOE_VISUAL_DISTAL_LOCK_FRACTION,
                        "distal_feather_fraction": _NUMI_HUMAN_TOE_VISUAL_DISTAL_FEATHER_FRACTION,
                        "lock_boundary_mm": longitudinal_lock_mm,
                        "feather_boundary_mm": longitudinal_feather_mm,
                        "locked_vertex_count": longitudinal_locked,
                        "feathered_vertex_count": longitudinal_feathered,
                        "combined_locked_vertex_count": sum(
                            attenuation <= 1.0e-8 for attenuation in distal_attenuation
                        ),
                        "combined_feathered_vertex_count": sum(
                            1.0e-8 < attenuation < 1.0 - 1.0e-8
                            for attenuation in distal_attenuation
                        ),
                    },
                    "boundary": (
                        "This locks only the visual BodyParts3D terminal slip to its named "
                        "toe-bone frame. It does not alter the authored MyoSim route, create "
                        "independent toe actuators, or establish clinical enthesis geometry."
                    ),
                })
                if visual_enthesis_registration is not None:
                    toe_enthesis_weight_lock["visual_enthesis_registration"] = (
                        visual_enthesis_registration
                    )
                if hallux_rigid_compound is not None:
                    toe_enthesis_weight_lock["hallux_rigid_compound"] = (
                        hallux_rigid_compound
                    )
                toe_enthesis_weight_lock["toe_rigid_compounds"] = toe_rigid_compounds

        first_binding = len(bindings_payload)
        for target, transform in zip(
            binding_targets,
            [binding_transforms[index:index + 8] for index in range(0, len(binding_transforms), 8)],
            strict=True,
        ):
            bindings_payload.append(struct.pack(
                "<I8f", target["core_body_index"], *transform
            ))
        first_vertex, first_index = len(vertices_payload), len(indices_payload)
        for vertex, normal, weights in zip(stored_vertices_m, stored_normals, vertex_weights, strict=True):
            active = sorted(
                ((weight, index) for index, weight in enumerate(weights) if weight > 1.0e-8),
                reverse=True,
            )[:_BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_MAX_INFLUENCES]
            active_total = sum(weight for weight, _ in active)
            if not active or not math.isfinite(active_total) or active_total <= 0.0:
                raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has invalid body weights")
            influence_indices = [index for _, index in active]
            influence_weights = [weight / active_total for weight, _ in active]
            influence_indices.extend(
                [0xFFFFFFFF] * (
                    _BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_MAX_INFLUENCES - len(influence_indices)
                )
            )
            influence_weights.extend(
                [0.0] * (
                    _BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_MAX_INFLUENCES - len(influence_weights)
                )
            )
            if any(
                not math.isfinite(weight) or weight < 0.0 or weight > 1.0
                for weight in influence_weights
            ) or abs(sum(influence_weights) - 1.0) > 1.0e-6:
                raise ImportError(
                    f"BodyParts3D full-body tissue surface {member_id} has invalid sparse weights"
                )
            vertices_payload.append(struct.pack(
                "<6f4I4f", *vertex, *normal, *influence_indices, *influence_weights
            ))
        indices_payload.extend(first_vertex + index for triangle in triangles for index in triangle)
        records_payload.append(struct.pack(
            "<8I", first_binding, len(binding_names), first_vertex,
            len(stored_vertices_m), first_index, len(triangles) * 3,
            stable_id, layer,
        ))
        provenance.append({
            "stable_id": stable_id, "member_id": member_id, "member": member,
            "member_sha256": hashlib.sha256(obj).hexdigest(),
            "label": label, "layer": layer_name, "endpoint_source": endpoint_source,
            "body_bindings": [
                {"myosim_body": name, "core_body_index": target["core_body_index"]}
                for name, target in zip(binding_names, binding_targets[:len(binding_names)], strict=True)
            ],
            "matched_muscles": [
                {
                    "name": name,
                    "source_actuator_index": route["source_actuator_index"],
                    "source_route_node_count": route["source_route_node_count"],
                    "primary_body": route["primary_body"],
                    "secondary_body": route["secondary_body"],
                    "binding_bodies": route["binding_bodies"],
                }
                for name, route in zip(source_muscles, matched_routes, strict=True)
            ],
            "body_weight_count": len(binding_names), "vertex_count": len(stored_vertices_m), "triangle_count": len(triangles),
        })
        if layer == _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE:
            provenance[-1]["route_binding"] = route_binding_diagnostics
        if attachment_weight_lock is not None:
            provenance[-1]["secondary_attachment_weight_lock"] = attachment_weight_lock
        if primary_attachment_weight_lock is not None:
            provenance[-1]["primary_attachment_weight_lock"] = (
                primary_attachment_weight_lock
            )
        if toe_enthesis_weight_lock is not None:
            provenance[-1]["toe_enthesis_weight_lock"] = toe_enthesis_weight_lock
        if source_component_selection is not None:
            provenance[-1]["source_component_selection"] = source_component_selection
        if layer == _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE:
            source_surface_bindings[member_id] = {
                "member_id": member_id,
                "myosim_muscles": list(source_muscles),
                "binding_names": binding_names,
                "global_vertices": global_vertices,
                "vertex_weights": vertex_weights,
            }
    if len(vertices_payload) > 0xFFFFFFFF or len(indices_payload) > 0xFFFFFFFF:
        raise ImportError("BodyParts3D full-body tissue payload exceeds the uint32 native renderer capacity")
    registration_fingerprint = _bodyparts_visual_registration_fingerprint(registration_file)
    payload = b"".join([
        struct.pack("<8s6I32s", _BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_VISUAL_MAGIC,
                    _BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_VISUAL_ABI,
                    len(records_payload), len(bindings_payload), len(vertices_payload),
                    len(indices_payload), registration_fingerprint,
                    bytes.fromhex(source_sha)),
        *records_payload, *bindings_payload, *vertices_payload,
        struct.pack(f"<{len(indices_payload)}I", *indices_payload),
    ])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
        "payload": {"file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
                    "magic": _BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_VISUAL_MAGIC.rstrip(b"\0").decode("ascii"),
                    "payload_abi": _BODYPARTS_MYOSIM_ROUTE_SOFT_TISSUE_VISUAL_ABI,
                    "registration_fingerprint32": f"{registration_fingerprint:08x}",
                    "surface_count": len(records_payload),
                    "binding_count": len(bindings_payload),
                    "vertex_count": len(vertices_payload), "index_count": len(indices_payload)},
        "source": {"registration": {"file": registration_file.name, "sha256": sha256(registration_file)},
                   "bodyparts": expected_bodyparts, "myosim_source_archive_sha256": source_sha,
                   "myosim_manifest": {"file": "myosim-fullbody-reference.manifest.json", "sha256": sha256((myosim_artifact.resolve() / "myosim-fullbody-reference.manifest.json"))},
                   "surface_map": {"file": "bodyparts3d-myosim-surface-map.v1.json", "sha256": sha256(REPOSITORY_ROOT / "config/bodyparts3d-myosim-surface-map.v1.json")},
                   "surfaces": provenance},
        "coverage": {"configured_surface_count": len(specifications), "emitted_surface_count": len(provenance),
                     "selected_stable_ids": sorted(stable_id_subset) if stable_id_subset is not None else None,
                     "muscle_surface_count": sum(1 for entry in provenance if entry["layer"] == "muscle"),
                     "tendon_surface_count": sum(1 for entry in provenance if entry["layer"] == "tendon"),
                     "authored_myosim_muscle_count": len(myosim_manifest["muscles"])},
        "runtime_binding": "BodyParts3D source-topology surfaces use a variable exact MyoSim route-body table with four sparse influences per vertex; shared digital surfaces include every authored digit route instead of a middle-finger proxy, while each calcaneal-tendon surface inherits femur/tibia/calcaneus weights from its nearest named source muscle surface, registers its locked/feathered distal boundary to exact named calcaneal source triangles, and adds a separately labelled visual enthesis strip at the opened source cap",
        "status": "native_route_body_sparse_kinematic_surface_binding_input_not_collision_or_physics",
        "evidence_boundary": "This source-authored surface package visually follows exact named articulated endpoint bodies. It does not make the source surface a force-transmitting continuum, add a tendon constitutive law, create collision/contact, or establish a medical registration.",
    }
    write_json(output / "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json", manifest)
    return manifest


def _bodyparts_source_element_relation_names(sources: Path, hierarchy: str) -> set[tuple[str, str, str]]:
    """Read one BodyParts3D FMA-to-element table for exact map validation."""
    if hierarchy == "part_of":
        source = sources / "partof_element_parts.txt"
    elif hierarchy == "is_a":
        source = sources / "isa_element_parts.txt"
    else:
        raise ImportError(f"BodyParts3D source hierarchy is unsupported: {hierarchy}")
    if not source.is_file():
        raise ImportError(f"BodyParts3D element relation source is unavailable: {source.name}")
    result: set[tuple[str, str, str]] = set()
    for raw in source.read_text(encoding="utf-8").splitlines():
        columns = raw.split("\t")
        if len(columns) != 3:
            raise ImportError(f"BodyParts3D element relation source has an invalid row: {source.name}")
        result.add((columns[0], columns[1], columns[2]))
    return result


def _bodyparts_unit_vector(value: list[float], context: str) -> list[float]:
    if len(value) != 3 or not all(math.isfinite(component) for component in value):
        raise ImportError(f"{context} is not a finite 3-vector")
    magnitude = math.sqrt(sum(component * component for component in value))
    if magnitude <= 1.0e-12:
        raise ImportError(f"{context} has zero length")
    return [component / magnitude for component in value]


def bodyparts_myosim_torso_anatomy_visual_payload(
    sources: Path, anatomy: dict[str, Any], registration_path: Path, myosim_artifact: Path, output: Path,
) -> dict[str, Any]:
    """Package selected exact organs, vessels, and spinal cord for native posing.

    The BodyParts3D surfaces remain their supplied triangle topology.  Each
    selected component is converted into the named MyoSim torso or abdomen
    inertial frame at the registered default pose, so the Metal visual runtime
    moves it with that articulated link.  This is intentionally a compact
    anatomical inspection layer, never a material or continuum lowerer.
    """
    registration_file = registration_path.resolve()
    registration = read_json(registration_file)
    if registration.get("schema") not in {
        "numi.human.bodyparts3d-myosim-bone-registration-candidate.v1",
        "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2",
    } or registration.get("status") not in {
        "provisional_visual_registration_not_admitted_to_collision_or_physics",
        "inferred_attachment_surface_visual_registration_not_admitted_to_collision_or_physics",
    }:
        raise ImportError("BodyParts3D torso anatomy payload requires an unmodified visual registration")
    expected_bodyparts = {
        "id": anatomy.get("source_id"), "version": anatomy.get("version"),
        "archives": anatomy.get("archives"),
    }
    source = registration.get("source")
    if not isinstance(source, dict) or source.get("bodyparts") != expected_bodyparts:
        raise ImportError("BodyParts3D torso anatomy registration does not match parsed source provenance")
    myosim = source.get("myosim")
    source_sha = myosim.get("source", {}).get("archive_sha256") if isinstance(myosim, dict) else None
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ImportError("BodyParts3D torso anatomy payload has no MyoSim source SHA-256")
    coordinates = registration.get("coordinate_system")
    global_matrix = coordinates.get("global_source_mm_to_myosim_world_m") if isinstance(coordinates, dict) else None
    global_translation, global_quaternion, global_scale = _bodyparts_visual_local_pose(
        global_matrix, "BodyParts3D torso anatomy global transform",
    )
    bodies, _, myosim_manifest = _myosim_surface_route_context(myosim_artifact, source_sha)
    map_path = REPOSITORY_ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json"
    surface_map = read_json(map_path)
    if surface_map.get("schema") != "numi.human.bodyparts3d-myosim-torso-anatomy-map.v1":
        raise ImportError("BodyParts3D torso anatomy surface map schema is unsupported")
    specifications = surface_map.get("entries")
    if not isinstance(specifications, list) or not specifications:
        raise ImportError("BodyParts3D torso anatomy surface map has no entries")
    relation_cache = {
        hierarchy: _bodyparts_source_element_relation_names(sources, hierarchy)
        for hierarchy in {entry.get("hierarchy") for entry in specifications if isinstance(entry, dict)}
        if isinstance(hierarchy, str)
    }
    if not relation_cache:
        raise ImportError("BodyParts3D torso anatomy surface map has no source hierarchies")
    layer_codes = {
        "organ": _BODYPARTS_MYOSIM_TORSO_ANATOMY_LAYER_ORGAN,
        "vessel": _BODYPARTS_MYOSIM_TORSO_ANATOMY_LAYER_VESSEL,
        "nerve": _BODYPARTS_MYOSIM_TORSO_ANATOMY_LAYER_NERVE,
    }
    global_rotation = _myosim_matrix_from_quaternion_xyzw(global_quaternion)
    vertices_payload: list[tuple[float, float, float, float, float, float]] = []
    indices_payload: list[int] = []
    records_payload: list[bytes] = []
    provenance: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    for stable_id, specification in enumerate(specifications, start=1):
        if not isinstance(specification, dict):
            raise ImportError("BodyParts3D torso anatomy surface map has an invalid entry")
        concept_id = specification.get("concept_id")
        label = specification.get("source_name")
        member_id = specification.get("member_id")
        hierarchy = specification.get("hierarchy")
        layer_name = specification.get("layer")
        body_name = specification.get("myosim_body")
        if not all(isinstance(value, str) and value for value in (
            concept_id, label, member_id, hierarchy, layer_name, body_name,
        )) or member_id in seen_members or hierarchy not in relation_cache or layer_name not in layer_codes:
            raise ImportError("BodyParts3D torso anatomy surface map identity is invalid")
        seen_members.add(member_id)
        if (concept_id, label, member_id) not in relation_cache[hierarchy]:
            raise ImportError(f"BodyParts3D torso anatomy source relation drifted: {member_id}")
        target = bodies.get(body_name)
        if not isinstance(target, dict):
            raise ImportError(f"BodyParts3D torso anatomy has no named MyoSim body: {body_name}")
        body_index = target.get("core_body_index")
        position = target.get("default_com_position_world_m")
        quaternion = target.get("default_inertial_quaternion_world_xyzw")
        if not isinstance(body_index, int) or body_index < 0 or not isinstance(position, list) or not isinstance(quaternion, list):
            raise ImportError(f"BodyParts3D torso anatomy target body is malformed: {body_name}")
        archive_path, member, obj = _bodyparts_obj_member(sources, hierarchy, member_id)
        vertices_mm, triangles = _bodyparts_obj_triangles(obj, member)
        normals = _bodyparts_vertex_normals(vertices_mm, triangles, member)
        world_vertices = _bodyparts_source_mm_to_body_world(
            vertices_mm, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            global_translation, global_quaternion, global_scale,
        )
        stored_vertices = _bodyparts_world_to_body_stored_m(
            world_vertices, position, quaternion, [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0], 1.0, f"BodyParts3D torso anatomy {member_id}",
        )
        body_world_rotation = _myosim_matrix_from_quaternion_xyzw(quaternion)
        body_world_rotation_inverse = _matrix_transpose(body_world_rotation)
        stored_normals = []
        for normal in normals:
            world_normal = _bodyparts_unit_vector(
                _myosim_matrix_vector(global_rotation, list(normal)),
                f"BodyParts3D torso anatomy {member_id} world normal",
            )
            stored_normals.append(_bodyparts_unit_vector(
                _myosim_matrix_vector(body_world_rotation_inverse, world_normal),
                f"BodyParts3D torso anatomy {member_id} local normal",
            ))
        first_vertex, first_index = len(vertices_payload), len(indices_payload)
        vertices_payload.extend(
            (*vertex, *normal) for vertex, normal in zip(stored_vertices, stored_normals, strict=True)
        )
        indices_payload.extend(first_vertex + index for triangle in triangles for index in triangle)
        records_payload.append(struct.pack(
            "<8I", body_index, first_vertex, len(vertices_mm), first_index,
            len(triangles) * 3, stable_id, layer_codes[layer_name], 0,
        ))
        provenance.append({
            "stable_id": stable_id, "concept_id": concept_id, "label": label,
            "member_id": member_id, "member": member,
            "member_sha256": hashlib.sha256(obj).hexdigest(),
            "hierarchy": hierarchy, "layer": layer_name,
            "myosim_body": body_name, "core_body_index": body_index,
            "vertex_count": len(vertices_mm), "triangle_count": len(triangles),
        })
    if len(vertices_payload) > 0xFFFFFFFF or len(indices_payload) > 0xFFFFFFFF:
        raise ImportError("BodyParts3D torso anatomy payload exceeds the uint32 native renderer capacity")
    registration_fingerprint = _bodyparts_visual_registration_fingerprint(registration_file)
    payload = b"".join([
        struct.pack(
            "<8s5I32s", _BODYPARTS_MYOSIM_TORSO_ANATOMY_VISUAL_MAGIC,
            _BODYPARTS_MYOSIM_TORSO_ANATOMY_VISUAL_ABI, len(records_payload),
            len(vertices_payload), len(indices_payload), registration_fingerprint,
            bytes.fromhex(source_sha),
        ),
        *records_payload,
        b"".join(struct.pack("<6f", *vertex) for vertex in vertices_payload),
        struct.pack(f"<{len(indices_payload)}I", *indices_payload),
    ])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "bodyparts3d-myosim-torso-anatomy.nhanatomy"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-torso-anatomy-visual-payload.v1",
        "payload": {
            "file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
            "magic": _BODYPARTS_MYOSIM_TORSO_ANATOMY_VISUAL_MAGIC.rstrip(b"\0").decode("ascii"),
            "payload_abi": _BODYPARTS_MYOSIM_TORSO_ANATOMY_VISUAL_ABI,
            "registration_fingerprint32": f"{registration_fingerprint:08x}",
            "surface_count": len(records_payload), "vertex_count": len(vertices_payload),
            "index_count": len(indices_payload),
        },
        "source": {
            "registration": {"file": registration_file.name, "sha256": sha256(registration_file)},
            "bodyparts": expected_bodyparts, "myosim_source_archive_sha256": source_sha,
            "myosim_manifest": {
                "file": "myosim-fullbody-reference.manifest.json",
                "sha256": sha256((myosim_artifact.resolve() / "myosim-fullbody-reference.manifest.json")),
            },
            "surface_map": {"file": map_path.name, "sha256": sha256(map_path)},
            "surfaces": provenance,
        },
        "coverage": {
            "configured_surface_count": len(provenance),
            "organ_surface_count": sum(entry["layer"] == "organ" for entry in provenance),
            "vessel_surface_count": sum(entry["layer"] == "vessel" for entry in provenance),
            "nerve_surface_count": sum(entry["layer"] == "nerve" for entry in provenance),
        },
        "runtime_binding": "each exact BodyParts3D source component is converted into the declared MyoSim torso or abdomen inertial frame at the registered default pose and then follows that one articulated visual link in the native renderer",
        "status": "native_single_link_kinematic_anatomy_surface_binding_input_not_collision_or_physics",
        "evidence_boundary": "This compact source-surface layer does not create organ FEM or MPM bodies, vessel tube mechanics, neural mechanics, tissue material parameters, collision/contact, force transmission, or medical registration.",
    }
    write_json(output / "bodyparts3d-myosim-torso-anatomy.manifest.json", manifest)
    return manifest


def _bodyparts_skin_bbox_distance_squared(
    point: list[float], minimum: list[float], maximum: list[float],
) -> float:
    """Squared distance from a world-space point to an articulated bone AABB."""
    return sum(
        (minimum[axis] - point[axis]) ** 2 if point[axis] < minimum[axis]
        else (point[axis] - maximum[axis]) ** 2 if point[axis] > maximum[axis]
        else 0.0
        for axis in range(3)
    )


def _bodyparts_skin_bbox_surface_distance_squared(
    point: list[float], minimum: list[float], maximum: list[float],
) -> float:
    """Squared distance to an articulated bone envelope's *boundary*.

    An ordinary AABB distance is zero throughout its enclosed volume.  That is
    appropriate for broad-phase collision, but not for exterior skinning: a
    torso vertex can be inside several overlapping axial envelopes and would
    otherwise receive arbitrary equal influences.  This keeps the ordinary
    outside metric while, for an interior point, measuring the nearest face of
    the envelope.  It is still a deterministic offline visual-binding proxy,
    not a replacement for a closest triangle or anatomical skin-weight
    dataset.
    """
    outside_squared = _bodyparts_skin_bbox_distance_squared(point, minimum, maximum)
    if outside_squared > 0.0:
        return outside_squared
    if len(point) != 3 or len(minimum) != 3 or len(maximum) != 3 or any(
            not math.isfinite(value) for value in (*point, *minimum, *maximum)) or any(
            minimum[axis] > maximum[axis] for axis in range(3)
    ):
        raise ImportError("BodyParts3D skin envelope boundary has invalid bounds")
    return min(
        min(
            (point[axis] - minimum[axis]) ** 2,
            (maximum[axis] - point[axis]) ** 2,
        )
        for axis in range(3)
    )


@dataclass(frozen=True)
class _BodyPartsSkinSurfaceNode:
    """One deterministic source-bone surface sample in a balanced 3-D tree."""

    point: tuple[float, float, float]
    binding_index: int
    axis: int
    left: _BodyPartsSkinSurfaceNode | None = None
    right: _BodyPartsSkinSurfaceNode | None = None


def _bodyparts_skin_surface_sample_indices(vertex_count: int, maximum_samples: int = 64) -> list[int]:
    """Uniformly subsample an exact source mesh without randomisation."""
    if vertex_count <= 0 or maximum_samples <= 0:
        raise ImportError("BodyParts3D skin surface sampler has an invalid mesh extent")
    sample_count = min(vertex_count, maximum_samples)
    if sample_count == vertex_count:
        return list(range(vertex_count))
    return [index * (vertex_count - 1) // (sample_count - 1) for index in range(sample_count)]


def _bodyparts_skin_surface_index(
    samples: list[tuple[tuple[float, float, float], int]],
) -> _BodyPartsSkinSurfaceNode:
    """Build a balanced nearest-source-vertex index for registered bone meshes."""
    if not samples:
        raise ImportError("BodyParts3D skin surface index has no bone samples")
    if any(
        len(point) != 3 or not isinstance(binding_index, int) or binding_index < 0 or
        not all(math.isfinite(value) for value in point)
        for point, binding_index in samples
    ):
        raise ImportError("BodyParts3D skin surface index has an invalid bone sample")

    def build(
        candidates: list[tuple[tuple[float, float, float], int]], depth: int,
    ) -> _BodyPartsSkinSurfaceNode | None:
        if not candidates:
            return None
        axis = depth % 3
        candidates.sort(key=lambda value: (value[0][axis], value[0], value[1]))
        middle = len(candidates) // 2
        point, binding_index = candidates[middle]
        return _BodyPartsSkinSurfaceNode(
            point, binding_index, axis,
            build(candidates[:middle], depth + 1),
            build(candidates[middle + 1:], depth + 1),
        )

    result = build(list(samples), 0)
    if result is None:
        raise ImportError("BodyParts3D skin surface index did not build")
    return result


def _bodyparts_skin_nearest_surface_bindings(
    index: _BodyPartsSkinSurfaceNode,
    point: list[float],
    count: int = 4,
) -> list[tuple[float, int]]:
    """Return nearest distinct registered bodies from exact source-bone samples."""
    if len(point) != 3 or count <= 0 or not all(math.isfinite(value) for value in point):
        raise ImportError("BodyParts3D skin surface query has invalid input")
    nearest_by_binding: dict[int, float] = {}

    def worst_distance_squared() -> float:
        return (
            max(nearest_by_binding.values())
            if len(nearest_by_binding) >= count
            else math.inf
        )

    def visit(node: _BodyPartsSkinSurfaceNode | None) -> None:
        if node is None:
            return
        squared = sum((point[axis] - node.point[axis]) ** 2 for axis in range(3))
        prior = nearest_by_binding.get(node.binding_index)
        if prior is None or squared < prior:
            nearest_by_binding[node.binding_index] = squared
        if len(nearest_by_binding) > count:
            worst_binding = max(nearest_by_binding, key=nearest_by_binding.__getitem__)
            del nearest_by_binding[worst_binding]
        delta = point[node.axis] - node.point[node.axis]
        near, far = (node.left, node.right) if delta <= 0.0 else (node.right, node.left)
        visit(near)
        if delta * delta <= worst_distance_squared():
            visit(far)

    visit(index)
    result = sorted((distance, binding_index) for binding_index, distance in nearest_by_binding.items())
    if len(result) < count:
        raise ImportError("BodyParts3D skin surface index has fewer than four registered bodies")
    return result[:count]


def _bodyparts_skin_smooth_visual_normals(
    normals: list[list[float]], triangles: list[tuple[int, int, int]], iterations: int = 3,
) -> list[list[float]]:
    """Smooth only the source-derived visual normal field across skin faces.

    The BodyParts3D exterior has a comparatively sparse faceted tessellation.
    Its original triangles remain untouched; this limited normal-only filter
    removes the renderer-visible checkerboard without inventing displacement,
    texture, skin weights, or material mechanics.
    """
    if iterations <= 0 or not normals or not triangles:
        raise ImportError("BodyParts3D skin visual normal smoothing has invalid input")
    current = [_bodyparts_unit_vector(list(normal), "BodyParts3D skin source normal") for normal in normals]
    for _ in range(iterations):
        accumulated = [[0.0, 0.0, 0.0] for _ in current]
        for triangle in triangles:
            if len(triangle) != 3 or any(not 0 <= index < len(current) for index in triangle):
                raise ImportError("BodyParts3D skin visual normal smoothing has an invalid triangle")
            first, second, third = triangle
            combined = [
                current[first][axis] + current[second][axis] + current[third][axis]
                for axis in range(3)
            ]
            for index in triangle:
                for axis in range(3):
                    accumulated[index][axis] += combined[axis]
        current = [
            _bodyparts_unit_vector(value, "BodyParts3D skin smoothed visual normal")
            for value in accumulated
        ]
    return current


def _bodyparts_skin_outer_surface_component(
    vertices_mm: list[list[float]], triangles: list[tuple[int, int, int]], member: str,
) -> tuple[list[list[float]], list[tuple[int, int, int]], dict[str, Any]]:
    """Extract the outer sheet from BodyParts3D's compound skin solid.

    `FJ2810` carries two near-coincident full-body components: the outside and
    inside of a thin source skin solid, plus tiny disconnected details.  The
    exterior is the largest enclosing connected component.  Retaining both
    sheets is correct for a closed solid but produces an illegible doubled
    silhouette in the raster inspection path.  This selection preserves every
    source vertex and triangle of the external component; it does not simplify,
    displace, fill, or synthesize skin geometry.
    """
    if not vertices_mm or not triangles:
        raise ImportError("BodyParts3D skin outer-surface selection has empty source geometry")
    # A point-touching triangle is not a continuous anatomical surface.  Work
    # over triangles connected by a full edge, rather than merely combining
    # every face that happens to reuse one OBJ vertex.
    parent = list(range(len(vertices_mm)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for triangle in triangles:
        if len(triangle) != 3 or any(not 0 <= index < len(vertices_mm) for index in triangle):
            raise ImportError(f"BodyParts3D skin {member} has an invalid source triangle")
        join(triangle[0], triangle[1])
        join(triangle[1], triangle[2])
    component_vertices: dict[int, list[int]] = {}
    for index in range(len(vertices_mm)):
        component_vertices.setdefault(find(index), []).append(index)
    component_triangles: Counter[int] = Counter(find(triangle[0]) for triangle in triangles)
    components: list[dict[str, Any]] = []
    for root, indices in component_vertices.items():
        minimum = [min(vertices_mm[index][axis] for index in indices) for axis in range(3)]
        maximum = [max(vertices_mm[index][axis] for index in indices) for axis in range(3)]
        components.append({
            "root": root,
            "indices": indices,
            "vertex_count": len(indices),
            "triangle_count": component_triangles[root],
            "minimum_mm": minimum,
            "maximum_mm": maximum,
            "volume_mm3": math.prod(maximum[axis] - minimum[axis] for axis in range(3)),
        })
    components.sort(key=lambda component: (component["volume_mm3"], component["vertex_count"]), reverse=True)
    outer = components[0]
    major_components = [
        component for component in components
        if component["vertex_count"] >= 0.5 * outer["vertex_count"]
    ]
    if len(major_components) < 2:
        raise ImportError(
            f"BodyParts3D skin {member} does not contain the expected nested source skin sheets"
        )
    if not any(
        component is not outer and all(
            outer["minimum_mm"][axis] <= component["minimum_mm"][axis] and
            outer["maximum_mm"][axis] >= component["maximum_mm"][axis]
            for axis in range(3)
        )
        for component in major_components
    ):
        raise ImportError(
            f"BodyParts3D skin {member} cannot identify an enclosing outer source sheet"
        )
    old_to_new = {index: new_index for new_index, index in enumerate(outer["indices"])}
    selected_triangles = [
        tuple(old_to_new[index] for index in triangle)
        for triangle in triangles
        if find(triangle[0]) == outer["root"]
    ]
    if not selected_triangles:
        raise ImportError(f"BodyParts3D skin {member} selected outer sheet has no triangles")
    return [vertices_mm[index] for index in outer["indices"]], selected_triangles, {
        "method": "exact_outer_connected_component_of_bodyparts3d_compound_skin_solid",
        "source_vertex_count": len(vertices_mm),
        "source_triangle_count": len(triangles),
        "retained_vertex_count": outer["vertex_count"],
        "retained_triangle_count": outer["triangle_count"],
        "outer_bounds_mm": [outer["minimum_mm"], outer["maximum_mm"]],
        "component_count": len(components),
        "nested_major_component_count": len(major_components),
        "boundary": (
            "exact source outer skin-sheet subset; inner sheet and disconnected source details "
            "are not part of the presented exterior shell"
        ),
    }


def _bodyparts_largest_connected_surface_component(
    vertices_mm: list[list[float]], triangles: list[tuple[int, int, int]], member: str,
) -> tuple[list[list[float]], list[tuple[int, int, int]], dict[str, Any]]:
    """Retain one source mesh's dominant connected anatomical sheet.

    Some BodyParts3D tendon and hallucis OBJ members include numerous
    disconnected sliver components alongside their main sheet. They read as
    floating or stretched tissue shards once posed against a bone. This
    selector keeps the largest source-connected sheet without moving, filling,
    welding, or remeshing it; provenance retains the discarded source-component
    count.
    """
    if not vertices_mm or not triangles:
        raise ImportError(f"BodyParts3D {member} connected-surface selection has empty source geometry")
    parent = list(range(len(triangles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    edge_owner: dict[tuple[int, int], int] = {}
    for triangle_index, triangle in enumerate(triangles):
        if len(triangle) != 3 or any(not 0 <= index < len(vertices_mm) for index in triangle):
            raise ImportError(f"BodyParts3D {member} has an invalid source triangle")
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = (min(first, second), max(first, second))
            prior_triangle = edge_owner.setdefault(edge, triangle_index)
            if prior_triangle != triangle_index:
                join(triangle_index, prior_triangle)
    component_triangle_indices: dict[int, list[int]] = {}
    for triangle_index in range(len(triangles)):
        component_triangle_indices.setdefault(find(triangle_index), []).append(triangle_index)
    component_vertex_counts = {
        root: len({index for triangle_index in triangle_indices for index in triangles[triangle_index]})
        for root, triangle_indices in component_triangle_indices.items()
    }
    dominant_root = max(
        component_triangle_indices,
        key=lambda root: (len(component_triangle_indices[root]), component_vertex_counts[root]),
    )
    selected_triangle_indices = component_triangle_indices[dominant_root]
    if not selected_triangle_indices:
        raise ImportError(f"BodyParts3D {member} dominant source component has no triangles")
    selected_indices = sorted({
        index for triangle_index in selected_triangle_indices for index in triangles[triangle_index]
    })
    old_to_new = {index: new_index for new_index, index in enumerate(selected_indices)}
    selected_triangles = [
        tuple(old_to_new[index] for index in triangles[triangle_index])
        for triangle_index in selected_triangle_indices
    ]
    return [vertices_mm[index] for index in selected_indices], selected_triangles, {
        "method": "exact_largest_edge_connected_component_of_bodyparts3d_source_surface",
        "source_vertex_count": len(vertices_mm),
        "source_triangle_count": len(triangles),
        "retained_vertex_count": len(selected_indices),
        "retained_triangle_count": len(selected_triangles),
        "discarded_component_count": len(component_triangle_indices) - 1,
        "boundary": (
            "exact dominant edge-connected source sheet; disconnected or point-touching source sliver components are not "
            "presented as anatomical tissue geometry"
        ),
    }


def bodyparts_myosim_skinned_shell_visual_payload(
    sources: Path, anatomy: dict[str, Any], registration_path: Path, output: Path,
) -> dict[str, Any]:
    """Prepare BodyParts3D's exterior shell for native multi-bone posing.

    The source has one exact exterior mesh, while the registered skeleton has
    many named source bones.  This offline importer assigns each source skin
    vertex to its four nearest *sampled exact registered source-bone surfaces*
    in the shared rest frame.  The samples replace coarse box-distance
    selection; a short joint band permits blending only between genuinely local
    candidate bodies.  Source-to-body transforms are recorded separately so
    the C++/Metal renderer can linearly blend the shell at the current
    articulated pose without a Python process.  This is an improved visual
    shell, deliberately not a claimed FEM skin, collision shell, closest-triangle
    skin-weight dataset, or clinical soft-tissue registration.
    """
    registration_file = registration_path.resolve()
    registration = read_json(registration_file)
    if registration.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2" or \
            registration.get("status") not in {
                "provisional_visual_registration_not_admitted_to_collision_or_physics",
                "inferred_attachment_surface_visual_registration_not_admitted_to_collision_or_physics",
            }:
        raise ImportError("BodyParts3D skinned shell requires a supported v2 visual registration")
    expected_bodyparts = {
        "id": anatomy.get("source_id"), "version": anatomy.get("version"),
        "archives": anatomy.get("archives"),
    }
    source = registration.get("source")
    if not isinstance(source, dict) or source.get("bodyparts") != expected_bodyparts:
        raise ImportError("BodyParts3D skinned shell registration does not match parsed source provenance")
    myosim = source.get("myosim")
    source_sha = myosim.get("source", {}).get("archive_sha256") if isinstance(myosim, dict) else None
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ImportError("BodyParts3D skinned shell has no MyoSim source SHA-256")
    coordinates = registration.get("coordinate_system")
    global_matrix = coordinates.get("global_source_mm_to_myosim_world_m") if isinstance(coordinates, dict) else None
    if not isinstance(global_matrix, list) or len(global_matrix) != 4:
        raise ImportError("BodyParts3D skinned shell has no global rest-frame registration")
    _, global_quaternion, _ = _bodyparts_visual_local_pose(
        global_matrix, "BodyParts3D skinned shell global transform",
    )
    global_rotation = _myosim_matrix_from_quaternion_xyzw(global_quaternion)
    anchors = registration.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != len(_BODYPARTS_MYOSIM_BONE_ANCHORS):
        raise ImportError("BodyParts3D skinned shell has incomplete visual-skeleton anchors")

    def world_point(vertex_mm: tuple[float, float, float]) -> list[float]:
        return [
            sum(global_matrix[row][column] * vertex_mm[column] for column in range(3)) +
            global_matrix[row][3]
            for row in range(3)
        ]

    bindings_by_body: dict[int, dict[str, Any]] = {}
    for specification, anchor in zip(_BODYPARTS_MYOSIM_BONE_ANCHORS, anchors, strict=True):
        if not isinstance(anchor, dict):
            raise ImportError("BodyParts3D skinned shell has an invalid visual-skeleton anchor")
        source_record, target, fitted = anchor.get("source"), anchor.get("target"), anchor.get("registration")
        if not isinstance(source_record, dict) or not isinstance(target, dict) or not isinstance(fitted, dict):
            raise ImportError("BodyParts3D skinned shell visual-skeleton anchor is incomplete")
        if source_record.get("member_id") != specification["member_id"] or \
                source_record.get("hierarchy") != specification["hierarchy"] or \
                target.get("name") != specification["myosim_body"]:
            raise ImportError("BodyParts3D skinned shell visual-skeleton anchor identity drifted")
        body_index = target.get("core_body_index")
        if not isinstance(body_index, int) or body_index < 0:
            raise ImportError("BodyParts3D skinned shell anchor has an invalid Core body index")
        translation, quaternion, scale = _bodyparts_visual_local_pose(
            fitted.get("source_obj_mm_to_core_inertial_body_m"),
            f"BodyParts3D skinned shell {specification['member_id']} local transform",
        )
        body_position = _myosim_vector(
            target.get("default_com_position_world_m"),
            f"BodyParts3D skinned shell {specification['member_id']} body position",
        )
        body_quaternion = list(target.get("default_inertial_quaternion_world_xyzw", []))
        _myosim_matrix_from_quaternion_xyzw(body_quaternion)
        record = bindings_by_body.get(body_index)
        if record is None:
            record = {
                "body_index": body_index,
                "translation": translation,
                "quaternion": quaternion,
                "scale": scale,
                "rest_position": body_position,
                "rest_quaternion": body_quaternion,
                "minimum": [math.inf, math.inf, math.inf],
                "maximum": [-math.inf, -math.inf, -math.inf],
                "source_members": [],
                "surface_samples": [],
            }
            bindings_by_body[body_index] = record
        else:
            values = (*translation, *quaternion, scale, *body_position, *body_quaternion)
            existing = (*record["translation"], *record["quaternion"], record["scale"],
                        *record["rest_position"], *record["rest_quaternion"])
            if any(abs(float(current) - float(reference)) > 2.0e-5
                   for current, reference in zip(values, existing, strict=True)):
                raise ImportError("BodyParts3D skinned shell has inconsistent source-to-body transforms")
        archive_path, member, obj = _bodyparts_obj_member(
            sources, specification["hierarchy"], specification["member_id"],
        )
        if source_record.get("archive_sha256") != sha256(archive_path) or \
                source_record.get("member") != member or \
                source_record.get("member_sha256") != hashlib.sha256(obj).hexdigest():
            raise ImportError("BodyParts3D skinned shell bone-envelope provenance drifted")
        bone_vertices_mm, _ = _bodyparts_obj_triangles(obj, member)
        for vertex in bone_vertices_mm:
            point = world_point(vertex)
            for axis in range(3):
                record["minimum"][axis] = min(record["minimum"][axis], point[axis])
                record["maximum"][axis] = max(record["maximum"][axis], point[axis])
        # A small stratified sample from every named source mesh preserves
        # hands, vertebrae, and paired small bones that a global downsample
        # would otherwise erase before body-level balancing below.
        for sample_index in _bodyparts_skin_surface_sample_indices(len(bone_vertices_mm)):
            record["surface_samples"].append(tuple(world_point(bone_vertices_mm[sample_index])))
        record["source_members"].append(specification["member_id"])
    if not bindings_by_body:
        raise ImportError("BodyParts3D skinned shell has no registered bone envelopes")
    bindings = [bindings_by_body[index] for index in sorted(bindings_by_body)]
    for binding in bindings:
        if not all(math.isfinite(value) for value in (*binding["minimum"], *binding["maximum"])) or \
                any(binding["minimum"][axis] > binding["maximum"][axis] for axis in range(3)):
            raise ImportError("BodyParts3D skinned shell has an empty registered bone envelope")
        if not binding["surface_samples"]:
            raise ImportError("BodyParts3D skinned shell has an empty registered bone surface")
        # Axial bodies can own many small meshes.  Cap only after each source
        # member has contributed samples, so a dense spine cannot make the
        # offline nearest-source index needlessly dominate build time.
        samples = binding["surface_samples"]
        if len(samples) > 256:
            binding["surface_samples"] = [
                samples[index]
                for index in _bodyparts_skin_surface_sample_indices(len(samples), 256)
            ]
    surface_index = _bodyparts_skin_surface_index([
        (sample, binding_index)
        for binding_index, binding in enumerate(bindings)
        for sample in binding["surface_samples"]
    ])

    archive_path, member, obj = _bodyparts_obj_member(sources, "is_a", "FJ2810")
    source_vertices_mm, source_triangles = _bodyparts_obj_triangles(obj, member)
    vertices_mm, triangles, outer_surface = _bodyparts_skin_outer_surface_component(
        source_vertices_mm, source_triangles, member,
    )
    normals = _bodyparts_vertex_normals(vertices_mm, triangles, member)
    normals = _bodyparts_skin_smooth_visual_normals(normals, triangles)
    source_skin_sha = hashlib.sha256(obj).hexdigest()
    vertex_payload: list[bytes] = []
    maximum_rest_error_m = 0.0
    influence_histogram: Counter[tuple[int, int, int, int]] = Counter()
    for vertex, normal in zip(vertices_mm, normals, strict=True):
        position_m = [coordinate * 0.001 for coordinate in vertex]
        world = world_point(vertex)
        candidates = _bodyparts_skin_nearest_surface_bindings(surface_index, world)
        if len(candidates) != 4:
            raise ImportError("BodyParts3D skinned shell has fewer than four bone bindings")
        # Only a candidate genuinely close to the nearest registered source
        # bone surface may share this vertex.  The band preserves conventional
        # local blend at a source joint while leaving ordinary limb/torso skin
        # rigidly local rather than letting a distant box overlap pull it.
        nearest_distance_m = math.sqrt(candidates[0][0])
        joint_band_m = 0.0125
        unnormalized = [
            (1.0 / (0.0075 + math.sqrt(distance_squared)) ** 4)
            if math.sqrt(distance_squared) <= nearest_distance_m + joint_band_m
            else 0.0
            for distance_squared, _ in candidates
        ]
        normalizer = sum(unnormalized)
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise ImportError("BodyParts3D skinned shell has invalid envelope weights")
        weights = [weight / normalizer for weight in unnormalized]
        if not all(math.isfinite(weight) and 0.0 <= weight <= 1.0 for weight in weights):
            raise ImportError("BodyParts3D skinned shell has non-finite blend weights")
        reconstructed = [0.0, 0.0, 0.0]
        for weight, (_, binding_index) in zip(weights, candidates, strict=True):
            binding = bindings[binding_index]
            local_rotation = _myosim_matrix_from_quaternion_xyzw(binding["quaternion"])
            body_rotation = _myosim_matrix_from_quaternion_xyzw(binding["rest_quaternion"])
            local = [
                binding["translation"][axis] + binding["scale"] *
                _myosim_matrix_vector(local_rotation, position_m)[axis]
                for axis in range(3)
            ]
            posed = [
                binding["rest_position"][axis] + _myosim_matrix_vector(body_rotation, local)[axis]
                for axis in range(3)
            ]
            for axis in range(3):
                reconstructed[axis] += weight * posed[axis]
        maximum_rest_error_m = max(
            maximum_rest_error_m,
            math.sqrt(sum((reconstructed[axis] - world[axis]) ** 2 for axis in range(3))),
        )
        indices = [binding_index for _, binding_index in candidates]
        world_normal = _bodyparts_unit_vector(
            _myosim_matrix_vector(global_rotation, normal),
            "BodyParts3D skinned shell rest world normal",
        )
        vertex_payload.append(struct.pack(
            "<6f4I4f", *position_m, *world_normal, *indices, *weights,
        ))
        influence_histogram[tuple(indices)] += 1
    if maximum_rest_error_m > 2.0e-5:
        raise ImportError("BodyParts3D skinned shell does not reconstruct its registered rest pose")
    index_payload = [index for triangle in triangles for index in triangle]
    if len(bindings) > 0xFFFFFFFF or len(vertex_payload) > 0xFFFFFFFF or len(index_payload) > 0xFFFFFFFF:
        raise ImportError("BodyParts3D skinned shell exceeds the uint32 native renderer capacity")
    registration_fingerprint = _bodyparts_visual_registration_fingerprint(registration_file)
    payload = b"".join([
        struct.pack(
            "<8s5I32s", _BODYPARTS_MYOSIM_SKIN_VISUAL_MAGIC,
            _BODYPARTS_MYOSIM_SKIN_VISUAL_ABI, len(bindings), len(vertex_payload),
            len(index_payload), registration_fingerprint, bytes.fromhex(source_sha),
        ),
        *[
            struct.pack("<I8f", binding["body_index"], *binding["translation"],
                        *binding["quaternion"], binding["scale"])
            for binding in bindings
        ],
        *vertex_payload,
        struct.pack(f"<{len(index_payload)}I", *index_payload),
    ])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "bodyparts3d-myosim-skinned-shell.nhskin"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-skinned-shell-visual-payload.v4",
        "payload": {
            "file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
            "magic": _BODYPARTS_MYOSIM_SKIN_VISUAL_MAGIC.rstrip(b"\0").decode("ascii"),
            "payload_abi": _BODYPARTS_MYOSIM_SKIN_VISUAL_ABI,
            "registration_fingerprint32": f"{registration_fingerprint:08x}",
            "binding_count": len(bindings), "vertex_count": len(vertices_mm),
            "triangle_count": len(triangles), "index_count": len(index_payload),
        },
        "source": {
            "registration": {"file": registration_file.name, "sha256": sha256(registration_file)},
            "bodyparts": expected_bodyparts, "myosim_source_archive_sha256": source_sha,
            "skin": {
                "member_id": "FJ2810", "member": member, "archive": archive_path.name,
                "archive_sha256": sha256(archive_path), "member_sha256": source_skin_sha,
                "source_vertex_count": len(source_vertices_mm),
                "source_triangle_count": len(source_triangles),
            },
            "registered_bone_envelopes": [
                {
                    "core_body_index": binding["body_index"],
                    "source_members": binding["source_members"],
                    "minimum_world_m": binding["minimum"], "maximum_world_m": binding["maximum"],
                    "surface_sample_count": len(binding["surface_samples"]),
                }
                for binding in bindings
            ],
        },
        "coverage": {
            "influences_per_vertex": 4,
            "distinct_influence_quartets": len(influence_histogram),
            "joint_band_m": 0.0125,
            "source_bone_surface_sample_count": sum(
                len(binding["surface_samples"]) for binding in bindings
            ),
            "rest_pose_reconstruction_max_error_m": maximum_rest_error_m,
            "outer_source_surface": outer_surface,
            "normal_binding": (
                "registered_world_rest_normal blended through each articulated "
                "body's current-from-rest rotation"
            ),
            "normal_presentation": {
                "method": "three_pass_triangle_neighbour_source_normal_smoothing",
                "changes": "visual normals only; exact source vertices and triangle connectivity are retained",
            },
        },
        "runtime_binding": "Exact BodyParts3D source skin triangles use four nearest distinct registered source-bone surface samples with deterministic inverse-quartic weights, restricted to candidates within a 12.5 mm source-joint band of the nearest sample; each influence carries its source-to-Core local transform for native C++/Metal posing. The source normal is registered once into the shared world rest frame and follows each articulated influence through its current-from-rest body rotation.",
        "status": "native_four_body_source_surface_local_linear_blend_skin_shell_visual_input_not_collision_or_physics",
        "evidence_boundary": "This is a sampled-source-bone-surface-proximity-derived articulated visual shell, not FEM/MPM skin, a tissue material law, collision/contact geometry, clinical registration, closest-triangle anatomical skin weights, or a force-coupled soft-tissue model.",
    }
    write_json(output / "bodyparts3d-myosim-skinned-shell.manifest.json", manifest)
    return manifest


def _quaternion_xyzw_from_matrix(matrix: list[list[float]]) -> list[float]:
    """Return the deterministic xyzw quaternion for a proper rotation matrix."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        result = [
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        ]
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = 2.0 * math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])
        result = [
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[2][1] - matrix[1][2]) / scale,
        ]
    elif matrix[1][1] > matrix[2][2]:
        scale = 2.0 * math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])
        result = [
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
        ]
    else:
        scale = 2.0 * math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])
        result = [
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        ]
    magnitude = math.sqrt(sum(value * value for value in result))
    if not math.isfinite(magnitude) or not magnitude > 1.0e-12:
        raise ImportError("OpenSim frame orientation cannot form a unit quaternion")
    result = [value / magnitude for value in result]
    return [-value for value in result] if result[3] < 0.0 else result


def _inverse_symmetric3(inertia: dict[str, Any], context: str) -> list[list[float]]:
    required = {"xx", "xy", "xz", "yy", "yz", "zz"}
    if not isinstance(inertia, dict) or set(inertia) != required:
        raise ImportError(f"{context} has incomplete inertia")
    xx, xy, xz, yy, yz, zz = (
        _finite_scalar(inertia[key], f"{context} {key}")
        for key in ("xx", "xy", "xz", "yy", "yz", "zz")
    )
    determinant = (
        xx * (yy * zz - yz * yz)
        - xy * (xy * zz - yz * xz)
        + xz * (xy * yz - yy * xz)
    )
    if not determinant > 0.0 or not math.isfinite(determinant):
        raise ImportError(f"{context} is not positive definite")
    inverse = [
        [(yy * zz - yz * yz) / determinant, (xz * yz - xy * zz) / determinant, (xy * yz - xz * yy) / determinant],
        [(xz * yz - xy * zz) / determinant, (xx * zz - xz * xz) / determinant, (xy * xz - xx * yz) / determinant],
        [(xy * yz - xz * yy) / determinant, (xy * xz - xx * yz) / determinant, (xx * yy - xy * xy) / determinant],
    ]
    if not all(math.isfinite(value) for row in inverse for value in row):
        raise ImportError(f"{context} inverse is non-finite")
    return inverse


def _joint_frame_body_transform(
    joint: dict[str, Any], frame_reference: Any, context: str
) -> tuple[str | None, list[float], list[list[float]]]:
    """Resolve a joint socket to its source body and body-frame transform."""
    frames = {
        frame.get("id"): frame
        for frame in joint.get("frames", [])
        if isinstance(frame, dict) and isinstance(frame.get("id"), str)
    }

    def resolve(reference: Any, seen: set[str]) -> tuple[str | None, list[float], list[list[float]]]:
        if not isinstance(reference, str) or not reference:
            raise ImportError(f"OpenSim joint {joint.get('id')} has unresolved {context} frame")
        if reference in {"ground", "/ground"}:
            return None, [0.0, 0.0, 0.0], _matrix_from_body_fixed_xyz([0.0, 0.0, 0.0])
        if reference.startswith("/bodyset/"):
            body = reference.rsplit("/", 1)[-1]
            if not body:
                raise ImportError(f"OpenSim joint {joint.get('id')} has invalid {context} body frame")
            return body, [0.0, 0.0, 0.0], _matrix_from_body_fixed_xyz([0.0, 0.0, 0.0])
        local = reference.rsplit("/", 1)[-1]
        if local in seen:
            raise ImportError(f"OpenSim joint {joint.get('id')} has cyclic {context} frame chain")
        frame = frames.get(local)
        if frame is None:
            raise ImportError(f"OpenSim joint {joint.get('id')} cannot resolve {context} frame {reference}")
        parent, parent_translation, parent_rotation = resolve(
            frame.get("parent_frame"), seen | {local}
        )
        translation = _vector3(frame.get("translation_m"), f"OpenSim {joint['id']} {context} translation")
        rotation = _matrix_from_body_fixed_xyz(
            _vector3(frame.get("orientation_rad"), f"OpenSim {joint['id']} {context} orientation")
        )
        return (
            parent,
            [
                parent_translation[index] + _matrix_vector(parent_rotation, translation)[index]
                for index in range(3)
            ],
            _matrix_product(parent_rotation, rotation),
        )

    return resolve(frame_reference, set())


def _pack_float4(values: Iterable[float], context: str) -> bytes:
    materialized = list(values)
    if len(materialized) != 4:
        raise ImportError(f"{context} must have four scalar lanes")
    try:
        return struct.pack("<4f", *(_finite_scalar(value, context) for value in materialized))
    except OverflowError as error:
        raise ImportError(f"{context} is outside FP32 range") from error


def rajagopal_core_reference_artifact(
    model: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Compile the full Rajagopal tree into the Core FP64 reference payload.

    The artifact preserves source masses, COMs, inertia tensors, joint frames,
    coordinates, and all CustomJoint binary programs. It deliberately owns no
    BodyParts3D registration, collision geometry, muscle/tendon state, or
    accelerated Metal function-based solver contract.
    """
    skeleton = rajagopal_rigid_skeleton_ir(model)
    source_hash = skeleton["source"].get("sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ImportError("Rajagopal Core reference payload requires a source SHA-256")
    bodies_by_id = {body.get("id"): body for body in model.get("bodies", [])}
    if len(bodies_by_id) != len(model.get("bodies", [])) or not bodies_by_id:
        raise ImportError("Rajagopal Core reference payload has invalid body identities")
    unresolved = list(model.get("joints", []))
    ordered_joints: list[dict[str, Any]] = []
    ordered_body_ids = ["__ground__"]
    body_index = {"__ground__": 0}
    while unresolved:
        progressed = False
        for joint in list(unresolved):
            parent, _, _ = _joint_frame_body_transform(joint, joint.get("parent_frame"), "parent")
            child, _, _ = _joint_frame_body_transform(joint, joint.get("child_frame"), "child")
            parent_id = "__ground__" if parent is None else parent
            if parent_id not in body_index:
                continue
            if not isinstance(child, str) or child not in bodies_by_id or child in body_index:
                raise ImportError(f"Rajagopal Core reference tree has invalid child at {joint.get('id')}")
            body_index[child] = len(ordered_body_ids)
            ordered_body_ids.append(child)
            ordered_joints.append(joint)
            unresolved.remove(joint)
            progressed = True
        if not progressed:
            raise ImportError("Rajagopal Core reference tree is disconnected or cyclic")
    if len(ordered_joints) != len(model.get("joints", [])):
        raise ImportError("Rajagopal Core reference payload did not retain every source joint")

    body_records: list[bytes] = []
    # Fixed roots do not contribute a generalized coordinate. This synthetic
    # dynamic anchor exists solely because the current Core fixed-tree ABI
    # requires one in-articulation root body; its inertial fields are never a
    # source-anatomy claim and contribute no articulated mass column.
    body_records.append(
        struct.pack(
            "<4I", 0, 0xFFFFFFFF, 0xFFFFFFFF, _MR_MOTION_DYNAMIC
        )
        + _pack_float4([1.0, 1.0, 0.0, 0.0], "synthetic ground mass")
        + _pack_float4([0.0, 0.0, 0.0, 0.0], "synthetic ground COM")
        + _pack_float4([1.0, 0.0, 0.0, 0.0], "synthetic ground inertia row 0")
        + _pack_float4([0.0, 1.0, 0.0, 0.0], "synthetic ground inertia row 1")
        + _pack_float4([0.0, 0.0, 1.0, 0.0], "synthetic ground inertia row 2")
        + _pack_float4([1.0, 0.0, 0.0, 0.0], "synthetic ground inverse inertia row 0")
        + _pack_float4([0.0, 1.0, 0.0, 0.0], "synthetic ground inverse inertia row 1")
        + _pack_float4([0.0, 0.0, 1.0, 0.0], "synthetic ground inverse inertia row 2")
        + _pack_float4([0.0, 0.0, 1.0e6, 1.0e6], "synthetic ground damping")
    )
    for identifier in ordered_body_ids[1:]:
        body = bodies_by_id[identifier]
        mass = _finite_scalar(body.get("mass_kg"), f"OpenSim body {identifier} mass")
        if not mass > 0.0:
            raise ImportError(f"OpenSim body {identifier} mass must be positive")
        com = _vector3(body.get("mass_center_m"), f"OpenSim body {identifier} COM")
        inertia = body.get("inertia_kg_m2")
        inverse = _inverse_symmetric3(inertia, f"OpenSim body {identifier} inertia")
        matrix = [
            [inertia["xx"], inertia["xy"], inertia["xz"]],
            [inertia["xy"], inertia["yy"], inertia["yz"]],
            [inertia["xz"], inertia["yz"], inertia["zz"]],
        ]
        parent_joint_index = next(
            index for index, joint in enumerate(ordered_joints) if joint.get("child_frame")
            and _joint_frame_body_transform(joint, joint.get("child_frame"), "child")[0] == identifier
        )
        parent_body, _, _ = _joint_frame_body_transform(
            ordered_joints[parent_joint_index], ordered_joints[parent_joint_index].get("parent_frame"), "parent"
        )
        body_records.append(
            struct.pack(
                "<4I",
                0,
                body_index["__ground__" if parent_body is None else parent_body],
                parent_joint_index,
                _MR_MOTION_DYNAMIC,
            )
            + _pack_float4([mass, 1.0 / mass, 0.0, 0.0], f"OpenSim body {identifier} mass")
            + _pack_float4([*com, 0.0], f"OpenSim body {identifier} COM")
            + b"".join(_pack_float4([*row, 0.0], f"OpenSim body {identifier} inertia") for row in matrix)
            + b"".join(_pack_float4([*row, 0.0], f"OpenSim body {identifier} inverse inertia") for row in inverse)
            + _pack_float4([0.0, 0.0, 1.0e6, 1.0e6], f"OpenSim body {identifier} damping")
        )

    default_q: list[float] = []
    default_v: list[float] = []
    joint_records: list[bytes] = []
    dof_records: list[bytes] = []
    function_programs: list[tuple[int, bytes]] = []
    joint_manifest: list[dict[str, Any]] = []
    for joint_index, joint in enumerate(ordered_joints):
        identifier = joint.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ImportError("Rajagopal Core reference payload has unnamed joint")
        parent_body, parent_translation, parent_rotation = _joint_frame_body_transform(
            joint, joint.get("parent_frame"), "parent"
        )
        child_body, child_translation, child_rotation = _joint_frame_body_transform(
            joint, joint.get("child_frame"), "child"
        )
        if not isinstance(child_body, str) or child_body not in bodies_by_id:
            raise ImportError(f"Rajagopal Core reference joint {identifier} has invalid child")
        parent_id = "__ground__" if parent_body is None else parent_body
        if parent_id not in body_index:
            raise ImportError(f"Rajagopal Core reference joint {identifier} has invalid parent")
        child_com = _vector3(bodies_by_id[child_body].get("mass_center_m"), f"OpenSim body {child_body} COM")
        parent_com = [0.0, 0.0, 0.0] if parent_body is None else _vector3(
            bodies_by_id[parent_body].get("mass_center_m"), f"OpenSim body {parent_body} COM"
        )
        parent_anchor = [parent_translation[index] - parent_com[index] for index in range(3)]
        child_anchor = [child_translation[index] - child_com[index] for index in range(3)]
        kind = joint.get("kind")
        coordinates = joint.get("coordinates")
        if not isinstance(coordinates, list):
            raise ImportError(f"Rajagopal Core reference joint {identifier} has invalid coordinates")
        if kind == "PinJoint":
            joint_type, nq, nv = _MR_JOINT_REVOLUTE, 1, 1
            axis0 = [0.0, 0.0, 1.0]
        elif kind == "CustomJoint":
            joint_type, nq, nv = _MR_JOINT_FUNCTION_BASED, len(coordinates), len(coordinates)
            program, coordinate_ids = pack_opensim_spatial_transform_gpu(joint)
            if len(coordinate_ids) != nq:
                raise ImportError(f"Rajagopal Core reference program coordinates mismatch at {identifier}")
            function_programs.append((joint_index, program))
            axis0 = [0.0, 0.0, 0.0]
        elif kind == "UniversalJoint" and coordinates and all(
            coordinate.get("locked") in (True, "true", "True")
            and coordinate.get("default_value") == 0.0
            for coordinate in coordinates
        ):
            joint_type, nq, nv = _MR_JOINT_FIXED, 0, 0
            axis0 = [0.0, 0.0, 0.0]
        else:
            raise ImportError(f"Rajagopal Core reference cannot exactly lower {kind} {identifier}")
        q_offset = len(default_q)
        v_offset = len(default_v)
        for local_dof, coordinate in enumerate(coordinates[:nv]):
            default = _finite_scalar(coordinate.get("default_value"), f"OpenSim coordinate {identifier}")
            default_q.append(default)
            default_v.append(0.0)
            flags = 0
            limits = [0.0, 0.0, 0.0, 0.0]
            range_value = coordinate.get("range")
            if coordinate.get("clamped") in (True, "true", "True"):
                if not isinstance(range_value, list) or len(range_value) != 2:
                    raise ImportError(f"OpenSim coordinate {identifier} has invalid position range")
                lower = _finite_scalar(range_value[0], f"OpenSim coordinate {identifier} lower range")
                upper = _finite_scalar(range_value[1], f"OpenSim coordinate {identifier} upper range")
                if lower > upper or default < lower or default > upper:
                    raise ImportError(f"OpenSim coordinate {identifier} default violates source range")
                flags |= _MR_DOF_POSITION_LIMIT
                limits[0], limits[1] = lower, upper
            dof_records.append(
                struct.pack("<8I", 0, joint_index, q_offset + local_dof, v_offset + local_dof, local_dof, flags, 0, 0)
                + _pack_float4(limits, f"OpenSim coordinate {identifier} limits")
                + _pack_float4([0.0, 0.0, 0.0, 0.0], f"OpenSim coordinate {identifier} drive")
            )
        joint_records.append(
            struct.pack("<8I", body_index[parent_id], body_index[child_body], joint_type, 0, q_offset, nq, v_offset, nv)
            + _pack_float4([*axis0, 0.0], f"OpenSim joint {identifier} axis 0")
            + _pack_float4([0.0, 0.0, 0.0, 0.0], f"OpenSim joint {identifier} axis 1")
            + _pack_float4([0.0, 0.0, 0.0, 0.0], f"OpenSim joint {identifier} axis 2")
            + _pack_float4([*parent_anchor, 0.0], f"OpenSim joint {identifier} parent anchor")
            + _pack_float4([*child_anchor, 0.0], f"OpenSim joint {identifier} child anchor")
            + _pack_float4(_quaternion_xyzw_from_matrix(parent_rotation), f"OpenSim joint {identifier} parent rotation")
            + _pack_float4(_quaternion_xyzw_from_matrix(child_rotation), f"OpenSim joint {identifier} child rotation")
        )
        joint_manifest.append({
            "id": identifier,
            "source_kind": kind,
            "core_joint_type": joint_type,
            "q_offset": q_offset,
            "nq": nq,
            "v_offset": v_offset,
            "nv": nv,
            "parent_body": parent_id,
            "child_body": child_body,
        })

    nq, nv = len(default_q), len(default_v)
    world = struct.pack(
        "<16I8f",
        _MR_ENGINE_ABI_VERSION, len(body_records), 1, len(joint_records),
        0, 0, nq, nv,
        1, 1, 1, 1,
        0, 0, 0, 0,
        0.0, -9.81, 0.0, 1.0 / 1000.0,
        1.0e-8, 1.0e-9, 2.0, 1.0e-4,
    )
    articulation = struct.pack(
        "<12I", 0, _MR_ROOT_FIXED, 0, len(body_records), 0, len(joint_records), 0, nq, 0, nv, 0, 0
    )
    header = struct.pack(
        "<8s9I32s",
        _RAJAGOPAL_CORE_REFERENCE_MAGIC,
        _RAJAGOPAL_CORE_REFERENCE_ABI,
        _MR_ENGINE_ABI_VERSION,
        len(ordered_body_ids) - 1,
        len(body_records),
        len(joint_records),
        nq,
        nv,
        len(function_programs),
        0,
        bytes.fromhex(source_hash),
    )
    payload = b"".join(
        [
            header,
            world,
            articulation,
            *body_records,
            *joint_records,
            *dof_records,
            *(struct.pack("<2I", joint_index, 0) + program for joint_index, program in function_programs),
            struct.pack(f"<{nq}f", *default_q),
            struct.pack(f"<{nv}f", *default_v),
        ]
    )
    expected_bytes = 76 + 96 + 48 + 160 * len(body_records) + 144 * len(joint_records) + 64 * nv + 2520 * len(function_programs) + 4 * (nq + nv)
    if len(payload) != expected_bytes:
        raise ImportError("internal Rajagopal Core reference payload ABI size mismatch")
    return (
        {
            "schema": "numi.human.rajagopal-core-reference.v1",
            "source": skeleton["source"],
            "payload": {
                "file": "rajagopal-core-reference.nhrigid",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "byte_order": "little-endian IEEE-754 binary32",
                "payload_abi": _RAJAGOPAL_CORE_REFERENCE_ABI,
                "engine_abi": _MR_ENGINE_ABI_VERSION,
            },
            "source_body_count": len(ordered_body_ids) - 1,
            "engine_body_count": len(body_records),
            "joint_count": len(joint_records),
            "nq": nq,
            "nv": nv,
            "function_based_program_count": len(function_programs),
            "body_order": ordered_body_ids,
            "joints": joint_manifest,
            "runtime_requirement": (
                "Load this payload into the Core FP64 FunctionBased reference; Metal ABA, "
                "BodyParts3D registration/colliders, and muscle-tendon runtime remain separate gates."
            ),
            "evidence_boundary": (
                "Source-faithful rigid-tree data and canonical FunctionBased programs only. "
                "The synthetic fixed ground anchor has no source-anatomy meaning and no generalized mass column."
            ),
        },
        payload,
    )


def rajagopal_millard_reference_artifact(
    model: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Compile source Millard records into Core's fixed-layout reference ABI.

    This binary owns every source muscle scalar, curve property, COM-relative
    GeometryPath point, and source-order PathWrap cylinder. Empty optional
    curve properties are materialized only with their documented OpenSim class
    defaults, preserving the source curve semantics without fitting or
    approximating any curve.
    """
    millard = rajagopal_millard_muscle_ir(model)
    core_manifest, _ = rajagopal_core_reference_artifact(model)
    body_order = core_manifest.get("body_order")
    if not isinstance(body_order, list) or body_order[:1] != ["__ground__"]:
        raise ImportError("Millard reference requires canonical Core body order")
    body_index = {identifier: index for index, identifier in enumerate(body_order)}
    bodies = {body.get("id"): body for body in model.get("bodies", [])}
    if len(bodies) != len(model.get("bodies", [])):
        raise ImportError("Millard reference requires unique source body identities")
    wraps = {wrap.get("id"): wrap for wrap in model.get("wrap_objects", [])}
    if len(wraps) != len(model.get("wrap_objects", [])):
        raise ImportError("Millard reference requires unique source wrap identities")

    muscle_records: list[bytes] = []
    curve_records: list[bytes] = []
    point_records: list[bytes] = []
    wrap_records: list[bytes] = []
    muscle_manifest: list[dict[str, Any]] = []
    for muscle in millard["muscles"]:
        identifier = muscle["id"]
        parameters = muscle["parameters"]
        curves = muscle["curves"]

        def curve_values(
            curve_name: str,
            required: tuple[str, ...],
            defaults: dict[str, float],
        ) -> list[float]:
            curve = curves.get(curve_name)
            values = curve.get("parameters") if isinstance(curve, dict) else None
            if not isinstance(values, dict):
                raise ImportError(f"Millard muscle {identifier} has invalid {curve_name}")
            allowed = set(required) | set(defaults)
            unknown = set(values) - allowed
            if unknown:
                raise ImportError(
                    f"Millard muscle {identifier} has unsupported {curve_name} properties: "
                    + ", ".join(sorted(unknown))
                )
            missing = set(required) - set(values)
            if missing:
                raise ImportError(
                    f"Millard muscle {identifier} has incomplete {curve_name} properties: "
                    + ", ".join(sorted(missing))
                )
            materialized = {**defaults, **values}
            return [
                _finite_scalar(materialized[name], f"Millard muscle {identifier} {curve_name} {name}")
                for name in (*required, *defaults)
            ]

        active = curve_values(
            "ActiveForceLengthCurve",
            (
                "min_norm_active_fiber_length",
                "transition_norm_fiber_length",
                "max_norm_active_fiber_length",
                "shallow_ascending_slope",
                "minimum_value",
            ),
            {},
        )
        velocity = curve_values(
            "ForceVelocityCurve",
            (
                "concentric_slope_at_vmax",
                "concentric_slope_near_vmax",
                "isometric_slope",
                "eccentric_slope_at_vmax",
                "eccentric_slope_near_vmax",
                "max_eccentric_velocity_force_multiplier",
            ),
            {
                "concentric_curviness": 0.6,
                "eccentric_curviness": 0.9,
            },
        )
        passive = curve_values(
            "FiberForceLengthCurve",
            (
                "strain_at_zero_force",
                "strain_at_one_norm_force",
                "stiffness_at_low_force",
                "stiffness_at_one_norm_force",
                "curviness",
            ),
            {},
        )
        tendon = curve_values(
            "TendonForceLengthCurve",
            (),
            {
                "strain_at_one_norm_force": 0.049,
                "stiffness_at_one_norm_force": 1.375 / 0.049,
                "norm_force_at_toe_end": 2.0 / 3.0,
                "curviness": 0.5,
            },
        )
        curve_records.append(struct.pack("<22f", *(active + velocity + passive + tendon)))
        point_offset = len(point_records)
        for point in muscle["path_points"]:
            frame = point["parent_frame"]
            body_id = frame.rsplit("/", 1)[-1]
            body = bodies.get(body_id)
            if body is None or body_id not in body_index:
                raise ImportError(f"Millard muscle {identifier} has unresolved path body {body_id}")
            location = _vector3(point["location_m"], f"Millard muscle {identifier} path point")
            center = _vector3(body["mass_center_m"], f"Millard body {body_id} mass centre")
            point_records.append(
                struct.pack(
                    "<I3f",
                    body_index[body_id],
                    *[location[index] - center[index] for index in range(3)],
                )
            )
        wrap_offset = len(wrap_records)
        compiled_wraps: list[dict[str, Any]] = []
        for path_wrap in muscle["path_wraps"]:
            source_wrap = wraps.get(path_wrap["wrap_object"])
            if source_wrap is None or source_wrap.get("kind") != "WrapCylinder":
                raise ImportError(
                    f"Millard muscle {identifier} requires supported WrapCylinder "
                    f"{path_wrap.get('wrap_object')}"
                )
            parent = source_wrap.get("parent_frame")
            if not isinstance(parent, str) or parent not in bodies or parent not in body_index:
                raise ImportError(f"Millard muscle {identifier} has unresolved cylinder parent")
            source_parameters = source_wrap.get("parameters")
            if not isinstance(source_parameters, dict):
                raise ImportError(f"Millard cylinder {source_wrap.get('id')} has invalid parameters")
            translation = _vector3(
                source_parameters.get("translation"),
                f"Millard cylinder {source_wrap.get('id')} translation",
            )
            rotation = _vector3(
                source_parameters.get("xyz_body_rotation"),
                f"Millard cylinder {source_wrap.get('id')} rotation",
            )
            radius = _finite_scalar(
                source_parameters.get("radius"),
                f"Millard cylinder {source_wrap.get('id')} radius",
            )
            length = _finite_scalar(
                source_parameters.get("length"),
                f"Millard cylinder {source_wrap.get('id')} length",
            )
            if radius <= 0.0 or length <= 0.0:
                raise ImportError(f"Millard cylinder {source_wrap.get('id')} dimensions must be positive")
            method = path_wrap.get("method") or "hybrid"
            method_codes = {"hybrid": 0, "midpoint": 1, "axial": 2}
            if method not in method_codes:
                raise ImportError(
                    f"Millard muscle {identifier} has unsupported PathWrap method {method!r}"
                )
            source_range = path_wrap.get("range")
            if source_range is None:
                start_point, end_point = -1, -1
            else:
                if not isinstance(source_range, list) or len(source_range) != 2:
                    raise ImportError(
                        f"Millard muscle {identifier} PathWrap range must have two indices"
                    )
                range_values = [
                    _finite_scalar(value, f"Millard muscle {identifier} PathWrap range")
                    for value in source_range
                ]
                if any(value != int(value) for value in range_values):
                    raise ImportError(
                        f"Millard muscle {identifier} PathWrap range must use integral indices"
                    )
                start_point, end_point = (int(value) for value in range_values)
            if not (
                (start_point == -1 or 1 <= start_point <= len(muscle["path_points"]))
                and (end_point == -1 or 1 <= end_point <= len(muscle["path_points"]))
                and (start_point == -1 or end_point == -1 or start_point <= end_point)
            ):
                raise ImportError(
                    f"Millard muscle {identifier} has invalid PathWrap range "
                    f"[{start_point}, {end_point}]"
                )
            center = _vector3(bodies[parent]["mass_center_m"], f"Millard body {parent} mass centre")
            wrap_records.append(
                struct.pack(
                    "<I8fiiI",
                    body_index[parent],
                    *[translation[index] - center[index] for index in range(3)],
                    *rotation,
                    radius,
                    length,
                    start_point,
                    end_point,
                    method_codes[method],
                )
            )
            compiled_wraps.append(
                {
                    "source_wrap_object": path_wrap["wrap_object"],
                    "method": method,
                    "range": [start_point, end_point],
                }
            )
        ignore_tendon = parameters["ignore_tendon_compliance"] in (True, "true")
        muscle_records.append(
            struct.pack(
                "<7f5I",
                _finite_scalar(parameters["max_isometric_force"], f"Millard muscle {identifier} maximum force"),
                _finite_scalar(parameters["optimal_fiber_length"], f"Millard muscle {identifier} optimal fibre length"),
                _finite_scalar(parameters["tendon_slack_length"], f"Millard muscle {identifier} tendon slack length"),
                _finite_scalar(parameters["pennation_angle_at_optimal"], f"Millard muscle {identifier} pennation"),
                _finite_scalar(parameters["fiber_damping"], f"Millard muscle {identifier} damping"),
                _finite_scalar(parameters["default_activation"], f"Millard muscle {identifier} default activation"),
                _finite_scalar(parameters["minimum_activation"], f"Millard muscle {identifier} minimum activation"),
                point_offset,
                len(muscle["path_points"]),
                wrap_offset,
                len(muscle["path_wraps"]),
                1 if ignore_tendon else 0,
            )
        )
        muscle_manifest.append(
            {
                "id": identifier,
                "path_point_offset": point_offset,
                "path_point_count": len(muscle["path_points"]),
                "wrap_offset": wrap_offset,
                "wrap_count": len(muscle["path_wraps"]),
                "wraps": compiled_wraps,
            }
        )
    source_hash = millard["source"].get("sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ImportError("Millard reference requires a source SHA-256")
    header = struct.pack(
        "<8s6I32s",
        _RAJAGOPAL_MILLARD_REFERENCE_MAGIC,
        _RAJAGOPAL_MILLARD_REFERENCE_ABI,
        len(body_order) - 1,
        len(muscle_records),
        len(point_records),
        len(wrap_records),
        0,
        bytes.fromhex(source_hash),
    )
    payload = b"".join(
        [header, *muscle_records, *curve_records, *point_records, *wrap_records]
    )
    expected_bytes = (
        64
        + 48 * len(muscle_records)
        + 88 * len(curve_records)
        + 16 * len(point_records)
        + 48 * len(wrap_records)
    )
    if len(payload) != expected_bytes:
        raise ImportError("internal Rajagopal Millard reference payload ABI size mismatch")
    return (
        {
            "schema": "numi.human.rajagopal-millard-reference.v1",
            "source": millard["source"],
            "payload": {
                "file": "rajagopal-millard-reference.nhmuscle",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "payload_abi": _RAJAGOPAL_MILLARD_REFERENCE_ABI,
                "byte_order": "little-endian IEEE-754 binary32",
            },
            "body_order": body_order,
            "muscle_count": len(muscle_records),
            "path_point_count": len(point_records),
            "path_wrap_count": len(wrap_records),
            "curve_record_bytes": 88,
            "wrap_record_bytes": 48,
            "muscles": muscle_manifest,
            "curve_ir_schema": millard["schema"],
            "runtime_requirement": (
                "Load this payload with the matching rigid-tree payload into "
                "MillardMuscleReference, evaluate source curves, solve force equilibrium, "
                "then project GeometryPath tension into articulated generalized force."
            ),
            "evidence_boundary": (
                "Exact source scalars, curve properties (including OpenSim class defaults where "
                "the source leaves them empty), body-frame path points, and WrapCylinder definitions "
                "with source PathWrap method and range. "
                "Neither artifact is a validated OpenSim-equivalence or device-resident muscle result."
            ),
        },
        payload,
    )


def _wrap_objects(model: ET.Element) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for frame in model.iter():
        wrap_set = next((item for item in frame if _local_name(item) == "WrapObjectSet"), None)
        if wrap_set is None:
            continue
        objects = next((item for item in wrap_set if _local_name(item) == "objects"), None)
        if objects is None:
            continue
        for wrap in objects:
            identifier = wrap.get("name")
            if not identifier:
                continue
            result.append(
                {
                    "id": identifier,
                    "kind": _local_name(wrap),
                    "parent_frame": frame.get("name"),
                    "parameters": _opensim_direct_leaf_properties(wrap),
                    "source_xml": _source_xml(wrap),
                }
            )
    return result


def parse_opensim(path: Path, source_id: str) -> dict[str, Any]:
    """Extract serializable mechanics without changing OpenSim source values."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ImportError(f"{path.name} is not parseable OpenSim XML: {error}") from error

    model = next((item for item in root.iter() if _local_name(item) == "Model"), None)
    if model is None:
        raise ImportError(f"{path.name} has no OpenSim Model element")

    body_set = next((item for item in model.iter() if _local_name(item) == "BodySet"), None)
    joint_set = next((item for item in model.iter() if _local_name(item) == "JointSet"), None)
    force_set = next((item for item in model.iter() if _local_name(item) == "ForceSet"), None)
    if body_set is None or force_set is None:
        raise ImportError(f"{path.name} must contain BodySet and ForceSet")

    bodies: list[dict[str, Any]] = []
    for body in _children(body_set, "Body"):
        identifier = body.get("name")
        if not identifier:
            continue
        bodies.append(
            {
                "id": identifier,
                "mass_kg": _number_or_text(_text(body, "mass")),
                "mass_center_m": _number_or_text(_text(body, "mass_center")),
                "inertia_kg_m2": _body_inertia(body),
                "source_xml": _source_xml(body),
            }
        )

    joints: list[dict[str, Any]] = []
    source_joints: list[tuple[ET.Element, str | None]] = []
    if joint_set is not None:
        objects = next((item for item in joint_set if _local_name(item) == "objects"), None)
        source_joints.extend((joint, None) for joint in (list(objects) if objects is not None else []))
    else:
        # OpenSim 3 serializes each body's inbound Joint beneath that body
        # rather than in a model-level JointSet.  Preserve that legacy source
        # structure without pretending it already has modern socket frames.
        for body in _children(body_set, "Body"):
            owning_body = body.get("name")
            holder = next((item for item in body if _local_name(item) == "Joint"), None)
            if holder is None:
                continue
            source_joints.extend(
                (candidate, owning_body)
                for candidate in holder
                if isinstance(candidate.tag, str)
            )
    for joint, owning_body in source_joints:
        identifier = joint.get("name")
        if not identifier:
            continue
        coordinates: list[dict[str, Any]] = []
        for coordinate in _children(joint, "Coordinate"):
            coordinate_id = coordinate.get("name")
            if coordinate_id:
                coordinates.append(
                    {
                        "id": coordinate_id,
                        "default_value": _number_or_text(_text(coordinate, "default_value")),
                        "range": _number_or_text(_text(coordinate, "range")),
                        "clamped": _number_or_text(_text(coordinate, "clamped")),
                        "locked": _number_or_text(_text(coordinate, "locked")),
                    }
                )
        legacy_parent = _text(joint, "parent_body")
        legacy_frames = []
        if owning_body is not None:
            legacy_frames = [
                {
                    "id": "__legacy_parent__",
                    "kind": "OpenSim3LegacyFrame",
                    "parent_frame": legacy_parent,
                    "translation_m": _number_or_text(_text(joint, "location_in_parent")),
                    "orientation_rad": _number_or_text(_text(joint, "orientation_in_parent")),
                },
                {
                    "id": "__legacy_child__",
                    "kind": "OpenSim3LegacyFrame",
                    "parent_frame": owning_body,
                    "translation_m": _number_or_text(_text(joint, "location")),
                    "orientation_rad": _number_or_text(_text(joint, "orientation")),
                },
            ]
        joints.append(
            {
                "id": identifier,
                "kind": _local_name(joint),
                "parent_frame": _text(joint, "socket_parent_frame") or legacy_parent,
                "child_frame": _text(joint, "socket_child_frame") or _text(joint, "child_body") or owning_body,
                "coordinates": coordinates,
                "frames": _joint_frames(joint) or legacy_frames,
                "motion_axes": _joint_motion_axes(joint),
                "legacy_opensim3": owning_body is not None,
                "source_xml": _source_xml(joint),
            }
        )

    muscles: list[dict[str, Any]] = []
    force_objects = next((item for item in force_set if _local_name(item) == "objects"), None)
    for force in list(force_objects) if force_objects is not None else []:
        kind = _local_name(force)
        if "muscle" not in kind.lower():
            continue
        identifier = force.get("name")
        if not identifier:
            continue
        path_points: list[dict[str, Any]] = []
        for point in force.iter():
            point_kind = _local_name(point)
            if "PathPoint" not in point_kind:
                continue
            point_id = point.get("name")
            if point_id:
                path_points.append(
                    {
                        "id": point_id,
                        "kind": point_kind,
                        "parent_frame": _text(point, "socket_parent_frame") or _text(point, "body"),
                        "location_m": _number_or_text(_text(point, "location")),
                        "parameters": _opensim_direct_leaf_properties(
                            point,
                            ("socket_parent_frame", "body", "location"),
                        ),
                        "source_xml": _source_xml(point),
                    }
                )
        path_wraps = []
        for path_wrap in force.iter():
            if _local_name(path_wrap) != "PathWrap":
                continue
            path_wraps.append(
                {
                    "id": path_wrap.get("name"),
                    "kind": _local_name(path_wrap),
                    "wrap_object": _text(path_wrap, "socket_wrap_object") or _text(path_wrap, "wrap_object"),
                    "range": _number_or_text(_text(path_wrap, "range")),
                    "method": _text(path_wrap, "method"),
                    "parameters": _opensim_direct_leaf_properties(
                        path_wrap,
                        ("socket_wrap_object", "wrap_object", "range", "method"),
                    ),
                    "source_xml": _source_xml(path_wrap),
                }
            )
        muscles.append(
            {
                "id": identifier,
                "kind": kind,
                "parameters": _opensim_direct_leaf_properties(force),
                "curves": {
                    _local_name(curve): {
                        "kind": _local_name(curve),
                        "parameters": _opensim_direct_leaf_properties(curve),
                        "source_xml": _source_xml(curve),
                    }
                    for curve in force
                    if _local_name(curve).endswith("Curve")
                },
                "path_points": path_points,
                "path_wraps": path_wraps,
                "source_xml": _source_xml(force),
            }
        )

    return {
        "source_id": source_id,
        "source_file": path.name,
        "source_sha256": sha256(path),
        "opensim_document_version": root.get("Version"),
        "model_id": model.get("name"),
        "gravity_m_s2": _number_or_text(_text(model, "gravity")),
        "bodies": bodies,
        "joints": joints,
        "muscles": muscles,
        "wrap_objects": _wrap_objects(model),
    }


def _vector3(value: Any, context: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(entry, float) and math.isfinite(entry) for entry in value)
    ):
        raise ImportError(f"{context} must be a finite OpenSim three-vector")
    return value


def _matrix_from_body_fixed_xyz(orientation: list[float]) -> list[list[float]]:
    """OpenSim body-fixed XYZ and URDF RPY use the same Rz*Ry*Rx algebra."""
    roll, pitch, yaw = orientation
    cosine_roll, sine_roll = math.cos(roll), math.sin(roll)
    cosine_pitch, sine_pitch = math.cos(pitch), math.sin(pitch)
    cosine_yaw, sine_yaw = math.cos(yaw), math.sin(yaw)
    return [
        [
            cosine_yaw * cosine_pitch,
            cosine_yaw * sine_pitch * sine_roll - sine_yaw * cosine_roll,
            cosine_yaw * sine_pitch * cosine_roll + sine_yaw * sine_roll,
        ],
        [
            sine_yaw * cosine_pitch,
            sine_yaw * sine_pitch * sine_roll + cosine_yaw * cosine_roll,
            sine_yaw * sine_pitch * cosine_roll - cosine_yaw * sine_roll,
        ],
        [
            -sine_pitch,
            cosine_pitch * sine_roll,
            cosine_pitch * cosine_roll,
        ],
    ]


def _matrix_transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _matrix_product(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    return [
        [sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _matrix_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][index] * vector[index] for index in range(3)) for row in range(3)]


def _rpy_from_matrix(matrix: list[list[float]]) -> list[float]:
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) < 1.0e-8:
        return [0.0, pitch, math.atan2(-matrix[0][1], matrix[1][1])]
    return [
        math.atan2(matrix[2][1], matrix[2][2]),
        pitch,
        math.atan2(matrix[1][0], matrix[0][0]),
    ]


def _format_vector(values: Iterable[float]) -> str:
    return " ".join(format(value, ".17g") for value in values)


def _body_id_from_frame(frame: dict[str, Any], context: str) -> str:
    parent = frame.get("parent_frame")
    if not isinstance(parent, str) or not parent.startswith("/bodyset/"):
        raise ImportError(f"{context} is not attached to an OpenSim body frame")
    return parent.rsplit("/", 1)[-1]


def _preview_joint_frames(joint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    frames = {frame["id"]: frame for frame in joint["frames"]}
    parent_id = joint.get("parent_frame")
    child_id = joint.get("child_frame")
    if not isinstance(parent_id, str) or not isinstance(child_id, str):
        raise ImportError(f"OpenSim preview joint {joint['id']} has unresolved frame sockets")
    parent = frames.get(parent_id.rsplit("/", 1)[-1])
    child = frames.get(child_id.rsplit("/", 1)[-1])
    if parent is None or child is None:
        raise ImportError(f"OpenSim preview joint {joint['id']} omits resolved PhysicalOffsetFrames")
    return parent, child


def _append_urdf_link(root: ET.Element, body: dict[str, Any]) -> None:
    mass = body.get("mass_kg")
    centre = body.get("mass_center_m")
    inertia = body.get("inertia_kg_m2")
    if not isinstance(mass, float) or not math.isfinite(mass) or mass <= 0.0:
        raise ImportError(f"OpenSim preview body {body['id']} has no positive mass")
    centre = _vector3(centre, f"OpenSim preview body {body['id']} mass centre")
    if not isinstance(inertia, dict) or set(inertia) != {"xx", "yy", "zz", "xy", "xz", "yz"}:
        raise ImportError(f"OpenSim preview body {body['id']} has no complete inertia tensor")
    link = ET.SubElement(root, "link", {"name": body["id"]})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": _format_vector(centre), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": format(mass, ".17g")})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": format(inertia["xx"], ".17g"),
            "ixy": format(inertia["xy"], ".17g"),
            "ixz": format(inertia["xz"], ".17g"),
            "iyy": format(inertia["yy"], ".17g"),
            "iyz": format(inertia["yz"], ".17g"),
            "izz": format(inertia["zz"], ".17g"),
        },
    )


def build_rajagopal_distal_pin_preview(
    model: dict[str, Any], side: str
) -> tuple[str, dict[str, Any]]:
    """Create an explicitly limited native-cookable distal-leg URDF preview.

    It represents only the three source PinJoints for a single leg.  The root
    tibia is floating, no collision proxy is invented, and no muscle is
    lowered.  That makes the artifact useful for Numi's imported-URDF compiler
    while keeping its scope distinct from a source-faithful Human RobotPack.
    """
    suffix = {"right": "r", "left": "l"}.get(side)
    if suffix is None:
        raise ImportError("Rajagopal preview side must be 'right' or 'left'")
    joint_ids = [f"ankle_{suffix}", f"subtalar_{suffix}", f"mtp_{suffix}"]
    joints_by_id = {joint["id"]: joint for joint in model["joints"]}
    selected = [joints_by_id.get(identifier) for identifier in joint_ids]
    if any(joint is None for joint in selected):
        missing = [identifier for identifier, joint in zip(joint_ids, selected) if joint is None]
        raise ImportError("Rajagopal preview is missing source joints: " + ", ".join(missing))
    if any(joint["kind"] != "PinJoint" for joint in selected if joint is not None):
        raise ImportError("Rajagopal distal preview only accepts exact source PinJoints")
    source_joints = [joint for joint in selected if joint is not None]
    frames = [_preview_joint_frames(joint) for joint in source_joints]
    body_ids = [_body_id_from_frame(frames[0][0], f"{joint_ids[0]} parent")]
    for joint, (_, child_frame) in zip(source_joints, frames):
        body_ids.append(_body_id_from_frame(child_frame, f"{joint['id']} child"))
    for joint, (_, child_frame), expected_parent in zip(
        source_joints[1:], frames[1:], body_ids[1:-1], strict=True
    ):
        parent_frame, _ = _preview_joint_frames(joint)
        actual_parent = _body_id_from_frame(parent_frame, f"{joint['id']} parent")
        if actual_parent != expected_parent:
            raise ImportError(f"Rajagopal preview joints are not one serial distal-leg chain at {joint['id']}")
    bodies_by_id = {body["id"]: body for body in model["bodies"]}
    try:
        bodies = [bodies_by_id[identifier] for identifier in body_ids]
    except KeyError as error:
        raise ImportError(f"Rajagopal preview is missing body {error.args[0]}") from error

    robot = ET.Element(
        "robot",
        {"name": f"numilab_human_rajagopal_{side}_distal_pin_preview"},
    )
    for body in bodies:
        _append_urdf_link(robot, body)
    lowering: list[dict[str, Any]] = []
    for joint, (parent_frame, child_frame) in zip(source_joints, frames):
        coordinate = joint["coordinates"]
        if len(coordinate) != 1:
            raise ImportError(f"Rajagopal preview joint {joint['id']} is not scalar")
        range_value = coordinate[0].get("range")
        if (
            not isinstance(range_value, list)
            or len(range_value) != 2
            or not all(isinstance(value, float) and math.isfinite(value) for value in range_value)
        ):
            raise ImportError(f"Rajagopal preview joint {joint['id']} has invalid source range")
        parent_translation = _vector3(
            parent_frame.get("translation_m"), f"{joint['id']} parent translation"
        )
        parent_orientation = _matrix_from_body_fixed_xyz(
            _vector3(parent_frame.get("orientation_rad"), f"{joint['id']} parent orientation")
        )
        child_translation = _vector3(
            child_frame.get("translation_m"), f"{joint['id']} child translation"
        )
        if any(abs(value) > 1.0e-10 for value in child_translation):
            raise ImportError(
                f"Rajagopal preview cannot preserve nonzero child-frame translation at {joint['id']}"
            )
        child_orientation = _matrix_from_body_fixed_xyz(
            _vector3(child_frame.get("orientation_rad"), f"{joint['id']} child orientation")
        )
        origin_rotation = _matrix_product(parent_orientation, _matrix_transpose(child_orientation))
        axis = _matrix_vector(child_orientation, [0.0, 0.0, 1.0])
        axis_norm = math.sqrt(sum(value * value for value in axis))
        if not axis_norm > 0.0:
            raise ImportError(f"Rajagopal preview joint {joint['id']} has zero transformed axis")
        axis = [value / axis_norm for value in axis]
        urdf_joint = ET.SubElement(
            robot,
            "joint",
            {"name": joint["id"], "type": "revolute"},
        )
        ET.SubElement(urdf_joint, "parent", {"link": _body_id_from_frame(parent_frame, joint["id"])})
        ET.SubElement(urdf_joint, "child", {"link": _body_id_from_frame(child_frame, joint["id"])})
        ET.SubElement(
            urdf_joint,
            "origin",
            {"xyz": _format_vector(parent_translation), "rpy": _format_vector(_rpy_from_matrix(origin_rotation))},
        )
        ET.SubElement(urdf_joint, "axis", {"xyz": _format_vector(axis)})
        ET.SubElement(
            urdf_joint,
            "limit",
            {"lower": format(range_value[0], ".17g"), "upper": format(range_value[1], ".17g")},
        )
        lowering.append(
            {
                "source_joint": joint["id"],
                "source_coordinate": coordinate[0]["id"],
                "source_range_rad": range_value,
                "source_parent_frame": parent_frame["id"],
                "source_child_frame": child_frame["id"],
                "numi_urdf_joint": joint["id"],
                "numi_joint_kind": "revolute",
                "origin_xyz_m": parent_translation,
                "origin_rpy_rad": _rpy_from_matrix(origin_rotation),
                "axis_in_child_body_frame": axis,
            }
        )
    ET.indent(robot, space="  ")
    urdf = ET.tostring(robot, encoding="unicode") + "\n"
    all_custom = [joint["id"] for joint in model["joints"] if joint["kind"] != "PinJoint"]
    return urdf, {
        "schema": "numi.human.rajagopal-distal-pin-preview.v1",
        "source": {
            "id": model["source_id"],
            "file": model["source_file"],
            "sha256": model["source_sha256"],
            "model_id": model["model_id"],
        },
        "side": side,
        "urdf_name": robot.attrib["name"],
        "included_bodies": [
            {
                "id": body["id"],
                "mass_kg": body["mass_kg"],
                "mass_center_m": body["mass_center_m"],
                "inertia_kg_m2": body["inertia_kg_m2"],
            }
            for body in bodies
        ],
        "joint_lowering": lowering,
        "excluded_source_joints": all_custom,
        "excluded_source_muscles": len(model["muscles"]),
        "collision_geometry": "none; BodyParts3D geometry has not yet been source-frame registered",
        "actuation": (
            "No actuator is lowered; the bounded ABA probe applies zero generalized effort "
            "and no OpenSim Hill-type muscle-tendon lowering."
        ),
        "evidence_boundary": (
            "A source-derived distal-leg compile preview, not a complete Human RobotPack, "
            "collision-qualified limb, muscle-actuated model, or physical validation."
        ),
    }


@dataclass(frozen=True)
class Bp3dLabel:
    concept_id: str
    representation_id: str
    name: str
    hierarchy: str


def _tab_rows(path: Path, expected_columns: int) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    malformed = [row for row in rows if len(row) < expected_columns]
    if malformed:
        raise ImportError(f"{path.name} contains malformed tabular rows")
    return rows


def _labels(path: Path, hierarchy: str) -> list[Bp3dLabel]:
    return [Bp3dLabel(row[0], row[1], row[2], hierarchy) for row in _tab_rows(path, 3)]


def _edges(path: Path, hierarchy: str) -> list[dict[str, str]]:
    return [
        {
            "hierarchy": hierarchy,
            "parent_id": row[0],
            "parent_name": row[1],
            "child_id": row[2],
            "child_name": row[3],
        }
        for row in _tab_rows(path, 4)
    ]


def _element_meshes(path: Path, hierarchy: str) -> dict[str, list[dict[str, str]]]:
    """Keep BodyParts3D's concept-to-element mapping alongside its FMA trees.

    The hierarchy tables name representations with ``BP...`` identifiers, while
    the OBJ archives contain the element files named ``FJ...``.  Treating the
    representation identifier as an OBJ filename loses real geometry for most
    components, so both identifiers must remain in the intermediate artifact.
    """
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _tab_rows(path, 3):
        output[row[0]].append(
            {
                "concept_id": row[0],
                "name": row[1],
                "element_id": row[2],
                "hierarchy": hierarchy,
            }
        )
    return output


def _descendants(edges: list[dict[str, str]], roots: set[str]) -> set[str]:
    children: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        children[edge["parent_id"]].add(edge["child_id"])
    seen = set(roots)
    queue = deque(roots)
    while queue:
        parent = queue.popleft()
        for child in children[parent]:
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def _zip_members(path: Path) -> tuple[set[str], str]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = {
                Path(name).stem
                for name in archive.namelist()
                if not name.endswith("/") and Path(name).suffix.lower() == ".obj"
            }
    except zipfile.BadZipFile as error:
        raise ImportError(f"BodyParts3D archive is not a ZIP file: {path}") from error
    if not members:
        raise ImportError(f"BodyParts3D archive has no OBJ meshes: {path}")
    return members, sha256(path)


def _obj_mesh_topology(stream: Iterable[bytes]) -> dict[str, Any]:
    """Return topology facts from one OBJ member without modifying its geometry.

    BodyParts3D is a surface source.  This preflight deliberately records only
    exact source-member facts and a conservative edge-manifold candidate; it
    does not repair meshes, create tetrahedra, or infer material properties.
    """
    digest = hashlib.sha256()
    vertex_count = 0
    valid_vertex_count = 0
    face_count = 0
    triangle_count = 0
    invalid_vertex_rows = 0
    invalid_faces = 0
    invalid_face_references = 0
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    lower_bounds = [float("inf"), float("inf"), float("inf")]
    upper_bounds = [float("-inf"), float("-inf"), float("-inf")]

    for raw_line in stream:
        digest.update(raw_line)
        line = raw_line.decode("utf-8", errors="replace").split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0] == "v":
            vertex_count += 1
            if len(tokens) < 4:
                invalid_vertex_rows += 1
                continue
            try:
                position = [float(tokens[index]) for index in range(1, 4)]
            except ValueError:
                invalid_vertex_rows += 1
                continue
            valid_vertex_count += 1
            for index, coordinate in enumerate(position):
                lower_bounds[index] = min(lower_bounds[index], coordinate)
                upper_bounds[index] = max(upper_bounds[index], coordinate)
        elif tokens[0] == "f":
            if len(tokens) < 4:
                invalid_faces += 1
                continue
            indices: list[int] = []
            face_is_valid = True
            for token in tokens[1:]:
                try:
                    source_index = int(token.split("/", 1)[0])
                except ValueError:
                    invalid_face_references += 1
                    face_is_valid = False
                    continue
                if source_index == 0:
                    invalid_face_references += 1
                    face_is_valid = False
                    continue
                index = source_index - 1 if source_index > 0 else vertex_count + source_index
                if index < 0 or index >= vertex_count:
                    invalid_face_references += 1
                    face_is_valid = False
                    continue
                indices.append(index)
            if not face_is_valid or len(indices) != len(tokens) - 1:
                invalid_faces += 1
                continue
            face_count += 1
            triangle_count += len(indices) - 2
            for first, second in zip(indices, [*indices[1:], indices[0]]):
                edge_counts[tuple(sorted((first, second)))] += 1

    boundary_edges = sum(count == 1 for count in edge_counts.values())
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    closed_2_manifold_candidate = (
        face_count > 0
        and not boundary_edges
        and not nonmanifold_edges
        and not invalid_vertex_rows
        and not invalid_faces
        and not invalid_face_references
    )
    return {
        "sha256": digest.hexdigest(),
        "vertex_count": vertex_count,
        "valid_vertex_count": valid_vertex_count,
        "face_count": face_count,
        "triangulated_face_count": triangle_count,
        "edge_count": len(edge_counts),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "invalid_vertex_row_count": invalid_vertex_rows,
        "invalid_face_count": invalid_faces,
        "invalid_face_reference_count": invalid_face_references,
        "bounds": (
            {"minimum": lower_bounds, "maximum": upper_bounds}
            if valid_vertex_count
            else None
        ),
        "closed_2_manifold_candidate": closed_2_manifold_candidate,
        "topology_boundary": (
            "A closed 2-manifold candidate has no edge-count or parse defect; "
            "self-intersection, orientation, volume conversion, and material "
            "validation remain separate gates."
        ),
    }


def _topology_summary(meshes: Iterable[dict[str, Any]]) -> dict[str, int]:
    mesh_list = list(meshes)
    return {
        "mesh_count": len(mesh_list),
        "vertex_count": sum(mesh["vertex_count"] for mesh in mesh_list),
        "valid_vertex_count": sum(mesh["valid_vertex_count"] for mesh in mesh_list),
        "face_count": sum(mesh["face_count"] for mesh in mesh_list),
        "triangulated_face_count": sum(mesh["triangulated_face_count"] for mesh in mesh_list),
        "closed_2_manifold_candidates": sum(
            mesh["closed_2_manifold_candidate"] for mesh in mesh_list
        ),
        "open_or_invalid_meshes": sum(
            not mesh["closed_2_manifold_candidate"] for mesh in mesh_list
        ),
        "boundary_edge_count": sum(mesh["boundary_edge_count"] for mesh in mesh_list),
        "nonmanifold_edge_count": sum(
            mesh["nonmanifold_edge_count"] for mesh in mesh_list
        ),
        "invalid_face_reference_count": sum(
            mesh["invalid_face_reference_count"] for mesh in mesh_list
        ),
        "invalid_vertex_row_count": sum(
            mesh["invalid_vertex_row_count"] for mesh in mesh_list
        ),
        "invalid_face_count": sum(mesh["invalid_face_count"] for mesh in mesh_list),
    }


def bodyparts_geometry_preflight(sources: Path, anatomy: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint every separate BodyParts3D OBJ and report its raw topology.

    The returned records retain archive/member identity and hash.  They are an
    import preflight, not a registration or deformable-body conversion.
    """
    classes_by_element: dict[tuple[str, str], set[str]] = defaultdict(set)
    roles_by_element: dict[tuple[str, str], set[str]] = defaultdict(set)
    for component in anatomy["components"]:
        for element in component["element_meshes"]:
            if element["mesh_present"]:
                key = (component["hierarchy"], element["element_id"])
                classes_by_element[key].add(component["anatomy_class"])
                roles_by_element[key].add(component["numi_role"])

    archive_paths = {
        "is_a": sources / "isa_BP3D_4.0_obj_99.zip",
        "part_of": sources / "partof_BP3D_4.0_obj_99.zip",
    }
    archives: list[dict[str, Any]] = []
    all_meshes: list[dict[str, Any]] = []
    for hierarchy, archive_path in archive_paths.items():
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = sorted(
                    (
                        info for info in archive.infolist()
                        if not info.is_dir() and Path(info.filename).suffix.lower() == ".obj"
                    ),
                    key=lambda info: info.filename,
                )
                if not members:
                    raise ImportError(f"BodyParts3D archive has no OBJ meshes: {archive_path}")
                meshes: list[dict[str, Any]] = []
                for info in members:
                    element_id = Path(info.filename).stem
                    with archive.open(info) as stream:
                        topology = _obj_mesh_topology(stream)
                    mesh = {
                        "archive": archive_path.name,
                        "hierarchy": hierarchy,
                        "member": info.filename,
                        "element_id": element_id,
                        "uncompressed_bytes": info.file_size,
                        "anatomy_classes": sorted(
                            classes_by_element[(hierarchy, element_id)]
                        ),
                        "numi_roles": sorted(roles_by_element[(hierarchy, element_id)]),
                        **topology,
                    }
                    meshes.append(mesh)
                    all_meshes.append(mesh)
        except zipfile.BadZipFile as error:
            raise ImportError(f"BodyParts3D archive is not a ZIP file: {archive_path}") from error
        archives.append(
            {
                "file": archive_path.name,
                "hierarchy": hierarchy,
                "sha256": sha256(archive_path),
                "summary": _topology_summary(meshes),
                "meshes": meshes,
            }
        )
    classes = sorted(
        {
            anatomy_class
            for mesh in all_meshes
            for anatomy_class in mesh["anatomy_classes"]
        }
    )
    return {
        "schema": "numi.human.bodyparts3d-geometry-preflight.v1",
        "source_id": anatomy["source_id"],
        "source_version": anatomy["version"],
        "archives": archives,
        "summary": _topology_summary(all_meshes),
        "anatomy_class_summaries": {
            anatomy_class: _topology_summary(
                mesh for mesh in all_meshes if anatomy_class in mesh["anatomy_classes"]
            )
            for anatomy_class in classes
        },
        "evidence_boundary": (
            "Exact OBJ member topology only; this preflight does not establish "
            "frame registration, collision suitability, watertight volume meshes, "
            "material calibration, or validated physical behavior."
        ),
    }


def _bodyparts_obj_member(
    sources: Path, archive_kind: str, member_id: str
) -> tuple[Path, str, bytes]:
    """Read one exact BodyParts3D OBJ member without extracting its archive."""
    archive_names = {
        "is_a": "isa_BP3D_4.0_obj_99.zip",
        "part_of": "partof_BP3D_4.0_obj_99.zip",
    }
    try:
        archive_name = archive_names[archive_kind]
    except KeyError as error:
        raise ImportError(f"unknown BodyParts3D archive {archive_kind!r}") from error
    if not re.fullmatch(r"FJ[0-9]+M?", member_id):
        raise ImportError(f"BodyParts3D member identity is invalid: {member_id!r}")
    archive_path = sources / archive_name
    try:
        with zipfile.ZipFile(archive_path) as archive:
            candidates = [
                info for info in archive.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.lower() == ".obj"
                and Path(info.filename).stem == member_id
            ]
            if len(candidates) != 1:
                raise ImportError(
                    f"BodyParts3D archive {archive_name} has {len(candidates)} OBJ members named "
                    f"{member_id}"
                )
            return archive_path, candidates[0].filename, archive.read(candidates[0])
    except zipfile.BadZipFile as error:
        raise ImportError(f"BodyParts3D archive is not a ZIP file: {archive_path}") from error


def _bodyparts_obj_geometry(
    obj: bytes, source_name: str
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, int, int]],
    list[tuple[float, float, float]] | None,
]:
    """Parse BodyParts3D geometry, retaining compatible authored OBJ normals."""
    vertices: list[tuple[float, float, float]] = []
    source_normals: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    vertex_normal_indices: list[int] = []
    complete_authored_normals = True
    try:
        lines = obj.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ImportError(f"BodyParts3D OBJ is not UTF-8 text: {source_name}") from error
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v":
            if len(fields) < 4:
                raise ImportError(f"BodyParts3D OBJ vertex is truncated: {source_name}:{line_number}")
            try:
                position = tuple(float(value) for value in fields[1:4])
            except ValueError as error:
                raise ImportError(
                    f"BodyParts3D OBJ vertex is non-numeric: {source_name}:{line_number}"
                ) from error
            if not all(math.isfinite(value) for value in position):
                raise ImportError(f"BodyParts3D OBJ vertex is non-finite: {source_name}:{line_number}")
            vertices.append(position)
            vertex_normal_indices.append(-1)
        elif fields[0] == "vn":
            if len(fields) < 4:
                raise ImportError(f"BodyParts3D OBJ normal is truncated: {source_name}:{line_number}")
            try:
                normal = tuple(float(value) for value in fields[1:4])
            except ValueError as error:
                raise ImportError(
                    f"BodyParts3D OBJ normal is non-numeric: {source_name}:{line_number}"
                ) from error
            length = math.sqrt(sum(value * value for value in normal))
            if not all(math.isfinite(value) for value in normal) or length == 0.0:
                raise ImportError(f"BodyParts3D OBJ normal is invalid: {source_name}:{line_number}")
            source_normals.append(tuple(value / length for value in normal))
        elif fields[0] == "f":
            if len(fields) < 4:
                raise ImportError(f"BodyParts3D OBJ face is truncated: {source_name}:{line_number}")
            face: list[tuple[int, int | None]] = []
            for field in fields[1:]:
                components = field.split("/")
                index_text = components[0]
                try:
                    index = int(index_text)
                except ValueError as error:
                    raise ImportError(
                        f"BodyParts3D OBJ face index is invalid: {source_name}:{line_number}"
                    ) from error
                index = index - 1 if index > 0 else len(vertices) + index
                if index < 0 or index >= len(vertices):
                    raise ImportError(
                        f"BodyParts3D OBJ face index is outside its vertex set: "
                        f"{source_name}:{line_number}"
                    )
                normal_index: int | None = None
                if len(components) >= 3 and components[2]:
                    try:
                        normal_index = int(components[2])
                    except ValueError as error:
                        raise ImportError(
                            f"BodyParts3D OBJ normal index is invalid: {source_name}:{line_number}"
                        ) from error
                    normal_index = (
                        normal_index - 1
                        if normal_index > 0
                        else len(source_normals) + normal_index
                    )
                    if normal_index < 0 or normal_index >= len(source_normals):
                        raise ImportError(
                            f"BodyParts3D OBJ normal index is outside its normal set: "
                            f"{source_name}:{line_number}"
                        )
                else:
                    complete_authored_normals = False
                face.append((index, normal_index))
            for index in range(1, len(face) - 1):
                triangle = (face[0], face[index], face[index + 1])
                triangles.append(tuple(point[0] for point in triangle))
                for vertex_index, normal_index in triangle:
                    if normal_index is None:
                        complete_authored_normals = False
                        continue
                    previous = vertex_normal_indices[vertex_index]
                    if previous not in (-1, normal_index):
                        complete_authored_normals = False
                    else:
                        vertex_normal_indices[vertex_index] = normal_index
    if not vertices or not triangles:
        raise ImportError(f"BodyParts3D OBJ has no drawable triangles: {source_name}")
    normals = (
        [source_normals[index] for index in vertex_normal_indices]
        if complete_authored_normals and source_normals and all(index >= 0 for index in vertex_normal_indices)
        else None
    )
    return vertices, triangles, normals


def _bodyparts_obj_triangles(
    obj: bytes, source_name: str
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Parse the position/triangle subset needed by physics-facing import stages."""
    vertices, triangles, _ = _bodyparts_obj_geometry(obj, source_name)
    return vertices, triangles


_NUMI_HUMAN_PECTORAL_FASCIA_MAGIC = b"NHFASC1\0"
_NUMI_HUMAN_PECTORAL_FASCIA_ABI = 1
_NUMI_HUMAN_PECTORAL_FASCIA_MEMBERS = (
    ("FJ1446", "abdominal part of right pectoralis major", "PECM3"),
    ("FJ1447", "clavicular part of right pectoralis major", "PECM1"),
    ("FJ1464", "sternocostal part of right pectoralis major", "PECM2"),
    ("FJ1446M", "abdominal part of left pectoralis major", "PECM3_l"),
    ("FJ1447M", "clavicular part of left pectoralis major", "PECM1_l"),
    ("FJ1464M", "sternocostal part of left pectoralis major", "PECM2_l"),
)


def _signed_tetrahedron_volume(
    points: list[tuple[float, float, float]], tetrahedron: tuple[int, int, int, int],
) -> float:
    a, b, c, d = (points[index] for index in tetrahedron)
    ab = tuple(b[index] - a[index] for index in range(3))
    ac = tuple(c[index] - a[index] for index in range(3))
    ad = tuple(d[index] - a[index] for index in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return sum(cross[index] * ad[index] for index in range(3)) / 6.0


def bodyparts_pectoralis_fascia_payload(
    sources: Path,
    myosim_artifact: Path,
    output: Path,
    thickness_m: float = 0.0006,
    muscle_load_fraction: float = 0.10,
) -> dict[str, Any]:
    """Compile a bounded, explicit pectoral-fascia FEM fallback.

    BodyParts3D 4.0 has no separate pectoral-fascia member.  We therefore keep
    only the anterior-facing sheet of each exact pectoralis-major OBJ and
    extrude that source topology posteriorly by a declared thickness.  This is
    a mechanics input with generated connectivity, not source-authored fascia
    geometry and not a clinical segmentation.
    """
    if not math.isfinite(thickness_m) or not 0.0002 <= thickness_m <= 0.0012:
        raise ImportError("pectoralis fascia thickness must be in [0.2, 1.2] mm")
    if not math.isfinite(muscle_load_fraction) or not 0.0 < muscle_load_fraction <= 0.25:
        raise ImportError("pectoralis fascia muscle-load fraction must be in (0, 0.25]")
    sources = sources.resolve()
    myosim_artifact = myosim_artifact.resolve()
    myosim_manifest_path = myosim_artifact / "myosim-fullbody-reference.manifest.json"
    myosim_manifest = read_json(myosim_manifest_path)
    muscles = myosim_manifest.get("muscles")
    if not isinstance(muscles, list):
        raise ImportError("MyoSim full-body manifest has no muscle table")
    actuator_by_name = {
        row.get("name"): row.get("source_actuator_index")
        for row in muscles if isinstance(row, dict)
    }
    stable_id_by_member = {
        row["member_id"]: stable_id
        for stable_id, row in enumerate(_bodyparts_myosim_surface_specifications(), start=1)
    }
    archive_path = sources / "isa_BP3D_4.0_obj_99.zip"
    if not archive_path.is_file():
        raise ImportError("BodyParts3D is-a OBJ archive is unavailable")

    nodes: list[tuple[float, float, float]] = []
    node_source_indices: list[int] = []
    node_regions: list[int] = []
    node_flags: list[int] = []
    tetrahedra: list[tuple[int, int, int, int, int]] = []
    regions: list[dict[str, Any]] = []
    source_members: list[dict[str, Any]] = []
    for region_index, (member_id, source_name, muscle_name) in enumerate(
        _NUMI_HUMAN_PECTORAL_FASCIA_MEMBERS
    ):
        actuator = actuator_by_name.get(muscle_name)
        if not isinstance(actuator, int) or isinstance(actuator, bool):
            raise ImportError(f"pectoralis fascia route is absent from MyoSim: {muscle_name}")
        stable_id = stable_id_by_member.get(member_id)
        if not isinstance(stable_id, int):
            raise ImportError(f"pectoralis fascia surface is absent from the native surface map: {member_id}")
        _, member, obj = _bodyparts_obj_member(sources, "is_a", member_id)
        vertices_mm, source_triangles = _bodyparts_obj_triangles(obj, member)
        centroid_y = sorted(
            sum(vertices_mm[index][1] for index in triangle) / 3.0
            for triangle in source_triangles
        )
        anterior_limit_mm = centroid_y[int(0.45 * (len(centroid_y) - 1))]
        selected_triangles: list[tuple[int, int, int]] = []
        for triangle in source_triangles:
            a, b, c = (vertices_mm[index] for index in triangle)
            ab = tuple(b[index] - a[index] for index in range(3))
            ac = tuple(c[index] - a[index] for index in range(3))
            normal = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            magnitude = math.sqrt(sum(value * value for value in normal))
            projected_double_area = abs(ab[0] * ac[2] - ab[2] * ac[0])
            if (
                sum(point[1] for point in (a, b, c)) / 3.0 <= anterior_limit_mm
                and magnitude > 1.0e-9
                and abs(normal[1]) / magnitude >= 0.35
                and projected_double_area >= 0.02
            ):
                selected_triangles.append(triangle)
        used_source_vertices = sorted({index for triangle in selected_triangles for index in triangle})
        if len(used_source_vertices) < 32 or len(selected_triangles) < 32:
            raise ImportError(f"pectoralis fascia {member_id} has insufficient anterior source topology")
        # The exact source surface remains the high-resolution presentation
        # geometry.  A bounded mechanics mesh uses the x-z convex envelope of
        # that anterior selection so interactive implicit solves do not carry
        # tens of thousands of high-aspect-ratio source triangles.  Every
        # mechanics vertex is still one exact source vertex.
        projected: dict[tuple[float, float], int] = {}
        for source_index in used_source_vertices:
            point = vertices_mm[source_index]
            key = (point[0], point[2])
            previous = projected.get(key)
            if previous is None or point[1] < vertices_mm[previous][1]:
                projected[key] = source_index
        ordered = sorted(projected)
        def cross(
            origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float],
        ) -> float:
            return ((first[0] - origin[0]) * (second[1] - origin[1]) -
                    (first[1] - origin[1]) * (second[0] - origin[0]))
        lower: list[tuple[float, float]] = []
        for point in ordered:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[float, float]] = []
        for point in reversed(ordered):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        hull_keys = lower[:-1] + upper[:-1]
        hull_source_vertices = [projected[key] for key in hull_keys]
        hull_set = set(hull_source_vertices)
        mean_x = sum(key[0] for key in hull_keys) / len(hull_keys)
        mean_z = sum(key[1] for key in hull_keys) / len(hull_keys)
        interior_candidates = [index for index in used_source_vertices if index not in hull_set]
        if len(hull_source_vertices) < 8 or not interior_candidates:
            raise ImportError(f"pectoralis fascia {member_id} has no bounded mechanics envelope")
        centre_source_vertex = min(
            interior_candidates,
            key=lambda index: ((vertices_mm[index][0] - mean_x) ** 2 +
                               (vertices_mm[index][2] - mean_z) ** 2),
        )
        mechanics_source_vertices = [*hull_source_vertices, centre_source_vertex]
        mechanics_triangles = [
            (len(hull_source_vertices), index, (index + 1) % len(hull_source_vertices))
            for index in range(len(hull_source_vertices))
        ]
        first_node = len(nodes)
        source_points_m = [
            tuple(coordinate * 0.001 for coordinate in vertices_mm[index])
            for index in mechanics_source_vertices
        ]
        ranked_x = sorted(range(len(source_points_m)), key=lambda index: abs(source_points_m[index][0]))
        band_count = max(3, math.ceil(0.20 * len(source_points_m)))
        fixed_indices = set(ranked_x[:band_count])
        load_indices = set(ranked_x[-band_count:])
        region_flags: list[int] = []
        for local, (source_index, point) in enumerate(zip(
            mechanics_source_vertices, source_points_m, strict=True
        )):
            flags = (1 if local in fixed_indices else 0) | (2 if local in load_indices else 0)
            region_flags.append(flags)
            nodes.append(point)
            node_source_indices.append(source_index)
            node_regions.append(region_index)
            node_flags.append(flags)
        layer_width = len(source_points_m)
        for source_index, point, flags in zip(
            mechanics_source_vertices, source_points_m, region_flags, strict=True
        ):
            nodes.append((point[0], point[1] + thickness_m, point[2]))
            node_source_indices.append(source_index)
            node_regions.append(region_index)
            node_flags.append(flags)
        first_tetrahedron = len(tetrahedra)
        for local_triangle in mechanics_triangles:
            a, b, c = (first_node + index for index in local_triangle)
            aa, bb, cc = a + layer_width, b + layer_width, c + layer_width
            for candidate in ((a, b, c, aa), (b, bb, c, aa), (c, bb, cc, aa)):
                volume = _signed_tetrahedron_volume(nodes, candidate)
                if abs(volume) <= 1.0e-15:
                    continue
                oriented = candidate if volume > 0.0 else (candidate[1], candidate[0], candidate[2], candidate[3])
                tetrahedra.append((*oriented, region_index))
        region_tetrahedron_count = len(tetrahedra) - first_tetrahedron
        fixed_count = 2 * sum(bool(flags & 1) for flags in region_flags)
        load_count = 2 * sum(bool(flags & 2) for flags in region_flags)
        if fixed_count < 6 or load_count < 6 or region_tetrahedron_count < 18:
            raise ImportError(f"pectoralis fascia {member_id} has incomplete anchor/load/FEM coverage")
        regions.append({
            "member_id": member_id,
            "source_name": source_name,
            "myosim_muscle": muscle_name,
            "source_actuator_index": actuator,
            "soft_tissue_stable_id": stable_id,
            "first_node": first_node,
            "node_count": 2 * layer_width,
            "first_tetrahedron": first_tetrahedron,
            "tetrahedron_count": region_tetrahedron_count,
            "fixed_node_count": fixed_count,
            "load_node_count": load_count,
            "anterior_centroid_limit_source_mm": anterior_limit_mm,
        })
        source_members.append({
            "member_id": member_id,
            "source_name": source_name,
            "member": member,
            "obj_sha256": hashlib.sha256(obj).hexdigest(),
            "source_vertex_count": len(vertices_mm),
            "source_triangle_count": len(source_triangles),
            "anterior_source_vertex_count": len(used_source_vertices),
            "mechanics_source_vertex_count": layer_width,
            "mechanics_convex_hull_vertex_count": len(hull_source_vertices),
            "selected_source_triangle_count": len(selected_triangles),
        })

    nodal_mass = [0.0] * len(nodes)
    total_volume = 0.0
    density_kg_m3 = 1000.0
    for a, b, c, d, _ in tetrahedra:
        volume = _signed_tetrahedron_volume(nodes, (a, b, c, d))
        if not math.isfinite(volume) or volume <= 0.0:
            raise ImportError("pectoralis fascia payload contains a non-positive tetrahedron")
        total_volume += volume
        share = density_kg_m3 * volume / 4.0
        for node in (a, b, c, d):
            nodal_mass[node] += share

    header = struct.pack(
        "<8s4I2f2I32s32s",
        _NUMI_HUMAN_PECTORAL_FASCIA_MAGIC,
        _NUMI_HUMAN_PECTORAL_FASCIA_ABI,
        len(regions), len(nodes), len(tetrahedra),
        thickness_m, muscle_load_fraction, 0, 0,
        bytes.fromhex(sha256(archive_path)), bytes.fromhex(sha256(myosim_manifest_path)),
    )
    region_payload = b"".join(struct.pack(
        "<8s6I",
        region["member_id"].encode("ascii").ljust(8, b"\0"),
        region["source_actuator_index"], region["first_node"], region["node_count"],
        region["first_tetrahedron"], region["tetrahedron_count"],
        region["soft_tissue_stable_id"],
    ) for region in regions)
    node_payload = b"".join(struct.pack(
        "<4f4I", *point, nodal_mass[index], node_flags[index], node_regions[index],
        node_source_indices[index], 0,
    ) for index, point in enumerate(nodes))
    tetrahedron_payload = b"".join(struct.pack("<5I", *tetrahedron) for tetrahedron in tetrahedra)
    payload = header + region_payload + node_payload + tetrahedron_payload
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "bodyparts3d-pectoralis-fascia.nhfascia"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.human.pectoralis-fascia-mechanics-payload.v1",
        "payload": {
            "file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
            "magic": "NHFASC1", "payload_abi": _NUMI_HUMAN_PECTORAL_FASCIA_ABI,
            "region_count": len(regions), "node_count": len(nodes),
            "tetrahedron_count": len(tetrahedra),
        },
        "source": {
            "bodyparts3d_archive": {"file": archive_path.name, "sha256": sha256(archive_path), "license": "CC-BY-4.0"},
            "myosim_manifest": {"file": myosim_manifest_path.name, "sha256": sha256(myosim_manifest_path)},
            "members": source_members,
            "geometry_status": "generated_bounded_thin_solid_mechanics_fallback_from_exact_anterior_pectoralis_major_source_vertex_envelope",
        },
        "mechanics": {
            "thickness_m": thickness_m, "density_kg_m3": density_kg_m3,
            "total_rest_volume_m3": total_volume, "total_mass_kg": density_kg_m3 * total_volume,
            "muscle_terminal_load_fraction": muscle_load_fraction,
            "fixed_node_flag": 1, "muscle_load_node_flag": 2,
            "regions": regions,
            "constitutive_model": "human_pectoralis_fascia_goh_uniaxial_v1",
        },
        "literature": {
            "material": {"doi": "10.1016/j.jmbbm.2025.107283", "scope": "human pectoralis-major fascia uniaxial mean fit; female surgical and cadaver cohort"},
            "thickness": {"doi": "10.1007/s00276-016-1747-8", "mean_m": 0.000612, "selected_m": thickness_m},
        },
        "runtime_binding": "The six named MyoSim pectoralis terminal loads may contribute only the declared fraction to flagged fascia nodes. MyoSim J^T remains the sole rigid generalized-force authority.",
        "evidence_boundary": "No BodyParts3D pectoral-fascia mesh exists. The mechanics envelope retains exact selected source vertices, while its convex fill, posterior thickness, tetrahedral connectivity, anchor bands, and fascia load share are generated research assumptions. The high-resolution source muscle surfaces remain presentation geometry. This is not a clinical segmentation, biaxial calibration, or validated two-way fascia-muscle-bone solve.",
    }
    write_json(output / "bodyparts3d-pectoralis-fascia.manifest.json", manifest)
    return manifest


def _bodyparts_glb(
    vertices_mm: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> tuple[bytes, dict[str, list[float]]]:
    """Write a minimal self-contained glTF 2.0 binary with generated normals."""
    normals = [[0.0, 0.0, 0.0] for _ in vertices_mm]
    for first, second, third in triangles:
        a, b, c = vertices_mm[first], vertices_mm[second], vertices_mm[third]
        edge_one = [b[index] - a[index] for index in range(3)]
        edge_two = [c[index] - a[index] for index in range(3)]
        normal = [
            edge_one[1] * edge_two[2] - edge_one[2] * edge_two[1],
            edge_one[2] * edge_two[0] - edge_one[0] * edge_two[2],
            edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0],
        ]
        for vertex in (first, second, third):
            for component in range(3):
                normals[vertex][component] += normal[component]
    position_bytes = bytearray()
    normal_bytes = bytearray()
    minimum_mm = [float("inf")] * 3
    maximum_mm = [float("-inf")] * 3
    for position, normal in zip(vertices_mm, normals, strict=True):
        for component in range(3):
            minimum_mm[component] = min(minimum_mm[component], position[component])
            maximum_mm[component] = max(maximum_mm[component], position[component])
        normal_length = math.sqrt(sum(value * value for value in normal))
        unit_normal = (
            (0.0, 0.0, 1.0)
            if normal_length == 0.0
            else tuple(value / normal_length for value in normal)
        )
        position_bytes.extend(struct.pack("<3f", *(value * 0.001 for value in position)))
        normal_bytes.extend(struct.pack("<3f", *unit_normal))
    index_bytes = bytearray()
    for triangle in triangles:
        index_bytes.extend(struct.pack("<3I", *triangle))

    binary = bytearray()
    views: list[dict[str, int]] = []
    for data, target in ((position_bytes, 34962), (normal_bytes, 34962), (index_bytes, 34963)):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(data)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
    bounds_m = {
        "minimum": [value * 0.001 for value in minimum_mm],
        "maximum": [value * 0.001 for value in maximum_mm],
    }
    document = {
        "asset": {"version": "2.0", "generator": "numilab-human bodyparts preview v1"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "bodyparts3d_source_static", "mesh": 0}],
        "meshes": [{
            "name": "bodyparts3d_source_surface",
            "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}],
        }],
        "materials": [{
            "name": "source_surface_preview",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.72, 0.47, 0.35, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.72,
            },
        }],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices_mm),
                "type": "VEC3",
                "min": bounds_m["minimum"],
                "max": bounds_m["maximum"],
            },
            {"bufferView": 1, "componentType": 5126, "count": len(vertices_mm), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(triangles) * 3, "type": "SCALAR"},
        ],
    }
    json_chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    while len(binary) % 4:
        binary.append(0)
    length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    glb = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary), b"BIN" + bytes((0,))),
            bytes(binary),
        )
    )
    return glb, {"minimum_mm": minimum_mm, "maximum_mm": maximum_mm}


def bodyparts_visual_preview(
    sources: Path,
    output: Path,
    archive_kind: str = "is_a",
    member_id: str = "FJ2810",
) -> dict[str, Any]:
    """Export one source OBJ surface as a static, non-physical GLB preview."""
    archive_path, member, obj = _bodyparts_obj_member(sources, archive_kind, member_id)
    vertices, triangles = _bodyparts_obj_triangles(obj, member)
    glb, bounds = _bodyparts_glb(vertices, triangles)
    output.mkdir(parents=True, exist_ok=True)
    glb_path = output / f"{member_id}-source-static.glb"
    glb_path.write_bytes(glb)
    manifest = {
        "schema": "numi.human.bodyparts3d-visual-preview.v1",
        "source": {
            "archive": archive_path.name,
            "archive_sha256": sha256(archive_path),
            "member": member,
            "member_sha256": hashlib.sha256(obj).hexdigest(),
            "member_id": member_id,
        },
        "geometry": {
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
            **bounds,
        },
        "preview": {
            "glb": glb_path.name,
            "glb_sha256": sha256(glb_path),
            "unit_conversion": "source millimetres to preview metres",
            "attachment": "static BodyParts3D source geometry only",
        },
        "evidence_boundary": (
            "This generated GLB is a static visual conversion of one exact BodyParts3D OBJ "
            "member. It does not establish a BodyParts3D-to-OpenSim frame registration, skin "
            "deformation, collision, material law, or a physical Human RobotPack."
        ),
    }
    write_json(output / f"{member_id}-source-static.manifest.json", manifest)
    return manifest


# These source members form a coherent right lower-leg reference in the shared
# BodyParts3D rest frame. They are deliberately a source-static anatomy bundle:
# exact geometry is useful for assessing muscle and calcaneal-tendon shape, but
# the bundle does not pretend that those surfaces are already registered to or
# driven by the MyoSim articulated tree.
_BODYPARTS_RIGHT_LOWER_LEG_ANATOMY = (
    ("FJ3365", "right femur", "bone"),
    ("FJ3381", "right patella", "bone"),
    ("FJ3387", "right tibia", "bone"),
    ("FJ3366", "right fibula", "bone"),
    ("FJ3385", "right talus", "bone"),
    ("FJ3360", "right calcaneus", "bone"),
    ("FJ1394", "right lateral head of gastrocnemius", "muscle"),
    ("FJ1397", "right medial head of gastrocnemius", "muscle"),
    ("FJ1437", "right soleus", "muscle"),
    ("FJ1439", "right tibialis anterior", "muscle"),
    ("FJ1405", "right calcaneal tendon", "tendon"),
)


# A deliberately uncluttered posterior-chain view of the source geometry.
# Tibialis anterior and the femur/patella are absent so that the actual
# calcaneal-tendon surface and its two ends are visible rather than hidden by
# otherwise valid but irrelevant structures.
_BODYPARTS_RIGHT_CALCANEAL_TENDON_CONTINUITY = (
    ("FJ3387", "right tibia", "bone"),
    ("FJ3366", "right fibula", "bone"),
    ("FJ3385", "right talus", "bone"),
    ("FJ3360", "right calcaneus", "bone"),
    ("FJ1394", "right lateral head of gastrocnemius", "muscle"),
    ("FJ1397", "right medial head of gastrocnemius", "muscle"),
    ("FJ1437", "right soleus", "muscle"),
    ("FJ1405", "right calcaneal tendon", "tendon"),
)


def _bodyparts_normals(
    vertices_mm: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> list[tuple[float, float, float]]:
    normals = [[0.0, 0.0, 0.0] for _ in vertices_mm]
    for first, second, third in triangles:
        a, b, c = vertices_mm[first], vertices_mm[second], vertices_mm[third]
        edge_one = [b[index] - a[index] for index in range(3)]
        edge_two = [c[index] - a[index] for index in range(3)]
        normal = [
            edge_one[1] * edge_two[2] - edge_one[2] * edge_two[1],
            edge_one[2] * edge_two[0] - edge_one[0] * edge_two[2],
            edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0],
        ]
        for vertex in (first, second, third):
            for component in range(3):
                normals[vertex][component] += normal[component]
    result: list[tuple[float, float, float]] = []
    for normal in normals:
        length = math.sqrt(sum(value * value for value in normal))
        result.append(
            (0.0, 0.0, 1.0)
            if length == 0.0
            else tuple(value / length for value in normal)
        )
    return result


def _bodyparts_anatomy_bundle_glb(
    surfaces: list[dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    """Write a multi-surface GLB with semantic source-layer materials."""
    if not surfaces:
        raise ImportError("BodyParts3D anatomy bundle requires at least one source surface")
    colors = {
        "bone": [0.78, 0.69, 0.53, 1.0],
        "muscle": [0.56, 0.022, 0.012, 1.0],
        "tendon": [0.94, 0.74, 0.34, 1.0],
    }
    if any(surface.get("layer") not in colors for surface in surfaces):
        raise ImportError("BodyParts3D anatomy bundle has an unsupported source layer")
    binary = bytearray()
    views: list[dict[str, int]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    minimum_mm = [float("inf")] * 3
    maximum_mm = [float("-inf")] * 3
    authored_normal_surface_count = 0

    def append_view(data: bytes, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        index = len(views)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target})
        binary.extend(data)
        return index

    for surface in surfaces:
        vertices = surface["vertices_mm"]
        triangles = surface["triangles"]
        normals = surface.get("normals")
        if normals is None:
            normals = _bodyparts_normals(vertices, triangles)
        else:
            if len(normals) != len(vertices):
                raise ImportError("BodyParts3D anatomy bundle authored normal count is invalid")
            authored_normal_surface_count += 1
        positions = bytearray()
        normal_bytes = bytearray()
        indices = bytearray()
        local_minimum = [float("inf")] * 3
        local_maximum = [float("-inf")] * 3
        for position, normal in zip(vertices, normals, strict=True):
            for component in range(3):
                local_minimum[component] = min(local_minimum[component], position[component])
                local_maximum[component] = max(local_maximum[component], position[component])
                minimum_mm[component] = min(minimum_mm[component], position[component])
                maximum_mm[component] = max(maximum_mm[component], position[component])
            positions.extend(struct.pack("<3f", *(value * 0.001 for value in position)))
            normal_bytes.extend(struct.pack("<3f", *normal))
        for triangle in triangles:
            indices.extend(struct.pack("<3I", *triangle))
        position_view = append_view(bytes(positions), 34962)
        normal_view = append_view(bytes(normal_bytes), 34962)
        index_view = append_view(bytes(indices), 34963)
        accessor_base = len(accessors)
        accessors.extend((
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": [value * 0.001 for value in local_minimum],
                "max": [value * 0.001 for value in local_maximum],
            },
            {"bufferView": normal_view, "componentType": 5126, "count": len(vertices), "type": "VEC3"},
            {"bufferView": index_view, "componentType": 5125, "count": len(triangles) * 3, "type": "SCALAR"},
        ))
        material = ("bone", "muscle", "tendon").index(surface["layer"])
        mesh_index = len(meshes)
        meshes.append({
            "name": surface["label"],
            "primitives": [{
                "attributes": {"POSITION": accessor_base, "NORMAL": accessor_base + 1},
                "indices": accessor_base + 2,
                "material": material,
            }],
        })
        nodes.append({"name": surface["label"], "mesh": mesh_index})
    while len(binary) % 4:
        binary.append(0)
    document = {
        "asset": {"version": "2.0", "generator": "numilab-human bodyparts anatomy bundle v1"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [
            {
                "name": layer,
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": colors[layer],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.64,
                },
            }
            for layer in ("bone", "muscle", "tendon")
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    glb = b"".join((
        struct.pack("<4sII", b"glTF", 2, length),
        struct.pack("<I4s", len(json_chunk), b"JSON"),
        json_chunk,
        struct.pack("<I4s", len(binary), b"BIN" + bytes((0,))),
        bytes(binary),
    ))
    return glb, {
        "minimum_mm": minimum_mm,
        "maximum_mm": maximum_mm,
        "authored_normal_surface_count": authored_normal_surface_count,
        "generated_normal_surface_count": len(surfaces) - authored_normal_surface_count,
    }


def _bodyparts_nearest_vertex_distance_mm(
    first: list[tuple[float, float, float]],
    second: list[tuple[float, float, float]],
) -> float:
    """Return an exact source-vertex separation for a small inspection bundle."""
    if not first or not second:
        raise ImportError("BodyParts3D continuity check requires nonempty source surfaces")
    squared_distance = float("inf")
    for first_point in first:
        for second_point in second:
            candidate = sum(
                (first_point[component] - second_point[component]) ** 2
                for component in range(3)
            )
            squared_distance = min(squared_distance, candidate)
    return math.sqrt(squared_distance)


def _bodyparts_anatomy_preview(
    sources: Path,
    output: Path,
    selection: tuple[tuple[str, str, str], ...],
    *,
    stem: str,
    schema: str,
    evidence_boundary: str,
    source_mesh_proximity: tuple[tuple[str, str, str], ...] = (),
) -> dict[str, Any]:
    """Export selected source surfaces without inventing an anatomical transform."""
    surfaces: list[dict[str, Any]] = []
    archive_path: Path | None = None
    for member_id, label, layer in selection:
        member_archive, member, obj = _bodyparts_obj_member(sources, "is_a", member_id)
        if archive_path is None:
            archive_path = member_archive
        elif member_archive != archive_path:
            raise ImportError("BodyParts3D anatomy bundle members must share one source archive")
        vertices, triangles, normals = _bodyparts_obj_geometry(obj, member)
        surfaces.append({
            "member_id": member_id,
            "member": member,
            "member_sha256": hashlib.sha256(obj).hexdigest(),
            "label": label,
            "layer": layer,
            "vertices_mm": vertices,
            "triangles": triangles,
            "normals": normals,
        })
    if archive_path is None:
        raise ImportError("BodyParts3D anatomy bundle has no source archive")
    glb, bounds = _bodyparts_anatomy_bundle_glb(surfaces)
    output.mkdir(parents=True, exist_ok=True)
    glb_path = output / f"{stem}-source-static.glb"
    glb_path.write_bytes(glb)
    manifest = {
        "schema": schema,
        "source": {"archive": archive_path.name, "archive_sha256": sha256(archive_path)},
        "surfaces": [
            {
                "member_id": surface["member_id"],
                "member": surface["member"],
                "member_sha256": surface["member_sha256"],
                "label": surface["label"],
                "layer": surface["layer"],
                "vertex_count": len(surface["vertices_mm"]),
                "triangle_count": len(surface["triangles"]),
                "normal_source": (
                    "authored_obj_vertex_normals"
                    if surface["normals"] is not None
                    else "generated_from_source_triangles"
                ),
            }
            for surface in surfaces
        ],
        "geometry": {
            "surface_count": len(surfaces),
            "vertex_count": sum(len(surface["vertices_mm"]) for surface in surfaces),
            "triangle_count": sum(len(surface["triangles"]) for surface in surfaces),
            **bounds,
        },
        "preview": {
            "glb": glb_path.name,
            "glb_sha256": sha256(glb_path),
            "unit_conversion": "source millimetres to preview metres",
            "attachment": "shared BodyParts3D source rest frame only",
            "materials": "semantic preview materials for bone, muscle, and tendon source layers",
        },
        "evidence_boundary": evidence_boundary,
    }
    by_member_id = {surface["member_id"]: surface for surface in surfaces}
    if source_mesh_proximity:
        manifest["source_mesh_proximity"] = [
            {
                "relationship": relationship,
                "method": "exact nearest source-vertex separation in shared BodyParts3D rest frame",
                "distance_mm": _bodyparts_nearest_vertex_distance_mm(
                    by_member_id[first_member]["vertices_mm"],
                    by_member_id[second_member]["vertices_mm"],
                ),
                "boundary": (
                    "A small mesh-vertex separation makes the selected source surfaces "
                    "inspectable together. It is not a topological weld, tissue-attachment "
                    "certificate, MyoSim registration, or mechanical tendon constraint."
                ),
            }
            for first_member, second_member, relationship in source_mesh_proximity
        ]
    write_json(output / f"{stem}-source-static.manifest.json", manifest)
    return manifest


def bodyparts_right_lower_leg_anatomy_preview(sources: Path, output: Path) -> dict[str, Any]:
    """Export exact static BodyParts3D muscle, tendon, and bone surfaces."""
    return _bodyparts_anatomy_preview(
        sources,
        output,
        _BODYPARTS_RIGHT_LOWER_LEG_ANATOMY,
        stem="bodyparts3d-right-lower-leg-anatomy",
        schema="numi.human.bodyparts3d-right-lower-leg-anatomy-preview.v2",
        evidence_boundary=(
            "This is exact BodyParts3D source-static lower-leg geometry. It does not establish "
            "a MyoSim-body transform, anatomical attachment transfer, articulated skinning, "
            "collision/contact, tissue mechanics, or physical Human RobotPack."
        ),
    )


def bodyparts_right_calcaneal_tendon_continuity_preview(
    sources: Path, output: Path
) -> dict[str, Any]:
    """Export the source posterior lower-leg chain with visible tendon ends."""
    return _bodyparts_anatomy_preview(
        sources,
        output,
        _BODYPARTS_RIGHT_CALCANEAL_TENDON_CONTINUITY,
        stem="bodyparts3d-right-calcaneal-tendon-continuity",
        schema="numi.human.bodyparts3d-right-calcaneal-tendon-continuity-preview.v1",
        evidence_boundary=(
            "This is a focused exact BodyParts3D source-static posterior lower-leg bundle. "
            "It makes the source calcaneal-tendon surface and selected neighbouring surfaces "
            "legible, but does not establish a topological tissue weld, MyoSim-body transform, "
            "articulated skinning, collision/contact, tissue mechanics, or a physical Human RobotPack."
        ),
        source_mesh_proximity=(
            ("FJ1405", "FJ3360", "calcaneal_tendon_to_right_calcaneus"),
            ("FJ1405", "FJ1394", "calcaneal_tendon_to_right_lateral_gastrocnemius"),
            ("FJ1405", "FJ1397", "calcaneal_tendon_to_right_medial_gastrocnemius"),
            ("FJ1405", "FJ1437", "calcaneal_tendon_to_right_soleus"),
        ),
    )


def parse_bodyparts3d(sources: Path, classification_path: Path) -> dict[str, Any]:
    files = {
        "isa_labels": sources / "isa_parts_list_e.txt",
        "partof_labels": sources / "partof_parts_list_e.txt",
        "isa_edges": sources / "isa_inclusion_relation_list.txt",
        "partof_edges": sources / "partof_inclusion_relation_list.txt",
        "isa_elements": sources / "isa_element_parts.txt",
        "partof_elements": sources / "partof_element_parts.txt",
        "isa_archive": sources / "isa_BP3D_4.0_obj_99.zip",
        "partof_archive": sources / "partof_BP3D_4.0_obj_99.zip",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ImportError("BodyParts3D input is incomplete:\n" + "\n".join(missing))
    classification = read_json(classification_path)
    isa_labels = _labels(files["isa_labels"], "is_a")
    partof_labels = _labels(files["partof_labels"], "part_of")
    isa_edges = _edges(files["isa_edges"], "is_a")
    partof_edges = _edges(files["partof_edges"], "part_of")
    isa_element_meshes = _element_meshes(files["isa_elements"], "is_a")
    partof_element_meshes = _element_meshes(files["partof_elements"], "part_of")
    isa_members, isa_hash = _zip_members(files["isa_archive"])
    partof_members, partof_hash = _zip_members(files["partof_archive"])

    classes = classification["classes"]
    isa_by_class = {key: _descendants(isa_edges, set(values)) for key, values in classes.items()}
    partof_by_class = {key: _descendants(partof_edges, set(values)) for key, values in classes.items()}
    physical_roles = classification["physical_roles"]
    component_priority = classification.get("classification_priority", list(classes))
    if set(component_priority) != set(classes):
        raise ImportError("anatomy classification priority must name every anatomy class exactly once")
    components: list[dict[str, Any]] = []
    for label in [*isa_labels, *partof_labels]:
        members = isa_members if label.hierarchy == "is_a" else partof_members
        element_lookup = (
            isa_element_meshes if label.hierarchy == "is_a" else partof_element_meshes
        )
        element_meshes = [
            {
                **element,
                "mesh_present": element["element_id"] in members,
            }
            for element in element_lookup.get(label.concept_id, [])
        ]
        classes_for_tree = isa_by_class if label.hierarchy == "is_a" else partof_by_class
        anatomy_class = next(
            (name for name in component_priority if label.concept_id in classes_for_tree[name]),
            "unclassified_surface",
        )
        components.append(
            {
                "concept_id": label.concept_id,
                "representation_id": label.representation_id,
                "name": label.name,
                "hierarchy": label.hierarchy,
                "representation_mesh_present": label.representation_id in members,
                "element_meshes": element_meshes,
                "mesh_present": bool(element_meshes) and any(
                    element["mesh_present"] for element in element_meshes
                ),
                "anatomy_class": anatomy_class,
                "numi_role": physical_roles.get(anatomy_class, "manual_classification_required"),
            }
        )
    return {
        "source_id": "bodyparts3d_4",
        "version": "4.0",
        "archives": [
            {"file": files["isa_archive"].name, "sha256": isa_hash, "hierarchy": "is_a"},
            {"file": files["partof_archive"].name, "sha256": partof_hash, "hierarchy": "part_of"},
        ],
        "components": components,
        "hierarchy_edges": [*isa_edges, *partof_edges],
    }


def bodyparts_nerve_annotation(anatomy: dict[str, Any]) -> dict[str, Any]:
    """Emit an annotation-only nerve graph without inventing neural physics."""
    components = [
        component
        for component in anatomy.get("components", [])
        if component.get("anatomy_class") == "nerve_surface"
    ]
    components.sort(
        key=lambda component: (
            str(component.get("hierarchy")),
            str(component.get("concept_id")),
            str(component.get("representation_id")),
        )
    )
    concepts_by_hierarchy = {
        hierarchy: {
            component["concept_id"]
            for component in components
            if component.get("hierarchy") == hierarchy
        }
        for hierarchy in {component.get("hierarchy") for component in components}
    }
    edges = [
        edge
        for edge in anatomy.get("hierarchy_edges", [])
        if edge.get("hierarchy") in concepts_by_hierarchy
        and (
            edge.get("parent_id") in concepts_by_hierarchy[edge["hierarchy"]]
            or edge.get("child_id") in concepts_by_hierarchy[edge["hierarchy"]]
        )
    ]
    edges.sort(
        key=lambda edge: (
            str(edge.get("hierarchy")),
            str(edge.get("parent_id")),
            str(edge.get("child_id")),
        )
    )
    return {
        "schema": "numi.human.bodyparts3d-nerve-annotation.v1",
        "source": {
            "id": anatomy.get("source_id"),
            "version": anatomy.get("version"),
            "archives": anatomy.get("archives"),
        },
        "component_count": len(components),
        "hierarchy_edge_count": len(edges),
        "components": components,
        "hierarchy_edges": edges,
        "numi_role": "anatomical_geometry_only",
        "evidence_boundary": (
            "BodyParts3D nerve labels, element meshes, and source hierarchy only. "
            "This artifact has no conduction, activation, attachment, collision, or "
            "deformable-physics semantics."
        ),
    }


def _locked_file_gate(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    expected_hash = metadata.get("sha256")
    expected_bytes = metadata.get("bytes")
    result: dict[str, Any] = {
        "file": path.name,
        "role": metadata.get("role"),
        "expected_sha256": expected_hash,
        "expected_bytes": expected_bytes,
    }
    if not path.is_file():
        result["status"] = "missing"
        return result
    result["actual_bytes"] = path.stat().st_size
    if expected_bytes is not None and result["actual_bytes"] != expected_bytes:
        result["status"] = "size_mismatch"
        return result
    result["actual_sha256"] = sha256(path)
    if not isinstance(expected_hash, str) or not expected_hash:
        result["status"] = "source_lock_requires_sha256"
        return result
    result["status"] = (
        "verified" if result["actual_sha256"] == expected_hash else "sha256_mismatch"
    )
    return result


def runtime_compatibility_report(
    model: dict[str, Any], runtime_contract: dict[str, Any]
) -> dict[str, Any]:
    """Compare imported OpenSim semantics with the revision-pinned Numi ABI.

    This is deliberately a compatibility report, never an approximation pass.
    A source model that asks for a primitive the Metal runtime cannot execute
    remains blocked instead of being silently converted to joint torque or a
    linear tendon surrogate.
    """
    joint_contract = runtime_contract["opensim_joint_lowering"]
    joint_kinds = Counter(joint["kind"] for joint in model["joints"])
    unsupported_counter: Counter[str] = Counter()
    exact_locked_lowerings: list[dict[str, Any]] = []
    for joint in model["joints"]:
        kind = joint["kind"]
        contract = joint_contract.get(kind, {})
        if contract.get("status") in {
            "supported",
            "supported_bounded_fixed_root_free_motion",
            "supported_bounded_fixed_root_direct_effort",
            "supported_bounded_mobile_root_direct_effort",
        }:
            continue
        locked_lowering = contract.get("fully_locked_zero_default_lowering")
        coordinates = joint.get("coordinates", [])
        all_locked_zero = (
            isinstance(locked_lowering, dict)
            and isinstance(coordinates, list)
            and bool(coordinates)
            and all(
                coordinate.get("locked") in (True, "true", "True")
                and coordinate.get("default_value") == 0.0
                for coordinate in coordinates
            )
        )
        if all_locked_zero:
            exact_locked_lowerings.append(
                {
                    "source_joint": joint.get("id"),
                    "source_kind": kind,
                    "numi_primitive": locked_lowering["numi_primitive"],
                    "condition": locked_lowering["condition"],
                }
            )
            continue
        unsupported_counter[kind] += 1
    unsupported = dict(sorted(unsupported_counter.items()))
    unknown = {
        kind: count
        for kind, count in sorted(joint_kinds.items())
        if kind not in joint_contract
    }
    transform_functions = Counter(
        axis["function_kind"] or "unspecified"
        for joint in model["joints"]
        for axis in joint["motion_axes"]
    )
    muscle_kinds = Counter(muscle["kind"] for muscle in model["muscles"])
    muscle_curves = Counter(
        curve_kind
        for muscle in model["muscles"]
        for curve_kind in muscle.get("curves", {})
    )
    wrap_kinds = Counter(
        wrap["kind"]
        for wrap in model["wrap_objects"]
        if isinstance(wrap.get("kind"), str) and wrap["kind"]
    )
    unsupported_wrap_kinds = {
        kind: count
        for kind, count in sorted(wrap_kinds.items())
        if kind != "WrapCylinder"
    }
    massless_bodies = sorted(
        body["id"]
        for body in model.get("bodies", [])
        if isinstance(body.get("id"), str) and
        _finite_scalar(body.get("mass_kg"), f"OpenSim body {body['id']} mass") == 0.0
    )
    path_points = sum(len(muscle["path_points"]) for muscle in model["muscles"])
    path_wraps = sum(len(muscle["path_wraps"]) for muscle in model["muscles"])
    bounded_function_based = False
    bounded_admission: dict[str, Any] | None = None
    if not unsupported and any(joint["kind"] == "CustomJoint" for joint in model["joints"]):
        try:
            skeleton_ir = rajagopal_rigid_skeleton_ir(model)
            coordinate_count = sum(
                len(joint["coordinates"]) for joint in skeleton_ir["joints"]
            )
            if skeleton_ir["body_count"] > 32 or coordinate_count > 40:
                bounded_admission = {
                    "status": "outside_bounded_capacity",
                    "body_count": skeleton_ir["body_count"],
                    "coordinate_count": coordinate_count,
                    "maximum_body_count": 32,
                    "maximum_coordinate_count": 40,
                }
            else:
                bounded_function_based = True
                bounded_admission = {
                    "status": "eligible_mobile_root_direct_effort_contact",
                    "body_count": skeleton_ir["body_count"],
                    "coordinate_count": coordinate_count,
                    "maximum_body_count": 32,
                    "maximum_coordinate_count": 40,
                }
        except ImportError as error:
            bounded_admission = {
                "status": "not_a_supported_function_based_tree",
                "reason": str(error),
            }
    supported_millard = all(
        muscle["kind"] == "Millard2012EquilibriumMuscle"
        for muscle in model["muscles"]
    ) and not unsupported_wrap_kinds
    return {
        "schema": "numi.human.runtime-compatibility.v1",
        "runtime": runtime_contract["runtime"],
        "source_model": {
            "id": model["model_id"],
            "joint_kinds": dict(sorted(joint_kinds.items())),
            "transform_functions": dict(sorted(transform_functions.items())),
            "muscle_kinds": dict(sorted(muscle_kinds.items())),
            "muscle_curve_kinds": dict(sorted(muscle_curves.items())),
            "muscle_path_points": path_points,
            "muscle_path_wraps": path_wraps,
            "wrap_objects": len(model["wrap_objects"]),
            "wrap_object_kinds": dict(sorted(wrap_kinds.items())),
            "unclassified_wrap_objects": (
                len(model["wrap_objects"]) - sum(wrap_kinds.values())
            ),
        },
        "skeleton": {
            "status": (
                "blocked"
                if unsupported or massless_bodies or (
                    bounded_admission is not None and not bounded_function_based
                )
                else (
                    "compatible_bounded_mobile_root_direct_effort_contact"
                    if bounded_function_based
                    else "compatible"
                )
            ),
            "unsupported_joint_kinds": unsupported,
            "massless_source_bodies": massless_bodies,
            "massless_body_requirement": (
                "No massless source bodies require a special anchor policy."
                if not massless_bodies else
                "Each massless source body requires an explicit kinematic-anchor "
                "or zero-inertia-carrier policy before articulated lowering; "
                "the current OpenSim Core payload refuses to invent its inertia."
            ),
            "exact_locked_joint_lowerings": exact_locked_lowerings,
            "bounded_admission": bounded_admission,
            "unknown_joint_kinds": unknown,
            "requirement": (
                "All source joint semantics must lower exactly into supported "
                "Numi articulated primitives. FunctionBased admission is "
                "bounded to one direct-effort articulation in fixed-root or "
                "source-default mobile-root free motion or synthetic temporal-cone "
                "contact."
            ),
        },
        "muscle_tendon": {
            "status": (
                "compatible"
                if not model["muscles"]
                else (
                    "compatible_bounded_static_equilibrium"
                    if supported_millard
                    else "blocked"
                )
            ),
            "current_contract": runtime_contract["muscle_tendon"]["current_contract"],
            "unsupported_source_wrap_kinds": unsupported_wrap_kinds,
            "requirements": runtime_contract["muscle_tendon"][
                "source_faithful_requirements"
            ],
        },
    }


def runtime_checkout_gate(
    runtime_root: Path | None, runtime_contract: dict[str, Any]
) -> dict[str, Any]:
    """Verify that an audit inspected the exact Numi runtime revision it names."""
    runtime = runtime_contract["runtime"]
    result: dict[str, Any] = {
        "repository": runtime["repository"],
        "expected_revision": runtime["revision"],
    }
    if runtime_root is None:
        return {"status": "not_provided", **result}
    if not runtime_root.is_dir():
        return {"status": "missing", "path": str(runtime_root), **result}
    try:
        revision = subprocess.run(
            ["git", "-C", str(runtime_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "-C", str(runtime_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "status": "unverifiable",
            "path": str(runtime_root),
            "error": str(error),
            **result,
        }
    if not revision.startswith(runtime["revision"]):
        status = "revision_mismatch"
    elif changes:
        status = "dirty_checkout"
    else:
        status = "verified"
    return {
        "status": status,
        "path": str(runtime_root),
        "actual_revision": revision,
        "working_tree": "dirty" if changes else "clean",
        **result,
    }


def parse_opensim_archive(path: Path, source_id: str) -> dict[str, Any]:
    """Parse an authenticated OpenSim archive without losing member provenance."""
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = sorted(
                member for member in archive.namelist() if member.lower().endswith(".osim")
            )
            if not candidates:
                raise ImportError(f"{path.name} contains no OpenSim model")
            member = next(
                (candidate for candidate in candidates if "bimanual" in candidate.lower()),
                candidates[0],
            )
            contents = archive.read(member)
    except zipfile.BadZipFile as error:
        raise ImportError(f"{path.name} is not a valid OpenSim archive") from error
    with tempfile.NamedTemporaryFile(suffix=".osim", delete=False) as temporary:
        temporary.write(contents)
        extracted = Path(temporary.name)
    try:
        model = parse_opensim(extracted, source_id)
    finally:
        extracted.unlink(missing_ok=True)
    model["source_file"] = member
    model["source_archive"] = {
        "file": path.name,
        "sha256": sha256(path),
    }
    return model


def parse_opensim_source_input(path: Path, source_id: str) -> dict[str, Any]:
    """Parse either an authenticated archive or a provenance-pinned `.osim` file.

    The public MoBL 4.1 mirror is a single OpenSim file, whereas the original
    SimTK release is a bimanual ZIP.  Keeping that distinction at the source
    boundary lets the importer report the public unimanual variant honestly
    instead of relabelling it as the authenticated original release.
    """
    if path.suffix.lower() != ".osim":
        return parse_opensim_archive(path, source_id)
    model = parse_opensim(path, source_id)
    model["source_direct_file"] = {
        "file": path.name,
        "sha256": sha256(path),
    }
    return model


def gate_report(
    *,
    sources: Path,
    upper_archive: Path | None,
    source_lock: dict[str, Any],
    runtime_contract: dict[str, Any],
    runtime_root: Path | None = None,
    upper_public_model: Path | None = None,
) -> dict[str, Any]:
    """Report every Human v1 dependency without pretending open gates passed."""
    bodyparts = source_lock["sources"]["bodyparts3d_4"]
    bodyparts_files = [
        _locked_file_gate(sources / filename, metadata)
        for filename, metadata in bodyparts["files"].items()
    ]
    lower_metadata = source_lock["sources"]["rajagopal_lai_uhlrich_2023"]
    lower_gate = _locked_file_gate(
        sources / "RajagopalLaiUhlrich2023.osim",
        {"role": "lower_body_opensim", "sha256": lower_metadata["sha256"]},
    )
    myosim_metadata = source_lock["sources"].get("myosim_fullbody")
    myosim_gate: dict[str, Any] = {"status": "not_selected"}
    if isinstance(myosim_metadata, dict):
        storage_dir = myosim_metadata.get("storage_dir")
        archive_file = myosim_metadata.get("archive_file")
        expected_file = myosim_metadata.get("expected_file")
        if not all(isinstance(value, str) and value for value in (
            storage_dir, archive_file, expected_file,
        )):
            myosim_gate = {
                "status": "invalid_source_lock",
                "role": myosim_metadata.get("role"),
            }
        else:
            archive = sources / storage_dir / archive_file
            myosim_gate = _locked_file_gate(
                archive,
                {
                    "role": myosim_metadata.get("role"),
                    "sha256": myosim_metadata.get("archive_sha256"),
                },
            )
            myosim_gate["source_revision"] = myosim_metadata.get("revision")
            myosim_gate["license"] = myosim_metadata.get("license")
            checkout_file = sources / storage_dir / "checkout" / expected_file
            myosim_gate["expected_source_file"] = str(checkout_file)
            myosim_gate["source_checkout"] = (
                "verified" if checkout_file.is_file() else "missing"
            )

    original_upper = source_lock["sources"]["mobl_arms_upper_extremity"]
    upper_gate: dict[str, Any] = {
        "source_variant": "authenticated_bimanual_original",
        "required_file": original_upper["release_file"],
        "terms": original_upper["license"],
        "status": "missing_authenticated_archive",
    }
    upper_source: Path | None = None
    upper_source_id: str | None = None
    if upper_archive is not None:
        upper_gate["path"] = str(upper_archive)
        if upper_archive.is_file():
            upper_gate["actual_sha256"] = sha256(upper_archive)
            try:
                with zipfile.ZipFile(upper_archive) as archive:
                    osim_members = sorted(
                        member for member in archive.namelist() if member.lower().endswith(".osim")
                    )
            except zipfile.BadZipFile:
                upper_gate["status"] = "invalid_archive"
            else:
                upper_gate["osim_members"] = osim_members
                upper_gate["status"] = (
                    "ready_for_import" if osim_members else "missing_osim_model"
                )
                if osim_members:
                    upper_source = upper_archive
                    upper_source_id = "mobl_arms_upper_extremity"
    elif upper_public_model is not None:
        public_upper = source_lock["sources"].get("mobl_arms_ceinms_41_public_mirror")
        upper_gate = {
            "source_variant": "public_unimanual_mirror",
            "terms": (
                public_upper.get("license") if isinstance(public_upper, dict)
                else "missing public MoBL source-lock entry"
            ),
            "status": "missing_public_model",
        }
        if not isinstance(public_upper, dict):
            upper_gate["status"] = "invalid_source_lock"
        else:
            model_file = public_upper.get("model_file")
            expected_sha = public_upper.get("sha256")
            upper_gate["required_file"] = model_file
            upper_gate["expected_sha256"] = expected_sha
            upper_gate["path"] = str(upper_public_model)
            if upper_public_model.is_file():
                actual_sha = sha256(upper_public_model)
                upper_gate["actual_sha256"] = actual_sha
                if actual_sha != expected_sha:
                    upper_gate["status"] = "sha256_mismatch"
                elif upper_public_model.name != model_file:
                    upper_gate["status"] = "unexpected_model_filename"
                else:
                    upper_gate["status"] = "ready_for_import_public_mirror"
                    upper_source = upper_public_model
                    upper_source_id = "mobl_arms_ceinms_41_public_mirror"

    bodyparts_ready = all(item["status"] == "verified" for item in bodyparts_files)
    myosim_source_ready = (
        myosim_gate.get("status") == "verified" and
        myosim_gate.get("source_checkout") == "verified"
    )
    source_import_ready = (
        bodyparts_ready
        and lower_gate["status"] == "verified"
        and upper_gate["status"] == "ready_for_import"
    )
    public_upper_source_ready = (
        bodyparts_ready
        and lower_gate["status"] == "verified"
        and upper_gate["status"] == "ready_for_import_public_mirror"
    )
    runtime_compatibility: dict[str, Any] = {
        "schema": "numi.human.runtime-compatibility-report.v1",
        "lower_body_and_pelvis": {
            "status": "lower_source_not_verified",
        },
        "upper_extremities": {
            "status": upper_gate["status"],
        },
    }
    if lower_gate["status"] == "verified":
        runtime_compatibility["lower_body_and_pelvis"] = runtime_compatibility_report(
            parse_opensim(
                sources / "RajagopalLaiUhlrich2023.osim",
                "rajagopal_lai_uhlrich_2023",
            ),
            runtime_contract,
        )
    if upper_source is not None and upper_source_id is not None:
        runtime_compatibility["upper_extremities"] = runtime_compatibility_report(
            parse_opensim_source_input(upper_source, upper_source_id),
            runtime_contract,
        )
    return {
        "schema": "numi.human.gate-report.v1",
        "source_artifacts": {
            "bodyparts3d_4": bodyparts_files,
            "rajagopal_lai_uhlrich_2023": lower_gate,
            "mobl_arms_upper_extremity": upper_gate,
            "myosim_fullbody": myosim_gate,
        },
        "runtime_compatibility": runtime_compatibility,
        "runtime_checkout": runtime_checkout_gate(runtime_root, runtime_contract),
        "mechanics_execution": {
            "status": "qualified_fullbody_static_device_reference",
            "runtime_revision": runtime_contract["runtime"]["revision"],
            "active_fullbody": runtime_contract["myosim_fullbody"],
            "comparative_lower_body": {
                "contract": "One FunctionBased articulation executes resident q/v/effort state on device with either a fixed root or a source-default-preserving 7-q/6-v physical pelvis root. Source Millard static-equilibrium forces are reduced into that same effort arena before every microstep. Excitation is supplied either by an explicit per-control stream or by a fail-closed complete, source-ordered native-task action surface mapped from signed action to normalized excitation and advanced by device first-order activation dynamics.",
                "remaining_evidence": runtime_contract["muscle_tendon"][
                    "source_faithful_requirements"
                ],
            },
        },
        "gates": [
            {
                "id": "source_faithful_import",
                "status": "ready" if source_import_ready else "blocked",
                "requirement": "The legacy BodyParts3D + Rajagopal + authenticated MoBL-ARMS manifest requires all three exact source artifacts.",
            },
            {
                "id": "public_upper_source_variant",
                "status": "ready" if public_upper_source_ready else "not_selected",
                "requirement": "The public CEINMS MoBL-ARMS 4.1 mirror is a hash-pinned, non-commercial unimanual source variant. It does not replace the authenticated bimanual archive.",
            },
            {
                "id": "free_human_foundation_source_stack",
                "status": (
                    "ready_for_source_import_unimanual_upper_variant"
                    if public_upper_source_ready else "not_selected"
                ),
                "requirement": "The free Human foundation is the verified BodyParts3D 4.0 geometry/hierarchy, RajagopalLaiUhlrich2023 lower-body mechanics, and public CEINMS MoBL-ARMS 4.1 upper-extremity source. Its upper source is a non-commercial unimanual variant; this source-import gate does not claim the authenticated bimanual archive, bilateral upper-body completion, frame registration, or physical qualification.",
            },
            {
                "id": "active_myosim_fullbody_mechanics",
                "status": (
                    "qualified_static_device_reference"
                    if myosim_source_ready else "blocked"
                ),
                "requirement": "The active full-body mechanical route requires the pinned Apache-2.0 MyoSim source archive and extracted composition source. Its 416 default-state routes and static actuator forces have Apple-GPU parity; device-resident J^T scatter and forward dynamics remain separate work.",
            },
            {
                "id": "skeleton_robotpack_lowering",
                "status": "qualified_bounded_device",
                "requirement": "Qualified: one FunctionBased source tree advances with direct effort and source-mass streamed temporal-cone responses in a synthetic device contact probe, including a source-default-preserving mobile pelvis root. The mobile reduction is not arbitrary ground_pelvis Euler-coordinate equivalence. BodyParts3D frame registration, anatomical collision/material calibration, replay, and broader RobotPack admission remain separate source or calibration work.",
            },
            {
                "id": "muscle_tendon_lowering",
                "status": "qualified_bounded_device",
                "requirement": "Qualified: source Millard curves, static fiber-tendon equilibrium, finite-cylinder paths/wraps, per-control or fail-closed complete native-task first-order activation, and per-muscle forces execute and reduce into MetalWorld's resident effort arena for fixed and source-default mobile roots. Dynamic fibre/tendon state, OpenSim equivalence, registered anatomical contact, replay, and held-out force/moment-arm validation remain evidence gates.",
            },
            {
                "id": "skin_shell",
                "status": "blocked",
                "requirement": "A repaired shell topology, thickness, and cited skin material calibration.",
            },
            {
                "id": "organ_fem_or_mpm",
                "status": "blocked",
                "requirement": "Watertight organ volume meshes and organ-specific constitutive calibration.",
            },
            {
                "id": "ligament_and_tendon_tensile",
                "status": "blocked",
                "requirement": "Registered attachment paths plus nonlinear ligament/tendon parameters and calibration.",
            },
            {
                "id": "cartilage_contact",
                "status": "blocked",
                "requirement": "Cartilage thickness, compliant-contact law, and contact validation.",
            },
            {
                "id": "vessel_tube",
                "status": "blocked",
                "requirement": "Centreline/tube conversion, vessel wall model, and explicit fluid/solid scope.",
            },
            {
                "id": "nerve_annotation",
                "status": "ready" if bodyparts_ready else "blocked",
                "requirement": "A verified BodyParts3D source import; geometry alone must remain annotation-only.",
            },
            {
                "id": "native_physics_evidence",
                "status": "qualified_bounded_device",
                "requirement": "Apple-GPU evidence qualifies bounded source mechanics and a synthetic source-contact response probe; it does not qualify registered anatomical contact, deformable anatomy, source authentication, or tissue calibration.",
            },
        ],
        "evidence_boundary": "This report is source and integration status, not medical or physical validation.",
    }


def _normalized(value: str) -> str:
    value = value.lower().replace("_", " ").replace("/", " ")
    value = re.sub(r"\b([rl])\b", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def registration_work_items(
    anatomy: dict[str, Any], mechanics: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    components = anatomy["components"]
    output: list[dict[str, Any]] = []
    for model in mechanics:
        for body in model["bodies"]:
            body_name = _normalized(body["id"])
            candidates = [
                {"concept_id": item["concept_id"], "name": item["name"], "representation_id": item["representation_id"]}
                for item in components
                if item["mesh_present"]
                and item["anatomy_class"] == "bone"
                and (body_name in _normalized(item["name"]) or _normalized(item["name"]) in body_name)
            ][:12]
            output.append(
                {
                    "mechanics_source": model["source_id"],
                    "body_id": body["id"],
                    "status": "unresolved_registration",
                    "candidate_geometry": candidates,
                    "required_checks": ["frame_transform", "scale", "origin", "mesh_to_collision_proxy"],
                }
            )
    return output


def build_manifest(
    *,
    sources: Path,
    upper_archive: Path | None,
    classification_path: Path,
    target_mapping_path: Path,
    source_lock: dict[str, Any],
    upper_public_model: Path | None = None,
) -> dict[str, Any]:
    bodyparts_lock = source_lock["sources"]["bodyparts3d_4"]["files"]
    bodyparts_gates = [
        _locked_file_gate(sources / filename, metadata)
        for filename, metadata in bodyparts_lock.items()
    ]
    unverified_bodyparts = [
        gate["file"] for gate in bodyparts_gates if gate["status"] != "verified"
    ]
    if unverified_bodyparts:
        raise ImportError(
            "BodyParts3D inputs are not all provenance-verified: "
            + ", ".join(unverified_bodyparts)
        )
    anatomy = parse_bodyparts3d(sources, classification_path)
    geometry_preflight = bodyparts_geometry_preflight(sources, anatomy)
    lower_path = sources / "RajagopalLaiUhlrich2023.osim"
    lower = parse_opensim(lower_path, "rajagopal_lai_uhlrich_2023")
    expected_lower_hash = source_lock["sources"]["rajagopal_lai_uhlrich_2023"]["sha256"]
    if lower["source_sha256"] != expected_lower_hash:
        raise ImportError("RajagopalLaiUhlrich2023.osim SHA-256 differs from sources.lock.json")

    if (upper_archive is None) == (upper_public_model is None):
        raise ImportError("provide exactly one of the authenticated MoBL archive or public MoBL 4.1 model")
    if upper_archive is not None:
        upper = parse_opensim_archive(upper_archive, "mobl_arms_upper_extremity")
        upper_terms = source_lock["sources"]["mobl_arms_upper_extremity"]["license"]
        upper_provenance = {
            "variant": "authenticated_bimanual_original",
            "archive_file": upper_archive.name,
            "archive_sha256": sha256(upper_archive),
        }
    else:
        assert upper_public_model is not None
        public_upper = source_lock["sources"].get("mobl_arms_ceinms_41_public_mirror")
        if not isinstance(public_upper, dict):
            raise ImportError("sources.lock.json has no public MoBL-ARMS 4.1 mirror entry")
        if not upper_public_model.is_file():
            raise ImportError(
                f"public MoBL-ARMS 4.1 model does not exist: {upper_public_model}"
            )
        if upper_public_model.name != public_upper.get("model_file"):
            raise ImportError("public MoBL-ARMS 4.1 model filename differs from sources.lock.json")
        actual_sha = sha256(upper_public_model)
        if actual_sha != public_upper.get("sha256"):
            raise ImportError("public MoBL-ARMS 4.1 model SHA-256 differs from sources.lock.json")
        upper = parse_opensim(upper_public_model, "mobl_arms_ceinms_41_public_mirror")
        upper_terms = public_upper["license"]
        upper_provenance = {
            "variant": "public_unimanual_mirror",
            "repository": public_upper["repository"],
            "revision": public_upper["revision"],
            "model_file": upper_public_model.name,
            "model_sha256": actual_sha,
            "not_a_substitute_for": "authenticated_bimanual_original",
        }

    mechanics = [lower, upper]
    manifest = {
        "schema": "numi.human.v1",
        "revision": 1,
        "provenance": {
            "source_lock_schema": source_lock["schema"],
            "bodyparts_attribution": source_lock["sources"]["bodyparts3d_4"]["attribution"],
            "upper_extremity_terms": upper_terms,
            "upper_source": upper_provenance,
        },
        "anatomy": anatomy,
        "geometry_preflight": geometry_preflight,
        "musculoskeletal_mechanics": {
            "lower_body_and_pelvis": lower,
            "upper_extremities": upper,
        },
        "numi_targets": read_json(target_mapping_path),
        "registration_work_items": registration_work_items(anatomy, mechanics),
        "evidence_boundary": "source-faithful import only; not a compiled Numi RobotPack or validated physics run",
    }
    return manifest


def report_for(manifest: dict[str, Any]) -> dict[str, Any]:
    anatomy = manifest["anatomy"]
    lower = manifest["musculoskeletal_mechanics"]["lower_body_and_pelvis"]
    upper = manifest["musculoskeletal_mechanics"]["upper_extremities"]
    class_counts: dict[str, int] = defaultdict(int)
    for component in anatomy["components"]:
        class_counts[component["anatomy_class"]] += 1
    return {
        "schema": "numi.human.import-report.v1",
        "source_hashes": {
            "bodyparts_archives": anatomy["archives"],
            "rajagopal": lower["source_sha256"],
            "mobl_arms": upper["source_sha256"],
        },
        "counts": {
            "anatomy_components": len(anatomy["components"]),
            "hierarchy_edges": len(anatomy["hierarchy_edges"]),
            "anatomy_classes": dict(sorted(class_counts.items())),
            "lower_bodies": len(lower["bodies"]),
            "lower_joints": len(lower["joints"]),
            "lower_muscles": len(lower["muscles"]),
            "upper_bodies": len(upper["bodies"]),
            "upper_joints": len(upper["joints"]),
            "upper_muscles": len(upper["muscles"]),
            "unresolved_geometry_registrations": len(manifest["registration_work_items"]),
        },
        "physical_status": manifest["evidence_boundary"],
    }
