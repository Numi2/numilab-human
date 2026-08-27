from __future__ import annotations

import hashlib
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
from itertools import permutations, product
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
_MYOSIM_MUSCLE_REFERENCE_MAGIC = b"NHMYO1\0\0"
_MYOSIM_MUSCLE_REFERENCE_ABI = 1
# Source-authored full-body foot support witnesses.  These are compiled
# MuJoCo capsule/ellipsoid surface points against MyoSim's own ground plane,
# not BodyParts3D collision proxies.
_MYOSIM_SUPPORT_CONTACT_MAGIC = b"NHCNT1\0\0"
_MYOSIM_SUPPORT_CONTACT_ABI = 1
_MR_MOTION_STATIC = 0
_MR_ROOT_FLOATING = 1
_MR_JOINT_PRISMATIC = 1
_MR_DOF_ROOT = 1 << 0
_MYOSIM_ROUTE_SITE = 1
_MYOSIM_ROUTE_SPHERE = 2
_MYOSIM_ROUTE_CYLINDER = 3


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
    context: str,
) -> bytes:
    return (
        struct.pack("<8I", 0, joint_index, q_index, v_index, local_dof, flags, 0, 0)
        + _pack_float4(limits, context + " limits")
        + _pack_float4([0.0, 0.0, armature, 0.0], context + " drive")
    )


def myosim_fullbody_reference_artifacts(
    exported: dict[str, Any],
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
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
    sites = exported.get("sites")
    geometries = exported.get("wrap_geometries")
    muscles = exported.get("muscles")
    if not all(isinstance(value, list) for value in (bodies, joints, sites, geometries, muscles)):
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
        muscle_manifest.append({
            "source_actuator_index": muscle.get("id"), "name": muscle.get("name"),
            "source_tendon_index": muscle.get("tendon"), "route_nodes": len(route),
            "oracle_length_m": floats[-2], "oracle_force_n_at_activation_0_5": floats[-1],
        })
    muscle_header = struct.pack(
        "<8s9I32s", _MYOSIM_MUSCLE_REFERENCE_MAGIC, _MYOSIM_MUSCLE_REFERENCE_ABI,
        len(body_records), len(muscle_records), len(site_records), len(geom_records), len(route_records),
        int(model.get("tendon_count")), 0, 0, bytes.fromhex(source_hash)
    )
    muscle_payload = b"".join([muscle_header, *site_records, *geom_records, *route_records, *muscle_records])
    expected_muscle_bytes = 76 + 16 * len(site_records) + 64 * len(geom_records) + 16 * len(route_records) + 164 * len(muscle_records)
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
    return manifest, rigid_payload, muscle_payload, support_payload


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
        # MyoSim's toes bodies provide the available articulated foot parent;
        # bind the remaining source tarsals to that side rather than pretending
        # the lower-body source contains their individual rigid joints.
        ("toes_r", "navicular bone of right foot", "FJ3308"),
        ("toes_r", "right medial cuneiform bone", "FJ3377"),
        ("toes_r", "right intermediate cuneiform bone", "FJ3370"),
        ("toes_r", "right lateral cuneiform bone", "FJ3373"),
        ("toes_r", "right cuboid bone", "FJ3364"),
        ("toes_l", "navicular bone of left foot", "FJ3307"),
        ("toes_l", "left medial cuneiform bone", "FJ3271"),
        ("toes_l", "left intermediate cuneiform bone", "FJ3264"),
        ("toes_l", "left lateral cuneiform bone", "FJ3267"),
        ("toes_l", "left cuboid bone", "FJ3258"),
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
        # MyoSim supplies one articulated toes segment per side, rather than
        # individual toe joints.  These source meshes therefore share that
        # corresponding side's live parent body.
        ("toes_r", "right first metatarsal bone", "FJ3351"),
        ("toes_r", "right second metatarsal bone", "FJ3353"),
        ("toes_r", "right third metatarsal bone", "FJ3355"),
        ("toes_r", "right fourth metatarsal bone", "FJ3357"),
        ("toes_r", "right fifth metatarsal bone", "FJ3359"),
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
        ("toes_l", "left first metatarsal bone", "FJ3241"),
        ("toes_l", "left second metatarsal bone", "FJ3244"),
        ("toes_l", "left third metatarsal bone", "FJ3247"),
        ("toes_l", "left fourth metatarsal bone", "FJ3250"),
        ("toes_l", "left fifth metatarsal bone", "FJ3253"),
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
    expected_bytes = header_bytes + 16 * site_count + 64 * wrap_count + 16 * route_count + 164 * muscle_count
    if (
        magic != _MYOSIM_MUSCLE_REFERENCE_MAGIC or abi != _MYOSIM_MUSCLE_REFERENCE_ABI
        or body_count == 0 or tendon_count == 0 or reserved0 != 0 or reserved1 != 0
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
    """Infer a constrained per-bone translation from source muscle sites.

    The base common-frame fit aligns only mesh centroids to inertial COMs,
    which is insufficient evidence for a muscle insertion.  This importer
    therefore uses the exact source site records on each already-bound link to
    refine *translation only* against the corresponding BodyParts3D triangle
    vertices.  It never changes source paths, forces, body binding, scale, or
    orientation, and remains visual-only until independently reviewed.
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
    summary: list[dict[str, Any]] = []
    for anchor in candidate["anchors"]:
        target = anchor["target"]
        registration = anchor["registration"]
        body_index = target["core_body_index"]
        sites = sites_by_body.get(body_index, [])
        source = anchor["source"]
        archive_path, member, obj = _bodyparts_obj_member(sources, source["hierarchy"], source["member_id"])
        if archive_path.name != source["archive"] or hashlib.sha256(obj).hexdigest() != source["member_sha256"]:
            raise ImportError("BodyParts3D attachment registration source mesh provenance drifted")
        vertices_mm, _ = _bodyparts_obj_triangles(obj, member)
        translation, quaternion, scale = _bodyparts_visual_local_pose(
            registration["source_obj_mm_to_core_inertial_body_m"],
            f"BodyParts3D attachment local transform for {source['member_id']}",
        )
        rotation = _myosim_matrix_from_quaternion_xyzw(quaternion)
        vertices = [
            _myosim_add(translation, [
                scale * value for value in _myosim_matrix_vector(
                    rotation, [coordinate * 0.001 for coordinate in vertex]
                )
            ])
            for vertex in vertices_mm
        ]

        def nearest(point: list[float]) -> tuple[list[float], float]:
            vertex = min(vertices, key=lambda candidate_vertex: sum(
                (point[axis] - candidate_vertex[axis]) ** 2 for axis in range(3)
            ))
            return vertex, math.sqrt(sum((point[axis] - vertex[axis]) ** 2 for axis in range(3)))

        initial_pairs = [(*nearest(point), point) for point in sites]
        initial_distances = [distance for _, distance, _ in initial_pairs if distance <= max_correspondence_distance_m]
        delta = [0.0, 0.0, 0.0]
        if len(initial_distances) >= 3:
            for _ in range(maximum_iterations):
                pairs = [(*nearest(point), point) for point in sites]
                accepted = [(vertex, point) for vertex, distance, point in pairs if distance <= max_correspondence_distance_m]
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
        final_distances = [distance for _, distance, _ in final_pairs if distance <= max_correspondence_distance_m]
        initial_median = _bodyparts_attachment_quantile(initial_distances, 0.5) if initial_distances else None
        final_median = _bodyparts_attachment_quantile(final_distances, 0.5) if final_distances else None
        apply = (
            initial_median is not None and final_median is not None
            and final_median + 0.001 < initial_median
        )
        if apply:
            local_matrix = registration["source_obj_mm_to_core_inertial_body_m"]
            for axis in range(3):
                local_matrix[axis][3] += delta[axis]
            body_rotation = _myosim_matrix_from_quaternion_xyzw(
                target["default_inertial_quaternion_world_xyzw"]
            )
            world_delta = _myosim_matrix_vector(body_rotation, delta)
            registration["default_pose_vertex_centroid_world_m"] = _myosim_add(
                registration["default_pose_vertex_centroid_world_m"], world_delta
            )
            registration["status"] = "inferred_visual_attachment_surface_refinement"
        registration["attachment_surface_refinement"] = {
            "method": "iterative_nearest_bodyparts3d_vertex_to_exact_myosim_site_translation_only",
            "maximum_correspondence_distance_m": max_correspondence_distance_m,
            "maximum_iterations": maximum_iterations,
            "source_site_count": len(sites),
            "accepted_site_count_before": len(initial_distances),
            "accepted_site_count_after": len(final_distances),
            "median_distance_before_m": initial_median,
            "p90_distance_before_m": _bodyparts_attachment_quantile(initial_distances, 0.9) if initial_distances else None,
            "median_distance_after_m": final_median,
            "p90_distance_after_m": _bodyparts_attachment_quantile(final_distances, 0.9) if final_distances else None,
            "translation_delta_core_body_m": delta,
            "applied": apply,
        }
        summary.append({
            "myosim_body": target["name"], "core_body_index": body_index,
            "applied": apply, "median_distance_before_m": initial_median,
            "median_distance_after_m": final_median,
        })
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
_BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE = 1
_BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON = 2


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
    vertices_payload: list[tuple[float, float, float, float, float, float]] = []
    indices_payload: list[int] = []
    records_payload: list[bytes] = []
    provenance_anchors: list[dict[str, Any]] = []
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
    if registration.get("status") != "provisional_visual_registration_not_admitted_to_collision_or_physics":
        raise ImportError("BodyParts3D posterior-chain visual payload requires an unmodified visual-only registration")
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
                global_vertices_m, primary_weights, bone_vertices_world_m,
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
            "the calcaneal-tendon insertion uses a source-mesh-proximity secondary-calcaneus weight lock"
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
    lock_radius_m = 0.003
    feather_radius_m = 0.015
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
            "This changes only visual two-body blend weights at a named tendon insertion; "
            "it is not a topological weld, a tendon constitutive model, force transfer, "
            "or medical attachment validation."
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
    magic, abi, body_count, muscle_count, site_count, geometry_count, route_count, _, _, _, payload_sha = struct.unpack_from(
        "<8s9I32s", payload
    )
    if magic != _MYOSIM_MUSCLE_REFERENCE_MAGIC or abi != _MYOSIM_MUSCLE_REFERENCE_ABI or payload_sha.hex() != source_sha:
        raise ImportError("MyoSim surface binding muscle payload ABI or source provenance is invalid")
    expected_bytes = header_size + 16 * site_count + 64 * geometry_count + 16 * route_count + 164 * muscle_count
    if len(payload) != expected_bytes or body_count == 0 or muscle_count == 0 or site_count == 0:
        raise ImportError("MyoSim surface binding muscle payload length is invalid")
    offset = header_size
    sites = [struct.unpack_from("<I3f", payload, offset + index * 16) for index in range(site_count)]
    offset += 16 * site_count + 64 * geometry_count
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
        first, last = routes[route_offset], routes[route_offset + count - 1]
        if first[0] != _MYOSIM_ROUTE_SITE or last[0] != _MYOSIM_ROUTE_SITE or first[1] >= len(sites) or last[1] >= len(sites):
            raise ImportError("MyoSim surface binding route has no source-site endpoints")
        primary_index, secondary_index = sites[first[1]][0], sites[last[1]][0]
        if primary_index == secondary_index or primary_index not in body_by_index or secondary_index not in body_by_index:
            raise ImportError("MyoSim surface binding route endpoint bodies are invalid")
        routes_by_muscle[name] = {
            "source_actuator_index": index,
            "source_route_node_count": count,
            "primary_body": body_by_index[primary_index]["name"],
            "secondary_body": body_by_index[secondary_index]["name"],
        }
    return bodies, routes_by_muscle, manifest


def bodyparts_myosim_fullbody_soft_tissue_visual_payload(
    sources: Path, anatomy: dict[str, Any], registration_path: Path, myosim_artifact: Path, output: Path,
) -> dict[str, Any]:
    """Package source-authored limb, shoulder, arm, hand and abdominal surfaces.

    Every ordinary row uses the first and final **authored MyoSim route sites**
    of its named muscle. The limited calcaneal-tendon row is explicitly marked
    as an anatomical shared-tendon pair because its three contributing routes
    begin on two distinct links. This is still a two-body kinematic visual
    binding, not a deformable tissue or a replacement for the MyoSim force path.
    """
    registration_file = registration_path.resolve()
    registration = read_json(registration_file)
    if registration.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2" or \
            registration.get("status") != "provisional_visual_registration_not_admitted_to_collision_or_physics":
        raise ImportError("BodyParts3D full-body tissue payload requires the unmodified v2 visual registration")
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
    registration_anchors = registration.get("anchors")
    if not isinstance(registration_anchors, list):
        raise ImportError("BodyParts3D full-body tissue payload has no visual-skeleton anchors")
    secondary_bone_sources: dict[str, dict[str, Any]] = {}
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

    vertices_payload: list[tuple[float, float, float, float, float, float, float]] = []
    indices_payload: list[int] = []
    records_payload: list[bytes] = []
    provenance: list[dict[str, Any]] = []
    for stable_id, specification in enumerate(specifications, start=1):
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
        explicit_primary, explicit_secondary = specification.get("primary_body"), specification.get("secondary_body")
        if explicit_primary is None and explicit_secondary is None:
            route_pairs = {(route["primary_body"], route["secondary_body"]) for route in matched_routes}
            if len(route_pairs) != 1:
                raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has non-uniform MyoSim endpoints")
            primary_name, secondary_name = next(iter(route_pairs))
            endpoint_source = "all_named_authored_myosim_route_endpoints"
        elif isinstance(explicit_primary, str) and isinstance(explicit_secondary, str):
            primary_name, secondary_name = explicit_primary, explicit_secondary
            endpoint_source = "explicit_anatomical_shared_tendon_pair_with_named_contributing_authored_routes"
        else:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has incomplete explicit endpoint bodies")
        primary_target, secondary_target = bodies.get(primary_name), bodies.get(secondary_name)
        if primary_target is None or secondary_target is None or primary_name == secondary_name:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has unresolved endpoint bodies")
        primary_body_index, secondary_body_index = primary_target["core_body_index"], secondary_target["core_body_index"]
        primary_position = _myosim_vector(primary_target.get("default_com_position_world_m"), f"BodyParts3D {member_id} primary position")
        secondary_position = _myosim_vector(secondary_target.get("default_com_position_world_m"), f"BodyParts3D {member_id} secondary position")
        primary_quaternion = list(primary_target.get("default_inertial_quaternion_world_xyzw", []))
        secondary_quaternion = list(secondary_target.get("default_inertial_quaternion_world_xyzw", []))
        _myosim_matrix_from_quaternion_xyzw(primary_quaternion)
        _myosim_matrix_from_quaternion_xyzw(secondary_quaternion)
        archive_path, member, obj = _bodyparts_obj_member(sources, "is_a", member_id)
        vertices_mm, triangles = _bodyparts_obj_triangles(obj, member)
        normals = _bodyparts_vertex_normals(vertices_mm, triangles, member)
        primary_transform = _bodyparts_local_registration_matrix(global_matrix, primary_position, primary_quaternion)
        secondary_transform = _bodyparts_local_registration_matrix(global_matrix, secondary_position, secondary_quaternion)
        primary_translation, primary_rotation, primary_scale = _bodyparts_visual_local_pose(primary_transform, f"BodyParts3D {member_id} primary transform")
        secondary_translation, secondary_rotation, secondary_scale = _bodyparts_visual_local_pose(secondary_transform, f"BodyParts3D {member_id} secondary transform")
        body_axis = _myosim_subtract(secondary_position, primary_position)
        squared_axis = sum(value * value for value in body_axis)
        if squared_axis <= 1.0e-10:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has coincident endpoint centres")
        global_vertices = [[sum(global_matrix[row][column] * vertex[column] for column in range(3)) + global_matrix[row][3] for row in range(3)] for vertex in vertices_mm]
        projections = [sum((vertex[axis] - primary_position[axis]) * body_axis[axis] for axis in range(3)) for vertex in global_vertices]
        minimum, maximum = min(projections), max(projections)
        if maximum - minimum <= 1.0e-6:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has no two-body blend extent")
        primary_weights = [
            max(0.0, min(1.0, (maximum - projection) / (maximum - minimum)))
            for projection in projections
        ]
        first_vertex, first_index = len(vertices_payload), len(indices_payload)
        layer_name = specification.get("layer", "muscle")
        layer = _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE if layer_name == "muscle" else _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON if layer_name == "tendon" else None
        if layer is None:
            raise ImportError(f"BodyParts3D full-body tissue surface {member_id} has an invalid layer")
        attachment_weight_lock: dict[str, Any] | None = None
        if layer == _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON:
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
            bone_vertices_world_m = [
                [
                    sum(global_matrix[row][column] * vertex[column] for column in range(3)) +
                    global_matrix[row][3]
                    for row in range(3)
                ]
                for vertex in bone_vertices_mm
            ]
            primary_weights, attachment_weight_lock = _bodyparts_secondary_attachment_weight_lock(
                global_vertices, primary_weights, bone_vertices_world_m, bone_triangles,
            )
            attachment_weight_lock.update({
                "secondary_body": secondary_name,
                "secondary_bone_member_id": bone_member_id,
                "secondary_bone_member": bone_member,
                "secondary_bone_member_sha256": hashlib.sha256(bone_obj).hexdigest(),
            })
        for vertex, normal, primary_weight in zip(vertices_mm, normals, primary_weights, strict=True):
            vertices_payload.append((*[coordinate * 0.001 for coordinate in vertex], *normal, primary_weight))
        indices_payload.extend(first_vertex + index for triangle in triangles for index in triangle)
        records_payload.append(struct.pack(
            "<8I16f", primary_body_index, secondary_body_index, first_vertex, len(vertices_mm), first_index,
            len(triangles) * 3, stable_id, layer,
            *primary_translation, *primary_rotation, primary_scale,
            *secondary_translation, *secondary_rotation, secondary_scale,
        ))
        provenance.append({
            "stable_id": stable_id, "member_id": member_id, "member": member,
            "member_sha256": hashlib.sha256(obj).hexdigest(),
            "label": label, "layer": layer_name, "endpoint_source": endpoint_source,
            "primary_myosim_body": primary_name, "primary_core_body_index": primary_body_index,
            "secondary_myosim_body": secondary_name, "secondary_core_body_index": secondary_body_index,
            "matched_muscles": [dict(route, name=name) for name, route in zip(source_muscles, matched_routes, strict=True)],
            "primary_weight_range": [0.0, 1.0], "vertex_count": len(vertices_mm), "triangle_count": len(triangles),
        })
        if attachment_weight_lock is not None:
            provenance[-1]["secondary_attachment_weight_lock"] = attachment_weight_lock
    if len(vertices_payload) > 0xFFFFFFFF or len(indices_payload) > 0xFFFFFFFF:
        raise ImportError("BodyParts3D full-body tissue payload exceeds the uint32 native renderer capacity")
    registration_fingerprint = _bodyparts_visual_registration_fingerprint(registration_file)
    payload = b"".join([
        struct.pack("<8s5I32s", _BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_MAGIC, _BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_ABI,
                    len(records_payload), len(vertices_payload), len(indices_payload), registration_fingerprint,
                    bytes.fromhex(source_sha)),
        *records_payload, b"".join(struct.pack("<7f", *vertex) for vertex in vertices_payload),
        struct.pack(f"<{len(indices_payload)}I", *indices_payload),
    ])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
        "payload": {"file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
                    "magic": _BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_MAGIC.rstrip(b"\0").decode("ascii"),
                    "payload_abi": _BODYPARTS_MYOSIM_SOFT_TISSUE_VISUAL_ABI,
                    "registration_fingerprint32": f"{registration_fingerprint:08x}",
                    "surface_count": len(records_payload),
                    "vertex_count": len(vertices_payload), "index_count": len(indices_payload)},
        "source": {"registration": {"file": registration_file.name, "sha256": sha256(registration_file)},
                   "bodyparts": expected_bodyparts, "myosim_source_archive_sha256": source_sha,
                   "myosim_manifest": {"file": "myosim-fullbody-reference.manifest.json", "sha256": sha256((myosim_artifact.resolve() / "myosim-fullbody-reference.manifest.json"))},
                   "surface_map": {"file": "bodyparts3d-myosim-surface-map.v1.json", "sha256": sha256(REPOSITORY_ROOT / "config/bodyparts3d-myosim-surface-map.v1.json")},
                   "surfaces": provenance},
        "coverage": {"configured_surface_count": len(specifications), "muscle_surface_count": sum(1 for entry in provenance if entry["layer"] == "muscle"),
                     "tendon_surface_count": sum(1 for entry in provenance if entry["layer"] == "tendon"),
                     "authored_myosim_muscle_count": len(myosim_manifest["muscles"])},
        "runtime_binding": "exact BodyParts3D source surfaces use a per-vertex two-body linear blend in their named Core articulated endpoint frames; ordinary entries derive both endpoint bodies directly from the named authored MyoSim route sites, while the calcaneal-tendon insertion is secondary-calcaneus locked by exact source-mesh proximity",
        "status": "native_two_body_kinematic_surface_binding_input_not_collision_or_physics",
        "evidence_boundary": "This source-authored surface package visually follows exact named articulated endpoint bodies. It does not make the source surface a force-transmitting continuum, add a tendon constitutive law, create collision/contact, or establish a medical registration.",
    }
    write_json(output / "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json", manifest)
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
    )
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
        },
        "skeleton": {
            "status": (
                "blocked"
                if unsupported or (bounded_admission is not None and not bounded_function_based)
                else (
                    "compatible_bounded_mobile_root_direct_effort_contact"
                    if bounded_function_based
                    else "compatible"
                )
            ),
            "unsupported_joint_kinds": unsupported,
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


def gate_report(
    *,
    sources: Path,
    upper_archive: Path | None,
    source_lock: dict[str, Any],
    runtime_contract: dict[str, Any],
    runtime_root: Path | None = None,
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

    upper_gate: dict[str, Any] = {
        "required_file": source_lock["sources"]["mobl_arms_upper_extremity"]["release_file"],
        "terms": source_lock["sources"]["mobl_arms_upper_extremity"]["license"],
        "status": "missing_authenticated_archive",
    }
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
    if upper_gate["status"] == "ready_for_import" and upper_archive is not None:
        runtime_compatibility["upper_extremities"] = runtime_compatibility_report(
            parse_opensim_archive(upper_archive, "mobl_arms_upper_extremity"),
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
    upper_archive: Path,
    classification_path: Path,
    target_mapping_path: Path,
    source_lock: dict[str, Any],
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

    upper = parse_opensim_archive(upper_archive, "mobl_arms_upper_extremity")

    mechanics = [lower, upper]
    manifest = {
        "schema": "numi.human.v1",
        "revision": 1,
        "provenance": {
            "source_lock_schema": source_lock["schema"],
            "bodyparts_attribution": source_lock["sources"]["bodyparts3d_4"]["attribution"],
            "upper_extremity_terms": source_lock["sources"]["mobl_arms_upper_extremity"]["license"],
            "upper_archive_sha256": sha256(upper_archive),
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
