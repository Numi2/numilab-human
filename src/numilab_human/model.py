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
from pathlib import Path
from typing import Any, Iterable


class ImportError(ValueError):
    """Raised when a source cannot make a source-faithful Human v1 artifact."""


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
            "The qualified Core path evaluates source static fiber-tendon equilibrium, "
            "GeometryPath wrapping, and body-frame moment-arm force scatter on device. "
            "Dynamic activation/fiber/tendon state advancement and held-out validation "
            "remain separate requirements."
        ),
        "evidence_boundary": (
            "Exact OpenSim muscle, curve, GeometryPath, and wrap source records only. "
            "This IR alone does not evaluate or apply a Hill-type force; that occurs in "
            "the qualified owner Core path."
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
            "requires_core": [
                "mobile FunctionBased MetalWorld state advancement",
                "articulated contact response for FunctionBased root",
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
                "accelerated_runtime": "bounded_fixed_root_free_motion",
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
            "The qualified Core path assembles one fixed-root source body/frame/inertia "
            "tree and FunctionBased programs into MetalWorld-resident free-motion "
            "state. BodyParts3D registration, collision/contact, and broader model "
            "admission remain separate work."
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

    body_set = next((item for item in model if _local_name(item) == "BodySet"), None)
    joint_set = next((item for item in model if _local_name(item) == "JointSet"), None)
    force_set = next((item for item in model if _local_name(item) == "ForceSet"), None)
    if body_set is None or joint_set is None or force_set is None:
        raise ImportError(f"{path.name} must contain BodySet, JointSet, and ForceSet")

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
    objects = next((item for item in joint_set if _local_name(item) == "objects"), None)
    for joint in list(objects) if objects is not None else []:
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
        joints.append(
            {
                "id": identifier,
                "kind": _local_name(joint),
                "parent_frame": _text(joint, "socket_parent_frame") or _text(joint, "parent_body"),
                "child_frame": _text(joint, "socket_child_frame") or _text(joint, "child_body"),
                "coordinates": coordinates,
                "frames": _joint_frames(joint),
                "motion_axes": _joint_motion_axes(joint),
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


def _bodyparts_obj_triangles(
    obj: bytes, source_name: str
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Parse a conservative vertex/face-only subset of an OBJ source member."""
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
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
        elif fields[0] == "f":
            if len(fields) < 4:
                raise ImportError(f"BodyParts3D OBJ face is truncated: {source_name}:{line_number}")
            face: list[int] = []
            for field in fields[1:]:
                index_text = field.split("/", 1)[0]
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
                face.append(index)
            for index in range(1, len(face) - 1):
                triangles.append((face[0], face[index], face[index + 1]))
    if not vertices or not triangles:
        raise ImportError(f"BodyParts3D OBJ has no drawable triangles: {source_name}")
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
                    "status": "eligible_fixed_root_free_motion",
                    "body_count": skeleton_ir["body_count"],
                    "coordinate_count": coordinate_count,
                    "maximum_body_count": 32,
                    "maximum_coordinate_count": 40,
                }
        except ImportError as error:
            bounded_admission = {
                "status": "not_a_supported_fixed_root_tree",
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
                    "compatible_bounded_fixed_root_free_motion"
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
                "bounded to one fixed-root articulation in free motion with "
                "direct effort."
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
        },
        "runtime_compatibility": runtime_compatibility,
        "runtime_checkout": runtime_checkout_gate(runtime_root, runtime_contract),
        "mechanics_execution": {
            "status": "qualified_bounded_device",
            "runtime_revision": runtime_contract["runtime"]["revision"],
            "contract": "One fixed-root FunctionBased articulation executes resident q/v/effort state on device; source Millard static-equilibrium forces are reduced into that same effort arena before every microstep.",
            "remaining_evidence": runtime_contract["muscle_tendon"][
                "source_faithful_requirements"
            ],
        },
        "gates": [
            {
                "id": "source_faithful_import",
                "status": "ready" if source_import_ready else "blocked",
                "requirement": "All three exact source artifacts must verify before a local manifest is built.",
            },
            {
                "id": "skeleton_robotpack_lowering",
                "status": "qualified_bounded_device",
                "requirement": "Qualified: one fixed-root FunctionBased source tree advances in MetalWorld free motion with direct effort. BodyParts3D frame registration, collision/contact, and broader RobotPack admission remain separate source or calibration work.",
            },
            {
                "id": "muscle_tendon_lowering",
                "status": "qualified_bounded_device",
                "requirement": "Qualified: source Millard curves, static fiber-tendon equilibrium, finite-cylinder paths/wraps, and per-muscle forces execute and reduce into MetalWorld's resident effort arena. OpenSim equivalence and held-out force/moment-arm validation remain evidence gates.",
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
                "requirement": "Apple-GPU evidence qualifies the bounded source mechanics path; it does not qualify contact, deformable anatomy, source authentication, or tissue calibration.",
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
