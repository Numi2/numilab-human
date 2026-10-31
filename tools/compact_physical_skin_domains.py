"""Exact active-domain reindexing before physical skin/fat native emission.

This removes only unreferenced node rows and derives compact role-local source
namespaces. It never welds by proximity, changes cells, allocates mass, or admits
physical tissue. Existing emitters must carry the returned source crosswalks.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _indices(value: Any, width: int | None, count: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name}: indices must be integers")
    a = raw.astype(np.int64, copy=False)
    if ((width is None and a.ndim != 1) or
        (width is not None and (a.ndim != 2 or a.shape[1] != width)) or
        a.size == 0 or np.any(a < 0) or np.any(a >= count)):
        raise ValueError(f"{name}: invalid index extent or domain")
    if width is not None and np.any(np.diff(np.sort(a, axis=1), axis=1) == 0):
        raise ValueError(f"{name}: repeated corner")
    return a


def _points(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.dtype != np.float32 or a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"{name}: finite emitted Float32 xyz rows required")
    return a


def compact_volume(points: Any, tetrahedra: Any) -> dict[str, np.ndarray]:
    """Keep every incident point, including unused-role points inside a volume."""
    p = _points(points, "volume")
    t = _indices(tetrahedra, 4, len(p), "tetrahedra")
    active = np.unique(t)
    inverse = np.full(len(p), -1, dtype=np.int64)
    inverse[active] = np.arange(len(active), dtype=np.int64)
    result = {
        "points_world_m": p[active].copy(),
        "tetrahedra": inverse[t],
        "output_node_to_input_node": active,
        "input_node_to_output_node": inverse,
    }
    # Every cell keeps its order and all four exact emitted corner coordinates.
    if not np.array_equal(result["points_world_m"][result["tetrahedra"]].view(np.uint32),
                          p[t].view(np.uint32)):
        raise ValueError("volume compaction changed emitted cell geometry")
    return result


def compact_role(points: Any, faces: Any, volume_node_rows: dict[str, Any],
                 volumes: dict[str, dict[str, np.ndarray]],
                 source_volume_points: dict[str, np.ndarray]) -> dict[str, Any]:
    """Derive a new role namespace; retain original source rows as provenance."""
    p = _points(points, "role")
    f = _indices(faces, 3, len(p), "role faces")
    active = np.unique(f)
    inverse = np.full(len(p), -1, dtype=np.int64)
    inverse[active] = np.arange(len(active), dtype=np.int64)
    mappings = {}
    for label, raw in volume_node_rows.items():
        if label not in volumes:
            raise ValueError("role names an unknown volume")
        original = _points(source_volume_points[label], label)
        rows = _indices(raw, None, len(original), "registered source-to-volume rows")
        if rows.shape != (len(p),) or len(np.unique(rows)) != len(rows):
            raise ValueError("registered role map is not a full injective source map")
        if not np.array_equal(original[rows].view(np.uint32), p.view(np.uint32)):
            raise ValueError("registered role differs from emitted volume coordinates")
        current = volumes[label]["input_node_to_output_node"][rows[active]]
        if np.any(current < 0):
            raise ValueError("referenced role point has no incident physical cell")
        mappings[label] = current
    if not mappings:
        raise ValueError("role has no physical volume owner")
    return {
        "points_world_m": p[active].copy(),
        "faces": inverse[f],
        "output_source_vertex_to_input_source_vertex": active,
        "input_source_vertex_to_output_source_vertex": inverse,
        "volume_node_rows": mappings,
    }


def prepare_paired_active_domains(
        skin_points: Any, skin_tets: Any, fat_points: Any, fat_tets: Any,
        role_inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Prepare exact compact O/I/D namespaces and one shared physical I.

    Cell order is Skin then Fat. Cell material/mass/parent rows keep that order.
    The detailed role face order and face ancestry do not change. Returned point
    source crosswalks must be retained alongside derived authoring namespaces.
    """
    if set(role_inputs) != {"O", "I", "D"}:
        raise ValueError("exact O/I/D role inputs required")
    original = {"Skin": _points(skin_points, "Skin"), "Fat": _points(fat_points, "Fat")}
    volumes = {
        "Skin": compact_volume(original["Skin"], skin_tets),
        "Fat": compact_volume(original["Fat"], fat_tets),
    }
    expected = {"O": {"Skin"}, "I": {"Skin", "Fat"}, "D": {"Fat"}}
    roles = {}
    for role in ("O", "I", "D"):
        entry = role_inputs[role]
        if set(entry["volume_node_rows"]) != expected[role]:
            raise ValueError(f"{role}: wrong physical layer ownership")
        roles[role] = compact_role(entry["points_world_m"], entry["faces"],
                                  entry["volume_node_rows"], volumes, original)
    # One volume point cannot acquire two authoring role identities.
    for label in volumes:
        maps = [r["volume_node_rows"][label] for r in roles.values()
                if label in r["volume_node_rows"]]
        joined = np.concatenate(maps)
        if len(np.unique(joined)) != len(joined):
            raise ValueError(f"{label}: role source maps overlap")

    sp, fp = (volumes[k]["points_world_m"] for k in ("Skin", "Fat"))
    si = roles["I"]["volume_node_rows"]["Skin"]
    fi = roles["I"]["volume_node_rows"]["Fat"]
    if not np.array_equal(sp[si].view(np.uint32), fp[fi].view(np.uint32)):
        raise ValueError("shared I coordinates differ")
    # Establish the direct O prefix without relying on input point order.
    outer = roles["O"]["volume_node_rows"]["Skin"]
    other_skin = np.ones(len(sp), dtype=bool)
    other_skin[outer] = False
    skin_order = np.concatenate((outer, np.flatnonzero(other_skin)))
    skin_joint = np.empty(len(sp), dtype=np.int64)
    skin_joint[skin_order] = np.arange(len(sp), dtype=np.int64)
    fat_joint = np.full(len(fp), -1, dtype=np.int64)
    fat_joint[fi] = skin_joint[si]
    unshared_fat = np.flatnonzero(fat_joint < 0)
    fat_joint[unshared_fat] = len(sp) + np.arange(len(unshared_fat), dtype=np.int64)
    jp = np.vstack((sp[skin_order], fp[unshared_fat]))
    jt = np.vstack((skin_joint[volumes["Skin"]["tetrahedra"]],
                    fat_joint[volumes["Fat"]["tetrahedra"]]))
    if not np.array_equal(np.unique(jt), np.arange(len(jp), dtype=np.int64)):
        raise ValueError("combined domain retains a nonphysical orphan row")
    joint_maps = {"Skin": skin_joint, "Fat": fat_joint}
    for label, v in volumes.items():
        v["output_node_to_joint_node"] = joint_maps[label]
    for role, entry in roles.items():
        maps = {label: joint_maps[label][rows]
                for label, rows in entry["volume_node_rows"].items()}
        first = next(iter(maps.values()))
        if any(not np.array_equal(first, other) for other in maps.values()):
            raise ValueError(f"{role}: shared ownership differs")
        entry["joint_node_rows"] = first
        if not np.array_equal(jp[first].view(np.uint32),
                              entry["points_world_m"].view(np.uint32)):
            raise ValueError(f"{role}: combined geometry mismatch")
    return {"volumes": volumes, "roles": roles, "points_world_m": jp,
            "tetrahedra": jt, "physical_admission": False}
