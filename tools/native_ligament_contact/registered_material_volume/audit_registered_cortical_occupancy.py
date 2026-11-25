"""Find positive-barycentric cortical samples inside the actual registered skin/fat tetrahedra."""

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import igl
import numpy as np

# The native ligament reader owns the NHBONES layout and typed cortical IDs.
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "registered_bed_work/first_contact/material_response"
    ),
)

from rebind_ligament_material import banks
from scipy.spatial.transform import Rotation


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def loaded_deformation(reference, loaded, cells, materials, *, compensated=False):
    """Keep the native reference precision and the observed displacement precision."""
    reference = reference.astype(np.float32).astype(np.float64)
    if not compensated:
        loaded = loaded.astype(np.float32).astype(np.float64)
    result = {
        material: {
            "invalid_reference_cells": 0,
            "relative_volume_failures": 0,
            "principal_stretch_failures": 0,
            "minimum_stretch": None,
            "maximum_stretch": None,
            "worst_stretch_cell": None,
        }
        for material in (15, 16)
    }
    for start in range(0, len(cells), 32768):
        block = cells[start : start + 32768]
        src = reference[block]
        cur = loaded[block]
        source_edges = src[:, 1:] - src[:, :1]
        valid = np.linalg.det(source_edges) > 0.0
        rows = np.flatnonzero(valid)
        gradient = np.linalg.solve(source_edges[rows], (cur[:, 1:] - cur[:, :1])[rows])
        relative_volume = np.linalg.det(gradient)
        stretch = np.linalg.svd(gradient, compute_uv=False)
        if not np.all(np.isfinite(stretch)) or not np.all(np.isfinite(relative_volume)):
            raise ValueError("Nonfinite deformation in the registered material")
        for material, stats in result.items():
            belongs = materials[start : start + len(block)] == material
            stats["invalid_reference_cells"] += int(np.sum(belongs & ~valid))
            selected = np.flatnonzero(belongs[rows])
            if not len(selected):
                continue
            j = relative_volume[selected]
            s = stretch[selected]
            # Same release envelope as the supported material admission owner.
            stats["relative_volume_failures"] += int(np.sum((j < 0.4) | (j > 2.5)))
            stats["principal_stretch_failures"] += int(
                np.sum((s[:, -1] < 2 / 3) | (s[:, 0] > 1.5))
            )
            lowest = float(s[:, -1].min())
            peak = int(np.argmax(s[:, 0]))
            if stats["minimum_stretch"] is None or lowest < stats["minimum_stretch"]:
                stats["minimum_stretch"] = lowest
            if (
                stats["maximum_stretch"] is None
                or s[peak, 0] > stats["maximum_stretch"]
            ):
                stats["maximum_stretch"] = float(s[peak, 0])
                stats["worst_stretch_cell"] = start + int(rows[selected[peak]])
    return result


def accepted_bed_load(lines, physical_step_s):
    """Read the force from the dense FEM/bed owner at its last accepted root."""
    result = {}
    for line in lines:
        if line.startswith("resting_initialization="):
            fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
            result["initialization"] = fields["resting_initialization"]
            result["initial_balanced"] = fields["balanced"] == "true"
            result["initial_maximum_acceleration_residual_m_s2"] = float(
                fields["maximum_acceleration_residual"]
            )
        elif line.startswith("resting_fem_skin_bed_accepted_root_reaction "):
            fields = dict(word.split("=", 1) for word in line.split()[1:] if "=" in word)
            if fields["scope"] != "last_accepted_root_existing_frame_reduction":
                raise ValueError("Unmatched dense support reaction interval")
            dt = float(fields["root_dt_s"])
            impulse = float(fields["normal_impulse_Ns"])
            force = float(fields["mean_normal_force_N"])
            if (dt <= 0 or impulse < 0 or force < 0
                    or not np.isfinite((dt, impulse, force)).all()
                    or not np.isclose(dt, physical_step_s, rtol=1e-7, atol=0)
                    or not np.isclose(impulse / dt, force, rtol=1e-7, atol=1e-10)):
                raise ValueError("Inconsistent bed impulse, force or physical time")
            result["last_root_mean_normal_force_N"] = force
            result["last_root_normal_impulse_Ns"] = impulse
            result["last_root_contact_samples"] = int(fields["valid_contact_sample_count"])
    return result


def read_material_capture(
    directory, cortex, reference, elements, labels, cells, material_rows
):
    """Read the terminal physical FEM state and skeleton from the same native transaction."""
    cfg = json.loads((directory / "launch.json").read_bytes())
    if json.loads((directory / "execution.json").read_bytes())["exit"] != 0:
        raise ValueError("The native capture did not complete")
    settings = cfg["environment"]
    pre = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    for key, expected in (
        ("TETGEN_NODE", reference),
        ("TETGEN_ELEMENTS", elements),
        ("CHILD_LABELS", labels),
    ):
        if settings[pre + key + "_SHA256"] != digest(expected):
            raise ValueError(
                "The observed scene and the audit use different material inputs"
            )
    if digest(cortex) != cfg["pins"].get(cfg["argv"][3]):
        raise ValueError("The observed scene uses a different cortical asset")
    traces = Path(cfg["argv"][4])
    prefix = Path(settings["NUMI_HUMAN_REGISTERED_CELL_NODE_PREFIX"])
    if (
        traces.resolve().parent != directory.resolve()
        or prefix.resolve().parent != directory.resolve()
    ):
        raise ValueError("The accepted outputs were not produced in the supplied run")
    with (traces / "resting-coupled.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    steps = int(cfg["argv"][cfg["argv"].index("--muscle-step-count") + 1])
    if not rows or int(rows[-1]["step"]) != steps:
        raise ValueError("Missing terminal physiology sample")
    receipt = traces / "accepted-geometry" / f"step-{steps}.receipt.json"
    capture = json.loads(receipt.read_bytes())
    if (
        capture["accepted_step"] != steps
        or not np.isfinite([capture["accepted_time_s"], float(rows[-1]["time_s"])]).all()
        or abs(capture["accepted_time_s"] - float(rows[-1]["time_s"])) > 1e-10
    ):
        raise ValueError("Terminal material and rigid capture times disagree")
    pack = receipt.with_name(f"step-{steps}.mrvpack")
    if (
        Path(capture["accepted_pack_path"]).resolve() != pack.resolve()
        or digest(pack) != capture["pack_file_sha256"]
    ):
        raise ValueError("The terminal render pack differs from its accepted receipt")
    poses = {
        rec["body_index"]: (
            Rotation.from_quat(rec["quaternion_xyzw"]),
            rec["position_m"],
        )
        for rec in capture["accepted_registered_body_poses"]
    }
    if len(poses) != len(capture["accepted_registered_body_poses"]):
        raise ValueError("Repeated registered rigid body")
    initial_path = Path(settings["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    if digest(initial_path) != cfg["pins"].get(str(initial_path)):
        raise ValueError("Unmatched loaded material in the native trial")

    def nodes(path):
        values = np.loadtxt(path, skiprows=1, usecols=(0, 1, 2, 3))
        if (
            not np.array_equal(values[:, 0], np.arange(1, len(values) + 1))
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid accepted material nodes")
        return values[:, 1:]

    source = nodes(initial_path)
    opening = nodes(prefix.with_name(prefix.name + "-reference.node"))
    state = prefix.with_name(prefix.name + "-accepted.node")
    final = nodes(state)
    if not np.array_equal(source, opening) or final.shape != opening.shape:
        raise ValueError(
            "Captured material node identities or initial coordinates differ"
        )
    if "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ" in settings:
        cavity = Path(settings["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
        if digest(cavity) != settings[
            "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"
        ] or digest(cavity) != cfg["pins"].get(str(cavity)):
            raise ValueError("Unmatched cavity topology")
        with np.load(cavity) as change:
            retained = change["retained_input_cell_rows"]
            new = change["replacement_tetrahedra"]
            offsets = change["replacement_parent_offsets"]
            donors = change["replacement_parent_cell_rows"]
            if (
                new.ndim != 2 or new.shape[1] != 4
                or np.any(new < 0) or np.any(new >= len(final))
                or len(np.unique(retained)) != len(retained)
                or not np.array_equal(
                    np.unique(np.r_[retained, donors]), np.arange(len(cells))
                )
                or np.intersect1d(retained, donors).size
            ):
                raise ValueError("Cavity does not close the source cell ledger")
            if (
                len(offsets) != len(new) + 1
                or np.any(offsets[1:] <= offsets[:-1])
                or offsets[0] != 0
                or offsets[-1] != len(donors)
            ):
                raise ValueError("Invalid cavity parent ranges")
            parent_material = material_rows[donors]
            child_material = parent_material[offsets[:-1]]
            if not np.array_equal(
                parent_material, np.repeat(child_material, np.diff(offsets))
            ):
                raise ValueError("Cavity crosses physical material identities")
            cells = np.vstack([cells[retained], new])
            material_rows = np.r_[material_rows[retained], child_material]
    step_s = float(cfg["argv"][cfg["argv"].index("--muscle-step-seconds") + 1])
    with (directory / "stdout.log").open() as log:
        bed = accepted_bed_load(log, step_s)
    displacement = np.linalg.norm(final - opening, axis=1)
    observation = {
        "launch_sha256": digest(directory / "launch.json"),
        "receipt_sha256": digest(receipt),
        "accepted_node_sha256": digest(state),
        "bed_support": bed,
        "loaded_node_sha256": digest(initial_path),
        "step": steps,
        "time_s": capture["accepted_time_s"],
        "cell_count": len(cells),
        "maximum_node_displacement_m": float(displacement.max()),
        "median_node_displacement_m": float(np.median(displacement)),
        "worst_node_row": int(displacement.argmax()),
    }
    return final, poses, cells, material_rows, receipt, observation


def validate(
    scene,
    volume_launch,
    registration,
    body_ids,
    proposal=None,
    sample_output=None,
    accepted_run=None,
):
    source = json.loads(volume_launch.read_bytes())
    scene_bytes = scene.read_bytes()
    if scene.suffix == ".gz":
        scene_bytes = gzip.decompress(scene_bytes)
    cfg = json.loads(scene_bytes)
    cortex = Path(cfg["argv"][3])
    if digest(cortex) != cfg["pins"][str(cortex)]:
        raise ValueError("Wrong native cortex")
    reg = json.loads(registration.read_bytes())
    if digest(volume_launch) != reg["source_launch"]:
        raise ValueError("Different physical volume owner")
    pose_path = Path(reg["initial_bodies_path"])
    raw = json.loads(pose_path.read_bytes())
    if digest(pose_path) != reg["initial_frame_sha256"]:
        raise ValueError("Wrong physical skeleton reference")
    poses = {
        r["body_index"]: (
            Rotation.from_quat(r["current"]["orientation_world_xyzw"]),
            r["current"]["position_world_m"],
        )
        for r in raw["bodies"]
    }
    e = source["environment"]
    pref = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    tet = Path(e[pref + "TETGEN_ELEMENTS"])
    labels = Path(e[pref + "CHILD_LABELS"])
    if (
        digest(tet) != e[pref + "TETGEN_ELEMENTS_SHA256"]
        or digest(labels) != e[pref + "CHILD_LABELS_SHA256"]
    ):
        raise ValueError("Unmatched cell ledger")
    node_path = e.get("NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE")
    node = Path(node_path) if node_path else tet.parent / "current.node"
    if node_path and source["pins"].get(node_path) != digest(node):
        raise ValueError("Unmatched loaded material input in the volume launch")
    if digest(node) != reg["loaded_nodes_sha256"]:
        raise ValueError("Unmatched material coordinate frame")
    table = np.loadtxt(node, skiprows=1, usecols=(0, 1, 2, 3))
    if not np.array_equal(table[:, 0], np.arange(1, len(table) + 1)):
        raise ValueError("Unordered material nodes")
    x = np.ascontiguousarray(table[:, 1:])
    if proposal is not None:
        with np.load(proposal) as field:
            ids = field["node_rows"]
            delta = field["delta_m"]
            if (
                ids.ndim != 1
                or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != len(ids)
                or np.any((ids < 0) | (ids >= len(x)))
                or delta.shape != (len(ids), 3)
                or not np.isfinite(delta).all()
            ):
                raise ValueError("Invalid material update")
            x[ids] += delta
    elements = np.loadtxt(tet, skiprows=1, usecols=range(5), dtype=np.int64)
    if not np.array_equal(elements[:, 0], np.arange(1, len(elements) + 1)):
        raise ValueError("Unordered cell ancestry")
    cells = elements[:, 1:] - 1
    if cells.min() < 0 or cells.max() >= len(x):
        raise ValueError("Invalid material connectivity")
    lab = np.loadtxt(labels, delimiter=",", skiprows=1, dtype=np.int64, usecols=(0, 1))
    if not np.array_equal(lab[:, 0], np.arange(len(cells))) or set(
        np.unique(lab[:, 1])
    ) != {15, 16}:
        raise ValueError("Unmatched skin and fat materials")
    reference_path = Path(e[pref + "TETGEN_NODE"])
    if digest(reference_path) != e[pref + "TETGEN_NODE_SHA256"]:
        raise ValueError("Unmatched stress-free material reference")
    reference = np.loadtxt(reference_path, skiprows=1, usecols=(0, 1, 2, 3))
    if reference.shape != table.shape or not np.array_equal(
        reference[:, 0], table[:, 0]
    ):
        raise ValueError("The stress-free and loaded node rows differ")
    if not np.isfinite(reference).all() or not np.isfinite(x).all():
        raise ValueError("Nonfinite registered coordinates")
    observation = None
    if accepted_run is not None:
        if proposal is not None:
            raise ValueError("An accepted run supplies its own loaded coordinates")
        x, poses, cells, materials, pose_path, observation = read_material_capture(
            accepted_run, cortex, reference_path, tet, labels, cells, lab[:, 1]
        )
        lab = np.column_stack((np.arange(len(cells)), materials))
    if observation is None:
        x = x.astype(np.float32).astype(np.float64)
    else:
        observation["coordinate_representation"] = (
            "native rest + high displacement + low displacement"
        )
    deformation = loaded_deformation(
        reference[:, 1:], x, cells, lab[:, 1], compensated=observation is not None
    )
    lo = np.empty((len(cells), 3))
    hi = np.empty_like(lo)
    for begin in range(0, len(cells), 32768):
        sl = slice(begin, begin + 32768)
        v = x[cells[sl]]
        lo[sl] = v.min(axis=1)
        hi[sl] = v.max(axis=1)
    from torso_cell_geometry import geom

    quality = {
        m: {"cells": 0, "inverted": 0, "tiny": 0, "low_shape": 0} for m in (15, 16)
    }
    for begin in range(0, len(cells), 32768):
        sl = slice(begin, begin + 32768)
        vol, shape = geom(x, cells[sl])
        if not (np.isfinite(vol).all() and np.isfinite(shape).all()):
            raise ValueError("Nonfinite material state")
        for m, stats in quality.items():
            on = lab[sl, 1] == m
            stats["cells"] += int(sum(on))
            stats["inverted"] += int(np.sum(vol[on] <= 0))
            stats["tiny"] += int(np.sum(vol[on] <= 1e-18))
            stats["low_shape"] += int(np.sum(shape[on] < 1e-5))
    if observation:
        declared = {}
        with (accepted_run / "stdout.log").open() as stream:
            for line in stream:
                if not line.startswith("registered_cell_shape phase=accepted "):
                    continue
                fields = dict(
                    part.split("=", 1) for part in line.split()[1:] if "=" in part
                )
                if int(fields["step"]) == observation["step"]:
                    declared[int(fields["material"])] = {
                        "cells": int(fields["cells"]),
                        "inverted": int(fields["inverted"]),
                        "tiny": int(fields["volume_le_1e_18"]),
                        "low_shape": int(fields["shape_lt_1e_5"]),
                    }
        if declared != quality:
            raise ValueError(
                "Terminal node geometry differs from the native accepted material certificate"
            )
        observation["native_cell_quality_reproduced"] = True
    rows = {}
    sample_records = []
    visited = set()
    for (body, sid), (f, y, _) in banks(cortex).items():
        if body not in body_ids:
            continue
        visited.add(body)
        rot, t = poses[body]
        # Transform each exact seam position once, preserving source sample IDs.
        unique, seam = np.unique(y, axis=0, return_inverse=True)
        v = (rot.apply(unique) + t)[seam]
        used = np.unique(f)
        pts = np.ascontiguousarray(np.vstack((v[used], v[f].mean(1))))
        selected = np.flatnonzero(
            np.all((hi >= pts.min(0)) & (lo <= pts.max(0)), axis=1)
        )
        bank = np.ascontiguousarray(cells[selected])
        idx = np.full(len(pts), -1, dtype="i8")
        if len(bank):
            tree = igl.AABB()
            tree.init(x, bank)
            idx = igl.in_element(x, bank, pts, tree)
        samples = np.flatnonzero(idx >= 0)
        counts = {15: 0, 16: 0}
        witness = {}
        if len(samples):
            ids = selected[idx[samples]]
            verts = x[cells[ids]]
            edges = verts[:, 1:] - verts[:, :1]
            inv = np.linalg.inv(edges.transpose(0, 2, 1))
            coord = np.einsum("ijk,ik->ij", inv, pts[samples] - verts[:, 0])
            bary = np.column_stack((1 - coord.sum(1), coord))
            # Signed distance to the nearest facet certifies strictly interior samples.
            gradients = np.concatenate((-inv.sum(1)[:, None, :], inv), axis=1)
            depths = (bary / np.linalg.norm(gradients, axis=2)).min(1)
            det = np.linalg.det(edges)
            physical_volume = det / 6
            for material in counts:
                ok = np.flatnonzero(
                    (bary.min(1) > 1.0e-6)
                    & (depths > 1.0e-9)
                    & (physical_volume > 1.0e-18)
                    & (lab[ids, 1] == material)
                )
                counts[material] = len(ok)
                if sample_output is not None:
                    sample_records.extend(
                        (body, sid, int(samples[j]), material, int(ids[j])) for j in ok
                    )
                if not len(ok):
                    continue
                j = ok[np.argmax(depths[ok])]
                i = int(samples[j])
                which = "vertex" if i < len(used) else "face_centroid"
                witness[material] = {
                    "cortical_sample": which,
                    "source_index": int(used[i]) if i < len(used) else i - len(used),
                    "material_tet_row": int(ids[j]),
                    "material_nodes": cells[ids[j]].tolist(),
                    "point_m": pts[i].tolist(),
                    "barycentric": bary[j].tolist(),
                    "min_distance_to_cell_faces_mm": float(depths[j] * 1000),
                    "volume_m3": float(physical_volume[j]),
                }
        rows[f"{body}/{sid}"] = {
            "body": body,
            "stable": sid,
            "tested_samples": len(pts),
            "interior_samples_by_material": counts,
            "witness": witness,
        }
    if visited != set(body_ids):
        raise ValueError("Missing requested cortical body")
    if sample_output is not None:
        np.save(sample_output, np.array(sample_records, dtype=np.int64).reshape(-1, 5))
    return {
        "accepted_observation": observation,
        "scene": hashlib.sha256(scene_bytes).hexdigest(),
        "volume_launch": digest(volume_launch),
        "registration": digest(registration),
        "proposal": digest(proposal) if proposal is not None else None,
        "material_quality": quality,
        "stress_free_nodes": digest(reference_path),
        "loaded_deformation": deformation,
        "cortex": digest(cortex),
        "material_nodes": digest(node),
        "elements": digest(tet),
        "labels": digest(labels),
        "frame": digest(pose_path),
        "scope": (
            "Accepted native cortical pose and terminal material coordinates."
            if observation
            else "Current cortical asset in the registered loaded material reference frame. Strict interior of actual labelled positive-volume cells. Counts describe sampled cortical points; a zero sample count does not prove separation."
        ),
        "rows": rows,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, required=True, help="Native integrated launch.json"
    )
    parser.add_argument(
        "--volume-launch",
        type=Path,
        required=True,
        help="Hash-bound physical volume launch.json",
    )
    parser.add_argument(
        "--material-registration",
        type=Path,
        required=True,
        help="Compiled material-boundary.json",
    )
    parser.add_argument(
        "--bodies", nargs="+", type=int, default=[34, 41, 42, 43, 84, 91, 92, 93]
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        help="Optional source-node rows and displacements in metres",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-output",
        type=Path,
        help="Optional array of body, cortical bank, sample, material and cell row",
    )
    parser.add_argument(
        "--require-clear",
        action="store_true",
        help="Return failure on any certified interior sample or invalid material cell",
    )
    parser.add_argument(
        "--require-deformation",
        action="store_true",
        help="Also enforce the registered whole-volume Jacobian and principal-stretch envelope",
    )
    parser.add_argument(
        "--accepted-run",
        type=Path,
        help="Completed dense native run with terminal tissue nodes and registered rigid poses",
    )
    args = parser.parse_args()
    result = validate(
        args.scene,
        args.volume_launch,
        args.material_registration,
        args.bodies,
        args.proposal,
        args.sample_output,
        args.accepted_run,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    count = sum(
        sum(r["interior_samples_by_material"].values()) for r in result["rows"].values()
    )
    invalid = any(
        any(row[k] for k in ("inverted", "tiny", "low_shape"))
        for row in result["material_quality"].values()
    )
    print(
        f"Confirmed cortical samples inside material: {count}. Invalid material cells present: {invalid}."
    )
    deformation_invalid = any(
        row[key]
        for row in result["loaded_deformation"].values()
        for key in (
            "invalid_reference_cells",
            "relative_volume_failures",
            "principal_stretch_failures",
        )
    )
    stretch_count = sum(
        row["principal_stretch_failures"]
        for row in result["loaded_deformation"].values()
    )
    print(
        f"Loaded skin/fat cells outside the principal-stretch envelope: {stretch_count}."
    )
    if (args.require_clear and (invalid or count)) or (
        args.require_deformation and deformation_invalid
    ):
        raise SystemExit(1)
