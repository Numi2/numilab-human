"""Compare the two registered muscle envelopes in matched full native captures."""

import hashlib
import json
from pathlib import Path
import runpy
import sys
import subprocess

import numpy as np

from solve_extensor_sites import Neighbors
from extensor_registration import sha


def loaded(case, initial, dense):
    receipt = json.loads((case / "terminal.receipt.json").read_text())
    poses = (
        receipt["initial_anatomical_registration"]["body_poses"]
        if initial
        else receipt["accepted_body_poses"]
    )
    poses = {p["body_index"]: p for p in poses}
    if sorted(poses) != list(range(receipt["accepted_body_pose_count"])):
        raise ValueError("Incomplete native skeleton")
    pack = case / ("initial.mrvpack.zst" if initial else "accepted.mrvpack.zst")
    if dense:
        if initial:
            raise ValueError("Dense control retained only its accepted pack")
        pack = case / "accepted-terminal.mrvpack.zst"
    raw = subprocess.check_output(["zstd", "-qdc", pack])
    if (
        hashlib.sha256(raw).hexdigest()
        != receipt["base_pack_file_sha256" if initial else "pack_file_sha256"]
    ):
        raise ValueError("Native geometry identity mismatch")
    return receipt, poses, raw


def compare(control, edited, reader, predicate, initial=False, dense=False):
    old, pose0, data0 = loaded(control, initial, dense)
    new, pose1, data1 = loaded(edited, initial, dense)
    if pose0 != pose1:
        raise ValueError("Unmatched body poses")
    if not initial:
        for k in (
            "accepted_step",
            "accepted_body_state_sha256",
            "accepted_respiration_state_sha256",
            "physical_endpoint",
            "surface_audit_endpoint",
        ):
            if old[k] != new[k]:
                raise ValueError("Accepted motion or physiology changed")
    sys.path.insert(0, str(reader.parent / "src"))
    api = runpy.run_path(str(reader))
    pr = api["primitives"]
    targets = {}
    for raw, poses in ((data0, pose0), (data1, pose1)):
        for (kind, sid), _, v, f in pr(raw, poses):
            if kind == 51005 and sid in (137, 138):
                targets.setdefault(sid, []).append((v.copy(), f.copy()))
    neighbors = {sid: [] for sid in targets}
    boxes = {
        sid: (
            np.concatenate([p[0] for p in versions]).min(0),
            np.concatenate([p[0] for p in versions]).max(0),
        )
        for sid, versions in targets.items()
    }
    same = 0
    for a, b in zip(pr(data0, pose0), pr(data1, pose1), strict=True):
        key, prim, v, f = a
        if b[0] != key or b[1] != prim or not np.array_equal(f, b[3]):
            raise ValueError("Anatomical topology or identities changed")
        if key[0] == 51005 and key[1] in targets:
            continue
        if not np.array_equal(v, b[2]):
            raise ValueError(f"Unrelated structure changed at {key}")
        same += 1
        tri = v[f]
        lo = tri.min(1)
        hi = tri.max(1)
        for sid, (lower, upper) in boxes.items():
            mask = ((hi >= lower) & (lo <= upper)).all(1)
            if mask.any():
                ids, idx = np.unique(f[mask], return_inverse=True)
                neighbors[sid].append((*key, v[ids], idx.reshape(-1, 3), 1.0))
    if set(targets) != {137, 138}:
        raise ValueError("Missing bilateral muscle bank")
    lo, hi = boxes[137]
    lo2, hi2 = boxes[138]
    if ((hi >= lo2) & (hi2 >= lo)).all():
        raise ValueError("Bilateral envelopes overlap in the broad phase")
    result = {}
    categories = ("self", "skin", "bone", "other_muscles", "other")
    for sid, versions in targets.items():
        old_xyz, faces = versions[0]
        new_xyz = versions[1][0]
        scene = Neighbors(neighbors[sid], faces, old_xyz, predicate)
        report = {}
        bybank = []
        type_list = [
            (kind, stable)
            for kind, stable, v, f, _ in neighbors[sid]
            for i in range(len(f))
        ]
        for label, points in [("control", old_xyz), ("edited", new_xyz)]:
            witnesses, counts = scene.evaluate(points)
            report[label] = {k: int(n) for k, n in zip(categories, counts)}
            affected = {}
            for i, j in witnesses:
                key = type_list[i] if i < scene.start else (51005, sid)
                tag = f"{key[0]}/{key[1]}"
                affected[tag] = affected.get(tag, 0) + 1
            bybank.append(affected)
        delta = np.linalg.norm(old_xyz - new_xyz, axis=1)
        report.update(
            maximum_displacement_mm=float(delta.max() * 1000),
            median_displacement_mm=float(np.median(delta) * 1000),
            new_affected_banks=sorted(bybank[1].keys() - bybank[0].keys()),
            cleared_banks=sorted(bybank[0].keys() - bybank[1].keys()),
            by_bank={"control": bybank[0], "edited": bybank[1]},
        )
        result[sid] = report
    report = dict(
        time_s=0 if initial else old["accepted_time_s"],
        unchanged_other_native_banks=same,
        source_pack_sha256=hashlib.sha256(data0).hexdigest(),
        edited_pack_sha256=hashlib.sha256(data1).hexdigest(),
        reader_sha256=sha(reader),
        predicate_sha256=sha(predicate),
        banks=result,
    )
    prior_dir = control / ("initial-envelope" if initial else "envelope")
    class_file = prior_dir / "physical-banks.json"
    if class_file.is_file():
        envelope_file = prior_dir / (
            "initial-envelope.json" if initial else "accepted-envelope.json"
        )
        classes = json.loads(class_file.read_text())
        envelope = json.loads(envelope_file.read_text())
        if (
            classes["pack_sha256"] != report["source_pack_sha256"]
            or envelope["pack_sha256"] != classes["pack_sha256"]
            or classes["audit_sha256"] != sha(envelope_file)
            or classes["checked_banks"] != same + 1
        ):
            raise ValueError("Whole-body audit refers to different geometry")
        expected = {
            (sem, stable): count for sem, stable, _, count in envelope["affected_banks"]
        }
        change = 0
        for sid, regional in result.items():
            original = regional["control"]["skin"]
            if expected.get((51005, sid), 0) != original:
                raise ValueError(
                    "Independent whole-body audit disagrees on an edited muscle"
                )
            change += regional["edited"]["skin"] - original
        # Every other emitted vertex and triangle, including the skin itself,
        # was compared above. Only these two nonocular banks can change the sum.
        report["derived_skin_clearance"] = {
            "control_nonocular_pairs": classes["physical_nonocular_pairs"],
            "edited_nonocular_pairs": classes["physical_nonocular_pairs"] + change,
            "unchanged_skin_self_pairs": envelope["skin_self_intersections"],
            "control_classification_sha256": sha(class_file),
        }
    if dense:
        for name in ("physical-reference.node", "physical-accepted.node"):
            if sha(control / name) != sha(edited / name):
                raise ValueError(f"Physical skin/fat coordinates changed in {name}")
        report["physical_node_sha256"] = sha(edited / "physical-accepted.node")
        control, edited = control / "native-traces", edited / "native-traces"
    if not initial:
        for file in ("resting-coupled.csv", "resting-com-support-impulses.csv"):
            if sha(control / file) != sha(edited / file):
                raise ValueError("Changed coupled trace or physical support impulse")
        report.update(
            coupled_trace_sha256=sha(edited / "resting-coupled.csv"),
            support_trace_sha256=sha(edited / "resting-com-support-impulses.csv"),
        )
    return report


if __name__ == "__main__":
    print(
        json.dumps(
            compare(
                *(Path(p).resolve() for p in sys.argv[1:5]),
                initial="--initial" in sys.argv[5:],
                dense="--dense" in sys.argv[5:],
            ),
            indent=2,
        )
    )
