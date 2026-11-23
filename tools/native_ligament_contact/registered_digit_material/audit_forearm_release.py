"""Identify skin/anatomy pairs already present when the supported body is released."""

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

from verify_native_extensor_pair import loaded
from registered_forearms import read, sha


def audit(case, reference, reader, banks, out):
    receipt, pose, raw = loaded(case, True, False)
    final = read(reference / "terminal.receipt.json")
    if (
        receipt["base_pack_file_sha256"] != final["base_pack_file_sha256"]
        or receipt["pack_file_sha256"] != final["pack_file_sha256"]
        or receipt["initial_anatomical_registration"]
        != final["initial_anatomical_registration"]
        or not (
            receipt["physical_endpoint"] == "accepted"
            and receipt["accepted_step"] == 8000
        )
    ):
        raise ValueError("Different initial pose or final anatomy")
    out.mkdir(exist_ok=False)
    sys.path.insert(0, str(reader.parent / "src"))
    api = runpy.run_path(str(reader))
    scan = runpy.run_path(str(banks))["scan_banks"]
    ((_, _, v, f),) = api["primitives"](raw, pose, skin_only=True)
    initial = dict(
        pack_sha256=receipt["base_pack_file_sha256"],
        pose_count=len(pose),
        skin_vertices=len(v),
        skin_faces=len(f),
        launch_sha256=sha(case / "launch.json"),
        scope="Native release pack with its captured initial skeletal registration",
    )
    scan(api, raw, pose, v, f, initial, out, "initial-envelope.json", check_skin=True)
    subprocess.run(
        [
            sys.executable,
            banks.parent / "classify_registered_envelope.py",
            out / "initial-envelope.json",
            case / "stdout.log",
            out / "physical-banks.json",
            "--window-receipt",
            case / "resting-ocular-sclera-window.json",
        ],
        check=True,
    )
    eyes = read(case / "resting-ocular-sclera-window.json")
    if not eyes["presentation_only"] or eyes["physical_state_created"]:
        raise ValueError("Physical eye geometry marked as display")
    displays = {
        row["derived_skin_identity"]["stable_id"]
        for row in eyes["windows"] + eyes["posterior_lens_backings"]
    }

    def pairs(p):
        rows = set()
        for line in (p / "target-witnesses.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["ocular_source_id"] or (
                r["semantic"] == 51010 and r["stable_id"] in displays
            ):
                continue
            rows.update(
                (r["semantic"], r["stable_id"], r["primitive"], i, j)
                for i, j in r["triangle_pairs"]
            )
        if len(rows) != read(p / "physical-banks.json")["physical_nonocular_pairs"]:
            raise ValueError("Physical triangle witness set mismatch")
        return rows

    poses = [
        pairs(p) for p in (out, reference / "envelope-3999", reference / "envelope")
    ]
    record = dict(
        physical_pairs=[len(p) for p in poses],
        remembered_times_s=[
            0.0,
            read(reference / "step-3999.receipt.json")["accepted_time_s"],
            final["accepted_time_s"],
        ],
        release_to_terminal_persist=len(poses[0] & poses[2]),
        mid_to_terminal_persist=len(poses[1] & poses[2]),
        new_since_release=len(poses[2] - poses[0]),
        resolved_since_release=len(poses[0] - poses[2]),
        release_self=initial["skin_self_intersections"],
        release=sha(out / "initial-envelope.json"),
        end=sha(reference / "envelope/accepted-envelope.json"),
    )
    (out / "trajectory.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "reference", "reader", "banks", "out"):
        p.add_argument(name, type=lambda s: Path(s).resolve())
    audit(**vars(p.parse_args()))
