"""Audit the outer skin against registered anatomy in a retained physical native frame."""

from pathlib import Path
import json, mmap, runpy, subprocess, sys, tempfile, hashlib
from certify_heart import sha


def check(case, audit_source, bank_source, require_clear=False, accepted_step=None):
    launch = json.loads((case / "launch.json").read_text())
    horizon = int(launch["argv"][launch["argv"].index("--muscle-step-count") + 1])
    steps = horizon if accepted_step is None else accepted_step
    if steps <= 0 or steps > horizon:
        raise ValueError("Capture is outside the requested native horizon")
    terminal = steps == horizon
    if not terminal:
        scheduled = launch["environment"]["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"].split(",")
        if str(steps) not in scheduled:
            raise ValueError("Intermediate native state was not scheduled for capture")
    stem = "terminal" if terminal else f"step-{steps}"
    saved = case / (stem + ".mrvpack")
    receipt = case / (stem + ".receipt.json")
    state = json.loads(receipt.read_text())
    if not terminal:
        terminal_receipt = json.loads((case / "terminal.receipt.json").read_text())
        if state["base_pack_file_sha256"] != terminal_receipt["base_pack_file_sha256"]:
            raise ValueError("Intermediate geometry belongs to a different native launch")
    expected = Path(launch["argv"][4]) / f"accepted-geometry/step-{steps}.mrvpack"
    if (
        state["physical_endpoint"] != "accepted"
        or state["surface_audit_endpoint"] != "passed"
        or state["accepted_step"] != steps
        or Path(state["accepted_pack_path"]) != expected
    ):
        raise ValueError("Native geometry does not match physical acceptance")
    poses = {r["body_index"]: r for r in state["accepted_body_poses"]}
    if len(poses) != state["accepted_body_pose_count"] or sorted(poses) != list(
        range(len(poses))
    ):
        raise ValueError("Invalid skeleton")
    sys.path.insert(0, str(audit_source.parent / "src"))
    module = runpy.run_path(str(audit_source))
    audit = runpy.run_path(str(bank_source))["scan_banks"]
    out = case / ("envelope" if terminal else f"envelope-{steps}")
    out.mkdir(exist_ok=True)
    if saved.exists():
        stream = saved.open("rb")
    else:
        compressed = case / ("accepted.mrvpack.zst" if terminal else f"step-{steps}.mrvpack.zst")
        if not compressed.exists() and not terminal:
            raise FileNotFoundError(compressed)
        if not compressed.exists():
            recovery = json.loads((case / "frame-recovery.json").read_text())
            compressed = case / recovery["url"].split("/")[-1]
            if (
                not recovery["independently_restored"]
                or sha(compressed) != recovery["compressed_sha256"]
            ):
                raise ValueError("Unverified compressed frame")
        stream = tempfile.TemporaryFile(dir=case)
        subprocess.run(["zstd", "-qdc", compressed], stdout=stream, check=True)
        stream.seek(0)
    with stream as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as geometry:
        if hashlib.sha256(geometry).hexdigest() != state["pack_file_sha256"]:
            raise ValueError("Changed frame")
        prior = out / "physical-banks.json"
        if prior.exists():
            banks = json.loads(prior.read_text())
            if (
                banks["pack_sha256"] == state["pack_file_sha256"]
                and banks["audit_sha256"] == sha(out / "accepted-envelope.json")
                and banks["native_log_sha256"] == sha(case / "stdout.log")
                and banks["window_receipt_sha256"] == sha(case / "resting-ocular-sclera-window.json")
            ):
                check_clearance(out, require_clear)
                return
        ((_, _, skin, faces),) = module["primitives"](geometry, poses, skin_only=True)
        result = dict(
            accepted_step=steps,
            accepted_time_s=state["accepted_time_s"],
            pack_sha256=state["pack_file_sha256"],
            pose_count=len(poses),
            native_root=state["accepted_root_fingerprint_hex"],
            skin_vertices=len(skin),
            skin_faces=len(faces),
            scope="Single complete accepted endpoint, frozen native transforms and exact Float32 world surface predicates.",
        )
        audit(
            module,
            geometry,
            poses,
            skin,
            faces,
            result,
            out,
            "accepted-envelope.json",
            check_skin=True,
        )
        del skin, faces
    proc = [
        sys.executable,
        bank_source.parent / "classify_registered_envelope.py",
        out / "accepted-envelope.json",
        case / "stdout.log",
        out / "physical-banks.json",
        "--window-receipt",
        case / "resting-ocular-sclera-window.json",
    ]
    subprocess.run(proc, check=True)
    check_clearance(out, require_clear)


def check_clearance(out, required):
    if required:
        banks = json.loads((out / "physical-banks.json").read_text())
        frame = json.loads((out / "accepted-envelope.json").read_text())
        crossings = banks["physical_nonocular_pairs"]
        folds = frame["skin_self_intersections"]
        if type(crossings) is not int or type(folds) is not int or min(crossings, folds) < 0:
            raise ValueError("Invalid endpoint geometry counts")
        if crossings or folds:
            raise SystemExit(f"Endpoint clearance failed with {crossings} physical skin/anatomy pairs and {folds} skin self-intersections")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "audit_source", "bank_source"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    parser.add_argument("--require-clear", action="store_true", help="Require zero physical nonocular skin/anatomy pairs and zero skin self-intersections at this endpoint.")
    parser.add_argument("--accepted-step", type=int, help="Inspect a scheduled accepted intermediate frame.")
    check(**vars(parser.parse_args()))
