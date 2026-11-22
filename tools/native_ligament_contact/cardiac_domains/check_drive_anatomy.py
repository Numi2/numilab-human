"""Audit the outer skin against registered anatomy in a retained physical native frame."""

from pathlib import Path
import json, mmap, runpy, subprocess, sys, tempfile, hashlib
from certify_heart import sha


def check(case, audit_source, bank_source):
    saved = case / "terminal.mrvpack"
    receipt = case / "terminal.receipt.json"
    state = json.loads(receipt.read_text())
    launch = json.loads((case / "launch.json").read_text())
    expected = Path(launch["argv"][4]) / "accepted-geometry/step-30000.mrvpack"
    if (
        state["physical_endpoint"] != "accepted"
        or state["surface_audit_endpoint"] != "passed"
        or state["accepted_step"] != 30000
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
    out = case / "envelope"
    out.mkdir(exist_ok=True)
    if saved.exists():
        stream = saved.open("rb")
    else:
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
        if (
            prior.exists()
            and json.loads(prior.read_text())["pack_sha256"]
            == state["pack_file_sha256"]
            and sha(out / "accepted-envelope.json")
            == json.loads(prior.read_text())["audit_sha256"]
        ):
            return
        ((_, _, skin, faces),) = module["primitives"](geometry, poses, skin_only=True)
        result = dict(
            accepted_step=30000,
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


if __name__ == "__main__":
    check(*(Path(p).resolve() for p in sys.argv[1:]))
