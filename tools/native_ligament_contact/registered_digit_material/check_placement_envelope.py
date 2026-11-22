"""Audit a proposed material placement against every native anatomical bank at the matched skeleton pose."""

import csv, hashlib, mmap, runpy, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from prepare_connected_pose import read, sha


def verify_outer_rows(node_map, expected):
    rows = np.full(len(expected), -1, dtype=np.int64)
    with node_map.open() as f:
        for r in csv.DictReader(f):
            if r["registered_surface_layer"] != "O":
                continue
            i = int(r["registered_surface_vertex_row"])
            node = int(r["joint_node_id"]) - 1
            if i < 0 or i >= len(rows) or (rows[i] != -1 and rows[i] != node):
                raise ValueError("Conflicting outer role map")
            rows[i] = node
    if np.any(rows < 0) or not np.array_equal(rows, expected):
        raise ValueError("Changed outer material rows")


def check(seed, physical, fine, compact, registration, impl, bank, out):
    out.mkdir(exist_ok=False)
    j = read(seed / "quality.json")
    trial = seed / "positions.npy"
    if sha(trial) != j["candidate_sha256"]:
        raise ValueError("Changed material proposal")
    installed = read(fine / "launch.json")["environment"]
    pfx = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_"
    facet = Path(installed[pfx + "FACES_CSV"])
    face = np.loadtxt(facet, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype="i4")
    original = np.load(physical)
    if sha(facet) != installed[pfx + "FACES_SHA256"] or not np.array_equal(
        face, original["Of"]
    ):
        raise ValueError("Changed physical outer topology")
    map_prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    node_map = Path(installed[map_prefix + "COMBINED_NODE_MAP_CSV"])
    if sha(node_map) != installed[map_prefix + "COMBINED_NODE_MAP_SHA256"]:
        raise ValueError("Changed registered node map")
    verify_outer_rows(node_map, original["Orows"])
    points = np.load(trial)[original["Orows"]]
    r = read(compact / "terminal.receipt.json")
    source = read(registration / "terminal.receipt.json")
    if (
        r["physical_endpoint"] != "accepted"
        or r["accepted_registered_body_poses"]
        != source["accepted_registered_body_poses"]
        or r["accepted_step"] != 30000
    ):
        raise ValueError("Mismatched skeletal endpoint")
    recover = read(compact / "frame-recovery.json")
    packed = compact / recover["url"].split("/")[-1]
    if (
        not recover["independently_restored"]
        or sha(packed) != recover["compressed_sha256"]
    ):
        raise ValueError("Changed compact native endpoint")
    sys.path.insert(0, str(impl.parent / "src"))
    mod = runpy.run_path(str(impl))
    audit = runpy.run_path(str(bank))["scan_banks"]
    with tempfile.TemporaryFile(dir=out) as f:
        subprocess.run(["zstd", "-qdc", packed], stdout=f, check=True)
        f.seek(0)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            if hashlib.sha256(raw).hexdigest() != r["pack_file_sha256"]:
                raise ValueError("Changed decoded endpoint")
            result = dict(
                pack_sha256=r["pack_file_sha256"],
                material_candidate_sha256=sha(trial),
                physical_layers_sha256=sha(physical),
                surface_topology_sha256=sha(facet),
                scope="Counterfactual material boundary versus accepted target anatomy in the same recorded skeletal pose. Geometry only; this proposed tissue state has not run in the native solver.",
            )
            poses = {p["body_index"]: p for p in r["accepted_body_poses"]}
            audit(
                mod,
                raw,
                poses,
                points,
                face,
                result,
                out,
                "envelope.json",
                check_skin=True,
            )
    subprocess.run(
        [
            sys.executable,
            bank.parent / "classify_registered_envelope.py",
            out / "envelope.json",
            compact / "stdout.log",
            out / "physical-banks.json",
            "--window-receipt",
            compact / "resting-ocular-sclera-window.json",
        ],
        check=True,
    )


if __name__ == "__main__":
    check(*(Path(s).resolve() for s in sys.argv[1:]))
