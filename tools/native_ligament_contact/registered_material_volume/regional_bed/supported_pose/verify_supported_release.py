"""Check the coupled material support experiment against its native endpoint and controls."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(case, comparison, intersections, registration):
    launch = read(case / "launch.json")
    trial = read(comparison)["candidate"]
    surface = read(intersections)
    cert = read(registration)
    native_dir = Path(launch["argv"][0]).parent
    if digest(case / "launch.json") != trial["launch"]:
        raise ValueError("Comparison belongs to another native launch")
    if read(case / "execution.json")["exit"] != 0:
        raise ValueError("Incomplete native motion")
    step_count = surface["accepted_step"]
    dt = trial["dt_s"]
    if step_count != 64 or abs(step_count * dt - surface["accepted_time_s"]) > 1e-9:
        raise ValueError("Unmatched accepted motion horizon")
    if launch["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] != "1":
        raise ValueError("Missing variable exterior pressure")
    if cert["candidate_inc"] != digest(native_dir / "NumiHumanCorticalRegistration.inc"):
        raise ValueError("Ligament certificate belongs to another build")
    if cert["bone_count"] != 185 or len(cert["paths"]) != 6 or any(p["crossings"] for p in cert["paths"]):
        raise ValueError("Incomplete cortical registration")
    traces = case / "native-traces"
    receipt_path, = (traces / "accepted-geometry").glob("*.receipt.json")
    receipt = read(receipt_path)
    pack_hash = digest(Path(receipt["accepted_pack_path"]))
    if any(h != pack_hash for h in (receipt["pack_file_sha256"], surface["accepted_pack_sha256"], trial["pack"])):
        raise ValueError("Different native states in the geometry and force records")
    signals = trial["signals"]
    if any(signals[k] != [0., 0.] for k in ("root_assistance_n", "root_assistance_nm")):
        raise ValueError("Unexpected root support")
    if signals["pleural_pa"][0] >= signals["pleural_pa"][1]:
        raise ValueError("Pleural pressure did not vary")
    with (traces / "resting-com-support-impulses.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    forces = {}
    for name, index in (("right", 20), ("left", 30)):
        selected = [r for r in rows if int(r["contact_index"]) == index]
        if [int(r["accepted_step"]) for r in selected] != list(range(1, step_count + 1)):
            raise ValueError(f"Missing or duplicate accepted {name} heel samples")
        values = np.array([float(r["normal_impulse_ns"]) / dt for r in selected])
        if not np.isfinite(values).all() or values.min() <= 25.:
            raise ValueError(f"Insufficient {name} heel load")
        forces[name] = [float(values.min()), float(values.max())]
    log = (case / "stdout.log").read_text().splitlines()
    for gate in ("resting_FEM_terminal_audit=passed", "resting_fascia_fem_terminal_audit=passed"):
        if not any(line.startswith(gate) for line in log):
            raise ValueError(f"Missing native certificate {gate}")
    timing, = [line for line in log if line.startswith("resting_integrated_throughput ")]
    speed = float(re.search(r"\breal_time_factor=([.0-9e+-]+)", timing).group(1))
    binaries = [native_dir / "numi-human-native", native_dir / "metalrobo/HumanRestingSkin.metallib",
                native_dir / "metalrobo/MetalRobo.metallib"]
    for path in binaries:
        if digest(path) != launch["pins"][str(path)]:
            raise ValueError(f"Stale native binary {path}")
    return {
        "device": next(line.split("device=")[1].split(" world_fingerprint=")[0]
                       for line in log if "device=" in line and " world_fingerprint=" in line),
        "launch_sha256": trial["launch"], "endpoint_pack_sha256": pack_hash,
        "binary_sha256": {p.name: launch["pins"][str(p)] for p in binaries},
        "accepted_steps": step_count, "simulated_s": trial["time_s"], "process_wall_s": trial["walls"],
        "native_interval_real_time_factor": speed, "process_real_time_factor": trial["whole_RTF"],
        "heel_normal_force_N": forces, "final_bed_force_xyz_N": trial["last_contact_N"],
        "horizontal_bed_force_rms_N": trial["horizontal_contact_rms_N"],
        "terminal_horizontal_com_speed_m_s": trial["com_horizontal_speed_m_s"],
        "raw_skin_to_anatomy_pairs": sum(surface["crossing_pairs_by_semantic"].values()),
        "skin_self_pairs": surface["self_intersections"]["count"], "signals": signals,
    }


if __name__ == "__main__":
    print(json.dumps(verify(*(Path(arg) for arg in sys.argv[1:])), indent=2))
