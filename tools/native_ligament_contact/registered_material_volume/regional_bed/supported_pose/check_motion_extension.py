"""Compare the accepted start of a longer native run and locate changes in bed support."""
import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def compare(short, long):
    manifest = json.loads((long / "launch.json").read_bytes())
    short_manifest = json.loads((short / "launch.json").read_bytes())
    args = manifest["argv"]
    short_args = short_manifest["argv"]
    if args[:4] != short_args[:4] or manifest["registered_regional_material"] != short_manifest["registered_regional_material"]:
        raise ValueError("Different physical owner")
    if manifest["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] != "1":
        raise ValueError("External pressure is fixed")
    reference = short / "native-traces"
    trial = long / "native-traces"
    short_momentum = read_csv(reference / "resting-com-momentum-diagnostic.csv")
    momentum = read_csv(trial / "resting-com-momentum-diagnostic.csv")
    steps = len(momentum)
    if steps < len(short_momentum) or [int(r["accepted_step"]) for r in momentum] != list(range(1, steps+1)):
        raise ValueError("Missing accepted momentum samples")
    compared = {}
    for name, ref, data, fields in (
        ("articulated", short_momentum, momentum, ("stand_q_fingerprint_fnv64", "stand_v_fingerprint_fnv64")),
        ("coupled", read_csv(reference / "resting-coupled.csv"), read_csv(trial / "resting-coupled.csv"),
         ("time_s", "pleural_pa", "lung_volume_ml", "PaO2_mmhg", "PaCO2_mmhg", "lv_mmhg", "rv_mmhg", "blood_error_ml")),
    ):
        if len(ref) != len(short_momentum) or len(data) != steps:
            raise ValueError("Accepted body and physiological clocks differ")
        compared[name] = sum(any(a[k] != b[k] for k in fields) for a, b in zip(ref, data))
        if name == "coupled" and any(float(r[k]) != 0 for r in data for k in ("root_assistance_n", "root_assistance_nm")):
            raise ValueError("Unexpected root assistance")
    bed = read_csv(trial / "resting-com-support-impulses.csv")
    if len(bed) != steps*32:
        raise ValueError("Incomplete bed impulse table")
    dt = float(args[args.index("--muscle-step-seconds")+1])
    vertical, horizontal, right, left = [], [], [], []
    for i in range(steps):
        block = bed[32*i:32*(i+1)]
        if ([int(r["contact_index"]) for r in block] != list(range(32)) or
                any(int(r["accepted_step"]) != i+1 or int(r["segment_steps"]) != 1 for r in block)):
            raise ValueError("Unexpected bed impulse ownership")
        force = [sum(float(r["impulse_world_"+axis+"_ns"]) for r in block)/dt for axis in "xyz"]
        if not all(map(math.isfinite, force)):
            raise ValueError("Nonfinite bed reaction")
        horizontal.append(math.hypot(*force[:2]))
        vertical.append(force[2])
        right.append(float(block[20]["normal_impulse_ns"])/dt)
        left.append(float(block[30]["normal_impulse_ns"])/dt)
    last = momentum[-1]
    return dict(requested_steps=int(args[args.index("--muscle-step-count")+1]), accepted_steps=steps,
                last_trace_s=float(last["time_s"]), matched_prefix_steps=len(short_momentum),
                different_prefix_steps=compared, zero_root_assistance=True,
                horizontal_force_rms_N=math.sqrt(sum(x*x for x in horizontal)/steps),
                vertical_force_range_N=[min(vertical), max(vertical)], final_vertical_force_N=vertical[-1],
                final_horizontal_speed_m_s=math.hypot(float(last["com_vx_m_s"]),float(last["com_vy_m_s"])),
                right_heel_range_N=[min(right),max(right)], left_heel_range_N=[min(left),max(left)],
                execution=json.loads((long / "execution.json").read_bytes()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("short", type=Path)
    parser.add_argument("long", type=Path)
    opts = parser.parse_args()
    print(json.dumps(compare(opts.short, opts.long), indent=2))
