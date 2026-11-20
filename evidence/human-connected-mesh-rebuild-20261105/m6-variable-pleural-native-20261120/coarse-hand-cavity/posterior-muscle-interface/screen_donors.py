"""Compare source-registered and anatomically plausible sacral load paths."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from audit import BONE, digest


def load_screen(previous):
    # Keep the reference screening owner and make the only scoped override
    # explicit for the alternative donor inventory.
    source = previous.read_text()
    old = 'ply_gap=0.05):'
    assert source.count(old) == 1
    source = source.replace(old, 'ply_gap=0.05, sacrum=None):')
    old = 'banks = bone[np.isin(bone["body"], available)]'
    assert source.count(old) == 1
    source = source.replace(old, '''mask = np.isin(bone["body"], available)
        if sacrum is not None:
            mask |= (bone["body"] == sacrum[0]) & (bone["bone"] == sacrum[1])
        banks = bone[mask]''')
    namespace = {"__name__":"posterior_donor_screen"}
    exec(compile(source, str(previous), "exec"), namespace)
    return namespace["paths"]


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("prior", "reference", "repaired", "bone", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run = args.reference / "native-candidate-128"
    launch = json.loads((run / "launch.json").read_text())
    bone_payload = next(Path(value) for value in launch["argv"] if value.endswith(".nhbones"))
    manifest = bone_payload.with_suffix(".manifest.json")
    source = json.loads(manifest.read_text())["source"]["anchors"]
    entries = [row for row in source if row["myosim_body"] == "sacrum"]
    assert len(entries) == 1
    record = entries[0]
    sacrum = (record["core_body_index"], record["source_record_index"])
    bone = np.memmap(args.bone, mode="r", dtype=BONE)
    assert len(bone[(bone["body"] == sacrum[0]) & (bone["bone"] == sacrum[1])]) > 0
    screen_path = args.prior / "screen_gluteal_columns.py"
    screen = load_screen(screen_path)
    crossings = json.loads((args.prior / "bone-endpoint-audit.json").read_text())
    reference = json.loads((args.prior / "gluteal-column-screen.json").read_text())
    result = {"sacrum_source": record, "manifest_sha256":digest(manifest), "screen_sha256":digest(screen_path)}
    for label, geometry in (("original", args.reference), ("repaired", args.repaired)):
        result[label] = {}
        for donors, extra in (("registered", None), ("plus_sacrum", sacrum)):
            observation = screen(run, args.bone, geometry, crossings, sacrum=extra)
            if label == "original" and donors == "registered":
                assert observation == reference
            observation["metrics"] = dict(
                rows=sum(len(b["rows"]) for b in observation["banks"]),
                sacrum_rows=sum((r["bone_body"], r["bone"]) == sacrum
                                for b in observation["banks"] for r in b["rows"]),
                upper_sum_projected_area_cm2=1e4 * sum(r["area_m2"] for b in observation["banks"] for r in b["rows"]),
            )
            print(label, donors, json.dumps(observation["metrics"]), flush=True)
            result[label][donors] = observation
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
