"""Verify that the loaded native runtime enforces the short-step latch."""
from argparse import ArgumentParser
from pathlib import Path
import json


def check(guarded, full):
    stages = json.loads(guarded.read_text())
    parity = json.loads(full.read_text())
    reference, candidate = stages["control"], stages["candidate"]
    assert parity["shared_accepted_steps"] == 90
    assert parity["first_accepted_difference"] is None
    assert not parity["rejections"]
    assert reference["accepted_steps"] == candidate["accepted_steps"] == 90
    assert reference["ligaments_passed"] == candidate["ligaments_passed"] == 34
    assert any("step=2 tier=0" in row for row in candidate["verified_rollbacks"])
    assert candidate["work"]["used_columns_sum"] < reference["work"]["used_columns_sum"]
    assert candidate["wall_s"] < reference["wall_s"]
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("guarded", type=Path)
    parser.add_argument("full", type=Path)
    args = parser.parse_args()
    assert check(args.guarded, args.full)
    print("Loaded guard preserves full-budget trajectory and rejects the unlatched trial")
