#!/usr/bin/env python3
"""Run the matched feedback arm and bind its exact pre-step initial state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "remote_stand_trial.py"


def load_base():
    spec = importlib.util.spec_from_file_location("remote_stand_trial_base", BASE)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the bound native owner adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def read_terminal_state(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("stand_terminal_state="):
            state = json.loads(line.split("=", 1)[1])
            if state.get("schema") != "numi.human.legacy-stand-terminal.v1":
                raise ValueError("unexpected terminal-state schema")
            return state
    raise ValueError("native stdout has no terminal-state record")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("treatment",), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    args = parser.parse_args()
    base = load_base()
    run_dir = args.run.resolve()
    output = base.execute_arm(args.arm, run_dir, args.trial_id)
    raw = (run_dir / "native-stdout.txt").read_text()
    terminal = read_terminal_state(raw)
    if terminal.get("step_count") != 1600:
        raise ValueError("terminal-state endpoint differs from registered horizon")
    output["initial_q_sha256"] = canonical_sha(terminal["initial_q"])
    output["initial_v_sha256"] = canonical_sha(terminal["initial_v"])
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"native treatment adapter failed: {error}", file=sys.stderr)
        raise SystemExit(2)
