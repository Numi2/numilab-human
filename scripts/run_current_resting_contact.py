"""Replay the registered full-contact Human scene on its native Numi host."""

import argparse
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument(
        "--check",
        action="store_true",
        help="verify the native owner and all pinned inputs",
    )
    options.add_argument("--steps", type=int, help="number of 0.125 ms physical steps")
    options.add_argument(
        "--seconds",
        type=int,
        choices=(16, 60),
        help="requested integrated duration in seconds",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg_path = (
        repo
        / "evidence/human-connected-mesh-rebuild-20261105/m6-contact-feature-ownership-20261125/launch.json.gz"
    )
    payload = gzip.decompress(cfg_path.read_bytes())
    if (
        hashlib.sha256(payload).hexdigest()
        != "f2c198650665551959263aabb2cd967088cabf56292151f573e83eae90a3624b"
    ):
        raise SystemExit("The launch manifest differs from the verified checkpoint.")
    cfg = json.loads(payload)
    argv = cfg["argv"]
    settings = cfg["environment"]
    old_output = Path(argv[4]).parent
    steps = args.steps if args.steps is not None else 256
    if args.seconds is not None:
        ticks = args.seconds / Fraction(argv[argv.index("--muscle-step-seconds") + 1])
        if ticks.denominator != 1:
            parser.error("duration does not end on a physical step")
        steps = ticks.numerator
    if not 0 < steps < 2**32:
        parser.error("steps must be a positive 32-bit count")
    for name, expected in cfg["pins"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"Missing or changed native input: {path}")
    helper = (
        Path.home()
        / "Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/extended-equilibrium-027/pelvic-contact-response-048/repartition/digital-corridor-049/posterior-equilibrium-050/thoracic-bed-port-051/run_cavity_motion.py"
    )
    if (
        not helper.is_file()
        or digest(helper)
        != "624d95ee5f007fd8de4c2479db94ab968f335e6382c8c64587f03d42a63a6893"
    ):
        raise SystemExit(f"Missing or changed process monitor: {helper}")
    print(
        f"Verified {len(cfg['pins'])} native input identities. Physical timestep {argv[argv.index('--muscle-step-seconds') + 1]} s.",
        flush=True,
    )
    if args.check:
        return 0

    out = Path(tempfile.mkdtemp(prefix="current-contact-", dir=old_output.parent))
    cfg["parent_launch"] = hashlib.sha256(payload).hexdigest()
    argv[4] = str(out / "native-traces")
    for flag in ("--muscle-step-count", "--persistent-metal-stand-steps"):
        if flag in argv:
            argv[argv.index(flag) + 1] = str(steps)
    for name, value in list(settings.items()):
        path = Path(value)
        if (
            path.is_absolute()
            and path.is_relative_to(old_output)
            and value not in cfg["pins"]
        ):
            settings[name] = str(out / path.relative_to(old_output))
    settings["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = str(steps)
    (out / "launch.json").write_text(json.dumps(cfg, indent=2) + "\n")
    environ = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("NM_", "NUMI_", "DYLD_"))
    }
    environ.update(settings)
    spec = importlib.util.spec_from_file_location("resting_contact_monitor", helper)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    print(
        f"Running {steps} physical steps. Native capture and traces: {out}", flush=True
    )
    result = monitor.execute_native(
        argv,
        cfg["cwd"],
        environ,
        out,
        minimum_free_bytes=4 * 2**30,
        timeout_s=float("inf"),
    )
    print(json.dumps(result), flush=True)
    return result["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
