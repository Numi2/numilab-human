"""Exercise a fixed-cavity proposal through the same registered full-scene entrypoint."""

import hashlib, json, os, shutil, subprocess, tempfile, time
from pathlib import Path


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(parent, owner, proposal, out, diagnose=False, steps=2):
    config = json.loads((parent / "launch.json").read_text())
    argv = config["argv"].copy()
    ids = json.loads((owner / "identities.json").read_text())
    old = Path(argv[4]).parent
    for name, pin in config["pins"].items():
        if digest(name) != pin:
            raise ValueError("Changed native input " + name)
    if (
        config["pins"][argv[0]] != ids["parent_binary_sha256"]
        or digest(owner / "numi-human-native") != ids["binary_sha256"]
    ):
        raise ValueError("Changed owner")
    if argv[argv.index("--muscle-step-count") + 1] != "2":
        raise ValueError("Expected exact two-step event")
    if steps <= 0:
        raise ValueError("Expected a positive step count")
    argv[argv.index("--muscle-step-count") + 1] = str(steps)
    argv[0] = str(owner / "numi-human-native")
    out.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        scratch = Path(tmp)
        argv[4] = str(scratch / "native")
        env = {
            k: v.replace(str(old) + "/", str(scratch) + "/")
            for k, v in config["environment"].items()
        }
        env.update(
            NUMI_HUMAN_REGISTERED_CELL_NODE_PREFIX=str(scratch / "physical"),
            NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ=str(proposal),
            NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256=digest(proposal),
            NUMI_HUMAN_SKIN_FAT_FIXED_CAVITY_MASS_MOMENTS="1",
            NUMI_HUMAN_CAVITY_ROUTE_WITNESS="1",
        )
        if diagnose:
            env.update(
                NM_FEM_NEWTON_TRACE_ROOT="1",
                NM_FEM_BACKTRACKING_TRACE_ROOT="1",
                NM_FEM_BACKTRACKING_TRACE_OBJECT="15",
                NM_FEM_BACKTRACKING_TRACE_NEWTON="15",
                NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT="1",
                NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT="1",
            )
        pins = dict(config["pins"])
        pins[argv[0]] = ids["binary_sha256"]
        pins[str(proposal)] = digest(proposal)
        (out / "launch.json").write_text(
            json.dumps(
                dict(
                    config,
                    argv=argv,
                    environment=env,
                    pins=pins,
                    parent_launch=digest(parent / "launch.json"),
                ),
                indent=2,
            )
            + "\n"
        )
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("NUMI_", "NM_"))
        } | env
        start = time.monotonic()
        with (out / "stdout.log").open("w") as so, (out / "stderr.log").open("w") as se:
            p = subprocess.Popen(argv, cwd=config["cwd"], env=env, stdout=so, stderr=se)
            while p.poll() is None:
                if (
                    shutil.disk_usage(out).free < 2 * 2**30
                    or time.monotonic() - start > 900
                ):
                    p.terminate()
                    p.wait()
                    raise RuntimeError("Resource limit")
                time.sleep(1)
        (out / "execution.json").write_text(
            json.dumps(
                dict(exit=p.returncode, wall_s=time.monotonic() - start), indent=2
            )
            + "\n"
        )
        for f in scratch.iterdir():
            if f.is_file() and (
                f.suffix in (".json", ".csv") or f.name.endswith("accepted.node")
            ):
                shutil.copy2(f, out / f.name)
        traces = out / "native-traces"
        traces.mkdir(exist_ok=True)
        native = scratch / "native"
        for name in (
            "resting-coupled.csv",
            "resting-surface-audit.csv",
            "resting-com-support-impulses.csv",
            "resting-support-selected-skin-witness.csv",
        ):
            if (native / name).is_file():
                shutil.copy2(native / name, traces / name)
        if (native / "accepted-geometry").is_dir():
            for receipt in (native / "accepted-geometry").rglob("*.json"):
                target = traces / receipt.relative_to(native)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(receipt, target)
        if p.returncode == 0:
            with (out / "terminal.mrvpack.zst").open("wb") as sink:
                subprocess.run(
                    [
                        "zstd",
                        "-q",
                        "-3",
                        "-T1",
                        "-c",
                        str(native / "resting-human.mrvpack"),
                    ],
                    stdout=sink,
                    check=True,
                )
        if p.returncode:
            raise RuntimeError(
                "Native admission or motion rejected; inspect retained owner output"
            )
        import numpy as np

        with np.load(proposal) as arrays:
            count = len(arrays["replacement_tet_offsets"]) - 1
        if (
            f"registered_volume_fixed_cavities=PASS cavities={count} "
            not in (out / "stdout.log").read_text()
        ):
            raise ValueError("Cavities did not enter the native scene")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    for arg in ("parent", "owner", "proposal", "out"):
        parser.add_argument(arg, type=lambda x: Path(x).resolve())
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    run(**vars(parser.parse_args()))
