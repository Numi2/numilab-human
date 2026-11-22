"""Exercise a fixed-cavity proposal through the same registered full-scene entrypoint."""

import hashlib, json, os, shutil, subprocess, tempfile, time
from pathlib import Path


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def execute_native(argv, cwd, env, out, minimum_free_bytes=2 * 2**30, timeout_s=900):
    start = time.monotonic()
    limit = None
    with (out / "stdout.log").open("w") as so, (out / "stderr.log").open("w") as se:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=so, stderr=se)
        while process.poll() is None:
            free = shutil.disk_usage(out).free
            if free < minimum_free_bytes:
                limit = dict(
                    stop_reason="disk_reserve",
                    available_bytes=free,
                    min_free_bytes=minimum_free_bytes,
                )
            elif time.monotonic() - start > timeout_s:
                limit = dict(stop_reason="wall_time", timeout_s=timeout_s)
            if limit:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    result = dict(exit=process.returncode, wall_s=time.monotonic() - start)
    if limit:
        result.update(limit)
    (out / "execution.json").write_text(json.dumps(result, indent=2) + chr(10))
    return result


def copy_recruitment_cap(argv, reference, fraction=1.0):
    """Use the same biological recruitment bound as a recorded supported run."""
    control = json.loads((reference / "launch.json").read_text())
    receipt = json.loads((reference / "terminal.receipt.json").read_text())
    if receipt["physical_endpoint"] != "accepted" or receipt["accepted_time_s"] < 16:
        raise ValueError("Expected an accepted supported control")
    for suffix in (".nhrigid", ".nhmyo", ".nheq"):
        (a,) = [v for v in argv if v.endswith(suffix)]
        (b,) = [v for v in control["argv"] if v.endswith(suffix)]
        if digest(a) != digest(b) or control["pins"][b] != digest(b):
            raise ValueError("Different musculoskeletal owner " + suffix)
    flag = "--muscle-activation"
    cap = control["argv"][control["argv"].index(flag) + 1]
    if (
        not 0 < float(cap) <= 1
        or argv.count(flag) != 1
        or float(argv[argv.index(flag) + 1]) != 0
    ):
        raise ValueError("Incorrect baseline activation configuration")
    if not 0 < fraction <= 1:
        raise ValueError(
            "Recruitment sensitivity must stay within the tested control bound"
        )
    # The native source optimizer chooses the individual muscle activations.
    selected = float(cap) * fraction
    argv[argv.index(flag) + 1] = cap if fraction == 1 else repr(selected)
    return dict(
        scale=fraction,
        selected_cap=selected,
        launch_sha256=digest(reference / "launch.json"),
        accepted_receipt_sha256=digest(reference / "terminal.receipt.json"),
        activation_cap=float(cap),
    )


def run(
    parent,
    owner,
    proposal,
    out,
    diagnose=False,
    steps=2,
    trace_root=None,
    recruitment_reference=None,
    recruitment_fraction=1.0,
    capture_fem_body_load=False,
):
    config = json.loads((parent / "launch.json").read_text())
    argv = config["argv"].copy()
    if recruitment_reference is not None:
        config["recruitment_reference"] = copy_recruitment_cap(
            argv, recruitment_reference, recruitment_fraction
        )
    elif recruitment_fraction != 1.0:
        raise ValueError("A sensitivity scale requires a matched control")
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
        if capture_fem_body_load:
            env["NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE"] = "1"
        if env.get("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS") != "2":
            raise ValueError("Unexpected capture schedule in the parent")
        env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = str(steps)
        if diagnose or trace_root is not None:
            root = 1 if trace_root is None else trace_root
            if root < 0 or root >= steps:
                raise ValueError("Trace root outside requested horizon")
            env.update(
                NM_FEM_NEWTON_TRACE_ROOT=str(root),
                NM_FEM_BACKTRACKING_TRACE_ROOT=str(root),
                NM_FEM_BACKTRACKING_TRACE_OBJECT="15",
                NM_FEM_BACKTRACKING_TRACE_NEWTON="15",
                NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT=str(root),
                NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT=str(root),
            )
        pins = dict(config["pins"])
        pins[argv[0]] = ids["binary_sha256"]
        pins[str(proposal)] = digest(proposal)
        override = ids.get("loaded_pose_override")
        if override:
            key = override["key"]
            pose = Path(override["path"])
            if (
                key != "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"
                or pins[env[key]] != override["parent_sha256"]
                or digest(pose) != override["sha256"]
                or digest(owner / "NumiHumanLoadedSkinPoseData.inc")
                != override["header_sha256"]
            ):
                raise ValueError("Invalid compiled interior pose override")
            env[key] = str(pose)
            pins[str(pose)] = override["sha256"]
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
        execution = execute_native(argv, config["cwd"], env, out)
        if "stop_reason" in execution:
            # Preserve the small native failure receipt on a forced stop. A
            # prefix can contain accepted steps even if the final export failed.
            failure = scratch / "failure.json"
            if failure.is_file():
                shutil.copy2(failure, out / failure.name)
            raise RuntimeError(
                "Native run stopped by "
                + execution["stop_reason"]
                + "; see execution.json"
            )
        for f in scratch.iterdir():
            if f.is_file() and (
                f.suffix in (".json", ".csv")
                or f.name in ("physical-reference.node", "physical-accepted.node")
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
        if execution["exit"] == 0:
            receipt = native / "accepted-geometry" / f"step-{steps}.receipt.json"
            state = json.loads(receipt.read_text())
            pack = native / "accepted-geometry" / f"step-{steps}.mrvpack"
            if (
                state["accepted_step"] != steps
                or state["physical_endpoint"] != "accepted"
                or state["surface_audit_endpoint"] != "passed"
                or Path(state["accepted_pack_path"]) != pack
                or digest(pack) != state["pack_file_sha256"]
            ):
                raise ValueError("Missing or mismatched terminal physical state")
            dst = out / "accepted-terminal.mrvpack.zst"
            with dst.open("wb") as sink:
                subprocess.run(
                    ["zstd", "-q", "-3", "-T1", "-c", str(pack)],
                    stdout=sink,
                    check=True,
                )
            shutil.copy2(receipt, out / "terminal.receipt.json")
            (out / "admission.json").write_text(
                json.dumps(
                    dict(
                        exit_code=0,
                        accepted_steps=steps,
                        requested_roots=steps,
                        wall_seconds=json.loads((out / "execution.json").read_text())[
                            "wall_s"
                        ],
                        terminal_pack_sha256=state["pack_file_sha256"],
                        terminal_archive_sha256=digest(dst),
                    ),
                    indent=2,
                )
                + "\n"
            )
            ocular = native / "resting-ocular-sclera-window.json"
            if ocular.is_file():
                shutil.copy2(ocular, traces / ocular.name)
        if execution["exit"]:
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
    parser.add_argument("--capture-fem-body-load", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--trace-root", type=int)
    parser.add_argument("--recruitment-reference", type=lambda x: Path(x).resolve())
    parser.add_argument("--recruitment-fraction", type=float, default=1.0)
    run(**vars(parser.parse_args()))
