"""Reproduce the first failed physical step with the owner's bounded contact and Newton telemetry."""

import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from run_bed_work import digest


def trace(parent, out, object_index=15):
    config = json.loads((parent / "launch.json").read_text())
    for name, pin in config["pins"].items():
        if digest(name) != pin:
            raise ValueError("Changed native input " + name)
    argv = config["argv"][:]
    if argv[argv.index("--muscle-step-count") + 1] != "2":
        raise ValueError("Expected the exact two-root stress test")
    out.mkdir(exist_ok=False)
    if shutil.disk_usage(out).free < 11 * 2**30:
        raise RuntimeError(
            "Insufficient space for the previously measured dense replay"
        )
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        scratch = Path(tmp)
        old = Path(argv[4]).parent
        argv[4] = str(scratch / "native")
        env = {
            k: v.replace(str(old) + "/", str(scratch) + "/")
            for k, v in config["environment"].items()
        }
        env["NUMI_HUMAN_REGISTERED_CELL_NODE_PREFIX"] = str(scratch / "physical")
        env.update(
            NM_FEM_NEWTON_TRACE_ROOT="1",
            NM_FEM_BACKTRACKING_TRACE_ROOT="1",
            NM_FEM_BACKTRACKING_TRACE_OBJECT=str(object_index),
            NM_FEM_BACKTRACKING_TRACE_NEWTON="15",
            NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT="1",
            NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT="1",
        )
        (out / "launch.json").write_text(
            json.dumps(
                dict(
                    config,
                    argv=argv,
                    environment=env,
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
                    raise RuntimeError("Dense replay resource limit")
                time.sleep(1)
        (out / "execution.json").write_text(
            json.dumps(
                dict(exit=p.returncode, wall_s=time.monotonic() - start), indent=2
            )
            + "\n"
        )
        for name in ("failure.json", "coupled.csv"):
            if (scratch / name).is_file():
                shutil.copy2(scratch / name, out / name)
        failure = json.loads((out / "failure.json").read_text())
        prior = json.loads((parent / "failure.json").read_text())
        for key in (
            "segment_start",
            "matter_code",
            "matter_diagnostics",
            "accepted_common_coordinates",
        ):
            if failure[key] != prior[key]:
                raise ValueError("Different physical event: " + key)
        # The terminal certificate stores a shared global residual on all
        # continuum objects. The winning status writer may differ; retain both.
        (out / "status-witness.json").write_text(
            json.dumps(
                dict(
                    parent_object=prior["matter_object_index"],
                    observed_object=failure["matter_object_index"],
                    resolved_scene_residual=failure["matter_diagnostics"],
                ),
                indent=2,
            )
            + "\n"
        )
        lines = [
            line
            for line in (out / "stderr.log").read_text().splitlines()
            if line.startswith(
                (
                    "fem_backtracking_trials=",
                    "fem_deformable_limit_trace=",
                    "matter_fgmres_active_work=",
                    "fem_newton_iterate=",
                    "fem_newton_terminal_certificate=",
                    "fem_deformable_limit_minimum=",
                    "fem_residual_armijo_trace_skipped=",
                )
            )
        ]
        (out / "trace.log").write_text("\n".join(lines) + "\n")
        if not lines:
            raise ValueError("Missing trace")


if __name__ == "__main__":
    trace(*(Path(s).resolve() for s in sys.argv[1:]))
