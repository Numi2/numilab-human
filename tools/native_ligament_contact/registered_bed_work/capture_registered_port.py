"""Recapture the actual static respiratory registration from a short native supported run."""

import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from certify_heart import sha


def capture(source, out):
    launch = json.loads((source / "launch.json").read_text())
    previous = json.loads((source / "terminal.receipt.json").read_text())
    if previous["accepted_step"] != 30000 or launch["headless"]:
        raise ValueError("Expected the supported normal-drive trajectory")
    for name, d in launch["pins"].items():
        if sha(Path(name)) != d:
            raise ValueError("Changed launch payload " + name)
    out.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        argv = launch["argv"][:]
        argv[4] = str(Path(tmp) / "native")
        argv[argv.index("--muscle-step-count") + 1] = "64"
        env = launch["environment"].copy()
        env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = "64"
        env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"] = str(out / "failure.json")
        (out / "launch.json").write_text(
            json.dumps(
                dict(
                    launch,
                    argv=argv,
                    environment=env,
                    parent_launch=sha(source / "launch.json"),
                ),
                indent=2,
            )
            + "\n"
        )
        clean = {
            k: v for k, v in os.environ.items() if not k.startswith(("NUMI_", "NM_"))
        }
        with (
            (out / "stdout.log").open("wb") as so,
            (out / "stderr.log").open("wb") as se,
        ):
            subprocess.run(
                argv,
                cwd=launch["cwd"],
                env=clean | env,
                stdout=so,
                stderr=se,
                timeout=140,
                check=True,
            )
        end = Path(argv[4]) / "accepted-geometry/step-64.receipt.json"
        r = json.loads(end.read_text())
        if r["physical_endpoint"] != "accepted" or r["accepted_step"] != 64:
            raise ValueError("Missing native transaction")
        shutil.copy2(end, out / "terminal.receipt.json")
        for k in ("vertex_map", "anatomy_parameters"):
            item = r["skin_source_mapping"][k]
            p = Path(item["path"])
            if (
                item["sha256"] != previous["skin_source_mapping"][k]["sha256"]
                or sha(p) != item["sha256"]
            ):
                raise ValueError("Different respiratory registration " + k)
            shutil.copy2(p, out / p.name)
        for name in (
            "resting-coupled.csv",
            "resting-body-motion-partition.csv",
            "resting-support-selected-skin-witness.csv",
            "resting-com-support-impulses.csv",
        ):
            shutil.copy2(Path(argv[4]) / name, out / name)


if __name__ == "__main__":
    capture(*(Path(s).resolve() for s in sys.argv[1:]))
