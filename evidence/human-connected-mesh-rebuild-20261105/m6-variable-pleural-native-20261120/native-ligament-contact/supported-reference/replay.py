"""Repeat the best supported control with the fine reference's source mass."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, doc):
    path.write_text(json.dumps(doc, indent=2) + "\n")


def replay(control, fine_logs, workspace, evidence):
    launch_path = control / "launch.json"
    config = json.loads(launch_path.read_text())
    before_argv = config["argv"].copy()
    assert not config["headless"]
    assert config["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1"
    assert before_argv[before_argv.index("--muscle-step-count") + 1] == "8000"
    assert before_argv[before_argv.index("--muscle-step-seconds") + 1] == "0.002"
    assert "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS" not in config["environment"]
    decoder_path = control / "native/resting-human.mrvpack.zst"
    restored = workspace / "resting-human.mrvpack"
    with restored.open("wb") as out:
        subprocess.run(["zstd", "-dc", str(decoder_path)], stdout=out, check=True)
    old_cache = config["environment"]["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"]
    identity = config["environment"]["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"]
    assert config["pins"].pop(old_cache) == digest(restored) == identity
    config["environment"]["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"] = str(restored)
    config["pins"][str(restored)] = identity

    fine_launch = fine_logs / "launch.json"
    target_argv = json.loads(fine_launch.read_text())["argv"]
    target_mass = target_argv[target_argv.index("--resting-reference-mass-kg") + 1]
    assert 97.13 < float(target_mass) < 97.14
    mass_slot = before_argv.index("--resting-reference-mass-kg") + 1
    assert float(before_argv[mass_slot]) == 72.0
    config["argv"][mass_slot] = target_mass
    native = workspace / "native"
    config["argv"][4] = str(native)
    assert all(a == b for i, (a,b) in enumerate(zip(before_argv,config["argv"],strict=True))
               if i not in (4, mass_slot))
    config["matched_reference"] = {
        "supported_72kg_launch_sha256": digest(launch_path),
        "full_volume_launch_sha256": digest(fine_launch),
        "source_mass_kg": target_mass,
        "control_mass_kg": before_argv[mass_slot],
        "initial_cache_archive_sha256": digest(decoder_path),
    }
    for path, sha in config["pins"].items():
        assert digest(Path(path)) == sha, path
    for name in ("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT",):
        if name in config["environment"]:
            config["environment"][name] = str(evidence / "failure.json")
    config["environment"]["NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT"] = "1"
    save(evidence / "launch.json", config)
    child_environment = {k:v for k,v in os.environ.items() if not k.startswith(("NUMI_", "NM_"))}
    started = time.monotonic()
    reason = None
    with (evidence / "stdout.log").open("wb") as out, (evidence / "stderr.log").open("wb") as err:
        with subprocess.Popen(config["argv"], env=child_environment | config["environment"],
                              cwd=config["cwd"], stdout=out, stderr=err) as child:
            while child.poll() is None:
                if shutil.disk_usage("/System/Volumes/Data").free < 1024**3:
                    reason = "system disk watchdog at 1 GiB"
                if time.monotonic() - started > 360:
                    reason = "360 second wall budget"
                if reason:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                    break
                time.sleep(.2)
            code = child.wait()
    save(evidence / "execution.json", {"wall_seconds": time.monotonic() - started,
                                      "native_exit_code": code, "watchdog": reason})
    for name in ("resting-coupled.csv", "resting-com-momentum-diagnostic.csv",
                 "resting-com-support-impulses.csv", "resting-surface-audit.csv",
                 "resting-local-bed-sections.csv", "resting-body-motion-partition.csv", "resting-support-point-motion.csv", "resting-support-selected-skin-witness.csv"):
        if (native / name).is_file():
            shutil.copyfile(native / name, evidence / name)
    print(code)


if __name__ == "__main__":
    replay(*map(Path, sys.argv[1:]))
