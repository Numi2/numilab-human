"""Run a cold native 60-second control or drive perturbation with the same certified owner."""

import json, shutil, subprocess, sys, tempfile
from pathlib import Path
from certify_heart import sha


def run(control, launcher, owner, out, treatment):
    if shutil.disk_usage(out.parent).free < 12 * 2**30:
        raise RuntimeError("Native trial requires 12 GiB free")
    if treatment not in ("control", "treatment"):
        raise ValueError("Unknown experiment")
    original = control / "config.json"
    settings = json.loads(original.read_text())
    baseline = json.loads((control / "launch.json").read_text())
    binary = owner / "numi-human-native"
    identities = json.loads((owner / "identities.json").read_text())
    for path, digest in identities.items():
        if sha(owner / path) != digest:
            raise ValueError("Changed owner input " + path)
    if (
        baseline["headless"]
        or settings["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] != "1"
        or "--resting-drive-intervention" in settings["argv"]
    ):
        raise ValueError("Incompatible supported control")
    for p in (
        "metalrobo/HumanRestingSkin.metallib",
        "metalrobo/MetalRobo.metallib",
        "numi-human-native",
    ):
        settings["pins"][str(owner / p)] = identities[p]
    if treatment == "treatment":
        settings["argv"].extend(["--resting-drive-intervention", "15", "35", "0.6"])
    out.mkdir(exist_ok=False)
    config = out / "config.json"
    config.write_text(json.dumps(settings, indent=2) + "\n")
    with tempfile.TemporaryDirectory(dir=out, prefix="native-") as dir:
        scratch = Path(dir)
        cmd = [
            sys.executable,
            str(launcher),
            str(config),
            str(binary),
            str(scratch),
            str(out),
            "30000",
        ]
        proc = subprocess.run(cmd)
        if proc.returncode:
            raise RuntimeError("Native transaction rejected; retained first witness")
        receipt = scratch / "native/accepted-geometry/step-30000.receipt.json"
        data = json.loads(receipt.read_text())
        if data["accepted_step"] != 30000 or data["physical_endpoint"] != "accepted":
            raise ValueError("Terminal receipt unavailable")
        shutil.copyfile(receipt, out / "terminal.receipt.json")
    (out / "source.json").write_text(
        json.dumps(
            dict(
                control=sha(original),
                launcher=sha(launcher),
                identity=sha(owner / "identities.json"),
                configuration=sha(config),
                receipt=sha(out / "terminal.receipt.json"),
                mode=treatment,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*(Path(a).resolve() for a in sys.argv[1:5]), sys.argv[5])
