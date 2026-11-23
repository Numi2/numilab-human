"""Recoverable update to the shared dense parent with the native trial and material ledger attached."""

import json
import sys
from pathlib import Path
from archive_cavity_trial import archive_files, sha


def archive(fit, run, out):
    doc = json.loads((fit / "installation.json").read_text())
    native = json.loads((run / "launch.json").read_text())
    if (
        doc["binary"] != sha(Path(native["argv"][0]))
        or sha(fit / "current.node") != doc["loaded_pose"]
    ):
        raise ValueError("Different native trial")
    selected = (
        "fit.json",
        "preload.npz",
        "current.node",
        "layers.json",
        "installation.json",
        "owner/identities.json",
        "owner/numi-human-native",
        "owner/NumiHumanLoadedSkinPoseData.inc",
        "owner/numilab_human_myosim_visual_probe.mm",
    )
    files = {"fit/" + k: fit / k for k in selected}
    for p in run.rglob("*"):
        if p.is_file() and not p.name.endswith("github-recovery.json"):
            files["run/" + str(p.relative_to(run))] = p
    for name in (
        "fit_bonded_volume_modes.py",
        "install_bonded_volume_pose.py",
        "measure_fat_volume_forces.py",
        "archive_bonded_load.py",
    ):
        files["tools/" + name] = fit.parent / name
    archive_files(
        files,
        "bonded-material-start-" + sha(run / "launch.json")[:16] + ".tar.zst",
        out,
        "Complete changed source, physical input deck, native executable, attempt and material checks. Original assets and dependency hashes are pinned in the launch.",
    )


if __name__ == "__main__":
    archive(*(Path(s).resolve() for s in sys.argv[1:]))
