"""Recompile the same Human entry point with a bounded, initial-state-only gravity observation."""

import json
from pathlib import Path
import subprocess
import sys
from material_context import sha


def build(root, reference, lib, stage):
    stage.mkdir(exist_ok=True)
    src = reference / "owner/native-owner"
    dest = stage / "owner"
    if not dest.exists():
        subprocess.run(["cp", "-cR", str(src), str(dest)], check=True)
    meta = json.loads((src / "identities.json").read_text())
    source = dest / "numilab_human_myosim_visual_probe.mm"
    original = source.read_text()
    anchor = "    model=std::move(rebased);"
    recovery = json.loads((reference.parent / "recovery.json").read_text())[
        "verified_member_sha256"
    ]
    if (
        original.count(anchor) != 1
        or sha(source)
        != recovery[
            "paired-reference/native-owner/numilab_human_myosim_visual_probe.mm"
        ]
    ):
        raise ValueError("Different tissue mass owner")
    source.write_text(
        original.replace(
            anchor, (root / "registered_gravity.inc").read_text() + "\n" + anchor
        )
    )
    commands = [
        [s.replace(str(src) + "/", str(dest) + "/") for s in c]
        for c in meta["commands"]
    ]
    build = lib / "build"
    for i, s in enumerate(commands[1]):
        if s.endswith("/app-runtime.mm.o"):
            commands[1][i] = str(build / "app-runtime.mm.o")
        if s.endswith("/libmetalrobo.dylib"):
            commands[1][i] = str(build / "libmetalrobo.dylib")
    cwd = json.loads((reference / "native-prefix/launch.json").read_text())["cwd"]
    for cmd in commands:
        subprocess.run(cmd, check=True, cwd=cwd)
    binary = dest / "numi-human-native"
    meta.update(
        commands=commands,
        source=sha(source),
        binary_path=str(binary),
        binary=sha(binary),
        binary_sha256=sha(binary),
        gravity_observer_source_sha256=sha(root / "registered_gravity.inc"),
    )
    (dest / "identities.json").write_text(json.dumps(meta, indent=2) + "\n")
    parent = stage / "parent"
    parent.mkdir(exist_ok=True)
    j = json.loads((lib / "surface-parent/launch.json").read_text())
    j["environment"]["NUMI_HUMAN_INITIAL_GRAVITY_LEDGER"] = "1"
    (parent / "launch.json").write_text(json.dumps(j, indent=2) + "\n")


if __name__ == "__main__":
    build(*(Path(s).resolve() for s in sys.argv[1:]))
