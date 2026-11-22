"""Bind the isolated native library, executable and shader to the existing physical human."""

import json
import os
from pathlib import Path
import subprocess
import sys
from material_context import sha


def stage(root, base):
    builds = root / "build"
    owner = root / "owner"
    pins = json.loads((builds / "build-pins.json").read_text())
    for name, d in pins.items():
        if sha(builds / name) != d:
            raise ValueError("Changed build " + name)
    old = base / "owner/native-owner"
    if not owner.exists():
        subprocess.run(["cp", "-cR", str(old), str(owner)], check=True)
    binary = owner / "numi-human-native"
    binary.unlink()
    os.link(builds / "numi-human-native", binary)
    ids = json.loads((old / "identities.json").read_text())
    ids["binary_sha256"] = sha(binary)
    ip = owner / "identities.json"
    ip.unlink()
    ip.write_text(json.dumps(ids, indent=2) + "\n")
    for name, mode in [("control-parent", "0"), ("surface-parent", "1")]:
        parent = root / name
        parent.mkdir(exist_ok=True)
        j = json.loads((base / "contact-parent/launch.json").read_text())
        env = j["environment"]
        key = "NUMI_MATTER_METALLIB_OVERRIDE"
        if sha(Path(env[key])) != j["pins"].pop(env[key]):
            raise ValueError("Different parent library")
        env[key] = str(builds / "NumiMatter.metallib")
        env["NM_FEM_REFERENCE_AREA_PLANE"] = mode
        env["DYLD_LIBRARY_PATH"] = str(builds)
        env["DYLD_PRINT_LIBRARIES"] = "1"
        j["pins"][env[key]] = pins["NumiMatter.metallib"]
        j["pins"][str(builds / "libmetalrobo.dylib")] = pins["libmetalrobo.dylib"]
        (parent / "launch.json").write_text(json.dumps(j, indent=2) + "\n")


if __name__ == "__main__":
    stage(*(Path(v).resolve() for v in sys.argv[1:]))
