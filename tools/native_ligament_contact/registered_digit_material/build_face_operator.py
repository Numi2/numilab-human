"""Build the registered plane response against the exact Matter source and compiler options."""

import json
import subprocess
import sys
from pathlib import Path

from material_context import sha


def build(identities, dest):
    record = json.loads(identities.read_text())
    root = Path(record["source"]).parents[2]
    if sha(Path(record["source"])) != record["source_sha256"]:
        raise ValueError("Shared solver source changed")
    c, link = record["commands"]
    c = c[: c.index("-c")] + [
        "-I",
        str(root),
        "-c",
        str(dest / "registered_bed_face.metal"),
        "-o",
        str(dest / "face.air"),
    ]
    link = link[: link.index("metallib") + 1] + [
        str(dest / "face.air"),
        "-o",
        str(dest / "face.metallib"),
    ]
    host = [
        "xcrun",
        "clang++",
        "-O2",
        "-std=c++17",
        "-fobjc-arc",
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        str(dest / "run_registered_bed_face.mm"),
        "-o",
        str(dest / "face-check"),
    ]
    for cmd in (c, link, host):
        subprocess.run(cmd, check=True)
    out = {
        "commands": [c, link, host],
        "contact": sha(root / "src/metal/contact.metalinc"),
        "kernel": sha(dest / "registered_bed_face.metal"),
        "library": sha(dest / "face.metallib"),
        "host": sha(dest / "face-check"),
    }
    (dest / "build.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    build(*(Path(s).resolve() for s in sys.argv[1:]))
