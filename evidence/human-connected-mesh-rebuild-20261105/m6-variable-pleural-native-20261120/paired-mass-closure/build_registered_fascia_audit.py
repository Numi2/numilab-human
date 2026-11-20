"""Expose the fascia rest data actually passed to the native FEM owner."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import os
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--paired-offset", action="store_true")
parser.add_argument("--neighbors", action="store_true")
args = parser.parse_args()
prefix = ("fascia-neighbors-audit" if args.neighbors else
          "paired-offset-fascia-audit" if args.paired_offset else "registered-fascia-audit")
base = root.parent / "resting-clearance-and-atrium-015/staged-solver-accuracy-build"
build = root / (prefix + "-build")
build.mkdir()
name = "numilab_human_myosim_visual_probe.mm"
before = (base / name).read_text()
key = "    for (const auto& sourceTet : fascia.tetrahedra) {\n"
assert before.count(key) == 1
diagnostic = r"""    if (const char* path = std::getenv("NUMI_HUMAN_REGISTERED_FASCIA_AUDIT_CSV")) {
        std::ofstream audit(path);
        require(static_cast<bool>(audit), "could not open registered fascia audit");
        audit << "row,mapped_x,mapped_y,mapped_z,fixed\n" << std::setprecision(17);
        for (std::size_t i = 0; i < points.size(); ++i) {
            audit << i << ',' << points[i].x << ',' << points[i].y << ',' << points[i].z
                  << ',' << ((fascia.nodes[i].flags & 1u) != 0u) << '\n';
        }
        require(static_cast<bool>(audit), "could not write registered fascia audit");
    }
"""
if args.neighbors:
    diagnostic += (root / "registered_fascia_surrounding.inc").read_text()
source = before.replace(key, diagnostic + key)
if args.paired_offset:
    from paired_fascia_source_offsets import register_source_sheet
    source = register_source_sheet(source)
(build / name).write_text(source)
(root / (prefix + ".patch")).write_text("".join(difflib.unified_diff(
    before.splitlines(keepends=True), source.splitlines(keepends=True),
    "a/apps/" + name, "b/apps/" + name,
)))
for path in (*base.glob("*.hpp"), base / "runtime.mm.o"):
    os.link(path, build / path.name)
(build / "metalrobo").symlink_to((base / "metalrobo").resolve(), target_is_directory=True)
commands = [
    [arg.replace(str(base) + "/", str(build) + "/") for arg in command]
    for command in json.loads((base / "build-commands.json").read_text())
]
(build / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
with (root / (prefix + "-build.log")).open("wb") as log:
    for command in commands:
        subprocess.run(command, cwd=Path.home() / "Work/skin-fat-native-integration-209/build-native",
                       stdout=log, stderr=subprocess.STDOUT, check=True)
with (build / "numi-human-native").open("rb") as stream:
    print(hashlib.file_digest(stream, "sha256").hexdigest())
