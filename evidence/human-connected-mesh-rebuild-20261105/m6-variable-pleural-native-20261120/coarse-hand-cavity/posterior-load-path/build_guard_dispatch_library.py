"""Build the shared Matter owner that the native launch actually loads."""
from argparse import ArgumentParser
import difflib
import json
from pathlib import Path
import subprocess

from build_guard_dispatch_app import patch
from replay_outlier import digest


def build(runtime, recipes, out, cwd):
    out.mkdir()
    original = runtime.read_text()
    text = patch(original)
    (out / "runtime.mm").write_text(text)
    (out / "short-guard-runtime.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), text.splitlines(keepends=True),
        "a/matter/src/runtime.mm", "b/matter/src/runtime.mm")))
    names = {"runtime.mm", "runtime.mm.o", "libmetalrobo.dylib"}
    options = json.loads(recipes.read_text())
    source_command = next(command for command in options if str(recipes.parent / "runtime.mm") in command)
    link_command = next(command for command in options if str(recipes.parent / "libmetalrobo.dylib") in command)
    commands = [[str(out / Path(p).name) if Path(p).parent == recipes.parent and Path(p).name in names
                 else p for p in command] for command in (source_command, link_command)]
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with (out / "build.log").open("xb") as log:
        for command in commands:
            subprocess.run(command, cwd=cwd, stdout=log, stderr=log, check=True, timeout=240)
    sha = {"parent_runtime": digest(runtime), "patched_runtime": digest(out / "runtime.mm"),
           "library": digest(out / "libmetalrobo.dylib")}
    (out / "build.json").write_text(json.dumps(sha, indent=2) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("runtime", "recipes", "out", "cwd"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    build(**vars(parser.parse_args()))
