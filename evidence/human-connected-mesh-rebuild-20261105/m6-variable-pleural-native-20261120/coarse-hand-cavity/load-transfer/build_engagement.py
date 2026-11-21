"""Recompile the pinned native owner with the corresponding authored ledger."""
from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import subprocess

from audit import digest
from native_trial import render_header


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("build", "reference", "fitted", "output"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    args.output.mkdir()
    filename = "numilab_human_myosim_visual_probe.mm"
    shutil.copyfile(args.build / filename, args.output / filename)
    args.output.joinpath("PosteriorBedReference.inc").write_text(
        render_header(args.reference, args.fitted))
    commands = [[arg.replace(str(args.build), str(args.output)) for arg in argv]
                for argv in json.loads((args.build / "build-commands.json").read_text())]
    args.output.joinpath("build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with args.output.joinpath("build.log").open("xb") as log:
        for command in commands:
            subprocess.run(command, stdout=log, stderr=log, check=True, timeout=180)
    evidence = {name: digest(args.output / name)
                for name in (filename, "PosteriorBedReference.inc", "numi-human-native")}
    args.output.joinpath("build-identity.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
