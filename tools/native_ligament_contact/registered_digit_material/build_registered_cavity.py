"""Rebuild the registered Human owner using its pinned sources and existing Metal runtime."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(parent, out, patch, identities):
    if out.exists():
        raise FileExistsError(out)
    files = json.loads(identities.read_text())
    for name, pin in files["before"].items():
        if digest(parent / name) != pin:
            raise ValueError("Changed parent " + name)
    if patch.suffix == ".zst":
        with tempfile.TemporaryDirectory() as tmp:
            unpacked = Path(tmp) / "registered-cavity.patch"
            with unpacked.open("wb") as f:
                subprocess.run(["zstd", "-qdc", str(patch)], stdout=f, check=True)
            return prepare(parent, out, unpacked, identities)
    if digest(patch) != files["patch_sha256"]:
        raise ValueError("Patch hash mismatch")
    subprocess.run(["cp", "-cR", str(parent), str(out)], check=True)
    for options in (["--check"], []):
        subprocess.run(
            ["git", "apply", "--whitespace=error", *options, str(patch)],
            cwd=out,
            check=True,
        )
    for name, pin in files["after"].items():
        if digest(out / name) != pin:
            raise ValueError("Changed prepared source " + name)
    return files


def build(parent, out, patch, identities):
    prepare(parent, out, patch, identities)
    ids = json.loads((parent / "identities.json").read_text())
    if digest(parent / "numi-human-native") != ids["binary_sha256"]:
        raise ValueError("Parent executable differs")
    cmds = [
        [arg.replace(str(parent) + "/", str(out) + "/") for arg in command]
        for command in ids["commands"]
    ]
    for command in cmds:
        subprocess.run(command, check=True)
    sha = digest(out / "numi-human-native")
    (out / "identities.json").write_text(
        json.dumps(
            dict(
                ids,
                commands=cmds,
                parent_binary_sha256=ids["binary_sha256"],
                binary_sha256=sha,
                binary=sha,
                binary_path=str(out / "numi-human-native"),
                modified_header_sha256=digest(
                    out / "NumiHumanSkinFatFixedCavityReconditioning.hpp"
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    build(*(Path(x).resolve() for x in sys.argv[1:]))
