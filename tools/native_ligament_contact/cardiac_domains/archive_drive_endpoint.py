"""Retain a source-pinned accepted native frame in the project's existing recovery release."""

import json, subprocess, sys, tempfile
from pathlib import Path
from certify_heart import sha


def pack(case):
    source = case / "terminal.mrvpack"
    doc = json.loads((case / "terminal.receipt.json").read_text())
    d = sha(source)
    if (
        doc["pack_file_sha256"] != d
        or doc["physical_endpoint"] != "accepted"
        or doc["accepted_step"] != 30000
    ):
        raise ValueError("Unmatched endpoint")
    repo = "Numi2/numilab-human"
    tag = "human-historical-support-checkpoints-20261122"
    asset = case / ("supported-" + case.name + "-" + d[:16] + ".mrvpack.zst")
    if not asset.exists():
        subprocess.run(["zstd", "-q", "-4", source, "-o", asset], check=True)
    gh = "/opt/homebrew/bin/gh"
    existing = json.loads(
        subprocess.check_output(
            [gh, "release", "view", tag, "--repo", repo, "--json", "assets"]
        )
    )
    if asset.name not in [a["name"] for a in existing["assets"]]:
        subprocess.run(
            [gh, "release", "upload", tag, asset, "--repo", repo], check=True
        )
    with tempfile.TemporaryDirectory(dir=case) as tmp:
        subprocess.run(
            [
                gh,
                "release",
                "download",
                tag,
                "--repo",
                repo,
                "--pattern",
                asset.name,
                "--dir",
                tmp,
            ],
            check=True,
        )
        fetched = Path(tmp) / asset.name
        if sha(fetched) != sha(asset):
            raise ValueError("GitHub download differs")
        import hashlib

        with subprocess.Popen(["zstd", "-qdc", fetched], stdout=subprocess.PIPE) as z:
            if hashlib.file_digest(z.stdout, "sha256").hexdigest() != d or z.wait():
                raise ValueError("Restored frame differs")
    receipt = dict(
        source=doc["accepted_pack_path"],
        source_sha256=d,
        compressed_sha256=sha(asset),
        independently_restored=True,
        url=f"https://github.com/{repo}/releases/download/{tag}/{asset.name}",
    )
    (case / "frame-recovery.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt["url"], flush=True)


if __name__ == "__main__":
    pack(Path(sys.argv[1]).resolve())
