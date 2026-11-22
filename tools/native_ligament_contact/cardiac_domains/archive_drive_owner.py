"""Archive the tested Human owner and verify exact clean extraction from its release asset."""

import hashlib, json, subprocess, sys, tarfile, tempfile
from pathlib import Path
from certify_heart import sha


def archive(build, result, proof):
    identities = json.loads((build / "identities.json").read_text())
    native = result / "terminal.receipt.json"
    launch = json.loads((result / "launch.json").read_text())
    binary = Path(launch["argv"][0])
    if (
        binary != build / "numi-human-native"
        or sha(binary) != identities["numi-human-native"]
        or json.loads(native.read_text())["accepted_step"] != 30000
    ):
        raise ValueError("Owner did not reach the endpoint")
    files = dict(identities)
    for p in [
        build / "identities.json",
        build / "build-commands.json",
        build / "numilab_human_myosim_visual_probe.mm",
        *build.glob("*.hpp"),
    ]:
        files[str(p.relative_to(build))] = sha(p)
    for n, d in files.items():
        if sha(build / n) != d:
            raise ValueError(n)
    repo = "Numi2/numilab-human"
    tag = "human-historical-support-checkpoints-20261122"
    gh = "/opt/homebrew/bin/gh"
    name = "native-cardiopulmonary-drive-owner-" + sha(binary)[:16] + ".tar.zst"
    with tempfile.TemporaryDirectory(dir=proof) as tmp:
        tmp = Path(tmp)
        payload = tmp / name
        with (
            payload.open("xb") as sink,
            subprocess.Popen(
                ["zstd", "-q", "-3", "-T1", "-c"], stdin=subprocess.PIPE, stdout=sink
            ) as z,
        ):
            with tarfile.open(fileobj=z.stdin, mode="w|", dereference=True) as t:
                for n in sorted(files):
                    t.add(build / n, arcname=n, recursive=False)
            z.stdin.close()
            if z.wait():
                raise ValueError("Compression failed")
        compressed = sha(payload)
        existing = json.loads(
            subprocess.check_output(
                [gh, "release", "view", tag, "--repo", repo, "--json", "assets"]
            )
        )
        if name not in [a["name"] for a in existing["assets"]]:
            subprocess.run(
                [gh, "release", "upload", tag, payload, "--repo", repo], check=True
            )
        dest = tmp / "download"
        dest.mkdir()
        subprocess.run(
            [
                gh,
                "release",
                "download",
                tag,
                "--repo",
                repo,
                "--dir",
                dest,
                "--pattern",
                name,
            ],
            check=True,
        )
        if sha(dest / name) != compressed:
            raise ValueError("Independent download differs")
        outputs = {}
        with subprocess.Popen(
            ["zstd", "-qdc", dest / name], stdout=subprocess.PIPE
        ) as dec:
            with tarfile.open(mode="r|", fileobj=dec.stdout) as t:
                for m in t:
                    if not m.isfile() or m.name not in files or m.name in outputs:
                        raise ValueError("Unexpected archive member")
                    outputs[m.name] = hashlib.file_digest(
                        t.extractfile(m), "sha256"
                    ).hexdigest()
            dec.stdout.read()
            if dec.wait() or outputs != files:
                raise ValueError("Recovered owner differs")
        (proof / "owner-recovery.json").write_text(
            json.dumps(
                dict(
                    url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                    compressed_sha256=compressed,
                    verified_member_sha256=outputs,
                    native_launch=sha(result / "launch.json"),
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    archive(*(Path(a).resolve() for a in sys.argv[1:]))
