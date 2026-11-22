"""Recover the exact experimental Human executable and its preparation source from GitHub."""

import hashlib, json, subprocess, sys, tarfile, tempfile
from pathlib import Path


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(owner, trial, out):
    manifest = json.loads((owner / "identities.json").read_text())
    launch = json.loads((trial / "launch.json").read_text())
    exe = owner / "numi-human-native"
    if (
        str(exe) != launch["argv"][0]
        or sha(exe) != manifest["binary_sha256"]
        or sha(exe) != launch["pins"][str(exe)]
    ):
        raise ValueError("Native owner mismatch")
    proposal = Path(launch["environment"]["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    assert sha(proposal) == launch["pins"][str(proposal)]
    files = {
        p.name: p
        for p in owner.iterdir()
        if p.is_file()
        and (
            p.suffix in {".hpp", ".h", ".inc", ".metalinc", ".mm", ".metal"}
            or p.name in {"identities.json", "numi-human-native"}
        )
    }
    files["cavity.npz"] = proposal
    name = "dense-cavity-owner-" + sha(exe)[:16] + "-" + sha(proposal)[:12] + ".tar.zst"
    archive_files(
        files,
        name,
        out,
        "Executable, owner-local sources and exact tested proposal. External inputs and libraries are identified by the retained launch and its parent.",
    )


def archive_files(files, name, out, scope):
    pins = {k: sha(p) for k, p in files.items()}
    repo = "Numi2/numilab-human"
    tag = "human-historical-support-checkpoints-20261122"
    gh = "/opt/homebrew/bin/gh"
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        tmp = Path(tmp)
        payload = tmp / name
        with (
            payload.open("wb") as f,
            subprocess.Popen(
                ["zstd", "-q", "-3", "-T1", "-c"], stdin=subprocess.PIPE, stdout=f
            ) as z,
        ):
            with tarfile.open(fileobj=z.stdin, mode="w|", dereference=True) as tar:
                for k, p in sorted(files.items()):
                    tar.add(p, arcname=k, recursive=False)
            z.stdin.close()
            if z.wait():
                raise OSError("Archive compression failed")
        h = sha(payload)
        remote = f"https://github.com/{repo}/releases/download/{tag}/{name}"
        result = subprocess.run(
            [gh, "release", "upload", tag, str(payload), "-R", repo],
            capture_output=True,
            text=True,
        )
        if result.returncode and "already exists" not in result.stderr:
            raise RuntimeError(result.stderr)
        recovery = tmp / "independent"
        recovery.mkdir()
        subprocess.run(
            [
                gh,
                "release",
                "download",
                tag,
                "-R",
                repo,
                "-p",
                name,
                "-D",
                str(recovery),
            ],
            check=True,
        )
        downloaded = recovery / name
        if sha(downloaded) != h:
            raise ValueError("Independent archive digest mismatch")
        actual = {}
        with subprocess.Popen(
            ["zstd", "-dq", "-c", str(downloaded)], stdout=subprocess.PIPE
        ) as dec:
            with tarfile.open(fileobj=dec.stdout, mode="r|") as tar:
                for item in tar:
                    if (
                        not item.isfile()
                        or item.name not in files
                        or item.name in actual
                    ):
                        raise ValueError("Unexpected recovered member")
                    actual[item.name] = hashlib.file_digest(
                        tar.extractfile(item), "sha256"
                    ).hexdigest()
            dec.stdout.read()
            if dec.wait() or actual != pins:
                raise ValueError("Restored member differs")
        out.write_text(
            json.dumps(
                dict(
                    url=remote,
                    archive_sha256=h,
                    size_bytes=payload.stat().st_size,
                    verified_member_sha256=actual,
                    scope=scope,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    run(*(Path(s).resolve() for s in sys.argv[1:]))
