"""Independently recover the exact executable, loaded pose and accepted physical coordinates."""

import hashlib, json, subprocess, sys, tarfile, tempfile
from pathlib import Path
from material_context import sha


def archive(root):
    case = root / "routed-target-1.5mm"
    native = root / "verified-motion-one"
    out = native / "physical-recovery.json"
    proof = json.loads((case / "supported-registration.json").read_text())
    run = json.loads((native / "matched-digital-interfaces.json").read_text())
    source = {
        "native/numi-human-native": case / "native-owner/numi-human-native",
        "native/NumiHumanLoadedSkinPoseData.inc": case
        / "native-owner/NumiHumanLoadedSkinPoseData.inc",
        "native/identities.json": case / "native-owner/identities.json",
        "reference.node": case / "reference.node",
        "loaded.node": case / "current.node",
        "accepted.node": native / "physical-accepted.node",
        "field.npz": case / "field.npz",
        "supported-registration.json": case / "supported-registration.json",
        "launch.json": native / "launch.json",
    }
    expected = {k: sha(v) for k, v in source.items()}
    if [
        expected[k]
        for k in (
            "native/numi-human-native",
            "reference.node",
            "loaded.node",
            "accepted.node",
        )
    ] != [
        proof["native_binary"],
        proof["reference"],
        proof["current"],
        run["repaired"]["accepted_material"],
    ]:
        raise ValueError("Archive not bound to accepted physical run")
    tag = "human-closed-skin-diagnostic-packs-20261121"
    repo = "Numi2/numilab-human"
    gh = "/opt/homebrew/bin/gh"
    zstd = "/opt/homebrew/bin/zstd"
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmp = Path(tmp)
        f = tmp / (
            "registered-digits-physical-" + expected["accepted.node"][:16] + ".tar.zst"
        )
        with (
            f.open("xb") as sink,
            subprocess.Popen(
                [zstd, "-q", "-3", "-T1", "-c"], stdin=subprocess.PIPE, stdout=sink
            ) as proc,
        ):
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                for name, p in source.items():
                    tar.add(p, arcname=name, recursive=False)
            proc.stdin.close()
            if proc.wait():
                raise ValueError("Archive compression failed")
        assets = json.loads(
            subprocess.check_output(
                [gh, "release", "view", tag, "-R", repo, "--json", "assets"]
            )
        )["assets"]
        if f.name not in {v["name"] for v in assets}:
            subprocess.run(
                [gh, "release", "upload", tag, str(f), "-R", repo], check=True
            )
        recovered = tmp / "independent"
        recovered.mkdir()
        subprocess.run(
            [
                gh,
                "release",
                "download",
                tag,
                "-R",
                repo,
                "-p",
                f.name,
                "-D",
                str(recovered),
            ],
            check=True,
        )
        downloaded = recovered / f.name
        got = {}
        with subprocess.Popen(
            [zstd, "-qdc", str(downloaded)], stdout=subprocess.PIPE
        ) as proc:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for row in tar:
                    if row.name not in expected or row.name in got or not row.isfile():
                        raise ValueError("Unexpected archive member")
                    got[row.name] = hashlib.file_digest(
                        tar.extractfile(row), "sha256"
                    ).hexdigest()
            proc.stdout.read()
            if proc.wait() or got != expected:
                raise ValueError("Restored data differ")
        out.write_text(
            json.dumps(
                dict(
                    verified_member_sha256=got,
                    independent_download_sha256=sha(downloaded),
                    repository=repo,
                    tag=tag,
                    url=f"https://github.com/{repo}/releases/download/{tag}/{f.name}",
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    archive(Path(sys.argv[1]).resolve())
