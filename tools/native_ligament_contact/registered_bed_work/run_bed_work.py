"""Run a cold supported native trajectory with a bounded GPU virtual-work audit."""

import hashlib, json, os, shutil, subprocess, tempfile, time
from pathlib import Path


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(source, owner, out, steps, enabled=True, intermediate_steps=()):
    parent = json.loads((source / "launch.json").read_text())
    if parent["headless"] or steps <= 0 or steps > 32000 or steps % 32:
        raise ValueError(
            "Expected visible supported 2ms scene and complete 32-step samples"
        )
    for name, d in parent["pins"].items():
        if digest(name) != d:
            raise ValueError("Stale parent pin " + name)
    identities = json.loads((owner / "identities.json").read_text())
    for name, d in identities.items():
        if digest(owner / name) != d:
            raise ValueError("Stale owner pin " + name)
    argv = parent["argv"][:]
    if argv[argv.index("--muscle-step-seconds") + 1] != "0.002":
        raise ValueError("Different timestep")
    argv[0] = str(owner / "numi-human-native")
    argv[argv.index("--muscle-step-count") + 1] = str(steps)
    # Native captures use zero-based submission endpoints. The true terminal
    # accepted state is always available separately.
    cadence = 32
    if (
        len(intermediate_steps) > 6
        or len(set(intermediate_steps)) != len(intermediate_steps)
        or any(
            step <= 0 or step >= steps or (step + 1) % cadence
            for step in intermediate_steps
        )
    ):
        raise ValueError("Invalid intermediate native capture schedule")
    if intermediate_steps and os.environ.get("NUMI_RETAIN_SUPPORT_GEOMETRY") != "1":
        raise ValueError("Intermediate geometry was requested without retention")
    out.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(dir=out, prefix="live-") as tmp:
        live = Path(tmp) / "native"
        argv[4] = str(live)
        env = parent["environment"].copy()
        env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = ",".join(
            map(str, sorted((*intermediate_steps, steps)))
        )
        env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"] = str(out / "failure.json")
        if enabled:
            env["NUMI_HUMAN_THORACIC_BED_AUDIT_END_STEP"] = str(steps)
        if os.environ.get("NUMI_CAPTURE_BED_COLUMNS") == "1":
            env["NUMI_HUMAN_BED_QUADRATURE_CAPTURE"] = "1"
        pins = {**parent["pins"], **{str(owner / n): d for n, d in identities.items()}}
        (out / "launch.json").write_text(
            json.dumps(
                dict(
                    parent,
                    argv=argv,
                    environment=env,
                    pins=pins,
                    parent_launch=digest(source / "launch.json"),
                ),
                indent=2,
            )
            + "\n"
        )
        clean = {
            k: v for k, v in os.environ.items() if not k.startswith(("NUMI_", "NM_"))
        }
        start = time.monotonic()
        with (
            (out / "stdout.log").open("wb") as so,
            (out / "stderr.log").open("wb") as se,
        ):
            p = subprocess.run(
                argv,
                cwd=parent["cwd"],
                env=clean | env,
                stdout=so,
                stderr=se,
                timeout=90 + steps * 0.08,
            )
        wall = time.monotonic() - start
        (out / "execution.json").write_text(
            json.dumps(
                dict(exit_code=p.returncode, cold_wall_s=wall, requested_steps=steps),
                indent=2,
            )
            + "\n"
        )
        if p.returncode:
            raise ValueError(
                "Native process failed; see preserved stderr and failure receipt"
            )
        receipt = live / f"accepted-geometry/step-{steps}.receipt.json"
        r = json.loads(receipt.read_text())
        if r["physical_endpoint"] != "accepted" or r["accepted_step"] != steps:
            raise ValueError("No matching accepted physical endpoint")
        shutil.copy2(receipt, out / "terminal.receipt.json")
        if os.environ.get("NUMI_RETAIN_SUPPORT_GEOMETRY") == "1":
            for name, artifact, identity in (
                ("initial", live / "resting-human.mrvpack", r["base_pack_file_sha256"]),
                (
                    "accepted",
                    receipt.with_suffix("").with_suffix(".mrvpack"),
                    r["pack_file_sha256"],
                ),
            ):
                if digest(artifact) != identity:
                    raise ValueError("Native geometry digest mismatch")
                with (out / (name + ".mrvpack.zst")).open("wb") as archive:
                    subprocess.run(
                        ["zstd", "-q", "-3", "-T1", "-c", str(artifact)],
                        stdout=archive,
                        check=True,
                    )
            for sample in intermediate_steps:
                frame = live / f"accepted-geometry/step-{sample}.mrvpack"
                proof = frame.with_suffix(".receipt.json")
                observed = json.loads(proof.read_text())
                if (
                    observed["physical_endpoint"] != "accepted"
                    or observed["accepted_step"] != sample
                    or observed["pack_file_sha256"] != digest(frame)
                    or observed["base_pack_file_sha256"] != r["base_pack_file_sha256"]
                ):
                    raise ValueError("Intermediate accepted geometry is inconsistent")
                shutil.copy2(proof, out / proof.name)
                with (out / (frame.name + ".zst")).open("wb") as output:
                    subprocess.run(
                        ["zstd", "-q", "-3", "-T1", "-c", str(frame)],
                        stdout=output,
                        check=True,
                    )
            for name in (
                "resting-local-bed-sections.csv",
                "resting-ocular-sclera-window.json",
                "resting-local-bed-footprint.csv",
                "resting-support-point-motion.csv",
                "resting-com-momentum-diagnostic.csv",
            ):
                if (live / name).is_file():
                    shutil.copy2(live / name, out / name)
        needed = [
            "resting-coupled.csv",
            "resting-support-selected-skin-witness.csv",
            "resting-com-support-impulses.csv",
            "resting-surface-audit.csv",
        ]
        if enabled:
            needed.append("resting-thoracic-bed-work.csv")
        for name in needed:
            shutil.copy2(live / name, out / name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "owner", "out"):
        parser.add_argument(key, type=lambda v: Path(v).resolve())
    parser.add_argument("steps", type=int)
    parser.add_argument("--no-bed-work-audit", "--disabled", dest="enabled", action="store_false", help="Skip only the optional thoracic bed virtual-work observer.")
    parser.add_argument("--intermediate-steps", type=int, nargs="+", default=[])
    run(**vars(parser.parse_args()))
