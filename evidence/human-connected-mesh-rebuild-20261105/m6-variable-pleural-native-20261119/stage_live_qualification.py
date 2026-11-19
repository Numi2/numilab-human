"""Stage compact reproducibility evidence for the native variable-pleural run."""
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np

from fit_cardiac_volume_grid import HERE, PREVIOUS, hash_file
from verify_cardiac_tube_membership import read_tubes


PUB = HERE / "publication-live"
BUILD = HERE / "cardiac-split-tube-native-build"
PROVED = HERE / "cardiac-conditioned-live-audit"
RUN = "cardiac-split-native-8000"
REUSE = "cardiac-split-native-8000-reuse"
GUARDED = "cardiac-split-native-8000-guarded"
WITNESS = "cardiac-split-native-2780-terminal"
ROLLBACK = "cardiac-split-native-32-resting-rollback"
REG = HERE.parent / "resting-clearance-and-atrium-002"
SHADER = HERE.parent / "coupled-thumb-reference-001"


def dest(loc):
    path = PUB / loc
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def text(source, path):
    raw = source.read_text()
    dest(path).write_text("\n".join(line.rstrip() for line in raw.splitlines()).rstrip()+"\n")


def compact(source, path):
    raw = source.read_bytes()
    target = dest(path)
    target.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    assert gzip.decompress(target.read_bytes()) == raw


def code_patch(base, candidate):
    p = subprocess.run(["/usr/bin/diff", "-u", str(base), str(candidate)], capture_output=True)
    assert p.returncode == 1,(base, candidate, p.stderr)
    out = p.stdout.decode("utf-8")
    raw = gzip.compress(out.encode("utf-8"),compresslevel=6,mtime=0)
    dest("runtime-diffs/"+candidate.name+".patch.gz").write_bytes(raw)
    return dict(base=str(base),base_sha256=hash_file(base),
                candidate_sha256=hash_file(candidate))


def lines(run, path, prefixes):
    logfile = HERE / run / path
    entries = [line for line in logfile.read_text().splitlines() if line.startswith(prefixes)]
    assert entries, (run, path, prefixes)
    dest(run/"selected.log").write_text("\n".join(entries)+"\n")


def main():
    PUB.mkdir(exist_ok=False)
    include=(
        "build_cardiac_tube_trial.py", "run_cardiac_split_trial.py",
        "summarize_cardiac_native_run.py", "verify_native_cardiac_split.py",
        "cardiac_trajectory_conditioned_audit.cpp", "cardiac_trajectory_refined_audit.cpp",
        "cardiac_certified_tube.metalinc", "prepare_cardiac_live_tubes.py",
        "verify_cardiac_tube_membership.py", "capture_cardiac_targets.py",
        "cardiac_coordinates_probe.swift", "cardiac_coordinates_probe.metal",
        "cardiac-live-membership.json", "cardiac-live-tube-tight-atlas.json",
        "cardiac-trajectory-fixtures.log", "fit_connected_skin_offset.py",
        "fit_connected_skin_vectors.py", "constrain_skin_vector_orientation.py",
        "skin-connected-offset.json", "skin-connected-vectors.json", "skin-connected-oriented.json",
        "compare_native_export_reuse.py", "native-export-reuse-comparison.json",
        "compare_historical_outer_surface.py", "historical-outer-registration.json",
        "verified-native-capture-dedup.json", "stage_live_qualification.py",
        "verified-compressed-dedup.json", "qualification-live.md",
        "guarded-variable-pressure-comparison.json", "compare_guarded_variable_pressure.py",
        "constrain_skin_vector_self.py", "skin-connected-self-fit.json")
    for name in include:
        f = HERE/name
        if f.is_file():
            text(f, "README.md" if name=="qualification-live.md" else name)
    for name in ("tube-contract.json","cardiac_proved_table.metalinc", "build-verification.json",
                 "build-commands.json","skin-build-commands.json"):
        text(BUILD/name,"compiled/"+name)
    for name in ("cardiac-split-native-build-atlas79.log","cardiac-split-native-build-reuse.log",
                 "cardiac-split-native-build-guarded.log"):
        text(HERE/name,"compiled/"+name)
    qualified = []
    for log_path in sorted(PROVED.glob("*.log")):
        text(log_path,"continuous-proof/"+log_path.name)
        for line in log_path.read_text().splitlines():
            counts = tuple(map(int,line.split()))
            if not any(counts[2:7]):
                qualified.append(counts[0])
    for log_path in sorted(HERE.glob("refined-live-window-*.log")):
        counts = tuple(map(int,log_path.read_text().split()))
        assert not any(counts[2:7])
        qualified.append(counts[0])
        text(log_path,"continuous-proof/"+log_path.name)
    assert sorted(set(qualified)) == list(range(79))
    text(PROVED/"result.json","continuous-proof/conditioned-first-pass.json")
    tubes=read_tubes()
    assert np.max(np.abs(tubes[:,:2,:7])) <= 1.
    raw = tubes.tobytes()
    dest("continuous-proof/float32-tube-bounds.bin.gz").write_bytes(gzip.compress(raw,mtime=0))
    identities = []
    for name, original in (
        ("NumiHumanRestingVisual.hpp",REG),("NumiHumanRestingCoupling.hpp",REG),
        ("NumiHumanRestingTransactionProbe.hpp",REG),("human_respiration.metal",SHADER)):
        identities.append(code_patch(original/name,BUILD/name))
    dest("runtime-diffs/identity.json").write_text(json.dumps(identities,indent=2)+"\n")
    for run, step in ((WITNESS,2780),(RUN,8000),(REUSE,8000),(GUARDED,8000),(ROLLBACK,32)):
        root=HERE/run
        for file in ("launch.json", "execution.json", "executed-build-verification.json"):
            if (root/file).is_file():
                text(root/file,run+"/"+file)
        for file in (run+"-summary.json",run+"-split-verification.json"):
            if (HERE/file).is_file():
                text(HERE/file,run+"/"+file)
        lines(Path(run),"stdout.log",
              ("resting_integrated_body=completed","resting_integrated_throughput",
               "resting_integrated_rejection=pass","resting_common_domain_rejection=pass",
               "resting_initial_visual_pack_cache=verified_clone","accepted_geometry_export="))
        native=root/"native"
        if run in (RUN,REUSE,GUARDED):
            for file in ("resting-coupled.csv","resting-surface-audit.csv","resting-com-momentum-diagnostic.csv"):
                compact(native/file,run+"/"+file+".gz")
        capture=native/"accepted-geometry"
        receipt=capture/f"step-{step}.receipt.json"
        if receipt.is_file():
            compact(receipt,run+"/"+receipt.name+".gz")
            for p in capture.glob("*.recovery.json"):
                text(p,run+"/"+p.name)
    compact(HERE/(RUN+"-skin-exact.json"),"terminal-skin-crossing-witnesses.json.gz")
    compact(HERE/(GUARDED+"-skin-exact.json"),"guarded-terminal-skin-crossing-witnesses.json.gz")
    text(HERE/"open-skin-clearance.json","open-skin-clearance.json")
    rows={}
    for p in sorted(PUB.rglob("*")):
        if p.is_file():
            rows[str(p.relative_to(PUB))]=dict(sha256=hash_file(p),bytes=p.stat().st_size)
    dest("manifest.json").write_text(json.dumps(rows,indent=2)+"\n")
    for name, spec in rows.items():
        assert hash_file(PUB/name) == spec["sha256"]
    print(len(rows), "verified artifacts",sum(x["bytes"] for x in rows.values()),"bytes",flush=True)


if __name__=="__main__":
    main()
