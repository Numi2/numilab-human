#!/usr/bin/env python3
"""Run the existing 936 scene owner with the separately verified 1173 reference."""
from pathlib import Path
import hashlib, importlib.util, json, sys
OWNER = Path("/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/v015/final_scene_936-v015.py")
OWNER_SHA = "9828b90ac3bca4a4e91068df6128bbde829fb0dc1f2d9e1d874ddc07af965eac"
REF = Path("/Users/n/numi-human-retained-delivery-20261009/native-terminal-cycle-1173")
VERIFY = REF.parent / "native-terminal-cycle-1173-review/verification.json"
PINS = {
    REF / "run-metadata.json": "24312b80992534d2d52c68818a3472994e5a005b20a7b6262e6448fe6fc057b7",
    REF / "resting-coupled.csv": "aa3b6f12f482f09becbdf20185240d9bb27a945df21dd0771096aac0243faffd",
    REF / "invocation.json": "fea4c8ada002ca9db9c02daf4484a04db6a0933be959dcf4dd2f0a10be141ae0",
    REF / "native.log": "9b62b8e5a8d8c5971b36ae4d022f380d291c25e9c467ede22bc1d4586683fd04",
    REF / "accepted-geometry/step-10000.mrvpack": "40edd587f96eb27b26d882a74d60243086905fa55a4e65633f1467500f13d764",
    REF / "accepted-geometry/step-10000.receipt.json": "673d17d899cf84f9bce25e762d5a8f8d5ee976eef3ff4337d8d1195276c49ab2",
    VERIFY: "13755a72927eb6fe38cd98d1b5512daf6ecbe1faa859b11ad147d71e8367bd16",
}
def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4194304), b""): h.update(block)
    return h.hexdigest()

def validate_reference_metadata(metadata, invocation, verification, pins, ref):
    if metadata.get("exit_code") != 0 or metadata.get("source_files_changed_during_run") != []:
        raise RuntimeError("reference did not complete with unchanged sources")
    if metadata.get("loaded_metal_runtime", {}).get("verified") is not True:
        raise RuntimeError("reference loaded runtime unverified")
    for key in ("argv", "asset_sha256", "environment"):
        if metadata.get(key) != invocation.get(key):
            raise RuntimeError("reference metadata/invocation mismatch: " + key)
    if verification.get("pass") is not True or verification.get("regenerated_run_path") != str(ref):
        raise RuntimeError("reference verification failed or names a different run")
    if verification.get("regenerated_run_metadata_sha256") != pins[ref / "run-metadata.json"]:
        raise RuntimeError("verification does not bind reference metadata")
    if verification.get("regenerated_invocation_sha256") != pins[ref / "invocation.json"]:
        raise RuntimeError("verification does not bind reference invocation")
    terminal = verification.get("terminal", {})
    if (terminal.get("accepted_step") != 10000 or
        terminal.get("no_additional_physical_or_controller_step") is not True or
        terminal.get("matches_final_physical_trace_time") is not True):
        raise RuntimeError("reference terminal accepted-state proof incomplete")
    for key, name in (("pack_file_sha256", "step-10000.mrvpack"), ("receipt_sha256", "step-10000.receipt.json")):
        if terminal.get(key) != pins[ref / "accepted-geometry" / name]:
            raise RuntimeError("reference terminal file identity mismatch")

def main():
    if sha(OWNER) != OWNER_SHA: raise RuntimeError("frozen 936 owner changed")
    for path, digest in PINS.items():
        if sha(path) != digest: raise RuntimeError("reference input changed: " + str(path))
    load = lambda p: json.loads(p.read_text())
    validate_reference_metadata(load(REF / "run-metadata.json"), load(REF / "invocation.json"), load(VERIFY), PINS, REF)
    if "--reference-self-test" in sys.argv:
        print(json.dumps({
            "status": "pinned_verified1173_reference_pass",
            "reference_run_path": str(REF),
            "verification_path": str(VERIFY),
            "verified_input_sha256": {str(p): h for p, h in PINS.items()},
            "accepted_terminal_step": 10000,
            "terminal_pack_sha256": PINS[REF / "accepted-geometry/step-10000.mrvpack"],
            "terminal_receipt_sha256": PINS[REF / "accepted-geometry/step-10000.receipt.json"],
            "historical_931_metadata_reconstructed": False,
        }, indent=2))
        return 0
    spec = importlib.util.spec_from_file_location("frozen_936_v015", OWNER)
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    historical = {"reference_path": str(owner.REF), "metadata_sha256": owner.P["refmeta"], "trace_sha256": owner.P["refcsv"]}
    if owner.P["refcsv"] != PINS[REF / "resting-coupled.csv"]:
        raise RuntimeError("retained trace is not byte-equivalent to historical reference hash")
    owner.REF = REF
    owner.P = dict(owner.P, refmeta=PINS[REF / "run-metadata.json"], refcsv=PINS[REF / "resting-coupled.csv"])
    original_writer = owner.wj
    def write_with_reference(path, report):
        if report.get("schema") in ("numi.human.final-native-scene-preflight.v1", "numi.human.final-20s-vs-931-comparison.v1"):
            report["reference_identity"] = {
                "run_path": str(REF), "role": "separately regenerated 1173 full-q reference; not historical 931 metadata",
                "verified_inputs_sha256": {str(p): h for p, h in PINS.items()},
                "historical_931": historical,
                "historical_metadata_reconstructed": False,
                "historical_trace_hash_equal": True,
                "scope": "20 s cadence/pose regression only; exact candidate parity is an observation, not an acceptance prerequisite.",
            }
            report["reference_driver"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__), "owner_path": str(OWNER), "owner_sha256": OWNER_SHA}
            if "source_sha256" in report:
                report["source_sha256"].update({str(p): h for p, h in PINS.items()})
                report["source_sha256"][str(Path(__file__).resolve())] = sha(__file__)
                report["post_run_comparison"] = "Use this reference driver with --compare-run/--compare-out; comparison uses separately verified1173 full-q outputs, with exact candidate owner argv/assets and accepted terminal checks."
            else:
                report["schema"] = "numi.human.final-20s-reference-comparison.v1"
        original_writer(path, report)
    owner.wj = write_with_reference
    result = owner.main()
    for path, digest in PINS.items():
        if sha(path) != digest: raise RuntimeError("reference input changed during owner operation: " + str(path))
    return result
if __name__ == "__main__": raise SystemExit(main())
