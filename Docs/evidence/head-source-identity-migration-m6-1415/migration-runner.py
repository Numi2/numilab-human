from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from numilab_human import model
from numilab_human.physiology import load_anatomy
from numilab_human.resting_anatomy import HEAD_MEMBER_COMPONENTS, rebind_head_member_identity_metadata
from numilab_human.resting_run import command as resting_run_command

D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
BASE = D / "current-manifest-recovery-004/registered-resting-scene-source-preparation-004/resting-anatomy-receipt.json"
BASE_SHA = "84d53a9ba9021df1a13327c4beb9f3d2845dcf4cc1d40819d247402f27512e4e"
PAYLOAD_SHA = "1e6da4fa8c550334f31a6b74fda7aed6f26b0c436bff0b9ad168fe97d41cbb23"
SOURCES = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/vein-owner-blend-source-append-dryrun-012-inputs/Sources")
OUT = D / "head-source-identity-migration-002"
RECEIPT_OUT = OUT / "resting-anatomy-receipt.json"
REPORT_OUT = OUT / "migration-report.json"
OLD_REPORT = D / "head-source-identity-migration-001/migration-report.json"
OLD_REPORT_SHA = "288f677f60c4c6b010f9098be6f1c331da724dc50c56e9a054385cd53cc186c6"
OLD_RECEIPT = D / "head-source-identity-migration-001/resting-anatomy-receipt.json"
OLD_RECEIPT_SHA = "4c13fddf83175b50ae0d1f930d01ebf938816f0eb294a12afc142d88401e5b05"
WRONG_BASE_RECEIPT_SHA = "98fc9e82233c4a4a6bc6af38c23e774d3a3373afec94d7cef9122108a76a238d"
WRONG_BASE_PAYLOAD_SHA = "9fb3ffb10e458589b2c69779bd1336bfc4dc8e704079a1e68e4b22a2b5b7ba09"
NATIVE_INVOCATION = D / "registered-resting-head-body-native-emission-diagnostic-007/native-run/invocation.json"
SIDECARS = {
    "common-cardiac-map-f32.bin": "0d88db155f7dfe9a93030badf6e570f49050164a7d0aa0ae011065aeada7d3da",
    "common-cardiac-domains-f32.bin": "7652785c9ea164b16ee96f64a3dcf940398af36cfd535a5f9b7a7d6bd2c7a611",
    "common-cardiac-volumes-f32.bin": "7be4f62996c996c6cfdd49480d27766c8253387bb24aeb504fef73cd553ae699",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"preserve existing migration output: {OUT}")
    if sha(BASE) != BASE_SHA:
        raise SystemExit("actual source004 base receipt SHA changed")
    if sha(OLD_REPORT) != OLD_REPORT_SHA or sha(OLD_RECEIPT) != OLD_RECEIPT_SHA:
        raise SystemExit("preserved migration-001 erratum pins changed")
    base = json.loads(BASE.read_text())
    payload = Path(base["payload"]["path"]).resolve()
    if not payload.is_file():
        raise SystemExit("actual source004 NHANAT payload is missing")
    payload_raw = payload.read_bytes()
    payload_sha = hashlib.sha256(payload_raw).hexdigest()
    if payload_sha != PAYLOAD_SHA or payload_sha != base["payload"]["sha256"]:
        raise SystemExit("actual source004 NHANAT payload does not match its receipt and pin")
    bodyparts3d = load_anatomy(SOURCES)
    member_sha = {}
    for member_id in HEAD_MEMBER_COMPONENTS:
        _, _member_name, obj = model._bodyparts_obj_member(SOURCES, "part_of", member_id)
        member_sha[member_id] = hashlib.sha256(obj).hexdigest()
    updated, audit = rebind_head_member_identity_metadata(
        base, bodyparts3d, member_sha, payload_raw=payload_raw
    )
    if set(audit["changed_stable_ids"]) != {"323", "326"}:
        raise SystemExit("identity migration changed unexpected stable IDs")
    if updated["payload"] != base["payload"]:
        raise SystemExit("payload metadata changed during source-label migration")
    if updated["functional_bindings"] != base["functional_bindings"]:
        raise SystemExit("functional bindings changed during source-label migration")
    if updated["mass_geometry_accounting"] != base["mass_geometry_accounting"]:
        raise SystemExit("mass accounting changed during source-label migration")
    for key in base:
        if key != "provenance" and updated[key] != base[key]:
            raise SystemExit(f"unexpected top-level receipt change: {key}")
    for key in base["provenance"]:
        if key != "source_id_map" and updated["provenance"][key] != base["provenance"][key]:
            raise SystemExit(f"unexpected provenance change: {key}")
    before_map = base["provenance"]["source_id_map"]
    after_map = updated["provenance"]["source_id_map"]
    if set(before_map) != set(after_map):
        raise SystemExit("source ID map keys changed")
    changed = set(audit["changed_stable_ids"])
    for stable_id in before_map:
        if stable_id not in changed:
            if after_map[stable_id] != before_map[stable_id]:
                raise SystemExit(f"unexpected source-map row change: {stable_id}")
            continue
        row = dict(after_map[stable_id])
        row["name"] = before_map[stable_id]["name"]
        row["source_owner_metadata"] = before_map[stable_id]["source_owner_metadata"]
        if row != before_map[stable_id]:
            raise SystemExit(f"unexpected metadata change in row {stable_id}")
    sidecar_sources = {}
    for filename, expected_sha in SIDECARS.items():
        source = BASE.parent / filename
        if not source.is_file() or sha(source) != expected_sha:
            raise SystemExit(f"source004 cardiac sidecar missing or hash mismatch: {source}")
        sidecar_sources[filename] = source.resolve()
    OUT.mkdir()
    for filename, source in sidecar_sources.items():
        link = OUT / filename
        link.symlink_to(source)
        if not link.is_file() or sha(link) != SIDECARS[filename]:
            raise SystemExit(f"migration sidecar link failed exact hash check: {link}")
    temp_receipt = RECEIPT_OUT.with_suffix(".json.tmp")
    temp_receipt.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    os.replace(temp_receipt, RECEIPT_OUT)
    report = {
        "status": audit["status"],
        "input_receipt_path": str(BASE),
        "input_receipt_sha256": BASE_SHA,
        "output_receipt_path": str(RECEIPT_OUT),
        "output_receipt_sha256": sha(RECEIPT_OUT),
        "anatomy_payload_path": str(payload),
        "anatomy_payload_sha256": payload_sha,
        "payload_typed_rows_verified": audit["payload_typed_rows_verified"],
        "typed_rows": audit["rows"],
        "native_payload_bytes_changed": False,
        "changed_source_id_map_rows_only": True,
        "functional_bindings_unchanged": True,
        "mass_geometry_accounting_unchanged": True,
        "source_tables": bodyparts3d["source"]["tables"],
        "source_members": audit["rows"],
        "source_dir": str(SOURCES),
        "source_archive_provenance": bodyparts3d["source"],
        "prior_attempt_erratum": {
            "status": "PRESERVED_WRONG_BASELINE_NOT_CURRENT",
            "attempt_001_report_path": str(OLD_REPORT),
            "attempt_001_report_sha256": OLD_REPORT_SHA,
            "attempt_001_output_receipt_path": str(OLD_RECEIPT),
            "attempt_001_output_receipt_sha256": OLD_RECEIPT_SHA,
            "attempt_001_input_receipt_sha256": WRONG_BASE_RECEIPT_SHA,
            "attempt_001_payload_sha256": WRONG_BASE_PAYLOAD_SHA,
            "correction": "attempt_001 was based on the integrated-source-004 receipt, not the current-manifest-recovery-004 registered source004 receipt; it is retained and must not be used as current provenance",
        },
        "no_geometry_predicates_or_admission_claimed": True,
    }
    invocation = json.loads(NATIVE_INVOCATION.read_text())
    argv0 = invocation["argv"]
    def option_value(name):
        return Path(argv0[argv0.index(name) + 1])
    body_scene_candidates = [
        Path(path) for path in invocation["asset_sha256"]
        if Path(path).name == "resting-supine-scene.manifest.json"
    ]
    if len(body_scene_candidates) != 1:
        raise SystemExit("native007 invocation does not identify one supine scene manifest")
    command_args = argparse.Namespace(
        body_scene=body_scene_candidates[0],
        anatomy_receipt=RECEIPT_OUT,
        tendon=option_value("--tendon-payload"),
        lab=Path("/Users/n/numi-human-resting-build-20261005"),
        build=Path(argv0[0]).resolve().parent.parent,
        output=OUT / "resting-run-command-preparation-only",
        circulation=option_value("--resting-scene"),
        respiration=Path(argv0[argv0.index("--resting-scene") + 2]),
        seconds=0.002,
        dt=0.002,
        dimension=512,
        mechanics_only=True,
        inspection_tour=False,
        drive_intervention=None,
        initial_posture=None,
    )
    prepared_argv, prepared_assets = resting_run_command(command_args)
    required_asset_hashes = {
        "cardiac_map": SIDECARS["common-cardiac-map-f32.bin"],
        "cardiac_domains": SIDECARS["common-cardiac-domains-f32.bin"],
        "cardiac_volumes": SIDECARS["common-cardiac-volumes-f32.bin"],
        "anatomy_payload": PAYLOAD_SHA,
        "native_muscle_surfaces": base["provenance"]["native_muscle_surfaces"]["sha256"],
    }
    for name, expected_sha in required_asset_hashes.items():
        if expected_sha not in prepared_assets.values():
            raise SystemExit(f"resting_run.command omitted required owner asset: {name}")
    parent_source_hashes = [
        record["sha256"] for record in base["passive_tissue_parent_bindings"]["source_payloads"]
    ]
    if any(digest not in prepared_assets.values() for digest in parent_source_hashes):
        raise SystemExit("resting_run.command omitted a passive parent extension asset")
    report["resting_run_command_preparation"] = {
        "owner": "numilab_human.resting_run.command",
        "native_launched": False,
        "argv": prepared_argv,
        "asset_sha256": prepared_assets,
        "required_assets_verified": required_asset_hashes,
        "passive_parent_source_count": len(parent_source_hashes),
        "passive_parent_source_hashes": parent_source_hashes,
        "sidecar_links": {
            name: {"path": str(OUT / name), "target": str(target), "sha256": sha(target)}
            for name, target in sidecar_sources.items()
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "receipt": str(RECEIPT_OUT),
        "receipt_sha256": report["output_receipt_sha256"],
        "report": str(REPORT_OUT),
        "report_sha256": sha(REPORT_OUT),
        "status": report["status"],
        "baseline_receipt_sha256": BASE_SHA,
        "payload_sha256": payload_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
