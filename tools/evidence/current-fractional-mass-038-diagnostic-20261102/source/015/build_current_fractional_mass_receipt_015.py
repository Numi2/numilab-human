#!/usr/bin/env python3
"""030d compatibility adapter over the immutable 014 source-only receipt replay."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
from typing import Any

CORE_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/receipt-builder-014/build_current_fractional_mass_receipt_014.py")
CORE_SHA256 = "f4ea5cf0beee09b6e1cf01dc4bf483c04ecb4fb53063d4308f2b09342b886a4a"
FLOW_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/postpass-030d-canonical-038/recorded_flow_contract_010_030d.py")
FLOW_SHA256 = "f2ee1d53880cb12864f1c3afab6bf336544a8b98f64b60adbd1ab3d3cd4bdba2"
VARIANT_030D = "030d-contiguous-preference-lookup"
STATUS_030D = "PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN"
RUN_CONTROL_ADAPTER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/receipt-builder-015/adapter_006_030d_run_control_015.py")
RUN_CONTROL_ADAPTER_SHA256 = "b8217031787e8ecdb53ae26ff46808893220b48e728d5a0b75bb29390f62a658"
RUN_CONTROL_VARS = {
    "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256",
    "NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_pinned(path: Path, digest: str, name: str):
    path = path.resolve(strict=True)
    if sha_file(path) != digest:
        raise ValueError(f"pinned {name} source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_control_manifest_015(run: dict[str, Any], control_path: Path, *, core, flow, legacy_validator=None):
    """Strict 030d closure dispatch; preserve the 014 validator for older runs."""
    control_path = control_path.resolve(strict=True)
    control = core.read_json(control_path)
    if control.get("variant") != VARIANT_030D:
        validator = legacy_validator or core.validate_control_manifest
        return validator(run, control_path)
    if control.get("status") != STATUS_030D:
        raise core.ReceiptError("030d control must remain full-allocator NOT_RUN")
    if run.get("schema") != core.RUN_SCHEMA:
        raise core.ReceiptError("030d run manifest schema mismatch")
    if run.get("physical_admission") is not False or run.get("native_steps") != 0:
        raise core.ReceiptError("030d run manifest contains native/physical claims")
    p, d = core.path_sha(run.get("source_control_manifest"), "run.source_control_manifest")
    if p != control_path or d != sha_file(control_path):
        raise core.ReceiptError("run does not pin supplied 030d source control")
    if run.get("source") != control.get("source") or run.get("binary") != control.get("binary"):
        raise core.ReceiptError("run source/binary differs from 030d source control")
    flow.verify_nested_control_refs(control)
    closure, closure_sha = core.path_sha(run.get("source_closure"), "run.source_closure")
    flow.BASE.verify_source_closure(closure, closure_sha, control_path, d, control)
    # The 038 source-control stays unchanged and pins frozen 006. This cloned
    # in-memory control selects a narrow 030d reader for the two run-control
    # hashes that 038 adds outside the 27 app environment variables.
    if control.get("admission_adapter", {}).get("sha256") != "249ebe4e548da2d4f1633c03dd994de313b1c591b47d82591fb8c5ea2a12609d":
        raise core.ReceiptError("030d source control does not pin the expected frozen 006 adapter")
    control = dict(control)
    control["admission_adapter"] = {
        "path": str(RUN_CONTROL_ADAPTER_PATH),
        "sha256": RUN_CONTROL_ADAPTER_SHA256,
    }
    return control


def parse_runtime_015(run: dict[str, Any], parser):
    """Keep run-control hashes out of the 006 app env while their adapter verifies them."""
    env, args = parser(run)
    present = RUN_CONTROL_VARS & set(env)
    if present and present != RUN_CONTROL_VARS:
        raise ValueError("030d actual argv must carry both separate run-control hashes")
    for key in RUN_CONTROL_VARS:
        env.pop(key, None)
    return env, args


def canonical_receipt_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_receipt_015(run_path: Path, execution_path: Path, stdout_path: Path,
                      config_path: Path, control_path: Path, output_path: Path):
    core = import_pinned(CORE_PATH, CORE_SHA256, "receipt_builder_014")
    flow = import_pinned(FLOW_PATH, FLOW_SHA256, "recorded_flow_010_030d")
    run_path, execution_path = run_path.resolve(strict=True), execution_path.resolve(strict=True)
    stdout_path, config_path = stdout_path.resolve(strict=True), config_path.resolve(strict=True)
    control_path, output_path = control_path.resolve(strict=True), output_path.resolve()
    run, control = core.read_json(run_path), core.read_json(control_path)
    original = core.validate_control_manifest
    original_parse_runtime = core.parse_runtime
    core.validate_control_manifest = lambda r, p: validate_control_manifest_015(
        r, p, core=core, flow=flow, legacy_validator=original)
    core.parse_runtime = lambda r: parse_runtime_015(r, original_parse_runtime)
    try:
        mass = core.build_receipt(run_path, execution_path, stdout_path, config_path, control_path)
    finally:
        core.validate_control_manifest = original
        core.parse_runtime = original_parse_runtime
    projection = None
    if control.get("variant") == VARIANT_030D:
        transport = mass.get("solver", {}).get("transport_receipt_sha256")
        if not isinstance(transport, str) or len(transport) != 64:
            raise ValueError("mass replay lacks exact transport receipt SHA")
        digest = hashlib.sha256(canonical_receipt_bytes(mass)).hexdigest()
        # This call validates only the 030d source-control projection. The later
        # 010 full-bundle verifier checks the populated per-role records.
        bundle = {
            "schema": flow.BASE.SCHEMA,
            "unified_mass_receipt": {"path": str(output_path), "sha256": digest},
            "source_control_manifest": run["source_control_manifest"],
            "source_closure": run["source_closure"],
            "paired_solver_run_sha256": transport,
            "roles": {},
        }
        flow.verify_source_control(run, bundle, mass_receipt=mass)
        projection = {
            "schema": "numi.h706.current-fractional-mass-source-projection-015.v1",
            "status": "PASS_SOURCE_ONLY_030D_CONTROL_PROJECTION",
            "physical_admission": False, "native_steps": 0,
            "run_manifest": {"path": str(run_path), "sha256": sha_file(run_path)},
            "source_control_manifest": {"path": str(control_path), "sha256": sha_file(control_path)},
            "source_closure": run["source_closure"],
            "mass_receipt": {"path": str(output_path), "sha256": digest},
            "receipt_builder_014": {"path": str(CORE_PATH), "sha256": CORE_SHA256},
            "recorded_flow_contract_010": {"path": str(FLOW_PATH), "sha256": FLOW_SHA256},
            "run_control_compatibility_adapter": {
                "path": str(RUN_CONTROL_ADAPTER_PATH),
                "sha256": RUN_CONTROL_ADAPTER_SHA256,
            },
            "full_recorded_flow_bundle": "NOT_VALIDATED_BY_THIS_SOURCE_PROJECTION_STAGE",
        }
    return mass, projection


def validate_output_paths(output_path: Path, projection_path: Path | None,
                          *, projection_required: bool) -> tuple[Path, Path | None]:
    output = output_path.resolve()
    projection = projection_path.resolve() if projection_path is not None else None
    if output.exists():
        raise ValueError("refusing to overwrite existing mass receipt")
    if projection_required and projection is None:
        raise ValueError("030d requires a distinct source-projection report path")
    if projection is not None:
        if projection == output:
            raise ValueError("mass receipt and source-projection report paths must differ")
        if projection.exists():
            raise ValueError("refusing to overwrite existing source-projection report")
    return output, projection


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical_receipt_bytes(value))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-manifest", required=True, type=Path)
    p.add_argument("--execution", required=True, type=Path)
    p.add_argument("--stdout-log", required=True, type=Path)
    p.add_argument("--identity-config", required=True, type=Path)
    p.add_argument("--control-manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--source-projection-report", type=Path)
    p.add_argument("--failure-report", type=Path)
    a = p.parse_args()
    try:
        control_preview = json.loads(a.control_manifest.resolve(strict=True).read_text(encoding="utf-8"))
        require_projection = control_preview.get("variant") == VARIANT_030D
        out, projection_out = validate_output_paths(
            a.output, a.source_projection_report, projection_required=require_projection)
        mass, projection = build_receipt_015(
            a.run_manifest, a.execution, a.stdout_log, a.identity_config,
            a.control_manifest, out)
        write_json(out, mass)
        if projection is not None:
            if projection_out is None:
                raise ValueError("030d projection output was not predeclared")
            rp = projection_out
            write_json(rp, projection)
            if sha_file(out) != projection["mass_receipt"]["sha256"]:
                raise ValueError("written mass receipt differs from projected receipt digest")
            print(f"recorded_flow_source_projection=PASS path={rp} sha256={sha_file(rp)} physical_admission=0 native_steps=0")
        print(f"fractional_mass_receipt=PASS schema={mass['schema']} path={out} sha256={sha_file(out)} physical_admission=0 native_steps=0")
        return 0
    except Exception as exc:
        failure = a.failure_report or a.output.with_suffix(".failure.json")
        write_json(failure.resolve(), {
            "schema": "numi.h706.current-fractional-mass-receipt-build-failure-015.v1",
            "status": "REJECTED", "physical_admission": False, "native_steps": 0,
            "reason": str(exc),
        })
        print(f"fractional_mass_receipt=REJECTED reason={exc} failure_report={failure.resolve()}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
