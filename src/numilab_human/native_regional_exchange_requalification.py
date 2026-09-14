"""Bind the current physical-M4 regional blood/exchange replay.

The native transaction is admitted as a source-graph amount/conservation
subgate.  It does not assign an anatomical lumen, physical tissue volume,
mechanical blood mass, organ mechanics, material calibration, or subject
calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Docs/media/native-human-regional-exchange-20260914/receipt.json"
NATIVE_LOG = ROOT / "Docs/media/native-human-regional-exchange-20260915/native.log"
BUILD_LOG = ROOT / "Docs/media/native-human-regional-exchange-20260915/build.log"
OUTPUT = ROOT / "Docs/media/native-human-regional-exchange-20260915/receipt-v1.json"
SCHEMA = "HumanPack.native-human-regional-exchange-current-requalification.v1"
CURRENT_SOURCE = {
    "branch": "numi-human-equilibrium-20260914",
    "commit": "c45fa9622f6c73b58febdc24a7115aecf3d7699f",
    "device": "Mac mini M4 Pro",
    "binary": "numi-matter-vascular-check",
    "binary_sha256": "dd4663054a9fc52b52dfff082a8afa1d7895180ac4527fddcaccaf8d205e4fa0",
}
FIXTURE = {
    "path": "Docs/media/cvsim21-circulation-20260912/cvsim21.native.v3.json",
    "sha256": "eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("native regional exchange requalification: " + message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_regular(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path.read_bytes()


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} is not finite")
    return float(value)


def _read_upstream(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_regular(path, "upstream receipt")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HumanImportError("native regional exchange requalification: upstream receipt is invalid JSON") from error
    _require(isinstance(value, dict), "upstream receipt is not an object")
    _require(value.get("schema") == "HumanPack.native-human-regional-exchange.v1",
             "upstream receipt schema changed")
    source = value.get("source", {})
    host = value.get("host", {})
    topology = value.get("topology", {})
    clock = value.get("clock", {})
    conservation = value.get("conservation", {})
    qualification = value.get("qualification", {})
    _require(source.get("source_graph_sha256") == "b118d58230a3f2766108d05501a3ac9b910124ef823567a7e5fe23f6df0b4f67",
             "source graph hash changed")
    _require(source.get("authored_graph_sha256") ==
             "0b39959800fa666671ed7f8ca598564cc58a0207179701e255e967484f40a9a0",
             "authored graph hash changed")
    _require(host.get("chip") == "Apple M4 Pro" and host.get("binary_sha256") == CURRENT_SOURCE["binary_sha256"],
             "upstream native host or binary changed")
    _require(topology.get("source_compartment_count") == 21 and
             topology.get("source_connection_count") == 24 and
             topology.get("regional_bed_count") == 7,
             "native topology changed")
    _require(clock == {
        "timestep_seconds": 0.0000125,
        "timestep_nanoseconds": 12500,
        "attempted_steps": 512,
        "accepted_steps_environment_0": 511,
        "accepted_steps_environment_1": 512,
        "rejected_step_environment_0": 37,
    }, "native exact-clock counters changed")
    for key in ("maximum_relative_volume_residual", "maximum_relative_blood_mass_residual",
                "maximum_relative_oxygen_residual"):
        _finite(conservation.get(key), key)
    _require(conservation.get("blood_density_candidate_kg_per_m3") == 1060.0 and
             conservation.get("blood_density_provenance") == "engineering_candidate_unresolved" and
             conservation.get("rejected_candidate_state_neutral") is True and
             conservation.get("bitwise_replay") is True,
             "native conservation boundary changed")
    for key in ("source_graph_identity_retained", "regional_blood_transport_bound",
                "oxygen_amount_exchange_executed", "accepted_step_conservation", "exact_clock"):
        _require(qualification.get(key) is True, f"native qualification {key} changed")
    for key in ("anatomical_vessel_lumen", "physical_tissue_volume_owner",
                "mechanical_blood_mass_owner", "organ_mechanics", "material_calibration",
                "subject_calibration", "standing_walking"):
        _require(qualification.get(key) is False, f"native boundary {key} was promoted")
    return value, _sha(path)


def _parse_log(path: Path) -> dict[str, Any]:
    text = _read_regular(path, "native log").decode("utf-8")
    line = next((row for row in text.splitlines() if row.startswith("human_regional_exchange=pass ")), None)
    _require(line is not None, "native log has no passing regional exchange summary")
    fields: dict[str, Any] = {}
    for key, value in re.findall(r"([a-zA-Z0-9_]+)=([^ ]+)", line):
        if value.isdigit():
            fields[key] = int(value)
        else:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value
    _require(fields.get("source_compartments") == 21 and fields.get("source_edges") == 24 and
             fields.get("regional_beds") == 7 and fields.get("attempted_steps") == 512 and
             fields.get("accepted_environment0") == 511 and fields.get("rejected_step") == 37 and
             fields.get("timestep_ns") == 12500 and fields.get("rollback") == "bitwise" and
             fields.get("replay") == "bitwise" and fields.get("source_identity") == "retained",
             "native log counters or replay boundary changed")
    _require(fields.get("blood_density_candidate_kg_m3") == 1060.0 and
             fields.get("density_material_calibration") == "unqualified" and
             "vascular_native_qualification=pass" in text and
             "biological_calibration=unqualified" in text,
             "native log unresolved-density boundary changed")
    return fields


def compile_receipt(*, upstream: Path = UPSTREAM, native_log: Path = NATIVE_LOG,
                    build_log: Path = BUILD_LOG) -> dict[str, Any]:
    upstream_doc, upstream_sha = _read_upstream(Path(upstream))
    log_fields = _parse_log(Path(native_log))
    build_bytes = _read_regular(Path(build_log), "native build log")
    _require(b"ninja: no work to do." in build_bytes or b"ninja: no work to do" in build_bytes,
             "native build log does not prove the current target")
    return {
        "schema": SCHEMA,
        "status": "partial",
        "subject": "one adult male source package",
        "source": {
            **CURRENT_SOURCE,
            "upstream_receipt_sha256": upstream_sha,
            "upstream_source_commit": upstream_doc["source"]["native_source_commit"],
            "native_log_sha256": _sha(Path(native_log)),
            "build_log_sha256": _sha(Path(build_log)),
        },
        "inputs": {"fixture": FIXTURE},
        "results": {
            "source_compartment_count": log_fields["source_compartments"],
            "source_connection_count": log_fields["source_edges"],
            "regional_bed_count": log_fields["regional_beds"],
            "attempted_steps": log_fields["attempted_steps"],
            "accepted_steps_environment_0": log_fields["accepted_environment0"],
            "rejected_step_environment_0": log_fields["rejected_step"],
            "timestep_nanoseconds": log_fields["timestep_ns"],
            "blood_density_candidate_kg_per_m3": log_fields["blood_density_candidate_kg_m3"],
            "maximum_relative_volume_residual": log_fields["volume_conservation_max"],
            "maximum_relative_blood_mass_residual": log_fields["blood_mass_conservation_max"],
            "maximum_relative_oxygen_residual": log_fields["oxygen_conservation_max"],
            "rollback": log_fields["rollback"],
            "replay": log_fields["replay"],
        },
        "qualification": {
            "current_native_replay": True,
            "regional_blood_transport": True,
            "oxygen_amount_exchange": True,
            "accepted_step_conservation": True,
            "anatomical_vessel_lumen": False,
            "physical_tissue_volume_owner": False,
            "mechanical_blood_mass_owner": False,
            "organ_mechanics": False,
            "material_calibration": False,
            "subject_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "Current source-graph amount, exact-clock advection, bidirectional oxygen "
            "exchange, conservation, rejected-step rollback and bitwise Apple M4 Pro "
            "replay. The density, transit/tissue volumes and exchange coefficient "
            "remain engineering candidates; no anatomical lumen, physical tissue "
            "volume, mechanical blood mass, organ mechanics, material calibration, "
            "subject calibration, standing or walking is admitted."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "receipt output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "receipt output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_receipt(upstream=arguments.upstream, native_log=arguments.native_log,
                             build_log=arguments.build_log)
    receipt_sha = _immutable_write(arguments.receipt.resolve(), result)
    print(json.dumps({"schema": SCHEMA, "receipt": str(arguments.receipt.resolve()),
                      "sha256": receipt_sha, "status": result["status"]}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--native-log", type=Path, default=NATIVE_LOG)
    parser.add_argument("--build-log", type=Path, default=BUILD_LOG)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
