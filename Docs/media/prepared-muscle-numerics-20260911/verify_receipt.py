"""Recheck fixed-state muscle numerics and bounded coupled replay evidence."""
import base64
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("trace_audit",
    BASE.parent / "source-compliant-equilibrium-20260910/verify_receipt.py")
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def require(value, message):
    if not value:
        raise ValueError(message)


def data(path):
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def force_scales(base=BASE):
    raw = data(base / "muscle.nhmyo.gz")
    magic, abi, bodies, count, sites, wraps, routes, tendons, architectures, size, archive = struct.unpack_from("<8s9I32s", raw)
    require((magic, abi, bodies, count, architectures, size) == (b"NHMYO2\0\0", 2, 157, 416, 416, 32), "muscle payload header")
    offset = 76 + sites * 16 + wraps * 64 + routes * 16
    require(len(raw) == offset + count * (164 + 32), "muscle payload extent")
    result = []
    for i in range(count):
        values = struct.unpack_from("<37f", raw, offset + i * 164 + 16)
        result.append(values[8] / values[2] if values[7] < 0 else values[7])
    require(all(math.isfinite(x) and x > 0 for x in result), "force scales")
    return result


def audit_probe(text, scales, expected_pass):
    rows = [json.loads(s.split("=", 1)[1]) for s in text.splitlines() if s.startswith("prepared_path={")]
    require([r["index"] for r in rows] == list(range(416)), "missing or duplicate muscle rows")
    for row in rows:
        require(all(math.isfinite(v) for v in row.values()), "nonfinite muscle row")
        require(row["native_fp64_m"] > 0 and row["metal_fp32_m"] > 0, "nonpositive route")
        require(abs(abs(row["native_fp64_m"] - row["metal_fp32_m"]) - row["absolute_error_m"]) < 1e-15, "path error mismatch")
    summaries = [json.loads(s.split("=", 1)[1]) for s in text.splitlines() if s.startswith("prepared_path_summary=")]
    require(len(summaries) == 1, "probe summary missing or duplicate")
    summary = summaries[0]
    measured = {
        "maximum_error_m": max(r["absolute_error_m"] for r in rows),
        "maximum_same_path_force_error_n": max(abs(r["metal_fp32_force_n"] - r["same_path_fp64_force_n"]) for r in rows),
        "maximum_force_error_n": max(abs(r["metal_fp32_force_n"] - r["native_fp64_force_n"]) for r in rows),
        "maximum_normalized_same_path_force_error": max(abs(r["metal_fp32_force_n"] - r["same_path_fp64_force_n"]) / scale for r, scale in zip(rows, scales)),
        "maximum_fiber_residual": max(abs(r["metal_fiber_residual"]) for r in rows)}
    for key, value in measured.items():
        if key in summary:
            require(math.isclose(value, summary[key], rel_tol=1e-12, abs_tol=1e-14), "summary disagrees with muscle rows: " + key)
    require(summary["passed"] is expected_pass, "probe qualification")
    require(summary["tolerance_m"] == 2e-6, "path tolerance changed")
    if expected_pass:
        require(summary["path_passed"] and summary["fiber_passed"], "incomplete probe gate")
        require(measured["maximum_error_m"] <= 2e-6 and
                measured["maximum_normalized_same_path_force_error"] <= 1e-5 and
                measured["maximum_fiber_residual"] <= 1e-5, "numerical budget exceeded")
        require(summary["normalized_force_tolerance"] == 1e-5 and
                summary["publication_tolerance_ulps"] == 0.501 and
                summary["maximum_fiber_publication_error_ulps"] <= 0.501, "fibre publication gate")
    return summary


def verify(base=BASE):
    receipt = json.loads((base / "receipt.json").read_text())
    require(receipt["schema"] == "numi.human.prepared-muscle-numerics.v1", "receipt schema")
    actual = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()
              and p.name != "receipt.json" and "__pycache__" not in p.parts}
    require(actual == set(receipt["artifacts"]), "artifact inventory")
    for name, digest in receipt["artifacts"].items():
        require(hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, "artifact hash: " + name)
    require(hashlib.sha256(Path(SPEC.origin).read_bytes()).hexdigest() == receipt["trace_auditor_sha256"], "trace auditor drift")
    scales = force_scales(base)
    for label in ["baseline", "angular-control-1us", "angular-control-100us"]:
        audit_probe(data(base / (label + ".log.gz")).decode(), scales, False)
    results = {}
    for us in [1, 5, 10, 100]:
        label = "published-" + str(us) + "us"
        launch = json.loads((base / (label + "-launch.json")).read_text())
        require(launch["source_revision"] == receipt["native_revision"] and not launch["source_status"] and
                launch["returncode"] == 0 and not launch["competing_workloads"], "probe execution provenance")
        require(launch["command"][-2:] == ["--timestep-us", str(us)], "probe timestep command")
        for suffix, artifact in [("/myosim-fullbody-muscle-reference.nhmyo", "muscle.nhmyo.gz"),
                                 ("/prepared.nhinit", "prepared.nhinit")]:
            require([v for k, v in launch["sha256"].items() if k.endswith(suffix)] ==
                    [hashlib.sha256(data(base / artifact)).hexdigest()], "probe input binding")
        for name, artifact in [("src/metal/MujocoMuscleReference.metal", "MujocoMuscleReference-wrap-fiber-strain.metal.gz"),
                               ("apps/numilab_human_myosim_reference_probe.cpp", "reference-probe.cpp")]:
            hashes = [v for k, v in launch["sha256"].items() if k.endswith("/" + name)]
            require(hashes == [hashlib.sha256(data(base / artifact)).hexdigest()], "probe source binding")
        results[us] = audit_probe(data(base / (label + ".log.gz")).decode(), scales, True)
        expected_dt = struct.unpack("<f", struct.pack("<f", us * 1e-6))[0]
        require(results[us]["timestep_seconds"] == expected_dt, "authored timestep changed")
    launch = json.loads((base / "horizon-wrapped-fiber-launch.json").read_text())
    require(launch["returncode"] == 0 and not launch["competing_workloads"], "coupled runner failed")
    for owner, revision in [("native", receipt["native_revision"]), ("brain", receipt["brain_revision"])]:
        require(launch["source_state"][owner] == {"revision": revision, "status": ""}, "coupled source state")
    text = data(base / "horizon-wrapped-fiber.log.gz").decode()
    cohort = TRACE.audit_trace(text, 64)
    initial = (base / "prepared.nhinit").read_bytes()
    require(len(initial) == 96 + 257 * 4 + 416 * 16, "prepared fibre input extent")
    lengths = [struct.unpack_from("<f", initial, 96 + 257 * 4 + i * 16 + 8)[0] for i in range(416)]
    records = {}
    for line in text.splitlines():
        if line.startswith("prepared_recruitment={"):
            row = json.loads(line.split("=", 1)[1])
            if row["kind"] in {"fiber_length", "fiber_velocity"}:
                records[row["scenario"], row["root"], row["kind"]] = struct.unpack("<416f", base64.b64decode(row["fp32_le_base64"]))
    max_ulps = 0.0
    dt = struct.unpack("<f", struct.pack("<f", 1e-4))[0]
    for scenario in TRACE.SCENARIOS:
        previous = lengths
        for root in range(1, 65):
            current = records[scenario, root, "fiber_length"]
            for old, new, velocity in zip(previous, current, records[scenario, root, "fiber_velocity"]):
                bits = struct.unpack("<I", struct.pack("<f", new))[0]
                ulp = struct.unpack("<f", struct.pack("<I", bits + 1))[0] - new
                max_ulps = max(max_ulps, abs(new - old - dt * velocity) / ulp)
            previous = current
    require(max_ulps <= 0.501, "coupled fibre clock/publication mismatch")
    diagnostic = json.loads((base / "validation-off4-launch.json").read_text())
    require(diagnostic["returncode"] == 0 and "MTL_DEBUG_LAYER" not in diagnostic["environment"], "validation-off diagnostic")
    off = data(base / "validation-off4.log.gz").decode()
    on = data(base / "wrapped-fiber4.log.gz").decode()
    TRACE.audit_trace(off, 4)
    def physical_records(text):
        result = {}
        for line in text.splitlines():
            if line.startswith("prepared_recruitment={"):
                r = json.loads(line.split("=", 1)[1])
                result[r["scenario"], r["root"], r["kind"]] = r["fp32_le_base64"]
        return result
    require(physical_records(on) == physical_records(off), "validation mode changed physical results")
    independent = json.loads((base / "independent-source-path-audit.json").read_text())
    require(independent["passed"] and independent["source_files_verified"] == 57 and
            independent["initial_muscle_path_diagnostic"]["maximum_length_error_m"] <= 2e-6,
            "independent source path audit")
    require("100% tests passed out of 10" in data(base / "native-regressions.log.gz").decode(), "native regressions")
    require(receipt["qualification"] == {"inside_wrap_paths": True, "same_path_fibre_solver": True,
        "authored_sub_ten_us_timestep": True, "bounded_coupled_replay": True,
        "full_source_force_convergence": False, "standing_walking": False,
        "registered_anatomical_tissue": False, "experimental_calibration": False,
        "performance": False}, "qualification boundary")
    return {"probes": results, "coupled": cohort, "maximum_coupled_fibre_publication_error_ulps": max_ulps,
            "independent_path_error_m": independent["initial_muscle_path_diagnostic"]["maximum_length_error_m"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
