"""Audit measured candidate cost and retained prepared-state refinement failures.

This reads immutable device results; it never advances a physical state.
"""
import base64
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import struct

BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent / "prepared-muscle-numerics-20260911"
SPEC = importlib.util.spec_from_file_location("prepared_numerics", PRIOR / "verify_receipt.py")
NUMERICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NUMERICS)
require = NUMERICS.require
data = NUMERICS.data
KINDS = NUMERICS.TRACE.COUNTS_V2
DEPENDENCIES = (
    "prepared-muscle-numerics-20260911/verify_receipt.py",
    "source-compliant-equilibrium-20260910/verify_receipt.py",
    "prepared-muscle-numerics-20260911/horizon-wrapped-fiber.log.gz",
    "prepared-muscle-numerics-20260911/muscle.nhmyo.gz",
    "prepared-muscle-numerics-20260911/prepared.nhinit",
)


def rows(text, prefix):
    # Atomic device records can follow a partially flushed diagnostic line.
    # Decode the intact JSON record, require its terminating newline, and let
    # identity/count checks below reject duplicates or missing frames.
    result = []
    decoder = json.JSONDecoder()
    for match in re.finditer(re.escape(prefix + "{"), text):
        record, end = decoder.raw_decode(text, match.start() + len(prefix))
        require(text[end:end+1] == "\n", "truncated or interleaved JSON record")
        result.append(record)
    return result


def physical_records(text, scenarios, roots, timestep):
    result = {}
    for r in rows(text, "prepared_recruitment="):
        # Summary lines are filtered by the caller below; every JSON row must
        # have a unique identity, exact byte extent and accepted physical clock.
        key = r["scenario"], r["root"], r["kind"]
        require(r["schema"] == "numi.human.prepared-recruitment-trace.v2", "trace schema")
        require(key[0] in scenarios and type(key[1]) is int and 1 <= key[1] <= roots
                and key[2] in KINDS, "trace identity")
        require(key not in result and r["elapsed_microseconds"] == key[1] * timestep, "trace clock or duplicate")
        raw = base64.b64decode(r["fp32_le_base64"], validate=True)
        require(len(raw) == 4 * KINDS[key[2]], "trace extent")
        values = struct.unpack("<" + str(KINDS[key[2]]) + "f", raw)
        require(all(math.isfinite(v) for v in values), "nonfinite trace")
        if key[2] in {"activation", "motor"}:
            require(all(0 <= v <= 1 for v in values), "actuator bounds")
        if key[2] == "qv":
            require(abs(sum(v*v for v in values[3:7]) - 1) <= 16 * 2**-23, "quaternion")
        result[key] = raw
    require(len(result) == len(scenarios) * roots * len(KINDS), "incomplete scenario or root evidence")
    return result


def trace_records(text, scenarios, roots, timestep):
    return physical_records(text, scenarios, roots, timestep)


def audit_refinement(text, timestep, roots, completed):
    scenarios = ("zero", "replay") if completed else ("zero",)
    records = trace_records(text, scenarios, roots, timestep)
    for root in range(1, roots + 1):
        motor = struct.unpack("<416f", records["zero", root, "motor"])
        require(all(v == 0 for v in motor), "refinement motor changed")
        if completed:
            for kind in KINDS:
                require(records["zero", root, kind] == records["replay", root, kind], "refinement replay drift")
    if completed:
        expected = f"prepared_timestep=observed roots_per_scenario={roots} timestep_us={timestep} duration_us={roots*timestep} replay=bitwise"
        require(expected in text and "Executed 1 test, with 0 failures" in text, "refinement completion missing")
    else:
        require("code=10" in text and "prepared_timestep=observed" not in text,
                "failed refinement promoted or failure missing")
        failures = re.findall(r"mrnx_matter_status code=10[^\n]*diagnostics=(\[[^\n]+?\])", text)
        require(len(failures) == 1, "missing or duplicate physical failure")
        norm, threshold, support, rigid = json.loads(failures[0])
        require(math.isclose(threshold, 0.005, rel_tol=1e-7) and norm > threshold,
                "nonlinear acceptance tolerance changed")
        if support != 0 or rigid != 0:
            require(math.isclose(norm, math.hypot(support, rigid), rel_tol=1e-6), "failure decomposition drift")
    final = struct.unpack("<257f", records["zero", roots, "qv"])
    return {"completed": completed, "accepted_roots": roots, "duration_microseconds": roots*timestep,
            "terminal_root_position_m": list(final[:3]), "terminal_root_velocity_m_s": list(final[129:132]),
            "terminal_root_speed_m_s": math.sqrt(sum(v*v for v in final[129:132]))}


def audit_timing(text):
    records = rows(text, "candidate_gpu_timing=")
    require(len(records) == 512 and all(r["valid"] is True for r in records), "incomplete GPU timestamps")
    result = {"query_count": len(records)}
    for field in ("prepare_ns", "kinematics_ns", "materialize_ns"):
        require(all(type(r[field]) is int and r[field] > 0 for r in records), "invalid GPU interval")
        result["median_" + field] = statistics.median(r[field] for r in records)
    return result


def audit_iterates(text):
    encoded = rows(text, "human_support_iterate=")
    require([(r["root"], r["iteration"], r["certificate"]) for r in encoded] ==
            [(9, i, False) for i in range(16)] + [(9, 15, True)], "missing or duplicate Newton iterate")
    decoded = []
    extents = {"q": 129, "delta_v": 160, "free_v": 128, "sample": 18*40,
               "kkt": 18*16, "jacobian": 18*3*160}
    for record in encoded:
        require(set(record["arenas"]) == set(extents), "iterate arena inventory")
        arenas = {}
        for name, size in extents.items():
            raw = bytes.fromhex(record["arenas"][name])
            require(len(raw) == size * 4, "iterate arena extent")
            arenas[name] = struct.unpack("<" + str(size) + "f", raw)
            # Sample identity begins with uint32 indices, including UINT_MAX.
            values = arenas[name] if name != "sample" else tuple(v for i, v in enumerate(arenas[name]) if i % 40 >= 4)
            require(all(math.isfinite(v) for v in values), "nonfinite iterate")
        decoded.append(arenas)
    require(all(a["free_v"] == decoded[0]["free_v"] for a in decoded), "Newton free predictor changed")
    final = decoded[-1]
    residuals = [final["kkt"][i*16:i*16+3] for i in range(18)]
    worst = max(range(18), key=lambda i: sum(v*v for v in residuals[i]))
    norm = math.sqrt(sum(v*v for r in residuals for v in r))
    require(norm > 0.005, "failed support residual disappeared")
    require(worst == 5, "first divergent contact changed")
    steps = []
    for previous, current in zip(decoded, decoded[1:]):
        dv = [v-u for v,u in zip(current["delta_v"], previous["delta_v"])]
        actual = current["q"][2] - previous["q"][2]
        expected = 25e-6 * dv[2]
        steps.append({"stored_root_z_change_m": actual, "requested_root_z_change_m": expected})
    require(any(s["stored_root_z_change_m"] == 0 and abs(s["requested_root_z_change_m"]) > 1e-8 for s in steps),
            "root rounding evidence missing")
    return {"iterations": 17, "support_residual_norm": norm, "worst_contact_row": worst,
            "worst_contact_residual": list(residuals[worst]),
            "observed_root_z_spacing_m": max(abs(s["stored_root_z_change_m"]) for s in steps),
            "root_vertical_steps": steps}


def audit_artifacts(base=BASE):
    receipt = json.loads((base / "receipt.json").read_text())
    require(receipt["schema"] == "numi.human.candidate-precision-performance.v1", "receipt schema")
    actual = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()
              and p.name != "receipt.json" and "__pycache__" not in p.parts}
    require(actual == set(receipt["artifacts"]), "artifact inventory")
    for name, digest in receipt["artifacts"].items():
        require(hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, "artifact drift: " + name)
    require(hashlib.sha256(Path(SPEC.origin).read_bytes()).hexdigest() == receipt["numerics_auditor_sha256"], "prior auditor drift")
    require(set(receipt["dependencies"]) == set(DEPENDENCIES), "evidence dependency inventory")
    for path, digest in receipt["dependencies"].items():
        require(hashlib.sha256((BASE.parent / path).read_bytes()).hexdigest() == digest, "evidence dependency drift")
    def read(label): return data(base / (label + ".log.gz")).decode()
    def launch(label): return json.loads((base / (label + "-launch.json")).read_text())
    measured = {"timing": {}, "source": {}, "refinement": {}}
    original = trace_records(read("timing-baseline4"), NUMERICS.TRACE.SCENARIOS, 4, 100)
    for label in ("timing-baseline4", "timing-width2564", "timing-ancestry4"):
        run = launch(label)
        require(run["returncode"] == 0 and not run["competing_workloads"]
                and run["environment"]["MTL_DEBUG_LAYER"] == "1", "timing run provenance")
        require(trace_records(read(label), NUMERICS.TRACE.SCENARIOS, 4, 100) == original, "optimization changed physics")
        measured["timing"][label] = audit_timing(read(label)) | {"elapsed_seconds": run["elapsed_seconds"]}
    prior_horizon = data(PRIOR / "horizon-wrapped-fiber.log.gz").decode()
    require(trace_records(read("horizon-performance64"), NUMERICS.TRACE.SCENARIOS, 64, 100) ==
            trace_records(prior_horizon, NUMERICS.TRACE.SCENARIOS, 64, 100), "long optimization changed physics")
    for label in ("root-relative-100us", "root-relative-1us", "published-100us", "published-1us"):
        summary = NUMERICS.audit_probe(read(label), NUMERICS.force_scales(), True)
        source_rows = rows(read(label), "prepared_path=")
        full_normalized = max(abs(r["metal_fp32_force_n"]-r["native_fp64_force_n"])/s
                              for r,s in zip(source_rows, NUMERICS.force_scales()))
        measured["source"][label] = {"maximum_path_error_m": summary["maximum_error_m"],
            "maximum_source_force_error_n": summary["maximum_force_error_n"],
            "maximum_normalized_source_force_error": full_normalized}
        require(full_normalized > 1e-5, "unqualified full-source force result was erased")
    for label in ("published-100us", "published-1us"):
        run = launch(label)
        require(run["source_revision"] == receipt["native_revision"] and run["source_status"] == ""
                and run["returncode"] == 0 and not run["competing_workloads"], "published source probe provenance")
    cohort = NUMERICS.TRACE.audit_trace(read("horizon-published64"), 64)
    require(launch("horizon-published64")["returncode"] == 0, "long cohort failed")
    measured["coupled"] = cohort
    NUMERICS.TRACE.audit_trace(read("published4"), 4)
    published = launch("published4")
    for owner in ("native", "brain"):
        require(published["source_state"][owner] == {"revision": receipt[owner + "_revision"], "status": ""}, "published stack drift")
    for path, digest in published["source_sha256"].items():
        if "/MetalRobo-human-completion-20260907/" in path:
            relative = "sources/native/" + path.split("/MetalRobo-human-completion-20260907/", 1)[1]
        else:
            relative = "sources/brain/" + path.split("/numi-brain-human-completion-20260907/", 1)[1]
        require(hashlib.sha256(data(base / (relative + ".gz"))).hexdigest() == digest, "published source snapshot drift")
    long_run = launch("horizon-published64")
    require(long_run["brain_test_binary_sha256"] == published["brain_test_binary_sha256"]
            and long_run["artifact_sha256"] == published["artifact_sha256"]
            and long_run["source_state"] == published["source_state"]
            and long_run["source_sha256"] == published["source_sha256"], "long cohort executable drift")
    measured["coupled_elapsed_seconds"] = long_run["elapsed_seconds"]
    initial = data(PRIOR / "prepared.nhinit")
    reconstructed = (base / "fixture-100us/prepared.nhinit").read_bytes()
    require(reconstructed[96:1124] == initial[96:1124] and reconstructed[1124:] != initial[1124:],
            "retained certificate reconstruction mismatch missing")
    for dt in (100, 50, 25):
        payload = (base / f"fixture-exact-{dt}us/prepared.nhinit").read_bytes()
        require(payload[:8] == b"NHINIT1\0" and payload[96:] == initial[96:], "refinement initial state changed")
        if dt == 100: require(payload == initial, "baseline fixture identity changed")
    require((base / "fixture-exact-25us-iterations32/prepared.nhinit").read_bytes()[96:] == initial[96:], "iteration control state changed")
    for label, dt, count, passed in [("refine-100us",100,16,True), ("refine-50us-native",50,32,True),
            ("refine-25us-components",25,8,False), ("refine-25us-iterations32",25,8,False),
            ("refine-25us-iterate-trace",25,8,False)]:
        require((launch(label)["returncode"] == 0) is passed, "refinement exit status")
        measured["refinement"][label] = audit_refinement(read(label), dt, count, passed)
    for dt, count, passed in [(100,16,True),(50,32,True),(25,8,False)]:
        label = f"refine-{dt}us-published"
        run = launch(label)
        require((run["returncode"] == 0) is passed and run["source_state"] == published["source_state"]
                and run["source_sha256"] == published["source_sha256"]
                and run["brain_test_binary_sha256"] == published["brain_test_binary_sha256"]
                and not run["competing_workloads"], "published refinement provenance")
        require(run["environment"]["NUMANX_PREPARED_TIMESTEP_US"] == str(dt)
                and run["environment"]["NUMANX_PREPARED_DURATION_US"] == "1600", "refinement authored time")
        for key, name in [("NUMANX_INITIAL_STATE", "prepared.nhinit"),
                          ("NUMANX_MATTER_WORLD_PACKAGE", f"prepared-{dt}us.nmatterpack.gz")]:
            expected = hashlib.sha256(data(base / f"fixture-exact-{dt}us" / name)).hexdigest()
            require(run["artifact_sha256"][key] == expected, "refinement input binding")
        for key in ("NUMANX_METALROBO_LIBRARY", "NUMANX_METALROBO_METALLIB", "NUMANX_MATTER_METALLIB"):
            require(run["artifact_sha256"][key] == published["artifact_sha256"][key], "refinement executable drift")
        measured["refinement"][label] = audit_refinement(read(label), dt, count, passed)
    require(launch("refine-50us")["returncode"] != 0 and launch("refine-50us-epoch")["returncode"] != 0,
            "retained bootstrap controls missing")
    require("causal topology" in read("refine-50us-epoch"), "latency contract control missing")
    measured["contact_failure"] = audit_iterates(read("refine-25us-iterate-trace"))
    require(audit_iterates(read("refine-25us-published")) == measured["contact_failure"], "published contact failure drift")
    independent = json.loads((base / "independent-source-path-audit.json").read_text())
    require(independent["passed"] is True and independent["source_files_verified"] == 57
            and independent["initial_muscle_path_diagnostic"]["trace_sha256"] == hashlib.sha256(read("published4").encode()).hexdigest(),
            "independent source audit provenance")
    measured["independent_source_path_error_m"] = independent["initial_muscle_path_diagnostic"]["maximum_length_error_m"]
    require(measured["independent_source_path_error_m"] <= 2e-6, "independent path gate")
    require("100% tests passed out of 10" in read("native-published-regressions"), "native regressions")
    require("status=failed" in read("operator-context-final"), "retained allocation failure missing")
    for label in ("operator-host-final", "operator-context-lazy"):
        require("status=ok" in read(label), "operator qualification missing")
    require("generic articulated operator probe passed" in read("operator-gpu-final"), "generic GPU qualification missing")
    require(receipt["qualification"] == {"bounded_candidate_speedup": True, "improved_static_geometry": True,
        "bounded_coupled_replay": True, "timestep_convergence": False, "full_source_force_convergence": False,
        "registered_anatomical_tissue": False, "experimental_calibration": False, "standing_walking": False,
        "performance_envelope": False, "full_release": False}, "qualification boundary")
    return measured, receipt


def verify(base=BASE):
    measured, receipt = audit_artifacts(base)
    require(measured == receipt["measurements"], "recomputed measurement drift")
    return measured


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
