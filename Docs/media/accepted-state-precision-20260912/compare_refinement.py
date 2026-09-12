"""Report common-time differences without declaring timestep convergence.

All inputs must first pass the complete accepted-state/replay verifier.
The fourth registered level, 12.5 microseconds, is still unavailable through
the integer-microsecond joint transaction clock.
"""
import base64
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct

from verify_trajectory import audit, f32, require

RUNS = ((100, "precision-100us-002"), (50, "precision-50us-002"), (25, "precision-25us-004"))
FIXTURE_ENV = {"NUMANX_MATTER_WORLD_PACKAGE", "NUMANX_MATTER_WORLD_FP",
               "NUMANX_INITIAL_STATE", "NUMANX_INITIAL_STATE_FP", "NUMANX_PREPARED_TIMESTEP_US"}
STATIC_FILES = ("NUMANX_METALROBO_LIBRARY", "NUMANX_METALROBO_METALLIB", "NUMANX_MATTER_METALLIB",
                "NUMANX_FULLBODY_RIGID", "NUMANX_FULLBODY_MUSCLE", "NUMANX_FULLBODY_CONTACT",
                "NUMANX_FULLBODY_VISUAL_PACK", "NUMANX_FULLBODY_VISION_PROFILE", "NUMANX_MATTER_MATERIAL",
                "NUMANX_JOINT_EQUALITIES", "NUMANX_JOINT_LIMITS", "NUMANX_PREPARED_RECRUITMENT")


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fnv(raw, seed=1469598103934665603):
    for byte in raw:
        seed = ((seed ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return seed


def validate_receipt(receipt, timestep):
    require(receipt["schema"] == "numilab-human.native-check-execution.v1", "execution schema mismatch")
    require(receipt["status"] == "pass" and type(receipt["returncode"]) is int and receipt["returncode"] == 0
            and receipt["source_unchanged"] is True and receipt["artifacts_unchanged"] is True,
            "unqualified refinement input")
    for name in ("source", "artifacts"):
        before, after = receipt[name + "_before"], receipt[name + "_after"]
        require(type(before) is dict and before and before == after, "inconsistent retained " + name + " identities")
        require(all(type(v) is dict and set(v) == {"bytes", "sha256"} and type(v["bytes"]) is int
                    and v["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", v["sha256"])
                    for v in before.values()), "invalid retained " + name + " identity")
    request = receipt["request"]
    env = request["environment"]
    require(type(env) is dict and all(type(v) is str for v in env.values()), "invalid execution environment")
    require(timestep in (25, 50, 100) and env["NUMANX_PREPARED_TIMESTEP_US"] == str(timestep)
            and env["NUMANX_PREPARED_DURATION_US"] == "1600", "request clock mismatch")
    require(env.get("NM_HUMAN_SUPPORT_TRACE_ROOT") in (None, "5"), "unregistered trace selector")
    require(set(request["sources"]) == set(receipt["source_before"]), "source inventory mismatch")
    require(request["binaries"] and all(path in receipt["artifacts_before"] for path in request["binaries"]),
            "missing native test executable identity")
    for name in (*STATIC_FILES, "NUMANX_MATTER_WORLD_PACKAGE", "NUMANX_INITIAL_STATE"):
        require(env[name] in receipt["artifacts_before"], "missing authored artifact identity: " + name)


def compatible_receipts(receipts):
    """Only timestep fixtures and a named diagnostic selector may vary."""
    baseline = None
    for timestep, receipt in receipts.items():
        validate_receipt(receipt, timestep)
        request, artifacts = receipt["request"], receipt["artifacts_before"]
        env = request["environment"]
        fixture_paths = {env["NUMANX_MATTER_WORLD_PACKAGE"], env["NUMANX_INITIAL_STATE"]}
        signature = (receipt["source_before"], receipt["runner_identity"],
                     {k: request[k] for k in ("command", "cwd", "binaries", "repositories")},
                     {k: v for k, v in env.items() if k not in FIXTURE_ENV | {"NM_HUMAN_SUPPORT_TRACE_ROOT"}},
                     {k: v for k, v in artifacts.items() if k not in fixture_paths})
        require(baseline is None or signature == baseline, "refinement source/build/static-input identity mismatch")
        baseline = signature


def fixture_state(base, timestep, receipt):
    """Bind actual copied NHINIT2 and package18/ABI34 bytes to each run.

    Compare all authored state bytes after removing only verified clock fields
    and their dependent world/header hashes; no source or solver field is ignored.
    """
    directory = base / f"precision-fixture-{timestep}us"
    package_name = f"prepared-{timestep * 1000}ns.nmatterpack"
    compiler = json.loads((directory / "execution.json").read_text())
    require(type(compiler["returncode"]) is int and compiler["returncode"] == 0
            and compiler["initial_q_v_muscle_bytes_unchanged"] is True
            and type(compiler["initial_header_bytes"]) is int and compiler["initial_header_bytes"] == 160,
            "unqualified fixture compilation")
    payloads = {name: (directory / name).read_bytes() for name in
                ("prepared.json", "prepared.nhinit", package_name, "compile.log")}
    require(all(compiler[name] == identity(raw) for name, raw in payloads.items()), "fixture artifact identity mismatch")
    metadata = json.loads(payloads["prepared.json"])
    require(set(metadata) == {"schema", "human_source_fp", "composed_human_source_fp", "world_fp", "initial_state_fp", "scope"}
            and metadata["schema"] == "numi.human.prepared-stance-fixture.v1", "fixture schema mismatch")
    require(all(re.fullmatch(r"[0-9a-f]{16}", metadata[k]) and int(metadata[k], 16) != 0
                for k in ("human_source_fp", "composed_human_source_fp", "world_fp", "initial_state_fp")),
            "fixture fingerprint is invalid")
    env = receipt["request"]["environment"]
    require(env["NUMANX_HUMAN_SOURCE_FP"] == metadata["human_source_fp"]
            and env["NUMANX_MATTER_WORLD_FP"] == metadata["world_fp"]
            and env["NUMANX_INITIAL_STATE_FP"] == metadata["initial_state_fp"], "fixture/request fingerprint mismatch")
    for key, name in (("NUMANX_INITIAL_STATE", "prepared.nhinit"), ("NUMANX_MATTER_WORLD_PACKAGE", package_name)):
        remote = Path(env[key])
        require(remote.name == name and remote.parent.name == directory.name
                and receipt["artifacts_before"][env[key]] == identity(payloads[name]), "fixture/request artifact mismatch")
    initial = payloads["prepared.nhinit"]
    require(len(initial) == 7844 and initial[:8] == b"NHINIT2\0"
            and struct.unpack_from("<8I", initial, 8) == (2, 160, 129, 128, 416, 4, 1, 0), "NHINIT2 fixture schema mismatch")
    require(struct.unpack_from("<QQQ", initial, 40) ==
            (int(metadata["composed_human_source_fp"], 16), int(metadata["world_fp"], 16), timestep)
            and struct.unpack_from("<Q", initial, 96)[0] == timestep * 1000
            and initial[152:160] == bytes(8), "NHINIT2 source/world/clock mismatch")
    require(f"{fnv(initial, 14695981039346656037):016x}" == metadata["initial_state_fp"], "NHINIT2 fingerprint mismatch")
    normalized_initial = initial[:48] + bytes(16) + initial[64:96] + bytes(8) + initial[104:]
    package = payloads[package_name]
    require(package[:16] == b"NUMIMATTERPKG\0\0\0" and
            struct.unpack_from("<4I", package, 16) == (18, 0x01020304, 34, 64), "package schema mismatch")
    world, fingerprint, header_hash = struct.unpack_from("<3Q", package, 32)
    require(world == fingerprint == int(metadata["world_fp"], 16)
            and header_hash == fnv(package[:48] + bytes(8)), "package world/header identity mismatch")
    offset, sections = 56, []
    # V18 appends the cavity sections before the pre-existing generated text.
    for expected_id in (*range(1, 58), *range(59, 64), 58, 64):
        require(offset + 32 <= len(package), "truncated package section")
        section, size, count, byte_count, content_hash = struct.unpack_from("<IIQQQ", package, offset)
        offset += 32
        require(section == expected_id and size * count == byte_count and offset + byte_count <= len(package),
                "package section layout mismatch")
        data = package[offset:offset + byte_count]
        require(fnv(data) == content_hash, "package section hash mismatch")
        if section == 1:
            require(size == 192 and count == 1 and struct.unpack_from("<I", data)[0] == 34
                    and data[172:176] == struct.pack("<f", timestep / 1_000_000), "package timestep mismatch")
            data = data[:172] + bytes(4) + data[176:]
        sections.append((section, size, count, data))
        offset += byte_count
    require(offset == len(package), "extra package payload")
    return ({k: v for k, v in metadata.items() if k not in ("world_fp", "initial_state_fp")},
            normalized_initial, sections)


def validate_inputs(base):
    receipts = {step: json.loads((base / name / "execution.json").read_text()) for step, name in RUNS}
    compatible_receipts(receipts)
    states = [fixture_state(base, step, receipt) for step, receipt in receipts.items()]
    require(all(state == states[0] for state in states[1:]), "refinement authored payload differs beyond timestep")
    return receipts


def read_run(directory, timestep):
    receipt = json.loads((directory / "execution.json").read_text())
    raw = gzip.decompress((directory / "run.log.gz").read_bytes())
    validate_receipt(receipt, timestep)
    require(receipt["log_identity"] == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "refinement log identity mismatch")
    log = raw.decode()
    audit(log, timestep_us=timestep, duration_us=1600)
    decoder, frames, pending = json.JSONDecoder(), {}, None
    for match in re.finditer(r"(mrnx_accepted_root_translation=|prepared_recruitment=)(?=\{)", log):
        row, _ = decoder.raw_decode(log, match.end())
        if match.group(1) == "mrnx_accepted_root_translation=":
            words = [int(word, 16) for word in row["words"]]
            pending = [sum((Fraction(f32(words[axis + offset])) for offset in (0, 4, 8)), Fraction())
                       for axis in range(3)]
        elif row["scenario"] == "zero":
            frame = frames.setdefault(row["elapsed_microseconds"], {})
            data = base64.b64decode(row["fp32_le_base64"], validate=True)
            frame[row["kind"]] = struct.unpack("<" + str(len(data) // 4) + "f", data)
            if row["kind"] == "qv":
                frame["root_translation_m"] = pending
    return frames


def compare(coarse, fine, sample_times):
    differences = []
    for timestamp in sample_times:
        a, b = coarse[timestamp], fine[timestamp]
        delta = lambda name: max(float(abs(x-y)) for x, y in zip(a[name], b[name]))
        differences.append({"elapsed_microseconds": timestamp,
                            "root_translation_max_axis_difference_m": delta("root_translation_m"),
                            "applied_force_max_difference_n": delta("applied_force"),
                            "tendon_tension_max_difference_n": delta("tendon_tension"),
                            "path_length_max_difference_m": delta("path_length"),
                            "fiber_length_max_difference_m": delta("fiber_length"),
                            "activation_max_difference": delta("activation"),
                            "qv_max_difference_in_mixed_native_units": delta("qv")})
    return {"common_samples": len(differences), "samples": differences,
            "maximum_over_common_samples": {key: max(row[key] for row in differences)
                for key in differences[0] if key != "elapsed_microseconds"}}


if __name__ == "__main__":
    base = Path(__file__).parent
    validate_inputs(base)
    runs = {step: read_run(base / name, step) for step, name in RUNS}
    sample_times = sorted(set(runs[100]) & set(runs[50]) & set(runs[25]))
    print(json.dumps({"schema": "numi.human.refinement-observation.v1", "duration_microseconds": 1600,
                      "differences": {"100_vs_50_us": compare(runs[100], runs[50], sample_times),
                                      "50_vs_25_us": compare(runs[50], runs[25], sample_times)},
                      "convergence_qualified": False,
                      "remaining_gate": "12.5-us transaction clock, full source-force consistency and registered convergence criteria"}, indent=2))
