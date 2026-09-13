"""Independently join committed root expansions to complete native trace bytes.

This consumes retained observations only. It does not step physics or establish
spatial/temporal convergence, anatomical qualification, or sustained behavior.
"""
import argparse
import base64
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct

COUNTS = {"qv": 257, **{name: 416 for name in (
    "motor", "activation", "fiber_length", "fiber_velocity", "path_length",
    "path_velocity", "applied_force", "tendon_tension", "fiber_residual")}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def round_exact_f32(value):
    """Round an exact rational using adjacent FP32 values and ties-to-even.

    The initial Float64 approximation may double-round at a midpoint; choosing
    among its finite FP32 neighbors with exact distances removes that ambiguity.
    """
    word = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    candidates = {word, max(0, word - 1), min(0xffffffff, word + 1), 0, 0x80000000}
    finite = [w for w in candidates if math.isfinite(f32(w))]
    return min(finite, key=lambda w: (abs(Fraction(f32(w)) - value), w & 1, w))


def audit(log, *, timestep_us, duration_us, initial_time_us=100):
    require(timestep_us > 0 and duration_us > 0 and duration_us % timestep_us == 0,
            "invalid physical interval")
    roots = duration_us // timestep_us
    records, translations = {}, {}
    pending = None
    committed_record = None
    kinds = tuple(COUNTS)
    expected_pairs = [(scenario, root) for scenario in ("zero", "replay")
                      for root in range(1, roots + 1)]
    pair_index = kind_index = 0
    prior_transactions = {"zero": set(), "replay": set()}
    decoder = json.JSONDecoder()
    pattern = r"(mrnx_accepted_root_translation=|prepared_recruitment=)(?=\{)"
    for match in re.finditer(pattern, log):
        row, end = decoder.raw_decode(log, match.end())
        require(log[end:end+1] == "\n", "truncated or interleaved native record")
        if match.group(1) == "mrnx_accepted_root_translation=":
            require(pair_index < len(expected_pairs) and kind_index == 0 and pending is None,
                    "unpaired, out-of-order or duplicated committed root")
            require(row["root"] == expected_pairs[pair_index][1], "committed root order mismatch")
            require(row["schema"] == "numi.human.accepted-root-translation.v1" and row["valid"] is True,
                    "invalid committed translation record")
            require(type(row["root"]) is int and 1 <= row["root"] <= roots and
                    type(row["physics_generation"]) is int and row["physics_generation"] == row["root"], "committed generation mismatch")
            require(type(row["timestamp_microseconds"]) is int and row["timestamp_microseconds"] == initial_time_us + row["root"] * timestep_us,
                    "committed root clock mismatch")
            require(re.fullmatch(r"[0-9a-f]{16}", row["transaction_fingerprint"]) is not None and
                    int(row["transaction_fingerprint"], 16) != 0, "invalid transaction fingerprint")
            require(type(row["words"]) is list and len(row["words"]) == 12 and
                    all(re.fullmatch(r"[0-9a-f]{8}", w) for w in row["words"]), "translation byte extent")
            scenario = expected_pairs[pair_index][0]
            require(row["transaction_fingerprint"] not in prior_transactions[scenario],
                    "transaction fingerprint repeated within a scenario")
            prior_transactions[scenario].add(row["transaction_fingerprint"])
            words = [int(w, 16) for w in row["words"]]
            require(all(math.isfinite(f32(w)) for w in words), "nonfinite translation")
            require(all(words[i] == 0 for i in (3, 7, 11)), "noncanonical translation padding")
            pending = (row, words)
            committed_record = row
            continue
        require(row["schema"] == "numi.human.prepared-recruitment-trace.v3", "trace schema changed")
        key = row["scenario"], row["root"], row["kind"]
        require(key[0] in ("zero", "replay") and type(key[1]) is int and 1 <= key[1] <= roots and
                key[2] in COUNTS and key not in records, "unknown or duplicate trace identity")
        require(pair_index < len(expected_pairs) and key[:2] == expected_pairs[pair_index] and
                key[2] == kinds[kind_index], "trace root/scenario/kind order mismatch")
        require(type(row["elapsed_microseconds"]) is int and row["elapsed_microseconds"] == key[1] * timestep_us,
                "trace elapsed time mismatch")
        require(committed_record is not None and
                row.get("transaction_fingerprint") == committed_record["transaction_fingerprint"] and
                type(row.get("physics_generation")) is int and
                row["physics_generation"] == committed_record["physics_generation"] and
                type(row.get("accepted_timestamp_microseconds")) is int and
                row["accepted_timestamp_microseconds"] == committed_record["timestamp_microseconds"],
                "trace identity disagrees with committed publication")
        raw = base64.b64decode(row["fp32_le_base64"], validate=True)
        require(len(raw) == 4 * COUNTS[key[2]], "trace byte extent mismatch")
        values = struct.unpack("<" + str(COUNTS[key[2]]) + "f", raw)
        require(all(math.isfinite(x) for x in values), "nonfinite accepted trace")
        if key[2] == "qv":
            require(pending is not None and pending[0]["root"] == key[1], "q/v lacks committed expansion")
            pair_key = key[:2]
            require(pair_key not in translations, "duplicate root expansion")
            words = pending[1]
            q_words = struct.unpack("<3I", raw[:12])
            for axis in range(3):
                exact = sum((Fraction(f32(words[axis + offset])) for offset in (0, 4, 8)), Fraction())
                # Immutable unadvanced signed zero follows the retained legacy q.
                expected = words[axis] if exact == 0 and words[axis + 4] == words[axis + 8] == 0 else round_exact_f32(exact)
                require(expected == q_words[axis], "q projection disagrees with exact authored expansion")
                displacement = Fraction(f32(words[axis+4])) + Fraction(f32(words[axis+8]))
                normalized_high = round_exact_f32(displacement)
                normalized_low = round_exact_f32(displacement - Fraction(f32(normalized_high)))
                require((normalized_high, normalized_low) == (words[axis+4], words[axis+8]),
                        "displacement pair is not canonical")
            require(abs(sum(x*x for x in values[3:7]) - 1) <= 16 * 2**-23, "root quaternion drift")
            translations[pair_key] = words
            pending = None
        if key[2] == "motor":
            require(all(x == 0 for x in values), "refinement motor command changed")
        if key[2] == "activation":
            require(all(0 <= x <= 1 for x in values), "activation outside source bounds")
        records[key] = raw
        kind_index += 1
        if kind_index == len(kinds):
            kind_index = 0
            pair_index += 1
            committed_record = None
    require(pair_index == len(expected_pairs) and kind_index == 0 and pending is None and len(records) == 2 * roots * len(COUNTS) and
            len(translations) == 2 * roots, "incomplete root/scenario evidence")
    reference = translations["zero", 1][:4]
    for root in range(1, roots + 1):
        require(translations["zero", root][:4] == reference, "episode translation reference changed")
        require(translations["zero", root] == translations["replay", root], "retained translation replay drift")
        for kind in COUNTS:
            require(records["zero", root, kind] == records["replay", root, kind], "accepted trace replay drift")
    marker = f"prepared_timestep=observed roots_per_scenario={roots} timestep_us={timestep_us} duration_us={duration_us} replay=bitwise"
    require(marker in log and "Executed 1 test, with 0 failures" in log and
            "0 unexpected" in log, "missing native test completion")
    last = translations["zero", roots]
    physical = [float(sum((Fraction(f32(last[a+i])) for i in (0, 4, 8)), Fraction())) for a in range(3)]
    return {"accepted_roots_per_scenario": roots, "duration_microseconds": duration_us,
            "timestep_microseconds": timestep_us, "translation_bytes_per_root": 48,
            "complete_trace_replay": "bitwise", "terminal_root_position_m": physical,
            "qualification": "bounded_native_execution_and_replay_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--timestep-us", type=int, required=True)
    parser.add_argument("--duration-us", type=int, default=1600)
    args = parser.parse_args()
    execution = json.loads(args.execution.read_text())
    require(execution["status"] == "pass" and type(execution["returncode"]) is int and execution["returncode"] == 0 and
            execution["source_unchanged"] is True and execution["artifacts_unchanged"] is True, "execution was not qualified")
    raw = args.log.read_bytes()
    if args.log.suffix == ".gz":
        raw = gzip.decompress(raw)
    require(execution["log_identity"] == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "retained log identity mismatch")
    print(json.dumps(audit(raw.decode(), timestep_us=args.timestep_us, duration_us=args.duration_us), indent=2))


if __name__ == "__main__":
    main()
