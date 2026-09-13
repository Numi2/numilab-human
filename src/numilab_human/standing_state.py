from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from .model import ImportError, write_json

SCHEMA = "numi.human.standing-initial-state-audit.v1"
NQ = 129
NV = 128
NMUSCLE = 416


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError(message)


def decode_initial_state(raw: bytes) -> dict[str, Any]:
    _require(len(raw) >= 96, "prepared state is truncated")
    magic = raw[:8]
    _require(magic in {b"NHINIT1\0", b"NHINIT2\0"}, "unsupported initial-state ABI")
    version, header_bytes, nq, nv, muscles, width, flag0, flag1 = struct.unpack_from("<8I", raw, 8)
    _require((nq, nv, muscles, width) == (NQ, NV, NMUSCLE, 4), "unexpected Human state dimensions")
    _require(flag1 == 0, "reserved initial-state flag is non-zero")
    human_fp, world_fp, legacy_us = struct.unpack_from("<3Q", raw, 40)
    if magic == b"NHINIT1\0":
        _require((version, header_bytes, flag0) == (1, 96, 0), "invalid NHINIT1 header")
        clock_ns = legacy_us * 1000
        clock_authority = "legacy_microseconds"
    else:
        _require((version, header_bytes, flag0) == (2, 160, 1), "invalid NHINIT2 header")
        _require(len(raw) >= 160 and raw[152:160] == bytes(8), "invalid NHINIT2 extension")
        clock_ns = struct.unpack_from("<Q", raw, 96)[0]
        _require(clock_ns > 0, "NHINIT2 has no authoritative nanosecond clock")
        clock_authority = "nanoseconds"
    count = NQ + NV + 4 * NMUSCLE
    _require(len(raw) == header_bytes + 4 * count, "prepared-state byte count mismatch")
    values = struct.unpack_from(f"<{count}f", raw, header_bytes)
    _require(all(math.isfinite(value) for value in values), "prepared state contains non-finite values")
    q = list(values[:NQ])
    v = list(values[NQ:NQ + NV])
    muscle_values = values[NQ + NV:]
    muscle_state = []
    for index in range(0, len(muscle_values), 4):
        excitation, activation, fiber_length, fiber_velocity = muscle_values[index:index + 4]
        _require(0.0 <= excitation <= 1.0 and 0.0 <= activation <= 1.0,
                 "prepared activation is outside [0, 1]")
        _require(fiber_length > 0.0, "prepared fiber length is not positive")
        muscle_state.append((excitation, activation, fiber_length, fiber_velocity))
    return {
        "abi": version,
        "header_bytes": header_bytes,
        "human_source_fp": f"{human_fp:016x}",
        "world_fp": f"{world_fp:016x}",
        "clock_nanoseconds": clock_ns,
        "clock_authority": clock_authority,
        "source_sha256": raw[64:96].hex(),
        "q": q,
        "v": v,
        "muscles": muscle_state,
    }


def audit(arguments: argparse.Namespace) -> int:
    path = arguments.initial_state.resolve()
    raw = path.read_bytes()
    state = decode_initial_state(raw)
    epsilon = arguments.activation_epsilon
    _require(math.isfinite(epsilon) and 0.0 < epsilon < 0.1, "activation epsilon is invalid")
    _require(math.isfinite(arguments.maximum_initial_speed) and arguments.maximum_initial_speed >= 0.0,
             "maximum initial speed is invalid")
    _require(math.isfinite(arguments.maximum_fiber_speed) and arguments.maximum_fiber_speed >= 0.0,
             "maximum fiber speed is invalid")
    activations = [row[1] for row in state["muscles"]]
    excitations = [row[0] for row in state["muscles"]]
    max_velocity = max((abs(value) for value in state["v"]), default=0.0)
    max_fiber_velocity = max((abs(row[3]) for row in state["muscles"]), default=0.0)
    saturated = sum(value >= 1.0 - epsilon for value in activations)
    nonzero = sum(value > epsilon for value in activations)
    uniform_maximal = saturated == NMUSCLE and all(value >= 1.0 - epsilon for value in excitations)
    reasons = []
    if uniform_maximal:
        reasons.append("uniform maximal activation is a force-path diagnostic, not an equilibrium stance")
    if max_velocity > arguments.maximum_initial_speed:
        reasons.append("generalized velocity exceeds the standing-state bound")
    if max_fiber_velocity > arguments.maximum_fiber_speed:
        reasons.append("fiber velocity exceeds the standing-state bound")
    candidate = not reasons
    receipt = {
        "schema": SCHEMA,
        "status": "standing_state_candidate" if candidate else "partial",
        "initial_state": {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "abi": state["abi"],
            "human_source_fp": state["human_source_fp"],
            "world_fp": state["world_fp"],
            "source_sha256": state["source_sha256"],
            "clock_nanoseconds": state["clock_nanoseconds"],
            "clock_authority": state["clock_authority"],
        },
        "state": {
            "activation_min": min(activations),
            "activation_max": max(activations),
            "activation_mean": sum(activations) / NMUSCLE,
            "activation_nonzero_count": nonzero,
            "activation_upper_bound_count": saturated,
            "uniform_maximal_activation": uniform_maximal,
            "maximum_generalized_speed": max_velocity,
            "maximum_fiber_speed_m_s": max_fiber_velocity,
        },
        "qualification": {
            "nonmaximal_recruitment": not uniform_maximal,
            "stationary_generalized_state": max_velocity <= arguments.maximum_initial_speed,
            "stationary_fibers": max_fiber_velocity <= arguments.maximum_fiber_speed,
            "standing_initial_state_candidate": candidate,
            "force_convergence": False,
            "sustained_standing": False,
            "walking": False,
        },
        "gate": {"reasons": reasons},
    }
    write_json(arguments.output.resolve(), receipt)
    print(json.dumps({"status": receipt["status"], "activation_nonzero_count": nonzero,
                      "uniform_maximal_activation": uniform_maximal}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--maximum-initial-speed", type=float, default=1.0e-6)
    parser.add_argument("--maximum-fiber-speed", type=float, default=1.0e-6)
    parser.add_argument("--activation-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a prepared Human standing initial state")
    add_arguments(parser)
    return audit(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
