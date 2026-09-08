"""Offline source-bound muscle control authoring; Brain/Metal owns execution."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path


def _fnv(payload: bytes, seed: int = 0xcbf29ce484222325) -> int:
    for byte in payload:
        seed = ((seed ^ byte) * 0x100000001b3) & (2**64 - 1)
    return seed


def _prepared_recruitment(body: dict, state: bytes, muscles: int,
                          source: bytes) -> list[float]:
    identity = body.get("prepared_initial_state")
    if not isinstance(identity, dict) or len(state) < 96:
        raise ValueError("prepared recruitment requires the native-admitted NHINIT1 identity")
    magic, abi, header, nq, nv, count, scalar, flags, reserved, human, world, clock, archive = struct.unpack_from("<8s8I3Q32s", state)
    fingerprint = _fnv(state)
    if (magic != b"NHINIT1\0" or (abi, header, count, scalar, flags, reserved) != (1, 96, muscles, 4, 0, 0)
            or nq < 7 or nv < 6 or nq > 65536 or nv > 65536
            or len(state) != 96 + 4 * (nq + nv + 4 * muscles)
            or archive != source or not human or not world or not clock
            or any(type(identity.get(k)) is not int for k in ("fingerprint", "world_fingerprint", "q_count", "dof_count"))
            or identity.get("sha256") != hashlib.sha256(state).hexdigest()
            or (identity["fingerprint"], identity["world_fingerprint"], identity["q_count"], identity["dof_count"]) != (fingerprint, world, nq, nv)
            or type(body.get("timestep_microseconds")) is not int or body["timestep_microseconds"] != clock
            or _fnv(b"NHINIT1" + struct.pack("<Q", fingerprint), human) != body["model_source_fingerprint"]):
        raise ValueError("prepared recruitment source, dimensions, world, clock or artifact identity mismatch")
    values = struct.unpack_from(f"<{nq + nv + 4 * muscles}f", state, 96)
    if (not all(math.isfinite(x) for x in values)
            or abs(sum(x*x for x in values[3:7]) - 1) > 16 * 2**-23
            or any(x != 0 for x in values[nq:nq+nv])):
        raise ValueError("prepared recruitment requires a finite stationary pose and unit root quaternion")
    recruitment = []
    for i in range(muscles):
        excitation, activation, fiber, velocity = values[nq+nv+4*i:nq+nv+4*i+4]
        if not (0 <= excitation <= 1 and excitation == activation and fiber > 0 and velocity == 0):
            raise ValueError("prepared recruitment requires matched static excitation/activation and fibre state")
        recruitment.append(excitation)
    return recruitment


def compile_program(body: dict, payload: bytes, *, tonic: float | None = None, length_gain: float,
                    velocity_gain: float, maximum: float = 0.95,
                    period_us: int = 0, gait: list[dict] | None = None,
                    prepared_state: bytes | None = None) -> dict:
    if (not isinstance(body, dict) or body.get("format") != "numanx-locomotor-body-v1"
            or body.get("muscle_payload_sha256") != hashlib.sha256(payload).hexdigest()
            or any(type(body.get(k)) is not int or not 0 < body[k] < 2**64
                   for k in ("model_source_fingerprint", "sensory_profile_fingerprint"))):
        raise ValueError("native body description does not bind this muscle payload")
    if len(payload) < 76:
        raise ValueError("truncated muscle header")
    magic, abi, bodies, muscles, sites, wraps, routes, tendons, architectures, stride, source = struct.unpack_from("<8s9I32s", payload)
    if magic != b"NHMYO2\0\0" or abi != 2 or not 0 < muscles <= 4096 or type(body.get("actuator_count")) is not int or body.get("actuator_count") != muscles:
        raise ValueError("locomotor authoring requires the admitted NHMYO2 source and actuator count")
    offset = 76 + sites * 16 + wraps * 64 + routes * 16
    if architectures != muscles or stride != 32 or len(payload) != offset + muscles * 164 + architectures * stride:
        raise ValueError("invalid NHMYO2 table extent")
    if prepared_state is not None:
        if tonic is not None or length_gain != 0 or velocity_gain != 0 or period_us != 0 or gait is not None:
            raise ValueError("prepared recruitment is a static tonic candidate; feedback and gait require separately authored references")
        recruitment = _prepared_recruitment(body, prepared_state, muscles, source)
    else:
        if body.get("prepared_initial_state") is not None:
            raise ValueError("native prepared body requires its matching prepared-state recruitment artifact")
        recruitment = [tonic] * muscles
    if (not all(type(x) in (int, float) and math.isfinite(x) for x in (*recruitment, length_gain, velocity_gain, maximum))
            or not 0 <= maximum <= 1 or any(not 0 <= x <= maximum for x in recruitment) or not 0 <= length_gain <= 10
            or not 0 <= velocity_gain <= 1 or type(period_us) is not int
            or (period_us != 0 and not 100_000 <= period_us <= 10_000_000)):
        raise ValueError("invalid excitation, feedback gains or physical period")
    if gait is not None and not isinstance(gait, list):
        raise ValueError("gait map must be an array of explicit muscle rows")
    phases = {}
    for row in gait or []:
        if not isinstance(row, dict) or set(row) != {"muscleIdentifier", "gaitSine", "gaitCosine"}:
            raise ValueError("gait rows require explicit muscle identifier and sine/cosine coefficients")
        i, a, b = row["muscleIdentifier"], row["gaitSine"], row["gaitCosine"]
        if (type(i) is not int or not 0 <= i < muscles or i in phases
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (a, b))
                or abs(a) + abs(b) > 0.25 or period_us == 0):
            raise ValueError("invalid, duplicate or unclocked gait row")
        phases[i] = (a, b)
    channels = []
    for i in range(muscles):
        reference_length = struct.unpack_from("<f", payload, offset + i * 164 + 16 + 35 * 4)[0]
        if not math.isfinite(reference_length) or not 0.0001 <= reference_length <= 10:
            raise ValueError("source reference path length is invalid")
        a, b = phases.get(i, (0, 0))
        channels.append(dict(muscleIdentifier=i, referenceLengthMeters=reference_length,
                             tonicExcitation=recruitment[i], lengthGain=length_gain,
                             velocityGainSeconds=velocity_gain, gaitSine=a, gaitCosine=b,
                             maximumExcitation=maximum))
    return dict(version=1, modelSourceFingerprint=body["model_source_fingerprint"],
                sensoryProfileFingerprint=body["sensory_profile_fingerprint"],
                calibrationArtifactSHA256=hashlib.sha256(prepared_state if prepared_state is not None else payload).hexdigest(),
                epochMicroseconds=0, periodMicroseconds=period_us, channels=channels)


def add_arguments(parser):
    parser.add_argument("body_description", type=Path)
    parser.add_argument("muscle_payload", type=Path)
    parser.add_argument("output", type=Path)
    recruitment = parser.add_mutually_exclusive_group(required=True)
    recruitment.add_argument("--tonic", type=float)
    recruitment.add_argument("--prepared-state", type=Path, help="native-admitted stationary NHINIT1 recruitment; requires zero feedback gains and period")
    parser.add_argument("--length-gain", type=float, required=True)
    parser.add_argument("--velocity-gain-seconds", type=float, required=True)
    parser.add_argument("--maximum-excitation", type=float, default=0.95)
    parser.add_argument("--period-microseconds", type=int, default=0)
    parser.add_argument("--gait-map", type=Path)
    parser.set_defaults(handler=run)


def run(args):
    try:
        program = compile_program(json.loads(args.body_description.read_text()), args.muscle_payload.read_bytes(),
                                  tonic=args.tonic, length_gain=args.length_gain,
                                  velocity_gain=args.velocity_gain_seconds, maximum=args.maximum_excitation,
                                  period_us=args.period_microseconds,
                                  gait=json.loads(args.gait_map.read_text()) if args.gait_map else None,
                                  prepared_state=args.prepared_state.read_bytes() if args.prepared_state else None)
    except (ValueError, TypeError, struct.error) as error:
        from .model import ImportError as HumanImportError
        raise HumanImportError(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(program, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"program": str(args.output), "channels": len(program["channels"]),
                      "period_microseconds": program["periodMicroseconds"], "promotable": False,
                      "boundary": "immutable recruitment authoring; source reference lengths; no loaded balance or experimental calibration"}))
    return 0
