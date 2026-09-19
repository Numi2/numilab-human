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


def _fnv_u64(value: int, seed: int) -> int:
    return _fnv(struct.pack("<Q", value), seed)


def _float_u64(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _baseline_fingerprint(program: dict) -> int:
    """Match ``MuscleLocomotorProgram.baselineFingerprint`` byte-for-byte."""
    seed = _fnv(b"NBMUSCLELOCOMOTOR1")
    for value in (
        1,
        program["modelSourceFingerprint"],
        program["sensoryProfileFingerprint"],
    ):
        seed = _fnv_u64(value, seed)
    seed = _fnv(program["calibrationArtifactSHA256"].encode("ascii"), seed)
    for value in (
        program["epochMicroseconds"],
        program["periodMicroseconds"],
        len(program["channels"]),
    ):
        seed = _fnv_u64(value, seed)
    for channel in program["channels"]:
        seed = _fnv_u64(channel["muscleIdentifier"], seed)
        for key in (
            "referenceLengthMeters",
            "tonicExcitation",
            "lengthGain",
            "velocityGainSeconds",
            "gaitSine",
            "gaitCosine",
            "maximumExcitation",
        ):
            seed = _fnv_u64(_float_u64(channel[key]), seed)
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


def _balance_feedback(program: dict, balance: dict, artifact: bytes, muscles: int) -> dict:
    expected = {
        "format", "modelSourceFingerprint", "sensoryProfileFingerprint", "mode",
        "updatePeriodMicroseconds", "initializationDurationMicroseconds",
        "sources", "routes",
    }
    if not isinstance(balance, dict) or set(balance) != expected \
            or balance.get("format") != "numi-human-muscle-balance-map-v1":
        raise ValueError("balance map requires the exact v1 authoring fields")
    try:
        decoded = json.loads(artifact)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("balance calibration artifact is not valid JSON") from error
    if decoded != balance:
        raise ValueError("balance calibration bytes do not encode the supplied map")
    if (type(balance["modelSourceFingerprint"]) is not int
            or balance["modelSourceFingerprint"] != program["modelSourceFingerprint"]
            or type(balance["sensoryProfileFingerprint"]) is not int
            or balance["sensoryProfileFingerprint"] != program["sensoryProfileFingerprint"]):
        raise ValueError("balance map belongs to another body or sensory generation")
    modes = {"posture": 1, "supportAware": 2}
    mode = balance["mode"]
    if not isinstance(mode, str) or mode not in modes:
        raise ValueError("balance mode must be posture or supportAware")
    update = balance["updatePeriodMicroseconds"]
    initialization = balance["initializationDurationMicroseconds"]
    if (type(update) is not int or not 1_000 <= update <= 20_000
            or type(initialization) is not int or not 0 <= initialization <= 1_000_000
            or initialization % update != 0):
        raise ValueError("balance update and initialization clocks are outside executable bounds")
    sources = balance["sources"]
    routes = balance["routes"]
    if (not isinstance(sources, list) or not 0 < len(sources) <= 64
            or not isinstance(routes, list) or not 0 < len(routes) <= 65_536):
        raise ValueError("balance map requires bounded nonempty source and route arrays")

    source_ids: set[int] = set()
    binding_ids: set[int] = set()
    evidence: set[str] = set()
    compiled_sources = []
    maximum_delay = 0
    has_filter = False
    source_fields = {
        "identifier", "bodyReceptorBindingIdentifier", "referenceValue",
        "filterTimeConstantSeconds", "conductionDelayMicroseconds", "evidenceKind",
    }
    for source in sources:
        if not isinstance(source, dict) or set(source) != source_fields:
            raise ValueError("balance source fields are incomplete or ambiguous")
        identifier = source["identifier"]
        binding = source["bodyReceptorBindingIdentifier"]
        reference = source["referenceValue"]
        filtering = source["filterTimeConstantSeconds"]
        delay = source["conductionDelayMicroseconds"]
        kind = source["evidenceKind"]
        if (type(identifier) is not int or not 0 < identifier < 2**32
                or identifier in source_ids or type(binding) is not int
                or not 0 < binding < 2**32 or binding in binding_ids
                or type(reference) not in (int, float) or not math.isfinite(reference)
                or type(filtering) not in (int, float) or not math.isfinite(filtering)
                or not 0 <= filtering <= 1
                or type(delay) is not int or not 0 <= delay <= 500_000
                or delay % update != 0
                or kind not in ("kinematic", "support")):
            raise ValueError("balance source identity, calibration or executable history is invalid")
        source_ids.add(identifier)
        binding_ids.add(binding)
        evidence.add(kind)
        maximum_delay = max(maximum_delay, delay)
        has_filter = has_filter or filtering > 0
        compiled_sources.append({
            "identifier": identifier,
            "bodyReceptorBindingIdentifier": binding,
            "referenceValue": reference,
            "filterTimeConstantSeconds": filtering,
            "conductionDelayMicroseconds": delay,
        })
    minimum_initialization = maximum_delay + (update if has_filter else 0)
    history_capacity = maximum_delay // update + 1
    if initialization < minimum_initialization or history_capacity > 501:
        raise ValueError("balance initialization does not cover bounded delay and filter history")
    if "kinematic" not in evidence or (mode == "supportAware" and "support" not in evidence):
        raise ValueError("balance mode lacks its required kinematic or support evidence")

    route_fields = {
        "sourceIdentifier", "muscleIdentifier", "gain", "maximumCorrection",
    }
    route_keys: set[tuple[int, int]] = set()
    maximum_by_muscle: dict[int, float] = {}
    compiled_routes = []
    for route in routes:
        if not isinstance(route, dict) or set(route) != route_fields:
            raise ValueError("balance route fields are incomplete or ambiguous")
        source = route["sourceIdentifier"]
        muscle = route["muscleIdentifier"]
        gain = route["gain"]
        maximum_correction = route["maximumCorrection"]
        key = (source, muscle)
        if (type(source) is not int or source not in source_ids
                or type(muscle) is not int or not 0 <= muscle < muscles
                or key in route_keys or type(gain) not in (int, float)
                or not math.isfinite(gain) or gain == 0 or abs(gain) > 10
                or type(maximum_correction) not in (int, float)
                or not math.isfinite(maximum_correction)
                or not 0 < maximum_correction <= 0.5):
            raise ValueError("balance route is unbound, duplicate or outside excitation bounds")
        cumulative = maximum_by_muscle.get(muscle, 0.0) + maximum_correction
        if cumulative > 0.5:
            raise ValueError("balance routes exceed the per-muscle correction budget")
        route_keys.add(key)
        maximum_by_muscle[muscle] = cumulative
        compiled_routes.append({
            "sourceIdentifier": source,
            "muscleIdentifier": muscle,
            "gain": gain,
            "maximumCorrection": maximum_correction,
        })

    return {
        "version": 1,
        "locomotorProgramFingerprint": _baseline_fingerprint(program),
        "modelSourceFingerprint": program["modelSourceFingerprint"],
        "sensoryProfileFingerprint": program["sensoryProfileFingerprint"],
        "calibrationArtifactSHA256": hashlib.sha256(artifact).hexdigest(),
        "mode": modes[mode],
        "updatePeriodMicroseconds": update,
        "initializationDurationMicroseconds": initialization,
        "sources": compiled_sources,
        "routes": compiled_routes,
    }


def compile_program(body: dict, payload: bytes, *, tonic: float | None = None, length_gain: float,
                    velocity_gain: float, maximum: float = 0.95,
                    period_us: int = 0, gait: list[dict] | None = None,
                    prepared_state: bytes | None = None,
                    balance: dict | None = None,
                    balance_artifact: bytes | None = None) -> dict:
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
    program = dict(version=1, modelSourceFingerprint=body["model_source_fingerprint"],
                   sensoryProfileFingerprint=body["sensory_profile_fingerprint"],
                   calibrationArtifactSHA256=hashlib.sha256(prepared_state if prepared_state is not None else payload).hexdigest(),
                   epochMicroseconds=0, periodMicroseconds=period_us, channels=channels)
    if (balance is None) != (balance_artifact is None):
        raise ValueError("balance map and calibration bytes must be supplied together")
    if balance is not None:
        program["balanceFeedback"] = _balance_feedback(
            program, balance, balance_artifact, muscles
        )
        program["version"] = 2
    return program


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
    parser.add_argument("--balance-map", type=Path,
                        help="source-bound v1 whole-body feedback map with bounded transactional delay and filtering")
    parser.set_defaults(handler=run)


def run(args):
    try:
        balance_artifact = args.balance_map.read_bytes() if args.balance_map else None
        program = compile_program(json.loads(args.body_description.read_text()), args.muscle_payload.read_bytes(),
                                  tonic=args.tonic, length_gain=args.length_gain,
                                  velocity_gain=args.velocity_gain_seconds, maximum=args.maximum_excitation,
                                  period_us=args.period_microseconds,
                                  gait=json.loads(args.gait_map.read_text()) if args.gait_map else None,
                                  prepared_state=args.prepared_state.read_bytes() if args.prepared_state else None,
                                  balance=json.loads(balance_artifact) if balance_artifact else None,
                                  balance_artifact=balance_artifact)
    except (ValueError, TypeError, struct.error, json.JSONDecodeError) as error:
        from .model import ImportError as HumanImportError
        raise HumanImportError(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(program, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"program": str(args.output), "channels": len(program["channels"]),
                      "period_microseconds": program["periodMicroseconds"],
                      "balance_feedback": "balanceFeedback" in program, "promotable": False,
                      "boundary": "immutable recruitment and bounded body-feedback authoring; no standing or human-response qualification"}))
    return 0