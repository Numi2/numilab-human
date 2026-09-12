"""Author the pinned CVSim21 source variant into native Matter; never step physics.

The native variant uses an exact continuous fixed-rate clock and complete,
stateless constitutive laws. Original C execution remains a separate reference.
No anatomy registration, transport parameters, or mechanical mass is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from .cvsim_parameters import parse_cvsim_parameters
from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/physionet/cvsim21"
INITIAL = ROOT / "tools/cvsim21_reference/evidence/20260912/original-initial.json"
CONFIG = ROOT / "config/cvsim21-source.v1.json"
SOURCE_LOCK_SHA256 = "b118d58230a3f2766108d05501a3ac9b910124ef823567a7e5fe23f6df0b4f67"
INITIAL_SHA256 = "fad558482f8bfeb94d531d4538e29d7a2c8a59ea63f32496467b087bcb343ff7"
PARAMETER_SHA256 = "c6a791e837e1542ebbb910dae1a2ac10bce5f7071ed1a9836b861b8d62267c03"
PA_PER_MMHG = 133.3224  # Declared rounded NIST HB44-2018 Appendix C convention.
ML_TO_M3 = 1e-6
LABELS = (
    "ascending_aorta", "brachiocephalic_arteries", "upper_body_arteries", "upper_body_veins",
    "superior_vena_cava", "descending_thoracic_aorta", "abdominal_aorta", "renal_arteries",
    "renal_veins", "splanchnic_arteries", "splanchnic_veins", "lower_body_arteries",
    "lower_body_veins", "abdominal_veins", "inferior_vena_cava", "right_atrium",
    "right_ventricle", "pulmonary_arteries", "pulmonary_veins", "left_atrium", "left_ventricle",
)
COMPLIANCE = (97, 98, 100, 36, 42, 99, 101, 102, 37, 103, 38, 104, 39, 40, 41, 43, 45, 47, 48, 49, 51)
ZPFV = (136, 137, 139, 77, 83, 138, 140, 141, 78, 142, 79, 143, 80, 81, 82, 84, 85, 86, 87, 88, 89)
FROM = (20, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 6, 9, 10, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19)
TO = (0, 1, 2, 3, 4, 15, 5, 6, 7, 8, 13, 9, 10, 13, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
RESISTANCE = (69, 105, 106, 53, 54, 63, 107, 108, 109, 55, 56, 110, 57, 58, 111, 59, 60, 61, 62, 64, 65, 66, 67, 68)
THORACIC = {0, 1, 4, 5, 14, 15, 16, 17, 18, 19, 20}
NONLINEAR = {10: 71, 12: 72, 13: 73}
CARDIAC = {15: (44, 119, None), 16: (46, 120, 118), 19: (50, 119, None), 20: (52, 120, 118)}
UNILATERAL = {0, 16, 19, 20, 23}
VARIANTS = {"upstream_equation", "heldt_table_aligned"}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise HumanImportError("CVSim21 authoring: " + message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise HumanImportError("CVSim21 authoring requires finite JSON") from error


def read_json(path: Path) -> dict:
    def unique(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=unique)
    except (ValueError, OSError) as error:
        raise HumanImportError(f"CVSim21 authoring: invalid JSON {path.name}") from error
    require(isinstance(result, dict), "expected JSON object")
    canonical(result)
    return result


def _verified(path: Path, sha256: str) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"missing or redirected pinned file: {path.name}")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha256, f"pinned SHA256 mismatch: {path.name}")
    return data


def _settings(config: dict) -> dict:
    require(set(config) == {"schema", "id", "source_manifest_sha256", "initial_sha256", "volume_coordinates", "source_mode", "numerical_settings"},
            "unsupported configuration fields or physiological overrides")
    require(config["schema"] == "HumanPack.cvsim21-source-config.v1" and config["id"] == "cvsim21_supine_continuous_fixed_rate", "unsupported source configuration")
    require(config["source_manifest_sha256"] == SOURCE_LOCK_SHA256 and config["initial_sha256"] == INITIAL_SHA256, "stale source or initialization identity")
    require(config["volume_coordinates"] in VARIANTS, "unsupported volume coordinates")
    require(config["source_mode"] == {"ABReflexOn": False, "CPReflexOn": False, "tiltTestOn": False, "clock": "continuous_fixed_rate_rational"}, "unsupported reflex, tilt or clock mode")
    # JSON booleans are deliberate mode selectors, not numeric C flag overrides.
    require(all(type(config["source_mode"][k]) is bool for k in ("ABReflexOn", "CPReflexOn", "tiltTestOn")), "mode flags must be booleans")
    units = {"volume_scale_m3": "m3", "flow_scale_m3_per_s": "m3/s", "pressure_scale_pa": "Pa", "residual_tolerance": "1"}
    settings = config["numerical_settings"]
    require(isinstance(settings, dict) and settings.keys() == units.keys(), "numerical settings incomplete")
    result = {}
    for key, unit in units.items():
        item = settings[key]
        require(isinstance(item, dict) and set(item) == {"value", "unit", "role"} and item["unit"] == unit and item["role"] == "numerical_residual_control_not_source_physiological_parameter", "numerical setting units or provenance mismatch")
        require(type(item["value"]) in (int, float) and math.isfinite(item["value"]) and item["value"] > 0, "numerical settings must be positive finite numbers")
        result[key] = float(item["value"])
    require(result["residual_tolerance"] < 1, "residual tolerance must be below one")
    return result


def compile_source(*, directory: Path = SOURCE, initial_path: Path = INITIAL, config: dict | None = None) -> tuple[dict, dict]:
    """Verify source and compiled-C initialization, then lower algebraic parameters."""
    try:
        return _compile_source(directory, initial_path, read_json(CONFIG) if config is None else config)
    except HumanImportError:
        raise
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
        raise HumanImportError(f"CVSim21 authoring: invalid source/configuration: {error}") from error


def _compile_source(directory: Path, initial_path: Path, config: dict) -> tuple[dict, dict]:
    canonical(config)
    numerical = _settings(config)
    _verified(directory / "source-lock.json", SOURCE_LOCK_SHA256)
    lock = read_json(directory / "source-lock.json")
    require(lock["schema"] == "NumiHuman.CVSim21-source-lock.v1" and lock["version"] == "1.0.0" and lock["license_designation"] == "ODC-By-1.0", "source identity or license differs")
    require(len(lock["files"]) == 17, "source file coverage differs")
    for name, row in lock["files"].items():
        require(Path(name).parts[0] in {"main", "sim"} and len(Path(name).parts) == 2 and ".." not in Path(name).parts, "unsafe source path")
        raw = _verified(directory / "21-comp-backend" / name, row["sha256"])
        require(len(raw) == row["bytes"], "source byte count differs")
    parsed = parse_cvsim_parameters(directory / "21-comp-backend/main/initial.c", expected_sha256=PARAMETER_SHA256)
    theta = parsed["parameter_vector"]
    _verified(initial_path, INITIAL_SHA256)
    initial = read_json(initial_path)
    require(initial["schema"] == "NumiHuman.CVSim21-original-initial.v1" and initial["source_parameters_unchanged"] is True, "initial state lacks original C owner")
    require(initial["parameter_vector"] == theta, "independent parser differs from original compiled C parameters")
    p, v, q = initial["pressure_mmHg"], initial["volume_mL"], initial["flow_mL_per_s"]
    require(len(p) == len(v) == 21 and len(q) == 24 and all(type(x) in (int, float) and math.isfinite(x) for x in p + v + q), "invalid initial state arity or numbers")
    require(math.isclose(math.fsum(v), theta[70], rel_tol=0, abs_tol=1e-8), "original initial blood budget differs")
    require(math.fsum(theta[i] for i in ZPFV) == theta[75], "zero-pressure filling budget differs")
    period = Fraction(60) / Fraction(str(theta[90]))
    require(period.numerator < 2**64 and period.denominator < 2**64, "cardiac period exceeds native exact rational representation")
    duration = float(period)
    translations = [0.0] * 21
    if config["volume_coordinates"] == "heldt_table_aligned":
        translations[2] = theta[138] - theta[139]
        translations[5] = -translations[2]
    compartments, provenance = [], []
    for i, label in enumerate(LABELS):
        baseline = theta[ZPFV[i]] + translations[i]
        external = theta[31] if i in THORACIC else 0.0
        compliance = theta[COMPLIANCE[i]]
        row = {"id": f"compartment_{i:02}_{label}", "stable_identifier": i + 1,
               "anatomical_region_id": f"source_aggregate:CVSim21:{label}",
               "physical_volume_owner_id": f"source_blood:CVSim21:v{i}", "storage_kind": "absolute_volume",
               "pressure_law": "linear_compliance", "reference_volume_m3": baseline * ML_TO_M3,
               "reference_pressure_pa": 0.0, "external_pressure_pa": external * PA_PER_MMHG,
               "compliance_m3_per_pa": compliance * ML_TO_M3 / PA_PER_MMHG,
               "initial_volume_m3": (v[i] + translations[i]) * ML_TO_M3,
               "volume_scale_m3": numerical["volume_scale_m3"], "volume_residual_tolerance": numerical["residual_tolerance"],
               "initial_species_mol": [], "elastance_min_pa_per_m3": 0.0, "elastance_max_pa_per_m3": 0.0,
               "period_seconds": 0.0, "activation_start": 0.0, "activation_end": 0.0, "source_pi": 0.0,
               "maximum_volume_displacement_m3": 0.0, "phase_delay": 0.0,
               "period_numerator_seconds": "0", "period_denominator": "0"}
        # Algebraic initialization check only; no candidate state or integration.
        displacement = compliance * (p[i] - external)
        if i in NONLINEAR:
            vmax = theta[NONLINEAR[i]]
            displacement = 2 * vmax / math.pi * math.atan(math.pi * compliance * (p[i] - external) / (2 * vmax))
            row.update(pressure_law="atan_compliance", source_pi=math.pi, maximum_volume_displacement_m3=vmax * ML_TO_M3)
        elif i in CARDIAC:
            systolic, systole, delay = CARDIAC[i]
            activation = theta[systole] / math.sqrt(duration)
            row.update(pressure_law="cosine_pulse_elastance", compliance_m3_per_pa=0.0,
                       elastance_min_pa_per_m3=PA_PER_MMHG / (ML_TO_M3 * compliance),
                       elastance_max_pa_per_m3=PA_PER_MMHG / (ML_TO_M3 * theta[systolic]),
                       activation_start=activation, activation_end=1.5 * activation,
                       phase_delay=0.0 if delay is None else theta[delay] / math.sqrt(duration), source_pi=math.pi,
                       period_numerator_seconds=str(period.numerator), period_denominator=str(period.denominator))
        require(math.isclose(v[i], theta[ZPFV[i]] + displacement, rel_tol=0, abs_tol=1e-8), f"original C storage initialization mismatch at v{i}")
        require(row["initial_volume_m3"] > 0 and row["reference_volume_m3"] > 0, "absolute blood volume must be positive")
        compartments.append(row)
        provenance.append({"source_index": i, "label": label, "compliance_parameter": COMPLIANCE[i], "raw_zpfv_parameter": ZPFV[i],
                           "volume_translation_mL": translations[i], "anatomy_registration": None, "mechanical_mass_owner": None})
    connections = []
    for i, (a, b, resistance) in enumerate(zip(FROM, TO, RESISTANCE, strict=True)):
        difference = p[a] - (max(p[b], theta[31]) if i == 4 else p[b])
        expected = (max(difference, 0.0) if i in UNILATERAL or i == 4 else difference) / theta[resistance]
        require(math.isclose(q[i], expected, rel_tol=0, abs_tol=1e-8), f"original C flow initialization mismatch at q{i}")
        connections.append({"id": f"flow_{i:02}", "stable_identifier": 22 + i, "from": a + 1, "to": b + 1,
                            "flow_law": "starling_resistance" if i == 4 else "one_way_resistance" if i in UNILATERAL else "resistance_inertance",
                            "resistance_pa_s_per_m3": theta[resistance] * PA_PER_MMHG / ML_TO_M3,
                            "inertance_pa_s2_per_m3": 0.0, "orifice_coefficient_m3_per_s_sqrt_pa": 0.0,
                            "downstream_pressure_floor_pa": theta[31] * PA_PER_MMHG if i == 4 else 0.0,
                            "initial_flow_m3_per_s": q[i] * ML_TO_M3, "flow_scale_m3_per_s": numerical["flow_scale_m3_per_s"],
                            "pressure_scale_pa": numerical["pressure_scale_pa"], "flow_residual_tolerance": numerical["residual_tolerance"]})
    native = {"schema": "HumanPack.physiology-native.v3", "model_id": config["id"] + "_" + config["volume_coordinates"],
              "qualification": "source_model_variant", "law": "closed_rational_elastance_regional_volume_v3",
              "authored_graph_sha256": hashlib.sha256(canonical(config)).hexdigest(), "source_graph_sha256": SOURCE_LOCK_SHA256,
              "residual_tolerance": numerical["residual_tolerance"], "species": [], "compartments": compartments,
              "connections": connections, "tissue_reservoirs": [], "exchanges": []}
    manifest = {"schema": "HumanPack.cvsim21-source-lowering.v1", "source_manifest_sha256": SOURCE_LOCK_SHA256,
                "initial_sha256": INITIAL_SHA256, "authored_config_sha256": native["authored_graph_sha256"],
                "native_content_sha256": hashlib.sha256(canonical(native) + b"\n").hexdigest(),
                "parameters": parsed, "compartment_bindings": provenance, "flow_resistance_parameters": list(RESISTANCE),
                "volume_coordinates": config["volume_coordinates"], "total_blood_volume_m3": theta[70] * ML_TO_M3,
                "total_zero_pressure_filling_volume_m3": theta[75] * ML_TO_M3,
                "unit_conversion": {"Pa_per_mmHg": PA_PER_MMHG, "m3_per_mL": ML_TO_M3,
                                    "pressure_convention": "rounded NIST HB44-2018 Appendix C; not an exact source constant",
                                    "reference_url": "https://nvlpubs.nist.gov/nistpubs/hb/2018/NIST.HB.44-2018.pdf"},
                "source_mode": config["source_mode"], "numerical_settings": config["numerical_settings"],
                "explicit_variants": ["continuous fixed-rate rational clock replaces original discrete SA-node scheduling",
                                      "signed atan leg storage remains continuous for nonpositive transmural pressure",
                                      "stateless Starling max law defines unassigned equality and reverse-flow source branches"] +
                                     (["paired arterial filling-volume coordinates aligned to source tables and Heldt thesis"] if any(translations) else []),
                "scientific_status": "source_variant_target_not_qualified_by_compilation",
                "boundary": "aggregate hydraulics only; no individual organ calibration, anatomy registration, mechanical mass partition, reflexes, tilt, species, tissue exchange or biological qualification"}
    return native, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=SOURCE)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        payload, manifest = compile_source(directory=args.source_directory, initial_path=args.initial, config=read_json(args.config))
        manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
        require(args.output.resolve() != manifest_path.resolve(), "payload and manifest require distinct paths")
        outputs = [(args.output, payload), (manifest_path, manifest)]
        for path, value in outputs:
            require(not path.is_symlink() and (not path.exists() or path.read_bytes() == canonical(value) + b"\n"), "output is immutable; select a new path")
        for path, value in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(canonical(value) + b"\n")
        print(json.dumps({"status": "compiled_source_variant", "payload": str(args.output), "manifest": str(manifest_path),
                          "sha256": manifest["native_content_sha256"], "compartments": 21, "connections": 24,
                          "scientific_status": manifest["scientific_status"]}))
    except (HumanImportError, OSError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
