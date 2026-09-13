"""Compile single-participant empirical protocols; never certify runtime outcomes.

Source records are provenance assertions bound to immutable bytes. This compiler
checks their consistency, not the truth of a source's demographic measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .model import ImportError as HumanImportError
from .gap_execution import read_json
from .target_coverage import canonical_bytes, digest

SCHEMA = "numi.human.capability-protocol.v1"
COMPILED_SCHEMA = "numi.human.compiled-capability-protocol.v1"
REQUIREMENTS = {
    "balance_recovery": ("quiet_stance", "weight_transfer", "altered_support", "unexpected_perturbation"),
    "locomotion": ("start_stop", "variable_speed", "turning", "backward_sideways"),
    "terrain": ("ramps", "stairs", "obstacles", "support_friction_change"),
    "transitions": ("sit_stand", "bending", "squatting", "floor_transfer"),
    "manipulation": ("reach", "grasp_lift", "place_release", "bimanual_transfer"),
    "combined": ("carry_walk_turn", "reach_balance", "lift_step"),
    "sustained": ("repeated_activity", "prolonged_activity"),
}
MEASUREMENTS = {family: {"kinematics", "external_force"} for family in REQUIREMENTS}
MEASUREMENTS["manipulation"] = {"kinematics", "object_kinematics", "hand_force"}
MEASUREMENTS["combined"] = {"kinematics", "external_force", "object_kinematics", "hand_force"}
MEASUREMENTS["sustained"] = {"kinematics", "external_force", "muscle_response"}
KINDS = {"kinematics", "external_force", "object_kinematics", "hand_force", "muscle_response", "mechanical_work", "tissue_response"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError(message)


def fields(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), f"{label}: unexpected or missing fields")
    return value


def text(value, label):
    require(type(value) is str and bool(value.strip()), f"{label}: nonempty string required")
    return value


def number(value, label, minimum=0):
    try:
        valid = type(value) in (float, int) and math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    require(valid, f"{label}: finite number >= {minimum} required")
    return value


def integer(value, label, minimum=0):
    require(type(value) is int and minimum <= value <= (1 << 64) - 1,
            f"{label}: uint64 >= {minimum} required")
    return value


class Artifacts:
    def __init__(self, base):
        self.base = base
        self.reads = {}

    def read(self, ref):
        fields(ref, {"path", "sha256", "bytes"}, "artifact")
        name = text(ref["path"], "artifact.path")
        path = (self.base / name).resolve()
        expected = ref["sha256"]
        require(type(expected) is str and len(expected) == 64 and
                all(c in "0123456789abcdef" for c in expected), "artifact SHA256 invalid")
        size = integer(ref["bytes"], "artifact.bytes", 1)
        data = path.read_bytes()
        require(len(data) == size and hashlib.sha256(data).hexdigest() == expected,
                f"artifact bytes changed: {name}")
        require(path not in self.reads or self.reads[path] == (expected, size), "artifact changed between reads")
        self.reads[path] = expected, size
        return data

    def json(self, ref):
        self.read(ref)
        value = read_json((self.base / ref["path"]).resolve())
        self.read(ref)
        return value

    def recheck(self):
        for path, (sha, size) in self.reads.items():
            data = path.read_bytes()
            require(len(data) == size and hashlib.sha256(data).hexdigest() == sha,
                    f"artifact changed during compilation: {path}")


def compile_protocol(value, *, base: Path):
    fields(value, {"schema", "subject", "model", "trials", "cells", "freeze"}, "protocol")
    require(value["schema"] == SCHEMA, "unsupported capability protocol")
    subject = fields(value["subject"], {"namespace", "participant_id", "sex", "age_years", "source"}, "subject")
    identity = (text(subject["namespace"], "subject namespace"), text(subject["participant_id"], "participant ID"))
    require(subject["sex"] == "male", "release requires one adult male")
    number(subject["age_years"], "subject age", 18)
    artifacts = Artifacts(base)
    demographics = artifacts.json(subject["source"])
    fields(demographics, {"schema", "namespace", "participant_id", "sex", "age_years", "raw_source", "source_locator"}, "demographics")
    require(demographics["schema"] == "numi.human.subject-demographics.v1", "unsupported demographics record")
    for key in ("namespace", "participant_id", "sex", "age_years"):
        require(type(demographics[key]) is type(subject[key]) and demographics[key] == subject[key],
                f"subject disagrees with demographics: {key}")
    artifacts.read(demographics["raw_source"])
    text(demographics["source_locator"], "demographic source locator")
    model = fields(value["model"], {"human_pack", "subject_binding"}, "model")
    artifacts.read(model["human_pack"])
    binding = artifacts.json(model["subject_binding"])
    fields(binding, {"schema", "namespace", "participant_id", "age_years", "human_pack_sha256", "parameters"}, "subject binding")
    require(binding["schema"] == "numi.human.subject-model-binding.v1", "unsupported subject binding")
    require((binding["namespace"], binding["participant_id"]) == identity and
            type(binding["age_years"]) is type(subject["age_years"]) and binding["age_years"] == subject["age_years"] and
            binding["human_pack_sha256"] == model["human_pack"]["sha256"], "model is not bound to this subject/age/pack")
    artifacts.read(binding["parameters"])
    freeze = fields(value["freeze"], {"before_controller_selection", "statistical_plan", "confidence", "numerical_error_fraction"}, "freeze")
    require(freeze["before_controller_selection"] is True, "protocol must be frozen before controller selection")
    require(number(freeze["confidence"], "confidence") == .95, "confidence must be 0.95")
    require(0 < number(freeze["numerical_error_fraction"], "numerical fraction") <= .1, "numerical error budget exceeds 10 percent")
    artifacts.read(freeze["statistical_plan"])
    require(type(value["trials"]) is list and value["trials"], "trial inventory is empty")
    trials, sessions, raw_roles = {}, {}, {}
    for trial in value["trials"]:
        fields(trial, {"id", "role", "session_id", "source"}, "trial")
        tid = text(trial["id"], "trial ID")
        require(tid not in trials, "duplicate trial ID")
        role = text(trial["role"], "trial role")
        require(role in {"calibration", "validation"}, "invalid trial role")
        session = text(trial["session_id"], "session ID")
        require(session not in sessions or sessions[session] == role, "calibration/validation sessions overlap")
        sessions[session] = role
        record = artifacts.json(trial["source"])
        fields(record, {"schema", "namespace", "participant_id", "age_years", "trial_id", "session_id", "capability_id", "conditions", "measurements"}, "trial source")
        require(record["schema"] == "numi.human.measured-trial.v1", "unsupported measured trial")
        require((record["namespace"], record["participant_id"]) == identity and
                type(record["age_years"]) is type(subject["age_years"]) and record["age_years"] == subject["age_years"],
                "trial belongs to another participant or age")
        require(record["trial_id"] == tid and record["session_id"] == session, "trial/session identity mismatch")
        text(record["capability_id"], "trial capability")
        artifacts.read(record["conditions"])
        require(type(record["measurements"]) is list and record["measurements"], "trial lacks measurements")
        measurements = {}
        for measurement in record["measurements"]:
            fields(measurement, {"id", "kind", "units", "origin", "artifact", "uncertainty"}, "measurement")
            mid = text(measurement["id"], "measurement ID")
            require(mid not in measurements, "duplicate measurement ID")
            require(text(measurement["kind"], "measurement kind") in KINDS, "unknown measurement kind")
            require(text(measurement["origin"], "measurement origin") in {"measured", "derived", "simulated"}, "unknown measurement origin")
            text(measurement["units"], "measurement units")
            artifacts.read(measurement["artifact"])
            sha = measurement["artifact"]["sha256"]
            require(sha not in raw_roles or raw_roles[sha] == (role, tid), "measurement bytes reused across trials or splits")
            raw_roles[sha] = (role, tid)
            artifacts.read(measurement["uncertainty"])
            measurements[mid] = measurement
        trials[tid] = {"role": role, "record": record, "measurements": measurements}
    require({t["role"] for t in trials.values()} == {"calibration", "validation"}, "both calibration and held-out trials are required")
    expected = {f"{family}/{scenario}" for family, scenarios in REQUIREMENTS.items() for scenario in scenarios}
    require(type(value["cells"]) is list, "cells must be a list")
    seen, used = set(), set()
    for cell in value["cells"]:
        fields(cell, {"id", "family", "trial_ids", "comparisons"}, "capability cell")
        cid = text(cell["id"], "cell ID")
        require(cid in expected and cid not in seen, "unknown or duplicate capability cell")
        seen.add(cid)
        require(cell["family"] == cid.split('/')[0], "cell family mismatch")
        require(type(cell["trial_ids"]) is list and cell["trial_ids"], "invalid cell trial IDs")
        for tid in cell["trial_ids"]:
            text(tid, "cell trial ID")
        require(len(set(cell["trial_ids"])) == len(cell["trial_ids"]), "duplicate cell trial IDs")
        require(all(t in trials and trials[t]["role"] == "validation" for t in cell["trial_ids"]), "cell must reference held-out trials")
        require(all(trials[t]["record"]["capability_id"] == cid for t in cell["trial_ids"]), "trial conditions belong to another capability")
        used.update(cell["trial_ids"])
        require(type(cell["comparisons"]) is list and cell["comparisons"], "cell has no comparisons")
        kinds, pairs = {tid: set() for tid in cell["trial_ids"]}, set()
        for comparison in cell["comparisons"]:
            fields(comparison, {"trial_id", "measurement_id", "metric", "tolerance", "numerical_error_limit", "margin_source"}, "comparison")
            tid, mid = text(comparison["trial_id"], "comparison trial"), text(comparison["measurement_id"], "comparison measurement")
            require(tid in cell["trial_ids"] and mid in trials[tid]["measurements"], "comparison has no matching measurement")
            measurement = trials[tid]["measurements"][mid]
            require(measurement["origin"] == "measured", "derived or simulated values cannot satisfy empirical comparisons")
            metric = text(comparison["metric"], "comparison metric")
            require(metric in {"rmse", "maximum_absolute_error"}, "unsupported comparison metric")
            require((tid, mid, metric) not in pairs, "duplicate comparison")
            pairs.add((tid, mid, metric))
            tolerance = number(comparison["tolerance"], "tolerance")
            require(tolerance > 0, "tolerance must be positive")
            require(number(comparison["numerical_error_limit"], "numerical error limit") <= tolerance * freeze["numerical_error_fraction"], "numerical error limit exceeds frozen fraction")
            margin = artifacts.json(comparison["margin_source"])
            fields(margin, {"schema", "measurement_id", "units", "metric", "tolerance", "basis", "source"}, "margin")
            require(margin["schema"] == "numi.human.empirical-margin.v1" and margin["measurement_id"] == mid and
                    margin["units"] == measurement["units"] and margin["metric"] == metric and
                    type(margin["tolerance"]) in (int, float) and margin["tolerance"] == tolerance,
                    "margin does not bind comparison units, metric and tolerance")
            require(text(margin["basis"], "margin basis") in {"measurement_repeatability", "source_validated_margin"}, "unsupported empirical margin basis")
            artifacts.read(margin["source"])
            kinds[tid].add(measurement["kind"])
        require(all(MEASUREMENTS[cell["family"]] <= measured for measured in kinds.values()), f"{cid}: mandatory measurement coverage is absent")
    require(seen == expected, "required whole-body capability cells are missing")
    require(used == {tid for tid, trial in trials.items() if trial["role"] == "validation"}, "unassigned validation trials")
    artifacts.recheck()
    result = {"schema": COMPILED_SCHEMA, "protocol": value, "protocol_sha256": digest(value),
              "subject_identity": list(identity), "age_years": subject["age_years"],
              "counts": {"families": len(REQUIREMENTS), "cells": len(seen), "trials": len(trials)},
              "empirical_qualification": "not_assessed", "runtime_qualification": "not_assessed",
              "source_authenticity": "not_assessed", "freeze_chronology": "not_attested",
              "artifact_sha256": sorted({sha for sha, _ in artifacts.reads.values()}),
              "boundary": "Source/coverage compilation only; no native execution, statistical equivalence, material calibration or whole-Human qualification."}
    result["compiled_sha256"] = digest(result)
    return result


def command(args):
    if args.requirements:
        result = {"schema": "numi.human.capability-requirements.v1", "subject": "one adult male at one sourced age",
                  "families": REQUIREMENTS, "required_measurement_kinds": {k: sorted(v) for k, v in MEASUREMENTS.items()},
                  "legacy_420_gate": "regression_only", "qualification": "not_assessed"}
    else:
        require(args.protocol is not None, "--protocol or --requirements is required")
        result = compile_protocol(read_json(args.protocol), base=args.protocol.resolve().parent)
    data = canonical_bytes(result) + b"\n"
    if args.output:
        require(not args.output.exists() or args.output.read_bytes() == data, "output is immutable; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.output.exists():
            with args.output.open('xb') as stream:
                stream.write(data)
    else:
        print(data.decode(), end='')
    return 0


def add_arguments(parser):
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--protocol", type=Path)
    mode.add_argument("--requirements", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.set_defaults(handler=command)
