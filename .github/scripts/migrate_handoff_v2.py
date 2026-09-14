#!/usr/bin/env python3
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "src/numilab_human/native_force_convergence.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    'STATIC_DYNAMIC_HANDOFF_SCHEMA = "numi.human.static-dynamic-handoff-audit.v1"\n',
    'STATIC_DYNAMIC_HANDOFF_SCHEMA = "numi.human.static-dynamic-handoff-audit.v2"\n'
    'HANDOFF_PASSIVE_BIAS_POLICY = "legacy_zero_activation_bias_excluded_compliant_tendon_force_retained"\n',
)
old_counts = '''_HANDOFF_COUNTS = {
    "activation": 416,
    "fiber_length": 416,
    "actuator_force": 416,
    "passive_actuator_force": 416,
    "generalized_muscle_force": 128,
    "generalized_passive_force": 128,
    "force_residual": 128,
}
'''
new_counts = '''_HANDOFF_COUNTS = {
    "activation": 416,
    "fiber_length": 416,
    "source_total_actuator_force": 416,
    "driven_actuator_force": 416,
    "excluded_passive_bias_force": 416,
    "generalized_muscle_force": 128,
    "generalized_passive_force": 128,
    "force_residual": 128,
}
'''
if text.count(old_counts) != 1:
    raise RuntimeError("unexpected Human handoff comparison registry")
text = text.replace(old_counts, new_counts)
old_threshold = '    "force_relative": 5.0e-5,\n    "residual_absolute": 1.0e-3,\n'
new_threshold = '    "force_relative": 5.0e-5,\n    "decomposition_absolute_n": 1.0e-6,\n    "residual_absolute": 1.0e-3,\n'
if text.count(old_threshold) != 1:
    raise RuntimeError("unexpected Human handoff threshold registry")
text = text.replace(old_threshold, new_threshold)
start = text.index("def _static_dynamic_handoff(")
end = text.index("\n\ndef audit(", start)
new_function = '''def _static_dynamic_handoff(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved, receipt = _load_receipt(
        path, STATIC_DYNAMIC_HANDOFF_SCHEMA, "static-dynamic handoff receipt"
    )
    qualification = receipt.get("qualification")
    coverage = receipt.get("coverage")
    thresholds = receipt.get("thresholds")
    comparisons = receipt.get("comparisons")
    decomposition = receipt.get("source_force_decomposition")
    gate = receipt.get("gate")
    if not all(isinstance(value, dict) for value in (
        qualification, coverage, thresholds, comparisons, decomposition, gate
    )):
        raise ImportError(
            "static-dynamic handoff receipt is missing coverage, thresholds, comparisons, decomposition, gate, or qualification"
        )
    threshold_evidence = all(
        _finite_nonnegative(thresholds.get(name))
        and thresholds[name] <= ceiling
        for name, ceiling in _HANDOFF_THRESHOLD_CEILINGS.items()
    )
    comparison_evidence = (
        set(comparisons) == set(_HANDOFF_COUNTS)
        and all(
            _handoff_comparison(name, comparisons.get(name), count)
            for name, count in _HANDOFF_COUNTS.items()
        )
    )
    decomposition_evidence = _handoff_comparison(
        "source_force_decomposition", decomposition, 416
    )
    maximum_equilibrium_residual = receipt.get("maximum_damped_equilibrium_residual")
    coverage_evidence = (
        coverage.get("muscles") == 416
        and coverage.get("generalized_coordinates") == 128
        and coverage.get("static_muscle_state") is True
        and coverage.get("static_generalized_forces") is True
        and coverage.get("static_zero_activation_force_diagnostic") is True
        and coverage.get("passive_bias_policy") == HANDOFF_PASSIVE_BIAS_POLICY
        and coverage.get("dynamic_pre_step_state") is True
        and isinstance(coverage.get("dynamic_state_owner"), str)
        and bool(coverage["dynamic_state_owner"])
        and isinstance(coverage.get("dynamic_force_owner"), str)
        and bool(coverage["dynamic_force_owner"])
    )
    qualification_evidence = (
        qualification.get("pre_step_snapshot_present") is True
        and qualification.get("activation_and_fiber_state_parity") is True
        and qualification.get("per_muscle_force_parity") is True
        and qualification.get("source_force_decomposition_closed") is True
        and qualification.get("generalized_force_parity") is True
        and qualification.get("fiber_tendon_equilibrium_closed") is True
    )
    evidence = (
        receipt.get("status") == "passed"
        and coverage_evidence
        and threshold_evidence
        and comparison_evidence
        and decomposition_evidence
        and _finite_nonnegative(maximum_equilibrium_residual)
        and maximum_equilibrium_residual
            <= thresholds["maximum_damped_equilibrium_residual"]
        and qualification_evidence
        and gate.get("reasons") == []
    )
    claimed = qualification.get("static_dynamic_handoff_parity") is True
    if claimed and not evidence:
        raise ImportError(
            "static-dynamic handoff receipt claims parity without complete supporting evidence"
        )
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "complete": claimed and evidence,
        "passive_bias_policy": coverage.get("passive_bias_policy"),
        "pre_step_snapshot_present": qualification.get("pre_step_snapshot_present") is True,
        "activation_and_fiber_state_parity": qualification.get("activation_and_fiber_state_parity") is True,
        "per_muscle_force_parity": qualification.get("per_muscle_force_parity") is True,
        "source_force_decomposition_closed": qualification.get("source_force_decomposition_closed") is True,
        "generalized_force_parity": qualification.get("generalized_force_parity") is True,
        "fiber_tendon_equilibrium_closed": qualification.get("fiber_tendon_equilibrium_closed") is True,
        "maximum_damped_equilibrium_residual": maximum_equilibrium_residual,
        "comparisons": comparisons,
        "source_force_decomposition": decomposition,
    }
'''
text = text[:start] + new_function + text[end:]
path.write_text(text, encoding="utf-8")
print("migrated native force convergence to Human handoff v2")
