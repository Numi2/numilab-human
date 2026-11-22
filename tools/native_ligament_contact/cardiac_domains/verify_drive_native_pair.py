"""Compare the old and updated native supported controls and account for the actual bed reaction."""

import csv, json, sys, re
from pathlib import Path
import numpy as np
from certify_heart import sha


def rows(case, filename):
    with (case / filename).open() as f:
        return list(csv.DictReader(f))


def verify(previous, c, t, out):
    data = [rows(p, "resting-coupled.csv") for p in (previous, c, t)]
    if data[0] != data[1]:
        raise ValueError("Normal-drive native trace changed")
    results = {}
    for name, case, trace in zip(
        ("control", "intervention"), (c, t), data[1:], strict=True
    ):
        terminal = json.loads((case / "terminal.receipt.json").read_text())
        execution = json.loads((case / "execution.json").read_text())
        if (
            terminal["accepted_step"] != 30000
            or execution["native_exit_code"] != 0
            or execution["watchdog"]
            or terminal["surface_audit_endpoint"] != "passed"
        ):
            raise ValueError("Incomplete physical terminal")
        clock = float(terminal["accepted_time_s"])
        dt = clock / 30000
        sections = rows(case, "resting-local-bed-sections.csv")
        keys = {int(r["contact_index"]) for r in sections}
        if len(keys) != len(sections) or len(keys) != 32:
            raise ValueError("Expected the registered supported scene")
        forces = {}
        with (case / "resting-com-support-impulses.csv").open() as f:
            for row in csv.DictReader(f):
                i = int(row["accepted_step"])
                if (
                    i * dt >= 58
                    and row["row_class"] == "external_support_contact"
                    and int(row["contact_index"]) in keys
                ):
                    forces[i] = (
                        forces.get(i, 0.0) + float(row["impulse_world_z_ns"]) / dt
                    )
        diagnostic = {
            int(r["accepted_step"]): float(r["normal_force_last_physical_step_n"])
            for r in rows(case, "resting-com-momentum-diagnostic.csv")
            if float(r["time_s"]) >= 58
        }
        if (
            not forces
            or forces.keys() != diagnostic.keys()
            or max(abs(f - diagnostic[i]) for i, f in forces.items()) > 0.015
        ):
            raise ValueError("Native bed force reduction differs")
        report = {}
        for key in (
            "pleural_pa",
            "lung_volume_ml",
            "PaO2_mmhg",
            "PaCO2_mmhg",
            "blood_error_ml",
            "body_mass_residual_kg",
            "oxygen_balance_error_stpd_ml",
            "co2_balance_error_stpd_ml",
        ):
            a = [float(r[key]) for r in trace]
            if not np.isfinite(a).all():
                raise ValueError(key)
            report[key] = [min(a), max(a)]
        s = (case / "stdout.log").read_text()
        callback = len(re.findall("^resting_present_profile step=", s, re.MULTILINE))
        results[name] = dict(
            accepted_s=clock,
            wall_s=execution["wall_seconds"],
            rtf=clock / execution["wall_seconds"],
            presented_callbacks=callback,
            presentations_per_wall_s=callback / execution["wall_seconds"],
            bed_contact_rows=len(keys),
            late_contact_samples=len(forces),
            late_mean_vertical_bed_reaction_N=float(np.mean(list(forces.values()))),
            body_weight_N=9.81 * float(trace[-1]["candidate_positive_body_mass_kg"]),
            breaths=int(trace[-1]["breaths"]),
            full_filling_ejection_cycles=int(
                trace[-1]["complete_filling_ejection_cycles"]
            ),
            ranges=report,
            receipt=sha(case / "terminal.receipt.json"),
            coupled_trace=sha(case / "resting-coupled.csv"),
        )
    results.update(
        original_compared_columns=len(data[0][0]),
        original_identical_trace_samples=len(data[0]),
        original_trace=sha(previous / "resting-coupled.csv"),
    )
    out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    verify(*(Path(p).resolve() for p in sys.argv[1:]))
