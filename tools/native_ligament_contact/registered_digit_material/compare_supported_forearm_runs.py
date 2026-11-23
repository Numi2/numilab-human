"""Compare matched native support, physiology and actual anatomy after a skin repair."""

import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
from registered_forearms import sha


def read(path):
    return json.loads(path.read_text())


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def audited_capture(case, name, receipt):
    folder = case / name
    f, b = read(folder / "accepted-envelope.json"), read(folder / "physical-banks.json")
    if (
        f["pack_sha256"] != receipt["pack_file_sha256"]
        or b["pack_sha256"] != f["pack_sha256"]
        or b["audit_sha256"] != sha(folder / "accepted-envelope.json")
        or b["checked_banks"] != 878
        or b["native_log_sha256"] != sha(case / "stdout.log")
        or f["accepted_step"] != receipt["accepted_step"]
    ):
        raise ValueError("Unmatched native whole-body geometry")
    return f, b


def paired_geometry(control, case, name, receipt):
    old, ob = audited_capture(control, name, read(control / receipt))
    new, nb = audited_capture(case, name, read(case / receipt))
    if (
        old["accepted_step"] != new["accepted_step"]
        or old["accepted_time_s"] != new["accepted_time_s"]
    ):
        raise ValueError("Different native capture clocks")

    def bank_rows(frame):
        rows = {
            (sem, bank): pairs for sem, bank, body, pairs in frame["affected_banks"]
        }
        if len(rows) != len(frame["affected_banks"]):
            raise ValueError("Repeated anatomical identity")
        return rows

    a, b = bank_rows(old), bank_rows(new)
    changes = [
        (sem, bank, a.get((sem, bank), 0), b.get((sem, bank), 0))
        for sem, bank in sorted(a.keys() | b.keys(), key=str)
        if a.get((sem, bank), 0) != b.get((sem, bank), 0)
    ]
    return dict(
        time_s=new["accepted_time_s"],
        checked_banks=nb["checked_banks"],
        source=[old["pack_sha256"], new["pack_sha256"]],
        external=[ob["physical_nonocular_pairs"], nb["physical_nonocular_pairs"]],
        skin_self=[old["skin_self_intersections"], new["skin_self_intersections"]],
        improved=[x for x in changes if x[2] > x[3]],
        worsened=[x for x in changes if x[2] < x[3]],
    )


def compare(control, case, baseline, ancestry, diagnostic):
    reference = read(baseline)
    if baseline.resolve() == (control / "envelope/accepted-envelope.json").resolve():
        f, b = audited_capture(
            control, "envelope", read(control / "terminal.receipt.json")
        )
        ref_pack = f["pack_sha256"]
        ref_external = b["physical_nonocular_pairs"]
        ref_self = f["skin_self_intersections"]
    else:
        ref_pack = reference["edited_pack_sha256"]
        ref_external = reference["derived_skin_clearance"]["edited_nonocular_pairs"]
        ref_self = reference["derived_skin_clearance"]["unchanged_skin_self_pairs"]
    a = read(control / "terminal.receipt.json")
    b = read(case / "terminal.receipt.json")
    env = read(case / "launch.json")["environment"]
    run = read(case / "execution.json")
    if (
        a["accepted_step"] != 8000
        or b["accepted_step"] != a["accepted_step"]
        or b["accepted_time_s"] != a["accepted_time_s"]
        or env.get("NM_VASCULAR_STEP_EXTERNAL_PRESSURE") != "1"
        or run["exit_code"] != 0
        or ref_pack != a["pack_file_sha256"]
    ):
        raise ValueError("Different run clocks or pleural configuration")
    frame = read(case / "envelope/accepted-envelope.json")
    banks = read(case / "envelope/physical-banks.json")
    if (
        frame["pack_sha256"] != b["pack_file_sha256"]
        or banks["audit_sha256"] != sha(case / "envelope/accepted-envelope.json")
        or banks["pack_sha256"] != frame["pack_sha256"]
        or banks["checked_banks"] != 878
    ):
        raise ValueError("Incomplete geometry audit")
    trace0, trace1 = [table(path / "resting-coupled.csv") for path in (control, case)]
    if (
        len(trace1) != len(trace0)
        or len(trace1) != 250
        or any(x["time_s"] != y["time_s"] for x, y in zip(trace0, trace1))
    ):
        raise ValueError("Mismatched sample clocks")
    keys = (
        "lung_volume_ml",
        "airflow_ml_s",
        "pleural_pa",
        "lv_ml",
        "rv_ml",
        "right_atrium_ml",
        "aortic_ejected_ml",
        "pulmonary_ejected_ml",
        "PaO2_mmhg",
        "PaCO2_mmhg",
        "blood_ml",
        "blood_error_ml",
        "breaths",
        "complete_filling_ejection_cycles",
    )
    trace = {
        k: float(max(abs(float(x[k]) - float(y[k])) for x, y in zip(trace0, trace1)))
        for k in keys
    }
    rr = table(case / "resting-com-momentum-diagnostic.csv")
    timed = [r for r in rr if float(r["time_s"]) >= 15]
    F = lambda k: np.array([float(r[k]) for r in timed])
    force = float(F("normal_force_last_physical_step_n").mean())
    grav = float(F("total_body_mass_kg")[-1] * 9.81)
    speed = float(
        np.linalg.norm(
            np.stack([F(f"com_v{k}_m_s") for k in "xyz"], axis=1), axis=1
        ).max()
    )
    lines = (case / "stdout.log").read_text()
    profile = {}
    for name in ("resting_integrated_body=completed", "resting_integrated_throughput"):
        (line,) = re.findall(r"^" + name + r" .*$", lines, re.M)
        info = dict(re.findall(r"(\w+)=([^ ]+)", line))
        profile[name.split("=")[0]] = dict(
            seconds=float(info["simulated_s"]),
            wall_s=float(info["wall_s"]),
            factor=float(info["real_time_factor"]),
        )
    if (
        "root_assistance=false" not in lines
        or "resting_bed_contact_owner=condensed_skin_fat" not in lines
    ):
        raise ValueError("Wrong support owner")
    bb = read(ancestry)
    src = Path(
        read(case / "launch.json")["argv"][
            read(case / "launch.json")["argv"].index("--skin-payload") + 1
        ]
    )
    if bb["replacement"] != sha(src):
        raise ValueError("Different registered skin")
    m = read(diagnostic)
    if m["old_pack"] != a["pack_file_sha256"] or m["new_pack"] != b["pack_file_sha256"]:
        raise ValueError("Motion attribution belongs to another run")
    return dict(
        skin_sha256=sha(src),
        pack_sha256=b["pack_file_sha256"],
        steps=b["accepted_step"],
        seconds=b["accepted_time_s"],
        device="Apple M6",
        native_profile=profile,
        cold_wall_s=run["cold_wall_s"],
        cold_factor=b["accepted_time_s"] / run["cold_wall_s"],
        support_last_second=dict(
            sample_count=len(timed),
            mean_sampled_last_step_normal_N=force,
            weight_N=grav,
            maximum_COM_speed_m_s=speed,
            COM_height_range_m=float(np.ptp(F("com_z_m"))),
        ),
        sources=dict(
            contact_owner="condensed skin/fat regional quadrature",
            variable_pleural_pressure=True,
            external_root_assistance=False,
        ),
        accepted_skeleton_byte_equal=b["accepted_body_state_sha256"]
        == a["accepted_body_state_sha256"],
        accepted_respiration_byte_equal=b["accepted_respiration_state_sha256"]
        == a["accepted_respiration_state_sha256"],
        impulses_byte_equal=sha(case / "resting-com-support-impulses.csv")
        == sha(control / "resting-com-support-impulses.csv"),
        bed_sections_byte_equal=sha(case / "resting-local-bed-sections.csv")
        == sha(control / "resting-local-bed-sections.csv"),
        peak_absolute_physiological_trace_differences=trace,
        terminal_measurements={k: float(trace1[-1][k]) for k in keys},
        clearance=dict(
            control_external=ref_external,
            edited_external=banks["physical_nonocular_pairs"],
            control_skin_self=ref_self,
            edited_skin_self=frame["skin_self_intersections"],
            edited_skin_bone=banks["bone_pairs"],
            semantic=frame["intersection_pairs_by_semantic"],
        ),
        measured_bank_changes={
            key: paired_geometry(control, case, key, receipt)
            for key, receipt in [
                ("envelope", "terminal.receipt.json"),
                ("envelope-3999", "step-3999.receipt.json"),
            ]
            if (control / key / "accepted-envelope.json").is_file()
            and (case / key / "accepted-envelope.json").is_file()
        },
        edited_other_captures={
            p.parent.name: dict(
                time_s=read(p)["accepted_time_s"],
                nonocular=read(p.parent / "physical-banks.json")[
                    "physical_nonocular_pairs"
                ],
                skin_self=read(p)["skin_self_intersections"],
            )
            for p in case.glob("envelope-*/accepted-envelope.json")
        },
    )


if __name__ == "__main__":
    print(
        json.dumps(compare(*(Path(k) for k in sys.argv[1:])), indent=2, allow_nan=False)
    )
