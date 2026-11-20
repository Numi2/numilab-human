"""Summarize the fixed-bed seed and the exact contact cache across matched roots."""
import csv
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P = ROOT / "operator-resident-030"
sys.path.insert(0, str(P))

from compare import sha256, summarize


def record(directory, prefix):
    with (directory / "stdout.log").open() as stream:
        rows = [row.strip() for row in stream if row.startswith(prefix)]
    assert len(rows) == 1, (prefix, len(rows))
    return rows[0]


def number(row, key):
    return float(re.search(r"\b" + key + r"=([^ ]+)", row)[1])


def json_lines(directory, prefix):
    with (directory / "stderr.log").open() as stream:
        return [json.loads(row.partition("=")[2]) for row in stream
                if row.startswith(prefix)]


def physics(directory):
    out = {
        "launch_sha256": sha256(directory / "launch.json"),
        "status": summarize(directory, require_domain_gate=False),
    }
    roots = json_lines(directory, "fem_newton_terminal_certificate=")
    assert out["status"]["exit"] == 0
    assert [row["root"] for row in roots] == list(range(out["status"]["roots_certified"]))
    assert all([status["code"] for status in row["status"]] == [0] for row in roots)
    with (directory / "native/resting-coupled.csv").open() as stream:
        pressure = [float(row["pleural_pa"]) for row in csv.DictReader(stream)]
    assert len(pressure) == len(roots)
    balance = record(directory, "resting_fem_skin_bed_accepted_root_reaction ")
    ligaments = record(directory, "resting_fascia_fem_terminal_audit=")
    mass = record(directory, "resting_tissue_mass_partition=")
    assert ligaments.startswith("resting_fascia_fem_terminal_audit=passed")
    assert number(ligaments, "registered_ligament_force_rows") == 34
    assert abs(number(mass, "executed_rigid_plus_FEM_kg") - number(mass, "initial_kg")) < 2.e-6
    out.update(
        native_success_roots=len(roots),
        pleural_pressure_range_pa=[min(pressure), max(pressure)],
        terminal_bed_force_n=number(balance, "mean_normal_force_N"),
        terminal_bed_contacts=int(number(balance, "valid_contact_sample_count")),
        terminal_ligaments=ligaments,
        total_mass_accounting=mass,
        startup_solid_audit=record(directory, "initial_skin_bone_solid_audit "),
        terminal_node_state=record(directory, "resting_FEM_terminal_audit="),
        startup_fem_bed=record(directory, "resting_fem_skin_bed_compiled="),
        release_force_balance=record(directory, "resting_initialization=dynamic_release "),
    )
    return out


def main():
    trial = HERE / "trial"
    cache = HERE / "contact-cache-128"
    control = P / "parallel-operators-128"
    seed = physics(trial)
    fast = physics(cache)
    normal = physics(control)
    assert seed["native_success_roots"] == 4
    assert fast["native_success_roots"] == normal["native_success_roots"] == 128
    output_rows = {}
    for name in ("native/resting-coupled.csv", "initial-skin-fem.csv", "skin-bone-witness.csv"):
        output_rows[name] = sha256(cache / name)
        assert output_rows[name] == sha256(control / name), name
    for prefix in ("matter_solver_domain_norms=", "matter_fgmres_active_work=",
                   "fem_newton_terminal_certificate="):
        assert json_lines(cache, prefix) == json_lines(control, prefix), prefix
    moving_nodes = []
    prediction = json.loads((HERE / "prediction.json").read_text())
    with (trial / "stdout.log").open() as stream:
        nodes = {}
        prefix = "resting_FEM_node,object=BodyParts3D whole-body skin-fat bonded shell,"
        for row in stream:
            if row.startswith(prefix):
                fields = dict(field.split("=", 1) for field in row[len(prefix):].strip().split(","))
                nodes[int(fields["local"])] = fields
    for initial in prediction["support"]:
        current = nodes[initial["outer_row"]]
        moving_nodes.append({**initial, "emitted_initial_gap_m": float(current["initial_z"]),
                             "emitted_terminal_gap_m": float(current["accepted_z"]),
                             "accepted_vz_m_s": float(current["vz"]),
                             "terminal_normal_force_n": float(current["contact_force_z_N"])})
    t = normal["status"]["body_wall_seconds"]
    u = fast["status"]["body_wall_seconds"]
    result = {
        "fixed_pose_seed": seed,
        "fixed_pose_seed_support_nodes": moving_nodes,
        "fixed_pose_seed_candidacy": seed["terminal_bed_force_n"] > 10.0,
        "released_control": normal,
        "released_exact_cache": fast,
        "cache_speedup_body": t / u,
        "cache_body_wall_reduction_fraction": 1.0 - u / t,
        "cache_wall_realtime_factor": fast["status"]["simulated_seconds"] / fast["status"]["wall_seconds"],
        "identical_sha256_shared_artifacts": output_rows,
        "identical_certificate_work_and_domain_residual_series": True,
        "shell_endpoint_audits": {
            name: json.loads((HERE / name).read_text())
            for name in ("endpoint-shell-audit.json", "cache-128-shell-audit.json")
        },
    }
    (HERE / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key.startswith("cache_") or key == "fixed_pose_seed_candidacy"}))


if __name__ == "__main__":
    main()
