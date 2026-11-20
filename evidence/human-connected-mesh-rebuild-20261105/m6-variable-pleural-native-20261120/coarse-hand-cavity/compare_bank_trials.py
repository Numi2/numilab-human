"""Compare exact native registration and accepted physical motion."""
from argparse import ArgumentParser
import json
import math
from pathlib import Path

import numpy as np

from analyze_bank_trial import summarize
from build_hand_trial import digest
from fit_hand_cavity import PARENT, determinant, quality, table


def witnesses(trial):
    rows = table(trial / "skin-bone-witness.csv")
    return {tuple(int(r[k]) for k in ("layer","row","body","bone")):r
            for r in rows if r["inside_rays"]=="3" and r["ambiguous"]=="0"}


def emitted(trial):
    nodes = table(trial / "initial-skin-fem.csv")
    nodes.sort(key=lambda r:int(r["global_node"]))
    ids = np.array([int(r["global_node"]) for r in nodes])
    assert np.array_equal(ids,np.arange(ids[0],ids[0]+len(ids)))
    return (np.array([[float(r[f"{axis}_m"]) for axis in "xyz"] for r in nodes]),
            np.array([float(r["mass_kg"]) for r in nodes]))


def native_motion(trial):
    trace = table(trial/"native/resting-coupled.csv")
    if not trace:
        raise ValueError(f"Missing accepted trace in {trial}")
    keys = ("time_s","pleural_pa","normal_impulse_ns","root_assistance_n","body_mass_residual_kg")
    return {key: [float(r[key]) for r in trace] for key in keys}


def endpoint_checks(trial):
    jacobians=[]
    ledger={}
    with (trial/"stdout.log").open() as stream:
        for line in stream:
            if line.startswith("resting_FEM_tet,object=BodyParts3D whole-body skin-fat bonded shell,"):
                fields=dict(item.split("=",1) for item in line.strip().split(",")[1:])
                jacobians.append((int(fields["global"]),float(fields["J"])))
            for prefix in ("resting_FEM_terminal_audit=", "resting_fascia_fem_terminal_audit=",
                           "resting_FEM_bed_release_initialization=", "whole_body_tissue_mass_reference="):
                if line.startswith(prefix):
                    ledger[prefix[:-1]]=line.strip()
    assert len(jacobians)==28632
    assert [i for i,_ in jacobians]==list(range(3315,3315+28632))
    assert all(math.isfinite(j) for _,j in jacobians)
    total_work=0
    exhausted=0
    with (trial/"stderr.log").open() as stream:
        for line in stream:
            if not line.startswith("matter_fgmres_active_work="):
                continue
            receipt=json.loads(line.partition("=")[2])
            total_work+=receipt["used_columns_sum"]
            exhausted+=receipt["budget_exhausted_solves"]
    return {
        "terminal_skin_fat_cells":len(jacobians),
        "terminal_skin_fat_min_j":min(j for _,j in jacobians),
        "terminal_skin_fat_max_j":max(j for _,j in jacobians),
        "terminal_nonpositive_skin_fat_cells":sum(j<=0 for _,j in jacobians),
        "active_krylov_columns":total_work,
        "budget_exhausted_linear_solves":exhausted,
        **ledger,
    }


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("candidate",type=Path)
    parser.add_argument("report",type=Path)
    args=parser.parse_args()
    base=summarize(args.baseline)
    cand=summarize(args.candidate)
    prev,witness=witnesses(args.baseline),witnesses(args.candidate)
    prior_points,prior_mass=emitted(args.baseline)
    next_points,next_mass=emitted(args.candidate)
    rows=table(PARENT.with_name(PARENT.name+".tets.csv"))
    tets=np.array([[int(row[f"parent_node{k}"]) for k in range(4)] for row in rows])
    base_det=determinant(prior_points,tets)
    next_det=determinant(next_points,tets)
    changed=np.any(next_points!=prior_points,axis=1)
    cells=changed[tets].any(axis=1)
    mat=np.array([int(row["material_index"]) for row in rows])
    motion=native_motion(args.candidate)
    cand.update({
        "baseline_launch_sha256":base["launch_sha256"],
        "baseline_coverage":base["coverage"],
        "cleared_initial_point_bank_pairs":[prev[key] for key in sorted(prev.keys()-witness.keys())],
        "new_initial_point_bank_pairs":[witness[key] for key in sorted(witness.keys()-prev.keys())],
        "strongest_initial_penalty_force_N_before":max(float(r["force_N"]) for r in prev.values()),
        "strongest_initial_penalty_force_N_after":max(float(r["force_N"]) for r in witness.values()),
        "cell_source_sha256":digest(PARENT.with_name(PARENT.name+".tets.csv")),
        "emitted_moved_nodes":int(np.count_nonzero(changed)),
        "emitted_changed_cells":int(np.count_nonzero(cells)),
        "minimum_emitted_tet_determinant_m3":float(next_det.min()),
        "nonpositive_emitted_cells_before":int(np.count_nonzero(base_det<=0.)),
        "nonpositive_emitted_cells_after":int(np.count_nonzero(next_det<=0.)),
        "minimum_relative_emitted_tet_quality":float(np.min(quality(next_points,tets[cells])/quality(prior_points,tets[cells]))),
        "emitted_material_volume_change_m3":{str(k):float(np.sum((next_det-base_det)[mat==k])/6)
                                             for k in (15,16)},
        "emitted_nodal_mass_change_kg":math.fsum(next_mass)-math.fsum(prior_mass),
        "terminal_checks":endpoint_checks(args.candidate),
        "physical_trace":motion,
        "accepted_physical_steps":len(motion["time_s"]),
        "process_realtime_ratio":motion["time_s"][-1]/cand["execution"]["wall_seconds"],
    })
    with args.report.open("x") as stream:
        json.dump(cand,stream,indent=2,allow_nan=False)
        stream.write("\n")


if __name__=="__main__":
    main()
