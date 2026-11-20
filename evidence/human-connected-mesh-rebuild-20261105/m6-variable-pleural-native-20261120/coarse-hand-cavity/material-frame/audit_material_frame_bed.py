"""Compare physical bed-node motion with the detailed skin audit."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

from build_hand_trial import digest
from compare_bank_trials import endpoint_checks, witnesses

KEYS = ("body_mass_residual_kg","normal_impulse_ns","root_assistance_n","root_assistance_nm","pleural_pa")
PREPARATION = ("resting_initialization=dynamic_release","resting_initial_FEM_outer_alignment=",
               "resting_fem_skin_bed_compiled=","resting_fem_skin_bed_terminal_history",
               "resting_fem_skin_bed_accepted_root_reaction","resting_integrated_body=completed")


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior",type=Path)
    parser.add_argument("reference_frame",type=Path)
    parser.add_argument("tilted",type=Path)
    parser.add_argument("orientation",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    prior, reference, trial=args.prior,args.reference_frame,args.tilted
    orientation=json.loads(args.orientation.read_text())
    assert orientation["launch_sha256"] == digest(reference/"launch.json")
    assert digest(prior/"initial-skin-fem.csv")==digest(reference/"initial-skin-fem.csv")
    assert digest(prior/"skin-bone-witness.csv")==digest(reference/"skin-bone-witness.csv")
    assert digest(prior/"native/resting-human.mrvpack")==digest(reference/"native/resting-human.mrvpack")
    path="native/resting-coupled.csv"
    first = rows(reference/path)
    assert rows(prior/path)[:len(first)] == first
    solid_a,solid_b=witnesses(reference),witnesses(trial)
    assert solid_a.keys()==solid_b.keys()
    pair_totals=defaultdict(int)
    for _,_,body,bone in solid_b:
        pair_totals[f"{body}/{bone}"]+=1
    trace=rows(trial/path)
    surface=rows(trial/"native/resting-surface-audit.csv")
    assert int(surface[0]["step"])==0 and int(surface[-1]["step"])==len(trace)
    owners = set(orientation["contact_outer_rows"])
    selected = {int(row["global_node"]):int(row["row"]) for row in
                rows(trial/"initial-skin-fem.csv") if row["layer"]=="0" and int(row["row"]) in owners}
    bed_nodes, solve = {}, []
    with (trial/"stdout.log").open() as f:
        for line in f:
            if line.startswith(PREPARATION):
                solve.append(line.strip())
            if line.startswith("resting_FEM_node,object=BodyParts3D whole-body skin-fat bonded shell,"):
                d=dict(item.split("=",1) for item in line.strip().split(",")[1:])
                i=int(d["global"])
                if i in selected:
                    bed_nodes[selected[i]]={k:float(d[k]) for k in
                        ("initial_z","accepted_z","vz","contact_force_z_N","free_internal_force_z_N")}
    assert bed_nodes.keys()==owners
    result = {
        "same_initial_emitted_nodal_geometry_and_mass": True,
        "same_initial_closed_solid_contact_witness_bytes": True,
        "same_initial_viewer_pack_bytes": True,
        "same_first_coupled_trace_rows": len(first),
        "tilted_trial_launch_sha256": digest(trial/"launch.json"),
        "same_tilted_contact_witness_keys":True,
        "point_penetrations_by_bank":dict(sorted(pair_totals.items())),
        "tilted_accepted_steps":len(trace),
        "tilted_terminal_seconds":float(trace[-1]["time_s"]),
        "tilted_trace_ranges":{k:[min(float(r[k]) for r in trace),
                                    max(float(r[k]) for r in trace)] for k in KEYS},
        "terminal_physical_bed_nodes":bed_nodes,
        "detailed_skin_surface_samples":len(surface),
        "detailed_skin_min_bed_gap_m": [float(r["min_skin_bed_gap_m"]) for r in (surface[0], surface[-1])],
        "terminal_checks":endpoint_checks(trial),
        "tilted_receipts":solve,
        "native_equilibrium_qualified":False,
        "sustained_crossing_check_qualified":False,
    }
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({k:result[k] for k in ("tilted_terminal_seconds","terminal_physical_bed_nodes",
                                           "detailed_skin_min_bed_gap_m")}))


if __name__=="__main__":
    main()
