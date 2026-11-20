"""Summarize the matched full-gravity residual, limiter and tissue observations."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path.cwd()
control = root / "glide-refresh-trial"
observer = root / "glide-refresh-witness-trial"
limiter = root / "glide-refresh-contact-trial"

def records(run, prefix):
    with (run / "stderr.log").open() as stream:
        return [json.loads(line[len(prefix):]) for line in stream if line.startswith(prefix)]

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def cert(run):
    return json.loads((run/"failure.json").read_text())

accepted_hash = digest(control/"native/resting-coupled.csv")
for run in (observer, limiter):
    assert digest(run/"native/resting-coupled.csv") == accepted_hash, run
    for key in ("matter_code", "matter_microsteps", "matter_diagnostics"):
        assert cert(run)[key] == cert(control)[key], (run, key)

newton = defaultdict(dict)
for row in records(observer, "fem_newton_iterate="):
    assert row["root"] == 7 and row["microtick"] == 85
    record = newton[row["iteration"]]
    assert row["stage"] not in record
    record[row["stage"]] = row

mins = records(limiter, "fem_deformable_limit_minimum=")
table = []
for iteration, stages in sorted(newton.items()):
    base = stages["before_contact_limits"]
    after = stages["after_deformable_limit"]
    end = stages["final_line_search"]
    witnesses = [r for r in mins if r["iteration"] == iteration]
    table.append({
        "iteration": iteration,
        "candidate_fem_minimum_J": min(row[1] for row in base["object_line_search"][2:]),
        "nonlinear_residual": base["fgmres"][0]["nonlinear"][1],
        "linear_residual_diagnostic": base["fgmres"][0]["diagnostics"][0],
        "alpha_before_deformable_contact": base["object_line_search"][7][0],
        "alpha_after_deformable_contact": after["object_line_search"][7][0],
        "alpha_after_all_limits": end["alpha"][0],
        "contact_witnesses": witnesses,
    })

nodes = defaultdict(dict)
cells = defaultdict(dict)
with (observer / "energy.csv").open() as stream:
    for row in csv.reader(stream):
        tag, step, micro, stage, status, i = row[:6]
        target = nodes if tag == "n" else cells
        key = tuple(map(int,(step,micro,stage,status)))
        target[key][int(i)] = np.asarray(row[6:], dtype=float)
energy = []
for key in sorted(nodes):
    assert len(nodes[key]) == 88 and len(cells[key]) == 129
    q = np.stack([nodes[key][i] for i in range(88)])
    e = np.stack([cells[key][i] for i in range(129)])
    tets = e[:,:4].astype(int) - 270
    assert tets.min() == 0 and tets.max() == 87
    v0, v1 = q[:,9:12], q[:,15:18]
    du = q[:,12:15] - q[:,6:9]
    ext = q[:,18:21]
    inv_dm = e[:,5:17].reshape(-1,3,4)[:,:,:3]
    u = q[:,12:15]
    jacobian = np.linalg.det(np.eye(3) + np.swapaxes(u[tets[:,1:]]-u[tets[:,:1]],1,2) @ inv_dm)
    energy.append(dict(root=key[0],microtick=key[1],stage=key[2],status=key[3],
        kinetic_j_before=float(np.sum(0.5*q[:,1]*np.sum(v0*v0,axis=1))),
        kinetic_j_after=float(np.sum(0.5*q[:,1]*np.sum(v1*v1,axis=1))),
        speed_max_m_s=float(np.linalg.norm(v1,axis=1).max()),
        speed_row357_m_s=float(np.linalg.norm(v1[-1])),
        force_max_n=float(np.linalg.norm(ext,axis=1).max()),
        force_row357_n=float(np.linalg.norm(ext[-1])),
        contact_load_work_j=float(np.sum(ext*du)),
        J_range=[float(jacobian.min()),float(jacobian.max())]))

report = {
    "native_certificate": cert(control)["matter_diagnostics"][:2],
    "accepted_csv_sha256_reproduced_by_both_observers": accepted_hash,
    "full_gravity_target_time_us": cert(control)["brain_candidate"]["timestamp_us"],
    "rolled_back_time_us": cert(control)["brain_input"]["source_time_us"],
    "geometry_witness": records(observer, "fem_backtracking_trials="),
    "newton": table,
    "region5_loads": energy,
}
(root/"full-gravity-contact-witness.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"certificate":report["native_certificate"],"first_iterations":table[:4],
    "last_iteration":table[-1],"last_loads":energy[-3:]},indent=2))
