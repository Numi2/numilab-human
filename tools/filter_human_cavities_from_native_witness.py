#!/usr/bin/env python3
"""Replay a fixed-boundary proposal minus exact, authenticated native failures."""
from pathlib import Path
import argparse,csv,importlib.util,json
import numpy as np
import build_human_conditioned_cavities as m

def retained_indices(total,witness):
 rows=witness["cavity_indices"]
 if witness.get("complete") is not True or witness.get("diagnostic_only") is not True or not rows:
  raise ValueError("incomplete native transport witness")
 if any(type(i) is not int or i<0 or i>=total for i in rows) or len(set(rows))!=len(rows):
  raise ValueError("invalid native cavity witness domain")
 return [i for i in range(total) if i not in set(rows)]

def pin(p):return dict(path=str(p),sha256=m.sha(p))
def run(a):
 if m.sha(a.proposal_report)!=a.report_sha256:raise ValueError("proposal report hash mismatch")
 d=a.native_dir;execution=d/"install-motion.execution.json"
 if m.sha(execution)!=a.execution_sha256:raise ValueError("native execution hash mismatch")
 ex=json.loads(execution.read_text());plan=d/"install-motion.plan.json";err=d/"install-motion.stderr.log"
 if m.sha(plan)!=ex["plan_sha256"] or m.sha(err)!=ex["stderr_sha256"] or ex["exit_code"]==0 or ex["resource_stop_reason"] is not None:
  raise ValueError("unbound or nonphysical native failure")
 if m.sha(d/"install-motion.stdout.log.gz")!=ex["stdout_sha256"]:raise ValueError("native stdout mismatch")
 trace=d/"install-motion/resting-coupled.csv"
 if trace.exists() and any(csv.DictReader(trace.open())):raise ValueError("native state advanced before filter")
 report=json.loads(a.proposal_report.read_text())
 if m.sha(m.__file__)!=report["owners"][str(Path(m.__file__))]:raise ValueError("cavity payload owner changed")
 env=json.loads(plan.read_text())["stages"][0]["environment"]
 if env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]!=report["output"]["sha256"] or m.sha(report["output"]["path"])!=report["output"]["sha256"]:raise ValueError("native failed a different proposal")
 lines=err.read_text().splitlines()
 witnesses=[json.loads(l.split("=",1)[1]) for l in lines if l.startswith("skin_fat_cavity_transport_failures=")]
 if len(witnesses)!=1 or not any("source flows exceed the frozen transport radius" in l and "no state changed" in l for l in lines):
  raise ValueError("not the exact supported transport failure")
 for v in report["input"].values():
  if m.sha(v["path"])!=v["sha256"]:raise ValueError("source input mismatch")
 receipt=json.loads(Path(report["input"]["mass_receipt"]["path"]).read_text())
 if witnesses[0]["maximum_transport_distance_m"]!=float(receipt["transport_distance_limit_m"]):raise ValueError("transport limit changed")
 kept=retained_indices(len(report["patches"]),witnesses[0]);patches=[report["patches"][i] for i in kept]
 if not patches:raise ValueError("no surviving cavities")
 with np.load(report["input"]["volume"]["path"],allow_pickle=False) as z:
  p=z["points_reference_world_m_f64"].astype(np.float32);t=z["volume_tetrahedra"].astype(np.int64)
  labels=np.column_stack([z[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]])
 owner=Path(__file__).with_name("simplify_human_skin_fat_boundary.py")
 if m.sha(owner)!=report["owners"][str(owner)]:raise ValueError("reference deformation owner changed")
 spec=importlib.util.spec_from_file_location("replay_deformation",owner);mesh=importlib.util.module_from_spec(spec);spec.loader.exec_module(mesh)
 ref=report["input"]["reference"];u,rr=mesh.load_reference_deformation(Path(ref["path"]),ref["sha256"],p,t,labels)
 gate=mesh.ReferenceDeformationGate(p,t,u,labels[:,1],rr)
 growth=sum(max(0,len(patch["new_cells"])-len(patch["old_rows"])) for patch in patches)
 if growth:m.reserve_reference_expansion(gate,growth)
 for patch in patches:
  proposal=m.propose_reference_patch(gate,t,np.array(patch["old_rows"]),np.array(patch["new_cells"]))
  if proposal is None:raise ValueError("subset violates reference-deformation gate")
  gate.commit(*proposal)
 a.output.mkdir(parents=True,exist_ok=False);out=a.output/"fixed-boundary-volume-proposal.npz"
 np.savez_compressed(out,**m.cavity_payload(report["input"]["volume"]["sha256"],len(t),patches))
 report.update(patches=patches,passing_cavities=len(patches),output=pin(out),reference_deformation=gate.report(),
  replay=dict(original_report=pin(a.proposal_report),failed_execution=pin(execution),native_plan=pin(plan),native_stderr=pin(err),excluded_cavities=witnesses[0]["cavity_indices"],retained_original_cavity_indices=kept,owner=pin(__file__),admission_thresholds_unchanged=True),
  status="PASS_FIXED_BOUNDARY_WITNESSED_SUBSET_PROPOSAL",physical_admission=False,native_steps=0)
 (a.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 print(json.dumps(dict(status=report["status"],retained_cavities=len(patches),excluded_cavities=len(witnesses[0]["cavity_indices"]),output=report["output"])),flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser(description=__doc__)
 for n in ["proposal-report","native-dir","output"]:ap.add_argument("--"+n,type=Path,required=True)
 for n in ["report-sha256","execution-sha256"]:ap.add_argument("--"+n,required=True)
 run(ap.parse_args())
