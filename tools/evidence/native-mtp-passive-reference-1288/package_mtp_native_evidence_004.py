from pathlib import Path
import json,csv,hashlib,gzip,subprocess,re
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
H=Path("/Users/n/numi-human-local-clearance-preservation-1280")
O=H/"tools/evidence/native-mtp-passive-reference-1288";O.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
junction=A/"mtp-native-foot-junction-audit-001/report.json";j=json.load(open(junction));assert j["complete"] and j["inputs_unchanged"]
summary={"scope":"Four16second native anatomical-mechanics sensitivity runs; no selected default or complete-body qualification.","device":"Apple M4 Pro Mac mini,24GB,macOS26.6; SSH execution only","runs":[],"literature":{"url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC5080733/","doi":"10.1186/s13047-016-0173-2","scope":"FirstMTP passive dorsiflexion quasi-stiffness in13healthy subjects; not a five-ray aggregate measurement","experienced_session1_mean_Nmm_per_deg":14.9,"converted_Nm_per_rad":14.9*.001*180/np.pi,"tested_aggregate_Nm_per_rad":[0,.5,1,2]},"tests":{"human_resting_run":"31passed,36subtests","numi_human_passive_joint":"1passed","builds":["numi-human-native","metalrobo_numilab_human_myosim_visual_probe"]}}
manifest={}
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());manifest[str(dst.relative_to(O))]={"retained_path":str(src),"sha256":sha(src)}
for name in ("prepare_mtp_native_sensitivity_001.py","verify_mtp_native_sensitivity_001.py","audit_mtp_native_foot_junction_001.py","compress_completed_mtp_outputs_001.py","mtp-final-visual-probe-build.log"):
 copy(A/name,O/name)
copy(junction,O/"foot-junction-report.json")
copy(Path(__file__),O/Path(__file__).name)
copy(A.parent/"native-initial-force-report-source-1187/matter/tools/inspect_resting_movie.swift",O/"inspect_resting_movie.swift")
for arm in ("zero","half","one","two"):
 D=A/f"mtp-passive-native-sensitivity-{arm}-001";N=D/"baseline/native-run";sub=O/arm
 report=json.load(open(D/"native-anatomy-verification-001/report.json"));assert report["complete"] and report["inputs_unchanged"]
 e=json.load(open(D/"baseline/execution.json"));decl=json.load(open(D/"baseline/run-declaration.json"));meta=json.load(open(N/"run-metadata.json"))
 assert e["returncode"]==0 and not e["changed_inputs"] and meta["loaded_metal_runtime"]["verified"]
 for p,d in decl["immutable_assets"].items():assert sha(p)==d,(p,d)
 for name in ("baseline/run-declaration.json","baseline/execution.json","baseline/execution-start.json","native-anatomy-verification-001/report.json","completed-output-lossless-compression.json"):
  copy(D/name,sub/Path(name).name)
 for name in ("invocation.json","run-metadata.json"):
  copy(N/name,sub/name)
 raw=(N/"resting-coupled.csv").read_bytes();dest=sub/"resting-coupled.csv.gz";dest.write_bytes(gzip.compress(raw,mtime=0));manifest[str(dest.relative_to(O))]={"retained_path":str(N/"resting-coupled.csv"),"uncompressed_sha256":sha(N/"resting-coupled.csv"),"sha256":sha(dest)}
 qrows=list(csv.DictReader(open(N/"resting-com-q-integration.csv")));compact=sub/"mtp-accepted-state.csv"
 with compact.open("w") as f:
  w=csv.writer(f);w.writerow(["accepted_step","time_s","right_q111_rad","left_q125_rad","right_limit_impulse_v110","left_limit_impulse_v124"])
  for r in qrows:
   q=r["q_accepted_f32_semicolon"].split(";");lim=r["gpu_stand_source_limit_impulses_f32_by_local_v_index_semicolon"].split(";");w.writerow([r["accepted_step"],r["time_s"],q[111],q[125],lim[110],lim[124]])
 manifest[str(compact.relative_to(O))]={"derived_from":str(N/"resting-com-q-integration.csv"),"source_sha256":sha(N/"resting-com-q-integration.csv"),"sha256":sha(compact)}
 native=[line.strip() for line in (N/"native.log").open() if line.startswith("resting_integrated_body=completed")];assert len(native)==1
 timing={k:float(re.search(r"\b"+k+r"=([0-9.eE+-]+)",native[0])[1]) for k in ("simulated_s","wall_s","real_time_factor")}
 movie=N/"native-viewer.mov"
 inspection=D/"movie-inspection.log"
 lines=[x for x in inspection.read_text().splitlines() if x.startswith("frames=")];assert len(lines)==1
 probe=dict(x.split("=",1) for x in lines[0].split());assert int(probe["frames"])==252 and int(probe["timing_markers"])==5
 copy(inspection,sub/"movie-inspection.log")
 rejection=[x for x in (N/"native.log").read_text().splitlines() if x.startswith("resting_integrated_rejection=")];assert len(rejection)==1 and "resting_integrated_rejection=pass" in rejection[0]
 (sub/"accepted-state-rejection.txt").write_text(rejection[0]+"\n")
 copy(N/"frame-skeleton.png",sub/"frame-skeleton.png")
 out={"arm":arm,"stiffness_nm_per_rad":report["stiffness_nm_per_rad"],"mtp_window":report["mtp_window"],"timing":timing,"launcher_wall_s":e["wall_seconds"],"trace":report["trace"],"foot_self_at_terminal":{str(x["stable_id"]):x["self_count"] for x in report["poses"][-1]["foot_muscles"]},"calcn_to_toes_terminal_triangle_pairs":next(x for x in j["runs"] if x["arm"]==arm)["poses"][-1]["count"],"movie":{"path":str(movie),"sha256":sha(movie),"probe":probe},"full_log":{"path":str(N/"native.log"),"sha256":sha(N/"native.log")},"root_assistance_zero":True,"inputs_unchanged":True}
 summary["runs"].append(out)
 for patch in D.glob("*.patch"):copy(patch,sub/patch.name)
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
(O/"retained-files.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(O),"runs":[{"k":r["stiffness_nm_per_rad"],"mean_deg":(np.asarray(r["mtp_window"]["angle_mean_rad"])*180/np.pi).tolist(),"junction_pairs":r["calcn_to_toes_terminal_triangle_pairs"],"rtf":r["timing"]["real_time_factor"]} for r in summary["runs"]]},indent=2))

