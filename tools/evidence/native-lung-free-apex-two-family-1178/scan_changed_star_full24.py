
#!/usr/bin/env python3
from pathlib import Path
import copy, gzip, hashlib, importlib.util, itertools, json, sys, time
import numpy as np
E=Path("/Users/n/numi-human-resting-evidence-20261005")
ROOT=E/"native-lung-free-apex-two-family-candidate-1177/source-trial-024"
OUT=ROOT/"exact-changed-star-full24"
RUNNER_PATH=E/"native-lung-late-pose-audit-runner-1171/audit_lung_cycle_1159.py"
NHA=E/"native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-thorax.nhanatomy"
DREPORT=E/"native-lung-free-apex-composition-1159-eightops-attempt1/composition-report.json"
LREPORT=E/"native-lung-free-apex-composition-1159-eightops-attempt1/current-reciprocal-map-report-v2.json"
PROBE_ROOT=ROOT/"production-probe-24"
NHA_SHA="c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713"
CHANGED={305:[40179,40180,40181,40182,40183,40184,40197,40198],308:[19661,19663,19664,19665,19666]}
COHORTS={
  "early8":{"run":E/"final-native-scene-preflight-936/skin-927-lung-1159-viewer-018-v015-attempt1/native-run","arm":None,"steps":[0,4991,5375,5759,6111,6495,7743,10000]},
  "baseline8":{"run":Path("/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/resting-baseline"),"arm":"control","steps":[47519,49151,51903,54047,55647,152191,154143,155000]},
  "drive_half8":{"run":Path("/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/resting-drive-half"),"arm":"treatment","steps":[47519,49151,51903,54047,55647,152447,154367,155000]},
}
def sha(path):
  h=hashlib.sha256()
  with Path(path).open("rb") as f:
    for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
  return h.hexdigest()
def load(path,name):
  sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def event_obj(step,label,kind,owner_or_pair,faces,vertex_ids,cls,pts,base):
  baseinfo=base.event_points(pts)
  if kind=="self": return {"step":step,"scenario":label,"type":kind,"owner":owner_or_pair,"faces":list(faces),"class":cls,**baseinfo}
  return {"step":step,"scenario":label,"type":kind,"owners":list(owner_or_pair),"faces":list(faces),"face_vertex_ids":[list(vertex_ids[0]),list(vertex_ids[1])],"class":cls,**baseinfo}
def scan_changed(base,step,label,pose,native_d,native_l,sink):
  changed_records={}; records={}; bad={};
  for sid in base.ROWS:
    records[sid],bad[sid]=base.exact_records(pose[sid]["v"],pose[sid]["f"])
  for sid,faceids in CHANGED.items():
    changed_records[sid]=[r for r in records[sid] if int(r[3]) in faceids]
  counts={"changed_self_pairs":0,"self_unallowed":0,"changed_cross_pairs":0,"cross_unclassified":0,"cross_classified":0,"degenerate_changed_faces":0}
  class_counts={}; seen_self=set(); seen_cross=set()
  for sid in CHANGED:
    counts["degenerate_changed_faces"]+=len(set(CHANGED[sid])&set(map(int,bad[sid])))
    for x,y in base.pred()._aabb_candidate_pairs(changed_records[sid],records[sid],same_surface=False):
      if x[3]==y[3]: continue
      face_key=tuple(sorted((int(x[3]),int(y[3]))))
      key=(sid,*face_key)
      if key in seen_self: continue
      pts=base.pred().triangle_intersection_points(x[0],y[0])
      if not pts: continue
      seen_self.add(key)
      common_ids=set(x[4])&set(y[4])
      allowed=(len(common_ids) in (1,2) and all(base.pred()._allowed_shared_point(z,{x[0][x[4].index(i)] for i in common_ids}) for z in pts))
      cls="allowed_indexed_adjacency" if allowed else "unallowed_self_intersection"
      counts["changed_self_pairs"]+=1;counts["self_unallowed"]+=int(not allowed)
      sink.write(json.dumps(event_obj(step,label,"self",sid,face_key,None,cls,pts,base),sort_keys=True,separators=(",",":"))+"\n")
  for a,b in itertools.combinations(base.ROWS,2):
    if a not in CHANGED and b not in CHANGED: continue
    hits={}
    if a in CHANGED:
      for x,y in base.pred()._aabb_candidate_pairs(changed_records[a],records[b],same_surface=False): hits[(int(x[3]),int(y[3]))]=(x,y)
    if b in CHANGED:
      for x,y in base.pred()._aabb_candidate_pairs(records[a],changed_records[b],same_surface=False): hits[(int(x[3]),int(y[3]))]=(x,y)
    for face_key,(x,y) in sorted(hits.items()):
      key=(a,b,*face_key)
      if key in seen_cross: continue
      pts=base.pred().triangle_intersection_points(x[0],y[0])
      if not pts: continue
      seen_cross.add(key)
      cls=base.classify(a,b,x,y,pts,pose,native_d,native_l)
      counts["changed_cross_pairs"]+=1
      isbad=cls.startswith("unclassified")
      counts["cross_unclassified"]+=int(isbad);counts["cross_classified"]+=int(not isbad)
      class_counts[cls]=class_counts.get(cls,0)+1
      sink.write(json.dumps(event_obj(step,label,"cross",[a,b],face_key,[x[4],y[4]],cls,pts,base),sort_keys=True,separators=(",",":"))+"\n")
  return {"counts":counts,"class_counts":class_counts,"pairs_examined_scope":"all exact intersections involving changed faces in rows305 or308; unmodified pairs transfer from the complete original-owner scans"}
def main():
  if OUT.exists(): raise SystemExit("fresh output required")
  OUT.mkdir()
  runner=load(RUNNER_PATH,"audit_runner_1171_full24")
  if sha(RUNNER_PATH)!="37bd87c0ac870e59141422db5f6cd8be9b16452155472e61bd84840bc8f8583a": raise SystemExit("runner hash changed")
  probe_report=json.loads((PROBE_ROOT/"probe-24-report.json").read_text())
  for cohort_name,item in COHORTS.items():
    probed=probe_report["cohorts"][cohort_name]
    if probed["returncode"]!=0 or len(probed["outputs"])!=8 or probed["steps"]!=item["steps"]: raise SystemExit("probe cohort identity mismatch: "+cohort_name)
  # Exact source point IDs in candidate NPZ: index 0=305:21520, 1=305:21522, 2=308:10628.
  point_output_index={305:{21520:0,21522:1},308:{10628:2}}
  all_results=[]; all_inputs={"candidate_npz":{"path":str(ROOT/"candidate-rows.npz"),"sha256":sha(ROOT/"candidate-rows.npz")},"probe_report":{"path":str(PROBE_ROOT/"probe-24-report.json"),"sha256":sha(PROBE_ROOT/"probe-24-report.json")},"probe_executable":{"path":str(probe_report["probe"]["path"]),"sha256":probe_report["probe"]["sha256"]},"map_composition_report":{"path":str(DREPORT),"sha256":sha(DREPORT)},"lobe_lineage_report":{"path":str(LREPORT),"sha256":sha(LREPORT)},"runner":{"path":str(RUNNER_PATH),"sha256":sha(RUNNER_PATH)}}
  report={"schema":"numi.human.lung-free-apex-candidate-024-24-pose-delta-audit.v1","status":"running","scope":"Exact changed-star delta audit over the early8 capture set and both registered long-study 8-pose arms. Only pairs incident to the changed 305/308 face stars are rescanned; complete original unchanged-pair scans remain separate evidence. This is predicted geometry from the unchanged production point kernel, not a new native scene capture.","candidate":{"npz_path":str(ROOT/"candidate-rows.npz"),"npz_sha256":sha(ROOT/"candidate-rows.npz"),"trial_report_path":str(ROOT/"trial-report.json"),"trial_report_sha256":sha(ROOT/"trial-report.json"),"changed_face_ids":{str(k):v for k,v in CHANGED.items()},"changed_owner_vertices":{"305":[21520,21522],"308":[10628]},"changed_rows_audited":[305,308],"audited_peer_rows":[305,306,307,308,309,311]},"cohorts":{},"inputs":all_inputs}
  write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
  for cohort_name,item in COHORTS.items():
    run_path=item["run"]
    cfg={"base":str(runner.BASE),"run":run_path,"out":OUT/cohort_name,"nha_path":NHA,"nha_sha":NHA_SHA,"map_reports":[],"lobe_lineage_report":LREPORT,"d_map_composition_report":DREPORT,"geometry_only_area_mismatch":False,"workers":1,"probe_report":None,"registered_arm":item["arm"]}
    base,adapters,ctx=runner.prepare_worker_context(cfg)
    if list(map(int,ctx["steps"]))!=item["steps"]: raise SystemExit("accepted state schedule mismatch: "+cohort_name)
    cohort_dir=OUT/cohort_name;cohort_dir.mkdir()
    cohort_inputs={"run_path":str(ctx["run_path"]),"registered_arm":item["arm"],"accepted_steps":ctx["steps"],"nha_path":str(ctx["nha_path"]),"nha_sha256":ctx["nha_sha"],"capture_input_hashes":ctx["input_hashes"],"registered_receipt_chain":ctx.get("registered_context")}
    all_inputs["cohort:"+cohort_name]=cohort_inputs
    event_path=cohort_dir/"changed-star-exact-events.jsonl.gz"
    pose_results=[]
    with gzip.open(event_path,"wt",compresslevel=1,encoding="utf-8") as sink:
      for step in item["steps"]:
        started=time.monotonic()
        pose,pack=base.row_pose(ctx["run_path"],int(step),ctx["nha_rows"])
        # Isolate only changed rows; all other capture arrays are shared read-only.
        candidate_pose=dict(pose)
        for sid in CHANGED:
          row=dict(pose[sid]);row["v"]=pose[sid]["v"].copy();candidate_pose[sid]=row
        probe_dir=PROBE_ROOT/cohort_name
        mapped=np.fromfile(probe_dir/("step-%d.receipt.xyz-f32.bin"%step),dtype="<f4")
        if mapped.size!=9: raise SystemExit("candidate point output must contain exactly three mapped xyz records")
        mapped=mapped.reshape(3,3)
        for sid,verts in point_output_index.items():
          for vid,index in verts.items(): candidate_pose[sid]["v"][vid,:3]=mapped[index]
        native_d={sid:base.native_map(sid,ctx["maps_doc"],candidate_pose) for sid in base.LOBES}
        native_l=base.native_lobe_maps(ctx["lineage"],candidate_pose)
        if not all(x.get("valid") for x in native_d.values()) or not all(x.get("valid") for x in native_l.values()): raise SystemExit("native mapped interface invalid at %s:%d"%(cohort_name,step))
        scan=scan_changed(base,int(step),"candidate024",candidate_pose,native_d,native_l,sink)
        row={"cohort":cohort_name,"step":int(step),"pack_path":str(pack),"pack_sha256":sha(pack),"pack_receipt_sha256":sha(pack.with_suffix(".receipt.json")),"counts":scan["counts"],"class_counts":scan["class_counts"],"native_D_map_valid":True,"native_lobe_maps_valid":True,"elapsed_seconds":time.monotonic()-started}
        pose_results.append(row);print(json.dumps(row),flush=True)
    cohort_summary={"identity":cohort_inputs,"steps":pose_results,"witness":{"path":str(event_path),"sha256":sha(event_path),"bytes":event_path.stat().st_size},"totals":{k:sum(x["counts"][k] for x in pose_results) for k in pose_results[0]["counts"]},"elapsed_seconds":sum(x["elapsed_seconds"] for x in pose_results)}
    report["cohorts"][cohort_name]=cohort_summary;write(OUT/"progress.json",report)
  report["status"]="complete"
  report["inputs"]=all_inputs
  report["summary"]={"pose_count":sum(len(x["steps"]) for x in report["cohorts"].values()),"total_unclassified_cross":sum(x["totals"]["cross_unclassified"] for x in report["cohorts"].values()),"total_unallowed_self":sum(x["totals"]["self_unallowed"] for x in report["cohorts"].values()),"total_degenerate_changed_faces":sum(x["totals"]["degenerate_changed_faces"] for x in report["cohorts"].values()),"delta_scope_complete_for_declared_changed_faces":True}
  write(OUT/"report.json",report)
  print(json.dumps({"report_path":str(OUT/"report.json"),"report_sha256":sha(OUT/"report.json"),"summary":report["summary"],"elapsed_by_cohort":{k:v["elapsed_seconds"] for k,v in report["cohorts"].items()}},indent=2))
if __name__=="__main__":main()
