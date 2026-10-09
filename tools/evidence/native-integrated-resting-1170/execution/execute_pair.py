import argparse,hashlib,json,pathlib,subprocess,time,sys
E=pathlib.Path("/Users/n/numi-human-resting-evidence-20261005")
R=pathlib.Path("/Users/n/numi-human-retained-delivery-20261009")
OUT=pathlib.Path(__file__).resolve().parent
STUDY=R/"native-integrated-resting-study-1170"
NUMI="/Users/n/numi-human-performance-source-014/tools/numi"
SMOKE=R/"terminal-capture-fix-smoke/verification.json"
SHORT=pathlib.Path("/Users/n/numi-human-delta-runner-fix-1159/tools/evidence/native-lung-delta-skin-clearance-1141/provenance/actual-1159/actual-1159-geometry-review.json")
SKIN=E/"native-lung-audit-exact-delta-1120/attempt-015-skin-transfer-1159-package-loader/result/skin-clearance-delta-transfer.json"
def sha(p):
    with p.open("rb") as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def command(label,argv):
    log=OUT/(label+".log"); receipt=OUT/(label+".json")
    if log.exists() or receipt.exists():raise RuntimeError("existing command evidence: "+label)
    started=time.time(); clock=time.monotonic()
    print(json.dumps({"started":label,"argv":argv}),flush=True)
    with log.open("x") as f:
        result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,text=True)
    evidence={"argv":argv,"returncode":result.returncode,"started_unix":started,
              "finished_unix":time.time(),"wall_seconds":time.monotonic()-clock,"log":str(log)}
    receipt.write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps(evidence),flush=True)
    return result.returncode
def require_command(label,argv):
    if command(label,argv):raise RuntimeError("command failed: "+label)
def protect_closed(trial):
    # Only after the science command returns: no live files or writable study directories.
    p=STUDY/"trials"/trial
    if p.is_dir():subprocess.run(["/usr/bin/chflags","-R","uchg",str(p)],check=True)
    count=0
    for f in (STUDY/"objects").rglob("*"):
        if f.is_file() and not f.is_symlink():
            subprocess.run(["/usr/bin/chflags","uchg",str(f)],check=True)
            count+=1
    (OUT/(trial+"-retention.json")).write_text(json.dumps({
        "trial":str(p),"closed_after_science_return":True,
        "reversible_macOS_user_immutable":True,"protected_object_files":count,
        "unprotect_command":"/usr/bin/chflags -R nouchg "+str(p)},indent=2)+"\n")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=pathlib.Path,required=True)
    parser.add_argument("--plan-sha256",required=True)
    args=parser.parse_args()
    if STUDY.exists():raise RuntimeError("fresh study already exists")
    if sha(args.plan)!=args.plan_sha256:raise RuntimeError("plan changed")
    if sha(SHORT)!="a38342daef11eeab0c02adf446fc1784d965804bfbe0c325cf25d23870824246":raise RuntimeError("historical short summary changed")
    if sha(SKIN)!="467f57dafc2e20d38947b62363152925aeea08aae10482892bced70f5675c72c":raise RuntimeError("short skin report changed")
    smoke=json.loads(SMOKE.read_text())
    if not (smoke["actual_run_native_exit_code"]==0 and smoke["preflight_physiology_byte_identical"]
            and smoke["terminal_capture"]["verified"] and smoke["terminal_capture"]["accepted_step"]==10000):
        raise RuntimeError("fixed owner native smoke did not pass")
    prerequisites={
        "plan":{"path":str(args.plan),"sha256":sha(args.plan)},
        "fixed_owner_smoke":{"path":str(SMOKE),"sha256":sha(SMOKE)},
        "historical_short_geometry_summary":{"path":str(SHORT),"sha256":sha(SHORT),
          "raw_lung_report_deleted_by_separate_cleanup":True,
          "scope":"Retained published short-run summary only; not final-run geometry acceptance."},
        "short_skin_report":{"path":str(SKIN),"sha256":sha(SKIN)},
        "final_geometry_requirement":"Fresh full 1171 lung and 1172 skin audits on each actual registered arm remain required.",
        "study":str(STUDY)}
    (OUT/"prerequisites.json").write_text(json.dumps(prerequisites,indent=2)+"\n")
    require_command("register",[NUMI,"science","register",str(args.plan),str(STUDY)])
    require_command("registered-status",[NUMI,"science","status",str(STUDY)])
    # These registration outputs are closed; later science runs only read them.
    for closed in ("objects","prediction","registration.json"):
        subprocess.run(["/usr/bin/chflags","-R","uchg",str(STUDY/closed)],check=True)
    (OUT/"registration-retention.json").write_text(json.dumps({
        "closed_paths":[str(STUDY/x) for x in ("objects","prediction","registration.json")],
        "reversible_macOS_user_immutable":True,
        "active_trial_and_attempt_directories_unchanged":True},indent=2)+"\n")
    require_command("protected-registration-status",[NUMI,"science","status",str(STUDY)])
    for trial in ("resting-baseline","resting-drive-half"):
        code=command(trial,[NUMI,"science","run",str(STUDY)])
        protect_closed(trial)
        if code:raise RuntimeError("science command failed: "+trial)
        receipt=json.loads((STUDY/"trials"/trial/"receipt.json").read_text())["payload"]
        if receipt.get("failure") is not None or receipt.get("returncode")!=0:
            raise RuntimeError("trial failed: "+trial)
    require_command("analyze",[NUMI,"science","analyze",str(STUDY)])
    require_command("verify",[NUMI,"science","verify",str(STUDY)])
    print(json.dumps({"registered_pair_complete":True,
          "final_geometry_audits_and_recording_review_pending":True}),flush=True)
if __name__=="__main__":main()
