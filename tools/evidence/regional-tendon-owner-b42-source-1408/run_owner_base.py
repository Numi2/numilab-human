import hashlib,json,sys,time,signal,traceback
from pathlib import Path
B=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004")
R=Path("/Users/n/numi-human-final-integration-publish-1397")
OUT=B/"bilateral-b42-native-tendon-owner-001"; OUT.mkdir(exist_ok=False)
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
old=B/"tendon-current-001/numi-human-tendon-attachments.manifest.json"
bone=B/"bilateral-b42-native-bone-owner-001/owner-base"
myo=B/"recovered-artifact"
pins={old:"153c2b74345f02ad658df12d655fdd63f76e8212833af9691bd5f07e199c8fd2",bone/"bodyparts3d-myosim-major-bones.nhbones":"4d74bc786176c1d4479e3be8ebb7c97797a0d55feca687f83c89136e7c6635b0",myo/"myosim-fullbody-muscle-reference.nhmyo":"e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b",myo/"myosim-fullbody-core-reference.nhrigid":"2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4",myo/"myosim-fullbody-reference.manifest.json":"99401961daa509af80bc089fd750ff9c83e0a2ae3a734d242efa920e12e067db"}
report={"schema":"numi.human.upper-limb-full-owner-base-compile.v1","status":"preparing","admission":False,"owner":"numilab_human.model.numi_human_tendon_attachment_envelope_payload","started_unix":time.time(),"output":str(OUT)}
def save(): (OUT/"compile-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
def timeout(*args): raise TimeoutError("bounded 20-minute tendon owner compile exceeded")
signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
try:
 for p,v in pins.items(): assert p.is_file() and h(p)==v, str(p)
 oldj=json.loads(old.read_text());bj=json.loads((bone/"bodyparts3d-myosim-major-bones.manifest.json").read_text())
 assert bj["payload"]["registration_fingerprint32"]=="31c77660"
 params=oldj["admission"]
 assert [params[k] for k in ["maximum_surface_distance_m","maximum_patch_radius_m","maximum_sampled_total_force_amplification","maximum_migrated_endpoint_distance_m"]]==[.012,.012,4.0,.025]
 assert params["route_private_surface_migration"] is True
 owner=R/"src/numilab_human/model.py"
 report.update(input_hashes={str(p):v for p,v in pins.items()},owner_sha256=h(owner),bone_manifest_sha256=h(bone/"bodyparts3d-myosim-major-bones.manifest.json"),parameters={"maximum_surface_distance_m":.012,"maximum_patch_radius_m":.012,"maximum_force_amplification":4.0,"migrate_semantic_rigid_foot_endpoints":True,"maximum_migrated_endpoint_distance_m":.025},phase="preflight_passed")
 save()
 sys.path.insert(0,str(R/"src"))
 from numilab_human.model import numi_human_tendon_attachment_envelope_payload
 report["phase"]="owner_compile";save()
 manifest=numi_human_tendon_attachment_envelope_payload(myo,bone,OUT/"owner-base",**report["parameters"])
 # Inspect the returned existing schema rather than assuming output coverage fields.
 mp=OUT/"owner-base/numi-human-tendon-attachments.manifest.json"
 saved=json.loads(mp.read_text());payload=mp.parent/saved["payload"]["file"]
 assert manifest==saved and h(payload)==saved["payload"]["sha256"]
 assert saved["source"]["bodyparts3d_bone_payload"]["sha256"]==pins[bone/"bodyparts3d-myosim-major-bones.nhbones"]
 assert saved["source"]["bodyparts3d_bone_payload"]["registration_fingerprint32"]=="31c77660"
 assert saved["source"]["myosim_muscle_payload_sha256"]==pins[myo/"myosim-fullbody-muscle-reference.nhmyo"]
 assert {str(p):h(p) for p in pins}==report["input_hashes"]
 report.update(status="compiled_source_unadmitted",phase="passed",finished_unix=time.time(),manifest_sha256=h(mp),payload_sha256=h(payload),coverage=saved.get("coverage"),evidence_boundary=saved["evidence_boundary"])
 save()
 print(json.dumps({k:report[k] for k in ["status","manifest_sha256","payload_sha256","coverage","started_unix","finished_unix"]}))
except BaseException as e:
 report.update(status="failed",error=repr(e),traceback=traceback.format_exc(),finished_unix=time.time());save();raise
finally: signal.alarm(0)
