#!/usr/bin/env python3
"""Use the existing Human launch owner; preparation does not execute physics."""
import argparse, hashlib, json, os, sys
from pathlib import Path
OWNER=Path("/Users/n/numi-human-final-integration-publish-1397").resolve()
D=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378").resolve()
A=D.parent
sys.path.insert(0,str(OWNER/"src"))
from numilab_human import resting_run
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=D/"registered-resting-native-emission-diagnostic-004"
RECEIPT=D/"current-manifest-recovery-004/registered-resting-scene-source-preparation-002/resting-anatomy-receipt.json"
EXPECTED="323819ed17f2309da471dfe1a44cf49cd4256df8a2cc237c5964925f853c924b"
def inputs():
 assert sha(RECEIPT)==EXPECTED
 parser=argparse.ArgumentParser()
 resting_run.add_arguments(parser)
 args=parser.parse_args([
 "--body-scene",str(D/"current-manifest-recovery-004/registered-resting-scene-source-preparation-002/resting-supine-scene.manifest.json"),
 "--anatomy-receipt",str(RECEIPT),
 "--tendon","/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/bilateral-b42-native-tendon-owner-001/owner-base/numi-human-tendon-attachments.nhtendon",
 "--lab","/Users/n/numi-lab-resting-qualification-publish-1395",
 "--build","/Users/n/numi-lab-source-allocation-build-1403-003",
 "--output",str(P/"native-run"),
 "--circulation","/Users/n/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json",
 "--respiration","/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-reference-respiration.json",
 "--seconds",".002","--dt",".002","--dimension","512",
 "--inspection-tour","--inspection-period-seconds","2",
 "--postural-activation-cap",".01","--release-initialization",
 "--mtp-passive-stiffness-nm-per-rad","2","--rigid-hands","--contact-iterations","64"])
 argv,assets=resting_run.command(args)
 ref=D/"vein008-forearm474-back158-native-emission-diagnostic-002/invocation.json"
 env=dict(json.loads(ref.read_text())["environment"])
 env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]="0"
 env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(P/"native-run/common-field-failure.json")
 pins={str(p.resolve()):sha(p) for p in [Path(__file__),Path(resting_run.__file__),ref]}
 return args,{"argv":argv,"asset_sha256":assets,"environment":env,
 "preparation_input_sha256":pins,"scope":"One-step unadmitted native Float32 emission diagnostic for 31c registered scene with existing back412 and index-only intercostal repair. Not anatomy admission, native16 confirmation, sustained physiology, or qualification."}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true");a=parser.parse_args()
 args,plan=inputs()
 if not a.execute:
  P.mkdir(exist_ok=False)
  (P/"invocation-plan.json").write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"status":"prepared_no_native_execution","preflight":str(P/"invocation-plan.json"),"sha256":sha(P/"invocation-plan.json")}))
  return 0
 frozen=json.loads((P/"invocation-plan.json").read_text())
 assert frozen==plan,"launch inputs changed after preparation"
 os.environ.update(plan["environment"])
 return resting_run.run(args)
if __name__=="__main__":sys.exit(main())
