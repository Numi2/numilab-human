import json,hashlib
from pathlib import Path
B=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004")
out=B/"bilateral-b42-native-tendon-owner-001"
old=json.loads((B/"tendon-current-001/numi-human-tendon-attachments.manifest.json").read_text())
new=json.loads((out/"owner-base/numi-human-tendon-attachments.manifest.json").read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
fields=("muscle","muscle_index","endpoint","body_index","source_site_index","route_node_index","source_local_point_m","resolved_local_point_m","endpoint_migration_m")
assert len(old["endpoints"])==len(new["endpoints"])==832
assert all(all(x[k]==y[k] for k in fields) for x,y in zip(old["endpoints"],new["endpoints"]))
changed=[]
for i,(x,y) in enumerate(zip(old["endpoints"],new["endpoints"])):
 if x!=y: changed.append({"endpoint_index":i,"muscle":x["muscle"],"endpoint":x["endpoint"],"body_index":x["body_index"],"old_attachment_mode":x["attachment_mode"],"new_attachment_mode":y["attachment_mode"]})
bodyids=sorted({x["body_index"] for x in changed})
# All changed maps must be in the refreshed arm registration; torso/head/lower-body owners remain exact.
assert all(41<=x<=126 for x in bodyids),bodyids
migrated=[i for i,x in enumerate(old["endpoints"]) if x["endpoint_migration_m"]!=0]
assert len(migrated)==18 and all(old["endpoints"][i]==new["endpoints"][i] for i in migrated)
assert old["runtime_contract"]==new["runtime_contract"]
r={"schema":"numi.human.upper-limb-full-owner-base-compile.v1","status":"source_endpoint_and_existing_owner_invariants_passed","admission":False,"old_manifest_sha256":sha(B/"tendon-current-001/numi-human-tendon-attachments.manifest.json"),"new_manifest_sha256":sha(out/"owner-base/numi-human-tendon-attachments.manifest.json"),"new_payload_sha256":new["payload"]["sha256"],"endpoint_count":832,"source_and_resolved_endpoint_identity_fields_byte_decoded_exact":list(fields),"changed_attachment_envelopes":changed,"changed_body_indices":bodyids,"unchanged_endpoint_record_count":832-len(changed),"all_18_existing_foot_migrations_unchanged":True,"runtime_contract_unchanged":True,"old_coverage":old["coverage"],"new_coverage":new["coverage"],"evidence_boundary":"Existing tendon owner compile source preparation. Changed arm registration recalculates force-transfer envelopes; no authored endpoint, force parameter or physical state change. Native/anatomical admission pending."}
p=out/"source-endpoint-invariants-001.json";p.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report_sha256":sha(p),"changed":len(changed),"bodyids":bodyids,"migrated":len(migrated)}))
