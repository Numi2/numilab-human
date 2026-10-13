from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276")
SRC=Path("/Users/numi/numi-human-anatomy-finish-1378/src")
sys.path.insert(0,str(SRC))
from numilab_human.resting_anatomy_interface_patch import HEADER,RECORD,parse_payload
BASE=Path("/Users/numi/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy")
BASE_RECEIPT=ROOT/"back412-skin-reference-native-021-003/resting-anatomy-receipt.json"
GEOMETRY_REPORT=ROOT/"full-anatomy-delivery-audit-1378/liver-id22-aggregate-geometry-audit-002.json"
IDENTITY_REPORT=ROOT/"full-anatomy-delivery-audit-1378/liver-id22-source-identity-audit.json"
OUT=ROOT/"full-anatomy-delivery-audit-1378/liver-runtime-single-shell-candidate-001"
OLD_SHA="1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
OLD_RECEIPT_SHA="00c154dc9f9abf7378f4621bb5a7db464182ee77ddec10a53b71efb1ed4b1382"
ID22_SHA="2f5e34fceea6c8b718bd3f50a125bf24c434ae7eb3d925b9795b0585272c1349"
def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()
def sha(path): return sha_bytes(Path(path).read_bytes())
assert sha(BASE)==OLD_SHA and sha(BASE_RECEIPT)==OLD_RECEIPT_SHA
assert sha(GEOMETRY_REPORT)=="18011492b1907b38ee5f5d1837ade3c384ac0706a86ee6652f5a0c129026cbac"
h,rows=parse_payload(BASE); receipt=json.loads(BASE_RECEIPT.read_text())
assert 22 in rows and set(rows)==set(map(int,receipt["provenance"]["source_id_map"]))|{22}
assert 22 not in set(map(int,receipt["provenance"]["source_id_map"]))
alias=receipt["provenance"]["retired_liver_alias_associations"]
assert alias["retired_stable_id"]==22 and alias["canonical_stable_id"]==21
assert receipt["provenance"]["pre_aggregate_liver_source_id_map"]["22"]["source_sha256"]==ID22_SHA
identity=json.loads(IDENTITY_REPORT.read_text())
assert identity["source_to_payload"]["source_triangle_count"]==21592
# Create an ABI5 composition that omits only retired stable ID22. Keep every other
# record's decoded Float32 vertices/normals, local triangle indices, body, layer and flags.
newrows={sid:r for sid,r in rows.items() if sid!=22}
source_ids=set(map(int,receipt["provenance"]["source_id_map"]))
assert set(newrows)==source_ids
surfaces=[];vertex_blocks=[];index_blocks=[];nv=ni=0
for sid in sorted(newrows):
 r=newrows[sid];v6=np.asarray(r["vertices6"],dtype="<f4");faces=np.asarray(r["faces"],dtype=np.uint32)
 assert np.isfinite(v6).all() and (not faces.size or (int(faces.min())>=0 and int(faces.max())<len(v6)))
 surfaces.append(RECORD.pack(int(r["body_index"]),nv,len(v6),ni,faces.size,int(sid),int(r["layer"]),int(r["flags"])))
 vertex_blocks.append(v6.tobytes(order="C"));index_blocks.append((faces.ravel()+nv).astype("<u4").tobytes(order="C"));nv+=len(v6);ni+=faces.size
payload=HEADER.pack(h[0],h[1],len(surfaces),nv,ni,h[5],h[6])+b"".join(surfaces)+b"".join(vertex_blocks)+b"".join(index_blocks)
OUT.mkdir(exist_ok=False)
payload_path=OUT/"resting-thorax.nhanatomy"
with payload_path.open("xb") as f:f.write(payload)
new_sha=sha(payload_path)
# Reread with the repository's existing ABI5 parser; stable IDs remain sparse and unique.
h2,check=parse_payload(payload_path)
assert set(check)==source_ids and 22 not in check
assert len(check)==523 and int(h2[3])==nv and int(h2[4])==ni
for sid in sorted(source_ids):
 a,b=rows[sid],check[sid]
 assert (a["body_index"],a["layer"],a["flags"])==(b["body_index"],b["layer"],b["flags"])
 assert a["vertices6"].tobytes()==b["vertices6"].tobytes(),f"vertices/normals changed for {sid}"
 assert a["faces"].tobytes()==b["faces"].tobytes(),f"faces changed for {sid}"
receipt["payload"].update({"path":str(payload_path),"sha256":new_sha,"surface_count":len(check),"vertex_count":nv,"index_count":ni,
    "registration_fingerprint32":f"{h2[5]:08x}","input_payload_sha256":OLD_SHA})
receipt["functional_bindings"]["anatomy_payload_sha256"]=new_sha
receipt["provenance"]["retired_runtime_surface_22"]={
  "stable_id":22,"canonical_aggregate_surface_stable_id":21,"disposition":"retired from runtime anatomy payload; original source asset and pre-aggregate identity record retained",
  "source_owner":"BodyParts3D v4.0 part_of/FJ2824 liver component; FMA7197; source SHA256 "+ID22_SHA,
  "reason":"current native evidence shows the composed 14-21 envelope is closed/outward-positive; ID22 is an independent exterior source surface with exact interior-through-shell triangle crossings and source-mesh self-intersections, not an internal liver segment partition or second validated volume",
  "geometry_evidence_report_path":str(GEOMETRY_REPORT),"geometry_evidence_report_sha256":sha(GEOMETRY_REPORT),
  "identity_evidence_report_path":str(IDENTITY_REPORT),"identity_evidence_report_sha256":sha(IDENTITY_REPORT),
  "source_assets_deleted":False,"body_mass_or_body_binding_changed":False,"stable_ids_reassigned":False}
receipt_path=OUT/"resting-anatomy-receipt.json"
with receipt_path.open("x") as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
receipt_sha=sha(receipt_path)
report={"schema":"numi.human.anatomy.retired-liver-source-alias-candidate.v1","status":"candidate_ready_for_native_replay",
 "scope":"New NHANAT1 ABI5 payload and receipt omit only stable ID22 from runtime geometry. Original payload, receipts, source assets, mesh assets, and native runtime were not edited.",
 "inputs":{"source_payload":str(BASE),"source_payload_sha256":OLD_SHA,"source_receipt":str(BASE_RECEIPT),"source_receipt_sha256":OLD_RECEIPT_SHA,
  "source_map_count":len(source_ids),"source_map_has_22":False,"source_payload_had_extra_22":True,
  "geometry_report":str(GEOMETRY_REPORT),"geometry_report_sha256":sha(GEOMETRY_REPORT),"id22_source_sha256":ID22_SHA},
 "candidate":{"payload":str(payload_path),"payload_sha256":new_sha,"receipt":str(receipt_path),"receipt_sha256":receipt_sha,
  "abi":int(h2[1]),"surface_count":len(check),"vertex_count":nv,"index_count":ni,"registration_fingerprint32":f"{h2[5]:08x}",
  "runtime_stable_ids_match_current_receipt_source_map":True,"stable_id_22_absent":True,"stable_id_gaps_preserved":True},
 "exact_regression":{"all_non22_rows_byte_identical_after_decode_and_repack":True,"non22_stable_id_count":len(source_ids),
  "record_body_layer_flags_vertices6_and_local_faces_equal":True,"mass_geometry_and_functional_bindings_other_than_payload_sha_unchanged":True,
  "source_mesh_and_preaggregate_id22_identity_preserved_in_receipt":True,"source_files_deleted_or_modified":False,"native_runtime_changed":False},
 "limits":"This is a bounded anatomy-payload candidate, not a native replay. Native acceptance must confirm that removing row22 keeps the accepted 14-21 liver surface and all other emitted anatomy stable; user-level mass remains the authorized generic 72 kg reference and no independent liver tissue-mass claim is added."}
report_path=OUT/"candidate-report.json"
with report_path.open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps({"candidate_payload":str(payload_path),"candidate_payload_sha256":new_sha,"candidate_receipt":str(receipt_path),"candidate_receipt_sha256":receipt_sha,
 "surface_count":len(check),"vertex_count":nv,"index_count":ni,"non22_rows_byte_identical":True,"report":str(report_path),"report_sha256":sha(report_path)},sort_keys=True),flush=True)
