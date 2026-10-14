from pathlib import Path
import struct,json,hashlib
d=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/bilateral-b42-native-bone-owner-001")
old=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/current-bone-registration-b1b410ad/bodyparts3d-myosim-major-bones.nhbones")
new=d/"owner-base/bodyparts3d-myosim-major-bones.nhbones"
a,b=old.read_bytes(),new.read_bytes()
assert hashlib.sha256(a).hexdigest()=="7d94e2f28ac6cea30c6c11a79919077acc40281636eb20b11b50b50aebd3ec49"
ha,hb=struct.unpack_from("<8s5I32s",a),struct.unpack_from("<8s5I32s",b)
assert ha[:5]==hb[:5] and ha[-1]==hb[-1] and ha[1:3]==(3,185)
rs=struct.calcsize("<6I8fI"); ve=60+185*rs
assert a[ve:]==b[ve:],"Bone XYZ/normal/index arrays changed"
changed=[]
for i in range(185):
 ra,rb=a[60+i*rs:60+(i+1)*rs],b[60+i*rs:60+(i+1)*rs]
 ua,ub=struct.unpack("<6I8fI",ra),struct.unpack("<6I8fI",rb)
 assert ua[:6]==ub[:6] and ua[-1]==ub[-1],"IDs/owners/offsets changed"
 if ra!=rb: changed.append(i+1)
expected=set(range(10,18))|set(range(72,126));expected-= {10,11}
assert set(changed)==expected,(changed,sorted(expected))
report={"source_payload_sha256":hashlib.sha256(a).hexdigest(),"candidate_payload_sha256":hashlib.sha256(b).hexdigest(),"changed_pose_record_stable_ids":changed,"unchanged_bone_rows":185-len(changed),"all_vertex_normal_triangle_index_arrays_byte_exact":True,"all_typed_source_owner_id_and_source_record_indices_byte_exact":True,"scapula10_11_exact":True,"source_and_native_admission":False}
out=d/"regional-row-byte-proof-002.json";assert not out.exists()
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),**report},indent=2))
