"""Export the certified local rib50 partition through the retained source owners."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H/"src"))
from numilab_human import composed_bone_body_frame as bones,model
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
from numilab_human.resting_pleura_proxy import _parse_payload
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bones-artifact",type=Path,required=True)
parser.add_argument("--nha",type=Path,required=True)
parser.add_argument("--candidate-directory",type=Path,required=True)
parser.add_argument("--source-ancestry",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
B=args.bones_artifact.resolve()
N=args.nha.resolve()
I=args.candidate_directory.resolve()
ANCESTRY=args.source_ancestry.resolve()
O=args.output.resolve()
C=O/"compiled"
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
start=time.monotonic();C.mkdir(parents=True,exist_ok=False)
prior_manifest=json.loads((B/"bodyparts3d-myosim-major-bones.manifest.json").read_text())
assert sha(B/"bodyparts3d-myosim-major-bones.nhbones")==prior_manifest["payload"]["sha256"]
assert N.is_file()
prepared=json.loads((I/"preparation-report.json").read_text())
assert sha(I/"candidate.npz")==prepared["candidate_sha256"]
strict=json.loads((I/'strict-body20-float32-partition-audit.json').read_text())
assert strict['candidate_sha256']==sha(I/'candidate.npz')
with np.load(I/'candidate.npz',allow_pickle=False) as data:
 attrs=data['rib50_source_vertices6'].copy();faces=data['rib50_faces'].copy()
 nha_attrs=data['intercostal313_source_vertices6'].copy();nha_faces=data['intercostal313_faces'].copy()
 pairs=data['rib50_interface_vertex_pairs'].tolist()
source_raw=(B/'bodyparts3d-myosim-major-bones.nhbones').read_bytes()
raw,proof=bones.replace_bone_source_partition(source_raw,stable_id=50,vertices6=attrs,faces=faces)
bone=C/'bodyparts3d-myosim-major-bones.nhbones';bone.write_bytes(raw)
manifest=json.loads((B/'bodyparts3d-myosim-major-bones.manifest.json').read_text())
repair=dict(input_artifact=str(B),input_manifest_sha256=sha(B/'bodyparts3d-myosim-major-bones.manifest.json'),
 input_payload_sha256=sha(B/'bodyparts3d-myosim-major-bones.nhbones'),proof=proof)
for k,p in dict(candidate=I/'candidate.npz',source_ancestry=ANCESTRY,
 preparation=I/'preparation-report.json',strict_audit=I/'strict-body20-float32-partition-audit.json').items():
 repair[k]=dict(path=str(p),sha256=sha(p))
manifest['source_partition_retriangulation']=repair
bh=bones._BONE_HEADER.unpack_from(raw);manifest['payload'].update(sha256=sha(bone),bytes=len(raw),vertex_count=bh[3],index_count=bh[4])
mp=C/'bodyparts3d-myosim-major-bones.manifest.json';write(mp,manifest)
validation=bones.validate_composed_rib_reframe(C,manifest,raw)
old=N.read_bytes();_,_,rows=_parse_payload(old)
original_nha_slots=len(rows[313]["vertices6"])
assert len(nha_attrs)>=original_nha_slots
assert np.array_equal(nha_attrs[:original_nha_slots],np.asarray(rows[313]["vertices6"],dtype="<f4"))
original_bone_slots=next(r for r in bones._decode_bones(source_raw)[1] if r[5]==50)[2]
row=dict(rows[313]);row.update(vertices6=nha_attrs,faces=nha_faces)
nha_raw,emission=encode_payload_with_row_replacements(old,{313:row})
nha=C/'resting-thorax.nhanatomy';nha.write_bytes(nha_raw)
_,_,outrows=_parse_payload(nha_raw)
for sid,oldrow in rows.items():
 if sid==313:continue
 assert np.array_equal(np.asarray(oldrow['vertices6']),np.asarray(outrows[sid]['vertices6']))
 assert np.array_equal(np.asarray(oldrow['faces']),np.asarray(outrows[sid]['faces']))
source_sha=manifest['source']['myosim_source_archive_sha256']
decoded,metadata,path=model._numi_human_bone_envelope_surfaces(C,source_sha)
assert sum(map(len,decoded.values()))==185 and sha(path)==sha(bone)
current=next(s for group in decoded.values() for s in group if s['stable_id']==50)
assert current['record_pose_f32']==[0.,0.,0.,0.,0.,0.,1.,1.]
assert np.array_equal(current['stored_vertex_attributes_f32'],attrs)
write(O/'rib50-interface-vertex-pairs.json',pairs)
result=dict(status='PASS_EXISTING_SOURCE_OWNER_EXPORT_ONLY',
 runner_sha256=sha(__file__),output_nha_path=str(nha),output_nha_sha256=sha(nha),
 output_bones_path=str(bone),output_bones_sha256=sha(bone),output_bones_manifest_sha256=sha(mp),
 input_nha_path=str(N),input_nha_sha256=sha(N),
 input_bones_sha256=sha(B/'bodyparts3d-myosim-major-bones.nhbones'),
 source_preparation_report_sha256=sha(I/'preparation-report.json'),
 strict_body20_float32_audit_sha256=sha(I/'strict-body20-float32-partition-audit.json'),
 source_face_ancestry_sha256=sha(ANCESTRY),
 rib50_interface_vertex_pairs_path=str(O/'rib50-interface-vertex-pairs.json'),
 rib50_interface_vertex_pairs_sha256=sha(O/'rib50-interface-vertex-pairs.json'),
 full_existing_185_bone_envelope_loader_passed=True,untouched_nha_rows_exact=len(rows)-1,
 retained_original_source_slots_exact=True,original_rib50_slots_explicitly_body20_baked=True,
 appended_rib50_slots=len(attrs)-original_bone_slots,appended_intercostal_slots=len(nha_attrs)-original_nha_slots,
 source_xyz_normals_allocations_and_attachment_maps_exact=True,
 new_partition_slots_appended=True,nha_emission=emission,
 physical_mass_or_state_changed=False,native_admission=False,retained_motion_admission=False,anatomical_admission=False,
 validation_sha256=validation['validation_sha256'],wall_seconds=time.monotonic()-start)
write(O/'source-preparation-report.json',result);print(json.dumps(result),flush=True)

