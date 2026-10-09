#!/usr/bin/env python3
"""Compose the fixed EPL143 NHSKIN and build an owner-native argv preview; never launch it."""
import argparse, copy, hashlib, json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187')
CAND=ROOT/'attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin'
CORR=ROOT/'attempt-006/epl143-position-correction.nhskin'
REPORT=ROOT/'attempt-006/report.json'
REVIEW=ROOT/'package-002/result/review.json'
REG=Path('/Users/n/numi-human-resting-evidence-20261005/native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/common-atlas-skin-geometry-registration.manifest.json')
BASE=Path('/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-1159-viewer-018-v015-attempt1')
SCENE=BASE/'resting-scene/resting-supine-scene.manifest.json'
RECEIPT=BASE/'anatomy/resting-anatomy-receipt.json'
INV=BASE/'native-run/invocation.json'
SUPPORT=BASE/'resting-scene/myosim-fullbody-resting-bed-support.nhcnt'
OWNER=Path('/Users/n/numi-human-free-apex-two-family-1178')
REV='f13b57547f25d601431e65a4e090b61f2b6672c7'
REVIEW_SHA='c030ff4f9421cfef096b6fba6662d3dde28f355af8516935e4262c7b95999868'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def bound(p):
 p=Path(p).resolve(); return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def need(ok,msg):
 if not ok: raise RuntimeError(msg)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 out=a.output.resolve();need(not out.exists(),f'refuse existing output: {out}');out.mkdir(parents=True)
 for p in (CAND,CORR,REPORT,REVIEW,REG,SCENE,RECEIPT,INV,SUPPORT):need(p.is_file(),f'missing input: {p}')
 need(sha(REVIEW)==REVIEW_SHA,'package-002 review changed')
 rep=json.loads(REPORT.read_text())
 need(rep['status']=='candidate_geometry_only_pass' and rep['source_frame_and_field']['selected_engineering_trial_amplitude_mm']==.5,'fixed candidate report changed')
 need(sha(CAND)==rep['candidate_assets']['composed_candidate_sha256'] and sha(CORR)==rep['candidate_assets']['position_only_correction_sha256'],'candidate/correction hash mismatch')
 sys.path.insert(0,str(OWNER/'src'))
 from numilab_human import resting_anatomy,resting_run
 files={n:bound(OWNER/'src/numilab_human'/n) for n in ('resting_anatomy.py','resting_run.py','resting_scene.py','skin_source_payload_preflight.py','model.py')}
 need(subprocess.run(['git','-C',str(OWNER),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()==REV,'owner revision changed')
 need(not subprocess.run(['git','-C',str(OWNER),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip(),'owner checkout dirty')
 base=json.loads(RECEIPT.read_text()); source=Path(base['mass_geometry_accounting']['skin_payload_path']).resolve(); source_sha=base['mass_geometry_accounting']['skin_payload_sha256']
 need(sha(source)==source_sha and source_sha==json.loads(REG.read_text())['output_payload']['sha256'],'base skin mismatch')
 reg=json.loads(REG.read_text()); candidate_sha=sha(CAND); manifest=copy.deepcopy(reg)
 manifest_path=out/'candidate-skin-geometry-registration.manifest.json'
 manifest['output_manifest']=str(manifest_path)
 manifest['output_payload']={'path':str(CAND.resolve()),'sha256':candidate_sha,'bytes':CAND.stat().st_size}
 manifest['inputs']['source_payload']={'path':str(source),'sha256':source_sha,'bytes':source.stat().st_size,'route':'exact 927 registered common-atlas NHSKIN input to fixed inferred EPL143 source-position correction'}
 manifest['inputs']['upstream_skin_provenance']={'immediate_source_path':str(REG.resolve()),'immediate_source_sha256':sha(REG),'immediate_source_record':{'path':str(source),'sha256':source_sha,'bytes':source.stat().st_size},'prior_lineage_retained':'927 manifest preserves 907/003 registration ancestry'}
 manifest['geometry_registration']={'schema':'numi.human.epl143-inferred-skin-clearance-candidate.v1','method':'fixed 0.5 mm compact geodesic-taper source-position field applied to exact 927 common-atlas NHSKIN','source_payload_sha256':source_sha,'correction_payload':bound(CORR),'output_payload_sha256':candidate_sha,'operation_report':bound(REPORT),'changed_position_vertex_ids':rep['source_frame_and_field']['changed_source_vertex_ids'],'changed_normal_vertex_ids':rep['candidate_assets']['changed_normal_vertex_ids'],'preservation':{'binding_records_byte_identical':True,'full_86_weight_matrix_byte_identical':True,'per_vertex_influence_records_byte_identical':True,'triangle_indices_and_order_byte_identical':True,'source_archive_identity_identical':True},'qualification':'inferred mixed-source reference geometry; 16 saved-pose geometry screens only; native contact/support response not qualified'}
 manifest['evidence_boundary']='The 0.5 mm amplitude is an engineering reference-geometry choice, not measured anatomy or certified clearance. Sixteen discrete geometry screens passed; dynamic support selector/Jacobian and native contact response remain untested.'
 manifest['preservation']={'all_86_canonical_binding_records_byte_identical':True,'all_86_binding_records_match_canonical_reference':True,'full_86_column_weight_matrix_byte_identical':True,'full_weight_matrix_byte_identical':True,'per_vertex_influence_records_byte_identical':True,'triangle_indices_and_order_byte_identical':True,'registration_fingerprint32_preserved':True,'source_archive_sha256_preserved':True,'indexed_rest_world_normals_recomputed':True,'nhtiss_tendon_muscle_and_physics_payloads_touched':False,'skin_contact_or_physics_owner_changed':False,'skin_topology_changed':False}
 manifest['qualification']={'EPL143_route_clearance':'passed changed-face exact screen on 16 discrete saved poses','native_pose_clearance':'not run for this payload','physical_or_collision_use':'not admitted','dynamic_support_selector_and_force_jacobian':'not tested; candidate overlaps active distal-thumb support taper','skin_self_intersection':'changed-star self gate passed on 16 offline saved poses; no native full-cycle qualification'}
 manifest["source_rights_and_provenance"]=copy.deepcopy(reg.get("geometry_registration",{}).get("source_rights_and_provenance",{}))
 manifest["source_rights_and_provenance"]["epl143_inferred_correction"]=bound(REPORT)
 manifest['status']='inferred_common_atlas_geometry_candidate_pending_native_clearance_and_pose_checks'
 manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 comp_dir=out/'composed-anatomy'
 composed=resting_anatomy.compose_skin_binding_candidate(RECEIPT,CAND,manifest_path,comp_dir)
 need(composed['status']=='skin_receipt_composed_with_recomputed_open_surface_accounting' and composed['anatomy_payload_unchanged'] and composed['functional_bindings_unchanged'],'owner receipt composition failed')
 receipt=Path(composed['receipt_path']); cdoc=json.loads(receipt.read_text())
 need(cdoc['payload']==base['payload'] and cdoc['functional_bindings']==base['functional_bindings'],'NHA or functional bindings changed')
 need(cdoc['mass_geometry_accounting']['skin_payload_sha256']==candidate_sha,'receipt does not bind candidate')
 scene=json.loads(SCENE.read_text()); scene['source']['skin']['path']=str(CAND.resolve());scene['source']['skin']['sha256']=candidate_sha
 scene_out=out/'resting-supine-scene.manifest.json';scene_out.write_text(json.dumps(scene,indent=2,sort_keys=True)+'\n')
 need(scene['outputs']['support_contact']['sha256']==sha(SUPPORT),'support payload identity changed')
 args=SimpleNamespace(body_scene=scene_out,anatomy_receipt=receipt,tendon=Path('/Users/n/numi-human-resting-evidence-20261005/tendon-semantic-foot-migration-834/numi-human-tendon-attachments.nhtendon'),lab=Path('/Users/n/numi-human-performance-source-014'),build=Path('/Users/n/numi-human-retired-alias-visibility-build-018-attempt2'),output=out/'native-run-not-executed',circulation=Path('/Users/n/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json'),respiration=Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-reference-respiration.json'),seconds=20.,dt=.002,dimension=512,mechanics_only=False,inspection_tour=True,inspection_period_seconds=2.5,drive_intervention=None,postural_activation_cap=.01,release_initialization=True,upper_passive_joints=False,rigid_hands=True,contact_iterations=64)
 argv,assets=resting_run.command(args); parent=json.loads(INV.read_text()); old=parent['argv']
 normalize={str(args.output.resolve()):str(Path(old[4]).resolve()),str(CAND.resolve()):str(source),str(receipt.resolve()):str(RECEIPT.resolve())}
 parent_output=str(Path(old[4]).resolve())
 normalized_argv=[x.replace(str(args.output.resolve()),parent_output) for x in argv]
 normalized_argv=[normalize.get(x,x) for x in normalized_argv]
 need(normalized_argv==old,'owner argv differs beyond declared skin/receipt/output paths')
 normalized_assets={str(Path(k).resolve()):v for k,v in assets.items()}; old_assets={str(Path(k).resolve()):v for k,v in parent['asset_sha256'].items()}
 asset_path_map={str(scene_out.resolve()):str(SCENE.resolve()),str(receipt.resolve()):str(RECEIPT.resolve()),str(CAND.resolve()):str(source)}
 # compose_skin_binding_candidate relocates the three cardiac sidecars beside
 # the derived receipt. Rebind only these known payloads to their original
 # receipt-relative paths, and require exact hashes before comparing argv inputs.
 sidecar_names={'common-cardiac-map-f32.bin','common-cardiac-volumes-f32.bin','common-cardiac-domains-f32.bin'}
 old_sidecars={Path(k).name:k for k in old_assets if Path(k).name in sidecar_names and Path(k).parent==RECEIPT.parent}
 candidate_sidecars={Path(k).name:k for k in normalized_assets if Path(k).name in sidecar_names and Path(k).parent==receipt.parent}
 need(set(old_sidecars)==sidecar_names and set(candidate_sidecars)==sidecar_names,'missing or unexpected cardiac sidecar assets')
 for name in sorted(sidecar_names):
  old_path=old_sidecars[name]; candidate_path=candidate_sidecars[name]
  need(normalized_assets[candidate_path]==old_assets[old_path],f'relocated cardiac sidecar changed: {name}')
  asset_path_map[candidate_path]=old_path
 compared_assets={asset_path_map.get(k,k):v for k,v in normalized_assets.items()}
 changed={k:{'parent':old_assets.get(k),'candidate':compared_assets.get(k)} for k in sorted(set(old_assets)|set(compared_assets)) if old_assets.get(k)!=compared_assets.get(k)}
 allowed={str(SCENE.resolve()),str(RECEIPT.resolve()),str(source)}
 need(set(changed)==allowed,f'unexpected input-identity changes: {changed}')
 need(normalized_assets[str(CAND.resolve())]==candidate_sha and normalized_assets[str(receipt.resolve())]==sha(receipt),'candidate absent from owner input bindings')
 need(not args.output.exists(),'native output unexpectedly created')
 outdoc={'schema':'numi.human.epl143-native-assembly-admission-preview.v1','status':'owner_composition_and_native_command_preparation_passed','qualification':'CPU-only anatomy receipt composition and native argv construction; no native process/GPU was launched','native_launch_executed':False,'candidate':bound(CAND),'correction':bound(CORR),'candidate_screen':bound(REPORT),'package_002_review':bound(REVIEW),'candidate_manifest':bound(manifest_path),'base_anatomy_receipt':bound(RECEIPT),'composed_anatomy_receipt':bound(receipt),'composed_anatomy_manifest':bound(Path(composed['manifest_path'])),'base_scene_manifest':bound(SCENE),'derived_scene_manifest':bound(scene_out),'NHCNT_support_payload':bound(SUPPORT),'owner':{'repo':str(OWNER),'revision':REV,'source_files':files,'compose_status':composed['status'],'NHA_unchanged':composed['anatomy_payload_unchanged'],'functional_bindings_unchanged':composed['functional_bindings_unchanged']},'native_owner_preview':{'argv':argv,'asset_sha256':normalized_assets,'candidate_vs_parent_input_changes':changed,'argv_matches_parent_after_declared_path_substitutions':True,'environment':'same pinned 1159 launch environment; not executed','reserved_output_path':str(args.output.resolve())},'support_caveat':'The correction lies within active right distal-thumb NHCNT region 15 taper. Static nearest-seed assignment and baseline winners are reported, but candidate dynamic minimum, selected point and force Jacobian are not checked; this preview is not contact or physics admission.','source_interpretation':'Inferred mixed-source registered reference geometry, not measured participant anatomy or measured skin thickness.'}
 report_path=out/'admission-preview.json';report_path.write_text(json.dumps(outdoc,indent=2,sort_keys=True)+'\n')
 (out/'native-owner-command.json').write_text(json.dumps({'argv':argv,'asset_sha256':normalized_assets,'qualification':outdoc['qualification'],'native_launch_executed':False},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':outdoc['status'],'report':str(report_path),'sha256':sha(report_path),'argv_sha256':hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest(),'native_launch_executed':False}))
if __name__=='__main__':main()
