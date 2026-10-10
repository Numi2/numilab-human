from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path('/Users/n/numi-human-retained-delivery-20261009')
E=ROOT/'source-seam-connectivity-1247'
HUMAN=Path('/Users/n/numi-human-q-audit-provenance-1240')
LAB=Path('/Users/n/numi-human-performance-source-014')
BUILD=Path('/Users/n/numi-human-q-integration-publish-build-002')
BUILD_SOURCE=Path('/Users/n/numi-human-q-integration-publish-1242')
PARENT_DIR=ROOT/'skin-resting-multipose-clearance-1218/native-baseline-310s-preparation'
PARENT_DECL=PARENT_DIR/'run-declaration.json'
PARENT_EXEC=PARENT_DIR/'execution.json'
PARENT_NATIVE=PARENT_DIR/'native-run/run-metadata.json'
BASE_RECEIPT=ROOT/'skin-resting-multipose-clearance-1218/package-preparation-002/composed-candidate/composed-anatomy/resting-anatomy-receipt.json'
SCENE=ROOT/'skin-resting-multipose-clearance-1218/package-preparation-002/composed-candidate/resting-supine-scene.manifest.json'
CAND_DIR=E/'fhl-current-7b23-compose-004/candidate'
CAND_PAYLOAD=CAND_DIR/'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
CAND_MANIFEST=CAND_DIR/'bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'
CAND_REPORT=CAND_DIR/'report.json'
CAND_RECEIPT=CAND_DIR/'resting-anatomy-receipt.json'
DIRECT_PARENT=ROOT/'passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003'
DIRECT_PARENT_RECEIPT=DIRECT_PARENT/'resting-anatomy-receipt.json'
DIRECT_PARENT_PAYLOAD=DIRECT_PARENT/'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
DIRECT_PARENT_MANIFEST=DIRECT_PARENT/'bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'
DEST=E/'native-fhl-10s-preparation-001'
OUT=DEST/'native-run'
RUNNER=DEST/'run.py'
PREPARER=DEST/'prepare_fhl_native_10s.py'


def sha(p:Path)->str:
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def pinmap(paths):
    out={}
    for p in paths:
        p=Path(p).resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        out[str(p)]=sha(p)
    return out

def replace_env(argv,name,value):
    prefix=name+'='
    hits=[i for i,x in enumerate(argv) if x.startswith(prefix)]
    if len(hits)!=1: raise ValueError(f'expected one {name} env entry, found {hits}')
    old=argv[hits[0]]
    argv[hits[0]]=prefix+value
    return old,argv[hits[0]]

def replace_option(argv,option,value):
    hits=[i for i,x in enumerate(argv) if x==option]
    if len(hits)!=1: raise ValueError(f'expected one {option}, found {hits}')
    i=hits[0]
    if i+1>=len(argv): raise ValueError(f'missing value for {option}')
    old=argv[i+1]
    argv[i+1]=value
    return old,value

def git_identity(repo):
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    if status: raise ValueError(f'expected clean source checkout: {repo}: {status}')
    return {'path':str(repo),'commit':commit,'worktree_clean':True}

if DEST.exists(): raise FileExistsError(f'refusing existing preparation dir: {DEST}')
if OUT.exists(): raise FileExistsError(f'refusing existing native output: {OUT}')
parent=json.loads(PARENT_DECL.read_text())
parent_exec=json.loads(PARENT_EXEC.read_text())
parent_metadata=json.loads(PARENT_NATIVE.read_text())
base_receipt=json.loads(BASE_RECEIPT.read_text())
scene=json.loads(SCENE.read_text())
candidate_receipt=json.loads(CAND_RECEIPT.read_text())
candidate_manifest=json.loads(CAND_MANIFEST.read_text())
candidate_report=json.loads(CAND_REPORT.read_text())
direct_receipt=json.loads(DIRECT_PARENT_RECEIPT.read_text())

# Closed parent run and all current source/scene identities must match their recorded pins.
if sha(PARENT_DECL)!='5319d47c6c2fd9fa0a581edbb02b07e6e6f18301407b060e6b52d13a5169f90f': raise ValueError('310 s parent declaration hash drift')
if parent_exec.get('returncode')!=0 or parent_exec.get('changed_inputs')!={}: raise ValueError('310 s parent run is not a closed unchanged success')
if parent_metadata.get('returncode',0)!=0: raise ValueError('310 s parent native metadata not successful')
if scene.get('schema')!='numi.human.resting-supine-source-scene.v1': raise ValueError('unexpected body scene schema')
if scene.get('bed',{}).get('heightfield') is not None: raise ValueError('expected the retained flat-bed scene, not a contoured bed')
if candidate_report.get('composer_source_sha256')!=sha(HUMAN.parent/'numi-human-fhl-source-composition-1248/src/numilab_human/passive_attachment_composition.py'):
    raise ValueError('FHL composition report is not bound to current tested composer')
if candidate_report.get('payload_sha256')!=sha(CAND_PAYLOAD) or candidate_manifest.get('payload',{}).get('sha256')!=sha(CAND_PAYLOAD): raise ValueError('candidate payload/report/manifest identity mismatch')
if candidate_manifest.get('source',{}).get('fhl_source_seam_correction',{}).get('changed_stable_ids')!=[27,28]: raise ValueError('candidate FHL stable IDs mismatch')
if candidate_report.get('changed_stable_ids') not in (None,[27,28]): raise ValueError('candidate changed IDs are not exactly the FHL pair')
if candidate_receipt.get('provenance',{}).get('native_muscle_surfaces',{}).get('sha256')!=sha(CAND_PAYLOAD): raise ValueError('candidate anatomy receipt does not bind payload')
if candidate_receipt.get('provenance',{}).get('native_muscle_surfaces',{}).get('manifest_sha256')!=sha(CAND_MANIFEST): raise ValueError('candidate anatomy receipt does not bind manifest')
if candidate_receipt.get('provenance',{}).get('fhl_source_seam_correction_binding',{}).get('changed_stable_ids')!=[27,28]: raise ValueError('receipt FHL binding missing')
if 'source_preserving_correction' in candidate_receipt['provenance']['fhl_source_seam_correction_binding']:
    raise ValueError('FHL binding incorrectly contains biceps correction detail')
if candidate_receipt['provenance'].get('biceps_source_preserving_correction_binding') != direct_receipt['provenance'].get('biceps_source_preserving_correction_binding'):
    raise ValueError('inherited biceps receipt binding was not preserved')
if candidate_receipt.get('payload')!=direct_receipt.get('payload') or candidate_receipt.get('functional_bindings')!=direct_receipt.get('functional_bindings') or candidate_receipt.get('mass_geometry_accounting')!=direct_receipt.get('mass_geometry_accounting'):
    raise ValueError('FHL receipt changed organ payload, functional bindings, or mass accounting')
if candidate_receipt.get('payload')!=base_receipt.get('payload') or candidate_receipt.get('functional_bindings')!=base_receipt.get('functional_bindings') or candidate_receipt.get('mass_geometry_accounting')!=base_receipt.get('mass_geometry_accounting'):
    raise ValueError('FHL receipt differs from 1218 baseline in organ payload/bindings/mass accounting')
if candidate_receipt['provenance'].get('skin_visual_binding_candidate') != base_receipt['provenance'].get('skin_visual_binding_candidate'):
    raise ValueError('FHL receipt does not retain the current ec5664 skin identity')
if base_receipt['provenance']['skin_visual_binding_candidate'].get('payload_sha256')!='ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7':
    raise ValueError('1218 skin candidate identity changed')
if candidate_report.get('binding_table_byte_exact') is not True: raise ValueError('candidate did not preserve the binding table')
if candidate_report.get('accepted_pose_forward_status') not in (None,'not_run'):
    raise ValueError('unexpected native pose qualification claim')

# Ask the existing Human owner to resolve every actual native input and native argv, without launching.
sys.path.insert(0,str(HUMAN/'src'))
from numilab_human.resting_run import command as owner_command
args=argparse.Namespace(body_scene=SCENE,anatomy_receipt=CAND_RECEIPT,
    tendon=Path('/Users/n/numi-human-resting-evidence-20261005/tendon-semantic-foot-migration-834/numi-human-tendon-attachments.nhtendon'),
    build=BUILD,lab=LAB,output=OUT,seconds=10.0,dt=.002,dimension=512,
    mechanics_only=False,postural_activation_cap=.01,release_initialization=True,
    contoured_bed=False,upper_passive_joints=False,hip_capsule_reference=False,
    hip_capsule_scale=None,rigid_hands=True,contact_iterations=64,drive_intervention=None,
    circulation=Path('/Users/n/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json'),
    respiration=Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-reference-respiration.json'))
native_argv,owner_assets=owner_command(args)

# Copy the exact closed-run owner launcher and derive its new invocation from the closed 310 s declaration.
d= dict(parent)
d['created_utc']=dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone.utc).isoformat().replace('+00:00','Z')
d['parent_declaration']={'path':str(PARENT_DECL),'sha256':sha(PARENT_DECL),'bytes':PARENT_DECL.stat().st_size}
d['parent_execution']={'path':str(PARENT_EXEC),'sha256':sha(PARENT_EXEC),'bytes':PARENT_EXEC.stat().st_size}
d['parent_native_metadata']={'path':str(PARENT_NATIVE),'sha256':sha(PARENT_NATIVE),'bytes':PARENT_NATIVE.stat().st_size}
d['parent_1217_physical_assets_preserved']=False
d['body_scene']=str(SCENE)
d['anatomy_receipt']=str(CAND_RECEIPT)
d['accepted_steps']=5000
d['seconds']=10
d['dt']=.002
d['capture_steps']=[0,5000]
d['capture_schedule']=[{'class':'initial','requested_time_s':0.0,'step':0},{'class':'terminal','requested_time_s':10.0,'step':5000}]
d['contact_iterations']=64
d['inspection_period_seconds']=8.0
argv=list(parent['argv'])
changes=[]
for env,value in [('NUMI_BUILD_DIR',str(BUILD)),('NUMI_HUMAN_ROOT',str(HUMAN)),
                  ('NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT',str(OUT/'common-field-failure.json')),
                  ('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS','0,5000')]:
    old,new=replace_env(argv,env,value)
    if old!=new: changes.append({'setting':env,'old':old,'new':new})
for opt,value in [('--anatomy-receipt',str(CAND_RECEIPT)),('--output',str(OUT)),('--seconds','10')]:
    old,new=replace_option(argv,opt,value)
    if old!=new: changes.append({'setting':opt,'old':old,'new':new})
if '--drive-intervention' in argv: raise ValueError('unexpected drive intervention in control arm')
if '--rigid-hands' not in argv or '--release-initialization' not in argv: raise ValueError('missing baseline release/rigid-hands setting')

# Verify the remaining physical/control arguments are identical to the 310 s parent.
def get_option(a,o):
    i=a.index(o); return a[i+1]
for opt in ('--body-scene','--tendon','--circulation','--respiration','--dt','--dimension','--postural-activation-cap','--contact-iterations','--inspection-period-seconds'):
    if get_option(argv,opt)!=get_option(parent['argv'],opt): raise ValueError(f'physical/control argument changed: {opt}')
if '--release-initialization' not in argv or '--rigid-hands' not in argv: raise ValueError('launch flags missing')
if 'NUMI_HUMAN_RESTING_TRANSACTION_PROBE=1' not in argv: raise ValueError('transaction probe setting changed')
if get_option(argv,'--body-scene')!=str(SCENE): raise ValueError('scene path changed')
if get_option(argv,'--seconds')!='10': raise ValueError('duration not 10 seconds')
if any(x.startswith('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=') and x!='NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=0,5000' for x in argv): raise ValueError('capture schedule mismatch')

# Bind all current owner inputs, composition proof inputs, source/runtime identities, and prior-lineage references.
assets=dict(parent['immutable_assets'])
# Remove the old actively selected build/runtime records; the parent declaration remains separately pinned above.
obsolete_prefixes=(
 '/Users/n/numi-human-performance-build-014/',
 '/Users/n/numi-human-retired-alias-visibility-build-018-attempt2/',
 '/Users/n/numi-human-retired-alias-visibility-018/apps/NumiHumanAcceptedGeometryCadence.hpp',
)
assets={p:h for p,h in assets.items() if not any(p.startswith(prefix) for prefix in obsolete_prefixes)}
for p,h in owner_assets.items():
    if p in assets and assets[p]!=h: raise ValueError(f'parent pin differs for actual current owner input {p}')
    assets[p]=h
for rel in ('lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/MetalRoboHyperPolicy.metallib',
            'shaders/NumiNeuron.metallib','matter/shaders/HumanRespiration.metallib',
            'matter/shaders/NumiMatter.metallib','matter/shaders/NumiMatterPhysicalStateDigest.metallib'):
    p=(BUILD/rel).resolve()
    if not p.is_file(): raise FileNotFoundError(p)
    assets[str(p)]=sha(p)
for p in [CAND_PAYLOAD,CAND_MANIFEST,CAND_REPORT,CAND_RECEIPT,E/'fhl-current-7b23-compose-004/command.json',
          E/'fhl-current-7b23-compose-001/fhl-source-seam-correction.json',
          E/'regen-4rows-001/fhl-row-patches-001/row-27.npz',E/'regen-4rows-001/fhl-row-patches-001/row-28.npz',
          E/'regen-4rows-001/fhl-row-patches-001/row-patch-preparation.json',
          E/'regen-4rows-001/regenerated-row-comparison-004.json',
          E/'regen-4rows-001/regenerated-row-exact-f32-self-audit-001.json',
          E/'source-selection-comparison.json',E/'regen-4rows-001/declaration.json',
          E/'fhl-current-7b23-compose-001/command-provenance-correction-001.json',
          DIRECT_PARENT_PAYLOAD,DIRECT_PARENT_MANIFEST,DIRECT_PARENT_RECEIPT,
          HUMAN/'src/numilab_human/resting_run.py',LAB/'tools/numi',LAB/'numi/commands/human-resting']:
    p=Path(p).resolve()
    if not p.is_file(): raise FileNotFoundError(p)
    assets[str(p)]=sha(p)
# Source git revisions plus clean-tree checks are part of the actual execution identity.
human_source=git_identity(HUMAN)
laboratory=git_identity(LAB)
build_source=git_identity(BUILD_SOURCE)
for p,h in assets.items():
    actual=sha(Path(p))
    if actual!=h: raise ValueError(f'input changed during preparation: {p}')

# The command helper's result is the exact physical native invocation; keep it beside the wrapper declaration.
d['argv']=argv
d['immutable_assets']=dict(sorted(assets.items()))
d['source_candidate']={
 'direct_parent_7b23':{'payload_path':str(DIRECT_PARENT_PAYLOAD),'payload_sha256':sha(DIRECT_PARENT_PAYLOAD),
                      'manifest_path':str(DIRECT_PARENT_MANIFEST),'manifest_sha256':sha(DIRECT_PARENT_MANIFEST),
                      'receipt_path':str(DIRECT_PARENT_RECEIPT),'receipt_sha256':sha(DIRECT_PARENT_RECEIPT)},
 'fhl_child':{'payload_path':str(CAND_PAYLOAD),'payload_sha256':sha(CAND_PAYLOAD),
              'manifest_path':str(CAND_MANIFEST),'manifest_sha256':sha(CAND_MANIFEST),
              'receipt_path':str(CAND_RECEIPT),'receipt_sha256':sha(CAND_RECEIPT),
              'composition_report_path':str(CAND_REPORT),'composition_report_sha256':sha(CAND_REPORT),
              'changed_stable_ids_vs_direct_parent':[27,28],
              'exact_f32_self_unallowed_pairs':candidate_manifest['source']['fhl_source_seam_correction']['exact_f32_self_unallowed_pairs'],
              'native_pose_qualification':'pending; this declaration has not been launched'},
 'ehl_rows_23_24':'unchanged and excluded; retained source evidence reports persistent exact self pairs; no EHL correction is included'}
d['runtime_identity']={
 'build_directory':str(BUILD),'binary_path':str(BUILD/'bin/numi-human-native'),'binary_sha256':sha(BUILD/'bin/numi-human-native'),
 'library_path':str((BUILD/'lib/libmetalrobo.dylib').resolve()),'library_sha256':sha((BUILD/'lib/libmetalrobo.dylib').resolve()),
 'human_runtime_source':human_source,'native_source_tree':build_source,'lab_owner_cli_source':laboratory,
 'human_resting_run_py':{'path':str(HUMAN/'src/numilab_human/resting_run.py'),'sha256':sha(HUMAN/'src/numilab_human/resting_run.py')},
 'lab_numi_wrapper':{'path':str(LAB/'tools/numi'),'sha256':sha(LAB/'tools/numi')},
 'lab_human_resting_command':{'path':str(LAB/'numi/commands/human-resting'),'sha256':sha(LAB/'numi/commands/human-resting')},
 'active_runtime_artifacts':{str(BUILD/rel):sha((BUILD/rel).resolve()) for rel in (
    'lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/MetalRoboHyperPolicy.metallib',
    'shaders/NumiNeuron.metallib','matter/shaders/HumanRespiration.metallib',
    'matter/shaders/NumiMatter.metallib','matter/shaders/NumiMatterPhysicalStateDigest.metallib')},
 'runtime_change_from_310s_parent':'The fixed q-integration-publish build002 and clean Human q-audit-provenance source checkout are selected for this short diagnostic; this is not a runtime-identical comparison to the 310 s baseline.'}
d['owner_resolved_native_argv']=native_argv
d['physical_scope']={
 'argv_changes':changes,
 'same_body_scene_and_flat_support':True,
 'same_ec5664_skin_sha256':'ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7',
 'same_72kg_reference_mass_and_organs_functional_bindings':True,
 'same_tendon_circulation_respiration_dt_dimension_activation_cap_release_hands_contact_iterations_and_inspection_period':True,
 'same_current_physiological_cli_inputs_as_310s_baseline':True,
 'only_source_geometry_change_vs_direct_7b23_parent':'FHL stable IDs 27 and 28; prior biceps source correction is inherited unchanged.',
 'cumulative_nhtiss_difference_vs_1218_310s_baseline':'The immediate source is 7b23, which already carries its prior biceps 103/104 correction; this FHL candidate adds 27/28 relative to that source. Do not attribute the cumulative difference from 1cd/1218 baseline solely to FHL.',
 'duration_change_s':{'from':310,'to':10},'capture_steps':{'from':parent['capture_steps'],'to':[0,5000]},
 'launch_status':'prepared only; not launched','no_complete_breath_capture_claim':'Only step 0 and terminal step 5000 packs are declared; continuous physiology traces remain available if the owner succeeds.'}
d['allowed_differences_from_parent']=[
 'FHL direct-child 7b23 NHTISS4 payload and receipt, changing only stable IDs 27/28 versus that immediate parent; inherited biceps 103/104 lineage remains intact.',
 'Fixed native build002 and clean Human source root used for this diagnostic; runtime identity is therefore not identical to the 310 s parent.',
 'Fresh native output and common-field failure receipt paths.',
 '10.0 simulated seconds / 5000 accepted steps instead of 310.0 / 155000.',
 'Accepted geometry captures only at steps 0 and 5000.'
]
DEST.mkdir(parents=True)
shutil.copy2(PARENT_DIR/'run.py',RUNNER)
shutil.copy2(Path(__file__),PREPARER)
d['run_py']={'path':str(RUNNER),'sha256':sha(RUNNER)}
(DEST/'run-declaration.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
report={
 'schema':'numi.human.fhl-native-10s-preparation-report.v1','status':'prepared_not_launched',
 'parent_declaration':{'path':str(PARENT_DECL),'sha256':sha(PARENT_DECL)},
 'parent_closed_success':True,'candidate_receipt_direct_parent':{'path':str(DIRECT_PARENT_RECEIPT),'sha256':sha(DIRECT_PARENT_RECEIPT)},
 'candidate_payload':{'path':str(CAND_PAYLOAD),'sha256':sha(CAND_PAYLOAD)},
 'candidate_receipt':{'path':str(CAND_RECEIPT),'sha256':sha(CAND_RECEIPT)},
 'source_test_hashes':{'composer':sha(HUMAN.parent/'numi-human-fhl-source-composition-1248/src/numilab_human/passive_attachment_composition.py'),
                       'focused_test':sha(HUMAN.parent/'numi-human-fhl-source-composition-1248/tests/test_passive_attachment_composition.py')},
 'owner_resolved_native_argv':native_argv,'owner_resolved_assets':dict(sorted(owner_assets.items())),
 'wrapper_argv':argv,'changed_settings':changes,'immutable_asset_count':len(assets),
 'test_command':'PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python3.13 -m pytest -q tests/test_passive_attachment_composition.py',
 'limitations':['No native run has been launched.','Two endpoint packs do not qualify full-cycle anatomy.','EHL 23/24 remain unchanged and excluded.','Build002 differs from the 310 s baseline runtime.']}
(DEST/'preparation-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'declaration':str(DEST/'run-declaration.json'),'declaration_sha256':sha(DEST/'run-declaration.json'),
                  'report':str(DEST/'preparation-report.json'),'report_sha256':sha(DEST/'preparation-report.json'),
                  'run_py_sha256':sha(RUNNER),'prep_script_sha256':sha(PREPARER),
                  'immutable_asset_count':len(assets),'native_argv':native_argv,'wrapper_argv':argv},indent=2))
