from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path('/Users/n/numi-human-retained-delivery-20261009')
PARENT_DIR = ROOT / 'skin-resting-multipose-clearance-1218/native-baseline-310s-preparation'
PARENT_DECL = PARENT_DIR / 'run-declaration.json'
PARENT_EXEC = PARENT_DIR / 'execution.json'
PARENT_NATIVE = PARENT_DIR / 'native-run/run-metadata.json'
PARENT_RECEIPT = ROOT / 'skin-resting-multipose-clearance-1218/package-preparation-002/composed-candidate/composed-anatomy/resting-anatomy-receipt.json'
PARENT_SCENE = ROOT / 'skin-resting-multipose-clearance-1218/package-preparation-002/composed-candidate/resting-supine-scene.manifest.json'
PARENT_SKIN = ROOT / 'skin-resting-multipose-clearance-1218/package-preparation-002/composed-candidate/bodyparts3d-myosim-skinned-shell.nhskin'
PARENT_SKIN_MANIFEST = ROOT / 'skin-resting-multipose-clearance-1218/package-preparation-002/composed-candidate/common-atlas-skin-geometry-registration.manifest.json'
CAND = ROOT / 'passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003'
CAND_BUILD = ROOT / 'passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003-build.py'
CAND_INPUTS = ROOT / 'passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003-inputs'
OLD_CAND = ROOT / 'passive-biceps-micro-overlap-1225/compose-current-1cd-attempt002'
HUMAN = Path('/Users/n/numi-human-free-apex-two-family-1178')
DEST = ROOT / 'passive-biceps-micro-overlap-1225/native-biceps-10s-attempt001'
OUT = DEST / 'native-run'
SCOPE = DEST / 'attempt-002-scope-correction.json'
DECL = DEST / 'run-declaration.json'
RUNNER = DEST / 'run.py'
SELF = DEST / 'prepare_native_biceps_10s.py'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha(path)


def jload(path: Path):
    return json.loads(path.read_text())

if DEST.exists():
    allowed_existing = {SELF.name, RUNNER.name, SCOPE.name}
    unexpected = sorted(p.name for p in DEST.iterdir() if p.name not in allowed_existing)
    if unexpected:
        raise FileExistsError(f'refusing nonempty preparation directory: {unexpected}')
else:
    DEST.mkdir(parents=True)

parent = jload(PARENT_DECL)
parent_exec = jload(PARENT_EXEC)
parent_receipt = jload(PARENT_RECEIPT)
child_receipt_path = CAND / 'resting-anatomy-receipt.json'
child_receipt = jload(child_receipt_path)
compose_exec = jload(CAND / 'execution.json')
compose_report = jload(CAND / 'report.json')
old_receipt_path = OLD_CAND / 'resting-anatomy-receipt.json'
old_receipt = jload(old_receipt_path)

assert sha(PARENT_DECL) == '5319d47c6c2fd9fa0a581edbb02b07e6e6f18301407b060e6b52d13a5169f90f'
assert parent_exec.get('returncode') == 0 and parent_exec.get('changed_inputs') == {}
assert sha(PARENT_RECEIPT) == 'ebfb61b926f33dd6497176734324e7208e1eeab2f2b518e6d4f884e0a8da99ec'
assert parent_receipt['provenance']['skin_visual_binding_candidate']['payload_sha256'] == 'ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7'
assert child_receipt['payload'] == parent_receipt['payload']
assert child_receipt['functional_bindings'] == parent_receipt['functional_bindings']
assert child_receipt['mass_geometry_accounting'] == parent_receipt['mass_geometry_accounting']
assert child_receipt['qualification'] == parent_receipt['qualification']
assert child_receipt['thorax_source_volume_m3'] == parent_receipt['thorax_source_volume_m3']
assert child_receipt['derived_lobe_overlap_partition'] == parent_receipt['derived_lobe_overlap_partition']
assert child_receipt['provenance']['skin_visual_binding_candidate'] == parent_receipt['provenance']['skin_visual_binding_candidate']
assert child_receipt['provenance']['native_muscle_surfaces']['sha256'] == compose_exec['output_payload_sha256'] == '7b23d0daf2eb73221944389716d01c8d9f2a9ebfb86dd815f605c8a45e54bbc9'
assert child_receipt['provenance']['native_muscle_surfaces']['manifest_sha256'] == compose_exec['output_manifest_sha256']
assert child_receipt['provenance']['biceps_source_preserving_correction_binding']
assert compose_exec['source_receipt_sha256'] == sha(PARENT_RECEIPT)
assert compose_exec['output_receipt_sha256'] == sha(child_receipt_path)
assert compose_exec['changed_stable_ids'] == [103, 104]
assert compose_exec['binding_table_byte_exact'] is True
assert compose_exec['other_row_vertex_bytes_match_current_parent'] is True
assert compose_exec['row_103_104_bytes_match_source_candidate'] is True
assert compose_exec['receipt_owner_matches_output'] is True
assert compose_exec['accepted_pose_forward_status'] == 'not_run'
assert compose_report['source_composition']['changed_stable_ids'] == [103, 104] if 'source_composition' in compose_report else True
assert sha(CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue') == sha(OLD_CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
assert old_receipt['provenance']['native_muscle_surfaces']['sha256'] == compose_exec['output_payload_sha256']
assert old_receipt['provenance']['native_muscle_surfaces']['manifest_sha256'] != compose_exec['output_manifest_sha256'] or old_receipt['provenance']['native_muscle_surfaces']['manifest_path'] != compose_exec['output_manifest_path']

# Only these five argv positions may differ from the closed 310 s control.
old_argv = list(parent['argv'])
new_argv = list(old_argv)
new_output = str(OUT)
new_failure = str(OUT / 'common-field-failure.json')
new_captures = '0,5000'
old_fail = next(x for x in old_argv if x.startswith('NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT='))
old_cap = next(x for x in old_argv if x.startswith('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS='))
new_argv[old_argv.index(old_fail)] = 'NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=' + new_failure
new_argv[old_argv.index(old_cap)] = 'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=' + new_captures

def option_value(argv, option):
    i = argv.index(option)
    return i + 1, argv[i + 1]

for flag, value in [('--anatomy-receipt', str(child_receipt_path)), ('--output', new_output), ('--seconds', '10')]:
    idx, _ = option_value(new_argv, flag)
    new_argv[idx] = value
changes = []
for i, (old, new) in enumerate(zip(old_argv, new_argv)):
    if old != new:
        if old.startswith('NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT='):
            setting = 'NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT'
        elif old.startswith('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS='):
            setting = 'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS'
        elif i and old_argv[i - 1] == '--anatomy-receipt':
            setting = '--anatomy-receipt'
        elif i and old_argv[i - 1] == '--output':
            setting = '--output'
        elif i and old_argv[i - 1] == '--seconds':
            setting = '--seconds'
        else:
            raise AssertionError(f'unapproved argv delta at {i}: {old!r} -> {new!r}')
        changes.append({'argv_index': i, 'setting': setting, 'old': old, 'new': new})
assert {x['setting'] for x in changes} == {
    'NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT', 'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS',
    '--anatomy-receipt', '--output', '--seconds'
}
assert len(changes) == 5
assert option_value(new_argv, '--body-scene')[1] == str(PARENT_SCENE)
assert option_value(new_argv, '--tendon')[1] == option_value(old_argv, '--tendon')[1]
assert option_value(new_argv, '--circulation')[1] == option_value(old_argv, '--circulation')[1]
assert option_value(new_argv, '--respiration')[1] == option_value(old_argv, '--respiration')[1]
assert option_value(new_argv, '--dt')[1] == '0.002'
assert option_value(new_argv, '--contact-iterations')[1] == '64'
assert option_value(new_argv, '--postural-activation-cap')[1] == '0.01'
assert '--release-initialization' in new_argv and '--rigid-hands' in new_argv and '--inspection-tour' in new_argv
assert option_value(new_argv, '--inspection-period-seconds')[1] == '8.0'
assert parent['body_scene'] == str(PARENT_SCENE)
assert not OUT.exists()

shutil.copy2(PARENT_DIR / 'run.py', RUNNER)
# Preserve the exact preparation program as the reproducible preflight source.
if Path(__file__).resolve() != SELF.resolve():
    shutil.copy2(Path(__file__), SELF)

scope_note = {
    'schema': 'numi.human.biceps-attempt-scope-correction.v1',
    'status': 'attempt-002-composition-retained-but-not-current-scene-bound',
    'attempt_002': {
        'payload_path': str(OLD_CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'),
        'payload_sha256': sha(OLD_CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'),
        'receipt_path': str(old_receipt_path),
        'receipt_sha256': sha(old_receipt_path),
        'receipt_source_parent': old_receipt['provenance']['biceps_source_preserving_correction_binding'].get('source_parent_receipt') if isinstance(old_receipt['provenance'].get('biceps_source_preserving_correction_binding'), dict) else None,
        'scope': 'The payload is the same checked biceps composition as attempt003; its receipt lineage is not bound to the intended current1218 scene receipt. This is an incompatibility for that intended run, not a claim that the composition itself is invalid.'
    },
    'attempt_003': {
        'payload_path': str(CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'),
        'payload_sha256': sha(CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'),
        'receipt_path': str(child_receipt_path),
        'receipt_sha256': sha(child_receipt_path),
        'source_receipt_path': str(PARENT_RECEIPT),
        'source_receipt_sha256': sha(PARENT_RECEIPT),
        'functional_bindings_unchanged': child_receipt['functional_bindings'] == parent_receipt['functional_bindings'],
        'mass_geometry_accounting_unchanged': child_receipt['mass_geometry_accounting'] == parent_receipt['mass_geometry_accounting'],
        'parent_payload_identity_unchanged': child_receipt['payload'] == parent_receipt['payload'],
        'skin_sha256': child_receipt['provenance']['skin_visual_binding_candidate']['payload_sha256'],
        'scope': 'This receipt is source-derived from the exact current1218 baseline receipt and carries the biceps operation provenance.'
    }
}
SCOPE.write_text(json.dumps(scope_note, indent=2, sort_keys=True) + '\n')

# Extend the parent's immutable source set with exact child-composition and preparation inputs.
assets = dict(parent['immutable_assets'])
extra_paths = [
    PARENT_DECL, PARENT_EXEC, PARENT_NATIVE, PARENT_RECEIPT, PARENT_SCENE, PARENT_SKIN, PARENT_SKIN_MANIFEST,
    CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue',
    CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json',
    CAND / 'resting-anatomy-receipt.json', CAND / 'report.json', CAND / 'execution.json',
    CAND / 'common-cardiac-map-f32.bin', CAND / 'common-cardiac-domains-f32.bin', CAND / 'common-cardiac-volumes-f32.bin',
    CAND_BUILD, CAND_INPUTS / 'biceps-correction-input.json',
    CAND_INPUTS / 'row-103.npz', CAND_INPUTS / 'row-104.npz',
    ROOT / 'passive-biceps-micro-overlap-1225/attempt-002/source-candidate-audit.json',
    OLD_CAND / 'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue', old_receipt_path,
    HUMAN / 'src/numilab_human/passive_attachment_composition.py',
    HUMAN / 'tests/test_passive_attachment_composition.py', SCOPE, SELF, RUNNER
]
# Resolve candidate row audit paths from source report if the guessed attempt-002 locations differ.
for row in compose_report.get('changed_rows', []):
    for k in ('candidate', 'report'):
        extra_paths.append(Path(row[k]))
for p in extra_paths:
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(f'missing immutable preparation input: {p}')
    digest = sha(p)
    if str(p) in assets and assets[str(p)] != digest:
        raise AssertionError(f'baseline asset changed: {p}')
    assets[str(p)] = digest

# Create a direct-parent declaration: the closed 310 s run, not its historical 1217 ancestor.
d = dict(parent)
d['created_utc'] = dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')
d['parent_declaration'] = {'path': str(PARENT_DECL), 'sha256': sha(PARENT_DECL), 'bytes': PARENT_DECL.stat().st_size}
d['parent_execution'] = {'path': str(PARENT_EXEC), 'sha256': sha(PARENT_EXEC), 'bytes': PARENT_EXEC.stat().st_size}
d['parent_native_metadata'] = {'path': str(PARENT_NATIVE), 'sha256': sha(PARENT_NATIVE), 'bytes': PARENT_NATIVE.stat().st_size}
d.pop('parent_1217_physical_assets_preserved', None)
d['accepted_steps'] = 5000
d['seconds'] = 10
d['capture_steps'] = [0, 5000]
d['capture_schedule'] = [
    {'class': 'initial', 'requested_time_s': 0.0, 'step': 0},
    {'class': 'terminal', 'requested_time_s': 10.0, 'step': 5000},
]
d['anatomy_receipt'] = str(child_receipt_path)
d['argv'] = new_argv
d['immutable_assets'] = dict(sorted(assets.items()))
d['source_candidate'] = {
    'baseline_1218_candidate': parent['source_candidate'],
    'biceps_source_preserving_composition': {
        'parent_receipt': {'path': str(PARENT_RECEIPT), 'sha256': sha(PARENT_RECEIPT)},
        'parent_nhtiss_payload': {'path': compose_exec['parent_payload_path'], 'sha256': compose_exec['parent_payload_sha256']},
        'candidate_payload': {'path': compose_exec['output_payload_path'], 'sha256': compose_exec['output_payload_sha256']},
        'candidate_manifest': {'path': compose_exec['output_manifest_path'], 'sha256': compose_exec['output_manifest_sha256']},
        'candidate_receipt': {'path': compose_exec['output_receipt_path'], 'sha256': compose_exec['output_receipt_sha256']},
        'composition_report': {'path': compose_exec['output_composition_report_path'], 'sha256': compose_exec['output_composition_report_sha256']},
        'composition_execution': {'path': str(CAND / 'execution.json'), 'sha256': sha(CAND / 'execution.json')},
        'composition_runner': {'path': compose_exec['runner_path'], 'sha256': compose_exec['runner_sha256']},
        'composition_owner': {'path': compose_exec['composer_path'], 'sha256': compose_exec['composer_sha256']},
        'changed_stable_ids': [103, 104],
        'binding_table_byte_exact': True,
        'other_rows_byte_exact': True,
        'row_103_104_match_source_candidates': True,
        'accepted_pose_forward_status': 'not_run',
        'scope': 'Source composition and lineage only; no native pose/contact clearance qualification.'
    }
}
d['physical_scope'] = {
    'argv_changes': changes,
    'same_body_scene': True,
    'same_current_skin_payload_sha256': 'ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7',
    'same_flat_bed_and_support_witnesses': True,
    'same_respiration_circulation_tendon_and_brain_cli_inputs': True,
    'same_release_initialization_rigid_hands_contact_iterations_activation_cap_and_dt': True,
    'same_native_program_runtime_environment_and_physics_assets': True,
    'only_anatomy_input_change': 'The exact 1218 receipt is replaced by attempt003 receipt, whose NHTISS4 payload differs only at stable IDs 103/104; receipt payload, functional bindings, mass accounting, skin, and scene identity match the baseline receipt.',
    'duration_change': {'from_s': 310, 'to_s': 10, 'from_steps': 155000, 'to_steps': 5000},
    'capture_change': {'from_steps': parent['capture_steps'], 'to_steps': [0, 5000]},
    'failure_receipt_and_native_output_are_fresh': True,
    'inspection_tour_period_s_unchanged': 8.0
}
d['allowed_differences_from_parent'] = [
    'Attempt003 current1218-derived anatomy receipt and its NHTISS4 payload/manifest; only stable IDs 103 and 104 differ from the 1218 NHTISS parent, with bindings, functional anatomy, mass accounting, skin, and body scene unchanged.',
    'Fresh native output, common-field failure-receipt, and derived movie paths.',
    '10.0 simulated seconds / 5000 accepted steps instead of 310.0 seconds / 155000 steps.',
    'Accepted geometry captures at initial step 0 and terminal step 5000.'
]
d['qualification_boundary'] = 'Prepared unlaunched 10 s full-scene smoke. It tests this biceps source correction in the existing native scene; it is not a long-horizon anatomy, contact, physiology, or clinical qualification.'
d['launch_status'] = 'not launched; prepared declaration only'
d['status'] = 'prepared_unlaunched'
# Embed preflight facts directly in the declaration so its hash binds the reviewed checks.
d['preparation_preflight'] = {
    'status': 'passed',
    'baseline_execution_successful_and_unchanged': True,
    'attempt003_parent_receipt_exact': True,
    'payload_functional_binding_mass_and_skin_identity_checks_passed': True,
    'attempt003_source_composition_claims': {k: compose_exec[k] for k in ['changed_stable_ids','binding_table_byte_exact','other_row_vertex_bytes_match_current_parent','row_103_104_bytes_match_source_candidate','receipt_owner_matches_output','accepted_pose_forward_status']},
    'unapproved_argv_differences': [],
    'argv_change_count': len(changes),
    'fresh_output_absent_at_prepare_time': True,
    'immutable_asset_count': len(assets),
    'attempt002_scope_correction': {'path': str(SCOPE), 'sha256': sha(SCOPE)},
}
DECL.write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')

# Reopen and independently check serialized declaration and every bound input before handoff.
final = jload(DECL)
assert final['argv'] == new_argv and final['accepted_steps'] == 5000
assert not OUT.exists()
for path, expected in final['immutable_assets'].items():
    if sha(Path(path)) != expected:
        raise AssertionError(f'pin mismatch: {path}')
# Exact five raw argv entries, no hidden physical/runtime switch.
actual_diff = [(i, a, b) for i, (a, b) in enumerate(zip(parent['argv'], final['argv'])) if a != b]
assert len(actual_diff) == 5
assert {changes[i]['argv_index'] for i in range(len(changes))} == {i for i, _, _ in actual_diff}

print(json.dumps({
    'status': 'prepared_unlaunched',
    'directory': str(DEST),
    'declaration': {'path': str(DECL), 'sha256': sha(DECL), 'bytes': DECL.stat().st_size},
    'runner': {'path': str(RUNNER), 'sha256': sha(RUNNER)},
    'preparation_script': {'path': str(SELF), 'sha256': sha(SELF)},
    'scope_correction': {'path': str(SCOPE), 'sha256': sha(SCOPE)},
    'parent': {'declaration_sha256': sha(PARENT_DECL), 'execution_sha256': sha(PARENT_EXEC), 'native_metadata_sha256': sha(PARENT_NATIVE)},
    'candidate': {'payload_sha256': sha(CAND/'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'), 'manifest_sha256': sha(CAND/'bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'), 'receipt_sha256': sha(child_receipt_path)},
    'argv_changes': changes,
    'immutable_asset_count': len(assets),
    'launch_command': f'cd {DEST} && python3 run.py',
    'output_absent': not OUT.exists(),
}, indent=2, sort_keys=True))
