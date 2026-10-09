#!/usr/bin/env python3
"""Read-only source/provenance preflight for the 1218 per-vertex reduction proposal.

This script verifies the frozen 1217/1216 inputs, the failed fit001/heldout002
ancestry, and the separately generated local-self-reduction-004 proposal. It
never writes anatomy assets and never treats proposal feasibility as an exact
geometry pass. The nine-pose exact audit is a separate required gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

R = Path('/Users/n/numi-human-retained-delivery-20261009')
E = Path('/Users/n/numi-human-resting-evidence-20261005')
ROOT = Path('/Users/n/numi-human-conforming-composition-source-1216')
sys.path.insert(0, str(ROOT / 'src'))
PKG = R / 'skin-resting-multipose-clearance-1218'
PARENT = R / 'muscle-conforming-refinement-1216/native-refinement-1217-attempt2'
FIT = PKG / 'fit-attempt-001'
HOLDOUT = PKG / 'heldout-002'
QP = PKG / 'local-self-reduction-004'
REPLAY = PKG / 'local-self-reduction-005'
AUDIT = PKG / 'local-self-reduction-audit-001/scan-001'

EXPECTED = {
    'owner_commit': '2eace39df60d72ba1197a775aea6ca4205d6e484',
    'owner_revision_sha256': '51e3a732e18fbdb6922871b435d98f42d01c00511d2d08a0fc4cac1009faba6f',
    'parent_declaration_sha256': 'a89a7fdfef2751b00db90d1b8f0332176ef4aea4981bcaa936b10ad71531c12b',
    'parent_execution_sha256': '4c5665166e3084546863b6aa6c48902f483c34d40a9f5e3982d3b0c25961ea69',
    'parent_receipt_sha256': '10ecac382b42e08b8ba296f361d419a0a7db1fb6e458448596e127acbea14d62',
    'parent_nhtiss_sha256': '1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48',
    'parent_nhtiss_manifest_sha256': 'f052eff736f040f2f6ca1a1a99d2939ca0ed5c0bead055dfb39d3141f05a3cac',
    'fit001_report_sha256': 'c497fcc209ba183761cf97425bc060ee91fa7767b98234e433dd0d26f8f05265',
    'fit001_positions_sha256': '48464230377c811f2cb313a13a8dbcbf7b50d008c2cec18fe4ef59f258d92e35',
    'fit001_position_data_sha256': '71fec6be6c6074ae4e3fb6c681941c884fd19a12ce337fac81207036f8fc44d7',
    'qp_feasibility_sha256': 'dd053cb43d5e992a36385baa0552573ffed84e70cb42fb3025ed21790a57afc6',
    'qp_proposal_sha256': '2a7b416c166b034030c84ac5098dcb5aa6735573726e250e92bdd7a21d2ab258',
    'replay_report_sha256': '3d679bd05d9a215347cf50c369eaebc31a614808bb9916bc179f0f2e89dbf03c',
    'replay_script_sha256': 'eda511bd8fd49fe5c7e997180c4afbea15fd0bf5a03b4f43948633c958550b8c',
    'heldout_summary_sha256': '9a5017bda14540d8e15aa40f682f8eb112336d58529a46eaa7717d6c3672dc19',
    'heldout_binding_sha256': '40fbb14542007eb3f1f27a366fe9ac0a175695377136722ccd458d0d9d70ce6b',
}

def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()

def pin(path: Path) -> dict:
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}

def load(path: Path) -> dict:
    need(path.is_file() and not path.is_symlink(), f'missing or symlinked JSON: {path}')
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f'expected JSON object: {path}')
    return value

def verify(path: Path, expected: str, label: str) -> dict:
    need(path.is_file() and not path.is_symlink(), f'missing or symlinked {label}: {path}')
    actual = sha(path)
    need(actual == expected, f'{label} SHA-256 changed: {actual}')
    return pin(path)

def check_hash_map(mapping: dict, label: str) -> list[dict]:
    need(isinstance(mapping, dict) and mapping, f'{label} has no pinned file map')
    checked = []
    for raw, digest in sorted(mapping.items()):
        path = Path(raw)
        need(path.is_file() and not path.is_symlink(), f'{label} input missing: {path}')
        actual = sha(path)
        need(actual == digest, f'{label} input changed: {path}')
        checked.append(pin(path))
    return checked

def preflight() -> dict:
    rev_path = ROOT / 'source-revision.json'
    owner_rev = verify(rev_path, EXPECTED['owner_revision_sha256'], 'frozen owner source-revision')
    revision = load(rev_path)
    need((revision.get('commit') or revision.get('revision') or revision.get('git_commit')) == EXPECTED['owner_commit'], 'frozen owner commit mismatch')

    declaration_path = PARENT / 'run-declaration.json'
    execution_path = PARENT / 'execution.json'
    parent_decl_pin = verify(declaration_path, EXPECTED['parent_declaration_sha256'], '1217 declaration')
    parent_exec_pin = verify(execution_path, EXPECTED['parent_execution_sha256'], '1217 execution')
    declaration = load(declaration_path)
    execution = load(execution_path)
    need(execution.get('returncode') == 0 and execution.get('changed_inputs') == {}, '1217 native run was unsuccessful or changed inputs')
    need(execution.get('declaration_sha256') == parent_decl_pin['sha256'], '1217 execution does not bind its declaration')
    need(declaration.get('capture_steps') == [0, 9983, 20000], '1217 accepted capture steps changed')
    parent_assets = check_hash_map(declaration.get('immutable_assets'), '1217 immutable assets')
    candidate = declaration.get('candidate', {})
    receipt_path = Path(candidate.get('candidate_receipt', ''))
    verify(receipt_path, EXPECTED['parent_receipt_sha256'], 'refined 1217 anatomy receipt')
    receipt = load(receipt_path)
    muscle = receipt.get('provenance', {}).get('native_muscle_surfaces', {})
    need(muscle.get('sha256') == EXPECTED['parent_nhtiss_sha256'], 'current refined NHTISS payload binding changed')
    need(muscle.get('manifest_sha256') == EXPECTED['parent_nhtiss_manifest_sha256'], 'current refined NHTISS manifest binding changed')
    nhtiss_path = Path(muscle.get('payload_path', ''))
    nhtiss_manifest_path = Path(muscle.get('manifest_path', ''))
    nhtiss_pin = verify(nhtiss_path, EXPECTED['parent_nhtiss_sha256'], 'refined NHTISS payload')
    nhtiss_manifest_pin = verify(nhtiss_manifest_path, EXPECTED['parent_nhtiss_manifest_sha256'], 'refined NHTISS manifest')
    scene_path = Path(declaration['body_scene'])
    scene = load(scene_path)
    skin = scene.get('source', {}).get('skin', {})
    base_skin = Path(receipt['mass_geometry_accounting']['skin_payload_path'])
    need(skin.get('path') == str(base_skin) and skin.get('sha256') == receipt['mass_geometry_accounting']['skin_payload_sha256'], '1217 scene no longer binds the retained 1187 NHSKIN')
    base_skin_pin = verify(base_skin, skin['sha256'], 'base 1187 NHSKIN')
    base_manifest = Path(receipt['provenance']['skin_visual_binding_candidate']['manifest_path'])
    base_manifest_pin = verify(base_manifest, receipt['provenance']['skin_visual_binding_candidate']['manifest_sha256'], 'base 1187 NHSKIN manifest')

    fit_report_path = FIT / 'candidate-report.json'
    fit_positions_path = FIT / 'candidate-source-positions-f32.npy'
    fit_report_pin = verify(fit_report_path, EXPECTED['fit001_report_sha256'], 'fit001 ancestry report')
    fit_positions_pin = verify(fit_positions_path, EXPECTED['fit001_positions_sha256'], 'fit001 ancestry positions')
    fit_report = load(fit_report_path)
    need(fit_report.get('status') == 'inferred_engineering_clearance_candidate_pending_native_replay', 'fit001 historical status changed')
    need(fit_report.get('candidate_source_positions_f32_sha256') == EXPECTED['fit001_position_data_sha256'], 'fit001 position data binding changed')
    fit_positions = np.load(fit_positions_path, allow_pickle=False)
    need(fit_positions.dtype == np.dtype('<f4') and fit_positions.shape == (54949, 3) and np.isfinite(fit_positions).all(), 'fit001 ancestry position layout invalid')
    need(hashlib.sha256(fit_positions.tobytes()).hexdigest() == EXPECTED['fit001_position_data_sha256'], 'fit001 ancestry position data changed')

    qp_report_path = QP / 'feasibility.json'
    qp_proposal_path = QP / 'unadmitted-source-proposal.npy'
    qp_report_pin = verify(qp_report_path, EXPECTED['qp_feasibility_sha256'], '004 QP feasibility report')
    qp_proposal_pin = verify(qp_proposal_path, EXPECTED['qp_proposal_sha256'], '004 QP proposal')
    qp = load(qp_report_path)
    need(qp.get('status') == 'feasible_pending_exact_full_checks' and qp.get('solver_success') is True and qp.get('inputs_unchanged') is True, '004 QP status/input stability changed')
    need(qp.get('proposal', {}).get('path') == str(qp_proposal_path) and qp.get('proposal', {}).get('sha256') == qp_proposal_pin['sha256'], '004 feasibility report does not bind its proposal')
    need(qp.get('inputs_before') == qp.get('inputs_after') and isinstance(qp.get('inputs_before'), dict), '004 QP before/after input inventory differs')
    qp_inputs = check_hash_map(qp['inputs_before'], '004 QP inputs')
    replay_script_path = REPLAY / 'feasibility.py'
    replay_report_path = REPLAY / 'feasibility.json'
    replay_proposal_path = REPLAY / 'unadmitted-source-proposal.npy'
    replay_script_pin = verify(replay_script_path, EXPECTED['replay_script_sha256'], '005 dependency-complete replay script')
    replay_report_pin = verify(replay_report_path, EXPECTED['replay_report_sha256'], '005 dependency-complete replay report')
    replay = load(replay_report_path)
    need(replay.get('status') == 'feasible_pending_exact_full_checks' and replay.get('inputs_unchanged') is True, '005 replay status/input stability changed')
    replay_gate = replay.get('dependency_complete_replay', {})
    need(replay_gate.get('array_bit_exact') is True and replay_gate.get('geometry_algorithm_unchanged') is True and replay_gate.get('inherited_preflight_pins_revalidated') == 49, '005 does not establish exact dependency-complete replay')
    need(replay.get('inputs_before') == replay.get('inputs_after') and isinstance(replay.get('inputs_before'), dict), '005 replay before/after inventory differs')
    replay_inputs = check_hash_map(replay['inputs_before'], '005 replay dependency closure')
    replay_proposal_record = replay_gate.get('original004_proposal', {})
    need(replay_proposal_record.get('path') == str(qp_proposal_path) and replay_proposal_record.get('sha256') == EXPECTED['qp_proposal_sha256'], '005 replay does not bind original004 proposal')
    replay_proposal_pin = verify(replay_proposal_path, EXPECTED['qp_proposal_sha256'], '005 replayed proposal')
    replay_copy_path = Path(replay.get('proposal', {}).get('path', ''))
    replay_copy_pin = verify(replay_copy_path, EXPECTED['qp_proposal_sha256'], '005 replay output proposal')
    need(replay_copy_path == replay_proposal_path, '005 replay proposal path is unexpected')
    proposal = np.load(qp_proposal_path, allow_pickle=False)
    replay_proposal = np.load(replay_proposal_path, allow_pickle=False)
    need(np.array_equal(proposal.view(np.uint32), replay_proposal.view(np.uint32)), '005 replayed array differs from004 proposal')
    need(proposal.dtype == np.dtype('<f4') and proposal.shape == (54949, 3) and np.isfinite(proposal).all(), '004 proposal array layout invalid')
    affected = set(map(int, qp.get('affected_vertices', [])))
    need(len(affected) == len(qp.get('affected_vertices', [])) and affected, '004 affected-vertex set malformed')
    changed = np.any(proposal.view(np.uint32) != fit_positions.view(np.uint32), axis=1)
    changed_ids = set(map(int, np.flatnonzero(changed)))
    need(changed_ids and changed_ids <= affected, '004 proposal changed vertices outside the declared QP support')
    fixed = set(map(int, fit_report.get('fixed_source_vertex_ids', [])))
    need(len(fixed) == 47 and not (fixed & changed_ids), '004 proposal moves bed/thorax fixed vertices')
    need(np.array_equal(proposal[list(sorted(fixed))].view(np.uint32), fit_positions[list(sorted(fixed))].view(np.uint32)), '004 proposal changed fixed vertices relative to fit001')
    proposal_delta_m = float(np.max(np.linalg.norm(proposal - fit_positions, axis=1)))
    need(proposal_delta_m < 0.001, '004 local reduction exceeds its pinned proposal delta bound')
    from numilab_human.skin_source_payload_preflight import decode_payload
    base_decoded = decode_payload(base_skin.read_bytes())
    need(base_decoded['vertex_count'] == 54949 and base_decoded['binding_count'] == 86, 'retained 1187 NHSKIN dimensions changed')
    need(np.array_equal(proposal[list(sorted(fixed))].view(np.uint32), base_decoded['vertices_u'][list(sorted(fixed)), :3]), '004 fixed vertices differ from canonical 1187 NHSKIN')

    # Historical failure is intentionally retained; it is not an admission gate pass.
    hold_summary_path = HOLDOUT / 'scan-output/summary.json'
    hold_binding_path = HOLDOUT / 'heldout-binding.json'
    hold_summary_pin = verify(hold_summary_path, EXPECTED['heldout_summary_sha256'], 'failed heldout002 summary')
    hold_binding_pin = verify(hold_binding_path, EXPECTED['heldout_binding_sha256'], 'failed heldout002 binding')
    hold = load(hold_summary_path)
    need(hold.get('all_steps_intersection_free') is False and hold.get('status') == 'complete_with_intersections_or_invalid_geometry', 'heldout002 historical failure status changed')
    old_results = hold.get('pose_results', [])
    need([r.get('step') for r in old_results] == [4991, 5375, 5759, 6111, 6495, 7743], 'heldout002 historical steps changed')
    old_self = [r.get('skin_self_crossing_pair_count') for r in old_results]
    old_cross = [r.get('all_skin_crossing_pair_count') for r in old_results]
    need(old_self == [14, 10, 6, 0, 0, 0] and old_cross == [0, 0, 0, 0, 0, 0], 'heldout002 historical failure counts changed')

    current_audit_summary = AUDIT / 'summary.json'
    audit_state = 'not_started'
    audit_pin = None
    if current_audit_summary.exists():
        audit_pin = pin(current_audit_summary)
        audit = load(current_audit_summary)
        audit_state = str(audit.get('status', 'present_unclassified'))

    return {
        'schema': 'numi.human.skin-resting-local-reduction-004-preflight.v1',
        'status': 'recorded_sources_verified_waiting_for_nine_pose_exact_audit',
        'composition_performed': False,
        'native_run_performed': False,
        'candidate_admitted': False,
        'interpretation': '005 replays the feasible 004 QP with a dependency-complete tracked input closure and an array-bit-exact result. The proposal is not an exact geometry pass and no anatomy asset has been composed.',
        'owner_source_revision': owner_rev,
        'native_parent_1217': {'declaration': parent_decl_pin, 'execution': parent_exec_pin, 'assets': parent_assets, 'receipt': pin(receipt_path), 'nhtiss': nhtiss_pin, 'nhtiss_manifest': nhtiss_manifest_pin, 'base_nhskin': base_skin_pin, 'base_nhskin_manifest': base_manifest_pin, 'scene': pin(scene_path)},
        'fit001_source_fit': {'report': fit_report_pin, 'positions': fit_positions_pin, 'fit_pose_steps': fit_report.get('fit_pose_steps'), 'final_nonocular_pair_count_by_pose': fit_report.get('final_nonocular_pair_count_by_pose'), 'final_ocular_pair_count_by_pose': fit_report.get('final_ocular_pair_count_by_pose'), 'status': 'three_pose_exact_fit_complete_not_full_admission'},
        'local_reduction_004': {'feasibility': qp_report_pin, 'proposal': qp_proposal_pin, 'proposal_data_sha256': hashlib.sha256(proposal.tobytes()).hexdigest(), 'status': qp['status'], 'max_proposal_delta_from_fit001_m': proposal_delta_m, 'changed_vertex_count_vs_fit001': len(changed_ids), 'affected_vertex_count': len(affected), 'all_changes_within_declared_support': True, 'fixed_bed_and_thorax_vertex_count': len(fixed), 'all_recorded_qp_inputs': qp_inputs, 'dependency_complete_replay_005': {'script': replay_script_pin, 'report': replay_report_pin, 'replayed_proposal': replay_copy_pin, 'tracked_dependency_count': len(replay_inputs), 'recorded_replay_inputs': replay_inputs, 'array_bit_exact': True, 'loaded_module_files_pinned_field_note': '542 tracked dependency files; the producer field name is not interpreted as 542 modules'}},
        'heldout002_failed_ancestry': {'summary': hold_summary_pin, 'binding': hold_binding_pin, 'status': hold['status'], 'steps': [4991, 5375, 5759, 6111, 6495, 7743], 'skin_self_pair_counts': old_self, 'target_pair_counts': old_cross, 'retained_as_failed': True},
        'nine_pose_exact_audit': {'expected_root': str(AUDIT), 'summary': audit_pin, 'status': audit_state, 'required': True, 'passed': False},
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-check', action='store_true', required=True, help='verify immutable source/QP ancestry only; does not scan or compose')
    parser.parse_args()
    result = preflight()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
