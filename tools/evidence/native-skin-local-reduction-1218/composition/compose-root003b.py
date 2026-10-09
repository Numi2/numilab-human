#!/usr/bin/env python3
"""Compose the 004 skin candidate only after exact nine-pose audit and root review.

Default mode is read-only preflight. --compose additionally requires
--root-reviewed, a complete intersection-free audit, and a fresh output path.
No command in this script launches native physics.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

R = Path('/Users/n/numi-human-retained-delivery-20261009')
ROOT = Path('/Users/n/numi-human-conforming-composition-source-1216')
PKG = R / 'skin-resting-multipose-clearance-1218'
PREP = PKG / 'package-preparation-002'
QP = PKG / 'local-self-reduction-004'
REPLAY = PKG / 'local-self-reduction-005'
REPLAY6 = PKG / 'local-self-reduction-006'
AUDIT_ROOT = PKG / 'local-self-reduction-audit-001/scan-001'
AUDIT_SUMMARY = AUDIT_ROOT / 'summary.json'
AUDIT_DECLARATION = AUDIT_ROOT / 'declaration.json'
AUDIT_SCRIPT = PKG / 'local-self-reduction-audit-001/audit_local_reduction_1218_root003.py'
BASE_COMPOSER = PKG / 'package-preparation-001/compose_native_candidate_1218.py'
PREFLIGHT = PREP / 'preflight_reduction_004.py'
OWNER_REVISION = ROOT / 'source-revision.json'
EXPECTED = {
    'owner_revision': '51e3a732e18fbdb6922871b435d98f42d01c00511d2d08a0fc4cac1009faba6f',
    'base_composer': '366be36b672796888bae9acb97c45ad1c6678b250c80aa691d6b018867b042fe',
    'audit_script': 'b89702107bb320ca48415781bdd3b62b2c048d490d4b3ec2c318667977b89d5b',
    'preflight_script': '9ce7e01033f11fbbdb7d20bf6a845d2b2c854e4b2b77d931424e787c19ff0883',
    'qp_report': 'dd053cb43d5e992a36385baa0552573ffed84e70cb42fb3025ed21790a57afc6',
    'qp_array': '2a7b416c166b034030c84ac5098dcb5aa6735573726e250e92bdd7a21d2ab258',
    'replay_report': '3d679bd05d9a215347cf50c369eaebc31a614808bb9916bc179f0f2e89dbf03c',
    'replay_script': 'eda511bd8fd49fe5c7e997180c4afbea15fd0bf5a03b4f43948633c958550b8c',
    'replay_tracked_array': '2a7b416c166b034030c84ac5098dcb5aa6735573726e250e92bdd7a21d2ab258',
    'replay6_report': 'cebee14945800d8f41a55c19fcedc33d69b805dbe51c4355ea99723f04fa5e16',
    'replay6_script': '6c062737cd7247273be5877de9f394e1544b961b65b673aae7b3038eb1c71316',
    'replay6_array': '2a7b416c166b034030c84ac5098dcb5aa6735573726e250e92bdd7a21d2ab258',
}
EXPECTED_STEPS = [0, 9983, 20000, 4991, 5375, 5759, 6111, 6495, 7743]
TARGETS = 859
MARGIN_MM = 0.25
SUPPORT_EDGE_FACTOR = 4.0
SKIN_SHAPE = (54949, 3)


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
    v = json.loads(path.read_text())
    need(isinstance(v, dict), f'expected JSON object: {path}')
    return v


def verified(path: Path, expected: str, label: str) -> dict:
    need(path.is_file() and not path.is_symlink(), f'missing {label}: {path}')
    actual = sha(path)
    need(actual == expected, f'{label} changed: {actual}')
    return pin(path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f'cannot load module: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verified_context() -> tuple[dict, dict, dict, np.ndarray]:
    verified(OWNER_REVISION, EXPECTED['owner_revision'], 'frozen owner revision')
    verified(BASE_COMPOSER, EXPECTED['base_composer'], 'historical static-input validator')
    verified(PREFLIGHT, EXPECTED['preflight_script'], '004/005 preflight script')
    # The sibling preflight verifies the 1217/1216 source chain, 004 proposal,
    # dependency-complete 005 replay, fixed vertices, and failed ancestry.
    pre = load_module(PREFLIGHT, 'preflight_reduction_004_frozen')
    pre_report = pre.preflight()
    static_module = load_module(BASE_COMPOSER, 'static_1218_composer')
    parent_decl = load(static_module.PARENT_DECL)
    static = static_module.fit_inputs(parent_decl)

    qp_path = QP / 'feasibility.json'
    qp_array = QP / 'unadmitted-source-proposal.npy'
    replay_path = REPLAY / 'feasibility.json'
    replay_array = REPLAY / 'unadmitted-source-proposal.npy'
    qp_pin = verified(qp_path, EXPECTED['qp_report'], '004 feasibility report')
    arr_pin = verified(qp_array, EXPECTED['qp_array'], '004 proposal array')
    replay_pin = verified(replay_path, EXPECTED['replay_report'], '005 replay report')
    verified(REPLAY / 'feasibility.py', EXPECTED['replay_script'], '005 replay source')
    replay = load(replay_path)
    gate = replay.get('dependency_complete_replay', {})
    need(gate.get('array_bit_exact') is True and gate.get('geometry_algorithm_unchanged') is True, '005 replay is not bit-exact or changed the solver')
    need(gate.get('original004_proposal', {}).get('sha256') == EXPECTED['qp_array'], '005 replay no longer binds 004 proposal')
    need(replay.get('inputs_unchanged') is True and replay.get('inputs_before') == replay.get('inputs_after'), '005 tracked dependency inputs changed')
    tracked = replay.get('inputs_before', {})
    need(isinstance(tracked, dict) and len(tracked) == 542, '005 dependency closure size changed')
    for raw, digest in tracked.items():
        p = Path(raw)
        need(p.is_file() and not p.is_symlink() and sha(p) == digest, f'005 tracked dependency changed: {p}')
    candidate = np.load(qp_array, allow_pickle=False)
    replay_candidate = np.load(replay_array, allow_pickle=False)
    replay6_path = REPLAY6 / 'feasibility.json'
    replay6_array = REPLAY6 / 'unadmitted-source-proposal.npy'
    replay6_pin = verified(replay6_path, EXPECTED['replay6_report'], '006 receipt-map replay report')
    replay6_array_pin = verified(replay6_array, EXPECTED['replay6_array'], '006 replayed proposal array')
    verified(REPLAY6 / 'feasibility.py', EXPECTED['replay6_script'], '006 receipt-map replay source')
    replay6 = load(replay6_path)
    gate6 = replay6.get('dependency_complete_replay', {})
    need(replay6.get('status') == 'feasible_pending_exact_full_checks' and replay6.get('solver_success') is True, '006 replay did not complete its feasibility replay')
    need(gate6.get('array_bit_exact') is True and gate6.get('geometry_algorithm_unchanged') is True, '006 replay is not bit-exact or changed the solver')
    need(gate6.get('inherited_preflight_pins_revalidated') == 49, '006 replay did not revalidate all inherited preflight pins')
    need(gate6.get('original004_proposal', {}).get('sha256') == EXPECTED['qp_array'], '006 replay no longer binds 004 proposal')
    # Historical field name counts 547 tracked dependencies, not 547 modules.
    need(gate6.get('loaded_module_files_pinned') == 547, '006 dependency closure count changed')
    need(replay6.get('inputs_unchanged') is True and replay6.get('inputs_before') == replay6.get('inputs_after'), '006 tracked inputs changed')
    tracked6 = replay6.get('inputs_before', {})
    need(isinstance(tracked6, dict) and len(tracked6) == 547, '006 dependency closure incomplete')
    for raw, digest in tracked6.items():
        p = Path(raw)
        need(p.is_file() and not p.is_symlink() and sha(p) == digest, f'006 tracked dependency changed: {p}')
    for raw, digest in tracked.items():
        if raw != str((REPLAY / 'feasibility.py').resolve()):
            need(tracked6.get(raw) == digest, f'006 does not preserve 005 dependency: {raw}')
    replay6_candidate = np.load(replay6_array, allow_pickle=False)
    need(candidate.dtype == np.dtype('<f4') and candidate.shape == SKIN_SHAPE and np.isfinite(candidate).all(), '004 proposal array invalid')
    need(np.array_equal(candidate.view(np.uint32), replay_candidate.view(np.uint32)), '005 replayed proposal differs from004')
    need(np.array_equal(candidate.view(np.uint32), replay6_candidate.view(np.uint32)), '006 receipt-map replay differs from004')
    return static, pre_report, {'004_report': qp_pin, '004_array': arr_pin, '005_report': replay_pin, 'tracked_count': len(tracked),
        '006_report': replay6_pin, '006_array': replay6_array_pin, 'tracked_count_006': len(tracked6)}, candidate


def verify_audit(candidate: np.ndarray, proposal_pins: dict) -> dict:
    audit_script_pin = verified(AUDIT_SCRIPT, EXPECTED['audit_script'], 'frozen root003 audit source')
    declaration = load(AUDIT_DECLARATION)
    summary = load(AUDIT_SUMMARY)
    need(summary.get('schema') == 'numi.human.skin-local-self-reduction-nine-pose-audit.summary.v1', 'nine-pose audit schema mismatch')
    need(summary.get('status') == 'complete_exact_intersection_free', 'nine-pose audit did not pass all exact geometry gates')
    need(summary.get('inputs_unchanged') is True and summary.get('inputs_before') == summary.get('inputs_after'), 'nine-pose audit input inventory changed')
    need(summary.get('target_surface_count') == TARGETS and summary.get('pair_coverage_complete_all_nine') is True, 'nine-pose audit did not complete all 859 target scans')
    need(summary.get('all_nine_target_self_degenerate_and_closed_envelope_gates_pass') is True, 'nine-pose aggregate geometry gates are not all true')
    need(summary.get('steps') == EXPECTED_STEPS and summary.get('pose_result_count') == len(EXPECTED_STEPS), 'nine-pose audit steps/result count differ')
    need(declaration.get('schema') == 'numi.human.skin-local-self-reduction-nine-pose-audit.declaration.v1', 'nine-pose declaration schema mismatch')
    need(declaration.get('steps') == EXPECTED_STEPS and declaration.get('target_surface_count') == TARGETS, 'nine-pose declaration scope differs')
    need(declaration.get('output_directory') == str(AUDIT_ROOT), 'nine-pose audit output root differs')
    need(declaration.get('candidate_qualification') == 'Unadmitted source-position proposal; offline forward audit only.', 'audit qualification boundary changed')
    need(declaration.get('proposal_004', {}).get('sha256') == proposal_pins['004_array']['sha256'], 'audit does not bind exact 004 proposal')
    need(declaration.get('complete_dependency_replay_005', {}).get('sha256') == proposal_pins['005_report']['sha256'], 'audit does not bind dependency-complete 005 replay')
    need(declaration.get('proposal_replay_005', {}).get('sha256') == EXPECTED['replay_tracked_array'], 'audit does not bind 005 replay array')
    need(declaration.get('receipt_map_dependency_replay_006', {}).get('sha256') == proposal_pins['006_report']['sha256'], 'audit does not bind 006 receipt-map replay')
    need(declaration.get('proposal_replay_006', {}).get('sha256') == EXPECTED['replay6_array'], 'audit does not bind 006 replay array')
    need(declaration.get('inputs_before_scan') == summary.get('inputs_before'), 'audit summary input set differs from declaration')
    tracked = summary.get('inputs_before')
    need(isinstance(tracked, dict) and tracked.get(str(AUDIT_SCRIPT.resolve())) == EXPECTED['audit_script'], 'audit input closure does not bind exact root003 source')
    need(isinstance(tracked, dict) and len(tracked) >= 542, 'nine-pose audit lacks dependency-complete input closure')
    for raw, digest in tracked.items():
        path = Path(raw)
        need(path.is_file() and not path.is_symlink() and sha(path) == digest, f'nine-pose audit input changed: {path}')
    rows = summary.get('pose_results')
    need(isinstance(rows, list) and len(rows) == len(EXPECTED_STEPS), 'nine-pose result rows are incomplete')
    by_step = {}
    for row in rows:
        step = row.get('accepted_step')
        need(step not in by_step, f'duplicate audit pose {step}')
        by_step[step] = row
    need(list(by_step) == EXPECTED_STEPS, 'nine-pose result ordering differs')
    checked_rows = []
    for step in EXPECTED_STEPS:
        row = by_step[step]
        need(row.get('status') == 'complete_pair_coverage' and row.get('pair_coverage_complete') is True and row.get('surface_target_count') == TARGETS, f'incomplete 859-target coverage at {step}')
        need(row.get('all_skin_crossing_pair_count') == 0 and row.get('skin_self_crossing_pair_count') == 0, f'exact intersections remain at {step}')
        need(row.get('skin_degenerate_face_rows') == [] and row.get('invalid_target_surface_count') == 0 and row.get('invalid_target_triangle_count') == 0, f'invalid/degenerate geometry at {step}')
        inside = row.get('closed_target_inside_counts')
        need(inside == {'51005:63': {'inside_vertex_count': 0}, '51005:64': {'inside_vertex_count': 0}}, f'vastus outer-envelope inside counts missing/nonzero at {step}')
        need(row.get('baseline_orientation_and_bed_gates_pass') is True, f'baseline orientation/bed gate failed at {step}')
        winding = row.get('source_to_pose_winding')
        need(isinstance(winding, dict) and float(winding.get('minimum_source_to_accepted_pose_normal_alignment', 0)) > 0 and float(winding.get('minimum_face_map_determinant', 0)) > 0, f'source-to-pose orientation failed/missing at {step}')
        need(float(row.get('source_surface_min_triangle_area_m2', 0)) > 0 and float(row.get('mapped_surface_min_triangle_area_m2', 0)) > 0, f'zero-area source/mapped skin triangle at {step}')
        result_path = AUDIT_ROOT / f'step-{step}.result.json'
        result_on_disk = load(result_path)
        need(result_on_disk == row, f'summary/result row mismatch at {step}')
        artifacts = {
            f'step-{step}.targets.jsonl': row.get('targets_sha256'),
            f'step-{step}.crossing-witnesses.jsonl': row.get('crossing_witnesses_sha256'),
            f'step-{step}.self-witnesses.jsonl': row.get('self_witnesses_sha256'),
            f'step-{step}.invalid-triangles.jsonl': row.get('invalid_triangles_sha256'),
        }
        for name, digest in artifacts.items():
            path = AUDIT_ROOT / name
            need(isinstance(digest, str) and path.is_file() and not path.is_symlink() and sha(path) == digest, f'nine-pose audit artifact changed/missing: {path}')
        checked_rows.append({'step': step, 'result': pin(result_path), 'targets_sha256': artifacts[f'step-{step}.targets.jsonl'],
            'crossing_sha256': artifacts[f'step-{step}.crossing-witnesses.jsonl'],
            'self_sha256': artifacts[f'step-{step}.self-witnesses.jsonl'],
            'invalid_sha256': artifacts[f'step-{step}.invalid-triangles.jsonl']})
    after = {p: sha(Path(p)) for p in tracked}
    need(after == tracked, 'nine-pose audit input changed during package validation')
    return {'declaration': pin(AUDIT_DECLARATION), 'summary': pin(AUDIT_SUMMARY), 'audit_script': audit_script_pin, 'summary_status': summary['status'],
        'steps': EXPECTED_STEPS, 'target_count': TARGETS, 'inputs_unchanged': True,
        'tracked_input_count': len(tracked), 'candidate_positions_sha256': EXPECTED['qp_array'],
        'pose_artifacts': checked_rows, 'aggregate_geometry_gate': True}


def compose(output: Path, static: dict, candidate: np.ndarray, pre_report: dict, proposal_pins: dict, audit: dict) -> dict:
    from numilab_human.skin_source_payload_preflight import decode_payload
    from numilab_human.common_atlas_skin_geometry_registration import compose_disjoint_skin_position_corrections
    from numilab_human.resting_anatomy import compose_skin_binding_candidate
    out = output.expanduser().resolve()
    need(not out.exists(), 'refuse to overwrite composition output')
    base = static['skin_path']
    receipt_path = static['receipt_path']
    receipt = static['receipt']
    manifest_path = static['skin_manifest_path']
    manifest = load(manifest_path)
    raw = base.read_bytes()
    decoded_base = decode_payload(raw)
    nv = decoded_base['vertex_count']
    nb = decoded_base['binding_count']
    need((nv, nb) == (54949, 86), 'base NHSKIN dimensions changed')
    vertex_offset = 60 + 36 * nb
    edited = bytearray(raw)
    np.frombuffer(edited, '<f4', count=14 * nv, offset=vertex_offset).reshape(nv, 14)[:, :3] = candidate
    registration_path = Path(manifest['inputs']['registration_path'])
    need(sha(registration_path) == manifest['inputs']['registration_sha256'], 'base skin registration changed')
    registration = load(registration_path)
    candidate_bytes, owner_composition = compose_disjoint_skin_position_corrections(
        raw, [bytes(edited)], global_source_matrix=registration['coordinate_system']['global_source_mm_to_myosim_world_m'])
    decoded = decode_payload(candidate_bytes)
    need(np.array_equal(decoded['vertices_u'][:, :3], candidate.view(np.uint32)), 'existing owner changed source candidate positions')
    need(candidate_bytes[:vertex_offset] == raw[:vertex_offset], 'composition changed header or binding prefix')
    need(candidate_bytes[vertex_offset + 56 * nv:] == raw[vertex_offset + 56 * nv:], 'composition changed topology stream')
    need(np.array_equal(decoded['indices'].reshape(-1, 3), decoded_base['indices'].reshape(-1, 3)), 'composition changed face rows')
    need(np.array_equal(decoded['vertices_u'][:, 6:], decoded_base['vertices_u'][:, 6:]), 'composition changed influence records')

    out.mkdir(parents=True)
    asset = out / 'bodyparts3d-myosim-skinned-shell.nhskin'
    asset.write_bytes(candidate_bytes)
    candidate_manifest_path = out / 'common-atlas-skin-geometry-registration.manifest.json'
    manifest['inputs']['source_payload'] = {**pin(base), 'route': '004 per-vertex local-self-reduction replayed by dependency-complete 005'}
    manifest['inputs']['upstream_skin_provenance'] = {'immediate_source_path': str(manifest_path), 'immediate_source_sha256': sha(manifest_path), 'immediate_source_record': pin(base)}
    manifest['output_payload'] = pin(asset)
    manifest['output_manifest'] = str(candidate_manifest_path)
    manifest['geometry_registration'] = {
        'method': 'inferred common-atlas source-position reduction within the previously selected local support',
        'source_payload': pin(base), 'source_positions': proposal_pins['004_array'],
        'qp_report': proposal_pins['004_report'], 'dependency_complete_replay': proposal_pins['005_report'],
        'receipt_map_dependency_replay_006': proposal_pins['006_report'], 'proposal_replay_006': proposal_pins['006_array'],
        'nine_pose_exact_audit': audit, 'owner_composition': owner_composition,
        'previous_registration_manifest': pin(manifest_path),
        'interpretation': 'Inferred reference geometry. The QP proposes a bounded local reduction of the earlier clearance candidate; exact geometry audit is separate. No measured tissue thickness or clinical clearance claim.'
    }
    manifest['method'] = 'Existing NHSKIN source-position composition, using the 004 per-vertex reduction after exact 9-pose audit.'
    manifest['evidence_boundary'] = 'The source edit is inferred reference anatomy. Geometry audit covers the listed nine captured poses only; contact geometry/Jacobians may change, so native physiological response remains unqualified.'
    manifest['qualification'] = {'nine_pose_geometry_audit': 'complete exact offline captured-pose audit', 'native_contact_response': 'pending fresh native qualification'}
    manifest['current_physics_payloads'] = {'nhtiss_payload': pin(Path(static['nhtiss_record']['payload_path'])),
        'nhtiss_manifest': pin(Path(static['nhtiss_record']['manifest_path'])), 'source_receipt': pin(receipt_path),
        'fit001_three_pose_fit': pre_report['fit001_source_fit'],
        'heldout002_failed_ancestry': pre_report['heldout002_failed_ancestry']}
    manifest['code'] = {'owner_revision': '2eace39df60d72ba1197a775aea6ca4205d6e484',
        'owner_revision_record': pin(ROOT / 'source-revision.json'),
        'source_files': {p.name: pin(p) for p in (
            ROOT/'src/numilab_human/resting_anatomy.py',
            ROOT/'src/numilab_human/common_atlas_skin_geometry_registration.py',
            ROOT/'src/numilab_human/skin_source_payload_preflight.py',
            ROOT/'src/numilab_human/passive_attachment_composition.py')}}
    candidate_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
    composed = compose_skin_binding_candidate(receipt_path, asset, candidate_manifest_path, out/'composed-anatomy')
    need(composed.get('anatomy_payload_unchanged') is True and composed.get('functional_bindings_unchanged') is True, 'receipt composition changed anatomy payload or functional bindings')
    composed_receipt_path = Path(composed['receipt_path'])
    composed_receipt = load(composed_receipt_path)
    need(composed_receipt.get('payload') == receipt.get('payload'), 'receipt composition changed NHA payload')
    need(composed_receipt.get('functional_bindings') == receipt.get('functional_bindings'), 'receipt composition changed functional bindings')
    need(composed_receipt.get('provenance', {}).get('native_muscle_surfaces') == receipt.get('provenance', {}).get('native_muscle_surfaces'), 'receipt composition changed refined NHTISS lineage')
    need(composed_receipt.get('thorax_source_volume_m3') == receipt.get('thorax_source_volume_m3'), 'receipt composition changed thorax volume')
    scene = copy.deepcopy(static['scene'])
    scene['source']['skin'].update(path=str(asset), sha256=sha(asset))
    scene_path = out / 'resting-supine-scene.manifest.json'
    scene_path.write_text(json.dumps(scene, indent=2, sort_keys=True, allow_nan=False)+'\n')
    expected_scene = copy.deepcopy(static['scene'])
    expected_scene['source']['skin'].update(path=str(asset), sha256=sha(asset))
    need(load(scene_path) == expected_scene, 'scene fields changed outside the NHSKIN path/hash')
    need(load(scene_path).get('bed') == static['scene'].get('bed'), 'fixed bed/support witnesses changed')

    report = {'schema':'numi.human.skin-resting-local-reduction-004-composition-review.v1',
        'status':'source_candidate_composed_native_pending',
        'native_run_performed':False, 'candidate_admitted':False,
        'source_revision':pin(OWNER_REVISION), 'parent_1217_declaration':pin(static_module_decl_path()),
        'source_receipt':pin(receipt_path), 'base_skin':pin(base), 'candidate_skin':pin(asset),
        'candidate_manifest':pin(candidate_manifest_path), 'composed_receipt':pin(composed_receipt_path),
        'candidate_scene':pin(scene_path), 'local_reduction':proposal_pins, 'nine_pose_audit':audit,
        'preservation':{'full_86_bindings_byte_identical':True,'indices_and_face_order_byte_identical':True,
            'NHA_payload_and_functional_bindings_unchanged':True,'current_refined_NHTISS_unchanged':True,
            'fixed_flat_bed_and_support_witnesses_unchanged':True,'normals_recomputed_by_existing_owner':owner_composition.get('rest_world_normals_recomputed_by_existing_skin_owner') is True},
        'receipt_composition':composed,
        'evidence_boundary':'Inferred reference skin source geometry, checked by exact intersection/topology predicates at nine captured poses only. No continuous-time, native physiological, or measured-tissue claim.'}
    (out/'composition-review.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return report


def static_module_decl_path() -> Path:
    return R / 'muscle-conforming-refinement-1216/native-refinement-1217-attempt2/run-declaration.json'


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-check',action='store_true',help='read-only validate inputs and candidate ancestry')
    mode.add_argument('--compose',action='store_true',help='compose only after exact nine-pose audit and explicit root review')
    ap.add_argument('--root-reviewed',action='store_true',help='explicit operator acknowledgement that root reviewed the successful audit')
    ap.add_argument('--output',type=Path,default=PREP/'composed-candidate')
    args=ap.parse_args()
    static, pre_report, proposal_pins, candidate = verified_context()
    audit_status='not_started'
    audit=None
    if AUDIT_SUMMARY.is_file() and AUDIT_DECLARATION.is_file():
        try:
            audit=verify_audit(candidate,proposal_pins)
            audit_status='complete_exact_intersection_free'
        except Exception as exc:
            audit_status='present_failed_or_unverified'
            if args.compose:
                raise
            audit={'failure':str(exc),'summary':pin(AUDIT_SUMMARY),'declaration':pin(AUDIT_DECLARATION)}
    if args.preflight_check:
        print(json.dumps({'status':'candidate_verified_waiting_for_reviewed_nine_pose_audit' if audit_status!='complete_exact_intersection_free' else 'nine_pose_audit_passed_waiting_for_root_review',
            'composition_performed':False,'native_run_performed':False,'candidate_admitted':False,
            'preflight_script':pin(PREFLIGHT),'qp_and_replay':proposal_pins,'audit_status':audit_status,'audit':audit,
            'ancestry':{'fit001_source_fit':pre_report['fit001_source_fit'],'heldout002_failed_admission':pre_report['heldout002_failed_ancestry']}},sort_keys=True,allow_nan=False))
        return 0
    need(args.root_reviewed, '--compose requires explicit --root-reviewed acknowledgement')
    need(audit_status=='complete_exact_intersection_free' and audit is not None, 'refuse composition without successful nine-pose audit')
    report=compose(args.output,static,candidate,pre_report,proposal_pins,audit)
    print(json.dumps({'status':report['status'],'composition_review':str(args.output.resolve()/'composition-review.json'),'native_run_performed':False},sort_keys=True))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
