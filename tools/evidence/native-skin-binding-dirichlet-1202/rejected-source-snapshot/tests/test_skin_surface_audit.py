"""Asset-backed whole-surface and adversarial visual skin checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

import numpy as np
import pytest

from numilab_human.skin_surface_audit import audit_skin_surface
from test_torso_anatomy_source_audit import _mutate_pack


@pytest.fixture(scope='module', params=['raw-source-rest', 'neutral', 'coupled-torso',
                                       'coupled-reach', 'knee-flexion'])
def skin_inputs(request):
    name = os.environ.get('NUMILAB_HUMAN_NATIVE_SKIN_AUDIT_ROOT')
    if not name:
        pytest.skip('skin source audit needs source-bound native captures')
    root = Path(name).resolve()
    repo = Path(__file__).resolve().parents[1]
    capture = root / ('native-' + request.param)
    assert (capture / 'exit.code').read_text().strip() == '0'
    q = json.loads((capture / 'pose.json').read_text())
    pose = None if q is None else tuple(map(tuple, q))
    return (repo / 'Sources', repo / 'Build/myosim-fullbody',
            repo / 'Build/knee-parity-registration-20260929/candidate.v6.registration.json',
            root / 'payload/bodyparts3d-myosim-skinned-shell.nhskin',
            next(capture.glob('*.mrvpack')), next(capture.glob('*.skin-poses.json')), pose)


def test_every_source_vertex_triangle_and_body_blend_reaches_native_skin(skin_inputs):
    report = audit_skin_surface(*skin_inputs)
    assert report['passed']
    assert report['vertex_count'] == 54949 and report['triangle_count'] == 109183
    assert report['binding_count'] == 86 and len(report['body_checks']) == 86
    assert report['payload_abi'] == 5 and report['runtime_influences_per_vertex'] == 86
    assert report['runtime_discarded_weight_mass'] == 0
    assert report['source_subset']['edge_component_count'] == 100
    certificate = report['source_weight_solution_certificate']
    assert certificate['passed'] and certificate['source_bone_count'] == 185
    assert certificate['seed_candidate_count'] == 12025 and certificate['seed_vertex_count'] == 5039
    assert certificate['rejected_seed_count'] == 696
    assert certificate['source_graph_vertex_count'] == 54663
    assert certificate['source_graph_edge_count'] == 163860
    assert certificate['exact_coincident_vertex_group_count'] == 257
    assert certificate['exact_coincident_redundant_vertex_count'] == 286
    assert report['native_source_seam_diagnostics']['maximum_native_seam_gap_m'] == 0
    assert report['native_surface_face_diagnostics']['face_count'] == 109183
    assert report['source_surface_topology']['exact_coordinate_quotient']['boundary_edge_count'] == 171
    assert not report['source_surface_topology']['closed_volume_admitted']
    assert certificate['maximum_relative_equation_residual'] <= 1e-10
    assert report['maximum_native_world_vertex_error_m'] <= 2e-5
    assert all(row['passed'] for row in report['body_checks'])
    skin_inputs[4].with_name('source-skin-audit.json').write_text(json.dumps(report, indent=2) + '\n')


@pytest.mark.parametrize('corruption', ['native_vertex', 'native_normal', 'native_topology',
                                       'native_owner', 'payload_owner', 'payload_weights',
                                       'native_current_pose', 'native_rest_pose', 'source_hash',
                                       'seed_association', 'harmonic_equation', 'false_physics',
                                       'hidden_truncation', 'projection_gap', 'full_runtime_weights',
                                       'claimed_runtime_truncation', 'unequal_exact_seam_weights',
                                       'false_seam_counts'])
def test_skin_oracle_rejects_rehashed_geometry_and_rest_invisible_influence_corruption(skin_inputs, tmp_path, corruption):
    from numilab_human.torso_anatomy_audit import _pack_sections
    args = list(skin_inputs)
    if corruption.startswith('native_') and corruption not in {'native_current_pose', 'native_rest_pose'}:
        sections = _pack_sections(args[4])
        primitives = np.frombuffer(sections[4][0], '<u4').reshape(-1, 16)
        selected = int(np.flatnonzero(primitives[:, 4] == 51007)[0])
        primitive = primitives[selected]
        packed_i = np.frombuffer(sections[3][0], '<u4')
        vertex = int(packed_i[int(primitive[0])])
        pack = tmp_path / 'rehash.mrvpack'
        if corruption in {'native_vertex', 'native_normal'}:
            def change(raw, offset, _):
                index = offset + vertex * 80 + (16 if corruption == 'native_normal' else 0)
                struct.pack_into('<f', raw, index, struct.unpack_from('<f', raw, index)[0] + .005)
            _mutate_pack(args[4], pack, 2, change)
            args[4] = pack
            report = audit_skin_surface(*args)
            assert not report['passed']
            if corruption == 'native_vertex':
                assert report['maximum_native_world_vertex_error_m'] > .0049
            else:
                assert report['maximum_native_normal_error'] > .0049
            return
        if corruption == 'native_topology':
            def change(raw, offset, _):
                index = offset + 4 * int(primitive[0])
                a, b = struct.unpack_from('<2I', raw, index)
                struct.pack_into('<2I', raw, index, b, a)
            _mutate_pack(args[4], pack, 3, change)
            message = 'native skin topology'
        else:
            _mutate_pack(args[4], pack, 4, lambda raw, offset, _: struct.pack_into('<I', raw, offset + selected * 64 + 24, 7))
            message = 'native skin semantic/owner'
        args[4] = pack
    elif corruption in {'native_current_pose', 'native_rest_pose'}:
        snapshot = json.loads(args[5].read_text())
        snapshot['bodies'][0]['current' if corruption == 'native_current_pose' else 'rest']['position_world_m'][0] += .02
        path = tmp_path / 'changed-poses.json'
        path.write_text(json.dumps(snapshot))
        args[5] = path
        report = audit_skin_surface(*args)
        assert not report['passed']
        assert any(not row['passed'] for row in report['body_checks'])
        return
    else:
        payload = tmp_path / args[3].name
        raw = bytearray(args[3].read_bytes())
        _, _, nb, nv, ni, *_ = struct.unpack_from('<8s5I32s', raw)
        if corruption == 'payload_owner':
            struct.pack_into('<I', raw, 60, 7)
            message = 'source skin body ownership'
        elif corruption == 'payload_weights':
            offset = 60 + 36 * nb
            for vertex in range(nv):
                index = offset + 56 * vertex + 40
                weights = list(struct.unpack_from('<4f', raw, index))
                if weights[0] > .25 and weights[1] > .15:
                    weights[0] += .1
                    weights[1] -= .1
                    struct.pack_into('<4f', raw, index, *weights)
                    break
            else:
                raise AssertionError('real skin has no blended vertex')
            # The weights still sum to one and all local source frames map
            # to the same world point at rest: static rest geometry cannot
            # identify this corruption; the source blend oracle must.
            message = 'source skin influence weights'
        elif corruption == 'full_runtime_weights':
            matrix = np.frombuffer(raw, '<f4', offset=60+36*nb+56*nv+4*ni).reshape(nv,nb)
            point = int(np.argmax(matrix[:,0]))
            donor = int(np.argmax(matrix[point]))
            receiver = (donor+1) % nb
            amount = min(.1, matrix[point,donor]/2)
            matrix[point,donor] -= amount
            matrix[point,receiver] += amount
            message = 'source skin full influence weights'
        else:
            message = 'skin member hash'
        payload.write_bytes(raw)
        manifest = json.loads(args[3].with_name('bodyparts3d-myosim-skinned-shell.manifest.json').read_text())
        manifest['payload']['sha256'] = hashlib.sha256(raw).hexdigest()
        solution = manifest['coverage']['binding_solution']
        proof = tmp_path / solution['file']
        shutil.copyfile(args[3].parent / solution['file'], proof)
        if corruption in {'seed_association', 'harmonic_equation', 'projection_gap', 'unequal_exact_seam_weights'}:
            with np.load(proof, allow_pickle=False) as archive:
                full, targets = archive['full_weights'].copy(), archive['seed_targets'].copy()
                gaps = archive['seed_projection_gaps_m'].copy()
            if corruption == 'seed_association':
                targets[0, 3] = (targets[0, 3] + 1) % nb
                message = 'source skin seed association'
            elif corruption in {'harmonic_equation', 'unequal_exact_seam_weights'}:
                # Keep positivity and partition unity while breaking a row
                # of the independently checked source graph equations.
                point = 54847 if corruption == 'unequal_exact_seam_weights' else int(np.argmax(full[:, 0]))
                donor = int(np.argmax(full[point]))
                receiver = (donor + 1) % nb
                amount = min(.1, full[point, donor] / 2)
                full[point, donor] -= amount
                full[point, receiver] += amount
                message = 'source exact seam weights' if corruption == 'unequal_exact_seam_weights' else 'source skin harmonic equations'
            else:
                gaps[0] += .01
                message = 'source bone-to-skin projection gaps'
            np.savez_compressed(proof, full_weights=full, seed_targets=targets, seed_projection_gaps_m=gaps)
            solution['sha256'] = hashlib.sha256(proof.read_bytes()).hexdigest()
            solution['bytes'] = proof.stat().st_size
        elif corruption == 'false_physics':
            manifest['status'] = 'anatomically_validated_physical_skin'
            message = 'payload type/ownership'
        elif corruption == 'hidden_truncation':
            manifest['coverage']['source_surface_binding']['maximum_discarded_weight_mass'] = 0
            message = 'source binding declared truncation'
        elif corruption == 'claimed_runtime_truncation':
            manifest['coverage']['maximum_runtime_discarded_weight_mass'] = .1
            message = 'full runtime weight ownership/coverage'
        elif corruption == 'false_seam_counts':
            manifest['coverage']['source_surface_binding']['exact_coincident_vertex_group_count'] = 0
            message = 'source exact seam declared counts'
        if corruption == 'source_hash':
            manifest['source']['skin']['member_sha256'] = '0' * 64
        payload.with_name('bodyparts3d-myosim-skinned-shell.manifest.json').write_text(json.dumps(manifest))
        args[3] = payload
    with pytest.raises(ValueError, match=message):
        audit_skin_surface(*args)



def _minimal_dirichlet_oracle_case():
    """Tiny closed source graph with one exact duplicated seam vertex."""
    import io
    from numilab_human.skin_surface_binding import DIRICHLET_METHOD
    from numilab_human.skin_surface_audit import _verified_weights

    world = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                        [0., 1., 0.], [0., 0., 0.]], dtype='<f8')
    faces = np.asarray([[0, 1, 2], [4, 2, 3]], dtype='<u4')
    # Vertex 4 is exactly the same material point as vertex 0 and must receive
    # the same fixed Dirichlet value. The two free vertices interpolate the
    # endpoint owner values harmonically on this symmetric graph.
    full = np.asarray([[1., 0.], [.5, .5], [0., 1.], [.5, .5], [1., 0.]], dtype='<f8')
    targets = np.asarray([[0, 0xffffffff, 0, 0, 1],
                          [1, 0xffffffff, 2, 1, 1]], dtype='<u4')
    gaps = np.asarray([1., 1.], dtype='<f8')
    proof = io.BytesIO()
    np.savez_compressed(proof, full_weights=full, seed_targets=targets,
                        seed_projection_gaps_m=gaps)
    result = _verified_weights(proof.getvalue(), world.tobytes(), faces.tobytes(),
                               targets.tobytes(), gaps.tobytes(), 2, DIRICHLET_METHOD)
    return result, full


def test_dirichlet_source_oracle_checks_exact_seams_fixed_shares_and_free_harmonic_rows():
    (_, _, full, certificate), expected = _minimal_dirichlet_oracle_case()
    np.testing.assert_array_equal(full, expected)
    assert certificate['passed'] is True
    assert certificate['seed_condition'] == 'dirichlet'
    assert certificate['maximum_fixed_seed_weight_error'] == 0
    assert certificate['maximum_relative_equation_residual'] < 1e-14
    assert certificate['source_graph_vertex_count'] == 4
    assert certificate['exact_coincident_redundant_vertex_count'] == 1
    assert certificate['minimum_source_projection_gap_m'] == 1


@pytest.mark.parametrize(('corruption', 'message'), [
    ('fixed_seed', 'source exact seam weights|source Dirichlet fixed-seed owner shares'),
    ('free_harmonic', 'source skin harmonic equations'),
    ('negative', 'binding solution positivity/partition'),
    ('seam', 'source exact seam weights'),
])
def test_dirichlet_source_oracle_rejects_seed_harmonic_positivity_and_seam_tampering(
        corruption, message):
    import io
    from numilab_human.skin_surface_binding import DIRICHLET_METHOD
    from numilab_human.skin_surface_audit import _verified_weights

    _, expected = _minimal_dirichlet_oracle_case()
    world = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                        [0., 1., 0.], [0., 0., 0.]], dtype='<f8')
    faces = np.asarray([[0, 1, 2], [4, 2, 3]], dtype='<u4')
    full = expected.copy()
    targets = np.asarray([[0, 0xffffffff, 0, 0, 1],
                          [1, 0xffffffff, 2, 1, 1]], dtype='<u4')
    gaps = np.asarray([1., 1.], dtype='<f8')
    if corruption == 'fixed_seed':
        full[[0, 4]] = [.9, .1]
    elif corruption == 'free_harmonic':
        full[1] = [.75, .25]
    elif corruption == 'negative':
        full[1] = [1.1, -.1]
    elif corruption == 'seam':
        full[4] = [.9, .1]
    proof = io.BytesIO()
    np.savez_compressed(proof, full_weights=full, seed_targets=targets,
                        seed_projection_gaps_m=gaps)
    with pytest.raises(ValueError, match=message):
        _verified_weights(proof.getvalue(), world.tobytes(), faces.tobytes(),
                          targets.tobytes(), gaps.tobytes(), 2, DIRICHLET_METHOD)
