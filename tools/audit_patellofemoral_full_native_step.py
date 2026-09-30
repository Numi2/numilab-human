"""Capture and independently verify the bounded full-source cartilage step.

This is an internal reproducibility receipt, not a loaded-knee qualification.
The four compressed accepted states are retained so the exact full-boundary
geometry check can be rerun without the original GPU or Matter checkout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from numilab_human.abdominal_organ_separation import exact_integer_meshes
from numilab_human.cardiac_cavity_intersections import point_location
from numilab_human.open_knee import parse_source
from numilab_human.surface_topology_audit import exact_embedding
from tools.audit_patellofemoral_pose_clearance import count_intersections, run as pose_run
from tools.export_patellofemoral_matter_input import run as export_run


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'Docs/media/patellofemoral-full-native-20260930'
WORK = ROOT / 'Build/full-patellofemoral-matter-20260930'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return digest(path.read_bytes())


def accepted_geometry(side: str, positions: bytes, source) -> dict:
    assert len(positions) == 50991 * 3 * 4
    nodes = np.frombuffer(positions, dtype='<f4').reshape((50991, 3))
    meshes = {}
    volume = {}
    self_embedding = {}
    offset = 0
    for region in ('PTC', 'FMC'):
        source_region = source.regions[region]
        local = {identifier: index for index, identifier in
                 enumerate(source_region.node_ids)}
        faces = np.asarray([
            [local[identifier] for identifier in face]
            for face in source.surfaces[f'{region}_All_Faces'].faces
        ], dtype=np.int64)
        size = len(source_region.node_ids)
        current = nodes[offset:offset + size].astype(np.float64)
        meshes[region] = (current, faces)
        used = np.unique(faces)
        compact = np.searchsorted(used, faces)
        result = exact_embedding(current[used].tolist(), compact.tolist())
        assert result['count'] == 0
        assert result['topology']['closed_oriented_manifold_candidate'] == (region == 'PTC')
        self_embedding[region] = {
            'exact_self_intersection_pairs': result['count'],
            'aabb_candidate_pairs': result['aabb_candidate_pairs'],
            'allowed_shared_vertex_or_edge_pairs':
                result['allowed_shared_vertex_or_edge_pairs'],
            'closed_oriented_manifold_candidate':
                result['topology']['closed_oriented_manifold_candidate'],
        }
        local = {identifier: index for index, identifier in
                 enumerate(source_region.node_ids)}
        tetrahedra = np.asarray([
            [local[identifier] for identifier in cell]
            for cell in source_region.elements], dtype=np.int64)
        if side == 'right':
            tetrahedra = tetrahedra[:, [1, 0, 2, 3]]
        signed_six = signed_six_volumes(current, tetrahedra)
        assert np.all(signed_six > 0)
        volume[region] = {
            'positive_tetrahedra': int(np.count_nonzero(signed_six > 0)),
            'zero_or_negative_tetrahedra': int(np.count_nonzero(signed_six <= 0)),
            'minimum_signed_tetrahedron_volume_m3': float(signed_six.min() / 6),
            'sum_signed_tetrahedron_volume_m3': float(signed_six.sum() / 6),
        }
        offset += size
    assert offset == 50991
    _, exact = exact_integer_meshes(meshes)
    intersections = count_intersections(exact)
    parity = {}
    for first, second in (('PTC', 'FMC'), ('FMC', 'PTC')):
        faces = exact[first]['faces']
        samples = []
        for face_index in (0, len(faces) // 2, len(faces) - 1):
            vertex_index = int(faces[face_index, 0])
            location = point_location(exact[first]['points'][vertex_index],
                                      exact[second]['records'])
            assert location['location'] == 'outside'
            samples.append({'source_surface_face_index': face_index,
                            'source_surface_vertex_index': vertex_index,
                            'exact_ray_parity': location})
        parity[f'{first}_in_{second}'] = samples
    return {'face_intersections': intersections,
            'mutual_surface_point_parity': parity,
            'exact_self_embedding': self_embedding,
            'accepted_tetrahedral_orientation': volume}


def signed_six_volumes(points: np.ndarray,
                       tetrahedra: np.ndarray) -> np.ndarray:
    a, b, c, d = (points[tetrahedra[:, index]] for index in range(4))
    return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)


def source_cartilage_chain(source) -> dict:
    """Check that the declared full surface is the oriented FEM boundary."""
    report = {}
    for region in ('PTC', 'FMC'):
        row = source.regions[region]
        tet = np.asarray(row.elements, dtype=np.int64)
        faces = np.concatenate((tet[:, [1, 2, 3]], tet[:, [0, 3, 2]],
                                tet[:, [0, 1, 3]], tet[:, [0, 2, 1]]))
        keys = np.sort(faces, axis=1)
        unique, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        assert np.all((counts == 1) | (counts == 2))
        boundary = faces[counts[inverse] == 1]
        declared = np.asarray(source.surfaces[f'{region}_All_Faces'].faces,
                              dtype=np.int64)

        def canonical_oriented(triangles: np.ndarray) -> np.ndarray:
            rotation = (np.argmin(triangles, axis=1)[:, None] +
                        np.arange(3)[None, :]) % 3
            oriented = np.take_along_axis(triangles, rotation, axis=1)
            return oriented[np.lexsort(oriented.T[::-1])]

        assert np.array_equal(canonical_oriented(boundary),
                              canonical_oriented(declared))
        index = {identifier: i for i, identifier in enumerate(row.node_ids)}
        local_tets = np.asarray([[index[identifier] for identifier in cell]
                                 for cell in row.elements], dtype=np.int64)
        points = np.asarray(row.nodes_mm, dtype=np.float64) * 0.001
        signed_six = signed_six_volumes(points, local_tets)
        assert np.all(signed_six > 0)
        report[region] = {
            'tetrahedra': len(tet),
            'unique_tetrahedral_faces': len(unique),
            'one_owner_boundary_faces': int(np.count_nonzero(counts == 1)),
            'two_owner_interior_faces': int(np.count_nonzero(counts == 2)),
            'three_or_more_owner_faces': int(np.count_nonzero(counts > 2)),
            'declared_full_boundary_oriented_face_match': True,
            'positive_source_tetrahedra': int(np.count_nonzero(signed_six > 0)),
            'minimum_source_tetrahedron_volume_m3': float(signed_six.min() / 6),
            'sum_source_tetrahedron_volume_m3': float(signed_six.sum() / 6),
        }
    return report


def fmc_defect_star(source) -> dict:
    """Distinguish a connected tetrahedral annulus from a split-able vertex."""
    node = 233523
    incident = [(index, tet) for index, tet in
                enumerate(source.regions['FMC'].elements) if node in tet]
    links = [tuple(value for value in tet if value != node)
             for _, tet in incident]
    assert len(links) == 12 and all(len(face) == 3 for face in links)
    edges = Counter(tuple(sorted((face[a], face[b])))
                    for face in links for a, b in ((0, 1), (1, 2), (2, 0)))
    assert set(edges.values()) == {1, 2}
    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    boundary_graph: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        boundary_graph[a].add(b)
        boundary_graph[b].add(a)
    assert all(len(neighbors) == 2 for neighbors in boundary_graph.values())
    pending = set(boundary_graph)
    loops = []
    while pending:
        stack = [pending.pop()]
        group = set(stack)
        while stack:
            for other in boundary_graph[stack.pop()] & pending:
                pending.remove(other)
                group.add(other)
                stack.append(other)
        loops.append(sorted(group))
    face_owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    for index, face in enumerate(links):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            face_owners[tuple(sorted((face[a], face[b])))].add(index)
    adjacency = {index: set() for index in range(len(links))}
    for owners in face_owners.values():
        for index in owners:
            adjacency[index].update(owners - {index})
    unseen = set(adjacency)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for other in adjacency[stack.pop()] & unseen:
                unseen.remove(other)
                stack.append(other)
    vertices = set(value for face in links for value in face)
    euler = len(vertices) - len(edges) + len(links)
    assert components == 1 and euler == 0 and sorted(map(len, loops)) == [6, 6]
    incident_surfaces = {
        name: sum(node in face for face in source.surfaces[name].faces)
        for name in ('FMC_@_FMB_TiesFaces', 'FMC_@_PTC_ContactFaces')
    }
    assert incident_surfaces == {
        'FMC_@_FMB_TiesFaces': 6, 'FMC_@_PTC_ContactFaces': 6}
    fmc = source.regions['FMC']
    defect_position = np.asarray(fmc.nodes_mm[fmc.node_ids.index(node)],
                                 dtype=np.float64)
    ptc = source.regions['PTC']
    distances = np.linalg.norm(np.asarray(ptc.nodes_mm, dtype=np.float64) -
                               defect_position, axis=1)
    nearest = int(np.argmin(distances))
    return {'source_node_id': node, 'incident_tetrahedra': len(incident),
            'tetrahedral_star_components': components,
            'link_vertices': len(vertices), 'link_edges': len(edges),
            'link_triangles': len(links), 'link_euler_characteristic': euler,
            'link_boundary_cycles_source_node_ids': sorted(loops),
            'incident_source_surface_faces': incident_surfaces,
            'nearest_PTC_source_node_id': ptc.node_ids[nearest],
            'nearest_PTC_source_node_distance_mm': float(distances[nearest]),
            'single_vertex_split_preserves_tetrahedral_adjacency': False}


def native_case(binary: Path, side: str, label: str,
                *, pose_um: int = 20, baseline: bool = False,
                contact_off: bool = False) -> tuple[dict, bytes]:
    input_name = f'open-knee-{side}-ptc-fmc'
    if pose_um != 20:
        input_name += f'-{pose_um}um'
    output = WORK / f'{side}-receipt-{label}.f32le'
    command = [str(binary), str(WORK / f'{input_name}.nhcar'), str(output)]
    if baseline:
        command.append('--baseline')
    else:
        command += ['--contact-slop-m', '0.00002',
                    '--approach-speed-mps', '0.1']
    if contact_off:
        command.append('--disable-contact')
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    row = json.loads(result.stdout)
    assert row['side'] == (0 if side == 'left' else 1)
    assert row['source_nodes'] == 50991
    assert row['source_tetrahedra'] == 208177
    return row, output.read_bytes()


def capture(matter_root: Path, build_dir: Path) -> dict:
    binary = build_dir / 'numi-matter-patellofemoral-full-surface-step'
    metallib = build_dir / 'shaders/NumiMatter.metallib'
    material = matter_root / 'matter/materials/open_knee_cartilage_isotropic_preflight.nmatter'
    for path in (binary, metallib, material):
        assert path.is_file(), path
    pose14 = pose_run(14)
    assert pose14['sides']['left']['candidate']['segment_or_polygon_crossing_pairs'] == 0
    inputs20 = export_run()
    inputs14 = export_run(WORK / 'pose-clearance-14um.json')
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    cases = {}
    geometry = {}
    for side in ('left', 'right'):
        cases[side] = {}
        baseline, _ = native_case(binary, side, 'baseline', baseline=True)
        near, _ = native_case(binary, side, '14um', pose_um=14)
        assert baseline['status_code'] == 6 and baseline['completed_microsteps'] == 0
        assert baseline['rollback_bitwise'] and baseline['diagnostics'][2] == -1
        assert near['status_code'] == 6 and near['completed_microsteps'] == 0
        assert near['rollback_bitwise'] and near['diagnostics'][2] == -4
        cases[side]['baseline'] = baseline
        cases[side]['near_plane_14um'] = near
        for mode in ('on', 'off'):
            row, positions = native_case(binary, side, mode,
                                         contact_off=(mode == 'off'))
            assert row['status_code'] == 0 and row['completed_microsteps'] == 1
            assert not row['rollback_bitwise']
            assert row['active_deformable_histories'] == (11 if mode == 'on' else 0)
            cases[side][mode] = row
            geometry[f'{side}_{mode}'] = accepted_geometry(side, positions, source)
            face_test = geometry[f'{side}_{mode}']['face_intersections']
            assert face_test['segment_or_polygon_crossing_pairs'] == 0
            assert face_test['point_contact_pairs'] == 0
            (EVIDENCE / f'{side}-{mode}.f32le.gz').write_bytes(
                gzip.compress(positions, compresslevel=9, mtime=0))
            cases[side][f'{mode}_positions_sha256'] = digest(positions)
        on = gzip.decompress((EVIDENCE / f'{side}-on.f32le.gz').read_bytes())
        off = gzip.decompress((EVIDENCE / f'{side}-off.f32le.gz').read_bytes())
        a = np.frombuffer(on, dtype='<f4').reshape((-1, 3))
        b = np.frombuffer(off, dtype='<f4').reshape((-1, 3))
        delta = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=1)
        cases[side]['contact_ab'] = {
            'changed_ptc_nodes': int(np.count_nonzero(delta[:26121])),
            'changed_fmc_nodes': int(np.count_nonzero(delta[26121:])),
            'maximum_position_delta_m': float(delta.max()),
        }
    replay, replay_positions = native_case(binary, 'left', 'replay')
    assert replay == cases['left']['on']
    assert digest(replay_positions) == cases['left']['on_positions_sha256']
    receipt = {
        'schema': 'numi.human.patellofemoral-full-native-step.v3',
        'status': 'bounded_source_cartilage_contact_preflight',
        'pose_status': 'unadopted_candidate',
        'native_source_revision': subprocess.run(
            ['git', '-C', str(matter_root), 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True).stdout.strip(),
        'source_inputs': {'20um': inputs20, '14um': inputs14},
        'native_identity_sha256': {
            'executable': sha(binary), 'metallib': sha(metallib),
            'material': sha(material),
            'contact_shader_source': sha(matter_root / 'matter/src/metal/contact.metalinc'),
            'runtime_source': sha(matter_root / 'matter/src/runtime.mm'),
            'probe_source': sha(matter_root / 'matter/tools/patellofemoral_full_surface_step.mm'),
        },
        'cases': cases,
        'accepted_full_boundary_geometry': geometry,
        'left_contact_on_replay_byte_identical': True,
        'source_fmc_vertex_link_manifold_defect_node_id': 233523,
        'source_fmc_defect_tetrahedral_star': fmc_defect_star(source),
        'source_cartilage_tetrahedral_boundary_chain': source_cartilage_chain(source),
        'limits': [
            'FMC source boundary has a two-fan vertex link; zero face crossings does not qualify disjoint volumes.',
            'Exact ray parity found sampled boundary points outside the opposite surface; FMC is not an embedded manifold solid, so this is not volume qualification.',
            'The source FMC defect has a connected annular tetrahedral star; splitting its vertex alone would sever valid tetrahedral adjacency.',
            'Exact self-intersection and tetrahedral orientation checks pass for retained states, but these do not calibrate material or physiological load.',
            'PTC current pose is an unadopted 20 micrometer candidate; patellar bone, PTB, QAT and PTL attachments are absent.',
            'The material uses a synthetic density and an unvalidated isotropic energy; no source FEBio material equivalence was established.',
            'One 1 microsecond zero-gravity step with 0.1 m/s prescribed approach is not sustained or physiological load.',
            'Active history count is not pressure or force; accepted barrier impulse field was zero.',
            'Only kinetic energy is reported; strain, contact work and whole-system energy closure are not established.',
            'The conservative FP32 final-state guard rejects unresolved near-plane vertices; it does not establish general clinical nonpenetration.',
        ],
        'loaded_knee_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    path = EVIDENCE / 'receipt.json'
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def verify() -> dict:
    receipt = json.loads((EVIDENCE / 'receipt.json').read_text())
    assert receipt['schema'] == 'numi.human.patellofemoral-full-native-step.v3'
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    assert fmc_defect_star(source) == receipt['source_fmc_defect_tetrahedral_star']
    assert source_cartilage_chain(source) == \
        receipt['source_cartilage_tetrahedral_boundary_chain']
    pose_path = ROOT / 'Docs/media/patellofemoral-pose-clearance-20260930/receipt.json'
    pose = json.loads(pose_path.read_text())
    assert pose['prescribed_translation_magnitude_m'] == 20.0e-6
    for file, expected in pose['source_files_sha256'].items():
        assert sha(ROOT / 'Sources/open-knee-oks003' / file) == expected
    for side in ('left', 'right'):
        row = receipt['source_inputs']['20um'][side]
        assert row['pose_receipt_sha256'] == sha(pose_path)
        assert row['nhknee_payload_sha256'] == pose['sides'][side]['payload_sha256']
    for side in ('left', 'right'):
        for mode in ('on', 'off'):
            positions = gzip.decompress((EVIDENCE / f'{side}-{mode}.f32le.gz').read_bytes())
            assert digest(positions) == receipt['cases'][side][f'{mode}_positions_sha256']
            assert accepted_geometry(side, positions, source) == \
                receipt['accepted_full_boundary_geometry'][f'{side}_{mode}']
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--matter-root', type=Path)
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    if args.capture:
        assert args.matter_root and args.build_dir
        output = capture(args.matter_root.resolve(), args.build_dir.resolve())
    else:
        output = verify()
    print(json.dumps({'schema': output['schema'], 'status': output['status'],
                      'loaded_knee_qualified': output['loaded_knee_qualified']},
                     sort_keys=True))
