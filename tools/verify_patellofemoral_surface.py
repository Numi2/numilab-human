"""Independently check source and compiled patellofemoral face orientation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from numilab_human.open_knee import (
    EXPECTED_HASHES, HEADER_STRUCT, NODE_SET_STRUCT, NODE_STRUCT,
    REGION_STRUCT, SURFACE_PAIR_STRUCT, SURFACE_STRUCT, TETRAHEDRON_STRUCT,
    parse_source,
)


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'Build/patellofemoral-surface-20260930'
SURFACES = {
    'PTC': ('PTC_@_FMC_ContactFaces', 'PTC_@_PTB_TiesFaces'),
    'FMC': ('FMC_@_PTC_ContactFaces', 'FMC_@_FMB_TiesFaces'),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError('patellofemoral surface: ' + message)


def name(value: bytes) -> str:
    return value.split(b'\0', 1)[0].decode('ascii')


def payload(path: Path, manifest: dict, source) -> dict:
    raw = path.read_bytes()
    require(sha(path) == manifest['payload']['sha256'] and
            len(raw) == manifest['payload']['bytes'], 'published payload identity')
    (magic, abi, header_bytes, region_count, node_count, tet_count,
     surface_count, face_count, node_set_count, _membership_count,
     pair_count, side, reserved, source_hashes) = HEADER_STRUCT.unpack_from(raw)
    require(magic == b'NHKNEE1\0' and abi == 3 and reserved == 0 and
            header_bytes == HEADER_STRUCT.size and side in (0, 1) and
            source_hashes == b''.join(bytes.fromhex(EXPECTED_HASHES[file]) for file in
                                       ('Geometry.feb', 'ModelProperties.xml',
                                        'FeBio_custom.feb', 'license.txt')),
            'native payload header and source identity')
    region_offset = header_bytes
    surface_offset = region_offset + region_count * REGION_STRUCT.size
    node_offset = (surface_offset + surface_count * SURFACE_STRUCT.size +
                   node_set_count * NODE_SET_STRUCT.size +
                   pair_count * SURFACE_PAIR_STRUCT.size)
    tet_offset = node_offset + node_count * NODE_STRUCT.size
    face_offset = tet_offset + tet_count * TETRAHEDRON_STRUCT.size
    regions = {
        name(values[0]): {'first_node': values[3], 'node_count': values[4],
                          'first_tet': values[5], 'tet_count': values[6]}
        for index in range(region_count)
        for values in (REGION_STRUCT.unpack_from(
            raw, region_offset + index * REGION_STRUCT.size),)
    }
    surfaces = {
        name(values[0]): {'region_index': values[1], 'first_face': values[2],
                          'face_count': values[3]}
        for index in range(surface_count)
        for values in (SURFACE_STRUCT.unpack_from(
            raw, surface_offset + index * SURFACE_STRUCT.size),)
    }
    positions = np.ndarray((node_count, 3), dtype='<f4', buffer=raw,
                           offset=node_offset,
                           strides=(NODE_STRUCT.size, 4)).astype(np.float64)
    tetrahedra = np.frombuffer(raw, '<u4', count=tet_count * 4,
                               offset=tet_offset).reshape(-1, 4)
    faces = np.frombuffer(raw, '<u4', count=face_count * 3,
                          offset=face_offset).reshape(-1, 3)
    require(set(SURFACES).issubset(regions) and
            all(surface in surfaces for pair in SURFACES.values()
                for surface in pair), 'patellofemoral source regions and surfaces')
    reflected = side == 1
    for region_name, names in SURFACES.items():
        region = regions[region_name]
        source_region = source.regions[region_name]
        require(region['node_count'] == len(source_region.node_ids) and
                region['tet_count'] == len(source_region.elements),
                f'{region_name} source region counts')
        local = {identifier: index for index, identifier in
                 enumerate(source_region.node_ids)}
        candidate_tets = tetrahedra[
            region['first_tet']:region['first_tet'] + region['tet_count']]
        expected_tets = np.array([
            [region['first_node'] + local[identifier]
             for identifier in ((cell[1], cell[0], cell[2], cell[3])
                                if reflected else cell)]
            for cell in source_region.elements], dtype=np.uint32)
        require(np.array_equal(candidate_tets, expected_tets),
                f'{region_name} exact source tetrahedron order/parity')
        for surface_name in names:
            row = surfaces[surface_name]
            candidate = faces[row['first_face']:row['first_face'] + row['face_count']]
            expected = np.array([
                [region['first_node'] + local[identifier]
                 for identifier in ((face[1], face[0], face[2])
                                    if reflected else face)]
                for face in source.surfaces[surface_name].faces], dtype=np.uint32)
            require(np.array_equal(candidate, expected),
                    f'{surface_name} exact source face order/parity')
    return {'raw': raw, 'side': 'right' if reflected else 'left',
            'regions': regions, 'surfaces': surfaces, 'positions': positions,
            'tetrahedra': tetrahedra, 'faces': faces}


def measured_surfaces(positions: np.ndarray, tetrahedra: dict[str, np.ndarray],
                      surfaces: dict[str, np.ndarray], anterior: np.ndarray) -> dict:
    result = {}
    for region_name, surface_names in SURFACES.items():
        tets = tetrahedra[region_name]
        wanted = {tuple(sorted(face)) for surface_name in surface_names
                  for face in surfaces[surface_name]}
        incidence: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for tet in tets:
            for opposite in range(4):
                key = tuple(sorted(int(tet[corner]) for corner in range(4)
                                   if corner != opposite))
                if key in wanted:
                    incidence[key].append(int(tet[opposite]))
        for surface_name in surface_names:
            faces = surfaces[surface_name]
            area_normal = np.zeros(3)
            total_area = 0.0
            for face in faces:
                owner = incidence.get(tuple(sorted(int(i) for i in face)), [])
                require(len(owner) == 1,
                        f'{surface_name} is not an exterior one-owner face')
                p = positions[face]
                normal = np.cross(p[1] - p[0], p[2] - p[0])
                length = float(np.linalg.norm(normal))
                require(np.isfinite(length) and length > 0 and
                        float(normal @ (positions[owner[0]] - p[0])) < 0,
                        f'{surface_name} has an inward or degenerate face')
                area_normal += normal
                total_area += length
            result[surface_name] = {
                'faces': len(faces), 'all_exterior_outward': True,
                'area_weighted_anterior_cosine': float(area_normal @ anterior /
                                                        total_area),
                'area_m2': total_area / 2,
            }
    require(result['PTC_@_FMC_ContactFaces']['area_weighted_anterior_cosine'] < -.7 and
            result['PTC_@_PTB_TiesFaces']['area_weighted_anterior_cosine'] > .7,
            'patellar cartilage contact/tie facing order')
    return result


def proximity(positions: np.ndarray, surfaces: dict[str, np.ndarray],
              close_distance: float) -> dict:
    def data(name: str):
        p = positions[surfaces[name]]
        normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        length = np.linalg.norm(normal, axis=1)
        require(bool(np.all(length > 0)), f'{name} nondegenerate contact faces')
        return p.mean(axis=1), normal / length[:, None]
    patella, patella_normal = data('PTC_@_FMC_ContactFaces')
    femur, femur_normal = data('FMC_@_PTC_ContactFaces')
    distance, nearest = cKDTree(femur).query(patella, k=1)
    near = distance < close_distance
    require(int(np.count_nonzero(near)) >= 1000,
            'source-defined near patellofemoral patch absent')
    direction = femur[nearest] - patella
    direction /= np.maximum(distance[:, None], 1e-15)
    opposing = np.einsum('ij,ij->i', patella_normal, femur_normal[nearest])
    patella_toward = np.einsum('ij,ij->i', patella_normal, direction)
    femur_toward = np.einsum('ij,ij->i', femur_normal[nearest], -direction)
    result = {
        'near_centroid_distance_limit_m': close_distance,
        'near_patellar_face_count': int(np.count_nonzero(near)),
        'near_median_centroid_distance_m': float(np.median(distance[near])),
        'near_normal_opposition_below_minus_half_fraction': float(
            np.mean(opposing[near] < -.5)),
        'near_patella_normal_toward_femur_fraction': float(
            np.mean(patella_toward[near] > 0)),
        'near_femur_normal_toward_patella_fraction': float(
            np.mean(femur_toward[near] > 0)),
    }
    require(result['near_normal_opposition_below_minus_half_fraction'] > .95 and
            result['near_patella_normal_toward_femur_fraction'] > .98 and
            result['near_femur_normal_toward_patella_fraction'] > .99,
            'near patellofemoral face normals do not oppose and face each other')
    return result


def run() -> dict:
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    source_positions = {}
    source_tetrahedra = {}
    source_surfaces = {}
    source_offset = 0
    for region_name in SURFACES:
        region = source.regions[region_name]
        local = {identifier: source_offset + index for index, identifier in
                 enumerate(region.node_ids)}
        source_positions[region_name] = np.asarray(region.nodes_mm) * .001
        source_tetrahedra[region_name] = np.asarray([
            [local[index] for index in tet] for tet in region.elements],
            dtype=np.uint32)
        for surface_name in SURFACES[region_name]:
            source_surfaces[surface_name] = np.asarray([
                [local[index] for index in face]
                for face in source.surfaces[surface_name].faces],
                dtype=np.uint32)
        source_offset += len(region.node_ids)
    source_points = np.vstack([source_positions[name] for name in SURFACES])
    source_anterior = np.asarray(source.landmarks['Yf_axis'], dtype=float)
    source_anterior /= np.linalg.norm(source_anterior)
    source_rows = measured_surfaces(source_points, source_tetrahedra,
                                    source_surfaces, source_anterior)
    source_near = proximity(source_points, source_surfaces, .001)
    compiled = {}
    for side, stem in [('left', 'open-knee-oks003-left'),
                       ('right', 'open-knee-oks003-right-mirrored')]:
        directory = BUILD / side
        manifest_path = directory / (stem + '.manifest.json')
        payload_path = directory / (stem + '.nhknee')
        manifest = json.loads(manifest_path.read_text())
        decoded = payload(payload_path, manifest, source)
        require(decoded['side'] == side, 'compiled side identity')
        tets = {name: decoded['tetrahedra'][
            row['first_tet']:row['first_tet'] + row['tet_count']]
            for name, row in decoded['regions'].items() if name in SURFACES}
        faces = {name: decoded['faces'][
            row['first_face']:row['first_face'] + row['face_count']]
            for name, row in decoded['surfaces'].items()
            if any(name in names for names in SURFACES.values())}
        rows = measured_surfaces(decoded['positions'], tets, faces,
                                 np.asarray([0., -1., 0.]))
        near = proximity(decoded['positions'], faces, .001)
        compiled[side] = {
            'payload_sha256': sha(payload_path),
            'manifest_sha256': sha(manifest_path),
            'outward_surfaces': rows,
            'near_contact_patch': near,
        }
    require(compiled['left']['near_contact_patch']['near_patellar_face_count'] ==
            compiled['right']['near_contact_patch']['near_patellar_face_count'],
            'bilateral mirror near-patch count')
    result = {
        'schema': 'numi.human.patellofemoral-surface-orientation.v1',
        'status': 'passed_source_and_bilateral_compiled_static_face_winding',
        'source_file_sha256': {file: sha(ROOT / 'Sources/open-knee-oks003' / file)
                               for file in EXPECTED_HASHES},
        'source': {'outward_surfaces': source_rows,
                   'near_contact_patch': source_near},
        'compiled': compiled,
        'source_compiler_sha256': sha(ROOT / 'src/numilab_human/open_knee.py'),
        'verifier_sha256': sha(Path(__file__)),
        'cartilage_face_winding_qualified': True,
        'static_near_patch_facing_qualified': True,
        'contact_gap_qualified': False,
        'contact_pressure_qualified': False,
        'loaded_knee_qualified': False,
        'clinical_anatomy_qualified': False,
        'boundary': ('All selected patellar and femoral cartilage contact/tie '
                     'faces are exterior and outward in the exact source and '
                     'both compiled sides. Close centroid pairs mostly face '
                     'each other; centroid proximity is not an exact surface '
                     'gap or loaded contact law.'),
    }
    output = ROOT / 'Docs/media/patellofemoral-surface-20260930/receipt.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result


if __name__ == '__main__':
    measured = run()
    print(json.dumps({'status': measured['status'],
                      'left_near': measured['compiled']['left']['near_contact_patch'],
                      'right_near': measured['compiled']['right']['near_contact_patch']}))
