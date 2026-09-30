"""Exact PTC/FMC volume-separation audit of retained native cartilage states.

Every input coordinate is the exact value of a retained binary32 metre. An
integer-grid broadphase finds every exact AABB pair; integer separating-axis
predicates classify the convex tetrahedra. Tetrahedral domains can be tested
directly even when an exposed boundary has a nonmanifold vertex.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from numilab_human.open_knee import parse_source


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Sources/open-knee-oks003'
EVIDENCE = ROOT / 'Docs/media/patellofemoral-full-native-20260930'
WORK = ROOT / 'Build/full-patellofemoral-matter-20260930'
OUTPUT = ROOT / 'Docs/media/patellofemoral-tetrahedral-separation-20260930/receipt.json'
SCALE = 1 << 40  # Exact for the retained Float32 positions in this SI range.
CELL = SCALE // 500  # Approximately 2 mm; integer arithmetic throughout.
HEADER = struct.Struct('<8s7I32s32s3d')
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def difference(a: tuple[int, int, int],
               b: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def cross(a: tuple[int, int, int],
          b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def tetrahedron_relation(first: tuple[tuple[int, int, int], ...],
                         second: tuple[tuple[int, int, int], ...]) -> str:
    """Closed convex SAT: separated, boundary contact, or interior overlap."""
    a_edges = [difference(first[j], first[i]) for i, j in EDGES]
    b_edges = [difference(second[j], second[i]) for i, j in EDGES]
    axes = [cross(difference(t[j], t[i]), difference(t[k], t[i]))
            for t in (first, second) for i, j, k in FACES]
    axes += [cross(a, b) for a in a_edges for b in b_edges]
    touching = False
    for axis in axes:
        if axis == (0, 0, 0):
            continue
        a = [dot(axis, point) for point in first]
        b = [dot(axis, point) for point in second]
        if max(a) < min(b) or max(b) < min(a):
            return 'separated'
        touching |= max(a) == min(b) or max(b) == min(a)
    return 'boundary_contact' if touching else 'interior_overlap'


def self_check() -> None:
    unit = ((0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4))
    far = tuple((x + 5, y, z) for x, y, z in unit)
    face = tuple((x + 4, y, z) for x, y, z in unit)
    near = tuple((x + 1, y, z) for x, y, z in unit)
    inside = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))
    assert tetrahedron_relation(unit, far) == 'separated'
    assert tetrahedron_relation(unit, face) == 'boundary_contact'
    assert tetrahedron_relation(unit, near) == 'interior_overlap'
    assert tetrahedron_relation(unit, inside) == 'interior_overlap'
    assert tetrahedron_relation(unit, unit) == 'interior_overlap'
    # Face normals alone miss this skew edge-edge separation.
    skew_a = ((-2, 5, -3), (3, 1, -5), (5, -4, -3), (4, -5, -1))
    skew_b = ((-5, -1, 2), (4, 1, 1), (1, 4, 2), (-3, 0, -4))
    for tet in (skew_a, skew_b):
        for i, j, k in FACES:
            axis = cross(difference(tet[j], tet[i]),
                         difference(tet[k], tet[i]))
            a = [dot(axis, point) for point in skew_a]
            b = [dot(axis, point) for point in skew_b]
            assert max(a) >= min(b) and max(b) >= min(a)
    assert tetrahedron_relation(skew_a, skew_b) == 'separated'


def exact_lattice(points: np.ndarray) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float64) * SCALE
    if not np.isfinite(scaled).all() or not np.array_equal(scaled, np.rint(scaled)):
        raise ValueError('retained coordinate is not exact on the binary32 lattice')
    if np.max(np.abs(scaled)) >= 2**62:
        raise ValueError('retained coordinate exceeds safe lattice range')
    return scaled.astype(np.int64)


def source_tetrahedra(source) -> tuple[np.ndarray, np.ndarray]:
    cells = []
    for region in ('PTC', 'FMC'):
        row = source.regions[region]
        index = {identifier: local for local, identifier in enumerate(row.node_ids)}
        cells.append(np.asarray([[index[identifier] for identifier in tet]
                                 for tet in row.elements], dtype=np.int64))
    assert cells[0].shape == (121105, 4) and cells[1].shape == (87072, 4)
    return cells[0], cells[1]


def verify_native_input(side: str, retained: dict,
                        ptc_tet: np.ndarray, fmc_tet: np.ndarray) -> str:
    """Bind the source-order cells to the exact native NHCAR1 input bytes."""
    path = WORK / f'open-knee-{side}-ptc-fmc.nhcar'
    raw = path.read_bytes()
    digest = sha(raw)
    assert digest == retained['source_inputs']['20um'][side]['input_sha256']
    header = HEADER.unpack_from(raw)
    assert header[0] == b'NHCAR1\0\0' and header[1] == 1
    assert header[2] == (0 if side == 'left' else 1)
    assert tuple(header[3:7]) == (26121, 121105, 24870, 87072)
    offset = HEADER.size + 2 * 12 * 26121
    first = np.frombuffer(raw, dtype='<u4', count=4 * 121105,
                          offset=offset).reshape(-1, 4)
    offset += first.nbytes + 12 * 24870
    second = np.frombuffer(raw, dtype='<u4', count=4 * 87072,
                           offset=offset).reshape(-1, 4)
    assert offset + second.nbytes == len(raw)
    assert np.array_equal(first, ptc_tet) and np.array_equal(second, fmc_tet)
    return digest


def aabb_candidates(a_min: np.ndarray, a_max: np.ndarray,
                    b_min: np.ndarray, b_max: np.ndarray):
    """Every closed AABB pair, with an integer XY grid and exact XYZ filter."""
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    for j in range(len(b_min)):
        for x in range(int(b_min[j, 0] // CELL), int(b_max[j, 0] // CELL) + 1):
            for y in range(int(b_min[j, 1] // CELL), int(b_max[j, 1] // CELL) + 1):
                bins[x, y].append(j)
    global_min = b_min.min(axis=0)
    global_max = b_max.max(axis=0)
    for i in range(len(a_min)):
        if np.any(a_max[i] < global_min) or np.any(a_min[i] > global_max):
            continue
        possible: set[int] = set()
        for x in range(int(a_min[i, 0] // CELL), int(a_max[i, 0] // CELL) + 1):
            for y in range(int(a_min[i, 1] // CELL), int(a_max[i, 1] // CELL) + 1):
                possible.update(bins.get((x, y), ()))
        for j in sorted(possible):
            if np.all(b_min[j] <= a_max[i]) and np.all(b_max[j] >= a_min[i]):
                yield i, j


def audit_state(positions: bytes, ptc_tet: np.ndarray,
                fmc_tet: np.ndarray) -> dict:
    if len(positions) != 50991 * 3 * 4:
        raise ValueError('retained native state has unexpected node count')
    nodes = np.frombuffer(positions, dtype='<f4').reshape((50991, 3))
    integer = exact_lattice(nodes)
    a_points = integer[:26121]
    b_points = integer[26121:]
    a = a_points[ptc_tet]
    b = b_points[fmc_tet]

    def exact_orientation(cells: np.ndarray) -> dict:
        positive = negative = zero = 0
        minimum_absolute = None
        for cell in cells:
            p = [tuple(int(value) for value in vertex) for vertex in cell]
            signed = dot(cross(difference(p[1], p[0]),
                               difference(p[2], p[0])), difference(p[3], p[0]))
            positive += signed > 0
            negative += signed < 0
            zero += signed == 0
            if signed:
                minimum_absolute = (abs(signed) if minimum_absolute is None else
                                    min(minimum_absolute, abs(signed)))
        if zero or (positive and negative):
            raise ValueError('retained tetrahedral orientation is degenerate or mixed')
        return {'positive': positive, 'negative': negative, 'zero': zero,
                'minimum_absolute_six_volume_lattice': minimum_absolute}

    orientation = {'PTC': exact_orientation(a), 'FMC': exact_orientation(b)}
    a_min, a_max = a.min(axis=1), a.max(axis=1)
    b_min, b_max = b.min(axis=1), b.max(axis=1)

    @lru_cache(maxsize=10000)
    def exact_tet(region: int, index: int) -> tuple[tuple[int, int, int], ...]:
        values = a[index] if region == 0 else b[index]
        return tuple(tuple(int(value) for value in point) for point in values)

    pair_count = contact_count = overlap_count = 0
    first_contacts = []
    first_overlaps = []
    for i, other in aabb_candidates(a_min, a_max, b_min, b_max):
        pair_count += 1
        relation = tetrahedron_relation(exact_tet(0, i), exact_tet(1, other))
        if relation == 'boundary_contact':
            contact_count += 1
            if len(first_contacts) < 12:
                first_contacts.append([i, other])
        elif relation == 'interior_overlap':
            overlap_count += 1
            if len(first_overlaps) < 12:
                first_overlaps.append([i, other])
    return {
        'aabb_candidate_tetrahedron_pairs': pair_count,
        'exact_boundary_contact_pairs': contact_count,
        'exact_interior_overlap_pairs': overlap_count,
        'first_boundary_contact_pairs': first_contacts,
        'first_interior_overlap_pairs': first_overlaps,
        'exact_tetrahedral_orientation': orientation,
        'solid_domains_disjoint': contact_count == overlap_count == 0,
    }


def run(output: Path) -> dict:
    self_check()
    source = parse_source(SOURCE)
    ptc_tet, fmc_tet = source_tetrahedra(source)
    retained = json.loads((EVIDENCE / 'receipt.json').read_text())
    loop = json.loads((ROOT / 'Docs/media/patellofemoral-loop-20260930/receipt.json').read_text())
    source_files = {name: sha((SOURCE / name).read_bytes()) for name in
                    ('Geometry.feb', 'FeBio_custom.feb', 'ModelProperties.xml', 'license.txt')}
    assert source_files == loop['source_file_sha256']
    assert retained['schema'] == 'numi.human.patellofemoral-full-native-step.v3'
    assert retained['left_contact_on_replay_byte_identical'] is True
    states = {}
    input_sha256 = {}
    for side in ('left', 'right'):
        states[side] = {}
        # NHKNEE1 reverses one local corner on the declared right mirror;
        # the native NHCAR1 exporter preserves that compiled parity.
        side_ptc = ptc_tet if side == 'left' else ptc_tet[:, [1, 0, 2, 3]]
        side_fmc = fmc_tet if side == 'left' else fmc_tet[:, [1, 0, 2, 3]]
        input_sha256[side] = verify_native_input(side, retained,
                                                 side_ptc, side_fmc)
        for mode in ('on', 'off'):
            compressed = (EVIDENCE / f'{side}-{mode}.f32le.gz').read_bytes()
            positions = gzip.decompress(compressed)
            assert sha(positions) == retained['cases'][side][f'{mode}_positions_sha256']
            states[side][mode] = {
                'accepted_positions_sha256': sha(positions),
                'compressed_file_sha256': sha(compressed),
                **audit_state(positions, side_ptc, side_fmc),
            }
    clear = all(row['solid_domains_disjoint'] for side in states.values()
                for row in side.values())
    result = {
        'schema': 'numi.human.patellofemoral-tetrahedral-separation.v1',
        'status': ('retained_candidate_cartilage_volumes_disjoint' if clear else
                   'failed_cartilage_volume_separation'),
        'algorithm': 'exact_binary32_integer_aabb_and_tetrahedron_separating_axes_v1',
        'predicate_self_checks_passed': 6,
        'producer_sha256': sha(Path(__file__).read_bytes()),
        'source_file_sha256': source_files,
        'full_native_receipt_sha256': sha((EVIDENCE / 'receipt.json').read_bytes()),
        'native_input_sha256': input_sha256,
        'native_input_tetrahedron_order_verified': True,
        'broadphase_cell_lattice': CELL,
        'ptc_tetrahedra': len(ptc_tet),
        'fmc_tetrahedra': len(fmc_tet),
        'states': states,
        'all_retained_native_states_volume_disjoint': clear,
        'clinical_anatomy_qualified': False,
        'loaded_contact_qualified': False,
        'boundary': ('This tests exact separation of the two retained cartilage '
                     'tetrahedral domains only. The 20 um patellar-cartilage pose '
                     'is unadopted and omits patellar bone, QAT/PTL tie, source '
                     'material equivalence, physiological load, pressure and energy.'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    row = run(args.output)
    print(json.dumps({'status': row['status'], 'states': {
        side: {mode: {'candidates': item['aabb_candidate_tetrahedron_pairs'],
                      'contacts': item['exact_boundary_contact_pairs'],
                      'overlaps': item['exact_interior_overlap_pairs']}
               for mode, item in cases.items()}
        for side, cases in row['states'].items()}}, sort_keys=True))
    raise SystemExit(0 if row['all_retained_native_states_volume_disjoint'] else 2)
