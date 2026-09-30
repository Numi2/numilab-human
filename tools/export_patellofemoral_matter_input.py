"""Export exact bilateral cartilage FEM regions for a bounded native Matter step.

The candidate shifts only PTC's initial current position. Its source rest
positions remain byte-for-byte tied to NHKNEE1. This is not an adopted pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from numilab_human.open_knee import parse_source
from tools.verify_patellofemoral_surface import payload


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'Build/full-patellofemoral-matter-20260930'
DEFAULT_POSE = ROOT / 'Docs/media/patellofemoral-pose-clearance-20260930/receipt.json'
HEADER = struct.Struct('<8s7I32s32s3d')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region(decoded: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    row = decoded['regions'][name]
    start, count = row['first_node'], row['node_count']
    first, size = row['first_tet'], row['tet_count']
    positions = np.asarray(decoded['positions'][start:start + count], dtype='<f4')
    tetrahedra = np.asarray(decoded['tetrahedra'][first:first + size], dtype='<u4')
    assert positions.shape == (count, 3) and tetrahedra.shape == (size, 4)
    assert np.all((tetrahedra >= start) & (tetrahedra < start + count))
    return positions, np.asarray(tetrahedra - start, dtype='<u4')


def export(side: str, source, pose: dict, pose_path: Path) -> dict:
    stem = ('open-knee-oks003-left' if side == 'left'
            else 'open-knee-oks003-right-mirrored')
    folder = ROOT / 'Build/patellofemoral-surface-20260930' / side
    input_path = folder / f'{stem}.nhknee'
    manifest_path = folder / f'{stem}.manifest.json'
    decoded = payload(input_path, json.loads(manifest_path.read_text()), source)
    assert pose['sides'][side]['payload_sha256'] == sha(input_path)
    for name in ('PTC', 'FMC'):
        material = source.materials[name]
        assert material['type'] == 'Mooney-Rivlin'
        assert (float(material['c1']), float(material['c2']), float(material['k'])) \
            == (2.54, 0.0, 100.0)
    ptc, ptc_tets = region(decoded, 'PTC')
    fmc, fmc_tets = region(decoded, 'FMC')
    shift = pose['sides'][side]['prescribed_pose_translation_m']
    current = np.asarray(ptc.astype(np.float64) + np.asarray(shift), dtype='<f4')
    assert np.all(np.isfinite(current))
    assert np.array_equal(current.astype(np.float64),
        (ptc.astype(np.float64) + np.asarray(shift)).astype('<f4').astype(np.float64))
    header = HEADER.pack(b'NHCAR1\0\0', 1, 0 if side == 'left' else 1,
                         len(ptc), len(ptc_tets), len(fmc), len(fmc_tets), 0,
                         bytes.fromhex(sha(input_path)), bytes.fromhex(sha(pose_path)),
                         *shift)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shift_um = round(pose['prescribed_translation_magnitude_m'] * 1e6)
    suffix = '' if pose_path == DEFAULT_POSE else f'-{shift_um}um'
    output = OUTPUT / f'open-knee-{side}-ptc-fmc{suffix}.nhcar'
    with output.open('wb') as stream:
        stream.write(header)
        for block in (ptc, current, ptc_tets, fmc, fmc_tets):
            stream.write(block.tobytes(order='C'))
    expected_bytes = HEADER.size + 12 * (2 * len(ptc) + len(fmc)) \
        + 16 * (len(ptc_tets) + len(fmc_tets))
    assert output.stat().st_size == expected_bytes
    return {
        'schema': 'numi.human.patellofemoral-matter-input.v1',
        'side': side,
        'input_sha256': sha(output),
        'input_bytes': expected_bytes,
        'nhknee_payload_sha256': sha(input_path),
        'nhknee_manifest_sha256': sha(manifest_path),
        'pose_receipt_sha256': sha(pose_path),
        'ptc_nodes': len(ptc), 'ptc_tetrahedra': len(ptc_tets),
        'fmc_nodes': len(fmc), 'fmc_tetrahedra': len(fmc_tets),
        'source_cartilage_c1_mpa': 2.54,
        'source_cartilage_c2_mpa': 0.0,
        'source_cartilage_bulk_mpa': 100.0,
        'boundary': 'Source FEM topology and rest positions plus an unadopted PTC current-pose candidate. Does not include patellar bone/tendon attachment, density, qualified source material execution, or load.',
    }


def run(pose_path: Path = DEFAULT_POSE) -> dict:
    pose_path = pose_path.resolve()
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    pose = json.loads(pose_path.read_text())
    assert pose['status'] == 'unadopted_bilateral_geometric_pose_candidate'
    receipt = {'left': export('left', source, pose, pose_path),
               'right': export('right', source, pose, pose_path)}
    shift_um = round(pose['prescribed_translation_magnitude_m'] * 1e6)
    receipt_path = OUTPUT / ('receipt.json' if pose_path == DEFAULT_POSE
                             else f'receipt-{shift_um}um.json')
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose-receipt', type=Path, default=DEFAULT_POSE)
    print(json.dumps(run(parser.parse_args().pose_receipt), sort_keys=True))
