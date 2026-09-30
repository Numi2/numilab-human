"""Run one exact Rodero source cell through Matter's active FEM ingress."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from . import cardiac_active_force_gate as reference
from . import cardiac_active_tension_gate as tension_gate
from . import cardiac_material_frames as frames


CELL = 1


def run(asset: Path, activation: Path, tension: Path, native_repo: Path,
        binary: Path, output: Path) -> dict:
    asset, activation, tension, native_repo, binary, output = (
        Path(p).resolve() for p in
        (asset, activation, tension, native_repo, binary, output))
    gate = tension_gate.audit(asset, activation, tension)
    gate_text = json.dumps(gate, sort_keys=True, indent=2) + '\n'
    reference.require((tension / 'independent-gate.json').read_text() == gate_text,
                      'published full-source tension gate changed')
    source_manifest = reference.human.read_json(asset / 'manifest.json')
    sheet_record = source_manifest['buffers']['sheets.f64le']
    reference.checked_file(asset / 'sheets.f64le',
                           sheet_record['sha256'], sheet_record['bytes'])
    labels = np.fromfile(asset / 'labels.u32le', '<u4')
    tetrahedra = np.fromfile(asset / 'tetrahedra.u32le', '<u4').reshape(-1, 4)
    positions = np.fromfile(asset / 'nodes.f64le', '<f8').reshape(-1, 3)
    fibres = np.fromfile(asset / 'fibres.f64le', '<f8').reshape(-1, 3)
    sheets = np.fromfile(asset / 'sheets.f64le', '<f8').reshape(-1, 3)
    summary = reference.human.read_json(tension / 'summary.json')
    frame = summary['frames'][0]
    reference.require(frame['time_ms'] == 100 and labels[CELL] == 1,
                      'pinned first LV source cell')
    cell_tension = np.fromfile(tension / frame['file'], '<f4')[CELL]
    quaternion, diagnostics = frames.convert_axes(
        tuple(fibres[CELL]), tuple(sheets[CELL]))
    nodes = positions[tetrahedra[CELL]]
    reference.require(cell_tension > 0 and
                      diagnostics['fibre_direction_correction_radians'] == 0.0,
                      'positive source tension and unchanged fibre direction')
    reference.require(binary.is_file() and not binary.is_symlink(),
                      'native Matter probe binary')
    revision = subprocess.run(['git', '-C', str(native_repo), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
    native_status = subprocess.run(['git', '-C', str(native_repo), 'status',
                                    '--porcelain=v1', '--untracked-files=no'],
                                   check=True, capture_output=True, text=True).stdout
    reference.require(not native_status, 'native source has uncommitted tracked changes')
    arguments = [str(binary), '--cell',
                 *(str(float(v)) for v in nodes.ravel()),
                 *(str(float(v)) for v in quaternion),
                 str(float(cell_tension))]
    runs = []
    for _ in range(2):
        completed = subprocess.run(arguments, check=True,
                                   capture_output=True, text=True)
        reference.require(not completed.stderr and
                          len(completed.stdout.strip().splitlines()) == 1,
                          'native source-cell stdout/stderr')
        runs.append(completed.stdout)
    reference.require(runs[0] == runs[1], 'native source-cell run replay')
    measured = json.loads(runs[0])
    reference.require(measured['abi'] == 35
                      and measured['device'].startswith('Apple M')
                      and measured['source_cell'] is True
                      and measured['accepted_active_steps'] == 1
                      and measured['baseline_code'] == 0
                      and measured['active_code'] == 0
                      and measured['rejected_code'] == 3
                      and measured['replay_bitwise'] is True
                      and measured['tip_delta_norm_m'] > 0,
                      'native accepted source-cell transaction and controls')
    result = {
        'schema': 'numi.human.cardiac-active-native-source-cell.v1',
        'status': 'accepted_bounded_source_cell_active_fem_step',
        'source_asset_manifest_sha256': reference.sha(asset / 'manifest.json'),
        'source_sheets_sha256': reference.sha(asset / 'sheets.f64le'),
        'candidate_activation_summary_sha256': reference.sha(activation / 'summary.json'),
        'tension_summary_sha256': reference.sha(tension / 'summary.json'),
        'full_source_tension_gate_sha256': reference.sha(
            tension / 'independent-gate.json'),
        'native_source_revision': revision,
        'native_probe_sha256': reference.sha(binary),
        'source_cell_index': CELL,
        'source_label': int(labels[CELL]),
        'source_node_ids': tetrahedra[CELL].tolist(),
        'source_positions_m': nodes.tolist(),
        'derived_material_frame_xyzw': [float(v) for v in quaternion],
        'source_candidate_tension_pa': float(cell_tension),
        'time_ms': 100,
        'native_result': measured,
        'native_stdout': runs[0].strip(),
        'outer_run_replay_bitwise': True,
        'fixture_density_kg_m3': 1050.0,
        'fixture_fixed_nodes': [0, 1, 2],
        'source_cell_native_accepted_steps': 1,
        'whole_wall_native_accepted_steps': 0,
        'heartbeat_qualified': False,
        'boundary': ('One exact source tetrahedron, source fibre frame and source-derived '
                     '100-ms candidate tension; synthetic inertial density, three fixed '
                     'nodes, no chamber loading, blood flow or whole-wall accepted state.'),
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        reference.require(output.read_text() == encoded,
                          'existing native source-cell receipt changed')
    else:
        pending = output.with_name(output.name + f'.{os.getpid()}.pending')
        pending.write_text(encoded)
        os.replace(pending, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--activation', type=Path, required=True)
    parser.add_argument('--tension', type=Path, required=True)
    parser.add_argument('--native-repo', type=Path, required=True)
    parser.add_argument('--native-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.asset, args.activation, args.tension, args.native_repo,
                 args.native_binary, args.output)
    print(json.dumps({'status': result['status'],
                      'native_result': result['native_result']}))


if __name__ == '__main__':
    main()
