"""Produce source-ordered Float32 cell tension for Matter's FEM stress input."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from . import cardiac_active_force_gate as reference


ROOT = reference.ROOT
REFERENCE = ROOT / 'Docs/media/cardiac-active-force-reference-20260930'


def produce(asset: Path, activation: Path, output: Path,
            times_ms: list[float] | None = None) -> dict:
    asset, activation, output = (Path(p).resolve() for p in
                                 (asset, activation, output))
    config, _, _, earlier, arrays = reference.source_inputs(
        asset, activation, REFERENCE)
    if times_ms is None:
        requested = [(frame['time_ms'], frame) for frame in earlier['frames']]
    else:
        requested_times = [float(time_ms) for time_ms in times_ms]
        reference.require(bool(requested_times) and
                          all(math.isfinite(time_ms) and time_ms >= 0.0
                              for time_ms in requested_times) and
                          all(right > left for left, right in
                              zip(requested_times, requested_times[1:])),
                          'requested frame times must be finite, nonnegative, and strictly increasing')
        reference_frames = {float(frame['time_ms']): frame
                            for frame in earlier['frames']}
        requested = [(time_ms, reference_frames.get(time_ms))
                     for time_ms in requested_times]
    source_nodes = arrays['ventricular-source-nodes.u32le']
    regions = arrays['ventricular-dof-regions.u32le']
    arrivals = arrays['refined-arrival.f64le'] * 1000.0
    nodes = arrays['nodes.f64le']
    tetrahedra = arrays['tetrahedra.u32le']
    labels = arrays['labels.u32le']
    lut = np.full(len(nodes), -1, dtype=np.int32)
    lut[source_nodes[:218077]] = np.arange(218077, dtype=np.int32)
    rv_lut = lut.copy()
    rv_lut[source_nodes[218077:]] = np.arange(218077, 218080, dtype=np.int32)
    reference.require(np.all(regions[:218077] == 0) and
                      regions[218077:].tolist() == [2, 2, 2],
                      'source electrical region ordering')
    selected = np.flatnonzero((labels == 1) | (labels == 2))
    reference.require(len(selected) == 1097534, 'source ventricular cell count')
    qa = (1.0 + 3.0 / math.sqrt(5.0)) / 4.0
    qb = (1.0 - qa) / 3.0
    p = config['source_parameters']
    peak = p['peak_isometric_tension_pa']
    delay = p['electromechanical_delay_ms']
    tc = p['contraction_time_constant_ms']
    tr = p['relaxation_time_constant_ms']
    duration = p['transient_duration_ms']
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for time_ms, earlier_frame in requested:
        tensions = np.zeros(len(labels), dtype='<f4')
        active_cells = 0
        for start in range(0, len(selected), 25000):
            index = selected[start:start + 25000]
            ids = tetrahedra[index]
            dofs = lut[ids].copy()
            rv = labels[index] == 2
            dofs[rv] = rv_lut[ids[rv]]
            reference.require(np.all(dofs >= 0), 'ventricular activation coverage')
            vertex_time = arrivals[dofs]
            quad_time = (qb * vertex_time.sum(axis=1)[:, None] +
                         (qa - qb) * vertex_time)
            elapsed = time_ms - quad_time - delay
            inside = (elapsed > 0) & (elapsed < duration)
            stress = np.zeros_like(elapsed)
            rise = np.tanh(elapsed[inside] / tc)
            fall = np.tanh((duration - elapsed[inside]) / tr)
            stress[inside] = peak * rise**2 * fall**2
            cell_tension = stress.mean(axis=1)
            active_cells += int(np.count_nonzero(cell_tension > 0))
            tensions[index] = cell_tension.astype('<f4')
        reference.require((earlier_frame is None or
                           active_cells == earlier_frame['native_metrics']['active_cells'])
                          and np.isfinite(tensions).all()
                          and np.all((tensions >= 0) & (tensions <= peak))
                          and np.all(tensions[(labels != 1) & (labels != 2)] == 0),
                          f'finite ventricular source tension at {time_ms} ms')
        time_token = (str(int(time_ms)) if float(time_ms).is_integer()
                      else format(time_ms, '.12g').replace('.', 'p'))
        path = output / f'active-tension-{time_token}ms.f32le'
        data = tensions.tobytes()
        if path.exists():
            reference.require(path.read_bytes() == data,
                              f'existing source tension changed at {time_ms} ms')
        else:
            pending = path.with_name(path.name + f'.{os.getpid()}.pending')
            pending.write_bytes(data)
            os.replace(pending, path)
        frame = {'time_ms': time_ms, 'file': path.name,
                 'bytes': len(data), 'sha256': reference.sha(path),
                 'active_ventricular_cells': active_cells,
                 'maximum_f32_tension_pa': float(tensions.max())}
        if earlier_frame is not None:
            frame['reference_internal_residual_sha256'] = earlier_frame['sha256']
        frames.append(frame)
    result = {
        'schema': 'numi.human.cardiac-active-tension-ingress.v1',
        'status': 'source_ordered_prescribed_active_tension_candidate',
        'source_asset_manifest_sha256': reference.sha(asset / 'manifest.json'),
        'candidate_activation_summary_sha256': reference.sha(activation / 'summary.json'),
        'fixed_reference_summary_sha256': reference.sha(REFERENCE / 'summary.json'),
        'producer_source_sha256': reference.sha(Path(__file__)),
        'source_cell_count': len(labels),
        'ordering': 'one little-endian Float32 Cauchy tension per source tetrahedron; labels 1/2 active, all other labels zero',
        'requested_times_ms': [frame['time_ms'] for frame in frames],
        'time_series_boundary': 'source-parameter Tanh reconstruction over the fixed activation-arrival field; no periodic excitation is inferred',
        'frames': frames,
        'accepted_native_anatomical_steps': 0,
        'source_model_reproduced': False,
        'heartbeat_qualified': False,
    }
    path = output / 'summary.json'
    encoded = json.dumps(result, sort_keys=True, indent=2) + '\n'
    if path.exists():
        reference.require(path.read_text() == encoded, 'existing ingress summary changed')
    else:
        pending = path.with_name(path.name + f'.{os.getpid()}.pending')
        pending.write_text(encoded)
        os.replace(pending, path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--activation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--times-ms', type=float, nargs='+',
                        help='source times to sample; defaults to the two fixed reference frames')
    args = parser.parse_args()
    result = produce(args.asset, args.activation, args.output, args.times_ms)
    print(json.dumps({'status': result['status'], 'frames': result['frames']}))


if __name__ == '__main__':
    main()
