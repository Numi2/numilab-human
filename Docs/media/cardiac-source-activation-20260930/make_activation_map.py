"""Render the actual case18 ventricular source coordinates and candidate times."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT/'Build/cardiac-electrical-source-20260930/asset'
RUN = ROOT/'Build/cardiac-source-activation-20260930'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ASSET/'manifest.json').read_text())
    summary = json.loads((RUN/'summary.json').read_text())
    assert sha(ASSET/'manifest.json') == summary['asset_manifest_sha256']
    assert summary['status'] == 'full_source_reconstruction_with_source_output_mismatch'
    for name in ('nodes.f64le', 'tetrahedra.u32le', 'labels.u32le'):
        assert sha(ASSET/name) == manifest['buffers'][name]['sha256']
    for name in ('ventricular-source-nodes.u32le',
                 'ventricular-dof-regions.u32le', 'refined-arrival.f64le'):
        assert sha(RUN/name) == summary['output_sha256'][name]
    positions = np.fromfile(ASSET/'nodes.f64le', '<f8').reshape(-1, 3)
    tets = np.fromfile(ASSET/'tetrahedra.u32le', '<u4').reshape(-1, 4)
    labels = np.fromfile(ASSET/'labels.u32le', '<u4')
    sources = np.fromfile(RUN/'ventricular-source-nodes.u32le', '<u4')
    regions = np.fromfile(RUN/'ventricular-dof-regions.u32le', '<u4')
    times = np.fromfile(RUN/'refined-arrival.f64le', '<f8')*1000
    normal = np.where(regions == 0)[0]
    lut = np.full(len(positions), -1, np.int32)
    lut[sources[normal]] = normal
    right_lut = lut.copy()
    extra = np.where(regions == 2)[0]
    right_lut[sources[extra]] = extra
    left = np.unique(tets[labels == 1])
    right = np.unique(tets[labels == 2])
    point_ids = np.concatenate([left, right])
    point_times = np.concatenate([times[lut[left]], times[right_lut[right]]])
    xyz = positions[point_ids]*1000

    plt.rcParams.update({'font.family': 'Arial', 'text.color': '#ecf2ff',
                         'axes.labelcolor': '#ecf2ff', 'xtick.color': '#9fb0c7',
                         'ytick.color': '#9fb0c7'})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='#0b1424')
    fig.text(.045, .945, 'NUMI HUMAN  /  CARDIAC SOURCE ACTIVATION',
             color='#56d5c6', fontsize=24, weight='bold')
    fig.text(.045, .888, 'Ventricular arrival-time reconstruction',
             color='#ecf2ff', fontsize=33, weight='bold')
    fig.text(.045, .847,
             'Rodero case18 source nodes and S4 conduction parameters  ·  offline Apple CPU solve',
             color='#b2bfd2', fontsize=17)
    axes = [fig.add_axes((.055, .23, .39, .55)),
            fig.add_axes((.525, .23, .39, .55))]
    for axis, (x, y, depth, title) in zip(axes, ((0, 2, 1, 'SOURCE FRONT  /  x,z'),
                                                (1, 2, 0, 'SOURCE SIDE  /  y,z'))):
        order = np.argsort(xyz[:, depth], kind='stable')
        scatter = axis.scatter(xyz[order, x], xyz[order, y],
                               c=point_times[order], s=.24,
                               cmap='turbo', vmin=0, vmax=80,
                               linewidths=0, rasterized=True)
        axis.set_aspect('equal')
        axis.set_facecolor('#0b1424')
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, loc='left', fontsize=16, color='#ecf2ff', pad=16)
        for spine in axis.spines.values():
            spine.set_color('#38516b')
    colorbar = fig.colorbar(scatter, ax=axes, orientation='horizontal',
                            fraction=.035, pad=.07, aspect=55)
    colorbar.set_label('Candidate activation arrival (ms) — simulated, not measured',
                       fontsize=14, color='#ecf2ff')
    colorbar.outline.set_edgecolor('#38516b')
    colorbar.ax.tick_params(colors='#ecf2ff')
    fig.text(.052, .13,
             'Source CARP simulation: LV span 71.99 ms  |  10–90% interval 29.84 ms',
             fontsize=17, color='#ecf2ff')
    fig.text(.052, .093,
             'This reconstruction:   LV span 75.19 ms  |  10–90% interval 37.33 ms',
             fontsize=17, color='#ecf2ff')
    fig.text(.052, .045,
             'Mismatch remains. No source voltage, native electrical transaction, ECG or heartbeat qualification.',
             fontsize=16, color='#ef9aa5', weight='bold')
    image = HERE/'cardiac-source-activation-map.png'
    fig.savefig(image, facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    provenance = {
        'schema': 'numi.human.cardiac-source-activation-figure.v1',
        'image_sha256': sha(image),
        'generator_sha256': sha(Path(__file__)),
        'source_nodes_sha256': sha(ASSET/'nodes.f64le'),
        'source_tetrahedra_sha256': sha(ASSET/'tetrahedra.u32le'),
        'source_labels_sha256': sha(ASSET/'labels.u32le'),
        'candidate_arrival_sha256': sha(RUN/'refined-arrival.f64le'),
        'candidate_summary_sha256': sha(RUN/'summary.json'),
        'source_output_role': 'published simulated comparator, not patient measurement',
        'clinical_or_native_qualification': False,
    }
    (HERE/'figure-provenance.json').write_text(json.dumps(provenance,
                                                         indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
