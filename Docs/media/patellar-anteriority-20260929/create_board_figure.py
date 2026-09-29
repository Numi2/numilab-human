"""Arrange actual native knee captures beside the decoded-bone orientation audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / 'Build/patellar-anteriority-20260929/compiled-bone-audit.json'
IMAGES = {
    'Right knee': ROOT / 'Build/knee-parity-registration-20260929/native-neutral-right.v6/views/myosim-fullbody-articulated-bodyparts-bones-focus-body-142-side.png',
    'Left knee': ROOT / 'Build/knee-parity-registration-20260929/native-neutral-left.v6/views/myosim-fullbody-articulated-bodyparts-bones-focus-body-156-side.png',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads(AUDIT.read_text())
    rows = report['poses']
    assert len(rows) == 8 and report['projected_patellar_anteriority_evaluation_count'] == 16
    assert all(len(p['patellar_anteriority']) == 2 for p in rows)
    assert all(x['passed'] for p in rows for x in p['patellar_anteriority'])
    assert len(report['failures']) == 5 and all('range' in x for x in report['failures'])
    assert all(x['centroid_signed_anterior_offset_m'] < 0
               for x in report['source_qpos0_patellar_anteriority'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor='#0e1726')
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0],
                           width_ratios=[1, 1, 1.35], left=.04, right=.96,
                           top=.81, bottom=.105, hspace=.19, wspace=.16)
    for idx, (name, path) in enumerate(IMAGES.items()):
        ax = fig.add_subplot(grid[0, idx])
        ax.imshow(plt.imread(path))
        ax.set_axis_off()
        ax.set_title(name + ' · projected neutral', color='#f0f4fb',
                     fontsize=15, weight='bold', pad=10)
    ax = fig.add_subplot(grid[:, 2])
    names = [
        'Neutral', 'Hip flexion', 'Knee flexion', 'Ankle dorsiflexion',
        'Subtalar rotation', 'MTP flexion', 'Deep crouch*', 'Functional crouch*',
    ]
    right = [next(x for x in p['patellar_anteriority'] if x['side'] == 'r')
             ['minimum_signed_anterior_offset_m'] * 1000 for p in rows]
    left = [next(x for x in p['patellar_anteriority'] if x['side'] == 'l')
            ['minimum_signed_anterior_offset_m'] * 1000 for p in rows]
    y = np.arange(len(rows))
    ax.barh(y - .17, right, height=.31, color='#60d5c4', label='Right')
    ax.barh(y + .17, left, height=.31, color='#9a9bff', label='Left')
    ax.set_yticks(y, names, color='#ecf2fa', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 38)
    ax.set_xlabel('Minimum anterior offset of any patella vertex (mm)',
                  color='#d3dce9', fontsize=10)
    ax.set_title('16 / 16 full-support checks pass', color='#f0f4fb',
                 fontsize=16, weight='bold', pad=14)
    ax.tick_params(colors='#d3dce9')
    ax.spines[:].set_color('#516179')
    ax.set_facecolor('#172236')
    ax.grid(axis='x', color='#516179', alpha=.4)
    ax.set_axisbelow(True)
    ax.legend(facecolor='#172236', edgecolor='#516179', labelcolor='#edf4fb',
              loc='lower right', fontsize=10)

    note = fig.add_subplot(grid[1, :2]);note.axis('off')
    note.text(.015, .89, 'Why a source-rest knee can look reversed',
              color='#f0f4fb', fontsize=18, weight='bold', va='top')
    note.text(.015, .69,
              'The literal unprojected source pose puts both patellar centroids '
              'behind the knee anchor\n(right −5.76 mm; left −9.25 mm). '
              'The source joint equalities move them forward\nbefore the native '
              'neutral/posed views shown above.',
              color='#d3dce9', fontsize=12, va='top', linespacing=1.6)
    note.text(.015, .22,
              'Geometric orientation only; no cartilage, loaded contact or clinical anatomy.\n'
              '* Crouch diagnostics retain five source joint-range conflicts.',
              color='#ffcc86', fontsize=11, va='top', linespacing=1.6)
    fig.text(.04, .955, 'NUMI HUMAN  /  PATELLAR PLACEMENT', color='#60d5c4',
             fontsize=13, weight='bold')
    fig.text(.04, .905, 'Projected patellae stay in front of the knee anchor',
             color='#f4f7fb', fontsize=24, weight='bold')
    fig.text(.04, .045,
             '29 Sep 2026  ·  BodyParts3D / MyoSim source-bound visual and compiled NHBONES1 audit  ·  '
             'BodyParts3D CC-BY-SA 2.1 Japan', color='#93a2b8', fontsize=9)
    output = HERE / 'executive-patellar-orientation.png'
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)
    public = {
        'schema': 'numi.human.patellar-anteriority-board-evidence.v1',
        'audit_sha256': sha256(AUDIT),
        'compiled_bone_sha256': report['inputs']['bone_payload']['sha256'],
        'registration_sha256': report['inputs']['registration']['sha256'],
        'source_qpos0': report['source_qpos0_patellar_anteriority'],
        'projected_poses': [{
            'name': p['name'], 'patellar_anteriority': p['patellar_anteriority'],
            'range_failure_count': sum(p['name'] in x for x in report['failures']),
        } for p in rows],
        'remaining_source_range_failure_count': len(report['failures']),
        'native_images': {name: {'path': str(path.relative_to(ROOT)),
                                 'sha256': sha256(path)} for name, path in IMAGES.items()},
        'figure_sha256': sha256(output),
        'physical_contact_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    (HERE / 'summary.json').write_text(json.dumps(public, indent=2, sort_keys=True) + '\n')
    print(output, public['figure_sha256'])


if __name__ == '__main__':
    main()
