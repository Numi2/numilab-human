"""Execute and scope the native Open Knee prescribed-closure contact preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from numilab_human.open_knee import parse_source


ROOT = Path(__file__).resolve().parents[1]
PAIRS_WITH_FMC_SLAVE = (
    'TBC-L_To_FMC', 'PTC_To_FMC', 'MNS-L_To_FMC',
    'MNS-M_To_FMC', 'TBC-M_To_FMC',
)
CODE_PATHS = (
    'src/core/NumiHumanKneeContact.cpp',
    'apps/numilab_human_knee_contact_probe.cpp',
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError('patellofemoral preflight scope: ' + message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *arguments: str) -> bytes:
    return subprocess.run(['git', '-C', str(repo), *arguments],
                          check=True, capture_output=True).stdout


def parse(output: str) -> dict:
    lines = output.splitlines()
    require(len(lines) == 8, 'expected one summary and seven pair rows')
    def fields(line: str) -> dict:
        pairs = [item.split('=', 1) for item in line.split()]
        require(all(len(pair) == 2 for pair in pairs) and
                len({pair[0] for pair in pairs}) == len(pairs),
                'malformed or duplicate native output field')
        return dict(pairs)
    summary = fields(lines[0])
    rows = [fields(line) for line in lines[1:]]
    require(summary.get('numi_human_knee_contact') == 'passed' and
            summary.get('articular_pairs') == '7' and
            summary.get('contact_samples') == '69701' and
            summary.get('sustained_steps') == '65' and
            float(summary.get('peak_closure_m', 'nan')) == 5e-5 and
            summary.get('replay') == 'bitwise' and
            summary.get('restore') == 'verified' and
            summary.get('evidence_level') == 'preflight' and
            summary.get('live_human_coupling') == 'false' and
            summary.get('nonpenetration_solve') == 'false',
            'native preflight scope or output changed')
    by_name = {row['articular_pair']: row for row in rows}
    require(len(by_name) == 7 and
            all(name in by_name for name in PAIRS_WITH_FMC_SLAVE),
            'seven articular pair rows')
    for name in PAIRS_WITH_FMC_SLAVE:
        row = by_name[name]
        require(row['samples'] == row['peak_active_samples'] == '11586' and
                row['slave_surface'] == f'FMC_@_{name.split("_To_")[0]}_ContactFaces',
                f'{name} broad femoral sample identity')
    patella = by_name['PTC_To_FMC']
    for name in ('TBC-L_To_FMC', 'TBC-M_To_FMC'):
        other = by_name[name]
        require(all(other[field] == patella[field] for field in (
                    'samples', 'area_m2', 'foundation_stiffness_pa_per_m',
                    'peak_active_samples', 'peak_pressure_pa',
                    'peak_normal_force_n', 'peak_energy_j')),
                f'{name} and patellar pair duplicated prescribed-closure response')
    return {'summary': summary, 'pairs': by_name,
            'patellar_pair': patella,
            'femoral_slave_pair_names': list(PAIRS_WITH_FMC_SLAVE)}


def audit(binary: Path, native_repo: Path, current_repo: Path) -> dict:
    binary, native_repo, current_repo = (Path(path).resolve() for path in
                                        (binary, native_repo, current_repo))
    require(binary.is_file(), 'native probe binary missing')
    native_revision = git(native_repo, 'rev-parse', 'HEAD').decode().strip()
    current_revision = git(current_repo, 'rev-parse', 'HEAD').decode().strip()
    require(native_revision == '24f8cdf4d876e1dccf0fb308affca84de972ce56' and
            current_revision == '5787938325d6d232074bc508940c86c6b8b7a29d' and
            not git(native_repo, 'status', '--porcelain=v1', '--untracked-files=no') and
            not git(current_repo, 'status', '--porcelain=v1', '--untracked-files=no'),
            'pinned clean native and current-source revisions')
    source_code_sha = {}
    for path in CODE_PATHS:
        native = git(native_repo, 'show', f'HEAD:{path}')
        current = git(current_repo, 'show', f'HEAD:{path}')
        require(native == current, f'current contact source differs: {path}')
        source_code_sha[path] = hashlib.sha256(native).hexdigest()

    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    pair_table = {name: (master, slave) for name, master, slave
                  in source.surface_pairs}
    femoral_faces = source.surfaces[
        pair_table['PTC_To_FMC'][1]].faces
    require(len(femoral_faces) == 22478 and all(
        source.surfaces[pair_table[name][1]].faces == femoral_faces
        for name in PAIRS_WITH_FMC_SLAVE),
        'source reuses the exact broad femoral slave surface')
    intersection_path = ROOT / 'Docs/media/patellofemoral-intersections-20260930/receipt.json'
    intersection = json.loads(intersection_path.read_text())
    require(intersection['status'] == 'failed_static_noninterpenetration' and
            intersection['source']['exact_segment_or_polygon_crossing_pairs'] == 18,
            'source initial intersections')

    executions = {}
    for side, stem in (
            ('left', 'open-knee-oks003-left'),
            ('right', 'open-knee-oks003-right-mirrored')):
        payload = ROOT / 'Build/patellofemoral-surface-20260930' / side / f'{stem}.nhknee'
        require(sha(payload) == intersection['compiled'][side]['payload_sha256'],
                f'{side} exact source payload')
        run = subprocess.run([str(binary), str(payload)], capture_output=True,
                             text=True, check=True)
        require(not run.stderr, f'{side} native preflight stderr')
        measured = parse(run.stdout)
        require(measured['summary']['side'] ==
                ('left' if side == 'left' else 'right_mirrored'),
                f'{side} native identity')
        executions[side] = {
            'payload_sha256': sha(payload),
            'stdout_sha256': hashlib.sha256(run.stdout.encode()).hexdigest(),
            'stdout': run.stdout,
            **measured,
        }
    return {
        'schema': 'numi.human.patellofemoral-preflight-scope.v1',
        'status': 'failed_localized_initial_contact_admission',
        'native_probe_revision': native_revision,
        'current_native_source_revision': current_revision,
        'native_contact_source_sha256': source_code_sha,
        'native_probe_binary_sha256': sha(binary),
        'intersection_receipt_sha256': sha(intersection_path),
        'broad_femoral_slave_face_count': len(femoral_faces),
        'source_initial_crossing_pair_count': 18,
        'executions': executions,
        'patellar_initial_contact_force_qualified': False,
        'localized_patellar_contact_qualified': False,
        'loaded_knee_qualified': False,
        'boundary': ('The CPU native preflight replays a 65-point prescribed '
                     'closure/restore curve. It zeros the source rest response '
                     'and activates every node of a broad femoral slave surface '
                     'shared by five pairs. Its positive pressure is not a '
                     'response to the 18 source-authored patellofemoral '
                     'intersections or a localized loaded-knee contact solve.'),
        'auditor_sha256': sha(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'native_repo', 'current_repo', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path,
                            required=True)
    args = parser.parse_args()
    result = audit(args.binary, args.native_repo, args.current_repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': result['status'],
                      'left_patellar_force_n': float(result['executions']['left'][
                          'patellar_pair']['peak_normal_force_n']),
                      'right_patellar_force_n': float(result['executions']['right'][
                          'patellar_pair']['peak_normal_force_n'])}))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
