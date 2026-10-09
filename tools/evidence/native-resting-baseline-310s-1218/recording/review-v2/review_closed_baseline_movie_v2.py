#!/usr/bin/env python3
"""Post-close movie/CSV frame correspondence and sequential decode review."""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

COUNT = Path('/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/recording-review-terminal-sequential-1217/count_pts.swift')
COUNT_SHA = '7436e6e4af15e2f34ce67bbd2c110991d3e3209f44e2ea4fc84571c5e20eced5'
DECODE = Path('/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/recording-review-terminal-sequential-1217/decode_sequential.swift')
DECODE_SHA = '446eefd138fd8fb472be21a92ff1ff150ea11377ce6df52ff66d92f220da9d18'
INSPECT = Path('/Users/n/numi-human-resting-final-source-028/matter/tools/inspect_resting_movie.swift')
INSPECT_SHA = '2d6704dd5f06bffd0b8aa0072171af805fdb3dd00dcc11c02e0b692513b21649'
PIXEL = Path(__file__).with_name('png_pixel_metrics.swift')
EXPECTED_PIXEL_SHA = '0ecfe9ae949b3e83eca2529fee36927ec63c275ec0ac8135d6870a03a38a9b44'
EXPECTED_DECL_STEPS = [0, 9983, 19999, 47519, 152191, 153183, 154143, 155000]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked required JSON: ' + str(path))
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError('expected JSON object: ' + str(path))
    return obj


def verify_helpers() -> dict:
    expected = {COUNT: COUNT_SHA, DECODE: DECODE_SHA, INSPECT: INSPECT_SHA, PIXEL: EXPECTED_PIXEL_SHA}
    for path, digest in expected.items():
        if not path.is_file() or sha(path) != digest:
            raise ValueError('movie review helper pin mismatch: ' + str(path))
    return {str(path): digest for path, digest in expected.items()}


def parse_pts(path: Path, *, zero_sample_markers: bool = False) -> list[tuple[int, Decimal | None, int, str]]:
    """Parse sample records; nonfinite timestamps are allowed only on empty markers."""
    result = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        parts = line.split('\t')
        if len(parts) != 3:
            raise ValueError('malformed PTS TSV line %d in %s' % (line_no, path))
        try:
            index, samples = int(parts[0]), int(parts[2])
        except Exception as exc:
            raise ValueError('invalid PTS TSV index/sample count in %s' % path) from exc
        raw_pts = parts[1]
        if index != len(result) or samples not in (0, 1):
            raise ValueError('invalid index/sample count in %s' % path)
        try:
            pts = Decimal(raw_pts)
        except Exception as exc:
            if not (zero_sample_markers and samples == 0):
                raise ValueError('invalid image PTS in %s' % path) from exc
            pts = None
        if pts is not None and not pts.is_finite():
            if not (zero_sample_markers and samples == 0):
                raise ValueError('nonfinite PTS is allowed only for zero-sample timing markers: ' + str(path))
            pts = None
        if samples == 1 and pts is None:
            raise ValueError('image sample has no finite PTS in %s' % path)
        if samples == 0 and not zero_sample_markers:
            raise ValueError('unexpected zero-sample timing marker in %s' % path)
        result.append((index, pts, samples, raw_pts))
    if not result:
        raise ValueError('empty PTS inventory: ' + str(path))
    return result


def run_logged(argv: list[str], logfile: Path) -> str:
    result = subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logfile.write_text(result.stdout, encoding='utf-8')
    if result.returncode != 0:
        raise ValueError('movie helper failed (%d): %s; see %s' % (result.returncode, argv[0], logfile))
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prepare-check', action='store_true', help='verify helper pins only; reads no run/movie data')
    args = parser.parse_args()
    helper_pins = verify_helpers()
    if args.prepare_check:
        print(json.dumps({'status': 'helpers_pinned', 'helpers': helper_pins}, sort_keys=True))
        return
    if args.run_dir is None or args.output_dir is None:
        parser.error('--run-dir and --output-dir are required unless --prepare-check is used')
    run_dir = args.run_dir.resolve(strict=True)
    prep_dir = run_dir.parent
    declaration_path = prep_dir / 'run-declaration.json'
    execution_path = prep_dir / 'execution.json'
    invocation_path = run_dir / 'invocation.json'
    metadata_path = run_dir / 'run-metadata.json'
    declaration = read_json(declaration_path)
    execution = read_json(execution_path)
    invocation = read_json(invocation_path)
    metadata = read_json(metadata_path)
    env = invocation.get('environment', {})
    period_raw = env.get('NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS')
    try:
        period = Decimal(str(period_raw))
        declared_period = Decimal(str(declaration.get('inspection_period_seconds')))
    except Exception as exc:
        raise ValueError('inspection tour period is missing or malformed in invocation/declaration') from exc
    if (not period.is_finite() or period <= 0 or not declared_period.is_finite() or
            period != declared_period):
        raise ValueError('invocation inspection period does not match the native run declaration')
    if (declaration.get('schema') != 'numi.human.resting.native-run.declaration.v1' or
            declaration.get('seconds') != 310 or declaration.get('accepted_steps') != 155000 or
            declaration.get('capture_steps') != EXPECTED_DECL_STEPS or
            declaration.get('inspection_period_seconds') is None):
        raise ValueError('run declaration is not the pinned 310 s baseline capture schedule')
    if (execution.get('returncode') != 0 or execution.get('changed_inputs') != {} or
            execution.get('declaration_sha256') != sha(declaration_path)):
        raise ValueError('owner execution receipt does not show a closed unchanged-input run')
    if (metadata.get('exit_code') != 0 or metadata.get('source_files_changed_during_run') != [] or
            metadata.get('argv') != invocation.get('argv') or
            metadata.get('asset_sha256') != invocation.get('asset_sha256') or
            metadata.get('loaded_metal_runtime', {}).get('verified') is not True):
        raise ValueError('native run metadata does not show a closed successful run')
    movie = run_dir / 'native-viewer.mov'
    surface = run_dir / 'resting-surface-audit.csv'
    for path in (movie, surface):
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing or linked native output: ' + str(path))
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise ValueError('refusing to overwrite movie review output: ' + str(output))
    source_paths = [declaration_path, execution_path, invocation_path, metadata_path, movie, surface]
    source_before = {str(path): sha(path) for path in source_paths}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'native-viewer.mov').symlink_to(movie)
    (output / 'resting-surface-audit.csv').symlink_to(surface)
    count_log = run_logged(['/usr/bin/swift', str(COUNT), str(movie), str(output)], output / 'count-pts.log')
    decode_log = run_logged(['/usr/bin/swift', str(DECODE), str(movie), str(output)], output / 'decode-sequential.log')
    inspect_log = run_logged(['/usr/bin/swift', str(INSPECT), str(output), str(period), '7'], output / 'inspect-movie.log')
    compressed = parse_pts(output / 'compressed-pts.tsv', zero_sample_markers=True)
    decoded = parse_pts(output / 'bgra-pts.tsv')
    image_samples = [pts for _, pts, count, _ in compressed if count == 1]
    marker_rows = [(index, raw_pts) for index, pts, count, raw_pts in compressed if count == 0]
    zero_markers = len(marker_rows)
    if any(pts is None for pts in image_samples) or any(count != 1 for _, _, count, _ in decoded):
        raise ValueError('movie image streams contain missing PTS or non-single-sample records')
    if len(image_samples) != sum(1 for _ in csv.DictReader(surface.open(newline=''))):
        raise ValueError('compressed nonempty movie frames do not match surface CSV data-row count')
    compressed_sorted = sorted(image_samples)
    decoded_pts = [pts for _, pts, _, _ in decoded]
    if any(pts is None for pts in decoded_pts):
        raise ValueError('decoded image frame contains a nonfinite PTS')
    if any(b <= a for a, b in zip(decoded_pts, decoded_pts[1:])):
        raise ValueError('BGRA sequential decoder PTS are not strictly increasing')
    preroll = False
    if decoded_pts == compressed_sorted:
        pass
    elif (len(decoded_pts) == len(compressed_sorted) + 1 and decoded_pts[0] == Decimal('0') and
          decoded_pts[1:] == compressed_sorted and Decimal('0') not in compressed_sorted):
        preroll = True
    else:
        raise ValueError('BGRA PTS differ from compressed nonempty PTS beyond a single decoded-only leading zero preroll')
    decode_match = re.search(r'samples=(\d+) nonmonotonic_or_duplicate_pts=(\d+) last_pts=([^ ]+) status=(\w+)', decode_log)
    if (decode_match is None or int(decode_match.group(1)) != len(decoded) or
            int(decode_match.group(2)) != 0 or decode_match.group(4) != 'completed'):
        raise ValueError('sequential AVAssetReader decode did not report complete unique frames')
    surface_rows = list(csv.DictReader(surface.open(newline='')))
    inspect_summary_line = next((line for line in inspect_log.splitlines() if line.startswith('frames=')), None)
    if inspect_summary_line is None:
        raise ValueError('AVAssetImageGenerator inspector omitted frame summary')
    inspect_summary = dict(token.split('=', 1) for token in inspect_summary_line.split())
    if (int(inspect_summary.get('frames', '-1')) != len(surface_rows) or
            int(inspect_summary.get('timing_markers', '-1')) != zero_markers):
        raise ValueError('presentation-ordered image samples do not match CSV frames/AV timing markers')
    try:
        first_wall_s = float(inspect_summary['first_wall_s'])
        max_gap_wall_s = float(inspect_summary['max_gap_wall_s'])
    except (KeyError, ValueError) as exc:
        raise ValueError('AV inspector summary lacks finite initial/max wall-gap values') from exc
    if not math.isfinite(first_wall_s) or not math.isfinite(max_gap_wall_s) or first_wall_s < 0 or max_gap_wall_s <= 0:
        raise ValueError('AV inspector wall timing summary is invalid')
    if abs(float(compressed_sorted[0]) - first_wall_s) > 1e-9:
        raise ValueError('inspector first image time differs from compressed PTS inventory')
    selected = [dict(token.split('=', 1) for token in line.split())
                for line in inspect_log.splitlines() if line.startswith('frame=')]
    expected_names = {'initial', 'middle', 'final', 'skin', 'muscles', 'skeleton', 'organs', 'lungs', 'heart', 'vessels'}
    if {item.get('frame') for item in selected} != expected_names:
        raise ValueError('representative frame set is incomplete')
    targets = []
    layer_names = ['skin', 'muscles', 'skeleton', 'organs', 'lungs', 'heart', 'vessels']
    for item in selected:
        name = item.get('frame')
        if name in layer_names:
            layer = layer_names.index(name)
            target_s = (Decimal(layer) + Decimal('0.5')) * period
            selected_s = Decimal(item['simulated_s'])
            targets.append({'frame': name, 'target_simulated_s': str(target_s),
                            'selected_simulated_s': str(selected_s),
                            'absolute_selection_error_s': str(abs(selected_s-target_s))})
    pngs = sorted(output.glob('reader-sample-*.png')) + sorted(output.glob('frame-*.png'))
    pixel_log = run_logged(['/usr/bin/swift', str(PIXEL), *map(str, pngs)], output / 'png-pixel-metrics.log')
    pixel_rows = [line for line in pixel_log.splitlines() if line.startswith('file=')]
    if len(pixel_rows) != len(pngs):
        raise ValueError('decoded PNG pixel metrics did not cover all tail/representative frames')
    source_after = {str(path): sha(path) for path in source_paths}
    if source_before != source_after:
        raise ValueError('a native run input/output source changed during movie inspection')
    report = {
        'schema': 'numi.human.closed-baseline-movie-review.v2',
        'status': 'pass',
        'scope': 'Post-close AVAssetReader PTS/count, AVAssetImageGenerator representative frames, sequential BGRA decode, and nonempty decoded PNG pixel check; no anatomy or physiology qualification.',
        'run_dir': str(run_dir),
        'source_hashes_before': source_before,
        'source_hashes_after': source_after,
        'helpers': helper_pins,
        'counts': {
            'compressed_track_sample_buffers': len(compressed),
            'compressed_zero_sample_timing_markers': zero_markers,
            'zero_sample_marker_records': [{'compressed_index': index, 'raw_pts': raw_pts,
                                            'is_image_frame': False} for index, raw_pts in marker_rows],
            'compressed_nonempty_image_samples': len(image_samples),
            'surface_csv_data_rows': len(surface_rows),
            'sequential_bgra_decoded_samples': len(decoded),
            'decoded_only_leading_zero_preroll': preroll,
            'zero_pts_blackout_interpretation': 'prohibited: when present, PTS 0.0 is an extra decoded preroll sample with no compressed nonempty image/surface-row counterpart; it is excluded from the presentation-frame count.',
        },
        'inspection_tour': {
            'period_seconds': str(period),
            'period_source': 'invocation.environment.NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS',
            'declaration_period_seconds': declaration['inspection_period_seconds'],
            'layers': targets,
        },
        'pts': {
            'compressed_nonempty_first_s': str(compressed_sorted[0]),
            'initial_gap_from_zero_to_first_image_s': first_wall_s,
            'longest_between_image_gap_s': max_gap_wall_s,
            'compressed_nonempty_last_s': str(compressed_sorted[-1]),
            'decoded_first_s': str(decoded_pts[0]),
            'decoded_last_s': str(decoded_pts[-1]),
            'decoded_pts_strictly_increasing': True,
            'compressed_nonempty_pts_match_decoded_after_optional_preroll': True,
        },
        'avassetimagegenerator': {'summary': inspect_summary, 'selected_frames': selected},
        'sequential_pixel_check': {'frames_checked': len(pixel_rows), 'per_frame_metrics_log': str(output / 'png-pixel-metrics.log')},
        'notes': ['Movie wall-clock PTS are compared in presentation order and are not equated with simulation seconds.',
                  'Zero-sample timing markers are preserved by compressed-stream index and raw PTS text; nonfinite marker PTS are never treated as image frames.',
                  'The initial gap from wall time zero to the first image and the longest inter-image gap are reported explicitly; the initial 7.265 s gap is not discarded.',
                  'Surface CSV rows are associated to sorted image samples by sequence index.',
                  'All native inputs were hash-identical before and after review.'],
    }
    (output / 'movie-review.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(output / 'movie-review.json'), 'status': 'pass', 'frame_count': len(surface_rows)}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        raise SystemExit('movie review refused: ' + str(exc))
