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


def parse_pts(path: Path) -> list[tuple[int, Decimal, int]]:
    result = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        parts = line.split('\t')
        if len(parts) != 3:
            raise ValueError('malformed PTS TSV line %d in %s' % (line_no, path))
        try:
            index, pts, samples = int(parts[0]), Decimal(parts[1]), int(parts[2])
        except Exception as exc:
            raise ValueError('invalid PTS TSV values in %s' % path) from exc
        if not pts.is_finite() or index != len(result) or samples < 0:
            raise ValueError('invalid index/time/sample count in %s' % path)
        result.append((index, pts, samples))
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
    if (declaration.get('schema') != 'numi.human.resting.native-run.declaration.v1' or
            declaration.get('seconds') != 310 or declaration.get('accepted_steps') != 155000 or
            declaration.get('capture_steps') != EXPECTED_DECL_STEPS):
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
    inspect_log = run_logged(['/usr/bin/swift', str(INSPECT), str(output), '2.5', '7'], output / 'inspect-movie.log')
    compressed = parse_pts(output / 'compressed-pts.tsv')
    decoded = parse_pts(output / 'bgra-pts.tsv')
    image_samples = [pts for _, pts, count in compressed if count == 1]
    zero_markers = sum(1 for _, _, count in compressed if count == 0)
    if any(count not in (0, 1) for _, _, count in compressed) or any(count != 1 for _, _, count in decoded):
        raise ValueError('movie stream contains batched or malformed sample buffers')
    if len(image_samples) != sum(1 for _ in csv.DictReader(surface.open(newline=''))):
        raise ValueError('compressed nonempty movie frames do not match surface CSV data-row count')
    compressed_sorted = sorted(image_samples)
    decoded_pts = [pts for _, pts, _ in decoded]
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
    selected = [dict(token.split('=', 1) for token in line.split())
                for line in inspect_log.splitlines() if line.startswith('frame=')]
    expected_names = {'initial', 'middle', 'final', 'skin', 'muscles', 'skeleton', 'organs', 'lungs', 'heart', 'vessels'}
    if {item.get('frame') for item in selected} != expected_names:
        raise ValueError('representative frame set is incomplete')
    pngs = sorted(output.glob('reader-sample-*.png')) + sorted(output.glob('frame-*.png'))
    pixel_log = run_logged(['/usr/bin/swift', str(PIXEL), *map(str, pngs)], output / 'png-pixel-metrics.log')
    pixel_rows = [line for line in pixel_log.splitlines() if line.startswith('file=')]
    if len(pixel_rows) != len(pngs):
        raise ValueError('decoded PNG pixel metrics did not cover all tail/representative frames')
    source_after = {str(path): sha(path) for path in source_paths}
    if source_before != source_after:
        raise ValueError('a native run input/output source changed during movie inspection')
    report = {
        'schema': 'numi.human.closed-baseline-movie-review.v1',
        'status': 'pass',
        'scope': 'Post-close AVAssetReader PTS/count, AVAssetImageGenerator representative frames, sequential BGRA decode, and nonempty decoded PNG pixel check; no anatomy or physiology qualification.',
        'run_dir': str(run_dir),
        'source_hashes_before': source_before,
        'source_hashes_after': source_after,
        'helpers': helper_pins,
        'counts': {
            'compressed_track_sample_buffers': len(compressed),
            'compressed_zero_sample_timing_markers': zero_markers,
            'compressed_nonempty_image_samples': len(image_samples),
            'surface_csv_data_rows': len(surface_rows),
            'sequential_bgra_decoded_samples': len(decoded),
            'decoded_only_leading_zero_preroll': preroll,
            'zero_pts_blackout_interpretation': 'prohibited: when present, PTS 0.0 is an extra decoded preroll sample with no compressed nonempty image/surface-row counterpart; it is excluded from the presentation-frame count.',
        },
        'pts': {
            'compressed_nonempty_first_s': str(compressed_sorted[0]),
            'compressed_nonempty_last_s': str(compressed_sorted[-1]),
            'decoded_first_s': str(decoded_pts[0]),
            'decoded_last_s': str(decoded_pts[-1]),
            'decoded_pts_strictly_increasing': True,
            'compressed_nonempty_pts_match_decoded_after_optional_preroll': True,
        },
        'avassetimagegenerator': {'summary': inspect_summary, 'selected_frames': selected},
        'sequential_pixel_check': {'frames_checked': len(pixel_rows), 'per_frame_metrics_log': str(output / 'png-pixel-metrics.log')},
        'notes': ['Movie wall-clock PTS are compared in presentation order and are not equated with simulation seconds.',
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
