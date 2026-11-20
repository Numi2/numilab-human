"""Run the existing exact native skin audit on a retained accepted geometry pack."""
import argparse
import json
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('pack', type=Path)
parser.add_argument('report', type=Path)
args = parser.parse_args()
work = Path(__file__).resolve().parent.parent
audit = work / 'resting-surface-clearance-001'
sys.path[:0] = [str(audit / 'src'), str(work / 'resting-clearance-and-atrium-005')]
from fast_skin_intersection_audit import install_for_float32

install_for_float32()
runpy.run_path(str(audit / 'audit.py'))['run'](
    args.pack.resolve(strict=True), args.report.resolve(),
)
result = json.loads(args.report.read_text())
print(json.dumps({
    'accepted_step': result['accepted_step'], 'accepted_time_s': result['accepted_time_s'],
    'skin_self_pairs': result['self_intersections']['count'],
    'crossing_pairs_by_semantic': result['crossing_pairs_by_semantic'],
    'nonocular_skin_target_pairs': sum(
        item['count'] for item in result['surfaces'] if not item['ocular_source_id']
    ),
    'limit': result['limit'],
}))
