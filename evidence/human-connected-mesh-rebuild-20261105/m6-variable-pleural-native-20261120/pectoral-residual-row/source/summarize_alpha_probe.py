"""Identify the first restrictive line-search stage at the rejecting microstep."""
import json
from pathlib import Path

run = Path('active-limiter-trial')
records = {}
for line in (run/'stderr.log').open():
    if not line.startswith('fem_newton_iterate='):
        continue
    row = json.loads(line.partition('=')[2])
    if row['root'] == 4 and row['microtick'] == 61:
        records.setdefault(row['iteration'], {})[row['stage']] = row
result = []
stages = ('before_contact_limits','after_deformable_limit','after_rigid_limit',
          'after_support_limit','after_vascular_limit','final_line_search')
for iteration, rows in sorted(records.items()):
    entry = {'iteration': iteration}
    prior = 1.
    for stage in stages:
        row = rows[stage]
        # Object alphas are synchronized at the contact boundaries. The
        # separate environment scalar is not written until a later stage.
        alpha = min(item[0] for item in row['object_line_search'])
        record = dict(alpha=alpha, environment_alpha=row['alpha'][0])
        if stage == 'before_contact_limits':
            record['due_fem_minimum_determinants'] = {
                index: item[1] for index, item in enumerate(row['object_line_search'])
                if item[1] is not None and item[2] > 0
            }
        if alpha < prior:
            record['decreases_object_alpha'] = True
        prior = alpha
        entry[stage] = record
    result.append(entry)
assert len(result) == 24
(run/'alpha-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for entry in result:
    if entry['iteration'] in (0, 1, 2, 3, 4, 8, 14, 23):
        print(entry['iteration'], [(stage, entry[stage]['alpha'],
               entry[stage]['environment_alpha']) for stage in stages])
