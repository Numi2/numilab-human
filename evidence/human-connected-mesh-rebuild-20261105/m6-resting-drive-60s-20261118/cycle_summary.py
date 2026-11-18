"""Exploratory breath-phase check performed after preregistered comparison."""
import csv
import json
from pathlib import Path
from statistics import mean

here = Path(__file__).resolve().parent

def summarize(arm):
    with (here/arm/'native'/'resting-coupled.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    unique = {}
    for r in rows:
        event = int(r['last_inspiration_step'])
        if event > 0:
            unique.setdefault(event,[float(r['last_inspiration_time_s']),
                                      float(r['last_complete_breath_inspired_ml'])])
    late = list(t for t in unique.values() if t[0]>=40)
    intervals=[following[0]-previous[0] for previous,following in zip(late,late[1:])]
    return {'complete_inspiration_events':list(unique.values()),
            'late_mean_complete_breath_ml':mean(v for _,v in late),
            'late_mean_inspiration_interval_s':mean(intervals),
            'late_minute_inspired_l':.06*mean(v for _,v in late)/mean(intervals),
            'minimum_airflow_ml_s':min(float(r['airflow_ml_s']) for r in rows),
            'maximum_airflow_ml_s':max(float(r['airflow_ml_s']) for r in rows)}

result={'scope':'Exploratory event-aligned analysis. Original recovery threshold remains failed.',
        'reference':summarize('reference'),'drive-0p6':summarize('drive-0p6')}
(here/'cycle-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
