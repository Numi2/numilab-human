from pathlib import Path
import json,csv,hashlib,math,sys
E=Path('/Users/n/numi-human-resting-evidence-20261005')
run=E/'native-composed-resting-cycle-909';out=E/'native-composed-cycle-review-910'
with (run/'resting-coupled.csv').open() as f:rows=list(csv.DictReader(f))
with (run/'resting-surface-audit.csv').open() as f:surfaces=list(csv.DictReader(f))
meta=json.loads((run/'run-metadata.json').read_text())
assert meta['exit_code']==0 and meta['source_files_changed_during_run']==[] and meta['loaded_metal_runtime']['verified']
assert len(rows)==10000 and [int(r['step']) for r in rows]==list(range(1,10001))
values={k:[float(r[k]) for r in rows] for k in rows[0]}
assert all(math.isfinite(v) for a in values.values() for v in a)
assert all(float(b['time_s'])>float(a['time_s']) for a,b in zip(rows,rows[1:]))
with (E/'native-area-guard-cycle-904/resting-coupled.csv').open() as f:prior=list(csv.DictReader(f))
phys=list(rows[0])[:list(rows[0]).index('step')]
parity={k:all(a[k]==b[k] for a,b in zip(rows,prior)) for k in phys}
ranges={k:{'minimum':min(v),'maximum':max(v),'mean':sum(v)/len(v),'final':v[-1]} for k,v in values.items()}
sranges={k:{'minimum':min(float(v[k]) for v in surfaces),'maximum':max(float(v[k]) for v in surfaces)} for k in surfaces[0] if k != 'geometry_mode' and all(v[k] for v in surfaces)}
sranges['geometry_mode']={'distinct_values':sorted({v['geometry_mode'] for v in surfaces})}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'run':str(run),'scope':'20-second integrated pilot, initialization included; numerical trace and native recording checks; full anatomy audit and 300-second acceptance remain separate.','accepted_steps':len(rows),'complete_breaths':int(rows[-1]['breaths']),'complete_filling_ejection_cycles':int(rows[-1]['complete_filling_ejection_cycles']),'native_elapsed_physical_s':float(rows[-1]['time_s']),'owner_wall_s':meta['wall_seconds'],'inclusive_real_time_factor':20/meta['wall_seconds'],'physiology_fields_exactly_equal_to_904':parity,'physiology_field_count':len(phys),'finite_fields':True,'monotone_physical_time':True,'both_inspiration_and_expiration':min(values['airflow_ml_s'])<0<max(values['airflow_ml_s']),'positive_aortic_and_pulmonary_accumulated_ejection':values['aortic_ejected_ml'][-1]>0 and values['pulmonary_ejected_ml'][-1]>0,'ranges':ranges,'surface_ranges':sranges,'displayed_accepted_frames':len(surfaces),'pins':{str(p):sha(p) for p in [run/'run-metadata.json',run/'invocation.json',run/'resting-coupled.csv',run/'resting-surface-audit.csv',run/'native-viewer.mov']}}
(out/'trace-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['accepted_steps','complete_breaths','complete_filling_ejection_cycles','owner_wall_s','inclusive_real_time_factor','both_inspiration_and_expiration','positive_aortic_and_pulmonary_accumulated_ejection','displayed_accepted_frames']}))
print('physiology_parity',sum(parity.values()),'/',len(parity))
