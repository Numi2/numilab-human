"""Measure accepted native trajectory only; never advances or changes physics."""
from pathlib import Path
import csv,hashlib,json,math
p=Path(__file__).resolve().parent
manifest=json.loads((p/'trace-manifest.json').read_text())
records={}
files=[]
for item in manifest['prefix']:
 path=p/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 files.append(path)
if (p/manifest['forward']).exists():files.append(p/manifest['forward'])
for path in files:
 for row in csv.DictReader(path.open()):
  if any(v is None or v=='' for v in row.values()):continue # writer may be publishing final row
  values={k:float(v) for k,v in row.items()}
  assert all(math.isfinite(v) for v in values.values()),(path,'nonfinite')
  step=int(values['accepted_step']);assert step==values['accepted_step']
  if step in records:assert records[step]==values,(path,step,'checkpoint boundary differs')
  records[step]=values
rows=[records[k] for k in sorted(records)]
assert rows and rows[0]['accepted_step']==0
assert all(b['accepted_step']==a['accepted_step']+1 and b['time_ms']>a['time_ms'] for a,b in zip(rows,rows[1:]))
assert all(abs(r['total_blood_ml']-5150)<=.01 and -.0001<=r['min_electric_potential']<=r['max_electric_potential']<=1.1 and 0<=r['min_recovery_gate']<=r['max_recovery_gate']<=1 for r in rows)
period=manifest['period_ms'];last=rows[-1]['time_ms']

def interval(start,end):
 # The native implicit transport advances volume with the accepted endpoint flow.
 # Fractional overlap apportions that step's conserved flux at a cycle boundary.
 flows={k:0. for k in ['aortic_ml_s','pulmonary_ml_s','mitral_ml_s','tricuspid_ml_s']}
 work={k:0. for k in ['lv','rv']}
 for a,b in zip(rows,rows[1:]):
  dt=b['time_ms']-a['time_ms'];overlap=max(0.,min(end,b['time_ms'])-max(start,a['time_ms']))
  if not overlap:continue
  for k in flows:flows[k]+=b[k]*overlap/1000
  for k in work:work[k]-=.5*(a[k+'_pressure_pa']+b[k+'_pressure_pa'])*(b[k+'_blood_ml']-a[k+'_blood_ml'])*1e-6*overlap/dt
 sample=[r for r in rows if start<=r['time_ms']<=end]
 return {'start_ms':start,'end_ms':end,'aortic_ejected_ml':flows['aortic_ml_s'],'pulmonary_ejected_ml':flows['pulmonary_ml_s'],'mitral_filled_ml':flows['mitral_ml_s'],'tricuspid_filled_ml':flows['tricuspid_ml_s'],'lv_pv_work_j':work['lv'],'rv_pv_work_j':work['rv'],'extrema':{k:{'min':min(r[k] for r in sample),'max':max(r[k] for r in sample)} for k in ['lv_blood_ml','rv_blood_ml','lv_pressure_pa','rv_pressure_pa']}}
cycles=[interval(i*period,(i+1)*period) for i in range(int(last//period))]
result={'accepted_steps':int(rows[-1]['accepted_step']),'time_ms':last,'validity_limits_satisfied':True,'max_blood_error_ml':max(abs(r['total_blood_ml']-5150) for r in rows),'observed_interval':interval(0,last),'complete_pacing_cycles':len(cycles),'cycles':cycles,'first_positive_outlet_samples':{k:next(({'accepted_step':int(r['accepted_step']),'time_ms':r['time_ms'],'flow_ml_s':r[k]} for r in rows if r[k]>0),None) for k in ['aortic_ml_s','pulmonary_ml_s']},'latest':rows[-1],'physics_lineage':manifest['physics_lineage'],'qualification':'Measurements only. Repeated electrical contraction, relaxation, both outlet ejections, PV closure and rollback must be assessed together. Partial prefix is not a heartbeat result.','pv_work_convention':'Minus trapezoidal integral of cavity pressure against blood volume; not whole-system energy closure.','flow_convention':'Native implicit endpoint flow times accepted timestep; split by overlap for off-grid cycle boundaries.'}
(p/'cycle-metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
