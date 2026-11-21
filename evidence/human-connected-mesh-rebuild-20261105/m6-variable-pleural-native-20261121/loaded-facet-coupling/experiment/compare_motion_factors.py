from pathlib import Path
import csv,json,re,sys
import numpy as np
from material_context import sha
a,b,out=map(Path,sys.argv[1:])
def table(p):return list(csv.DictReader(p.open()))
def read(p):return json.loads((p/'qualification.json').read_text())
matched=min(read(a)['accepted_steps'],read(b)['accepted_steps'])
def endpoints(p):
    t=(p/'stdout.log').read_text()
    return {int(n):float(ms) for ms,n in re.findall(r'authoritative_segment_end wall_elapsed_ms=([0-9.]+) stage_step=(\d+)',t) if int(n)<=matched}
def native(p):return {int(r['candidate_step']):r for r in table(p/'candidate-contact.csv') if int(r['candidate_step'])<matched}
def error(x,y):
    den=max(float(np.max(abs(x))),1e-30);peak=float(np.max(abs(y-x)))
    nonzero=x!=0
    return dict(max_abs=peak,max_abs_over_reference_peak=peak/den,
        max_relative_at_nonzero_reference=float(np.max(abs((y[nonzero]-x[nonzero])/x[nonzero]))) if nonzero.any() else None,
        reference_peak=float(np.max(abs(x))))
t,u=endpoints(a),endpoints(b);common=sorted(t.keys()&u.keys());sa,sb=native(a),native(b)
if set(sa)!=set(range(matched)) or set(sb)!=set(sa):raise ValueError('Missing physical endpoint')
fields=['force_r_n','force_l_n','energy_r_j','energy_l_j','min_J','max_J','stress_Pa','energy_density_J_m3']
result={'control':a.name,'trial':b.name,'matched_roots':matched,'control_accepted':read(a)['accepted_steps'],'trial_accepted':read(b)['accepted_steps'],
    'matched_native_segment_wall_ms':[dict(root=n,control=t[n],blocked=u[n]) for n in common],'physical_endpoints':{},'physiology':{}}
for k in fields:result['physical_endpoints'][k]=error(np.array([float(sa[i][k]) for i in range(matched)]),np.array([float(sb[i][k]) for i in range(matched)]))
def phys(p):return {round(float(r['time_s'])/.002):r for r in table(p/'resting-coupled.csv') if 0<round(float(r['time_s'])/.002)<=matched}
fa,fb=phys(a),phys(b);shared=sorted(fa.keys()&fb.keys());result['physiology_accepted_roots']=shared
for k in ['lung_volume_ml','pleural_pa','alveolar_pa','airflow_ml_s','PaO2_mmhg','PaCO2_mmhg','lv_ml','rv_ml','lv_mmhg','rv_mmhg','blood_ml','aortic_ejected_ml','pulmonary_ejected_ml','root_assistance_n','root_assistance_nm']:
    if shared and k in fa[shared[0]]:result['physiology'][k]=error(np.array([float(fa[i][k]) for i in shared]),np.array([float(fb[i][k]) for i in shared]))
out.write_text(json.dumps(result,indent=2)+'\n')
