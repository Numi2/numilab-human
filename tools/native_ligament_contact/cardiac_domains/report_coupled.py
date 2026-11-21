"""Summarize one complete supported native run with timing and physical ownership."""
import csv,hashlib,json,math,re,sys
from pathlib import Path
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def run(path,reference,owner,out):
    sys.path.insert(0,str(owner));from compare import audit
    d=audit(path,path);cfg=json.loads((path/'launch.json').read_text())
    data=read(path/'resting-coupled.csv');base=read(reference/'resting-coupled.csv')
    assert len(data)>=len(base) and d['native_exit_code']==0 and cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1'
    assert all(math.isfinite(float(row[k])) for row in data for k in ('time_s','blood_error_ml','pleural_pa','body_mass_residual_kg'))
    bad={}
    for k in data[0]:
        pairs=[(a[k],b[k]) for a,b in zip(data,base) if a[k]!=b[k]]
        if pairs: bad[k]=len(pairs)
    d['matched_16s_trace_columns']=len(base[0])-len(bad);d['differing_16s_trace_columns']=bad
    d['matched_16s_trace_rows']=len(base);d['simulated_duration_s']=float(data[-1]['time_s'])
    steps=int(cfg['argv'][cfg['argv'].index('--muscle-step-count')+1]);dt=float(cfg['argv'][cfg['argv'].index('--muscle-step-seconds')+1])
    assert d['final_accepted_step']==steps and abs(steps*dt-d['simulated_duration_s'])<1e-5
    assert d['max_root_force_n']==d['max_root_torque_nm']==0 and d['pleural_pa_range'][0]<d['pleural_pa_range'][1]
    for k in ('blood_ml','blood_error_ml','oxygen_balance_error_stpd_ml','co2_balance_error_stpd_ml','tidal_ml','last_lv_stroke_ml','PaO2_mmhg','PaCO2_mmhg','body_mass_residual_kg'):
        xx=[float(r[k]) for r in data];d[k+'_range']=[min(xx),max(xx)]
    d['presentations_per_wall_second']=d['presentation_callbacks']/d['wall_seconds']
    d['source_sha256']={p.name:digest(p) for p in [path/'launch.json',path/'resting-coupled.csv',path/'resting-com-support-impulses.csv',path/'execution.json',path/'stdout.log',path/'stderr.log']}
    out.write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
