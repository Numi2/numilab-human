"""Check native clocks, load, conservation and matched trajectories against the knee-supported control."""
import csv,hashlib,json,math,re,sys
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def table(d,t):return list(csv.DictReader((d/'native-traces'/t).open()))
def series(r,k):return np.array([float(x[k]) for x in r])
def trial(d):
    cfg=read(d/'launch.json');rows=table(d,'resting-coupled.csv');raw=table(d,'resting-com-support-impulses.csv')
    if not rows:return {'accepted_steps':0,'execution':read(d/'execution.json')}
    n=len(rows);dt=float(cfg['argv'][cfg['argv'].index('--muscle-step-seconds')+1])
    assert list(series(rows,'step'))==list(range(1,n+1)) and np.max(abs(series(rows,'time_s')/(np.arange(n)+1)-dt))<1e-10
    assert np.max(abs(series(rows,'root_assistance_n')))==0 and np.max(abs(series(rows,'root_assistance_nm')))==0
    assert cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1'
    assert len(raw)==32*n
    assert all(int(row['accepted_step'])==i//32+1 and int(row['contact_index'])==i%32 for i,row in enumerate(raw))
    force=np.stack([series(raw,'impulse_world_'+k+'_ns') for k in 'xyz'],-1).reshape(n,32,3)/dt
    assert np.isfinite(force).all()
    text=(d/'stdout.log').read_text(); scope={}
    for l in text.splitlines():
        if l.startswith('resting_integrated_throughput ') and 'real_time_factor='in l:scope['timed_integrated_interval']=l
    failure=read(d/'failure.json')if (d/'failure.json').is_file()else None
    if failure: assert int(failure['segment_start'])==n
    coverage={};t=d/'native-traces/accepted-geometry'
    for p in t.glob('*.receipt.json'):
        v=read(p);f=p.with_name(p.name.replace('.receipt.json','.mrvpack'));assert sha(f)==v['pack_file_sha256'];coverage[p.name]=dict(pack=sha(f),time=v['accepted_time_s'],accepted_root=v['accepted_root_fingerprint_hex'])
    end=read(d/'execution.json')
    return dict(accepted_steps=n,simulated_s=rows[-1]['time_s'],execution=end,process_rtf=float(rows[-1]['time_s'])/end['wall_s'],**scope,
                failure=({k:failure[k] for k in ('segment_start','matter_code','matter_object_index','matter_diagnostics')} if failure else None),
                max_abs_body_mass_residual_kg=float(np.max(abs(series(rows,'body_mass_residual_kg')))),max_abs_blood_error_ml=float(np.max(abs(series(rows,'blood_error_ml')))),
                max_abs_gas_balance_ml={k:float(np.max(abs(series(rows,k+'_balance_error_stpd_ml'))))for k in ('oxygen','co2')},
                last_breath_count=int(rows[-1]['breaths']),last_filling_ejection_cycles=int(rows[-1]['complete_filling_ejection_cycles']),
                ranges={k:[float(series(rows,k).min()),float(series(rows,k).max())]for k in ('pleural_pa','lung_volume_ml','lv_mmhg','PaCO2_mmhg','PaO2_mmhg')},
                launch_sha256=sha(d/'launch.json'),zero_root_assistance=True,material_bed_rows=32,captures=coverage),rows,force
def report(ref,stage,out):
    base,b,f=trial(ref);v,a,g=trial(stage);n=min(len(a),len(b))
    v['common_prefix']={'steps':n,'bed_force_relative_l2':float(np.linalg.norm(g[:n]-f[:n])/np.linalg.norm(f[:n])),
       'max_abs_delta':{k:float(np.max(abs(series(a[:n],k)-series(b[:n],k))))for k in ('pleural_pa','lung_volume_ml','lv_mmhg','rv_mmhg','PaO2_mmhg','PaCO2_mmhg')}}
    out.write_text(json.dumps(v,indent=2)+'\n')
if __name__=='__main__':report(*(Path(s)for s in sys.argv[1:]))
