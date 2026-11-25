"""Compare the two complete, time-matched physical transactions."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
def get(case):
    launch=json.loads((case/'launch.json').read_bytes());r=json.loads((case/'execution.json').read_bytes())
    if r['exit']!=0:raise ValueError('Incomplete trial')
    p=case/'native-traces';rc,=(p/'accepted-geometry').glob('*.receipt.json');rc=json.loads(rc.read_bytes())
    if digest(Path(rc['accepted_pack_path']))!=rc['pack_file_sha256'] or rc['accepted_step']!=64 or rc['physical_endpoint']!='accepted':raise ValueError('Invalid endpoint')
    def rows(table):
        with (p/(table+'.csv')).open() as f:return list(csv.DictReader(f))
    q=rows('resting-com-q-integration');m=rows('resting-com-momentum-diagnostic');coupled=rows('resting-coupled');bed=rows('resting-com-support-impulses');s=q[0];dt=float(s['dt_s'])
    if not q or q[0]['accepted_step']!='1' or [int(r['accepted_step']) for r in m]!=list(range(1,65)) or [int(r['step']) for r in coupled]!=list(range(1,65)) or len(bed)!=32*64 or launch['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Different acquisition')
    f=np.zeros((64,32,3));seen=set()
    for row in bed:
        i,j=int(row['accepted_step'])-1,int(row['contact_index']);seen.add((i,j))
        if int(row['segment_steps'])!=1:raise ValueError('Averaged contact step')
        f[i,j]=[float(row['impulse_world_'+a+'_ns'])/dt for a in 'xyz']
    if len(seen)!=2048:raise ValueError('Missing skin contact')
    v=np.array([[float(row['com_v'+a+'_m_s'])for a in 'xyz']for row in m])
    vec=f.sum(axis=1)
    log=(case/'stdout.log').read_text()
    audits=[line for line in log.splitlines() if line.startswith(('resting_FEM_terminal_audit=','resting_fascia_fem_terminal_audit=','resting_integrated_throughput ','initial_material_shear_recruitment ','resting_initialization=dynamic_release','resting_initialization=static_'))]
    if len([a for a in audits if 'terminal_audit=passed' in a])!=2:raise ValueError('Native tissue gate failed')
    signals={}
    for key in ['root_assistance_n','root_assistance_nm','pleural_pa','lung_volume_ml','blood_error_ml','PaCO2_mmhg','PaO2_mmhg']:
        a=np.array([float(r[key]) for r in coupled]);signals[key]=[a.min(),a.max()]
        if not np.isfinite(a).all():raise ValueError('Nonfinite physiology')
    return dict(launch=digest(case/'launch.json'),pack=rc['pack_file_sha256'],start_q=q[0]['q_before_fingerprint_fnv64'],dt_s=dt,time_s=rc['accepted_time_s'],
        walls=r['wall_s'],whole_RTF=rc['accepted_time_s']/r['wall_s'],horizontal_contact_rms_N=float(np.sqrt(np.mean(np.sum(vec[:,:2]**2,axis=1)))),last_contact_N=vec[-1].tolist(),
        com_horizontal_speed_m_s=float(np.linalg.norm(v[-1,:2])),first_contact_force_N=f[0].tolist(),last_contact_force_N=f[-1].tolist(),
        signals=signals,audits=audits)
if __name__=='__main__':
    a,b,out=map(Path,sys.argv[1:]);x=get(a);y=get(b)
    if x['start_q']!=y['start_q'] or x['dt_s']!=y['dt_s']:raise ValueError('Changed loaded pose or time base')
    ratios={k:y[k]/x[k]for k in ['horizontal_contact_rms_N','com_horizontal_speed_m_s']}
    out.write_text(json.dumps(dict(control=x,candidate=y,ratios=ratios),indent=2)+'\n')
    print(ratios,x['last_contact_N'],y['last_contact_N'])
