"""Compare load paths in two independently initialized native whole-body scenes."""
import csv,json,sys
from pathlib import Path
import numpy as np
from compare import get
from compile_connected_joint import digest
def more(case):
    base=case/'native-traces'
    with (base/'resting-com-support-impulses.csv').open() as f:
        rows=list(csv.DictReader(f))
    dt=get_dt(case)
    f=np.array([float(r['normal_impulse_ns']) for r in rows]).reshape(64,32)/dt
    with (base/'resting-surface-audit.csv').open() as st:
        audit=list(csv.DictReader(st))
    log=(case/'stdout.log').read_text().splitlines()
    forces=next(json.loads(l.split('=',1)[1])for l in log if l.startswith('persistent_initial_force_reference='))
    d={int(r['dof']):r for r in forces['rows']}
    v={}
    for side,which,hip,knee in [('right',[10,20,21],100,105),('left',[11,30,31],114,119)]:
        loads=f[:,which].sum(1)
        v[side]=dict(reaction_range_N=[float(loads.min()),float(loads.max())],count_above_25_N=int((loads>25).sum()),
            heel_reaction_N_first_last=[float(f[0,which[1]]),float(f[-1,which[1]])],hip=d[hip],knee=d[knee])
    return dict(legs=v,compiled_balance=[l for l in log if l.startswith(('initial_gravity_total=','resting_bed_section_preload=','resting_initialization=dynamic_release'))])
def get_dt(case):
    a=json.loads((case/'launch.json').read_bytes())['argv']
    return float(a[a.index('--muscle-step-seconds')+1])
def main(before,after,out):
    ref=get(before);trial=get(after)
    if ref['time_s']!=trial['time_s'] or ref['dt_s']!=trial['dt_s'] or ref['start_q']==trial['start_q']: raise ValueError('Unmatched time or unchanged construction')
    a,b=[json.loads((c/'launch.json').read_bytes())for c in (before,after)]
    for opt in ['--joint-equality-payload','--support-contact-payload','--tendon-payload','--skin-payload','--soft-tissue-payload','--pectoralis-fascia-payload','--torso-anatomy-payload']:
        getarg=lambda c:c['argv'][c['argv'].index(opt)+1]
        if getarg(a)!=getarg(b):raise ValueError(('Changed physical input',opt))
    for i in (1,2,3):
        if a['argv'][i]!=b['argv'][i]:raise ValueError('Changed musculoskeletal input')
    if any(trial['signals'][s]!=[0.,0.] for s in ['root_assistance_n','root_assistance_nm']):raise ValueError('Nonzero root assistance')
    ratios={k:trial[k]/ref[k]for k in ['horizontal_contact_rms_N','com_horizontal_speed_m_s']}
    result=dict(control=dict(**ref,**more(before)),candidate=dict(**trial,**more(after)),ratios=ratios)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(ratios,trial['last_contact_N'],{k:(v['reaction_range_N'],v['count_above_25_N'])for k,v in result['candidate']['legs'].items()})
if __name__=='__main__':main(*(Path(p).resolve() for p in sys.argv[1:]))
