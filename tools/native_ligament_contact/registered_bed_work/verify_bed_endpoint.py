"""Check the accepted prefix and the first tissue-domain failure against independent geometry."""
import json,re,sys
from pathlib import Path
from material_context import sha

def verify(control,prior_failure,prefix,rejected):
    def read(p):return json.loads(p.read_text())
    old,new=[read(p/'terminal.receipt.json') for p in (control,prefix)]
    if old['physical_endpoint']!='accepted' or new['accepted_step']!=old['accepted_step'] or new['accepted_body_poses']!=old['accepted_body_poses']:
        raise ValueError('Changed accepted physical endpoint')
    for k,b in old['accepted_cardiac_geometry_buffers'].items():
        if b['bytes']!=new['accepted_cardiac_geometry_buffers'][k]['bytes'] or b['sha256']!=new['accepted_cardiac_geometry_buffers'][k]['sha256']:
            raise ValueError('Changed accepted heart geometry')
    traces=('resting-coupled.csv','resting-com-support-impulses.csv','resting-support-selected-skin-witness.csv')
    if any(sha(control/n)!=sha(prefix/n) for n in traces):
        raise ValueError('Changed accepted coupled or support measurements')
    a,b=[read(p/'terminal-bed-certificate.json') for p in (control,prefix)]
    fields=('step','time_s','columns','map','influences','checked_columns','count_beyond_material_domain','largest_signed_excess_m')
    if any(a[k]!=b[k] for k in fields) or b['count_beyond_material_domain']!=0:
        raise ValueError('Changed endpoint physical column state')
    w=read(rejected.with_name(rejected.name+'-source')/'witness.json')
    comp=read(prior_failure/'terminal-bed-certificate.json')
    by_vertex={j['vertex']:j for j in comp['witnesses']}
    f=read(rejected/'failure.json')['first_failure_latch']
    receipt=read(rejected/'execution.json')
    n=new['accepted_step'];vid=w['vertex'];row=by_vertex[vid]
    expected_time=round(b['time_s']*1.e6)
    if not(receipt['exit_code'] and f['step']==w['material_step_index']==n and f['stand_code']==5 and f['stand_status'][1]==vid
           and f['respiration_accepted_status'][0]==n and f['brain_accepted_metadata'][1]==expected_time
           and f['brain_candidate_metadata'][1]>expected_time and comp['step']==n+1
           and comp['count_beyond_material_domain']==w['invalid_columns_observed'] and w['depth_m']>w['allowed_depth_m']
           and abs(w['depth_m']-row['physical_depth_m'])<3e-7 and abs(w['allowed_depth_m']-row['allowed_depth_m'])<1e-8):
        raise ValueError('Material rejection or accepted clock disagrees with independent pose')
    message=(rejected/'stderr.log').read_text()
    if not re.search(rf'code=5 completed_steps={n} failing_index={vid}\b',message) or (rejected/'terminal.receipt.json').exists():
        raise ValueError('Unexpected native acceptance report')
    return dict(scope='Registered lower surface and reduced skin/fat bed model. Full-volume motion is tested separately.',
        control=sha(control/'launch.json'),prefix=sha(prefix/'launch.json'),rejected=sha(rejected/'launch.json'),
        accepted_steps=n,accepted_s=b['time_s'],source_columns=b['checked_columns'],minimum_remaining_depth_m=-b['largest_signed_excess_m'],
        same_body_poses=len(new['accepted_body_poses']),identical_traces={k:sha(prefix/k) for k in traces},
        same_cardiac_buffers={k:v['sha256'] for k,v in new['accepted_cardiac_geometry_buffers'].items()},
        first_failure=w,independent_pose_depth_m=row['physical_depth_m'],accepted_brain_us=f['brain_accepted_metadata'][1],
        trial_brain_us=f['brain_candidate_metadata'][1],cold_wall_s=read(prefix/'execution.json')['cold_wall_s'])
if __name__=='__main__':print(json.dumps(verify(*(Path(x).resolve() for x in sys.argv[1:])),indent=2))
