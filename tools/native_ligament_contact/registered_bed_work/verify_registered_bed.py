"""Verify the matched source control, native failure boundary and physical transpose test."""
import csv,json,sys
from pathlib import Path
from run_bed_work import digest

def verify(case,parent):
    read=lambda p:json.loads(p.read_text())
    def rows(p):
        with p.open() as f:return list(csv.DictReader(f))
    control=case/'native-columns'
    for k in ('resting-com-support-impulses.csv','resting-support-selected-skin-witness.csv','resting-coupled.csv'):
        sample=rows(control/k)
        if not sample or sample!=rows(parent/'finite-16'/k)[:len(sample)]:raise ValueError("control changed "+k)
    a,b=[read(case/k/'terminal-bed-certificate.json') for k in ('entry-8','strict-9')]
    r=read(case/'material-32/failure.json')['first_failure_latch'];w=read(case/'resolved-motion-witness.json')
    if a['step']!=8 or a['count_beyond_material_domain'] or b['step']!=9 or b['count_beyond_material_domain']!=b['resolved_count'] or b['resolved_count']<=0:raise ValueError("endpoint qualification changed")
    if r['step']!=b['step'] or r['respiration_accepted_status'][0]!=r['step'] or r['brain_accepted_metadata'][1]!=round(b['time_s']*1e6):
        raise ValueError("transaction clocks do not match")
    q=b['witnesses'][0]
    if q['vertex']!=w['vertex'] or w['pre_dynamics_step']!=b['step'] or abs(w['depth_m']-q['physical_depth_m'])>b['opening_replay_error_m']:
        raise ValueError("native domain and independent replay diverge")
    c=read(case/'live-coupon.json')
    if c['max_relative_force_error']>.001 or c['max_relative_tangent_error']>.004 or c['max_force_normalized_transfer_error']>.001 or c['open_row_N']!=0 or not c['excess_strain_propagated']:raise ValueError("failed native constitutive check")
    return dict(parent=read(parent/'finite-16/motion.json')['pack_sha256'],device=read(parent/'finite-16/motion.json')['device'],
      gpu_shader=digest(case/'live-owner/metalrobo/HumanRestingSkin.metallib'),stand_shader=digest(case/'live-owner/metalrobo/MetalRobo.metallib'),
      executable=digest(case/'live-owner/numi-human-native'),control_match=True,admitted_endpoint=a,rejected_endpoint=b,
      live_material_witness=w,accepted_brain_time_us=r['brain_accepted_metadata'][1],first_failed_candidate_brain_time_us=r['brain_candidate_metadata'][1],virtual_work=c)
if __name__=='__main__':print(json.dumps(verify(*(Path(a) for a in sys.argv[1:])),indent=2,allow_nan=False))
