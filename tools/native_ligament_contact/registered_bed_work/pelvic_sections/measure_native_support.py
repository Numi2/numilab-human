"""Measure per-source normal support wrench at the native accepted sample times."""
import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(p):
    return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def table(p):
    with p.open() as f:return list(csv.DictReader(f))

def measure(case):
    launch=read(case/'launch.json')
    end=read(case/'terminal.receipt.json')
    if end['physical_endpoint']!='accepted':raise ValueError('Missing transaction receipt')
    n=end['accepted_step']
    argv=launch['argv']; dt=float(argv[argv.index('--muscle-step-seconds')+1])
    if dt!=.002 or n!=int(argv[argv.index('--muscle-step-count')+1]) or n%32: raise ValueError('Unmatched physical clock')
    if launch['environment'].get('NM_VASCULAR_STEP_EXTERNAL_PRESSURE')!='1':raise ValueError('Pleural coupling is disabled')
    points=case/'resting-support-selected-skin-witness.csv'
    impulses=case/'resting-com-support-impulses.csv'
    momentum=table(case/'resting-com-momentum-diagnostic.csv')
    trace=table(case/'resting-coupled.csv')
    frames={int(x['accepted_step']):x for x in momentum}
    common=sorted(frames)
    if common!=list(range(32,n+1,32)) or [int(x['step']) for x in trace]!=common:raise ValueError('Incomplete coupled telemetry')
    reg=defaultdict(list)
    for x,y in zip(table(points),table(impulses),strict=True):
        s=int(y['accepted_step']); sid=int(y['source_geometry_index'])
        if (s!=int(x['accepted_step']) or s!=int(x['pre_dynamics_step_index'])+1 or x['region']!=y['contact_index']
            or sid!=int(x['source_geometry_index']) or y['row_class']!='external_support_contact'):
            raise ValueError('Contact force and selected point are not matched')
        f=float(y['normal_impulse_ns'])/dt
        r=np.array([float(x['selected_point_world_'+a+'_m']) for a in 'xyz'])
        normal=np.array([float(y['normal_'+a]) for a in 'xyz'])
        if not np.all(np.isfinite(np.r_[r,normal,f])) or f<0 or abs(np.linalg.norm(normal)-1)>1e-6:
            raise ValueError('Invalid support state')
        reg[s,sid].append((f,r,normal))
    tail=[s for s in common if s*dt>n*dt-1]
    result={}
    for sid in (33529,20347):
        forces=[]; moments=[]; jumps=[]; active=[]; center=None
        for s in common:
            part=reg[s,sid]
            if not part:raise ValueError('Missing pelvic footprint')
            F=sum(f for f,_,_ in part)
            active.append(sum(f>0 for f,_,_ in part))
            com=np.array([float(frames[s]['com_'+a+'_m']) for a in 'xyz'])
            # The selected point is measured just before the accepted substep.
            # Use the sampled endpoint COM solely as the common torque origin.
            moment=sum((np.cross(r-com,norm*f) for f,r,norm in part),start=np.zeros(3))
            new=sum((f*r for f,r,_ in part),start=np.zeros(3))/F if F>0 else None
            if new is not None and center is not None:jumps.append(float(np.linalg.norm((new-center)[:2])*1000))
            center=new
            if s in tail: forces.append(F);moments.append(moment)
        result[str(sid)]=dict(last_s_mean_normal_N=np.mean(forces),mean_normal_moment_about_sampled_COM_Nm=np.mean(moments,axis=0).tolist(),
                              simultaneously_loaded_sections_max=max(active),load_center_jump_max_mm=max(jumps,default=None))
    F=np.array([float(frames[s]['normal_force_last_physical_step_n']) for s in tail])
    mass=float(frames[n]['total_body_mass_kg'])
    phys={k:[min(float(r[k]) for r in trace), max(float(r[k]) for r in trace)] for k in
          ('pleural_pa','PaO2_mmhg','PaCO2_mmhg','right_atrium_ml','lv_ml','blood_error_ml','root_assistance_n','root_assistance_nm','peak_penetration_m','max_abs_body_mass_residual_kg')}
    if not np.all(np.isfinite(np.array(list(phys.values())))):raise ValueError('Nonfinite physiology')
    surface=table(case/'resting-surface-audit.csv')
    if [int(r['step']) for r in surface] != [0, *(s-1 for s in common), n]:raise ValueError('Unmatched surface submission and terminal samples')
    wall=read(case/'execution.json')['cold_wall_s']
    return dict(accepted_seconds=end['accepted_time_s'], sample_count=len(common),end_to_end_wall_s=wall,
                cold_real_time_factor=end['accepted_time_s']/wall,mass_kg=mass,last_s_mean_whole_normal_N=float(F.mean()),
                last_s_whole_support_fraction_of_weight=float(F.mean()/(mass*9.81)), sources=result,
                geometry_pack=end['pack_file_sha256'],hashes={p.name:sha(p) for p in (points,impulses,case/'resting-coupled.csv',case/'launch.json')},
                coupled_ranges=phys, cycles={k:int(float(trace[-1][k])) for k in ('breaths','complete_filling_ejection_cycles')},
                mean_body_velocity_last_second_m_s=np.mean([[float(frames[s]['com_v'+a+'_m_s']) for a in 'xyz'] for s in tail],axis=0).tolist(),
                min_sampled_skin_bed_gap_m=min(float(r['min_skin_bed_gap_m']) for r in surface))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('case','output'):p.add_argument(k,type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(measure(a.case),indent=2,allow_nan=False)+'\n')
