"""Compare physical prefix and rejection from the two matched whole-body launches."""
import json
import re
from pathlib import Path
import sys
import numpy as np
from compare_dense_support import observed
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from measure_loaded_skin_motion import measure
from supported_material_admission import installed_topology, material_state, stress_limits
from summarize_bonded_evidence import full_run
from verify_bonded_native import matched_horizon


def read(p):return json.loads(p.read_text())


def measure_release(case, prefix, trial, bed_file):
    b=read(prefix/'launch.json')
    if matched_horizon(b)!=matched_horizon(read(trial/'launch.json')):
        raise ValueError('Different physical configuration or solver')
    report=read(prefix/'verified-endpoint.json'); q=read(case/'owner/qualification.json')
    bed=read(bed_file);cap=read(prefix/'terminal.receipt.json')
    if sha(prefix/'physical-accepted.node')!=report['physical_nodes_sha256'] or cap['accepted_step']!=report['accepted_steps'] or bed['receipt']!=sha(prefix/'terminal.receipt.json'):raise ValueError('Unmatched accepted frame')
    ref,_,cell,tags,pins=inputs(prefix)
    if pins.get('installed_pose_sha256')!=q['current'] or pins.get('reference_sha256')!=q['parent']['reference_sha256']:raise ValueError('Unexpected reference or loaded material')
    t,role=installed_topology(cell,tags,b['environment'])
    x=np.loadtxt(prefix/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    mat,hashes=materials(case/'owner/native-owner');caps=stress_limits(hashes);counts={}
    for start in range(0,len(t),32768):
        sl=slice(start,start+32768);r=role[sl];idx=t[sl]
        _,s,stress=material_state(ref[idx].astype(float),x[idx],mat[r-15])
        for i in (15,16):
            totals=counts.setdefault(i,dict(stretch_envelope=0,severe=0,stress_cap=0))
            for name,flag in [('stretch_envelope',(s[:,0]>1.5)|(s[:,-1]<2/3)),('severe',(s[:,0]>3)|(s[:,-1]<1/3)),('stress_cap',stress>caps[i-15])]:
                totals[name]+=int(np.sum(flag&(r==i)))
    grav,forces=observed(prefix)
    raw=(prefix/'stdout.log').read_text()
    timing=[float(v) for v in re.findall(r'^human_execution_stage=native_horizon_(?:begin|end) wall_elapsed_ms=([\d.eE+-]+)',raw,re.M)]
    if len(timing)!=2 or timing[-1]<=timing[0]:raise ValueError('Missing native horizon')
    body=(timing[-1]-timing[0])/1000
    current=dict(endpoint=report,loads=forces,articulated_gravity_N=grav,terminal_bed_N=bed['native']['mean_normal_force_N'],
                 weight_N=bed['weight_N'],loaded_bed_N=bed['initial_barrier_N'],body_s=body,
                 body_real_time_factor=report['simulated_s']/body,skin_motion=measure(prefix),material_stretch=counts,
                 pleural_owner_enabled='resting_vascular_external_pressure=accepted_absolute_pleural_gauge' in raw,
                 prefix_launch=sha(prefix/'launch.json'),three_step_launch=sha(trial/'launch.json'))
    if (trial/'failure.json').exists():
        state=read(trial/'failure.json')
        if state['segment_start']!=report['accepted_steps']:raise ValueError('Different failure horizon')
        def loads(log):return [s for s in log.splitlines() if s.startswith('accepted_fem_body_load_capture=accepted')]
        if len(loads(raw))!=report['accepted_steps'] or loads(raw)!=loads((trial/'stdout.log').read_text()):raise ValueError('Different prefix force histories')
        current['rejection']=dict(trial=full_run(trial),matter_object=state['matter_object_index'],code=state['matter_code'],
                common_rollback=state['candidate_common_coordinates']==state['accepted_common_coordinates'],
                respiratory_rollback=state['respiratory_candidate_status']==state['respiratory_accepted_status'])
    return current


def check(case,control,out):
    prefix=case/'native-prefix' if (case/'native-prefix').is_dir() else case/'native-three'
    trial=case/'native-three'
    a=read(control/'conserved-two/launch.json');b=read(prefix/'launch.json');full=read(trial/'launch.json')
    if not matched_horizon(a)==matched_horizon(b)==matched_horizon(full):raise ValueError('Unmatched physical runs')
    current=measure_release(case,prefix,trial,case/'native-bed.json')
    old=read(control/'conserved-summary.json')['native']
    selected=['terminal_bed_N','articulated_gravity_N','weight_N','loads','body_real_time_factor','material_stretch','endpoint','rejection']
    result=dict(baseline={k:old[k] for k in selected},candidate=current,source=sha(control/'conserved-summary.json'),qualification=sha(case/'owner/qualification.json'))
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':check(*(Path(s).resolve() for s in sys.argv[1:]))
