"""Compare accepted progress and measured tissue loads of matched native releases."""
import json
import re
import sys
from pathlib import Path
import numpy as np
from material_context import sha
from verify_bonded_native import matched_horizon
from fit_dorsal_material import inputs
from supported_material_admission import installed_topology,material_state,stress_limits
from fit_posterior_equilibrium import materials

def read(p):return json.loads(p.read_text())
def measure(root):
    case=root/'bonded-equilibrium'
    def config(run):
        j=read(run/'launch.json');cwd,args,pins,env=matched_horizon(j)
        for k in ('NUMI_MATTER_METALLIB_OVERRIDE','DYLD_LIBRARY_PATH'):
            path=Path(env.pop(k))
            if k=='DYLD_LIBRARY_PATH':path=path/'libmetalrobo.dylib'
            if pins.pop(str(path))!=sha(path):raise ValueError('Changed solver')
        for k in ('NM_FEM_NEWTON_TRACE_ROOT','NM_FEM_BACKTRACKING_TRACE_ROOT','NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT','NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT'):
            if env.pop(k) not in ('0','2'):raise ValueError('Unexpected diagnostic root')
        return cwd,args,pins,env
    def metrics(run):
        j=read(run/'launch.json');status=read(run/'execution.json')
        log=(run/'stdout.log').read_text();err=(run/'stderr.log').read_text()
        records=[json.loads(l.split('=',1)[1]) for l in err.splitlines() if l.startswith('matter_fgmres_microtick_work=')]
        loads=[];clocks=[]
        for l in log.splitlines():
            if l.startswith('accepted_fem_body_load_capture=accepted '):
                loads.append(json.loads(re.search(r'root_generalized_force_xyz_n=(\[[^\]]*\])',l)[1]))
            if l.startswith('human_standing_progress=accepted '):clocks.append(float(re.search(r'simulated_seconds=([^ ]+)',l)[1]))
        if len(loads)!=len(clocks):raise ValueError('Different accepted force and physical clocks')
        out=dict(launch=sha(run/'launch.json'),exit=status['exit'],wall_s=status['wall_s'],accepted=len(clocks),seconds=clocks[-1] if clocks else 0,loads_N=loads,
                 work=[{k:v for k,v in w.items() if k in ('root','used_columns','active_newton_begins','minimum_newton_norm','last_newton_norm','final_newton_latched','status_code_at_microtick_end')} for w in records])
        if status['exit']:
            f=read(run/'failure.json')
            if f['segment_start']!=len(clocks):raise ValueError('Unreported accepted step')
            out.update(matter_code=f['matter_code'],index=f['matter_failing_index'],diagnostics=f['matter_diagnostics'],
                       common_rollback=f['candidate_common_coordinates']==f['accepted_common_coordinates'],
                       respiration_rollback=f['respiratory_candidate_status']==f['respiratory_accepted_status'])
            if not (out['common_rollback'] and out['respiration_rollback']):raise ValueError('Incomplete rollback')
        else:
            r=read(run/'terminal.receipt.json')
            if r['accepted_step']!=len(clocks) or r['physical_endpoint']!='accepted':raise ValueError('Missing native endpoint')
        return out
    control=root/'material-direction/release/three-2';base=case/'three';exp=case/'bed-stop/three'
    paired=case/'bed-stop/paired/three'
    if not config(control)==config(base)==config(exp)==config(paired):raise ValueError('Unmatched physical scene or execution options')
    out={name:metrics(run) for name,run in [('unchanged_release',control),('fitted',base),('fitted_with_contact_stopping_test',exp),('fitted_with_matched_preflight_and_stopping_test',paired)]}
    if len({read(r/'launch.json')['environment']['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'] for r in (base,exp,paired)})!=1:raise ValueError('Different fitted states')
    # A terminal frame on the same material is independently compared against both
    # release references so a change of starting volume cannot hide a smaller cell.
    ref_run=root/'fold-repair/accepted-2'
    ref,baseline,src,roles,pins=inputs(ref_run);baseline=baseline.astype(float);ref=ref.astype(float)
    t,tags=installed_topology(src,roles,read(ref_run/'launch.json')['environment'])
    props,ident=materials(case/'owner');limits=stress_limits(ident)
    out['measured_endpoints']={}
    for run in [paired,paired.parent/'accepted-2',paired.parent/'accepted-1']:
        if not (run/'verified-endpoint.json').is_file():continue
        rep=read(run/'verified-endpoint.json');rc=read(run/'terminal.receipt.json')
        if sha(run/'physical-accepted.node')!=rep['physical_nodes_sha256'] or rc['accepted_step']!=rep['accepted_steps']:raise ValueError('Unmatched material endpoint')
        pts=np.loadtxt(run/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
        start=np.loadtxt(run/'physical-reference.node',skiprows=1,usecols=(1,2,3))
        stats=dict(min_J_parent=float('inf'),min_J_new=float('inf'),parent_floor_failures=0,new_floor_failures=0)
        stretch={str(m):dict(outside=0,severe=0,stress_cap=0) for m in (15,16)}
        for i in range(0,len(t),32768):
            c=t[i:i+32768];r=tags[i:i+len(c)]
            def det(a):return np.linalg.det(a[c[:,1:]]-a[c[:,:1]])
            v=det(pts)
            for key,arr in [('parent',baseline),('new',start)]:
                ratio=v/det(arr)
                if not np.isfinite(ratio).all():raise ValueError('Invalid ratio')
                stats['min_J_'+key]=min(stats['min_J_'+key],float(ratio.min()))
                stats[key+'_floor_failures']+=int(np.count_nonzero(ratio<=.35))
            _,sv,stress=material_state(ref[c],pts[c],props[r-15])
            for m in (15,16):
                sel=r==m
                for k,v0 in [('outside',(sv[:,0]>1.5)|(sv[:,-1]<2/3)),('severe',(sv[:,0]>3)|(sv[:,-1]<1/3)),('stress_cap',stress>limits[m-15])]:stretch[str(m)][k]+=int(np.count_nonzero(sel&v0))
        motion=read(run/'motion.json');text=(run/'stdout.log').read_text()
        fascia=next(l for l in text.splitlines() if l.startswith('resting_fascia_fem_terminal_audit=passed'))
        observed=metrics(run)
        out['measured_endpoints'][run.name]=dict(audit=rep,volume=stats,stretch=stretch,motion=motion,terminal_skeleton_upward_N=observed['loads_N'][-1][2],
                 bone_contact_normal_N=float(re.search('body_contact_normal_force_N=([^ ]+)',fascia)[1]))
    return out
if __name__=='__main__':print(json.dumps(measure(Path(sys.argv[1]).resolve()),indent=2,allow_nan=False))
