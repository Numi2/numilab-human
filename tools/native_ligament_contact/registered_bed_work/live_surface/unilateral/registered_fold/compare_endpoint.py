"""Check the accepted physical fold, matched rollback and support against the pinned parent."""
import json
from pathlib import Path
import sys
import hashlib
import subprocess

import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from verify_bonded_native import matched_horizon

def read(path):
    return json.loads(path.read_text())

def check(root):
    case=root/'fold-repair'
    failed=case/'three'
    failure=read(failed/'failure.json')
    prefix=case/f"accepted-{failure['segment_start']}"
    parent=root/'feasible-bed/two'
    law=read(case/'admission.json')
    a,b=(read(p/'launch.json') for p in (failed,prefix))
    if a['argv'][0]!=b['argv'][0] or a['pins']!=b['pins'] or sha(Path(a['argv'][0]))!=law['binary']:
        raise ValueError('Runs used different physical owners')
    runs=[matched_horizon(job) for job in (a,b)]
    # Only the sampled root of these four recording channels may differ.
    for record in runs:
        for name in ('NM_FEM_NEWTON_TRACE_ROOT','NM_FEM_BACKTRACKING_TRACE_ROOT',
                     'NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT','NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT'):
            record[3].pop(name)
    if runs[0]!=runs[1]:
        raise ValueError('Unmatched physical input, forcing, solver or capture')
    def loads(p):
        return [s for s in (p/'stdout.log').read_text().splitlines() if s.startswith('accepted_fem_body_load_capture=accepted ')]
    tr='native-traces/resting-coupled.csv'
    if (len(loads(prefix))!=failure['segment_start'] or loads(prefix)!=loads(failed) or
        (prefix/tr).read_bytes()!=(failed/tr).read_bytes()):
        raise ValueError('Different accepted physical history')
    src=read(case/'source.json')
    ids=src['registered_nodes']
    def gap(pos):
        p,a,b,c=pos[ids].astype(float)
        n=np.cross(b-a,c-a)
        return 1e9*float((p-a)@n)/np.linalg.norm(n)
    def measure(p):
        ref,loaded,tets,roles,pins=inputs(p)
        audit=read(p/'verified-endpoint.json')
        obs=p/'physical-accepted.node'
        with subprocess.Popen(['zstd','-dcq',str(p/'accepted-terminal.mrvpack.zst')],stdout=subprocess.PIPE) as proc:
            frame_hash=hashlib.file_digest(proc.stdout,'sha256').hexdigest()
            if proc.wait():raise ValueError('Could not restore accepted frame')
        if sha(obs)!=audit['physical_nodes_sha256'] or frame_hash!=audit['frame_sha256']:
            raise ValueError('Unmatched accepted frame')
        end=np.loadtxt(obs,skiprows=1,usecols=(1,2,3))
        from fit_posterior_equilibrium import materials
        from supported_material_admission import installed_topology,material_state,stress_limits
        tets,roles=installed_topology(tets,roles,read(p/'launch.json')['environment'])
        props,materials_used=materials(p.parent/'owner')
        limits=stress_limits(materials_used)
        stretch={str(m):dict(outside=0,severe=0,stress_cap=0) for m in (15,16)}
        for i in range(0,len(tets),32768):
            local=tets[i:i+32768]
            role=roles[i:i+len(local)]
            _,v,stress=material_state(ref[local].astype(float),end[local],props[role-15])
            for m in (15,16):
                use=role==m
                stretch[str(m)]['outside']+=int(np.count_nonzero(use&((v[:,0]>1.5)|(v[:,-1]<2/3))))
                stretch[str(m)]['severe']+=int(np.count_nonzero(use&((v[:,0]>3)|(v[:,-1]<1/3))))
                stretch[str(m)]['stress_cap']+=int(np.count_nonzero(use&(stress>limits[m-15])))
        return dict(material_stretch=stretch,registered_geometry=pins,endpoint=sha(p/'verified-endpoint.json'),
            accepted_s=audit['simulated_s'],complete_wall_s=audit['complete_wall_s'],complete_factor=audit['end_to_end_factor'],
            material_geometry=audit['material_audits'],crossings=audit['endpoint_anatomy'],
            feature_signed_gap_nm=dict(reference=gap(ref),loaded=gap(loaded),accepted=gap(end)))
    measures=[measure(p) for p in (parent,prefix)]
    if measures[0]['accepted_s']!=measures[1]['accepted_s']:raise ValueError('Different physical time')
    from compare_dense_support import observed
    target,force=observed(prefix)
    return dict(registered_source=sha(case/'source.json'),source_triangle=src['outer_triangle'],source_nodes=ids,
        before=measures[0],after=measures[1],bed=read(case/'motion.json'),skeleton_weight_N=target,
        FEM_on_articulation_N=force,matched_prefix_body_force_history=True,matched_coupled_trace=sha(prefix/tr),
        rejection_code=failure['matter_code'],rejection_diagnostics=failure['matter_diagnostics'],
        rollback_common=failure['candidate_common_coordinates']==failure['accepted_common_coordinates'],
        rollback_respiratory=failure['respiratory_candidate_status']==failure['respiratory_accepted_status'])
if __name__=='__main__':
    print(json.dumps(check(Path(sys.argv[1]).resolve()),indent=2,allow_nan=False))
