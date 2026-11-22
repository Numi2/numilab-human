"""Compare accepted dense-tissue loads on the skeleton with the simultaneous bed reaction."""
from pathlib import Path
import json
import re
import sys
import numpy as np
from material_context import sha


def measure(reference, observed):
    source = json.loads((reference/'launch.json').read_text())
    obs = json.loads((observed/'launch.json').read_text())
    if (source['argv'][:4] + source['argv'][5:] != obs['argv'][:4] + obs['argv'][5:]
            or source['cwd'] != obs['cwd'] or source['pins'] != obs['pins']
            or obs['environment'].get('NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE') != '1'):
        raise ValueError('Different owner, physical inputs or uninstrumented run')
    env1, env2 = (j['environment'].copy() for j in (source, obs))
    env2.pop('NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE')
    def baseline(env, j):
        tmp = str(Path(j['argv'][4]).parent) + '/'
        return {k:v.replace(tmp,'<capture>/') for k,v in env.items()}
    if baseline(env1, source) != baseline(env2, obs):
        raise ValueError('Physical run options changed')
    a,b=(json.loads((p/'terminal.receipt.json').read_text()) for p in (reference, observed))
    if a['accepted_step'] != b['accepted_step'] or a['physical_endpoint'] != 'accepted' or b['physical_endpoint'] != 'accepted':
        raise ValueError('Unmatched terminal physical steps')
    log=(observed/'stdout.log').read_text().splitlines()
    equilibrium=next(l for l in log if l.startswith('persistent_initial_force_reference='))
    gravity=next(r['gravity_target_n'] for r in json.loads(equilibrium.split('=',1)[1])['rows'] if r['dof']==2)
    tissue_line=next(l for l in log if l.startswith('resting_tissue_shared_runtime=bound '))
    tissue_mass=float(re.search(r' FEM_mass_kg=([\d.eE+-]+)', tissue_line)[1])
    weight=float(source['argv'][source['argv'].index('--resting-reference-mass-kg')+1])*9.81
    rows=[]
    for line in log:
        if not line.startswith('accepted_fem_body_load_capture=accepted '):
            continue
        step=int(re.search(r' source_step=(\d+)',line)[1])
        force=json.loads(re.search(r'root_generalized_force_xyz_n=(\[[^\]]+\])',line)[1])
        if len(force)!=3 or not np.isfinite(force).all() or (rows and step<=rows[-1]['source_step']):
            raise ValueError('Invalid generalized force observation')
        rows.append(dict(source_step=step, root_generalized_force_xyz_N=force))
    if not rows or rows[-1]['source_step']!=a['accepted_step']-1:
        raise ValueError('Terminal skeleton load was not captured')
    terminal_bed=next(l for l in reversed(log) if l.startswith('resting_fem_skin_bed_accepted_root_reaction '))
    reaction=float(re.search(r'mean_normal_force_N=([\d.eE+-]+)',terminal_bed)[1])
    parity=dict(physical_nodes=sha(reference/'physical-accepted.node')==sha(observed/'physical-accepted.node'),
                body_poses=a['accepted_body_poses']==b['accepted_body_poses'],
                coupled_trace=sha(reference/'native-traces/resting-coupled.csv')==sha(observed/'native-traces/resting-coupled.csv'),
                native_pack=a['pack_file_sha256']==b['pack_file_sha256'])
    return dict(accepted_steps=a['accepted_step'],root_observations=rows,root_vertical_gravity_target_N=gravity,
                total_weight_N=weight, tissue_weight_N=tissue_mass*9.81,
                weight_less_tissue_minus_root_gravity_N=weight-tissue_mass*9.81-gravity,
                final_bed_reaction_N=reaction,final_root_fem_minus_gravity_N=rows[-1]['root_generalized_force_xyz_N'][2]-gravity,
                parity=parity,launch_sha256=sha(observed/'launch.json'),matched_launch_sha256=sha(reference/'launch.json'),
                binary_sha256=source['pins'][source['argv'][0]],
                scope='Accepted FEM body contact and anchor loads projected onto the floating-root translation coordinates. Bed force acts on the skin. These forces can differ in a moving body. The archived initial gravity target is a balance reference, not a new runtime force.')


if __name__=='__main__':
    print(json.dumps(measure(*(Path(p).resolve() for p in sys.argv[1:])),indent=2,allow_nan=False))
