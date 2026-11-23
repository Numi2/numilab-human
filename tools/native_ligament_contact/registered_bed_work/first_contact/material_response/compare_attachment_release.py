"""Compare tissue reaction, strain, support and the accepted endpoint at a matched root interval."""
import json
from pathlib import Path
import re
import sys
import numpy as np
from trial_registered_release import sha


def capture(path):
    metrics=json.loads((path/'material-motion.json').read_bytes())
    launch=json.loads((path/'launch.json').read_bytes())
    lines=(path/'stdout.log').read_text().splitlines()
    rows=[json.loads(re.search(r'root_generalized_force_xyz_n=(\[[^\]]+\])',l)[1])
          for l in lines if l.startswith('accepted_fem_body_load_capture=accepted ')]
    if len(rows)!=metrics['accepted_steps'] or not metrics['variable_pleural'] or metrics['root_assistance']:
        raise ValueError('Coupled release evidence incomplete')
    if metrics['launch']!=sha(path/'launch.json') or metrics['physical_nodes']!=sha(path/'physical-accepted.node'):
        raise ValueError('Stale material or launch identity')
    text=next((l for l in lines if l.startswith('registered_attachment_release=compiled ')),None)
    if bool(text)!=bool(launch.get('registered_release_enabled',False)):
        raise ValueError('Incorrect release initialization')
    coordinates=np.loadtxt(path/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    step=metrics['accepted_seconds']/len(rows)
    return dict(**metrics,root_forces_N=rows,first_root_norm_N=float(np.linalg.norm(rows[0])),
                average_root_force_N=np.mean(rows,axis=0).tolist(),root_load_times_dt_Ns=(step*np.sum(rows,axis=0)).tolist(),
                registration=text),coordinates


def compare(paths):
    cases={}
    coords={}
    for name,path in zip(('original_full','aligned_full','original_half','aligned_half'),paths,strict=True):
        cases[name],coords[name]=capture(path)
    baseline=cases['original_full']
    for k,c in cases.items():
        if c['accepted_seconds']!=baseline['accepted_seconds'] or c['source_materials']!=baseline['source_materials'] or c['cells']!=baseline['cells']:
            raise ValueError('Unmatched endpoint or material')
        if c['accepted_steps']!=(2 if k.endswith('half') else 1):
            raise ValueError('Unexpected root subdivision')
    comparisons={}
    for key in ('original','aligned'):
        whole,half=cases[key+'_full'],cases[key+'_half']
        ds=np.linalg.norm(coords[key+'_full']-coords[key+'_half'],axis=1)
        comparisons[key]=dict(
            first_root_subdivision_delta_N=float(np.linalg.norm(np.subtract(whole['root_forces_N'][0],half['root_forces_N'][0]))),
            mean_root_force_subdivision_delta_N=float(np.linalg.norm(np.subtract(whole['average_root_force_N'],half['average_root_force_N']))),
            matched_time_coordinate_rms_m=float(np.sqrt(np.mean(ds**2))),
            matched_time_coordinate_max_m=float(ds.max()))
    return dict(cases=cases,subdivision=comparisons,
        scope='Coupled first physical interval, release pose. Root generalized force output on the translation degrees of freedom is sampled once per root.')


if __name__=='__main__':
    args=[Path(p) for p in sys.argv[1:]]
    args[-1].write_text(json.dumps(compare(args[:-1]),indent=2,allow_nan=False)+'\n')
