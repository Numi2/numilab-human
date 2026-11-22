"""Match the precision trial to the retained three-step run and compare accepted physical nodes."""
import json
from pathlib import Path
import sys
import numpy as np
from material_context import sha
from measure_posterior_release import read, measure_release

def summarize(root, out):
    prefix=root/'incremental-prefix'
    trial=root/'incremental-three'
    control=root/'native-prefix'
    comparison=read(root/'incremental-comparison.json')
    if comparison['candidate']!=sha(trial/'launch.json'):
        raise ValueError('Different three-step trial')
    changed=measure_release(root,prefix,trial,root/'incremental-bed.json')
    baseline=read(root/'motion-comparison.json')['candidate']
    delta=[]
    for case,sample in ((control,baseline),(prefix,changed)):
        path=case/'physical-accepted.node'
        if sample['endpoint']['physical_nodes_sha256']!=sha(path):
            raise ValueError('Unmatched material endpoint')
        delta.append(np.loadtxt(path,skiprows=1,usecols=(1,2,3)))
    if baseline['endpoint']['simulated_s']!=changed['endpoint']['simulated_s']:
        raise ValueError('Different accepted physical time')
    diff=np.linalg.norm(delta[1]-delta[0],axis=1)
    summary=dict(three_step_comparison=sha(root/'incremental-comparison.json'),
                 control=sha(root/'motion-comparison.json'),kinematics=changed,
                 accepted_material_difference_m=dict(maximum=float(np.max(diff)),
                 rms=float(np.sqrt(np.mean(diff**2))),largest_node_row=int(np.argmax(diff))),
                 accepted_bed_difference_N=changed['terminal_bed_N']-baseline['terminal_bed_N'])
    out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':summarize(*(Path(v).resolve() for v in sys.argv[1:]))
