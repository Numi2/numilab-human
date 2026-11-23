"""Bind native acceleration witnesses to the source joint map and compare physical releases."""
from pathlib import Path
import json,re,sys
import numpy as np
from material_context import sha
def run(base,loaded,out):
    def read(path):
        s=(path/'stdout.log').read_text()
        data,=[json.loads(l.split('=',1)[1]) for l in s.splitlines() if l.startswith('stand_terminal_state=')]
        return s,data
    log,second=read(loaded);_,first=read(base)
    if first['initial_q']!=second['initial_q'] or first['initial_v']!=second['initial_v'] or first['step_count']!=1 or second['step_count']!=1:raise ValueError('unmatched initial condition')
    a,b=[(np.array(s['v'])-s['initial_v'])/s['timestep_seconds'] for s in (first,second)]
    cfg=json.loads((base/'launch.json').read_text());rigid=Path(cfg['argv'][1])
    if sha(rigid)!=cfg['pins'][str(rigid)]:raise ValueError('changed rigid skeleton')
    prep=json.loads((rigid.parent/'projected-joint-envelope.json').read_text())
    if prep['output_rigid_sha256']!=sha(rigid):raise ValueError('unmatched source revision')
    audit=json.loads((rigid.parent.parent/'joint-coverage-audit.json').read_text());src=audit['source_inputs'];path=Path(src['source_manifest'])
    if sha(path)!=src['source_manifest_sha256']:raise ValueError('unmatched joint map')
    tree=json.loads(path.read_text())['core_tree'];mp={r['core_v_index']:r for r in tree['source_joint_map']}
    data=[]
    for line in log.splitlines():
        if not line.startswith('loaded_opening_coordinate='):continue
        tokens=dict(x.split('=',1) for x in line.split());v=int(tokens['loaded_opening_coordinate']);m=mp[v]
        if m['core_joint_index']!=int(tokens['joint']) or m['core_q_index']!=int(tokens['q']):raise ValueError('joint identity disagrees')
        data.append(dict(v=v,q=m['core_q_index'],joint=m['source_name'],static_acceleration=float(tokens['acceleration']),
                         static_force_residual_Nm=float(tokens['force']),bed_torque_Nm=float(tokens['bed']),
                         native_unloaded_recruitment_acceleration_rad_s2=a[v],native_loaded_recruitment_acceleration_rad_s2=b[v]))
    if len(data)!=8:raise ValueError('incomplete joint attribution')
    rows=dict(rigid_sha256=sha(rigid),source_manifest_sha256=sha(path),control=sha(base/'terminal.receipt.json'),trial=sha(loaded/'terminal.receipt.json'),
              dt_s=first['timestep_seconds'],control_max_root_translation_acceleration_m_s2=max(abs(a[:3])),
              loaded_max_root_translation_acceleration_m_s2=max(abs(b[:3])),control_max_joint_acceleration_rad_s2=max(abs(a[6:])),
              loaded_max_joint_acceleration_rad_s2=max(abs(b[6:])),worst_static_coordinates=data)
    out.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
