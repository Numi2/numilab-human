"""Check the emitted initial bed wrench using the native law and installed mass rows."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from fit_native_bed_barrier_load import force
from fit_dorsal_material import inputs
from material_context import sha

def verify(trial,proposal,flow,fit):
    x,base,t,tags,pins=inputs(trial)
    q=json.loads(fit.read_text());b=q['registered_bed']
    launch=json.loads((trial/'launch.json').read_text());env=launch['environment']
    if pins!=q['source'] or sha(proposal)!=q['proposal_sha256'] or sha(flow)!=q['mass_sha256']:
        raise ValueError('Unmatched material registration')
    with np.load(flow) as f:nm=f['nodal_mass_kg'].copy();mass=f['element_mass_kg']
    with np.load(proposal) as f:
        removed=np.setdiff1d(np.arange(len(t)),f['retained_input_cell_rows'])
        for corner in range(4):
            np.add.at(nm,t[removed,corner],-mass[removed]/4)
            np.add.at(nm,f['replacement_tetrahedra'][:,corner],f['replacement_mass_kg']/4)
    mp=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if sha(mp)!=q['map_sha256']:raise ValueError('Mismatched outer material')
    with mp.open() as s:outer=np.array([int(r['joint_node_id'])-1 for r in csv.DictReader(s) if r['registered_surface_layer']=='O'])
    if len(np.unique(outer))!=len(outer):raise ValueError('Duplicate outer IDs')
    new=base.copy();new[q['node_ids']]=np.array(q['emitted_coordinates'],dtype='f4')
    if not np.array_equal(new[q['node_ids']],q['emitted_coordinates']):raise ValueError('Non-Float32 compiler input')
    dt=float(launch['argv'][launch['argv'].index('--muscle-step-seconds')+1]);width=float.fromhex('0x1.4f8b580000000p-17')
    if b['width_m']!=width or b['dt_s']!=dt:raise ValueError('Changed native contact law')
    def support(p):
        v=p[outer].astype(float);live=v[:,2]<width
        if np.min(v[:,2])<=1e-4*width:raise ValueError('Invalid stationary contact')
        f=force(v[live,2],nm[outer[live]],dt,width)
        return dict(minimum_gap_m=float(v[:,2].min()),total_N=float(f.sum()),xy_first_moment_Nm=(v[live,:2].T@f).tolist(),active_nodes=outer[live].tolist())
    old=support(base);after=support(new)
    drift=after['total_N']-old['total_N'];moment=np.array(after['xy_first_moment_Nm'])-old['xy_first_moment_Nm']
    if old['minimum_gap_m']!=after['minimum_gap_m']:
        raise ValueError(f"Compiled first-contact gap changed from {old['minimum_gap_m']} to {after['minimum_gap_m']}")
    if abs(drift)>.002 or np.max(abs(moment))>.001:raise ValueError('Physical support wrench changed')
    return dict(minimum_gap_m=after['minimum_gap_m'],outer_nodes=len(outer),old_active_nodes=len(old['active_nodes']),new_active_nodes=len(after['active_nodes']),
        before_N=old['total_N'],after_N=after['total_N'],force_drift_N=drift,moment_drift_Nm=moment.tolist(),fit_sha256=sha(fit),
        scope='All registered outer skin points against the existing stationary bed at initialization.')
if __name__=='__main__':print(json.dumps(verify(*(Path(p).resolve() for p in sys.argv[1:])),indent=2))
