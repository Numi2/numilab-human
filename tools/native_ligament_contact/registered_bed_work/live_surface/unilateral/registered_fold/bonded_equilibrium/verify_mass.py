"""Verify the emitted pose using the original ledger's own groups and masses."""
from pathlib import Path
import json
import numpy as np
import sys
from fit_dorsal_material import inputs
from material_context import sha

def verify(root):
    s=root/'bonded-equilibrium';case=root/'fold-repair/accepted-2';m=root/'fold-repair/material'
    before=json.loads((s/'preparation.json').read_text())
    x,y,src,role,pins=inputs(case)
    y=y.astype(float)
    current=s/'current.node';ledger=m/'fine-mass-routed/current-density-closed-flow.npz'
    if sha(current)!=before['loaded_pose'] or sha(ledger)!=json.loads((ledger.parent/'mass-transfer.json').read_text())['flow_sha256']:raise ValueError('Unmatched material pose or ledger')
    end=np.loadtxt(current,skiprows=1,usecols=(1,2,3))
    delta=end-y
    changed=np.flatnonzero((delta!=0).any(axis=1));rows=np.flatnonzero(np.isin(src,changed).any(1))
    with np.load(ledger) as z:
        mass=z['element_mass_kg'][rows]; ids=z['group_for_cell'][rows];nodal=z['nodal_mass_kg'][changed]
    env=json.loads((case/'launch.json').read_text())['environment']
    with np.load(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']) as z:
        if not np.isin(rows,z['retained_input_cell_rows']).all() or np.any(np.isin(z['replacement_tetrahedra'],changed)):raise ValueError('A changed mass falls in a replacement region')
    g,inv=np.unique(ids,return_inverse=True);moment=np.zeros((len(g),3))
    np.add.at(moment,inv,mass[:,None]*delta[src[rows]].mean(1))
    a=mass[:,None]*y[src[rows]].mean(1)
    b=mass[:,None]*end[src[rows]].mean(1)
    independent=np.zeros_like(moment)
    np.add.at(independent,inv,b-a)
    nodal_error=float(abs(moment.sum(axis=0)-nodal@delta[changed]).max())
    mapped_error=float(abs(independent-moment).max())
    labels=m/'mesh/current-child.labels.csv'
    if sha(labels)!=json.loads(labels.with_suffix('.json').read_text())['labels_sha256']:raise ValueError('Changed donor ledger')
    region=np.loadtxt(labels,delimiter=',',skiprows=1,usecols=(1,2),dtype='i4')[rows]
    donors,invreg=np.unique(region,axis=0,return_inverse=True)
    regional=np.zeros((len(donors),3));np.add.at(regional,invreg,mass[:,None]*delta[src[rows]].mean(1))
    regional_change=float(abs(regional).max())
    if nodal_error>1e-14 or mapped_error>1e-14 or regional_change>8e-11:raise ValueError((nodal_error,mapped_error,regional_change))
    return dict(affected_source_groups=len(g),affected_donor_material_groups=len(donors),
                maximum_advected_source_group_first_moment_change_kg_m=float(abs(moment).max()),
                max_source_group_transport_accounting_error_kg_m=mapped_error,
                maximum_donor_material_first_moment_change_kg_m=regional_change,
                nodal_cell_moment_difference_kg_m=nodal_error,changed_source_masses=0,
                reference_sha256=pins['reference_sha256'],ledger_sha256=sha(ledger),
                loaded_pose=sha(current),source_pose=pins['installed_pose_sha256'])
if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1]).resolve()),indent=2))
