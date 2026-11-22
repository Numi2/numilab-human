"""Check the installed mass, nodal transport, and conservative cavity transfer in a deformed pose."""
from pathlib import Path
import json
import sys
import numpy as np
from cavity_mass import cavity_delta
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import installed_topology

def read(p):
    return json.loads(p.read_text())

def verify(root,held=False):
    case=root/'bonded-equilibrium'/('held-volume' if held else 'free-interface')
    run=root/'fold-repair/accepted-2'
    material=root/'fold-repair/material'
    env=read(run/'launch.json')['environment']
    _,before,source,role,pins=inputs(run)
    before=before.astype(float)
    after=np.loadtxt(case/'current.node',skiprows=1,usecols=(1,2,3))
    prep=read(case/'preparation.json')
    if sha(case/'current.node')!=prep['loaded_pose'] or sha(Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']))!=prep['source_pose']:
        raise ValueError('Unmatched fitted coordinates')
    actual,tags=installed_topology(source,role,env)
    path=material/'fine-mass-routed/current-density-closed-flow.npz'
    labels=material/'mesh/current-child.labels.csv'
    if sha(path)!=read(path.parent/'mass-transfer.json')['flow_sha256'] or sha(labels)!=read(labels.with_suffix('.json'))['labels_sha256']:
        raise ValueError('Different carrier masses or anatomical ownership')
    donor=np.loadtxt(labels,delimiter=',',usecols=2,skiprows=1,dtype=int)
    with np.load(path) as l:
        old=l['element_mass_kg']; old_nodal=l['nodal_mass_kg']; groups=l['group_for_cell']
    with np.load(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']) as z:
        ret=z['retained_input_cell_rows']; nt=z['replacement_tetrahedra']; mass=z['replacement_mass_kg']
        offsets=z['replacement_parent_offsets']; parent=z['replacement_parent_cell_rows']
        newdonor=donor[parent[offsets[:-1]]]
        if not np.array_equal(np.repeat(newdonor,np.diff(offsets)),donor[parent]):
            raise ValueError('Ambiguous replacement donor')
        err=[]
        for i,j in zip(z['replacement_tet_offsets'][:-1],z['replacement_tet_offsets'][1:]):
            prior=np.unique(parent[offsets[i]:offsets[j]])
            err.append(abs(old[prior].sum()-mass[i:j].sum()))
        dm,nodes,budget_mass=cavity_delta(source,old,ret,nt,mass,len(before))
        m=np.r_[old[ret],mass];d=np.r_[donor[ret],newdonor]
    delta=after-before;changed=np.flatnonzero(np.any(delta!=0,axis=1))
    k=np.flatnonzero(np.isin(actual,changed).any(1))
    disp=delta[actual[k]].mean(axis=1)
    region,index=np.unique(np.c_[tags[k],d[k]],axis=0,return_inverse=True)
    mom=np.zeros((len(region),3));np.add.at(mom,index,m[k,None]*disp)
    transport=mom.sum(axis=0)
    nodal=(old_nodal+dm)[changed]@delta[changed]
    # Separately advect the immutable pre-remesh ledger by source group.
    orig=np.flatnonzero(np.isin(source,changed).any(1))
    g,ix=np.unique(groups[orig],return_inverse=True)
    orig_mom=np.zeros((len(g),3));np.add.at(orig_mom,ix,old[orig,None]*delta[source[orig]].mean(axis=1))
    centroid=after[actual[k]].mean(1)-before[actual[k]].mean(1)
    direct=m[k]@centroid
    origin=after[nodes[0]]
    cm=dm[nodes]@(after[nodes]-origin);radius=abs(after[nodes]-origin).max()
    absolute_limit=1e-10*budget_mass*radius
    if (max(err)>1e-14 or abs(nodal-transport).max()>1e-14 or abs(direct-transport).max()>1e-14
        or abs(orig_mom.sum(0)+dm[changed]@delta[changed]-transport).max()>1e-14
        or abs(cm).max()>absolute_limit or np.max(abs(mom))>8e-11 or min(m)<=0):
        raise ValueError(('Installed transfer or loaded cavity first moment changed',cm,absolute_limit))
    return dict(cells=len(actual),installed_stars=len(k),source_groups_touched=len(g),registered_donor_material_groups=len(region),
        max_cavity_mass_difference_kg=max(err),max_regional_moment_change_kg_m=float(np.max(abs(mom))),
        max_nodal_transport_error_kg_m=float(abs(nodal-transport).max()),max_centroid_transport_error_kg_m=float(abs(direct-transport).max()),
        loaded_cavity_moment_kg_m=cm.tolist(),native_limit_kg_m=absolute_limit,reference=pins['reference_sha256'],
        material_ledger=sha(path),cavity=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256'],loaded_pose=prep['loaded_pose'])
if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1]).resolve(),'--hold-parent-volume' in sys.argv[2:]),indent=2,allow_nan=False))
