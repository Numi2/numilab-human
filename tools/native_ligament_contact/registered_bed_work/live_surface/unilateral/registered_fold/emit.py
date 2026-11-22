"""Emit the same source-owned repair to the native reference, loaded mesh and mass ledger."""
import json
from pathlib import Path
import sys
import numpy as np
from emit_constrained_material import write_coordinates
from fit_dorsal_material import inputs
from material_context import sha


def emit(root, parent, out):
    trial=root.parent/'feasible-bed/two'
    report=json.loads((root/'recheck.json').read_text())
    choice=report['selected']
    if choice is None or dict(choice,reason='passed local checks') not in report['trials']:
        raise ValueError('Missing conservative patch')
    package=root/f"routed-axis-{choice['axis']}.npz"
    x,y,t,tag,pins=inputs(trial)
    if report['source']!=pins or sha(trial/'physical-accepted.node')!=json.loads((trial/'verified-endpoint.json').read_text())['physical_nodes_sha256']:
        raise ValueError('Unmatched integrated material')
    with np.load(package) as p:
        nodes=p['node_rows'];dx=p['displacement_m'];cells=p['cell_rows']
        m=p['mass_kg'];old_mass=p['original_mass_kg']
    if np.any(dx[:,choice['axis']]!=np.array(choice['move_m'])) or not np.array_equal(nodes,report['node_rows']):
        raise ValueError('Different displacement from fitted source')
    active=np.any(dx!=0,axis=1);ids=nodes[active];delta=dx[active]
    env=json.loads((trial/'launch.json').read_text())['environment']
    key='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE'
    out.mkdir(exist_ok=False)
    sources=[(x,Path(env[key]),'reference.node'),(y,Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']),'current.node')]
    for coords,path,name in sources:
        value=(coords[ids].astype(float)+delta).astype('f4')
        if not np.array_equal(value.astype(float)-coords[ids].astype(float),delta):
            raise ValueError('Changed Float32 repair size')
        coords[ids]=value
        write_coordinates(path,out/name,ids,value)
    with np.load(parent/'proposal.npz') as prior:
        inherited=prior['cell']
    affected=np.union1d(inherited,np.flatnonzero(np.isin(t,ids).any(1)))
    nodes=np.unique(t[affected])
    np.savez_compressed(out/'proposal.npz',nodes=nodes,cell=affected,x=x[nodes],t=np.searchsorted(nodes,t[affected]).astype('i4'),role=tag[affected])
    affected_role=tag[cells]==15
    if np.any(m[~affected_role]!=old_mass[~affected_role]) or np.any(np.isin(t[tag!=15],ids)):
        raise ValueError('Repair changes adipose material outside the selected dermal route')
    cells,m,old_mass=cells[affected_role],m[affected_role],old_mass[affected_role]
    np.savez_compressed(out/'mass-patch.npz',cells=cells,element_mass_kg=m,old_mass_kg=old_mass)
    authority=dict(source=pins,reference=sha(out/'reference.node'),current=sha(out/'current.node'),
        proposal=sha(out/'proposal.npz'),mass_patch=sha(out/'mass-patch.npz'),rows=ids.tolist(),moves_m=delta.tolist(),
        prior_flow=sha(parent/'fine-mass-routed/current-density-closed-flow.npz'),
        cases=[dict(force_and_energy=[choice['force_errors'],choice['energy_errors']])],
        source_feature_sha256=report['feature_sha256'],fit_sha256=sha(root/'recheck.json'))
    (out/'material-identity.json').write_text(json.dumps(authority,indent=2)+'\n')


if __name__=='__main__':
    emit(*(Path(s).resolve() for s in sys.argv[1:]))
