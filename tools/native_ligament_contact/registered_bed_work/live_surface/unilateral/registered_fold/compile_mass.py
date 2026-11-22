"""Fit the repaired patch to imported native volumes and the unchanged carrier supply."""
import json
from pathlib import Path
import sys
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from registered_density import density_bounds


def restore_sum(mass, low, high, target):
    result=np.clip(mass,low,high)
    for _ in range(6):
        diff=target-result.sum()
        if diff==0:break
        room=high-result if diff>0 else result-low
        i=int(np.argmax(room))
        if room[i]<abs(diff):
            if room.sum()<abs(diff):raise ValueError('Source group cannot supply the registered volume')
            result+=diff*room/room.sum()
        else:result[i]+=diff
    return result


def compile_mass(run,parent,case):
    def read(p):return json.loads(p.read_text())
    x,y,t,role,pins=inputs(run)
    info=read(case/'material-identity.json')
    ledger=parent/'fine-mass-routed'
    flow=ledger/'current-density-closed-flow.npz'
    prior=read(ledger/'mass-transfer.json')
    if info['source']!=pins or sha(flow)!=info['prior_flow'] or sha(flow)!=prior['flow_sha256'] or sha(case/'mass-patch.npz')!=info['mass_patch']:
        raise ValueError('Unmatched physical inputs')
    volumes=np.fromfile(case/'native.volumes','f8')
    if volumes.shape!=(len(t),) or np.any(volumes<=1e-18):raise ValueError('Unqualified imported geometry')
    data=dict(np.load(flow))
    old=data['element_mass_kg'];m=old.copy();g=data['group_for_cell']
    supply=np.bincount(data['group'],weights=data['mass_flow_kg'])
    old_total=np.bincount(g,weights=old)
    remap=np.load(case/'mass-patch.npz');rows=remap['cells']
    if not np.array_equal(old[rows],remap['old_mass_kg']) or not np.all(role[rows]==15):
        raise ValueError('Different dermal source rows')
    m[rows]=remap['element_mass_kg']
    if set(np.unique(role))!={15,16}:raise ValueError('Unexpected material in the registered skin/fat carrier')
    owners=run.parent/'owner'
    skin,fat=(density_bounds(owners,r) for r in (15,16))
    low=np.where(role==15,skin[0],fat[0]);high=np.where(role==15,skin[1],fat[1])
    for group in np.unique(g[rows]):
        local=rows[g[rows]==group]
        if np.array_equal(m[local],old[local]):continue
        scale=supply[group]/old_total[group]
        lo=np.maximum(low[local],low[local]/scale)*volumes[local]
        hi=np.minimum(high[local],high[local]/scale)*volumes[local]
        target=old[local].sum()
        budget=min(target-lo.sum(),hi.sum()-target)
        if budget<0:raise ValueError(f'Insufficient imported mass capacity in group {group}')
        # Leave margin on both density checks, bounded by the actual donor supply.
        guard=min(1e-7,.1*budget/volumes[local].sum())*volumes[local]
        m[local]=restore_sum(m[local],lo+guard,hi-guard,target)
    actual=np.bincount(g,weights=m)
    for rho in (m/volumes,m*(supply/actual)[g]/volumes):
        bad=np.flatnonzero(~np.isfinite(rho)|(rho<low)|(rho>high))
        if len(bad):raise ValueError(f'Native density fails at {len(bad)} cells, first {bad[:8].tolist()}')
    z=np.loadtxt(run/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    ids=np.array(info['rows']);shift=np.array(info['moves_m'])
    nodes=np.unique(t[rows]);cells=np.searchsorted(nodes,t[rows]);delta=m[rows]-old[rows]
    delta_by_node=np.zeros((len(nodes),3));delta_by_node[np.searchsorted(nodes,ids)]=shift
    mass_error=0.;moment_error=0.;stats=[]
    for group in np.unique(g[rows]):
        local=g[rows]==group
        mass_error=max(mass_error,abs(delta[local].sum()))
        c=cells[local];moments=[]
        for state in (x,y,z):
            centers=state[nodes].astype(float)[c].mean(1);origin=centers.mean(0)
            error=delta[local]@(centers-origin)+m[rows[local]]@delta_by_node[c].mean(1)
            moments.append(error.tolist());moment_error=max(moment_error,np.max(abs(error)))
        stats.append(dict(source_group=int(group),first_moment_change_kg_m=moments))
    if mass_error>1e-17 or moment_error>1e-14:raise ValueError(('Conservative mapping failed',mass_error,moment_error))
    nodal=data['nodal_mass_kg'].copy()
    for corner in range(4):np.add.at(nodal,t[rows,corner],delta/4)
    if np.any(nodal<=0):raise ValueError('Invalid nodal mass')
    data.update(element_mass_kg=m,nodal_mass_kg=nodal)
    inp=read(parent/'material/native-role-launch.json');check=read(parent/'material/native-role-verification.json')
    if not check['passed'] or inp['argv'][-2]!=sha(flow):raise ValueError('Unqualified parent carrier deliveries')
    count=int(inp['argv'][-1]);out=case/'fine-mass-routed';out.mkdir()
    np.savez_compressed(out/flow.name,**data)
    report=dict(parent_flow_sha256=sha(flow),flow_sha256=sha(out/flow.name),geometry_sha256=sha(case/'proposal.npz'),
        native_volume_sha256=sha(case/'native.volumes'),admitted_route_distance_m=prior['admitted_route_distance_m'],
        maximum_carrier_mass_change_kg=0.,archived_compensation_unchanged=True,changed_edges=0,density_outliers=0,
        group_max_abs_kg=mass_error,maximum_first_moment_change_kg_m=moment_error,regional_mass=stats,
        corrected_cell_count=int(np.sum(old!=m)),solver_output=f'inherited source-carrier allocation augmentations={count}',
        inherited_allocation_augmentation_count=count,used_new_typed_routes=0,allocation_provenance=sha(parent/'material/native-role-launch.json'))
    (out/'mass-transfer.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    compile_mass(*(Path(s).resolve() for s in sys.argv[1:]))
