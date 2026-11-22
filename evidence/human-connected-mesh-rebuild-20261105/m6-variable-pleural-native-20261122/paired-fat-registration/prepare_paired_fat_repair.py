"""Emit the same local shape correction into the reference and loaded physical coordinates."""
import json
import sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from emit_constrained_material import write_coordinates
from torso_cell_geometry import geom

def prepare(native,geometry,trial,parent,out):
    r=json.loads(trial.read_text())
    x,y,t,tag,pins=inputs(native)
    z=native/'physical-accepted.node'
    if pins!=r['source'] or sha(z)!=r['accepted_sha256']:raise ValueError('Unmatched physical motion')
    z=np.loadtxt(z,skiprows=1,usecols=(1,2,3))
    previous=np.load(parent/'proposal.npz');cavity=np.load(geometry);unchanged=cavity['retained_input_cell_rows']
    before=[x.copy(),y.copy(),z.copy()]
    moved=[];edits=[];checked=[]
    for w in r['results']:
        choices=[o for o in w['trials'] if o['status'] in ('eligible geometry','passed') and o['reference_shift']]
        choice=max(choices,key=lambda o:(o['target_shapes'][-1],-o['peak_m']))
        ids=np.array(w['nodes']);move=np.array(choice['shift_m'])
        ids,move=ids[move!=0],move[move!=0]
        cells=np.flatnonzero(np.isin(t,ids).any(1))
        if (not len(ids) or np.intersect1d(ids,moved).size or not np.isin(cells,unchanged).all()
                or np.any(tag[cells]!=16) or np.isin(cavity['replacement_tetrahedra'],ids).any()):
            raise ValueError('Overlapping or unowned material edit')
        axis=choice['axis']
        for s in (x,y,z):s[ids,axis]=s[ids,axis].astype(float)+move
        checked.append(dict(source_row=w['source_row'],rows=ids.tolist(),axis=axis,movement_m=move.tolist(),force_checks=choice['mechanical_checks']))
        moved.extend(ids.tolist());edits.extend(move.tolist())
    rows=np.array(sorted(moved));cells=np.flatnonzero(np.any(np.isin(t,rows),axis=1))
    quality=[]
    for s in (x,y,z):
        v,q=geom(s.astype(float),t[cells]);summary=dict(minimum_volume_m3=v.min(),minimum_shape=q.min(),quality_failures=int(np.count_nonzero((v<=1e-18)|(q<1e-5))))
        if summary['quality_failures']:raise ValueError(summary)
        quality.append(summary)
    out.mkdir(exist_ok=False)
    e=json.loads((native/'launch.json').read_text())['environment']
    for points,key,name in ((x,'NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE','reference.node'),(y,'NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE','current.node')):
        write_coordinates(Path(e[key]),out/name,rows,points[rows])
    patchcells=np.union1d(previous['cell'],cells)
    nodes=np.union1d(previous['nodes'],np.unique(t[patchcells]))
    np.savez_compressed(out/'proposal.npz',nodes=nodes,cell=patchcells,x=x[nodes],t=np.searchsorted(nodes,t[patchcells]).astype('i4'),role=tag[patchcells])
    result=dict(native_source=pins,parent_proposal=sha(parent/'proposal.npz'),registration_fit=sha(trial),reference=sha(out/'reference.node'),
                current=sha(out/'current.node'),proposal=sha(out/'proposal.npz'),regions=checked,quality=quality,
                changed_mass_pending_native_role_check=True)
    (out/'material-identity.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':prepare(*(Path(a).resolve() for a in sys.argv[1:]))
