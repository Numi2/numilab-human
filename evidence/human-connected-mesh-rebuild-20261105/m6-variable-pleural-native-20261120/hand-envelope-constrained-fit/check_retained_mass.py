"""Check direct same-cell material mass retention on affected original elements."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from envelope_lp import prepare,tet_metrics,digest

if __name__=="__main__":
    launch,topo,bed,saved,summary=map(Path,sys.argv[2:])
    env=json.loads(launch.read_text())["environment"]
    x,tets,role,_=prepare(launch,topo,bed)
    with np.load(saved,allow_pickle=False) as a:
        p=x.copy()
        p[a["node_rows"]]=a["position_f32"]
    assert digest(saved)==json.loads(summary.read_text())["proposal_sha256"]
    with np.load(topo,allow_pickle=False) as a:
        orig=a["retained_input_cell_rows"]
    element=Path(env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS"])
    assert digest(element)==env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS_SHA256"]
    original_roles=np.loadtxt(element,dtype=np.int32,skiprows=1,usecols=(5,))
    changed=np.any(p!=x,axis=1)
    affected=np.any(changed[tets],axis=1)
    a=np.flatnonzero(affected[:len(orig)])
    before,_,_=tet_metrics(x[tets[a]].astype(float))
    after,_,_=tet_metrics(p[tets[a]].astype(float))
    assert (after>0).all()
    output=[]
    for material,name,limits in [(15,"SKIN",(900.,1100.)),(16,"FAT",(850.,1050.))]:
        path_key="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_"+name+"_CHILD_MASS_CSV"
        path=Path(env[path_key])
        assert digest(path)==env[path_key+"_SHA256"]
        data=np.loadtxt(path,delimiter=",",skiprows=1)
        original_role_rows=np.flatnonzero(original_roles==material)
        assert len(data)==len(original_role_rows) and np.array_equal(data[:,0],np.arange(len(data)))
        used=(role[a]==material)
        ids=np.searchsorted(original_role_rows,orig[a[used]])
        assert np.array_equal(original_role_rows[ids],orig[a[used]])
        old_volume,mass,density=data[ids,1:].T
        assert np.all(mass>0)
        source_density=mass/(before[used]/6)
        repaired_density=mass/(after[used]/6)
        old_lower,old_upper=(source_density<limits[0]-2e-7),(source_density>limits[1]+2e-7)
        lo,hi=(repaired_density<limits[0]-2e-7),(repaired_density>limits[1]+2e-7)
        output.append({"material":name,"source_mass_sha256":digest(path),
                       "affected_original_cells":len(ids),"affected_original_mass_kg":float(mass.sum()),
                       "source_density_range": [float(source_density.min()),float(source_density.max())],
                       "repaired_density_range":[float(repaired_density.min()),float(repaired_density.max())],
                       "old_below":int(old_lower.sum()),"old_above":int(old_upper.sum()),
                       "new_below":int(lo.sum()),"new_above":int(hi.sum()),
                       "newly_outside":int(((lo|hi)&~(old_lower|old_upper)).sum()),
                       "repaired_over_source_volume_ratio": [float((after[used]/before[used]).min()),
                                                            float((after[used]/before[used]).max())]})
    print(json.dumps({"scope":"A direct same-cell transfer screen for original retained ownership rows. Does not reconstruct accepted replacement-cell masses, donor routes or the complete mass transfer.",
                      "affected_replacement_cells_excluded":int(affected[len(orig):].sum()),
                      "source_scope":output},indent=2))
