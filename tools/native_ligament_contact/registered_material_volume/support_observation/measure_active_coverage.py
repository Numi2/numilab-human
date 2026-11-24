"""Measure which positive terminal bed-history rows are inside the authored material footprint."""
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
from bind_gluteal_receivers import digest, require
from reconstruct_gluteal_fat import bary


def run(prior, fitted, actual, volume, output):
    p=actual/'physical-posterior-bed-history.csv'
    with p.open() as f:
        rows=list(csv.DictReader(f))
    stdout=(actual/'stdout.log').read_text()
    record=next(l for l in stdout.splitlines() if l.startswith('resting_fem_skin_bed_terminal_history'))
    value=lambda k:float(re.search(r'\b'+k+r'=([^ ]+)',record)[1])
    ids=np.array([int(r['registered_skin_node']) for r in rows])
    impulse=np.array([float(r['terminal_microstep_normal_impulse_Ns']) for r in rows])
    outer=np.load(volume/'material-walls.npz')['O']
    require(len(rows)==int(value('loaded_rows')) and len(np.unique(ids))==len(ids) and np.all(impulse>0) and
            np.isin(ids,np.unique(outer)).all() and abs(impulse.sum()-value('normal_impulse_sum_Ns')) < 1e-15,
            'Accepted history does not reproduce the native terminal reduction')
    r=np.load(fitted/'registered-columns.npz')
    geom=actual/'physical-accepted.node'
    require(digest(geom)==json.loads((fitted/'registration.json').read_bytes())['physical_nodes'],
            'Changed physical endpoint for the registered region')
    xyz=np.loadtxt(geom,skiprows=1,usecols=(1,2,3))
    pos=(xyz[ids]-r['origin'])@r['frame']
    fit_faces=np.unique(r['outer'])
    foot_tris=((xyz[outer[fit_faces]]-r['origin'])@r['frame'])
    # Count face-boundary points, whose nodal basis may also extend outside the patch.
    ranges=np.zeros((len(ids),len(fit_faces)),bool)
    for i,t in enumerate(foot_tris):
        ranges[:,i]=bary(t,pos[:,:2]).min(1)>=-1e-9
    inside=ranges.any(1)
    nodes_inside=np.isin(ids,np.unique(outer[fit_faces]))
    ring_inside=np.array([np.isin(np.flatnonzero((outer==nid).any(1)),fit_faces).all() for nid in ids])
    require(inside[nodes_inside].all(),'Warped exterior projection')
    source=json.loads((prior/'registered-receivers.json').read_bytes())
    registered=np.isin(ids,[r['skin_node']for r in source['receivers']])
    report=dict(rows=len(rows),terminal_impulse_Ns=float(impulse.sum()),
        existing_registration=digest(fitted/'registration.json'),native_contact_rows=digest(p),
        counts={'nine_registered_starts':int(registered.sum()),'point_within_material_footprint':int(inside.sum()),
                'complete_exterior_basis_supported':int(ring_inside.sum())},
        terminal_impulse_fraction={'nine_registered_starts':float(impulse[registered].sum()/impulse.sum()),
                'point_within_material_footprint':float(impulse[inside].sum()/impulse.sum()),
                'complete_exterior_basis_supported':float(impulse[ring_inside].sum()/impulse.sum())},
        loaded_nodes=ids.tolist(),outside_node_ids=ids[~inside].tolist(),partial_basis_node_ids=ids[inside&~ring_inside].tolist(),
        scope='Positive terminal microstep history. Whole-root reaction is reported separately by the native owner.')
    output.mkdir(exist_ok=True);(output/'coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'fit.json').write_text(json.dumps(dict(contact_nodes=ids.tolist(),bed_normal=r['frame'][:,2].tolist()),indent=2)+'\n')
    print(report['counts'],report['terminal_impulse_fraction'])
if __name__=='__main__':run(*(Path(v) for v in sys.argv[1:]))
