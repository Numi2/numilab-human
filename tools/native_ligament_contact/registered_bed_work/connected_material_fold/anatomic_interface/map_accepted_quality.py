"""Locate native motion shape failures in the registered source material."""
import json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from supported_material_admission import geom
from condition_motion_witness import altitude


def run(work):
    stage = work / 'witness-material'
    volume = stage / 'volume'
    motion = json.loads((stage/'accepted-motion.json').read_bytes())
    capture = work/'accepted-material-07/physical-accepted.node'
    if digest(capture) != motion['file']:
        raise ValueError('Wrong accepted physical coordinates')
    load = lambda f: np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    x,y,z = (load(f) for f in (volume/'mesh.node',volume/'current.node',capture))
    raw = np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    labels = np.loadtxt(volume/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')
    if not np.array_equal(labels[:,0],np.arange(len(raw))) or not np.array_equal(labels[:,1],raw[:,4]):
        raise ValueError('Invalid cell provenance')
    rows = []
    for a in range(0,len(raw),32768):
        local = raw[a:a+32768,:4]-1
        _,shape = geom(z,local)
        bad = np.flatnonzero(shape<1e-5)
        if not len(bad):
            continue
        t = local[bad]
        _,restshape = geom(y,t)
        h = np.minimum(altitude(x,t),altitude(y,t))*1e6
        for i,(r,start,alt) in enumerate(zip(bad,restshape,h)):
            k=a+int(r)
            rows.append(dict(cell=k,tissue='skin' if raw[k,4]==15 else 'fat',nodes=local[r].tolist(),donor=int(labels[k,2]),region=int(labels[k,3]),source_face=int(labels[k,4]),initial_shape=float(start),accepted_shape=float(shape[r]),minimum_initial_altitude_um=float(alt)))
    for tissue,checks in motion['roles'].items():
        if checks['below_shape'] != sum(row['tissue']==tissue for row in rows):
            raise ValueError('Independent failure count differs')
    result=dict(geometry=digest(capture),elements=digest(volume/'mesh.ele'),source_labels=digest(volume/'current-child.labels.csv'),threshold=1e-5,submicron_at_start=sum(r['minimum_initial_altitude_um']<1 for r in rows),regions=sorted(set(r['region'] for r in rows)),witnesses=rows)
    (stage/'motion-quality.json').write_text(json.dumps(result,indent=2)+'\n')
    print(len(rows),result['submicron_at_start'],result['regions'])

if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
