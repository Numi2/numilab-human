"""Test bone components for complete exterior placement and assess attachment-to-cortex registration."""
from pathlib import Path
import json
import runpy
import struct
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from audit_route_clearance import TriangleDistance,Surface
from run_bed_work import digest
from verify_native_extensor_pair import loaded


def measure(root,reader):
    base=root/'material-source'
    arr=base/'corrected/outer-boundaries.npz'
    proof=json.loads((base/'corrected/connected-columns.json').read_text())
    envelopes=json.loads((base/'geometry/dense_loaded/physical-banks.json').read_text())
    if envelopes['bone_pairs']!=0 or digest(arr)!=proof['outer_boundaries']:raise ValueError('Enclosure requires a crossing-free bone bank')
    with np.load(arr) as z:skin=Surface(z['loaded_xyz'],z['triangles'])
    r,poses,raw=loaded(root/'pose-one',True,False)
    if r['base_pack_file_sha256']!=envelopes['pack_sha256']:raise ValueError('Different opening geometry')
    sys.path.insert(0,str(reader.parent/'src'));api=runpy.run_path(str(reader));lib=api['clearance']
    h=lib._PACK_HEADER;d=lib._PACK_DIRECTORY
    sections={k[0]:k for i in range(h.unpack_from(raw)[2]) for k in [d.unpack_from(raw,h.size+d.size*i)]}
    counts={};owners={};components=[]
    for (semantic,sid),primitive,v,f in api['primitives'](raw,poses):
        if semantic!=51004:continue
        row=struct.unpack_from('<8I',raw,sections[4][2]+64*primitive)
        inst=struct.unpack_from('<8f12I',raw,sections[5][2]+80*row[3])
        if row[-1]!=sid or row[4]!=51004 or inst[10]!=3 or inst[9] not in poses:raise ValueError('Wrong bone pose')
        bank=inst[9]
        vertices,ids=np.unique(v,axis=0,return_inverse=True);g=ids.ravel()[f]
        edges=np.concatenate((g[:,[0,1]],g[:,[1,2]],g[:,[2,0]]))
        graph=coo_matrix((np.ones(len(edges),dtype=np.uint8),(edges[:,0],edges[:,1])),shape=(len(vertices),len(vertices)))
        n,labels=connected_components(graph,directed=False)
        for i in range(n):
            points=vertices[labels==i]
            if not len(points):raise ValueError('Empty cortical component')
            center=points.mean(0);center_vertex=points[np.argmin(np.linalg.norm(points-center,axis=1))]
            report=skin.query(center_vertex)
            components.append(dict(bone=sid,body=bank,component=i,unique_vertices=len(points),representative_m=center_vertex.tolist(),**report))
            counts[report['side']]=counts.get(report['side'],0)+1
        # The cortical surface may contain zero-area rendering triangles. Its closed triangle
        # union is unchanged by dropping those lying wholly on edges of its valid faces.
        t=v[f].astype(float);usable=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)>0
        if not usable.all() and not set(g[~usable].ravel()).issubset(set(g[usable].ravel())):raise ValueError('Isolated degenerate bone vertex')
        owners.setdefault(bank,[]).append((sid,TriangleDistance(v,f[usable])))
    route=json.loads((base/'route-clearance.json').read_text())
    native=json.loads((base/'native-route-clearance.json').read_text())
    check=[]
    for ref in route['route_roles']:
        k=str(ref['site']);s=route['sites'][k];pair=native['sites'][k]
        if ref['kind']=='intermediate' or pair['native_side']!='outside' or pair['dense_side']!='outside':continue
        matches=owners.get(s['body'],[])
        nearest=min(((obj.nearest(np.asarray(s['world_m'])),sid) for sid,obj in matches),default=None)
        check.append(dict(**ref,body=s['body'],native_outside_m=pair['native_distance_m'],dense_outside_m=pair['dense_distance_m'],
                          nearest_owned_bone_id=nearest[1] if nearest else None,cortical_distance_m=nearest[0] if nearest else None))
    return dict(pack_sha256=envelopes['pack_sha256'],material_boundary=digest(arr),components=components,component_counts=counts,exposed_endpoints=check,
                scope='One vertex per exactly seam-identified connected bone component against dense skin with zero measured cortical surface crossings. Nearest distances for exposed endpoints use positive-area triangles on the named owning body; they do not identify an anatomical attachment location.')
if __name__=='__main__':
    p=Path(__file__).resolve().parent
    (p/'material-source/bone-enclosure.json').write_text(json.dumps(measure(p,Path(sys.argv[1]).resolve()),indent=2,allow_nan=False)+'\n')
