"""Measure edge scale, altitude, density, donor coverage and anatomical embedding of each witness."""
import json,sys
from pathlib import Path
import numpy as np
from itertools import combinations
def run(volume,fit,audit,out):
    def points(p):return np.loadtxt(p,skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    rd=json.loads(audit.read_bytes()); x=points(volume/'mesh.node');y=points(fit/'current.node')
    mesh=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    cells=np.loadtxt(volume/'current-child.labels.csv',delimiter=',',dtype='i8',skiprows=1)
    if not np.array_equal(cells[:,0],np.arange(len(mesh))):raise ValueError('Unordered source identities')
    mass=np.fromfile(volume/'cell-mass.f64','<f8'); nodal=np.fromfile(volume/'node-mass.f64','<f8')
    for d in rd['cells']:
        i=d['local_id']; q=mesh[i]; r=x[q]
        vol=abs(np.linalg.det(r[1:]-r[0]))/6
        area=max(np.linalg.norm(np.cross(r[b]-r[a],r[c]-r[a]))/2 for a,b,c in combinations(range(4),3))
        length=np.linalg.norm(np.array([r[a]-r[b] for a,b in combinations(range(4),2)]),axis=1)
        donors=np.all(cells[:,1:]==cells[i,1:],axis=1)
        d.update(material=int(cells[i,1]),body=int(cells[i,2]),region=int(cells[i,3]),source_face=int(cells[i,4]),
            same_label_cells=int(donors.sum()),same_label_mass_kg=mass[donors].sum(),cell_mass_kg=mass[i],rest_volume_m3=vol,
            rest_density_kg_m3=mass[i]/vol,smallest_altitude_m=3*vol/area,longest_edge_m=length.max(),
            nodal_mass_kg=nodal[q].tolist(),center_m=y[q].mean(0).tolist())
    out.write_text(json.dumps(rd,indent=2)+'\n')
if __name__=='__main__':run(*(Path(a).resolve() for a in sys.argv[1:]))
