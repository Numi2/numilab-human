"""Check anatomical landmark enclosure and true nearest distance on an oblique tetrahedron."""
import numpy as np
from audit_route_clearance import Surface

vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
f=np.array([[0,2,1],[0,1,3],[3,2,0],[3,1,2]])
s=Surface(vertices,f)
for p,side,distance in [([.12,.2,.14],'inside',.12),([.48,.51,.54],'outside',.53/np.sqrt(3)),
                        ([-.04,-.03,.3],'outside',.05)]:
    r=s.query(np.array(p));assert r['side']==side and abs(r['distance_m']-distance)<1e-12
opening=Surface(vertices,f[1:]);r=opening.query(np.array([.12,.2,.14]))
assert len(opening.boundary)==3 and r['side']=='inside' and r['qualified_axis_count']<3
print('Verified face, edge and opening-aware landmark queries.')
