"""Exact rigid displacement is a null-strain construction across bonded material IDs."""
import numpy as np
from scipy.spatial.transform import Rotation
from extend_material_donor_motion import fit
from construct_registered_donor_field import smooth_map

# A three-cell chain leaves a Dirichlet support on both sides of the central seam.
points = np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[0.,0.,.01],
                   [.01,.01,.01],[.011,.006,.002],[.013,.012,.004]])
cells = np.array([[0,1,2,3],[1,4,2,3],[4,1,5,3],[4,6,3,5]])
for row in cells:
    if np.linalg.det(points[row[1:]]-points[row[:1]])<0:
        row[1], row[2] = row[2], row[1]
rotation=Rotation.from_rotvec([.05,-.06,.12]).as_matrix()
target=points@rotation.T + [.01,.02,-.03]
np.testing.assert_allclose(smooth_map(points,target,.006),target,atol=1e-12,rtol=0)
owners=np.array([0,0,0,0,0,1,1])
proposed,report=fit(points,target,owners,cells,np.stack([rotation,rotation]),0,np.zeros(len(points),bool),None)
np.testing.assert_allclose(proposed,target,atol=1e-12,rtol=0)
assert report['relative_linear_residual']<1.e-12
print('rigid patch identity across donor seams passed')
