"""Conservative nearest-triangle broadphase for offline anatomical registration."""
import numpy as np
from scipy.spatial import cKDTree

class TriangleNearestIndex:
    """Index a fixed target; exact distance owner resolves every surviving face."""
    def __init__(self, triangles, distance_squared, initial_count=16):
        self.triangles=np.ascontiguousarray(triangles,dtype=np.float64)
        if self.triangles.ndim!=3 or self.triangles.shape[1:]!=(3,3) or len(self.triangles)==0:
            raise ValueError('expected nonempty triangle array')
        if not np.isfinite(self.triangles).all():
            raise ValueError('nonfinite target triangle')
        self.distance_squared=distance_squared
        self.last_stats={}
        # The measured small-target path costs less as a full owner scan.
        if len(self.triangles)<=1024:
            self.tree=None
            return
        self.centers=self.triangles.mean(axis=1)
        relative=self.triangles-self.centers[:,None,:]
        self.radii=np.sqrt(np.max(np.einsum('nki,nki->nk',relative,relative),axis=1))
        self.maximum_radius=float(self.radii.max())
        self.scale=max(1.,float(np.abs(self.triangles).max()))
        self.tree=cKDTree(self.centers)
        self.distance_squared=distance_squared
        self.initial_count=min(int(initial_count),len(self.triangles))
        if self.initial_count<1:raise ValueError('positive initial count required')
        self.last_stats={}
    def query(self, point):
        point=np.asarray(point,dtype=np.float64)
        if point.shape!=(3,) or not np.isfinite(point).all():
            raise ValueError('expected finite point')
        if self.tree is None:
            values=self.distance_squared(point,self.triangles)
            if not np.isfinite(values).all():raise ValueError('nonfinite exact distance')
            best=int(np.argmin(values))
            self.last_stats={'target_faces':len(self.triangles),'initial_faces':0,'exact_candidate_faces':len(self.triangles)}
            return best,float(values[best])
        initial=np.unique(np.atleast_1d(self.tree.query(point,k=self.initial_count,workers=1)[1]).astype(np.int64))
        d2=self.distance_squared(point,self.triangles[initial])
        if not np.isfinite(d2).all():raise ValueError('nonfinite exact distance')
        distance=float(np.sqrt(max(0.,float(np.min(d2)))))
        # A face can improve the current exact upper bound only if its
        # centroid is within distance + its enclosing vertex radius.
        # The small numeric pad protects the conservative bound at ties.
        pad=128*np.finfo(np.float64).eps*max(self.scale,float(np.abs(point).max()),distance)+1e-12
        ids=np.asarray(self.tree.query_ball_point(point,distance+self.maximum_radius+pad,workers=1,return_sorted=True),dtype=np.int64)
        lower=np.linalg.norm(self.centers[ids]-point,axis=1)-self.radii[ids]
        ids=ids[lower<=distance+pad]
        ids=np.union1d(ids,initial)
        values=self.distance_squared(point,self.triangles[ids])
        if not np.isfinite(values).all():raise ValueError('nonfinite exact distance')
        best=int(np.argmin(values))
        self.last_stats={'target_faces':len(self.triangles),'initial_faces':len(initial),'exact_candidate_faces':len(ids)}
        return int(ids[best]),float(values[best])
