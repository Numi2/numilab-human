"""Registered normal and shared tangential modes for paired tissue columns."""
import numpy as np
from scipy.sparse import coo_matrix

TANGENT_BOUND_M = .002


class ColumnMotion:
    def __init__(self, normals, normal_id, columns, tangent):
        self.normal_count = int(np.count_nonzero(normal_id >= 0))
        self.node_id = np.full((len(normals), 3), -1, dtype=np.intp)
        self.axes = np.zeros((len(normals), 3, 3))
        self.node_id[:, 0] = normal_id
        self.axes[:, 0] = normals
        self.count = self.normal_count
        self.nodes = np.asarray(columns)
        self.groups = normal_id[self.nodes]
        self.tangent_count = 0
        if tangent:
            unit_normal = normals[self.nodes[:, 0]]
            # Cross the normal with its least-aligned coordinate axis.
            local_axis = np.eye(3)[np.abs(unit_normal).argmin(axis=1)]
            first = np.cross(unit_normal, local_axis)
            first /= np.linalg.norm(first, axis=1)[:, None]
            second = np.cross(unit_normal, first)
            tangent_id = self.normal_count + np.arange(2 * len(columns)).reshape(-1, 2)
            self.axes[self.nodes, 1:] = np.stack((first, second), axis=1)[:, None, :, :]
            self.node_id[self.nodes, 1:] = tangent_id[:, None, :]
            self.groups = np.hstack((self.groups, tangent_id))
            self.tangent_count = tangent_id.size
            self.count += self.tangent_count

    def displacement(self, values):
        return np.einsum("nv,nvd->nd", values[self.node_id], self.axes)

    def recover(self, displacement):
        coordinates = np.einsum("nd,nvd->nv", displacement, self.axes)
        result = np.zeros(self.count)
        active_normal = self.node_id[:, 0] >= 0
        result[self.node_id[active_normal, 0]] = coordinates[active_normal, 0]
        if self.tangent_count:
            result[self.groups[:, 3:].ravel()] = coordinates[self.nodes[:, 0], 1:].ravel()
        np.testing.assert_allclose(displacement, self.displacement(result), atol=1.e-14, rtol=0.)
        return result

    def jacobian(self, gradients, cells):
        columns = self.node_id[cells]
        rows = np.broadcast_to(np.arange(len(cells))[:, None, None], columns.shape)
        values = np.einsum("nkd,nkvd->nkv", gradients, self.axes[cells])
        include = columns >= 0
        return coo_matrix((values[include], (rows[include], columns[include])),
                          shape=(len(cells), self.count)).tocsr()
