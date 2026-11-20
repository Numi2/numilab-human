"""Add local lateral tissue motion to the offline supported seed predictor."""
import numpy as np
from scipy.sparse import coo_matrix

from predict import SeedEnergy


class LateralSeedEnergy(SeedEnergy):
    def __init__(self, *args, width_m, **kwargs):
        super().__init__(*args, **kwargs)
        world = args[0]
        edges = world[self.cells[:, 1:]] - world[self.cells[:, :1]]
        self.base_f = np.einsum("tai,taj->tij", edges, self.g[:, 1:])
        np.testing.assert_allclose(self.base_f[:, 2], self.z0, atol=2.e-12, rtol=0.)
        lateral = np.flatnonzero(world[:self.nfree, 2] <= width_m)
        self.lateral = lateral
        count = self.nfree + 2 * len(lateral)
        self.id = np.full((self.anatomy_count, 3), count + 3, dtype=np.intp)
        self.id[:self.nfree, 2] = np.arange(self.nfree)
        self.id[lateral, :2] = np.arange(self.nfree, count).reshape(-1, 2)
        self.q = np.arange(count, count + 3)
        self.n = count + 3
        self.grav = np.r_[self.weights, np.zeros(2 * len(lateral)), self.body_weight, 0., 0.]
        self.dofs = self.id[self.cells].reshape(-1, 12)
        i, j = np.broadcast_arrays(self.dofs[:, :, None], self.dofs[:, None, :])
        interior = (i < count) & (j < count)
        self.full_free = interior
        self.full_indices = (i[interior], j[interior])
        dot = self.mu[:, None, None] * np.einsum("tad,tbd->tab", self.g, self.g)
        dot *= self.volume[:, None, None]
        self.shear = np.einsum("tab,ij->taibj", dot, np.eye(3)).reshape(-1, 12, 12)[interior]
        cols = self.q
        r = np.repeat(np.arange(self.outer), 3)
        c = np.tile(cols, self.outer)
        qi, qj = np.meshgrid(cols, cols, indexing="ij")
        self.contact_indices = (np.r_[np.arange(self.outer), r, c, qi.ravel()],
                                np.r_[np.arange(self.outer), c, r, qj.ravel()])

    def full_positions(self, v):
        displacement = np.r_[v, 0.][self.id]
        gap = self.base_gap + self.a @ v[self.q] + displacement[:self.outer, 2]
        f = self.base_f + np.einsum("tai,taj->tij", displacement[self.cells], self.g)
        return displacement, gap, f

    def positions(self, v):
        displacement, gap, f = self.full_positions(v)
        return displacement[:, 2], gap, np.linalg.det(f)

    def shape_rows(self, v, rows):
        f = self.full_positions(v)[2][rows]
        cofactor = np.stack((np.cross(f[:, 1], f[:, 2]), np.cross(f[:, 2], f[:, 0]),
                             np.cross(f[:, 0], f[:, 1])), axis=1)
        columns = self.dofs[rows]
        gradients = np.einsum("tij,taj->tai", cofactor, self.g[rows]).reshape(-1, 12)
        active = columns < self.q[0]
        local = np.broadcast_to(np.arange(len(rows))[:, None], columns.shape)
        return coo_matrix((gradients[active], (local[active], columns[active])),
                          shape=(len(rows), self.n)).tocsr()

    def state(self, v, gravity, hessian=False, penalty=None):
        _, gap, f = self.full_positions(v)
        jac = np.linalg.det(f)
        if jac.min() <= self.min_j or jac.max() >= self.max_j:
            return None
        if penalty is None and self.mattress is None and gap.min() <= 1.e-9:
            return None
        bed_energy, force, stiffness = self.contact(gap, penalty)
        log = np.log(jac)
        stretch = np.einsum("tij,tij->t", f - self.base_f, f + self.base_f)
        energy = .5 * (self.mu * (stretch - 2. * log) + self.bulk * log**2)
        objective = float(self.volume @ energy + bed_energy.sum() + gravity * (self.grav @ v))
        inverse_transpose = np.stack((np.cross(f[:, 1], f[:, 2]), np.cross(f[:, 2], f[:, 0]),
                                      np.cross(f[:, 0], f[:, 1])), axis=1) / jac[:, None, None]
        c = self.bulk * log - self.mu
        vgrad = np.einsum("tij,taj->tai", inverse_transpose, self.g)
        local = self.mu[:, None, None] * np.einsum("tij,taj->tai", f, self.g) + c[:, None, None] * vgrad
        forces = np.bincount(self.dofs.ravel(), weights=(self.volume[:, None, None] * local).ravel(),
                            minlength=self.n + 1)[:self.n]
        derivative = forces + gravity * self.grav
        derivative[self.q] -= self.a.T @ force
        derivative[:self.outer] -= force
        if not hessian:
            return objective, derivative, force, jac
        outer = np.einsum("tai,tbj->taibj", vgrad, vgrad)
        volume = (self.bulk[:, None, None, None, None] * outer -
                  c[:, None, None, None, None] * outer.swapaxes(2, 4))
        structured = (volume * self.volume[:, None, None, None, None]).reshape(-1, 12, 12)[self.full_free]
        external = self.a * stiffness[:, None]
        contact = np.r_[stiffness, external.ravel(), external.ravel(), (external.T @ self.a).ravel()]
        rows = np.r_[self.full_indices[0], self.contact_indices[0]]
        cols = np.r_[self.full_indices[1], self.contact_indices[1]]
        tangent = coo_matrix((np.r_[structured + self.shear, contact], (rows, cols)),
                            shape=(self.n, self.n)).tocsc()
        return objective, derivative, force, jac, tangent
