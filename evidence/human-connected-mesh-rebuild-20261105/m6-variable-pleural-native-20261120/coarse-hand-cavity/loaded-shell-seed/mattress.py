"""Series elimination of passive mattress indentation and IPC clearance."""

import numpy as np

from predict import ip_contact


def projected_area(world, cells, outer):
    """Lump downward projected outer-face area on its incident skin nodes."""
    result = np.zeros(outer)
    for opposite in range(4):
        face = cells[:, [i for i in range(4) if i != opposite]]
        selected = np.all(face < outer, axis=1)
        vertices = world[face[selected]]
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        interior = world[cells[selected, opposite]] - vertices[:, 0]
        inward = np.einsum("ni,ni->n", interior, normals) > 0.
        z = np.where(inward, -normals[:, 2], normals[:, 2])
        area = np.maximum(0., -.5 * z)
        np.add.at(result, face[selected].ravel(), np.repeat(area / 3., 3))
    return result


class Mattress:
    def __init__(self, area, modulus_pa, thickness, nu=.30):
        # Constrained small-strain modulus for a fixed-base spring foundation.
        modulus = modulus_pa * (1. - nu) / ((1. + nu) * (1. - 2. * nu))
        self.k = area * modulus / thickness
        self.area = area
        self.thickness = thickness
        self.modulus_pa = modulus_pa

    def clearance_for_load(self, nominal, width, mass, dt):
        active = (nominal < width) & (self.k > 0.)
        gap = nominal.copy()
        k, m = self.k[active], mass[active]
        low = np.maximum(nominal[active], width * 1.001e-4)
        high = np.full_like(low, width)
        for _ in range(32):
            half = .5 * (low + high)
            _, force, _ = ip_contact(half, width, m, dt)
            displace = half - force / k
            low = np.where(displace < nominal[active], half, low)
            high = np.where(displace >= nominal[active], half, high)
        gap[active] = .5 * (low + high)
        return gap, active

    def __call__(self, nominal, width, mass, dt):
        clearance, active = self.clearance_for_load(nominal, width, mass, dt)
        energy, load, tangent = (np.zeros_like(nominal) for _ in range(3))
        energy[active], load[active], tangent[active] = ip_contact(clearance[active], width, mass[active], dt)
        energy[active] += .5 * load[active]**2 / self.k[active]
        tangent[active] /= 1. + tangent[active] / self.k[active]
        return energy, load, tangent
