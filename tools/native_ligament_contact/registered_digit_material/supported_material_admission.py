"""Authoring checks for loaded, reference-registered skin and fat. Dynamic motion keeps its own certificate."""
from pathlib import Path
import numpy as np
import re
from material_context import sha
from fit_posterior_equilibrium import materials
from torso_cell_geometry import geom


def installed_topology(elements, roles, env):
    path = Path(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    if sha(path) != env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:
        raise ValueError('Cavity topology hash mismatch')
    with np.load(path) as p:
        lineage = p['replacement_parent_cell_rows']
        offsets = p['replacement_parent_offsets']
        retained = p['retained_input_cell_rows']
        replacement = p['replacement_tetrahedra']
        if (len(offsets) != len(replacement) + 1 or offsets[0] != 0 or offsets[-1] != len(lineage)
                or np.any(np.diff(offsets) <= 0) or np.any(lineage < 0) or np.any(lineage >= len(roles))):
            raise ValueError('Incomplete replacement material ancestry')
        mat = roles[lineage[offsets[:-1]]]
        if not np.array_equal(roles[lineage], np.repeat(mat, np.diff(offsets))):
            raise ValueError('Ambiguous replacement material')
        return np.vstack((elements[retained], replacement)), np.r_[roles[retained], mat]


def material_state(reference, loaded, properties):
    # Row edge matrices give F transposed. Principal stretches and the maximum
    # absolute world stress component are invariant to this representation.
    X = reference[:, 1:] - reference[:, :1]
    Y = loaded[:, 1:] - loaded[:, :1]
    if not (np.isfinite(X).all() and np.isfinite(Y).all() and (np.linalg.det(X) > 0).all()):
        raise ValueError('Invalid material reference or current coordinates')
    F = np.linalg.solve(X, Y)
    J = np.linalg.det(F)
    if not np.isfinite(J).all() or np.any(J <= 0):
        raise ValueError('Nonpositive or nonfinite relative volume')
    stretches = np.linalg.svd(F, compute_uv=False)
    shear, bulk = properties.T
    stress = shear[:, None, None] * F + (bulk * np.log(J) - shear)[:, None, None] * np.linalg.inv(F).transpose(0, 2, 1)
    return J, stretches, np.abs(stress).max(axis=(1, 2))


def stress_limits(identities):
    result = []
    for path, expected in identities.items():
        if sha(Path(path)) != expected:
            raise ValueError('Material program changed')
        text = re.sub(r'//[^\n]*', '', Path(path).read_text())
        bounds = re.findall(r'\bmaximum_stress\s*=\s*([\d.eE+-]+)\s*;', text)
        if len(bounds) != 1 or not (0 < float(bounds[0]) < np.inf):
            raise ValueError('Missing or invalid maximum stress')
        result.append(float(bounds[0]))
    if len(result) != 2:
        raise ValueError('Expected dermis and fat material limits')
    return np.array(result)


def certify(reference, loaded, cells, roles, properties, stress_cap=(5.e6, 5.e6)):
    if reference.shape != loaded.shape or not len(cells) or len(cells) != len(roles) or not np.isin(roles, [15, 16]).all():
        raise ValueError('Inconsistent loaded material domain')
    out = dict(cells=len(cells), invalid_geometry=0, outside_stretch_envelope=0, outside_stress_cap=0,
               minimum_stretch=float('inf'), maximum_stretch=0., peak_stress_Pa=0., peak_stress_row=None)
    loaded = loaded.astype(float, copy=False)
    limits = np.asarray(stress_cap)
    for start in range(0, len(cells), 32768):
        subset = slice(start, start + 32768)
        tet = cells[subset]
        X = reference[tet].astype(float)
        Y = loaded[tet].astype(float)
        J, stretch, stress = material_state(X, Y, properties[roles[subset] - 15])
        # geom uses the same scaled determinant as the existing element-shape audit.
        volume, quality = geom(loaded, tet)
        out['invalid_geometry'] += int(np.sum((volume <= 1.e-18) | (quality < 1.e-5) | (J < .4) | (J > 2.5)))
        out['outside_stretch_envelope'] += int(np.sum((stretch[:, -1] < 2/3) | (stretch[:, 0] > 1.5)))
        out['outside_stress_cap'] += int(np.sum(stress > limits[roles[subset] - 15]))
        out['minimum_stretch'] = min(out['minimum_stretch'], float(np.min(stretch[:, -1])))
        out['maximum_stretch'] = max(out['maximum_stretch'], float(np.max(stretch[:, 0])))
        if out['peak_stress_row'] is None or stress.max() > out['peak_stress_Pa']:
            index = int(np.argmax(stress))
            out['peak_stress_Pa'] = float(stress[index])
            out['peak_stress_row'] = start + index
    out['passed'] = not any(out[k] for k in ('invalid_geometry', 'outside_stretch_envelope', 'outside_stress_cap'))
    return out


def check_installed(reference, loaded, source_cells, source_roles, env, owner):
    cells, roles = installed_topology(source_cells, source_roles, env)
    properties, identities = materials(owner)
    result = certify(reference, loaded.astype(float), cells, roles, properties, stress_limits(identities))
    result['materials'] = identities
    if not result['passed']:
        raise ValueError(f'Loaded skin/fat registration failed material admission: {result}')
    return result
