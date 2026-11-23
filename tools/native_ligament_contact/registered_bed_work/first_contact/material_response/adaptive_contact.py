"""Build and certify a contact response on residual-selected tissue nodes.

Preparation uses the installed reference mass and constitutive tangent. Loads
are the 19 normal forces on registered outer skin. A larger independent patch
tests the estimated error for every linear combination of these loads.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy import sparse
from scipy.linalg import cholesky, eigh, solve_triangular
from registered_bed_reference import reference, read, sha, mass_ledger, native_plane
from material_extension import Factor
from local_contact_inverse import patch_stiffness, finite_difference

def response(x, t, role, props, mass, fixed, sites, active, dt):
    inside = np.zeros(len(x), dtype=bool)
    inside[active] = True
    rows = np.flatnonzero(inside[t].any(axis=1))
    ids, local = np.unique(t[rows], return_inverse=True)
    cells = local.reshape(-1, 4)
    K = patch_stiffness(x[ids], cells, props[role[rows] - 15])
    mass3 = np.repeat(mass[ids], 3)
    A = K + sparse.diags(mass3 / dt**2)
    free = np.flatnonzero(np.repeat(inside[ids], 3))
    ports = 3 * np.searchsorted(ids, sites) + 2
    rhs = np.zeros((len(mass3), len(sites)))
    rhs[ports, np.arange(len(sites))] = 1
    factor = Factor(A[free][:, free], Path('material_factor.dylib'))
    try:
        disp = factor.solve(rhs[free])
    finally:
        factor.close()
    residual = rhs - A[:, free] @ disp
    solve_error = np.linalg.norm(residual[free]) / np.linalg.norm(rhs)
    residual[np.repeat(fixed[ids], 3)] = 0
    C = disp[np.searchsorted(free, ports)]
    reciprocal_error = np.linalg.norm(C - C.T) / np.linalg.norm(C)
    C = (C + C.T) / 2
    # Normalize port directions by the energy of this Galerkin solution.
    L = cholesky(C, lower=True)
    W = solve_triangular(L, residual.T, lower=True).T * (dt / np.sqrt(mass3))[:, None]
    bound = W.T @ W
    upper = float(np.linalg.eigvalsh(bound)[-1])
    score = (W * W).sum(axis=1).reshape(-1, 3).sum(axis=1)
    if not np.isfinite(upper) or reciprocal_error > 1e-8 or solve_error > 1e-7:
        raise ArithmeticError((upper, reciprocal_error, solve_error))
    neighboring = ~inside[ids] & ~fixed[ids]
    candidates = np.flatnonzero(neighboring & (score > 0))
    order = candidates[np.lexsort((ids[candidates], -score[candidates]))]
    result = dict(active_nodes=len(active), incident_cells=len(rows),
                  upper_relative_energy_error=upper, linear_residual=float(solve_error),
                  reciprocity=float(reciprocal_error))
    return result, ids[order], score[order], C, disp.reshape(-1, 3, len(sites)), ids[neighboring]

def build(parent):
    x, t, role, outer, deep, props, pins = reference(parent)
    native = parent / 'native-terminal-16'
    cfg = read(native / 'launch.json')
    bed = next(Path(s) for s in cfg['argv'] if s.endswith('.nhcnt'))
    if sha(bed) != cfg['pins'][str(bed)]:
        raise ValueError('Bed identity mismatch')
    sites = outer[x[outer, 2] - native_plane(bed) <= .001]
    fixed = np.zeros(len(x), bool)
    fixed[deep] = True
    if fixed[sites].any() or len(sites) != 19:
        raise ValueError('Unexpected native interface')
    mass, total, mass_hash = mass_ledger(cfg['environment'], t, role, len(x))
    report = dict(reference=sha(native / 'physical-reference.node'), bed=sha(bed),
                  mass_ledger=mass_hash, mass_kg=total, materials=pins,
                  reference_nodes=len(x), reference_tets=len(t), deep_nodes=deep.tolist(),
                  source_nodes=sites.tolist(), cases=[])
    for dt in (1.5625e-5, .000125, .0005, .001, .002):
        active = np.sort(sites)
        case = dict(step_s=dt, trials=[], certified=False)
        for iteration in range(40):
            result, ids, score, C, disp, halo = response(x, t, role, props, mass, fixed, sites, active, dt)
            case['trials'].append(result)
            if result['upper_relative_energy_error'] <= .01:
                larger = np.union1d(active, halo)
                test, _, _, C2, _, _ = response(x, t, role, props, mass, fixed, sites, larger, dt)
                difference = eigh(C2-C, C, eigvals_only=True)
                if difference.min() < -1e-9 or difference.max() > result['upper_relative_energy_error'] + 1e-9:
                    raise ArithmeticError('Larger patch violates certified energy interval')
                case.update(certified=True, halo_test_nodes=len(larger),
                            measured_halo_relative_error=float(difference.max()),
                            halo_certified_bound=test['upper_relative_energy_error'])
                np.savez_compressed('adaptive-response-%g.npz' % dt, physical_nodes=active,
                                    ports=sites, displacement_m_per_N=disp, compliance_m_per_N=C)
                break
            if not len(ids) or len(active) > 16000:
                break
            # Target dominant exterior residuals across every normalized port direction.
            count = np.searchsorted(np.cumsum(score), .995 * score.sum()) + 1
            active = np.union1d(active, ids[:max(8, count)])
        report['cases'].append(case)
        print(dt,result['active_nodes'],result['upper_relative_energy_error'],case.get('measured_halo_relative_error'),flush=True)
        Path('adaptive-contact.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
if __name__=='__main__':
    build(Path(sys.argv[1]).resolve())
