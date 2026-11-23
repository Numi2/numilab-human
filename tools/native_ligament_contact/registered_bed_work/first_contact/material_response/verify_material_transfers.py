"""Independently replay solved carrier masses, geometric reach and parent/recipient conservation."""
import json
from fractions import Fraction
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import pinned, sha, tetgen_positions


def role_audit(tag, mat, packet, prior, p1, result, mesh, cells, x, m, source, labels, radius):
    role_rows = np.flatnonzero(cells[:, 4] == mat)
    parents = np.repeat(packet['global_source330_tet_rows'], np.diff(packet['parent_offsets']))
    bary = packet['carrier_barycentric']
    t, source_points, parent_mass = source
    points = np.einsum('ij,ijk->ik', bary, source_points[t[parents]])
    quota = parent_mass[parents] * packet['carrier_weights']
    if np.any(labels[parents, 0] != mat) or np.any(quota <= 0) or np.max(abs(bary.sum(1)-1)) > 2e-12:
        raise ValueError('Invalid source material or quadrature')
    q, g, w = (result[k] for k in ('carrier', 'role_cell', 'mass'))
    if (np.min(q) < 0 or np.max(q) >= len(quota) or np.min(g) < 0 or np.max(g) >= len(role_rows)
            or not np.isfinite(w).all() or np.any(w <= 0)):
        raise ValueError('Invalid carrier flow')
    key = g.astype('i8') * len(quota) + q
    if (np.diff(key) <= 0).any():
        raise ValueError('Unsorted or duplicate recipient/carrier')
    donor = np.bincount(q, weights=w, minlength=len(quota))
    total = np.bincount(g, weights=w, minlength=len(role_rows))
    if (total <= 0).any() or (donor <= 0).any():
        raise ValueError('Unfunded source or recipient')
    target = m[role_rows]
    tet = x[cells[role_rows, :4] - 1]
    center = tet.mean(axis=1)
    a, b, c = (tet[:, i] - tet[:, 0] for i in (1, 2, 3))
    det = (a[:,0]*(b[:,1]*c[:,2]-b[:,2]*c[:,1]) - a[:,1]*(b[:,0]*c[:,2]-b[:,2]*c[:,0])
           + a[:,2]*(b[:,0]*c[:,1]-b[:,1]*c[:,0]))
    rho = total/(det/6)
    lo, hi = (900., 1100.) if mat == 15 else (850., 950.)
    tol = (64*np.finfo(float).eps)*np.maximum(hi, abs(rho))
    invalid = (det <= 0) | (~np.isfinite(rho)) | (rho < lo-tol) | (rho > hi+tol)
    initial_outliers = int(np.count_nonzero(invalid))
    # Sliver volumes amplify cancellation. Re-evaluate only ambiguous determinants
    # in exact rational arithmetic on the same emitted coordinates.
    ax, bx, cx = abs(a), abs(b), abs(c)
    permanent = (ax[:,0]*(bx[:,1]*cx[:,2]+bx[:,2]*cx[:,1]) + ax[:,1]*(bx[:,0]*cx[:,2]+bx[:,2]*cx[:,0])
                 + ax[:,2]*(bx[:,0]*cx[:,1]+bx[:,1]*cx[:,0]))
    det_error = 12*np.finfo(float).eps*permanent
    near = invalid | (6*total < (lo-tol)*(det+det_error)) | (6*total > (hi+tol)*(det-det_error))
    for j in np.flatnonzero(near):
        corners = [[Fraction.from_float(float(value)) for value in row] for row in tet[j]]
        u, v, h = [[corner[k]-corners[0][k] for k in range(3)] for corner in corners[1:]]
        determinant = (u[0]*(v[1]*h[2]-v[2]*h[1]) - u[1]*(v[0]*h[2]-v[2]*h[0])
                       + u[2]*(v[0]*h[1]-v[1]*h[0]))
        if determinant > 0:
            rho[j] = float(Fraction.from_float(float(total[j]))*6/determinant)
            invalid[j] = rho[j] < lo-tol[j] or rho[j] > hi+tol[j]
    offsets = p1['cell_admissible_region_offsets']
    region = p1['cell_admissible_region_ids']
    typed = labels[parents, 2]
    inside = np.empty(len(w), bool)
    for label in np.unique(typed):
        ix = np.flatnonzero(typed[q] == label)
        rows = np.searchsorted(offsets[1:], np.flatnonzero(region == label), side='right')
        inside[ix] = np.isin(g[ix], rows)
    distance = np.empty(len(w))
    for start in range(0, len(w), 100000):
        s = slice(start, start + 100000)
        distance[s] = np.linalg.norm(points[q[s]] - center[g[s]], axis=1)
    if not np.isfinite(distance).all() or np.max(distance) > radius:
        raise ValueError('Unreachable spatial transfer')
    exc = ~inside
    historical = ~prior['eligible']
    keys = prior['role_cell'][historical].astype('i8')*len(quota) + prior['carrier'][historical]
    order = np.argsort(keys)
    if not np.array_equal(key[exc], keys[order]) or not np.array_equal(w[exc], prior['mass'][historical][order]):
        raise ValueError('Changed exceptional source, receiver or debit')
    if np.any(exc):
        if not (np.isin(parents[q[exc]], packet['regional_compensation_parent_rows']).all()
                and np.isin(role_rows[g[exc]], packet['regional_compensation_child_rows']).all()
                and w[exc].sum() <= packet['regional_compensation_limits'][0]
                and distance[exc].max() <= packet['regional_compensation_limits'][1]):
            raise ValueError('Unapproved cross-region mass')
    err = total-target
    dq = (donor-quota)/quota
    dg = err/target
    return dict(material=mat, source_carriers=len(quota), recipient_cells=len(total), positive_transfers=len(w),
                mass_kg=float(total.sum()), max_relative_source_error=float(np.max(abs(dq))),
                max_relative_recipient_error=float(np.max(abs(dg))), density_outliers=int(invalid.sum()),
                density_failures_global=role_rows[invalid].tolist(), exact_determinants=int(near.sum()),
                fast_determinant_density_flags=initial_outliers, min_density=float(rho.min()), max_density=float(rho.max()), max_distance_m=float(distance.max()),
                fixed_exception_edges=int(exc.sum()), fixed_exception_mass_kg=float(w[exc].sum()),
                fixed_exception_max_distance_m=float(np.max(distance[exc], initial=0)),
                max_cell_first_moment_sum_difference_kg_m=float(np.max(abs(err @ center))),
                verified=bool(not invalid.any() and np.max(abs(dq)) <= 1e-9 and np.max(abs(dg)) <= 1e-9))


def run(launch, mesh, packets, support, flows, out):
    e = json.loads(launch.read_bytes())['environment']
    rec = json.loads((mesh/'receipt.json').read_bytes())
    for f in rec['entries']:
        if sha(mesh/f['file']) != f['sha256']:
            raise ValueError('Changed mesh')
    x = tetgen_positions(mesh/'mesh.node').astype('f4').astype('f8')
    cells = np.loadtxt(mesh/'mesh.ele', skiprows=1, usecols=(1,2,3,4,5), dtype='i4')
    if sha(mesh/'cell-mass.f64') != rec['cell_mass_sha256']:
        raise ValueError('Changed target masses')
    m = np.fromfile(mesh/'cell-mass.f64', dtype='f8')
    files = [Path(e['NUMI_H706_FRACTIONAL_SOURCE330_'+k]) for k in ('TETS','NODES')]
    if any(sha(f) != e['NUMI_H706_FRACTIONAL_SOURCE330_'+k+'_SHA256'] for f,k in zip(files,('TETS','NODES'))):
        raise ValueError('Changed quadrature sources')
    t = np.loadtxt(files[0], delimiter=',', skiprows=1, usecols=(3,4,5,6), dtype='i4')
    pm = np.loadtxt(files[0], delimiter=',', skiprows=1, usecols=8)
    xp = np.loadtxt(files[1], delimiter=',', skiprows=1, usecols=(5,6,7))
    lab = np.loadtxt(pinned(e,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'), delimiter=',', skiprows=1, usecols=(1,2,3), dtype='i4')
    report = {}
    for tag, mat in (('SKIN',15),('FAT',16)):
        sr = json.loads((support/(tag.lower()+'-p1-receipt.json')).read_bytes())
        p = Path(sr['p1_one_hop_path'])
        if sha(p) != sr['p1_one_hop_sha256']:
            raise ValueError('Changed support')
        f = flows/(tag+'-physical-flow.npz')
        r = role_audit(tag,mat,np.load(packets/(tag.lower()+'-carriers.npz')),np.load(packets/(tag+'-candidate-flow.npz')),
                       np.load(p),np.load(f),mesh,cells,x,m,(t,xp,pm),lab,rec['radius_m'])
        report[tag] = dict(flow_sha256=sha(f), **r)
        print(tag, report[tag], flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    return int(not all(v['verified'] for v in report.values()))


if __name__ == '__main__':
    sys.exit(run(*(Path(v).resolve() for v in sys.argv[1:])))
