"""Conservative rest-volume overlay of two connected same-material tetrahedralizations."""
import ctypes
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from compile_connected_joint import digest

EDGES = list(itertools.combinations(range(4), 2))
BASIS = np.eye(4)
FACES = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]


def key(tet):
    return tuple(sorted(map(int, tet)))


def oriented_boundary(tets):
    faces = defaultdict(list)
    for cell in tets:
        for face in FACES:
            nodes = [int(cell[i]) for i in face]
            pos = nodes.index(min(nodes))
            ordered = tuple(nodes[pos:] + nodes[:pos])
            faces[tuple(sorted(nodes))].append(ordered)
    boundary = set()
    for directions in faces.values():
        if len(directions) == 1:
            boundary.add(directions[0])
        elif len(directions) != 2 or directions[0][1:] != directions[1][:0:-1]:
            raise ValueError('Nonmanifold or inconsistent interior orientation')
    return boundary


def intersection_fraction(old, new, points):
    # Intersect in barycentric coordinates of the source, including all 12 edge/face combinations per face.
    def bary(nodes, samples):
        xyz = points[nodes]
        tail = np.linalg.solve((xyz[1:] - xyz[0]).T, (points[samples] - xyz[0]).T).T
        coords = np.column_stack([1-tail.sum(axis=1), tail])
        for i, node in enumerate(samples):
            exact = np.flatnonzero(nodes == node)
            if len(exact):
                coords[i] = BASIS[exact[0]]
        return coords
    new_in_old, old_in_new = bary(old, new), bary(new, old)
    eps = 3e-11
    vertices = []
    for own, other in [(BASIS, old_in_new), (new_in_old, new_in_old)]:
        vertices.extend(own[(other >= -eps).all(axis=1), 1:])
        for a, b in EDGES:
            for k in range(4):
                u, v = other[a, k], other[b, k]
                if (u > eps and v < -eps) or (v > eps and u < -eps):
                    s = u/(u-v)
                    if np.min(other[a] + s*(other[b]-other[a])) >= -eps:
                        vertices.append((own[a] + s*(own[b]-own[a]))[1:])
    if len(vertices) < 4:
        return 0.
    xyz = np.unique(vertices, axis=0)
    if len(xyz) < 4 or np.linalg.matrix_rank(xyz[1:]-xyz[0], tol=1e-12) < 3:
        return 0.
    try:
        return ConvexHull(xyz).volume*6.
    except QhullError:
        if np.linalg.svd(xyz-xyz.mean(0), compute_uv=False)[-1] > 1e-11:
            raise
        return 0.


def collect_components(ops):
    parents = list(range(len(ops)))
    def root(n):
        while n != parents[n]:
            parents[n] = parents[parents[n]]
            n = parents[n]
        return n
    seen = {}
    for i, op in enumerate(ops):
        for cell in op['old']+op['new']:
            k = (op['material'], key(cell))
            if k in seen:
                parents[root(i)] = root(seen[k])
            seen[k] = i
    groups = defaultdict(list)
    for i, op in enumerate(ops):
        groups[root(i)].append(op)
    return groups.values()


def run(job, fit, output):
    output.mkdir()
    src = job/'interval-certified-scene/material'
    load = lambda p: np.ascontiguousarray(np.loadtxt(p, skiprows=1, usecols=(1, 2, 3)).astype('f4').astype('f8'))
    loaded = job/'verified-connected-material/current.node'
    proof = json.loads((fit/'result.json').read_bytes())
    for f, tag in [(src/'mesh.node','reference'),(loaded,'loaded'),(src/'mesh.ele','elements')]:
        if digest(f) != proof[tag]:
            raise ValueError('Source differs from repair')
    os.link(src/'mesh.node',output/'mesh.node')
    os.link(loaded,output/'current.node')
    x, y = load(src/'mesh.node'), load(loaded)
    old = np.loadtxt(src/'mesh.ele', dtype='i4', skiprows=1)
    t = np.ascontiguousarray(old[:,1:5]-1)
    role = old[:,5]
    z = np.load(fit/'patches.npz')
    old_id = {key(t[i]):i for i in np.r_[z['source0'],z['source1']]}
    oldmass = np.fromfile(src/'cell-mass.f64', '<f8')
    row, col, weight = [], [], []
    newcells, newroles, removed = [], [], set()
    worst_row = worst_col = 0.
    component_stats = []
    for group in collect_components(json.loads((fit/'ancestry.json').read_bytes())):
        change = Counter()
        cells = {}
        for op in group:
            for sign, name in [(-1,'old'),(1,'new')]:
                for c in op[name]:
                    k = key(c)
                    cells[k] = c
                    change[k] += sign
        if any(abs(n)>1 for n in change.values()):
            raise ValueError('Nonunique cavity ancestor')
        source = np.array([old_id[k] for k,n in change.items() if n == -1])
        after = np.array([cells[k] for k,n in change.items() if n == 1],dtype='i4')
        if set(source)&removed or oriented_boundary(t[source]) != oriented_boundary(after):
            raise ValueError('Cavity boundary changed')
        removed.update(source)
        ov = np.array([[intersection_fraction(c,d,x) for d in after] for c in t[source]])
        def volumes(q):
            return np.linalg.det(x[q[:,1:]]-x[q[:,:1]])/6
        v0, v1 = volumes(t[source]), volumes(after)
        if min(min(v0),min(v1)) <= 0:
            raise ValueError('Invalid orientation')
        err0 = max(abs(ov.sum(1)-1))
        err1 = max(abs((v0@ov)/v1-1))
        if max(err0,err1)>5e-8:
            raise ValueError(f'Invalid cavity overlay {source.tolist()} {err0} {err1}')
        worst_row, worst_col = max(err0,worst_row),max(err1,worst_col)
        ov /= ov.sum(1,keepdims=True)
        a,b = np.nonzero(ov>0)
        row.extend(source[a]);col.extend(len(newcells)+b);weight.extend(ov[a,b])
        newcells.extend(after)
        material = group[0]['material']
        if np.any(role[source] != material):
            raise ValueError('Cavity crossed material boundary')
        newroles.extend([material]*len(after))
        component_stats.append([material,source.tolist(),len(after),err0,err1])
    kept = np.ones(len(t),bool);kept[list(removed)]=False
    old_to_new = np.full(len(t),-1,'i4')
    old_to_new[kept]=np.arange(kept.sum(),dtype='i4')
    cell = np.ascontiguousarray(np.vstack([t[kept],newcells]),dtype='i4')
    mat = np.r_[role[kept],newroles].astype('i4')
    ir = np.asarray(row);ic=np.asarray(col)+kept.sum();w=np.asarray(weight)
    mass = np.r_[oldmass[kept],np.bincount(ic-kept.sum(),weights=oldmass[ir]*w,minlength=len(newcells))]
    nm = sum(np.bincount(cell[:,i],weights=mass/4,minlength=len(x)) for i in range(4))
    mm0 = np.fromfile(src/'node-mass.f64','<f8')
    for m in (15,16):
        if abs(np.sum(mass[mat==m])-np.sum(oldmass[role==m]))>1e-12:
            raise ValueError('Material mass changed')
    if np.any(nm<=0) or np.any(mass<=0):
        raise ValueError('Non-positive mass')
    # Run the native interval predicate on every tetrahedron in the emitted ordering.
    text = (job/'storage/interval-checker-locator.txt').read_text().split()
    if digest(Path(text[1]))!=text[0]:
        raise ValueError('Predicate identity')
    fn = ctypes.CDLL(text[1]).scanCellIntervals
    fp = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    ip = np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    fn.argtypes=[fp,fp,ctypes.c_size_t,ip,ctypes.c_size_t,fp]
    values=np.empty((len(cell),3))
    if fn(x,y,len(x),cell,len(cell),values)!=0:
        raise ValueError('Interval evaluation failed')
    margins = np.minimum(values[:,0]-values[:,1]-float(np.float32(.35)),2.5-values[:,0]-values[:,1])
    bad=np.flatnonzero((~np.isfinite(margins))|(margins<=0))
    if len(bad):
        raise ValueError(f'Uncertified initial intervals {bad.tolist()}')
    np.savez_compressed(output/'overlay.npz',source=ir,target=ic,fraction=w,old_to_new=old_to_new)
    with (output/'mesh.ele').open('w') as f:
        f.write(f'{len(cell)} 4 1\n')
        for i in range(0,len(cell),65536):
            n=min(i+65536,len(cell))
            np.savetxt(f,np.column_stack([np.arange(i+1,n+1),cell[i:n]+1,mat[i:n]]),fmt='%d')
    mass.astype('<f8').tofile(output/'cell-mass.f64');nm.astype('<f8').tofile(output/'node-mass.f64')
    report=dict(cells=len(cell),nodes=len(x),components=len(component_stats),input_affected=len(removed),new_cells=len(newcells),
                reference=digest(output/'mesh.node'),initial=digest(output/'current.node'),elements=digest(output/'mesh.ele'),
                mass=digest(output/'cell-mass.f64'),minimum_native_interval_margin=float(margins.min()),
                failed_intervals=len(bad),maximum_source_relative_error=worst_row,maximum_target_relative_error=worst_col,
                first_moment_change_kg_m=((nm-mm0)@x).tolist(),cavities=component_stats)
    (output/'transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='cavities'},flush=True)
if __name__=='__main__':
    run(*(Path(s).resolve() for s in sys.argv[1:]))
