"""Fit a shared local field correction without moving the zero-coordinate heart."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import struct

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "resting-clearance-and-atrium-003"
MODES = (1, 2, 3, 4)  # Row zero is position, the next four are the chamber coefficients.
SCALES = np.asarray((0.32, -0.58, 0.32, -0.58))
RESERVE_MM = 0.0004
LIMIT_MM = 0.03


def geometry(path):
    raw = path.read_bytes()
    n, f, w = struct.unpack_from("<III", raw)
    rows = np.frombuffer(raw, "<f4", 24*n, 12).reshape(n, 8, 3)
    faces = np.frombuffer(raw, "<u4", 3*f, 12+96*n).reshape(f, 3)
    windows = np.frombuffer(raw, "<f4", 14*w, 12+96*n+12*f).reshape(w, 2, 7)
    return raw, rows, faces, windows


def position(rows, q):
    xyz = rows[:, 0].copy()
    for basis, scale in zip(rows[:, 1:].transpose(1, 0, 2), q):
        xyz = (xyz.astype(np.float64) + basis.astype(np.float64)*float(scale)).astype("<f4")
    return xyz.astype(float)*1000.0


def separation_axes(x, y):
    dx, dy = np.roll(x, -1, axis=0)-x, np.roll(y, -1, axis=0)-y
    nx, ny = np.cross(dx[0], dx[1]), np.cross(dy[0], dy[1])
    edge_axes = np.cross(dx[:, None], dy).reshape(-1, 3)
    return np.vstack((nx, ny, edge_axes, np.cross(nx, dx), np.cross(ny, dy)))


def gap_pairs(a, b):
    shared = np.intersect1d(a, b)
    if len(shared) == 2:
        raise ValueError("shared-edge witness needs a dihedral constraint")
    return [(i, j) for i in a for j in b if i != j and (
        len(shared) == 0 or i in shared or j in shared)]


def best_axis(a, b, current, rest, pairs):
    pairs = np.asarray(pairs)
    diff = current[pairs[:, 1]]-current[pairs[:, 0]]
    rest_diff = rest[pairs[:, 1]]-rest[pairs[:, 0]]
    axes = np.vstack((separation_axes(current[a], current[b]),
                      separation_axes(rest[a], rest[b]), diff, rest_diff))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1.e-12]/norms[norms > 1.e-12, None]
    axes = np.r_[axes, -axes]
    # Find a separating plane through any canonical shared vertex. For a
    # disjoint pair, minimize the translation required to separate projections.
    for d in (diff, rest_diff):
        d = d/np.maximum(np.linalg.norm(d, axis=1)[:, None], 1.e-12)
        result = linprog([0., 0., 0., -1.], A_ub=np.c_[-d, np.ones(len(d))],
                         b_ub=np.zeros(len(d)), bounds=[(-1.,1.)]*3+[(None,None)],
                         method="highs")
        if result.success and (size := np.linalg.norm(result.x[:3])) > 1.e-10:
            axes = np.r_[axes, [result.x[:3]/size]]
    score = (diff @ axes.T).min(axis=0)
    center = (rest_diff @ axes.T).min(axis=0)
    return axes[np.argmax(score + 0.12*np.minimum(center, 0.0))]


class Inequalities:
    def __init__(self):
        self.row, self.col, self.values, self.rhs = [], [], [], []

    def upper(self, items, cap):
        for column, value in items:
            if value != 0:
                self.row.append(len(self.rhs))
                self.col.append(column)
                self.values.append(value)
        self.rhs.append(cap)

    def matrix(self, columns):
        return coo_matrix((self.values, (self.row, self.col)), (len(self.rhs), columns)).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--additional", action="append", type=Path, default=[])
    parser.add_argument("--tag", default="field-trial-1")
    opts = parser.parse_args()
    original = opts.input
    raw, rows, faces, windows = geometry(original)
    table = PRIOR/"common-cardiac-tight-witnesses.csv"
    source = rows[:, 0].astype(float)*1000.
    initial = [position(rows, q) for q in windows[:, 0]]
    linearization = initial
    if opts.previous:
        _, posed, topology, parameters = geometry(opts.previous)
        assert np.array_equal(faces, topology) and np.array_equal(windows, parameters)
        linearization = [position(posed, q) for q in windows[:, 0]]
    witnesses = set()
    for filename in [table, *opts.additional]:
        with filename.open() as f:
            for line in csv.DictReader(f):
                step, a, b = map(int, (line["window"], line["first_face"], line["second_face"]))
                if b<0:
                    raise ValueError("a candidate contains a degenerate face")
                witnesses.add((step, a, b))
    impacted = np.zeros(len(rows), dtype=bool)
    impacted[faces[np.asarray(sorted(witnesses))[:, 1:]].reshape(-1)] = True
    # Free two one-rings, then smooth to a third, fixed outer ring.
    for _ in range(2):
        impacted[faces[impacted[faces].any(axis=1)].ravel()] = True
    local = np.flatnonzero(impacted)
    index = np.full(len(rows), -1, dtype=int)
    index[local] = np.arange(len(local))
    neighbourhood = faces[impacted[faces].any(axis=1)]
    edge_set = np.sort(neighbourhood[:,[0,1,1,2,2,0]].reshape(-1,2), axis=1)
    edge_set = np.unique(edge_set, axis=0)
    edge_set = edge_set[impacted[edge_set].any(axis=1)]
    local_dofs = 3*len(local)
    nd = local_dofs*len(MODES)
    eq = Inequalities()

    def entries(node, vector, mode):
        ix = index[node]
        if ix>=0:
            yield from ((mode*local_dofs+3*ix+axis, c) for axis,c in enumerate(vector))

    def affected_entries(node, vector, q):
        return tuple(itertools.chain.from_iterable(entries(node, vector*factor, k)
            for k,factor in enumerate(q[np.asarray(MODES)-1]/SCALES)))

    nw = len(witnesses)
    for witness,(step,a,b) in enumerate(sorted(witnesses)):
        tri_a, tri_b = faces[a], faces[b]
        pairs = gap_pairs(tri_a, tri_b)
        axis = best_axis(tri_a, tri_b, linearization[step], source, pairs)
        q = windows[step, 0]
        for i,j in pairs:
            bound = float((initial[step][j]-initial[step][i])@axis)-RESERVE_MM
            eq.upper([*affected_entries(i, axis, q), *affected_entries(j, -axis, q), (2*nd+witness,-1.)], bound)
    # Keep the oriented projected area above half its retained value at each
    # fitted pose. Exact nonlinear orientation is audited after rounding.
    for step, q in enumerate(windows[:,0]):
        tri = initial[step][neighbourhood]
        cross = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
        area = np.linalg.norm(cross,axis=1)
        assert np.all(area>0)
        normals = cross/area[:,None]
        derivatives = np.cross(np.roll(tri, -1, axis=1)-np.roll(tri, -2, axis=1),
                               normals[:,None,:])
        for k,face in enumerate(neighbourhood):
            eq.upper(list(itertools.chain.from_iterable(affected_entries(v,-c,q) for v,c in zip(face,derivatives[k]))),
                     0.40*area[k])

    for i in range(nd):
        eq.upper(((i,1.),(nd+i,-1.)), 0.)
        eq.upper(((i,-1.),(nd+i,-1.)), 0.)
    for mode in range(len(MODES)):
        for e,(a,b) in enumerate(edge_set):
            for h,v in enumerate(np.eye(3)):
                off = 2*nd+nw+3*(mode*len(edge_set)+e)+h
                eq.upper([*entries(a,v,mode),*entries(b,-v,mode),(off,-1.)], 0.)
                eq.upper([*entries(a,-v,mode),*entries(b,v,mode),(off,-1.)], 0.)
    costs = np.r_[np.zeros(nd),np.ones(nd)*.15,np.full(nw,250.),np.full(3*len(edge_set)*len(MODES),.5)]
    solved = linprog(costs,A_ub=eq.matrix(len(costs)),b_ub=eq.rhs,
                     bounds=[(-LIMIT_MM,LIMIT_MM)]*nd+[(0.,None)]*(len(costs)-nd),
                     method="highs",options={"time_limit":110.})
    report = dict(witnesses=len(witnesses),local_vertices=len(local),equations=len(eq.rhs),
                  bounded_candidate=solved.success, solver_message=solved.message)
    if solved.success:
        update=solved.x[:nd].reshape(len(MODES),-1,3)
        updated=rows.copy()
        for k,mode in enumerate(MODES):
            updated[local,mode]=(updated[local,mode]+update[k]/(SCALES[k]*1000.)).astype("<f4")
        assert np.array_equal(updated[:,0], rows[:,0])
        report["maximum_peak_reference_displacement_mm"]=float(np.linalg.norm(update,axis=1).max())
        report["positive_slacks"]=int(np.count_nonzero(solved.x[2*nd:2*nd+nw]>1.e-6))
        report["changed_vertices"]=int(np.count_nonzero(np.any(updated[:,1:]!=rows[:,1:],axis=(1,2))))
        report["minimum_projected_area_ratio"]=[]
        for step,q in enumerate(windows[:,0]):
            a,b=(p[neighbourhood] for p in (initial[step],position(updated,q)))
            normal=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
            after=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
            report["minimum_projected_area_ratio"].append(float(np.min(
                np.sum(normal*after,axis=1)/np.sum(normal*normal,axis=1))))
        if min(report["minimum_projected_area_ratio"])>0.5:
            start=12+rows.nbytes
            (ROOT/(opts.tag+".bin")).write_bytes(raw[:12]+updated.tobytes()+raw[start:])
    (ROOT/(opts.tag+".json")).write_text(json.dumps(report,indent=2)+"\n")
    print(report,flush=True)


if __name__=="__main__":
    main()
