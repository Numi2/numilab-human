"""Screen existing fixed-boundary cavity operations on the current native mesh."""
import hashlib
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
retained = here.parents[3] / "resting-clearance-and-atrium-010/screen_fine_fixed_boundary_flips.py"
original = retained.read_text()
start = original.index("corners=tets.ravel()")
end = original.index("used=set()", start)
old = original[start:end]
print(str(retained), hashlib.sha256(original.encode()).hexdigest(), file=sys.stderr)
geometry_audit = (here / "audit_fixed_boundary.py").read_text()
anchors = ("def summarize", "# Pack three sorted indices", "result[\"frozen_boundary\"]")
summarize, faces, end_faces = map(geometry_audit.index, anchors)
core = geometry_audit[:summarize] + geometry_audit[faces:end_faces]
prior_import = original[original.index("script=Path.home()"):original.index("\ndata=Path")]
header = '''
from collections import Counter
import time
def volumes(points, tetrahedra):
    p = points[tetrahedra]
    d = p[:,1:] - p[:,:1]
    det = np.linalg.det(d)
    diameter = np.maximum.reduce([np.linalg.norm(p[:,a]-p[:,b],axis=1) for a,b in edges])
    return det / 6, det / diameter**3

tets, material, geometry = nodes, role, points
'''
trainer = '''
# Active physics, mass and ancestry are not touched in this diagnostic.
# Candidate energy is geometric. Centered singular values make its condition
# comparison independent of node order and the selected corner origin.
def metrics(local):
    rows = geometry[local]
    singular = np.linalg.svd(rows - rows.mean(axis=1,keepdims=True), compute_uv=False)
    basis = rows[:,1:] - rows[:,:1]
    inverse = np.linalg.inv(basis)
    local_mass = np.abs(np.linalg.det(basis)) / 6
    proxy = local_mass * (np.sum(inverse**2,axis=(1,2)) +
                          np.sum(inverse.sum(axis=2)**2,axis=1))
    return float(np.max(singular[:,0]/singular[:,-1])), float(proxy.sum())

begin = time.monotonic()
limit = int(sys.argv[2])
fixed = np.flatnonzero((fixed_count == 4) & (condition >= 1000))
fixed = fixed[np.argsort(-measure[fixed],kind="stable")][:limit]
selected, used, vertices_used = [], set(), set()
tested, rejections = Counter(), Counter()
for row in fixed:
    if row in used or any(v in vertices_used for v in tets[row]):
        continue
    best = None
    for kind, retired, replacement in make_proposals(row):
        tested[kind] += 1
        ri = sorted(retired)
        old = tets[ri]
        if np.any(material[ri] != material[row]) or any(v in vertices_used for v in old.ravel()):
            rejections["ownership_or_overlap"] += 1
            continue
        v, q = volumes(geometry, replacement)
        if np.any(np.abs(v) <= 1e-18) or np.any(np.abs(q) < 1e-5):
            rejections["absolute_or_shape_floor"] += 1
            continue
        flip = v < 0
        replacement[flip] = replacement[flip][:,[1,0,2,3]]
        if chain(old) != chain(replacement):
            rejections["oriented_cavity_boundary"] += 1
            continue
        if any(hits(t)-retired for t in replacement):
            rejections["duplicate_cell"] += 1
            continue
        old_volume = volume[ri].sum()
        if not np.isclose(old_volume, np.abs(v).sum(), rtol=1e-11, atol=1e-22):
            rejections["volume"] += 1
            continue
        before, old_proxy = metrics(old)
        after, new_proxy = metrics(replacement)
        if not (after <= .80*before and new_proxy <= .90*old_proxy):
            rejections["comparative_conditioning"] += 1
            continue
        score = old_proxy - new_proxy
        if best is None or score > best[0]:
            best = (score, ri, old, dict(kind=kind, retired=ri, material_id=int(material[row]),
                        new=replacement.tolist(), old_centered_condition=before,
                        new_centered_condition=after, old_gradient_energy_proxy=old_proxy,
                        new_gradient_energy_proxy=new_proxy, cavity_volume_m3=float(old_volume),
                        parent_union_from_retired_current_rows=ri))
    if best:
        score, ri, old, entry = best
        selected.append(entry)
        used.update(ri)
        vertices_used.update(old.ravel())

result = dict(input_sha256=digests, scope="Disjoint same-material cavity proposal. Mass ancestry, global intersections and native dynamics are not qualified.",
             target_count=len(fixed), tested=dict(tested), rejections=dict(rejections),
             selected=len(selected), affected_baseline_cells=len(used),
             new_cells=sum(len(e["new"]) for e in selected),
             rejected_or_unused_target_count=len(set(fixed)-used),
             whole_body_proxy_fraction_removed=float(sum(e["old_gradient_energy_proxy"]-e["new_gradient_energy_proxy"]
                                                        for e in selected)/measure.sum()),
             wall_s=time.monotonic()-begin, replacements=selected)
print(json.dumps(result, indent=2))
'''
(here / "screen_conditioning_flips.py").write_text(core + prior_import + header + old + trainer)
