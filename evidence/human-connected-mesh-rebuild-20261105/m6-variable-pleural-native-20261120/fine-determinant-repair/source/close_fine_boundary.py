"""Repair witnessed deep-layer folds while protecting sampled Float32 determinants."""
from pathlib import Path
from itertools import combinations
import json
import subprocess

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack
from scipy.spatial import cKDTree

from candidate import load

original, initial, cells, material, ancestors, _, roles, source_cells, source_material = load(include_interface_polish=False)
# The witnessed folded boundary is attached to one artificial 2->3 split.
# Restore the original two-tet cavity before smoothing its neighborhood.
boundary_group = 49
was_split = ancestors == -(boundary_group+1)
restored = [1278082, 1278083]
assert np.count_nonzero(was_split)==3 and np.all(source_material[restored]==16)
assert not np.any(np.isin(restored,ancestors))
cells = np.vstack((cells[~was_split], source_cells[restored]))
material = np.r_[material[~was_split], source_material[restored]]
bank = Path('fine-boundary-f32.bin').read_bytes()
witnesses = [list(map(int,l.split()[1:])) for l in Path('fine-boundary-audit.log').read_text().splitlines() if l.startswith('witness ')]
assert len(witnesses)==7 and all(row[:2]==[2,2] for row in witnesses)
must_move = np.unique([row[4:] for row in witnesses])
assert np.all(roles[must_move]=='D')
flat = cells.ravel()
ordered = np.argsort(flat, kind='stable')//4
offset = np.r_[0, np.cumsum(np.bincount(flat, minlength=len(initial)))]
tree=cKDTree(original)
edges=np.asarray(list(combinations(range(4),2)))
floor=1.e-5
axes=np.arange(3)

def incident(nodes):
    return np.unique(np.concatenate([ordered[offset[j]:offset[j+1]] for j in nodes]))

def calc(x,tet):
    p=x[tet].astype(float)
    e=p[:,1:]-p[:,:1]
    a,b=np.moveaxis(edges,1,0)
    side=p[:,a]-p[:,b]
    lengths=np.einsum('nki,nki->nk',side,side)
    longest=lengths.argmax(axis=1)
    row=np.arange(len(tet))
    scale=lengths[row,longest]**1.5
    cross=np.cross(e[:,[1,2,0]],e[:,[2,0,1]])
    q=np.einsum('ni,ni->n',e[:,0],cross[:,0])/scale
    g=np.concatenate((-cross.sum(axis=1)[:,None],cross),axis=1)/scale[:,None,None]
    sg=3*side[row,longest]/lengths[row,longest,None]
    g[row,a[longest]]-=q[:,None]*sg
    g[row,b[longest]]+=q[:,None]*sg
    return q,g

# Use disconnected witness components with one translation per patch. The protected
# triangles therefore recover their exact source arrangement to Float32 accuracy.
rest=original[must_move].astype(float)
graph=cKDTree(rest).query_ball_tree(cKDTree(rest),.008)
remaining=set(range(len(rest)))
components=[]
while remaining:
    todo=[min(remaining)]
    seen=set()
    while todo:
        k=todo.pop()
        if k in seen: continue
        seen.add(k)
        todo.extend(i for i in graph[k] if i not in seen)
    remaining-=seen
    components.append(must_move[sorted(seen)])

def restore(radius,center_mode):
    x=initial.copy()
    for seeds in components:
        seed_points=original[seeds].astype(float)
        if center_mode=='mean':
            center=np.mean(initial[seeds].astype(float)-seed_points,axis=0)
        else:
            center=np.median(initial[seeds].astype(float)-seed_points,axis=0)
        local=np.unique(np.concatenate(tree.query_ball_point(seed_points,radius)))
        local=local[roles[local]!='O']
        distance=cKDTree(seed_points).query(original[local])[0]
        weight=(1-np.minimum(1.,distance/radius))**2
        proposal=original[local].astype(float)+center
        x[local]=initial[local].astype(float)+weight[:,None]*(proposal-initial[local])
        x[seeds]=original[seeds].astype(float)+center
    return x

def improve(x,local,affected):
    t=cells[affected]
    relocal=np.searchsorted(local,t)
    valid=local[np.minimum(relocal,len(local)-1)]==t
    row,corner=np.nonzero(valid)
    n=3*len(local)
    for _ in range(9):
        current,g=calc(x,t)
        failing=current<floor
        if not np.any(failing):
            return
        padding=.55*np.sum(np.abs(g)*np.where(valid[:,:,None],np.abs(np.spacing(x[t])),0.),axis=(1,2))
        goal=np.where(failing,floor*1.6+padding,floor+np.minimum(padding,.5*(current-floor)))
        jac=coo_matrix(((g[row,corner]*.001).ravel(),
                        (np.repeat(row,3),(3*relocal[row,corner,None]+axes).ravel())),
                        shape=(len(t),n)).tocsr()
        scores=np.flatnonzero(failing)
        slack=coo_matrix((-np.ones(len(scores)),(scores,np.arange(len(scores)))),shape=(len(t),len(scores)))
        iden=eye(n,format='csr')
        bounds=.27
        rhs=np.r_[current-goal,np.zeros(2*n)]
        lhs=vstack((hstack((-jac,coo_matrix(jac.shape),slack)),
                    hstack((vstack((iden,-iden)),vstack((-iden,-iden)),coo_matrix((2*n,len(scores))))))).tocsr()
        objective=np.r_[np.zeros(n),np.repeat(np.where(roles[local]=='D',2.,1.),3),np.full(len(scores),2e5)]
        lp=linprog(objective,A_ub=lhs,b_ub=rhs,bounds=[(-bounds,bounds)]*n+[(0,bounds)]*n+[(0,None)]*len(scores),
                   method='highs',options={'primal_feasibility_tolerance':1e-9})
        if not lp.success:
            return
        base=x[local].copy()
        def rank(q):
            return int(np.count_nonzero(q<floor)),float(np.maximum(0.,floor-q).sum())
        best=rank(current),base
        for fraction in (1.,.875,.625,.5,.25):
            x[local]=base.astype(float)+fraction*.001*lp.x[:n].reshape(-1,3)
            q=calc(x,t)[0]
            if not np.any((current>=floor)&(q<floor)) and rank(q)<best[0]:
                best=rank(q),x[local].copy()
        x[local]=best[1]
        if not best[0]<rank(current):
            return

# First look for a pure position-only rollback around the two narrow witness
# regions. Only use the local LP if determinants require rebalancing.
for radius,kind in ((.001,'mean'),(.002,'mean'),(.0035,'mean'),(.005,'mean'),(.002,'median'),(.0035,'median')):
    x=restore(radius,kind)
    shifted=np.flatnonzero(np.any(x!=initial,axis=1))
    affected=incident(shifted)
    before=calc(x,cells[affected])[0]
    fail=np.flatnonzero(before<floor)
    if len(fail):
        local=np.unique(cells[affected[fail]])
        for _ in range(2):
            local=np.unique(cells[incident(local)])
        local=local[(roles[local]!='O')&~np.isin(local,must_move)]
        improve(x,local,incident(local))
    shifted=np.flatnonzero(np.any(x!=initial,axis=1))
    affected=incident(shifted)
    q=calc(x,cells[affected])[0]
    print('patch',radius,kind,'fixed',len(must_move),'changed',len(shifted),'quality_initial',len(fail),
          'quality_final',np.count_nonzero(q<floor),flush=True)
    if np.any(q<floor):
        for r,z in zip(affected[q<floor],q[q<floor]): print(r,z,cells[r],roles[cells[r]],np.isin(cells[r],must_move),flush=True)

        continue
    with open('fine-boundary-trial.bin','wb') as f:
        f.write(bank[:8+12*len(x)])
        x.astype('<f4',copy=False).tofile(f)
        f.write(bank[8+24*len(x):])
    out=subprocess.check_output(['./audit_fine_boundary','fine-boundary-trial.bin'],text=True)
    Path('fine-boundary-corrected-audit.log').write_text(out)
    counts=[int(l.split()[-1]) for l in out.splitlines() if l.startswith('pair ')]
    print('crossings',counts,flush=True)
    if any(counts):
        continue
    np.savez_compressed('fine-interface-polish.npz',changed_rows=shifted,positions_f32=x[shifted])
    result=dict(patch_radius_m=radius,patch_motion=kind,corrected_rows=shifted.tolist(),boundary_crossings=0,
                reversed_initial_flip=boundary_group, restored_source_rows=restored,
                changed_coordinates_by_role={str(r):int(np.count_nonzero(roles[shifted]==r)) for r in np.unique(roles)},
                minimum_local_quality=float(q.min()), affected_cells=len(affected),
                max_delta_m=float(np.max(np.linalg.norm((x-initial).astype(float),axis=1))))
    Path('fine-interface-polish.json').write_text(json.dumps(result,indent=2)+'\n')
    break
else:
    raise SystemExit('bounded interface repair exhausted')
