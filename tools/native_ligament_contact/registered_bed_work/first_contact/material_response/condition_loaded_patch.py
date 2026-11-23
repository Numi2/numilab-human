"""Fit all low-quality cells in the connected, load-bearing shoulder tissue at once."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, vstack, hstack
from condition_volume_quality import constraints
from fit_carpal_clearance import geom
from compile_connected_joint import digest


def ratios(x, t):
    v, q = geom(x[t])
    return np.minimum(v / 1.e-18, q / 1.e-5)


def run(mesh, planes, fixed_witness, out):
    z = np.load(mesh)
    base = z['x'].astype(float)
    x = base.copy()
    t = z['t']
    fixed = z['locked'].copy()
    fixed[np.unique(z['facets'][z['markers']==15])] = True
    fixed[np.load(fixed_witness)] = True
    d = np.load(planes)
    ids, norms, origins = (d[k] for k in ('node','normal','on_plane'))
    if len(base) <= max(ids):
        raise ValueError('Cortical plane references missing nodes')
    base_gap = np.einsum('ij,ij->i', norms, base[ids]-origins)*1.e6
    protected = np.minimum(base_gap, .08)
    initial = ratios(x,t)
    out.mkdir()
    history = []
    for iteration in range(48):
        values = ratios(x,t)
        failed = np.flatnonzero(values < 1.0)
        if not len(failed):
            break
        nodes = np.unique(t[failed])
        neighbors = np.unique(t[np.isin(t,nodes).any(1)])
        nodes = neighbors[~fixed[neighbors]]
        lookup = np.full(len(x),-1,int)
        lookup[nodes] = np.arange(len(nodes))
        local = lookup[t]
        cc = np.flatnonzero((local>=0).any(1))
        g = local[cc]
        e, grad, target = constraints(x,t[cc])
        nvars = 3*len(nodes)
        i,j,k = np.where(np.broadcast_to((g>=0)[:,:,None],grad.shape))
        coef = -grad[i,j,k]*1.e-6/target[i]
        A = coo_matrix((coef,(i,3*g[i,j]+k)),shape=(len(cc),nvars)).tocsr()
        # Increase the smallest normalized determinant along with all other deficits.
        # Keep the original safety margin of near-passing cells throughout the step.
        q = e/target + 1.
        original_bad = values[cc] < 1.0
        floor = np.where(original_bad,1.035,np.minimum(q,1.055))
        rhs = q - floor
        badrows = np.flatnonzero(original_bad)
        mask = lookup[ids]>=0
        ind = np.flatnonzero(mask)
        cols = (3*lookup[ids[ind],None]+np.arange(3)).ravel()
        normalA = coo_matrix((-norms[ind].ravel(),(np.repeat(np.arange(len(ind)),3),cols)),shape=(len(ind),nvars))
        gaps = np.einsum('ij,ij->i', norms[ind], x[ids[ind]]-origins[ind])*1.e6
        A = vstack((A,normalA)).tocsr()
        rhs = np.r_[rhs,gaps-protected[ind]]
        nbad=len(badrows)
        slack=coo_matrix((-np.ones(nbad),(badrows,np.arange(nbad))),shape=(len(rhs),nbad))
        prob=hstack((A,slack))
        best = None
        def score(value):
            return (int(np.count_nonzero(value<1.)),float(np.maximum(1.-value,0.).sum()))
        before = score(values)
        for trust in (5.,35.,200.,500.):
            delta = 1.e6*(x[nodes]-base[nodes])
            lo = np.maximum(-trust,-500.-delta).ravel()
            hi = np.minimum(trust,500.-delta).ravel()
            bounds=list(zip(lo,hi))+[(0.,None)]*nbad
            a = linprog(np.r_[np.zeros(nvars),np.ones(nbad)],A_ub=prob,
                        b_ub=rhs,bounds=bounds,method='highs')
            if not a.success:
                continue
            I = eye(nvars,format='csr')
            M = vstack((hstack((A,coo_matrix(A.shape))),hstack((I,-I)),hstack((-I,-I))))
            b = linprog(np.r_[np.zeros(nvars),np.ones(nvars)],A_ub=M,
                        b_ub=np.r_[rhs-slack@a.x[nvars:]+1e-9,np.zeros(2*nvars)],
                        bounds=bounds[:nvars]+[(0.,None)]*nvars,method='highs')
            if not b.success:
                continue
            step=b.x[:nvars].reshape(-1,3)*1e-6
            for alpha in (1.,.85,.6,.4,.2,.1,.04):
                new=x.copy()
                new[nodes]=(x[nodes]+alpha*step).astype('f4')
                val=ratios(new,t)
                value=score(val)
                if val.min()>0 and value<before and (best is None or value<best[0]):
                    best=(value,new)
        if best is None:
            break
        x=best[1]
        val=ratios(x,t)
        history.append(dict(iteration=iteration,low=int((val<1.).sum()),minimum=float(val.min()),deficit=best[0][1]))
        print(history[-1],flush=True)
    path=out/'domain.npz'
    np.savez_compressed(path,**{k:(x.astype('f4') if k=='x' else z[k]) for k in z.files})
    val=ratios(x,t)
    result=dict(source=digest(mesh),cortical_planes=digest(planes),fixed_witness=digest(fixed_witness),
                output=digest(path),initial_low=int((initial<1).sum()),final_low=int((val<1).sum()),
                inverted=int((val<=0).sum()),moved=int(np.any(x!=base,axis=1).sum()),max_move_m=float(np.linalg.norm(x-base,axis=1).max()),iterations=history)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['initial_low'],result['final_low'],result['moved'],result['max_move_m'])


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
