"""Follow existing quadrature routes into overlapping replacement tetrahedra.

A protected upstream allocation retains its own equality. Other allocations can
move only along that quadrature's certified routes. Outside material cells and
new receiving cells share the same density and global first-moment constraints.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack
from register_material_surfaces import sha
from registered_density import density_bounds
from supported_material_admission import geom


def run(job,mesh,domain,overlap,allocation,flows,out):
    cert=json.loads((overlap/'result.json').read_bytes())
    if cert['mesh']!=sha(domain) or cert['mapping']!=sha(overlap/'overlap.npz'):raise ValueError('Changed validated overlap')
    d=np.load(domain);x=d['x'].astype(float);t=d['t'];r=d['role'];z=np.load(overlap/'overlap.npz')
    oldx=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3));blk=np.loadtxt(mesh/'mesh.ele',dtype='i4',skiprows=1);tc=blk[:,1:5]-1;mat=blk[:,5]
    M=np.fromfile(mesh/'cell-mass.f64',dtype='<f8');O=z['parent_ids'];n=len(tc)
    owner=Path(json.loads((job/'launch.json').read_text())['compiled_material_owner'])
    out.mkdir();result={}
    for material,tag in ((15,'SKIN'),(16,'FAT')):
        lo,hi=density_bounds(owner,material);delta=np.load(allocation/(tag+'-delta.npz'));base=flows/(tag+'-physical-flow.npz')
        if str(delta['base_flow_sha256'])!=sha(base):raise ValueError('Wrong source flow')
        wfile=np.load(base);q=wfile['carrier'];g=np.flatnonzero(mat==material)[wfile['role_cell']];w=wfile['mass']
        inside=np.isin(g,O);free=np.zeros(len(g),bool);free[delta['original_edges']]=True
        edges=np.flatnonzero(free|(inside&(w>0)));parents=g[edges];fixed=~free[edges]
        group=q[edges].astype('i8');group[fixed]=np.arange(fixed.sum())+q.max()+1
        _,row=np.unique(group,return_inverse=True);quota=np.bincount(row,weights=w[edges])
        # An intermediate parent routes only to children with actual material overlap.
        links={}
        for p,c,v in zip(z['parent'],z['child'],z['volume_mm3']):
            if r[c]==material:links.setdefault(int(p),[]).append((int(c)+n,v))
        upstream=[];target=[];initial=[]
        for j,p in enumerate(parents):
            routes=links.get(int(p),[]) if inside[edges[j]] else [(int(p),1.)]
            total=sum(v for _,v in routes)
            for c,v in routes:
                if quota[row[j]]>0:
                    upstream.append(j);target.append(c);initial.append(w[edges[j]]*v/total)
        up=np.array(upstream);s=row[up];initial=np.array(initial);targets,dst=np.unique(target,return_inverse=True)
        donors,si=np.unique(s,return_inverse=True)
        if np.any(quota[np.setdiff1d(np.arange(len(quota)),donors)]>0):raise ValueError('Positive source has no allowed route')
        supply=quota[donors];external=targets<n;oldids=targets[external];newids=targets[~external]-n
        if not np.array_equal(newids,np.flatnonzero(r==material)):raise ValueError('Replacement has unreachable child')
        ext=np.flatnonzero(~inside[edges]);spent=np.bincount(np.searchsorted(oldids,parents[ext]),weights=w[edges[ext]],minlength=len(oldids))
        reserve=np.r_[M[oldids]-spent,np.zeros(len(newids))];reserve=np.maximum(reserve,0)
        v=np.r_[geom(oldx,tc[oldids])[0],z['child_volume_mm3'][newids]*1e-9];center=x[t[newids]].mean((0,1))
        cent=np.vstack((oldx[tc[oldids]].mean(1),x[t[newids]].mean(1)))-center
        wanted=(M[O[mat[O]==material]]@(oldx[tc[O[mat[O]==material]]].mean(1)-center)
                +spent@(oldx[tc[oldids]].mean(1)-center))
        H=hi*v-reserve;L=np.maximum(lo*v-reserve,0);tolerance=3e-12*hi*v
        if np.any(H< -tolerance):raise ValueError('Protected reserve exceeds child density bound')
        H=np.maximum(H,0);den=np.maximum(H,np.bincount(dst,weights=initial))
        mask=den[dst]>0;dst,si,initial,up=dst[mask],si[mask],initial[mask],up[mask]
        scale=np.minimum(supply[si],den[dst]);k=np.arange(len(si))
        donor=coo_matrix((scale/supply[si],(si,k)),shape=(len(supply),len(k))).tocsr()
        denom=np.where(den>0,den,1.)
        recv=coo_matrix((scale/denom[dst],(dst,k)),shape=(len(v),len(k))).tocsr()
        base_mass=np.bincount(dst,weights=initial,minlength=len(v))
        eq=hstack((donor,-donor));moment=(cent[dst]*scale[:,None]).T;size=np.max(abs(moment),axis=1)
        rhs=np.r_[(supply-np.bincount(si,weights=initial,minlength=len(supply)))/supply,(wanted-base_mass@cent)/size]
        eq=vstack((eq,coo_matrix(np.c_[moment,-moment]/size[:,None]))).tocsr();cap=hstack((recv,-recv))
        top=np.minimum(np.maximum(H-base_mass,0),(H-L)*1e-7);btm=np.minimum(np.maximum(base_mass-L,0),(H-L)*1e-7)
        # New cells use interior density bounds so independent native determinant
        # and summation order can be certified at the original physical limits.
        top[~external]=btm[~external]=(H[~external]-L[~external])*1e-7
        refs=dict(sources=len(supply),recipients=len(v),edges=len(k))
        # Earlier carrier-to-parent allocation gives a warm transport estimate.
        # Preserve strictly frozen allocations as separate supplies.
        cost=np.maximum(scale/scale.max(),1e-8);params=dict(A_ub=vstack((cap,-cap)).tocsr(),b_ub=np.r_[(H-top-base_mass)/denom,(base_mass-L-btm)/denom],bounds=[(0,None)]*len(k)+list(zip(np.zeros(len(k)),initial/scale)),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
        for moments in (True,False):
            count=len(rhs) if moments else len(supply)
            lp=linprog(np.r_[cost,cost],A_eq=eq[:count],b_eq=rhs[:count],**params)
            refs['with_moments' if moments else 'without_moments']=lp.message
            if lp.success:break
        if lp.success:
            m=np.maximum(initial+scale*(lp.x[:len(k)]-lp.x[len(k):]),0);sums=np.bincount(dst,weights=m,minlength=len(v))
            err=np.max(abs(np.bincount(si,weights=m,minlength=len(supply))/supply-1));mom=sums@cent-wanted;rho=(sums+reserve)/v
            ok=err<1e-9 and np.all(rho>=lo-1e-8) and np.all(rho<=hi+1e-8) and np.max(abs(mom))<1e-11
            refs.update(passed=bool(ok),max_source_relative_error=float(err),moment_error_kg_m=mom.tolist(),density_min=float(rho.min()),density_max=float(rho.max()),mass_error_kg=float(m.sum()-supply.sum()))
            np.savez_compressed(out/(tag+'-mapping.npz'),source_edge=edges[up],destination=targets[dst],mass_kg=m,recipient=targets,receiver_mass_kg=sums+reserve)
        result[tag]=refs
        print(tag,refs,flush=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*(Path(a) for a in sys.argv[1:]))
