"""Transfer source mass within conforming, face-adjacent children after Float32 emission."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from refine_shoulder_material import FACES,determinant


def map_mass(z, source_mass, labels):
    t, old, root, x = z['t'], z['base'], z['source_parent'], z['X'][0]
    mu = z['mass'].copy()
    centroid = x[t].mean(1)
    V = determinant(x,t)/6
    lower = np.where(z['roles']==15,900.,850.)
    upper = np.where(z['roles']==15,1100.,950.)
    # Work in nanograms for strict absolute equality at the compiler's kilogram scale.
    unit = 1e-12
    n=len(t);k=len(old)
    owners=list(root)+list(root);children=list(range(n))*2
    signs=[1.]*n+[-1.]*n;cost=[1.]* (2*n)
    faces={}
    for i,tt in enumerate(old):
        for f in tt[FACES]:
            key=tuple(sorted(f))
            if key in faces:
                j=faces[key]
                if np.array_equal(labels[i],labels[j]):
                    for donor,other in ((i,j),(j,i)):
                        # Spill only into descendants which touch the common original face.
                        verts=x[list(key)]
                        normal=np.cross(verts[1]-verts[0],verts[2]-verts[0]);normal/=np.linalg.norm(normal)
                        dst=np.flatnonzero(root==other)
                        d=abs((x[t[dst]]-verts[0])@normal)
                        near=dst[(d<1e-7).sum(1)>=3]
                        for child in near:
                            owners.append(donor);children.append(child);signs.append(1.);cost.append(5.+np.linalg.norm(centroid[child]-x[old[donor]].mean(0))/.01)
            else:faces[key]=i
    owners,children,signs=map(np.asarray,(owners,children,signs));u=np.arange(len(owners))
    footprint=x[old].mean(1)
    basis=np.column_stack((np.ones(len(u)),(centroid[children]-footprint[owners])/.05))
    eq=coo_matrix(((basis*signs[:,None]).ravel(),((4*owners[:,None]+np.arange(4)).ravel(),np.repeat(u,4))),shape=(4*k,len(u))).tocsr()
    b=np.zeros((k,4));b[:,0]=source_mass-np.bincount(root,weights=mu,minlength=k)
    for axis in range(3):
        b[:,axis+1]=-np.bincount(root,weights=mu*(centroid[:,axis]-footprint[root,axis])/.05,minlength=k)
    a=coo_matrix((signs,(children,u)),shape=(n,len(u))).tocsr()
    # Interior slack protects Float64 determinant roundoff at active density limits.
    epsilon=1e-7
    lb=((lower+epsilon)*V-mu)/unit;ub=((upper-epsilon)*V-mu)/unit
    bounds=np.column_stack((np.zeros(len(owners)),np.r_[np.full(n,np.inf),mu/unit,np.full(len(owners)-2*n,np.inf)]))
    res=linprog(cost,A_ub=vstack((a,-a)),b_ub=np.r_[ub,-lb],A_eq=eq,b_eq=b.ravel()/unit,bounds=bounds,method='highs',options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    if not res.success:raise ValueError(res.message)
    source=np.r_[root,owners[2*n:]];target=np.r_[np.arange(n),children[2*n:]]
    packet=np.r_[mu+(res.x[:n]-res.x[n:2*n])*unit,res.x[2*n:]*unit]
    positive=packet>0;source,target,packet=source[positive],target[positive],packet[positive]
    actual=np.bincount(target,weights=packet,minlength=n)
    moment=np.array([np.bincount(source,weights=packet*(centroid[target,j]-footprint[source,j]),minlength=k)for j in range(3)]).T
    err=abs(np.bincount(source,weights=packet,minlength=k)-source_mass).max()
    if packet.min()<=0 or err>1e-14 or np.linalg.norm(moment,axis=1).max()>1e-14 or (actual/V<lower).any() or (actual/V>upper).any() or np.any(labels[source]!=labels[root[target]]):
        raise ValueError('Source mass, centroid, donor, or child density changed outside its physical limits')
    return actual,source,target,packet,dict(max_source_mass_error_kg=float(err),max_source_moment_error_kg_m=float(np.linalg.norm(moment,axis=1).max()),adjacent_source_packets=int(np.sum(source!=root[target])),adjacent_source_mass_kg=float(packet[source!=root[target]].sum()),density_kg_m3={int(role):[float((actual/V)[z['roles']==role].min()),float((actual/V)[z['roles']==role].max())]for role in (15,16)})
