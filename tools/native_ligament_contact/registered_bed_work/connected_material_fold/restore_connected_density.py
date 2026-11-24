"""Restore local fat volume capacity on generated material corners, retaining measured boundaries."""
from pathlib import Path
from collections import Counter
import csv,json
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack
from measure_density_capacity import density
from compile_connected_joint import digest
from supported_material_admission import geom

def gradients(p):
    a,b,c=(p[:,i]-p[:,0] for i in (1,2,3))
    g=np.stack((np.cross(b,c),np.cross(c,a),np.cross(a,b)),axis=1)/6
    return np.concatenate((-g.sum(axis=1)[:,None,:],g),axis=1)

def run(q):
    volume=q/'remaining-fold/collar-2/fold-volume';out=q/'local-density-transfer';out.mkdir(exist_ok=True)
    x,t,lab,mass,v,center=density(volume);y=np.loadtxt(volume/'current.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8')
    bad=np.loadtxt(volume/'density-outliers.txt',ndmin=2)[:,0].astype(int);tag=lab[bad[0],1:]
    cells=np.flatnonzero((lab[:,1:]==tag).all(1));distance=cKDTree(center[bad]).query(center[cells])[0];cells=cells[distance<=.015]
    nodes=np.unique(t[cells]);other=np.ones(len(t),bool);other[cells]=False;fixed=np.unique(t[other])
    ids=np.load(q/'seam-volume/local-ancestry.npz')['surviving_source_nodes']
    cfg=json.loads((q/'gravity-scene/launch.json').read_bytes());mapping=Path(cfg['environment']['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if digest(mapping)!=cfg['pins'][str(mapping)]:raise ValueError('Unknown physical node registration')
    usable=[]
    with mapping.open()as stream:
        reader=csv.reader(stream);next(reader)
        for row in reader:
            if row[8]=='interior' and row[18]=='none' and row[19]=='-1' and row[20]=='-1':usable.append(int(row[0])-1)
    layers=np.load(q/'remaining-fold/source-surfaces.npz');bound=np.unique(np.concatenate([layers[n].ravel()for n in 'OID']))
    free=np.setdiff1d(nodes,np.union1d(fixed,bound));free=free[np.isin(ids[free],usable)&(x[free]==y[free]).all(1)]
    ix=np.full(len(x),-1,'i4');ix[free]=np.arange(len(free))
    inc=ix[t[cells]];g=gradients(x[t[cells]])
    gv=np.zeros((len(free),3));ii,jj=np.where(inc>=0);np.add.at(gv,inc[ii,jj],g[ii,jj])
    # Volume sensitivity of one isolated nodal move is exactly linear.
    mag=np.linalg.norm(gv,axis=1);order=np.argsort(-mag);radius=.005
    opts=[];trials=[];idx=np.arange(len(cells))
    for k in order:
        if mag[k]<1e-12:continue
        node=free[k];rows=np.where((inc==k).any(1))[0];sel=cells[rows];n=gv[k]/mag[k]
        for um in (.05,.15,.4,1.,2.):
            dest=(x[node]+n*um*1e-6).astype('f4').astype('f8');shift=dest-x[node]
            P=x[t[sel]].copy();P[t[sel]==node]=dest
            vv,ss=geom(P.reshape(-1,3),np.arange(4*len(P)).reshape(-1,4))
            if np.min(vv)<=1e-18 or np.min(ss)<1e-5:continue
            gain=950*np.sum(vv-v[sel]);cost=float(np.linalg.norm(shift))
            if gain<=0:continue
            opts.append((int(node),sel,dest,gain,cost,float(min(ss))))
    report=dict(source=digest(volume/'result.json'),bad_cells=len(bad),local_cells=len(cells),unmeasured_free_nodes=len(free),nodes_with_net_material_volume_response=int(np.sum(mag>1e-12)),eligible_moves=len(opts),trials=trials)
    # Select independent vertex stars. The LP chooses a local redistribution on
    # the exact represented geometry, with donor mass and its first moment fixed.
    oldtarget=center[cells].T@mass[cells];org=np.average(center[cells],axis=0,weights=mass[cells]);basecenter=center[cells]-org
    # Use the tight native bounds with a numerical margin only on changed masses.
    from scipy.sparse import csr_matrix,vstack
    from scipy.sparse.csgraph import connected_components
    from material_cavity_flips import faces
    facets=np.sort(faces(t[cells]),axis=1)
    owner=np.repeat(np.arange(len(cells)),4)
    order=np.lexsort(facets.T[::-1]);rows=order[1:][(facets[order[1:]]==facets[order[:-1]]).all(1)]
    cols=order[:-1][(facets[order[1:]]==facets[order[:-1]]).all(1)]
    aa,bb=owner[rows],owner[cols];ns=len(cells);ne=len(aa);edges=np.arange(ne)
    incidence=coo_matrix((np.r_[np.ones(ne),-np.ones(ne)],(np.r_[bb,aa],np.r_[edges,edges])),shape=(ns,ne)).tocsr()
    graph=coo_matrix((np.ones(ne),(aa,bb)),shape=(ns,ns))
    components,part=connected_components(graph,directed=False)
    distances=np.linalg.norm(center[cells[aa]]-center[cells[bb]],axis=1)
    D=hstack([incidence,-incidence],format='csr');bounds_matrix=vstack([D,-D],format='csr')
    original_bad=np.isin(cells,bad);report['face_connected_components']=components
    selected={};occupied=set();epsilon=2e-6
    for pos in range(12):
        choices=[o for o in opts if o[0]not in selected and not occupied.intersection(o[1]) and o[4]>0]
        if not choices:break
        def priority(o):
            closest=cKDTree(center[bad]).query(x[o[0]])[0]
            return min(o[3],6e-10)/(o[4]*(1+closest/radius))
        best=max(choices,key=priority);node,rows,dest,gain,cost,shape=best;selected[node]=dest;occupied.update(rows.tolist())
        xx=x.copy()
        for n,pt in selected.items():xx[n]=pt
        vol,qual=geom(xx,t[cells]);cc=xx[t[cells]].mean(1)-org
        # Solve only for signed mass deltas. Scaling preserves precision for
        # sub-microgram transfers on an otherwise unchanged body.
        sc=1e-9;lo=((850+epsilon)*vol-mass[cells])/sc;hi=((950-epsilon)*vol-mass[cells])/sc
        after=cc.T@mass[cells];before=basecenter.T@mass[cells]
        # Unchanged cells whose native densities already passed can keep their precise mass.
        original_valid=(~original_bad)&(~np.isin(cells,list(occupied)))
        hi[original_valid]=np.maximum(0,hi[original_valid]);lo[original_valid]=np.minimum(0,lo[original_valid])
        c_eq=[];rhs=[]
        for group in np.unique(part):
            sel=np.flatnonzero(part==group)
            if len(sel)<2:continue
            translated=center[cells[sel]].mean(0)-org
            basis=csr_matrix(((cc[sel]-translated)*1000).T)
            c_eq.append(basis@D[sel]);rhs.extend((basecenter[sel].T@mass[cells[sel]]-cc[sel].T@mass[cells[sel]])*1000/sc)
        res=linprog(np.r_[distances,distances]*1000,A_eq=vstack(c_eq,format='csr'),b_eq=rhs,A_ub=bounds_matrix,b_ub=np.r_[hi,-lo],bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-10})
        row=dict(node=int(node),shift_um=cost*1e6,gain_kg=float(gain),total_room_kg=float(hi.sum()*sc),feasible=res.success);trials.append(row)
        if res.success:
            directed=sc*(res.x[:ne]-res.x[ne:]);keep=np.flatnonzero(directed!=0)
            m=mass[cells]+incidence@directed;pr=m/vol
            if min(pr)<850-1e-6 or max(pr)>950+1e-6 or abs(m.sum()-mass[cells].sum())>1e-15:raise ValueError('Uncertified local allocation')
            active=np.flatnonzero(m!=mass[cells]);ns=np.array(sorted(selected),'i4');pin=out/'allocation.npz'
            np.savez_compressed(pin,nodes=ns,original_positions=x[ns].astype('f4'),positions=xx[ns].astype('f4'),credit_from=np.where(directed[keep]>0,cells[aa[keep]],cells[bb[keep]]),credit_to=np.where(directed[keep]>0,cells[bb[keep]],cells[aa[keep]]),credit_kg=abs(directed[keep]),cell_rows=cells[active],mass_before=mass[cells[active]],mass_after=m[active])
            report.update(candidate=digest(pin),max_density=float(max(pr)),minimum_density=float(min(pr)),changed_mass_cells=len(active),positive_adjacent_transfers=len(keep),longest_single_face_transfer_m=float(max(distances[keep])),max_local_shift_um=max(np.linalg.norm(xx[ns]-x[ns],axis=1))*1e6,local_mass_error_kg=float(m.sum()-mass[cells].sum()),first_moment_delta_kg_m=(cc.T@m-before).tolist())
            break
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k!='trials'},trials[-1:] if trials else [],flush=True)
if __name__=='__main__':run(Path(__file__).resolve().parent)
