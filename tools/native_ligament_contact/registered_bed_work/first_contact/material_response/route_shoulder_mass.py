"""Conserve quadrature mass through parent-to-child incidence without expanding every donor/child product."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack
from audit_registered_skeleton import digest
from registered_density import density_bounds
from supported_material_admission import geom

def run(job,mesh,domain,flow,edges,out):
    own=Path(json.loads((job/'launch.json').read_bytes())['compiled_material_owner']);mat=np.loadtxt(mesh/'mesh.ele',dtype='i4',skiprows=1);t=mat[:,1:5]-1;mat=mat[:,5];M=np.fromfile(mesh/'cell-mass.f64',dtype='<f8');x=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=[1,2,3]);z=np.load(domain);pos=z['x'].astype(float);tt=z['t'];ov=np.load(flow/'ancestry/overlap.npz');support=np.load(edges/'flows/SKIN-physical-flow.npz')
    if digest(domain)!=json.loads((flow/'ancestry/result.json').read_bytes())['mesh']:raise ValueError('Unmatched emitted volume')
    out.mkdir();report={}
    for m,tag in [(15,'SKIN'),(16,'FAT')]:
        a=np.load(edges/'flows'/(tag+'-physical-flow.npz'));f=np.load(edges/'allocation'/(tag+'-delta.npz'))
        bank=np.flatnonzero(mat==m);parent=bank[a['role_cell']];q=a['carrier'].copy();w=a['mass'];free=np.zeros(len(w),bool);free[f['original_edges']]=True;q[~free]=q.max()+1+np.arange(sum(~free))
        inside=np.isin(parent,ov['parent_ids']);O=np.unique(parent[inside]);external=np.unique(parent[~inside]);ch=np.flatnonzero(z['role']==m)
        use=np.isin(ov['child'],ch);op=ov['parent'][use];dest=ov['child'][use];P=len(O);E=len(external);C=len(ch)
        if np.any(~np.isin(op,O)):
            # Geometric parents with zero source mass can own zero-quantity edges.
            valid=np.isin(op,O);op,dest=op[valid],dest[valid]
        if np.any(~np.isin(ch,dest)):raise ValueError('Missing material route')
        lo,hi=density_bounds(own,m);cv=ov['child_volume_mm3'][ch]*1e-9;ev=geom(x,t[external])[0];volume=np.r_[ev,cv]
        spent=np.bincount(np.searchsorted(external,parent[~inside]),weights=w[~inside],minlength=E)
        if np.any(spent>M[external]+2e-14):raise ValueError('Negative reserved mass')
        reserve=np.r_[np.maximum(M[external]-spent,0),np.zeros(C)];floor=np.maximum(0,lo*volume-reserve);cap=np.maximum(0,hi*volume-reserve)
        if np.any(hi*volume-reserve< -np.maximum(hi*volume*3e-12,1e-23)):raise ValueError('Protected external tissue exceeds capacity')
        # Pinned old source weights may only flow through geometric or individually advected material ancestry.
        qid,qq=np.unique(q,return_inverse=True);quota=np.bincount(qq,weights=w);pi=np.searchsorted(O,op);cj=np.searchsorted(ch,dest);pj=np.searchsorted(O,parent[inside]);ej=np.searchsorted(external,parent[~inside]);parent_cap=np.bincount(pi,weights=cap[E+cj],minlength=P)
        born=np.zeros(len(w));born[inside]=parent_cap[pj];born[~inside]=cap[ej]
        head=np.minimum(quota[qq],born);tail=np.minimum(parent_cap[pi],cap[E+cj]);good=head>0;v1=np.flatnonzero(good);v2=np.flatnonzero(tail>0)
        scale=np.r_[head[v1],tail[v2]];n1=len(v1);n=len(scale)
        qmax=np.bincount(qq,weights=head,minlength=len(quota));cmax=np.bincount(cj,weights=np.minimum(tail,np.bincount(pj,weights=head[inside],minlength=P)[pi]),minlength=C)
        # Capacity auditing uses the physical inclusive limits for every receiver.
        bad_donor=np.flatnonzero(qmax+1e-18<quota);bad_cell=np.flatnonzero(cmax+1e-18<floor[E:])
        summary=dict(donors=len(quota),parents=P,recipients=E+C,variables=n,dead_end_donors=len(bad_donor),undersupplied_children=ch[bad_cell].tolist(),unreachable_source_mass_kg=float(np.maximum(quota-qmax,0).sum()),worst_child_lower_bound_deficit_kg=float(np.maximum(floor[E:]-cmax,0).max(initial=0.)))
        summary['total_kg']={'supply':float(quota.sum()),'minimum':float(floor.sum()),'maximum':float(cap.sum())}
        report[tag]=summary;(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(tag,summary,flush=True)
        from measure_mass_cut import check
        missing=check(Path(__file__).with_name('mass_capacity_cut'),out/(tag+'-capacity.json'),qid,quota,O,external,ch,qq,inside,P,E,op,dest,parent,floor,cap)
        summary['minimum_cut_deficit_kg']=missing
        from construct_mass_transfer import solve
        center=pos[tt[ch]].mean((0,1));cent=np.vstack((x[t[external]].mean(1),pos[tt[ch]].mean(1)))-center
        wanted=M[O]@(x[t[O]].mean(1)-center)+spent@cent[:E]
        rawweights=ov['volume_mm3'][use]
        if len(rawweights)!=len(op):rawweights=rawweights[valid]
        numerical=solve(out,tag,qq,quota,w,inside,parent,O,external,ch,op,dest,rawweights,volume,reserve,lo,hi,cent,wanted)
        summary['constructive_transfer']=numerical
        if numerical['passed'] or numerical['kg_deficit']>1e-17 or len(bad_donor) or len(bad_cell) or missing>1e-11:
            (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');continue
        h_in=np.flatnonzero(inside[v1]);h_out=np.flatnonzero(~inside[v1]);p_in=np.searchsorted(O,parent[v1[h_in]]);p_out=pi[v2]
        pnorm=np.maximum(np.maximum(M[O],np.bincount(p_in,weights=scale[h_in],minlength=P)),1e-25)
        # Count every active source, including donors with zero quota.
        norm=np.maximum(quota,1e-25);ri=np.r_[qq[v1],len(quota)+p_in,len(quota)+p_out];ci=np.r_[np.arange(n1),h_in,n1+np.arange(len(v2))];val=np.r_[scale[:n1]/norm[qq[v1]],scale[h_in]/pnorm[p_in],-tail[v2]/pnorm[p_out]]
        eq=coo_matrix((val,(ri,ci)),shape=(len(quota)+P,n)).tocsr();rhs=np.r_[quota/norm,np.zeros(P)]
        dst=np.r_[np.searchsorted(external,parent[v1[h_out]]),E+cj[v2]];col=np.r_[h_out,n1+np.arange(len(v2))];rn=np.maximum(cap,1e-25);recv=coo_matrix((scale[col]/rn[dst],(dst,col)),shape=(E+C,n)).tocsr()
        center=pos[tt[ch]].mean((0,1));cent=np.vstack((x[t[external]].mean(1),pos[tt[ch]].mean(1)))-center
        wanted=M[O]@(x[t[O]].mean(1)-center)+spent@cent[:E]
        mom=coo_matrix(((cent[dst]*scale[col,None]).T.ravel(),(np.repeat(np.arange(3),len(col)),np.tile(col,3))),shape=(3,n)).tocsr()
        mn=np.maximum(np.array(abs(mom).sum(1)).ravel(),1e-16);with_m=vstack((eq,mom.multiply(1/mn[:,None]))).tocsr()
        # The costs prefer the original donor and the larger true material overlaps.
        price=np.zeros(n);price[:n1]=scale[:n1]/max(scale.max(),1e-22)*(1-np.minimum(1,w[v1]/scale[:n1]))
        kw=dict(A_ub=vstack((recv,-recv),format='csr'),b_ub=np.r_[cap/rn,-floor/rn],bounds=(0,1),method='highs',options={'time_limit':100,'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
        for moments in (True,False):
            sol=linprog(price,A_eq=with_m if moments else eq,b_eq=np.r_[rhs,wanted/mn] if moments else rhs,**kw);summary['with_moments' if moments else 'capacity_only']=sol.message
            if sol.success:break
        if sol.success:
            mflow=scale*np.maximum(sol.x,0);recv_mass=np.bincount(dst,weights=mflow[col],minlength=E+C);rho=(recv_mass+reserve)/volume
            derr=abs(np.bincount(qq[v1],weights=mflow[:n1],minlength=len(quota))-quota)
            balance=np.bincount(p_in,weights=mflow[h_in],minlength=P)-np.bincount(p_out,weights=mflow[n1:],minlength=P)
            moment_error=recv_mass@cent-wanted
            checked=bool(np.all(derr<=1e-9*np.maximum(quota,1e-23)) and np.max(abs(balance))<1e-14 and np.linalg.norm(moment_error,np.inf)<1e-11 and rho.min()>=lo-1e-8 and rho.max()<=hi+1e-8)
            summary.update(passed=checked,max_source_error_kg=float(derr.max()),max_parent_error_kg=float(max(abs(balance))),mass_error_kg=float(recv_mass.sum()-quota.sum()),moment_error_kg_m=moment_error.tolist(),density_range=[float(rho.min()),float(rho.max())])
            np.savez_compressed(out/(tag+'-mapping.npz'),upstream=v1,upstream_kg=mflow[:n1],material_parent=op[v2],child=dest[v2],downstream_kg=mflow[n1:],external=external,external_mass_kg=(recv_mass+reserve)[:E],child_ids=ch,child_mass_kg=recv_mass[E:])
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(tag,summary,flush=True)
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
