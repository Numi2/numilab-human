"""Solve quadrature allocation as a warm-started, fixed-point circulation in units of 2^-120 kg."""
from pathlib import Path
import json,subprocess
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack

def solve(out,tag,qq,quota,w,inside,parent,O,external,ch,op,dest,weights,volume,reserve,lo,hi,cent,wanted):
    Q=len(quota);P=len(O);E=len(external);C=len(ch);K=Q+P;T=K+E+C;S=T+1
    pi=np.searchsorted(O,parent[inside]);p=np.searchsorted(O,op);j=np.searchsorted(ch,dest)
    to=np.where(inside,Q+np.searchsorted(O,parent),K+np.searchsorted(external,parent));ii=np.arange(len(w))
    total=float(quota.sum());parent_mass=np.bincount(pi,weights=w[inside],minlength=P)
    parent_area=np.bincount(p,weights=weights,minlength=P)
    initial=parent_mass[p]*weights/parent_area[p];end=np.r_[np.bincount(to[~inside]-K,weights=w[~inside],minlength=E),np.bincount(j,weights=initial,minlength=C)]
    L=np.maximum(0,lo*volume-reserve);H=np.maximum(0,hi*volume-reserve);guard=(H-L)*1.e-7
    # An unchanged allocation at a physical density limit is already admissible.
    # Protect rounding only to the extent that the previous allocation permits it.
    L+=np.minimum(guard,np.maximum(end-L,0.));H-=np.minimum(guard,np.maximum(H-end,0.))
    want=np.clip(end,L,H)
    coords=np.arange(T-K);a=np.r_[np.full(Q,S),qq,Q+p,K+coords,T];b=np.r_[np.arange(Q),to,K+E+j,np.full(E+C,T),S]
    low=np.r_[quota,np.zeros(len(w)+len(p)),L,0.];high=np.r_[quota,np.minimum(quota[qq],total),H[E+j],H,total*2.]
    seed=np.r_[quota,w,initial,want,total];buf=np.rec.fromarrays([a.astype('<i4'),b.astype('<i4'),low,high,seed],names='a,b,l,h,w')
    exe=Path(__file__).with_name('transport_material_mass')
    proc=subprocess.run([str(exe.resolve())],input=np.array([S+1,len(a)],'<i4').tobytes()+buf.tobytes(),capture_output=True,check=True)
    flow=np.frombuffer(proc.stdout,'<f8');assert len(flow)==len(a)+1;deficit=flow[0]
    up=flow[1+Q:1+Q+len(w)].copy();down=flow[1+Q+len(w):1+Q+len(w)+len(p)].copy()
    def measure():
        received=np.r_[np.bincount(to[~inside]-K,weights=up[~inside],minlength=E),np.bincount(j,weights=down,minlength=C)]
        ferr=np.max(abs(np.bincount(qq,weights=up,minlength=Q)-quota)/np.maximum(quota,1.e-25))
        mass=(received+reserve);rho=mass/volume;perr=np.max(abs(np.bincount(pi,weights=up[inside],minlength=P)-np.bincount(p,weights=down,minlength=P)))
        return received,dict(kg_deficit=float(deficit),max_source_relative_error=float(ferr),max_parent_error_kg=float(perr),density_range=[float(rho.min()),float(rho.max())],first_moment_kg_m=(received@cent-wanted).tolist(),mass_error_kg=float(received.sum()-quota.sum()))
    received,stat=measure();rec=dict(initial=stat.copy())
    # Restore all three first moments through equal-and-opposite transfers between
    # existing external destinations of the same quadrature carrier.
    if deficit<1e-17 and max(abs(received@cent-wanted))>=1e-11:
        eids=np.flatnonzero(~inside);oid=to[eids]-K;byq=np.argsort(qq[eids],kind='stable');groups=np.split(byq,np.flatnonzero(np.diff(qq[eids][byq]))+1)
        src=[];dst=[];lim=[]
        give=np.maximum(0,received-L);take=np.maximum(0,H-received);margin=np.maximum(0,(H-L)*1e-8)
        for group in groups:
            gi=oid[group];cap0=np.minimum(up[eids[group]],np.maximum(0,give[gi]-margin[gi]));cap1=np.maximum(0,take[gi]-margin[gi])
            A=group[np.argsort(-cap0)[:3]];B=group[np.argsort(-cap1)[:5]]
            for i in A:
                for k in B:
                    bound=min(up[eids[i]],max(0,give[oid[i]]-margin[oid[i]]),max(0,take[oid[k]]-margin[oid[k]]))
                    if i!=k and bound>1e-14:src.append(i);dst.append(k);lim.append(bound)
        if src:
            src=np.asarray(src);dst=np.asarray(dst);lim=np.asarray(lim);ii=np.arange(len(lim));dd=oid[dst];ss=oid[src]
            # Per-source-edge withdrawals cannot exceed currently allocated mass.
            donor,arr=np.unique(src,return_inverse=True);supply=up[eids[donor]]
            debit=coo_matrix((lim/supply[arr],(arr,ii)),shape=(len(supply),len(ii)))
            ext,rr=np.unique(np.r_[dd,ss],return_inverse=True);den=np.maximum(H[ext],1e-25)
            signed=coo_matrix((np.r_[lim,-lim]/den[rr],(rr,np.tile(ii,2))),shape=(len(ext),len(ii)))
            goal=wanted-received@cent;moment=(cent[dd]-cent[ss]).T*lim;sc=np.maximum(abs(moment).max(1),1e-16)
            corr=linprog(lim/lim.max(),A_eq=moment/sc[:,None],b_eq=goal/sc,A_ub=vstack((debit,signed,-signed)).tocsr(),b_ub=np.r_[np.ones(len(supply)),np.maximum(0,take[ext]-margin[ext])/den,np.maximum(0,give[ext]-margin[ext])/den],bounds=(0,1),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'time_limit':60})
            rec['moment_solve']=corr.message
            if corr.success:
                val=corr.x*lim;np.add.at(up,eids[src],-val);np.add.at(up,eids[dst],val)
                if np.min(up)<-1e-16:raise ValueError('Negative routed source mass')
                np.maximum(up,0,out=up);received,stat=measure()
    rho=stat['density_range']
    stat['passed']=bool(deficit<1e-17 and stat['max_source_relative_error']<1e-9 and stat['max_parent_error_kg']<1e-14 and max(map(abs,stat['first_moment_kg_m']))<1e-11 and rho[0]>=lo-1e-8 and rho[1]<=hi+1e-8)
    rec['final']=stat;(out/(tag+'-transport.json')).write_text(json.dumps(rec,indent=2)+'\n')
    np.savez_compressed(out/(tag+'-transport.npz'),source_edge=np.arange(len(up)),quadrature_group=qq,upstream_kg=up,parent=op,child=dest,downstream_kg=down,external=external,external_mass_kg=received[:E]+reserve[:E],child_ids=ch,child_mass_kg=received[E:])
    return stat
