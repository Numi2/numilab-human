"""Trace the retained source quadrature through both cavity mass allocations and current P1 regions."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from compile_connected_joint import digest
from solve_connected_carrier_flow import solve,neighbor_candidates

def survivor_map(index):
    size=int(index['child_begin'])+len(index['old_cavity_tets']);mask=np.ones(size,bool);mask[index['old_cavity_tets']]=False
    m=np.full(size,-1,'i4');m[mask]=np.arange(mask.sum());return m

def pair_credits(q,available,c,required,allowed,cost):
    # Normalize only the LP. Emitted credits keep their measured kilogram units.
    total=available.sum();ordered=np.argsort(cost,axis=None,kind='stable');a=available.copy();b=required.copy();i,j=np.unravel_index(ordered,cost.shape);keep=allowed[i,j];records=[]
    for u,v in zip(i[keep],j[keep]):
        mass=min(a[u],b[v])
        if mass>0:records.append((q[u],c[v],mass));a[u]-=mass;b[v]-=mass
    if max(a.sum(),b.sum())<=max(3e-14*total,1.e-24):return records
    u,v=np.nonzero(allowed)
    if not len(u):return None
    ids=np.arange(len(u));A=coo_matrix((np.ones(2*len(u)),(np.r_[u,len(q)+v],np.r_[ids,ids])),shape=(len(q)+len(c),len(u))).tocsr()
    sol=linprog(cost[u,v]/max(cost[u,v].max(),1e-24),A_eq=A,b_eq=np.r_[available,required]/total,bounds=(0,None),method='highs')
    if not sol.success:return None
    w=np.maximum(sol.x,0)*total
    if np.max(abs(A@w-np.r_[available,required]))>max(2e-12*total,1e-23):return None
    return [(q[a],c[b],w[i]) for i,(a,b) in enumerate(zip(u,v)) if w[i]>0]

def run(h,p66,job,out):
    out.mkdir();env=json.loads((job/'scene-material-p1/launch.json').read_bytes())['environment'];prefix='NUMI_H706_FRACTIONAL_'
    loc=h/'closed-native/native';wloc=h/'closed-native/source';wroot=p66/'registered-envelope-068';comp=h.parent/'composed-flow'
    wi=np.load(wloc/'material-source-indices.npz');si=np.load(loc/'material-source-indices.npz');W=survivor_map(wi);S=survivor_map(si)
    WM=np.loadtxt(wloc/'mesh.ele',skiprows=1,usecols=(5,),dtype='i4');A=np.loadtxt(loc/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    x=np.loadtxt(loc/'mesh.node',skiprows=1,usecols=(1,2,3));cent=x[A[:,:4]-1].mean(1);destmass=np.fromfile(loc/'cell-mass.f64','<f8');wrmass=np.fromfile(wloc/'cell-mass.f64','<f8')
    source=np.loadtxt(env[prefix+'SOURCE330_NODES'],delimiter=',',skiprows=1,usecols=(5,6,7));t0=np.loadtxt(env[prefix+'SOURCE330_TETS'],delimiter=',',skiprows=1,usecols=(3,4,5,6),dtype='i4')
    lab=np.loadtxt(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_PARENT_LABELS_CSV'],delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    radius=float(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M'])
    original=p66/'motion-repair-work/paired-p1';control=json.loads((comp/'result.json').read_bytes());output={}
    for tag,m in [('SKIN',15),('FAT',16)]:
        k=tag.lower();oldgraph=p66/'motion-repair-work/full-flow'/(tag+'-physical-flow.npz');cgraph=comp/'flows'/(tag+'-physical-flow.npz');wa=wroot/'guarded-mass'/(tag+'-mapping.npz')
        for file,key in [(oldgraph,'original_flow'),(cgraph,'reindexed'),(wa,'protected_wrist')]:
            if digest(file)!=control[tag][key]:raise ValueError('Original '+key+' identity differed')
        graph=dict(np.load(oldgraph));wr=dict(np.load(wa));sgraph=dict(np.load(cgraph));route=dict(np.load(h/'closed-flow'/(tag+'-transport.npz')))
        oldgeo=np.load(original/(k+'-role-geometry.npz'));bank=oldgeo['combined_child_rows'];oi=bank[graph['role_cell']];q0=graph['carrier'];kg0=graph['mass'];qmany=q0.max()+1
        replaced=W[oi]<0;prior=np.load(wroot/'wrist-allocation'/(tag+'-delta.npz'))
        if str(prior['base_flow_sha256'])!=digest(oldgraph):raise ValueError('Wrist protected flow mismatch')
        replaced[prior['original_edges']]=True;keep=np.flatnonzero(~replaced);edge=wr['source_edge'];external=wr['destination']<len(W)
        g=np.r_[W[oi[keep]],np.where(external,W[np.minimum(wr['destination'],len(W)-1)],int(wi['child_begin'])+wr['destination']-len(W))]
        q=np.r_[q0[keep],q0[edge]];mass=np.r_[kg0[keep],wr['mass_kg']]
        if np.any(g<0) or np.any(WM[g]!=m):raise ValueError('Invalid wrist incidence')
        wids=np.flatnonzero(WM==m);werr=np.abs(np.bincount(g,weights=mass,minlength=len(WM))[wids]-wrmass[wids])
        if np.any(werr>np.maximum(wrmass[wids]*3e-12,3e-19)):raise ValueError('Wrong reconstructed wrist masses')
        oldord=np.full(len(q0),-1,'i4');oldord[keep]=np.arange(len(keep))
        routeid=sgraph['wrist_route'];idx=np.where(routeid>=0,len(keep)+routeid,oldord[sgraph['original_edge']])
        if np.min(idx)<0 or len(np.unique(idx))!=len(idx) or not np.array_equal(q[idx],sgraph['carrier']) or not np.array_equal(g[idx],wids[sgraph['role_cell']]) or not np.array_equal(mass[idx],sgraph['mass']):raise ValueError('Cannot prove intermediate quadrature identity')
        retain=np.ones(len(g),bool);retain[idx]=False
        upstream=route['upstream_kg'];which=idx[route['source_edge']];inputq=q[which];pg=g[which];ext=S[pg]>=0
        if np.any(S[g[retain]]<0) or np.any(upstream<0) or np.any(route['downstream_kg']<0):raise ValueError('Lost upstream shoulder source')
        ids=np.flatnonzero(A[:,4]==m);local=np.full(len(A),-1,'i4');local[ids]=np.arange(len(ids))
        packet=dict(np.load(env[prefix+'CARRIERS_'+tag+'_NPZ']));par=np.repeat(packet['global_source330_tet_rows'],np.diff(packet['parent_offsets']))
        qpos=np.einsum('ij,ijk->ik',packet['carrier_barycentric'],source[t0[par]])
        p1=dict(np.load(env[prefix+'P1_'+tag+'_NPZ']));offsets=p1['cell_admissible_region_offsets'];regions=p1['cell_admissible_region_ids'];target_region=lab[par,2]
        sub=np.flatnonzero(~ext&(upstream>0));si0=np.argsort(pg[sub],kind='stable');sub=sub[si0];gc=route['parent'];maps=np.argsort(gc,kind='stable')
        allpar=np.union1d(pg[sub],gc[route['downstream_kg']>0]);extra=[];mismatch=[];bad=[]
        begin=int(si['child_begin']);ip=pg[sub];op=gc[maps];newcells=np.sort(np.unique(begin+route['child']));keys=[];ancestry=[]
        for a in allpar:
            sl=np.searchsorted(ip,a,side='left'),np.searchsorted(ip,a,side='right')
            sr=np.searchsorted(op,a,side='left'),np.searchsorted(op,a,side='right')
            inbound=sub[slice(*sl)];outs=maps[slice(*sr)];all_outputs=outs;outs=outs[route['downstream_kg'][outs]>0]
            qq,inv=np.unique(inputq[inbound],return_inverse=True);available=np.bincount(inv,weights=upstream[inbound],minlength=len(qq));child=int(si['child_begin'])+route['child'][outs];required=route['downstream_kg'][outs]
            if not len(outs) or not len(qq):
                bad.append(dict(parent=int(a),reason='one-sided group',kg=float(max(available.sum(),required.sum()))));continue
            l=local[child];elig=np.column_stack([np.isin(target_region[qq],regions[offsets[j]:offsets[j+1]]) for j in l]);cost=((qpos[qq,None,:]-cent[child][None,:,:])**2).sum(2)
            elig&=cost<=radius**2;delta=abs(available.sum()-required.sum());mismatch.append(delta)
            if delta>max(3e-12*required.sum(),1e-23):raise ValueError('Intermediate mass mismatch')
            cols=np.unique(begin+route['child'][all_outputs]);where=local[cols]
            ancestry.append((qq[:,None].astype('i8')*len(newcells)+np.searchsorted(newcells,cols)).ravel())
            geometric=np.column_stack([np.isin(target_region[qq],regions[offsets[j]:offsets[j+1]]) for j in where])
            geometric&=((qpos[qq,None,:]-cent[cols][None,:,:])**2).sum(2)<=radius**2
            u,v=np.nonzero(geometric)
            if len(u):keys.append(qq[u].astype('i8')*len(newcells)+np.searchsorted(newcells,cols[v]))
            rec=pair_credits(qq,available,child,required,elig,cost)
            if rec is None:
                bad.append(dict(parent=int(a),reason='no admissible source-to-recipient flow',kg=float(required.sum()),carriers=qq.tolist(),children=child.tolist(),source_regions=target_region[qq].tolist(),eligible_edges=int(elig.sum())));continue
            extra+=rec
        accepted=None;unmatched=bad
        if bad:
            target=np.bincount(np.searchsorted(newcells,begin+route['child']),weights=route['downstream_kg'],minlength=len(newcells))
            quota=np.bincount(inputq[sub],weights=upstream[sub],minlength=qmany)
            tentative,accepted=solve(h/'transport_material_mass',keys,quota,target,newcells,qmany,extra)
            direct=dict(accepted)
            if accepted['child_deficits'] or accepted['carrier_deficits']:
                history=[direct]
                for hops in (1,2):
                    neighbor_keys=neighbor_candidates(A[newcells,:4],ancestry,keys,qpos,target_region,cent[newcells],offsets,regions,local[newcells],radius,hops)
                    tentative,accepted=solve(h/'transport_material_mass',neighbor_keys,quota,target,newcells,qmany,extra)
                    accepted['allowed_material_face_hops']=hops
                    if not (accepted['child_deficits'] or accepted['carrier_deficits']):break
                    history.append(dict(accepted))
                accepted['earlier_restrictions']=history
                used=tentative[:,0].astype('i8')*len(newcells)+np.searchsorted(newcells,tentative[:,1].astype('i4'))
                outside=~np.isin(used,np.unique(np.concatenate(keys)))
                accepted['transport_through_adjacent_cells_kg']=float(tentative[outside,2].sum())
            if accepted['child_deficits']==0 and accepted['carrier_deficits']==0:extra=tentative;unmatched=[]
            print(tag,'joint',accepted,flush=True)
        raw=np.asarray(extra).reshape(-1,3);Q=np.r_[q[retain],inputq[ext],raw[:,0].astype('i4')];G=np.r_[S[g[retain]],S[pg[ext]],raw[:,1].astype('i4')];KG=np.r_[mass[retain],upstream[ext],raw[:,2]]
        active=KG>0;Q=Q[active];G=G[active];KG=KG[active]
        source_err=abs(np.bincount(Q,weights=KG,minlength=qmany)-np.bincount(q0,weights=kg0,minlength=qmany));target_err=abs(np.bincount(G,weights=KG,minlength=len(A))[ids]-destmass[ids])
        rec=dict(role=tag,old_flow=digest(oldgraph),new_skin_fat=digest(loc/'cell-mass.f64'),carrier_count=int(qmany),cells=len(ids),flow_edges=len(Q),maximum_intermediate_error_kg=max(mismatch,default=0),maximum_source_error_kg=float(source_err.max()),maximum_receiver_error_kg=float(target_err.max()),unresolved_parents=len(unmatched),independent_parent_failures=len(bad),joint_allocation=accepted,unallocated_kg=sum(a['kg'] for a in unmatched),witnesses=unmatched[:20])
        if not unmatched:
            if np.any(target_err>np.maximum(3e-12*destmass[ids],3e-19)) or source_err.max()>2e-12:raise ValueError('Composed conservation failed')
            f=out/(tag+'-physical-flow.npz');np.savez_compressed(f,carrier=Q.astype('i4'),role_cell=local[G],mass=KG);rec['flow_sha256']=digest(f)
        else:
            f=out/(tag+'-unmatched.json');f.write_text(json.dumps(bad)+'\n');rec['unmatched_sha256']=digest(f)
        output[tag]=rec;(out/'result.json').write_text(json.dumps(output,indent=2)+'\n');print(tag,len(bad),rec['unallocated_kg'],rec['maximum_receiver_error_kg'],flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
