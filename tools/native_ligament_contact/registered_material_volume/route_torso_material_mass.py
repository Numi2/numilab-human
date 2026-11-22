"""Reassign actual native carrier masses after a connected, fixed-topology reference registration."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from emit_skin_source_registration import sha
from torso_cell_geometry import geom
def main():
    p=argparse.ArgumentParser()
    for k in ('baseline','reg','inputs','geometry','out'):p.add_argument(k,type=Path)
    p.add_argument('--region-support',type=Path);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    base,reg,out=args.baseline,args.reg,args.out
    material=reg/'material';mr=json.loads((reg/'fine-mass-routed/mass-transfer.json').read_text())
    flow_file=reg/'fine-mass-routed/current-density-closed-flow.npz';assert sha(flow_file)==mr['new_flow_sha256']
    flow=dict(np.load(flow_file));wr=json.loads((material/'witnesses/receipt.json').read_text())
    assert wr['verified_flow_sha256']==sha(flow_file)
    p0=reg/'current-parent-capture/compiled-parent-cell-mass.csv';parent=np.loadtxt(p0,delimiter=',',skiprows=1)
    node=np.loadtxt(str(p0)+'.nodes.csv',delimiter=',',skiprows=1);ct=np.loadtxt(str(p0)+'.tets.csv',delimiter=',',skiprows=1)
    labels=np.loadtxt(str(p0)+'.labels.csv',delimiter=',',skiprows=1,dtype='i4')
    src=[];quota=[];par=[]
    for row in wr['roles']:
        loc=Path(row['carrier_path']);assert sha(loc)==row['carrier_sha256'];c=np.load(loc);pp=np.repeat(c['global_source330_tet_rows'],np.diff(c['parent_offsets']));par.extend(pp)
        src.extend(np.einsum('ni,nij->nj',c['carrier_barycentric'],node[ct[pp,3:7].astype('i4'),5:8]))
        quota.extend(c['carrier_weights']*parent[pp,2])
    src,quota,par=np.asarray(src),np.asarray(quota),np.asarray(par)
    assert np.array_equal(par,flow['source_parent'])
    old_mesh=base/'constitutive-witness-repaired-tetgen';paths=[old_mesh/'mesh.node',old_mesh/'mesh.ele',base/'refit-motion-qualification/constitutive-witness-repaired.volumes',base/'close_live_parent_density_compiled']
    for f in paths:assert sha(f)==mr['files'][str(f)]
    with paths[0].open() as f:z=int(next(f).split()[0]);x=np.loadtxt(f,usecols=(1,2,3),max_rows=z)
    with paths[1].open() as f:z=int(next(f).split()[0]);el=np.loadtxt(f,usecols=(1,2,3,4,5),max_rows=z,dtype='i4')
    t=el[:,:4]-1;role=el[:,4];s=np.load(args.inputs);proposal=np.load(args.geometry);ix=s['cell']
    assert np.array_equal(t[ix],s['nodes'][s['t']]) and np.array_equal(x[s['nodes']].astype('f4'),s['x'])
    y=proposal['x'].astype('f8');x[s['nodes']]=y
    assert np.array_equal(y[s['fix']],s['x'][s['fix']]);vnew,qnew=geom(y,s['t'])
    if min(vnew)<=1e-18 or min(qnew)<1e-5:raise ValueError('Candidate failed emitted tetrahedron thresholds')
    vol=np.fromfile(paths[2],dtype='f8');vol[ix]=vnew
    center=sum(x[t[:,i]]*.25 for i in range(4))
    g=flow['group_for_cell'];carrier=flow['carrier_row'].copy();receiver=flow['group'].copy();prev=flow['mass_flow_kg'].copy();hist=flow['archived_compensation'].astype(bool)
    archived_cap=prev.copy();limit=mr['distance_bound_m']
    group_size=np.bincount(g);starts=np.r_[0,np.cumsum(group_size)];order=np.argsort(g,kind='stable');ng=len(group_size)
    max_distance=np.zeros(len(carrier))
    for start in range(0,len(carrier),40000):
        end=min(start+40000,len(carrier));rr=receiver[start:end];nn=group_size[rr]
        i=np.repeat(np.arange(len(rr)),nn);off=np.r_[0,np.cumsum(nn)[:-1]]
        cell=order[starts[rr[i]]+np.arange(len(i))-off[i]]
        max_distance[start:end]=np.maximum.reduceat(np.linalg.norm(center[cell]-src[carrier[start:end][i]],axis=1),off)
    eligible=max_distance<=limit
    from torso_current_region_support import allowed
    support=None
    if args.region_support is not None:
        support=np.load(args.region_support);bits=support['group_regions'];source_region=support['source_region'];assert np.array_equal(source_region,labels[par,3])
        # Pre-existing capped compensation has its own authenticated two-hop
        # parent/child proof. Its recorded budget can only decrease. Every
        # ordinary or newly created edge requires current one-hop admission.
        eligible &= (allowed(bits,receiver,source_region[carrier]) | hist)
    old_carrier=carrier.copy();old_receiver=receiver.copy()
    # Reuse only routes verified for their current children. Compensation rows
    # can release material but cannot receive additional mass.
    flow_now=np.where(eligible,prev,0.);lost=quota-np.bincount(carrier,weights=flow_now,minlength=len(quota))
    ordinary=eligible&~hist;room=np.bincount(carrier,weights=ordinary*prev,minlength=len(quota))
    flow_now[ordinary]+=prev[ordinary]*np.divide(lost,room,out=np.zeros_like(lost),where=room>0)[carrier[ordinary]]
    flow_now=np.maximum(0,flow_now)
    kind=np.unique(labels[:,1:4],axis=0,return_inverse=True)[1].ravel()[par]
    orphan=np.flatnonzero((lost>1e-13)&(room==0));initial_peer=[];new_c=[];new_r=[];new_m=[];new_d=[]
    centroids=np.column_stack([np.bincount(g,weights=center[:,k])/group_size for k in range(3)])
    for k in np.unique(kind[orphan]):
        donor_rows=np.flatnonzero((kind[carrier]==k)&(prev>0)&~hist)
        destinations,first=np.unique(receiver[donor_rows],return_index=True);peer=donor_rows[first]
        if not len(first):raise ValueError('Carrier tuple has no physically supported recipient')
        tree=cKDTree(centroids[destinations])
        for c in orphan[kind[orphan]==k]:
            # A verified zero-valued edge can receive material under the same
            # ordinary routing rules as an existing positive edge.
            existing=np.flatnonzero((carrier==c)&ordinary)
            if len(existing):
                selected=existing[np.argmin(max_distance[existing])];flow_now[selected]+=lost[c];continue
            candidates=tree.query_ball_point(src[c],limit)
            for j in sorted(candidates,key=lambda j:np.linalg.norm(centroids[destinations[j]]-src[c])):
                r=destinations[j];ids=order[starts[r]:starts[r+1]]
                dist=float(np.linalg.norm(center[ids]-src[c],axis=1).max())
                if dist>limit or np.any((carrier==c)&(receiver==r)):continue
                if support is not None and not allowed(bits,r,source_region[c]):continue
                new_c.append(c);new_r.append(r);new_m.append(lost[c]);new_d.append(dist);initial_peer.append(peer[j]);break
            else:raise ValueError(('Carrier lacks typed reachable current recipient',int(c),int(k),float(lost[c])))
    carrier=np.r_[carrier,np.array(new_c,dtype=carrier.dtype)];receiver=np.r_[receiver,np.array(new_r,dtype=receiver.dtype)]
    flow_now=np.r_[flow_now,new_m];max_distance=np.r_[max_distance,new_d]
    eligible=np.r_[eligible,np.ones(len(new_c),bool)];hist=np.r_[hist,np.zeros(len(new_c),bool)]
    low=np.where(role==15,900.,850.);high=np.where(role==15,1100.,950.)
    reserve=3e-10
    lo=np.bincount(g,weights=(low+reserve)*vol);hi=np.bincount(g,weights=(high-reserve)*vol)
    kind=np.unique(labels[:,1:4],axis=0,return_inverse=True)[1].ravel()[par]
    by_kind={int(k):(np.flatnonzero(kind==k),cKDTree(src[kind==k])) for k in np.unique(kind)}
    recv_sort=np.argsort(old_receiver,kind='stable');recv_starts=np.r_[0,np.cumsum(np.bincount(old_receiver,minlength=ng))]
    direct_by_kind={};direct_by_receiver={};native_cell_labels=None
    if support is not None:
        # Use actual, newly authenticated direct P1 slots as source-tuple
        # witnesses. The first candidate at a receiver need not have carried
        # positive mass in the previous (different) registration.
        state_dir=args.region_support.parent.parent/'native-inputs/material/p1'
        if not state_dir.is_dir():raise ValueError('Missing native P1 state for new route support')
        if sha(state_dir/'current-paired-volume-state.json')!=str(support['current_state_sha256']):raise ValueError('P1 source changed')
        all_tuples=np.unique(labels[:,1:4],axis=0);lookup={tuple(q):i for i,q in enumerate(all_tuples)}
        for role_name in ('skin','fat'):
            rec=json.loads((state_dir/(role_name+'-p1-receipt.json')).read_text());assert sha(Path(rec['p1_path']))==rec['p1_sha256']
            qp=np.load(rec['p1_path']);geometry=np.load(rec['role_geometry_path']);children=geometry['combined_child_rows']
            cell=np.repeat(children,np.diff(qp['cell_group_offsets']));types=np.array([lookup[tuple(k)] for k in qp['group_tuples']])[qp['cell_group_ids']]
            codes=types.astype('i8')*ng+g[cell];key,first=np.unique(codes,return_index=True)
            for code,child in zip(key,cell[first]):
                typ,j=divmod(int(code),ng);direct_by_kind.setdefault(typ,{})[j]=int(child);direct_by_receiver.setdefault(j,{})[typ]=int(child)
    history=[];new_route_source=list(initial_peer);test_hash=sha(args.geometry)
    for round_id in range(5):
        e=np.flatnonzero(eligible);inp=out/'routing-input.bin';sol=out/'routing-solution.bin';cut=out/'routing-cut.bin'
        with inp.open('wb') as f:
            for a,dt in [([len(quota),ng,len(e)],'u4'),(carrier[e],'u4'),(receiver[e],'u4'),(flow_now[e],'f8'),(quota,'f8'),(lo,'f8'),(hi,'f8'),(hist[e],'u1')]:np.asarray(a,dtype=dt).tofile(f)
        run=subprocess.run([paths[3],inp,sol,cut],capture_output=True,text=True,check=True)
        flow_now[:]=0;flow_now[e]=np.fromfile(sol,'f8')
        assigned=np.bincount(receiver,weights=flow_now,minlength=ng);delta=np.maximum(lo-assigned,assigned-hi);bad=np.flatnonzero(delta>1e-15)
        line=dict(iteration=round_id,unmet_groups=len(bad),unmet_kg=float(np.maximum(delta,0).sum()),solver=run.stdout.strip())
        history.append(line);print(line['iteration'],line['unmet_groups'],line['unmet_kg'],flush=True)
        if not len(bad):break
        # Supply additional destinations with the exact material, donor, region
        # witness already present at this receiver. All represented child
        # centers must satisfy the same production transport distance.
        existing=set(zip(carrier.tolist(),receiver.tolist()))
        requests=set(map(int,bad))
        if np.any(assigned[bad]>hi[bad]):
            # Allow nearby same-tuple receivers to absorb oversupply.
            centroids=np.column_stack([np.bincount(g,weights=center[:,k])/group_size for k in range(3)])
            tree=cKDTree(centroids)
            for j in bad[assigned[bad]>hi[bad]]:
                requests.update(map(int,tree.query(centroids[j],k=24)[1]))
        nc=[];nr=[];nd=[];peer=[]
        if support is not None:
            # Upper-capacity cuts need outgoing routes for the actual carriers
            # trapped at a saturated recipient. More incoming routes do not
            # create a path out of that cut.
            centers_by_group=np.column_stack([np.bincount(g,weights=center[:,axis])/group_size for axis in range(3)])
            upper_groups=bad[assigned[bad]>hi[bad]]
            donor_rows=np.flatnonzero(np.isin(receiver,upper_groups)&(flow_now>1e-16))
            for car in np.unique(carrier[donor_rows]):
                key=kind[car];witness=np.flatnonzero((kind[old_carrier]==key)&(prev>0)&~hist[:len(prev)])
                targets,first=np.unique(old_receiver[witness],return_index=True)
                ok=(hi[targets]-assigned[targets]>1e-15)&allowed(bits,targets,np.full(len(targets),source_region[car],dtype='i4'))
                targets,first=targets[ok],first[ok]
                provenance={int(j):int(witness[i]) for j,i in zip(targets,first)}
                for j,child in direct_by_kind.get(int(key),{}).items():
                    if hi[j]-assigned[j]>1e-15 and allowed(bits,j,source_region[car]):
                        provenance.setdefault(j,-child-1)
                targets=np.fromiter(provenance,dtype='i4')
                order_distance=np.argsort(np.linalg.norm(centers_by_group[targets]-src[car],axis=1),kind='stable')
                admitted=0
                for ix in order_distance:
                    j=targets[ix];edge=(int(car),int(j))
                    if edge in existing:continue
                    child=order[starts[j]:starts[j+1]];dist=float(np.linalg.norm(center[child]-src[car],axis=1).max())
                    if dist>limit:continue
                    existing.add(edge);nc.append(int(car));nr.append(j);nd.append(dist);peer.append(provenance[int(j)])
                    admitted+=1
                    if admitted>=32:break
        if support is not None and np.any(assigned[bad]<lo[bad]):
            # Find augmenting edges across the exact residual min-cut. The
            # source and every target child must agree in material and body;
            # the source region must be admissible in every child's one-hop row.
            with cut.open('rb') as f:
                residual=np.fromfile(f,'f8',count=ng);reach=np.fromfile(f,'u1',count=len(quota)+ng+2)
            reachable=reach[:len(quota)]!=0
            if native_cell_labels is None:
                labels_file=base/'paired-floor-child.labels.csv'
                assert sha(labels_file)==json.loads((base/'paired-floor-wall-lineage.json').read_text())['labels_sha256']
                native_cell_labels=np.loadtxt(labels_file,delimiter=',',skiprows=1,usecols=(1,2),dtype='i4')
            cache={}
            for j in bad[assigned[bad]<lo[bad]]:
                child=order[starts[j]:starts[j+1]];md=np.unique(native_cell_labels[child],axis=0)
                if len(md)!=1:continue
                mat,donor=map(int,md[0]);key=(mat,donor,tuple(map(int,bits[j])))
                if key not in cache:
                    use=np.flatnonzero(reachable&(labels[par,1]==mat)&(labels[par,2]==donor)&allowed(bits,np.full(len(par),j),source_region))
                    cache[key]=(use,cKDTree(src[use]) if len(use) else None)
                use,tree=cache[key]
                if not len(use):continue
                mid=center[child].mean(0);_,near=tree.query(mid,k=min(32,len(use)))
                for car in use[np.atleast_1d(near)]:
                    edge=(int(car),int(j))
                    if edge in existing:continue
                    dist=float(np.linalg.norm(center[child]-src[car],axis=1).max())
                    if dist>limit:continue
                    existing.add(edge);nc.append(int(car));nr.append(int(j));nd.append(dist)
                    # The separate negative range stores a native child donor
                    # witness. The source region comes from the authenticated CSR.
                    peer.append(-len(t)-int(child[0])-1)
        for j in sorted(requests):
            ids=recv_sort[recv_starts[j]:recv_starts[j+1]];ids=ids[(prev[ids]>0)&~hist[ids]]
            cc=order[starts[j]:starts[j+1]];mid=center[cc].mean(0)
            for k in np.union1d(np.unique(kind[old_carrier[ids]]),list(direct_by_receiver.get(j,{}))).astype('i4'):
                candidates,tree=by_kind[int(k)];_,ri=tree.query(mid,k=min(len(candidates),16*(round_id+1)))
                for car in candidates[np.atleast_1d(ri)]:
                    edge=(int(car),j)
                    if edge in existing:continue
                    dist=float(np.linalg.norm(center[cc]-src[car],axis=1).max())
                    if dist>limit:continue
                    if support is not None and not allowed(bits,j,source_region[car]):continue
                    existing.add(edge);nc.append(int(car));nr.append(j);nd.append(dist);eligible_peers=ids[kind[old_carrier[ids]]==k];peer.append(int(eligible_peers[0]) if len(eligible_peers) else -direct_by_receiver[j][int(k)]-1)
        if not nc:break
        new_route_source.extend(peer);carrier=np.r_[carrier,np.asarray(nc,dtype=carrier.dtype)];receiver=np.r_[receiver,np.asarray(nr,dtype=receiver.dtype)]
        eligible=np.r_[eligible,np.ones(len(nc),bool)];hist=np.r_[hist,np.zeros(len(nc),bool)];max_distance=np.r_[max_distance,nd];flow_now=np.r_[flow_now,np.zeros(len(nc))]
    assigned=np.bincount(receiver,weights=flow_now,minlength=ng);delta=np.maximum(lo-assigned,assigned-hi);bad=np.flatnonzero(delta>1e-15);err=float(np.max(abs(np.bincount(carrier,weights=flow_now,minlength=len(quota))-quota)))
    report=dict(recipient_partition=None,one_hop_support_sha256=sha(args.region_support) if support is not None else None,geometry_sha256=test_hash,baseline_flow_sha256=sha(flow_file),source_carriers=len(quota),source_parent_rows=len(parent),full_mesh_cells=len(t),new_typed_routes=len(new_route_source),retained_routes=len(old_carrier),invalidated_distance_routes=int(sum(~eligible)),maximum_used_route_distance_m=float(max(max_distance[flow_now>0])),admitted_route_distance_m=limit,source_mass_kg=float(quota.sum()),maximum_carrier_mass_error_kg=err,unmet_groups=len(bad),history=history)
    if not len(bad):
        if support is not None:assert np.all(allowed(bits,receiver[flow_now>0],source_region[carrier[flow_now>0]])|hist[flow_now>0])
        old_count=len(old_carrier);new_used=flow_now[old_count:]>0
        keep=np.r_[np.ones(old_count,bool),new_used];new_route_source=list(np.asarray(new_route_source,dtype='i4')[new_used])
        carrier,receiver,hist,flow_now,max_distance=carrier[keep],receiver[keep],hist[keep],flow_now[keep],max_distance[keep]
        report['used_new_typed_routes']=len(new_route_source)
    np.savez_compressed(out/'flow-witnesses.npz',unmet_group=bad,delta_kg=delta[bad],supplied_kg=assigned[bad],minimum_kg=lo[bad],maximum_kg=hi[bad],new_route_peer=np.maximum(np.asarray(new_route_source,dtype='i4'),-1),direct_p1_child=np.where(np.asarray(new_route_source)<-len(t),-1,np.maximum(-np.asarray(new_route_source,dtype='i4')-1,-1)),same_donor_one_hop_child=np.maximum(-np.asarray(new_route_source,dtype='i4')-len(t)-1,-1))
    if not len(bad):
        assert err<1e-14 and np.all(flow_now>=0) and np.all(flow_now[:len(prev)][flow['archived_compensation']]<=archived_cap[flow['archived_compensation']]+1e-17)
        # Keep every unchanged mass where possible, then make each recipient's
        # mass sum equal its sourced allocation strictly within its density range.
        em=np.clip(flow['element_mass_kg'],(low+reserve)*vol,(high-reserve)*vol)
        remain=assigned-np.bincount(g,weights=em);room=np.where(remain[g]>=0,(high-reserve)*vol,(low+reserve)*vol)-em
        capacity=np.bincount(g,weights=room)
        em+=room*np.clip(np.divide(remain,capacity,out=np.zeros_like(remain),where=capacity!=0),0,1)[g]
        em*=np.divide(assigned,np.bincount(g,weights=em))[g];rho=em/vol
        outside=(rho<low-1e-8)|(rho>high+1e-8);assert not outside.any()
        flow.update(carrier_row=carrier,group=receiver,archived_compensation=hist,mass_flow_kg=flow_now,element_mass_kg=em,nodal_mass_kg=sum(np.bincount(t[:,i],weights=em*.25,minlength=len(x)) for i in range(4)),prior_edge_row=np.r_[flow['prior_edge_row'],np.full(len(new_route_source),-1,dtype=flow['prior_edge_row'].dtype)])
        dst=out/'current-density-closed-flow.npz';np.savez_compressed(dst,**flow)
        report.update(flow_sha256=sha(dst),density_outliers=int(sum(outside)),result_mass_kg=float(em.sum()),nodal_mass_kg=float(flow['nodal_mass_kg'].sum()))
    (out/'mass-transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    if len(bad):sys.exit(1)
if __name__=='__main__':main()
