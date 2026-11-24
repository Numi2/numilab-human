"""Transport prior source quadrature through measured rest-volume intersections and certify recipients."""
import json, os, sys, subprocess
from pathlib import Path
import numpy as np
from balance_cavity_credits import rebalance
from compile_connected_joint import digest
from register_material_surfaces import pinned
from write_material_witnesses import write_role
from replay_connected_allocation import run as replay
from source_face_identity import primary_source_face

def run(job,out):
    rd=lambda p:json.loads(p.read_bytes())
    cfg=rd(job/'verified-connected-material/scene/launch.json');env=cfg['environment']
    qpre='NUMI_H706_FRACTIONAL_';pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    p1=out/'material-p1';vol=out/'volume';env.update(rd(p1/'result.json')['environment'])
    oldsrc=job/'deep-fat-ledger';overlay=np.load(vol/'overlay.npz');mapping=overlay['old_to_new'];src=overlay['source'];dst=overlay['target'];frac=overlay['fraction']
    oldmat=np.loadtxt(job/'deep-fat-native/mesh.ele',usecols=5,skiprows=1,dtype='i4')
    cells=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    x=np.loadtxt(vol/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8')
    for k,n in [('TETGEN_NODE','mesh.node'),('TETGEN_ELEMENTS','mesh.ele')]:
        env[pre+k]=str(vol/n);env[pre+k+'_SHA256']=digest(vol/n)
    previous_labels=pinned(env,'CHILD_LABELS','CHILD_LABELS_SHA256')
    oldlabels=np.loadtxt(previous_labels,delimiter=',',dtype='i8',skiprows=1)
    src_label=np.loadtxt(pinned(env,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),delimiter=',',skiprows=1,dtype='i8')
    outlabels=np.empty((len(cells),5),dtype='i8');surv=mapping>=0;outlabels[mapping[surv]]=oldlabels[surv]
    rows=np.r_[np.unique(dst)];outlabels[rows,1]=cells[rows,4];outlabels[:,0]=np.arange(len(cells))
    oldmass=np.fromfile(job/'deep-fat-native/cell-mass.f64','<f8')
    for k in rows:
        part=np.flatnonzero(dst==k);dominant=src[part[np.argmax(frac[part]*oldmass[src[part]])]]
        outlabels[k,2:]=oldlabels[dominant,2:]
    source=[]
    for field,cols,dtype in [('TETS',(3,4,5,6),'i4'),('NODES',(5,6,7),'f8')]:
        key=qpre+'SOURCE330_'+field;fp=Path(env[key])
        if digest(fp)!=env[key+'_SHA256']:raise ValueError('Quadrature source changed')
        source.append(np.loadtxt(fp,skiprows=1,delimiter=',',usecols=cols,dtype=dtype))
    flows=out/'flows';packets=out/'packets';stage=out/'scene-material-p1'
    for folder in (flows,packets,stage):folder.mkdir(exist_ok=True)
    if not (out/'posed-app').exists():os.symlink(job/'posed-app',out/'posed-app')
    role_meta=rd(oldsrc/'flows/result.json');qpins={};proof={}
    cellmass=np.fromfile(vol/'cell-mass.f64','<f8')
    for role,m in [('SKIN',15),('FAT',16)]:
        oldids=np.flatnonzero(oldmat==m);ids=np.flatnonzero(cells[:,4]==m)
        idx=np.full(len(cells),-1,'i4');idx[ids]=np.arange(len(ids))
        support=dict(np.load(env[qpre+'P1_'+role+'_NPZ']))
        # Select a P1 maximum and an actual source face for every inserted cell.
        for i in np.flatnonzero(np.isin(ids,rows)):
            off=support['cell_group_offsets'];slots=np.arange(off[i],off[i+1]);scores=support['cell_group_p1_score_numerators'][slots]/support['cell_group_p1_score_denominators'][slots]
            valid=[]
            for a in slots[scores==scores.max()]:
                gid=support['cell_group_ids'][a];parents=support['cell_group_parent_tet_offsets'];donors=support['cell_group_parent_tet_rows'][parents[a]:parents[a+1]]
                group=support['group_tuples'][gid];faces=np.unique(src_label[donors,4])
                if not np.all(src_label[donors,1:4]==group):raise ValueError('Foreign donor')
                prior=outlabels[ids[i]];face=primary_source_face(gid,faces,prior[4])
                valid.append((not np.array_equal(group,prior[1:4]),face!=prior[4],int(gid),int(face)))
            _,_,gid,face=min(valid);outlabels[ids[i],1:4]=support['group_tuples'][gid];outlabels[ids[i],4]=face
        k=qpre+'CARRIERS_'+role
        packet=dict(np.load(env[k+'_NPZ']))
        parent=np.repeat(packet['global_source330_tet_rows'],np.diff(packet['parent_offsets']))
        pt=np.einsum('ij,ijk->ik',packet['carrier_barycentric'],source[1][source[0][parent]])
        path=oldsrc/'flows'/(role+'-physical-flow.npz')
        if digest(path)!=role_meta[role]['flow_sha256']:raise ValueError('Unknown accepted flow')
        old=dict(np.load(path));q=old['carrier'];receiver=oldids[old['role_cell']];w=old['mass']
        retained=mapping[receiver]>=0;qs=[q[retained]];to=[idx[mapping[receiver[retained]]]];amt=[w[retained]]
        affected=np.flatnonzero(~retained)
        for a in np.unique(receiver[affected]):
            edges=affected[receiver[affected]==a];v=np.flatnonzero(src==a)
            for j in v:
                qs.append(q[edges]);to.append(np.full(len(edges),idx[dst[j]],'i4'));amt.append(w[edges]*frac[j])
        q1,c1,w1=map(np.concatenate,(qs,to,amt));n=int(retained.sum())
        center=x[cells[ids,:4]-1].mean(1);radius=float(env[pre+'MAXIMUM_MASS_TRANSPORT_DISTANCE_M'])
        rr=src_label[parent,3];off=support['cell_admissible_region_offsets'];rids=support['cell_admissible_region_ids']
        # Historical exceptions are tied to existing recipients. Never extend them to a new cell.
        dist=np.linalg.norm(pt[q1[n:]]-center[c1[n:]],axis=1)
        good=np.array([rr[a] in rids[off[b]:off[b+1]] for a,b in zip(q1[n:],c1[n:])],dtype=bool)&(dist<=radius)
        if not np.all(good):
            f=out/(role+'-ineligible-overlay.npz');np.savez_compressed(f,carrier=q1[n:][~good],recipient=ids[c1[n:][~good]],mass=w1[n:][~good],distance=dist[~good])
            sol,info=rebalance(q1,c1,w1,n,good,center,pt,rr,off,rids,radius,cells[ids,:4]-1,ids,rd(vol/'transfer.json')['cavities'],src,dst)
            if sol is None:
                (out/(role+'-infeasible.json')).write_text(json.dumps(info))
                print(role,'infeasible cavities',len(info),flush=True)
                continue
            q1,c1,w1=sol
            print(role,'balanced local cavity systems',info,flush=True)
        original=np.bincount(q,weights=w,minlength=len(pt))
        assigned=np.bincount(q1,weights=w1,minlength=len(pt))
        if np.min(assigned)<=0:raise ValueError('Missing donor after local solve')
        ratio=original/assigned;selected=abs(ratio[q1]-1)>1e-12
        w1[selected]*=ratio[q1[selected]]
        closed=np.bincount(c1,weights=w1,minlength=len(ids))
        if np.max(abs(closed-cellmass[ids])/cellmass[ids])>1e-11:raise ValueError('Relative receiver closure failed')
        err=max(abs(np.bincount(q,weights=w,minlength=len(pt))-np.bincount(q1,weights=w1,minlength=len(pt))))
        cerr=max(abs(np.bincount(c1,weights=w1,minlength=len(ids))-cellmass[ids]))
        source_total=np.bincount(q,weights=w,minlength=len(pt))
        rel=max(abs(np.bincount(q1,weights=w1,minlength=len(pt))-source_total)/source_total)
        if rel>1e-11 or err>5e-14 or cerr>3e-14:raise ValueError('Mass closure failed')
        oldtargets=packet['carrier_target_child_rows'];direct=mapping[oldids[oldtargets]]
        primary=np.full(len(direct),-1,'i4');valid=direct>=0;primary[valid]=idx[direct[valid]]
        for a in np.flatnonzero(~valid):
            ci=np.unique(c1[q1==a]);dd=np.linalg.norm(center[ci]-pt[a],axis=1)
            allow=np.array([rr[a] in rids[off[b]:off[b+1]] for b in ci])&(dd<=radius)
            if not allow.any():raise ValueError('Lost quadrature origin')
            primary[a]=ci[allow][np.argmin(dd[allow])]
        packet['carrier_target_child_rows']=primary.astype(oldtargets.dtype)
        packet['carrier_target_distance_m']=np.linalg.norm(pt-center[primary],axis=1)
        if 'regional_compensation_child_rows' in packet:
            rev=packet['regional_compensation_child_rows']
            if np.any(mapping[rev]<0):raise ValueError('Rebuilt registered exception cell')
            packet['regional_compensation_child_rows']=np.sort(mapping[rev]).astype(rev.dtype)
        flow=dict(carrier=q1,role_cell=c1,mass=w1);fp=flows/path.name
        np.savez_compressed(fp,**flow);role_meta[role]['flow_sha256']=digest(fp)
        fpack=packets/(role.lower()+'-carriers.npz');np.savez_compressed(fpack,**packet)
        witness=write_role(role,m,env,cells,x,src_label[:,1:4],*source,packet,support,flow,radius,packets/(role.lower()+'-witness.csv.gz'))
        for name,file,h in [('CARRIERS_'+role,fpack,digest(fpack)),('CARRIER_WITNESS_'+role,Path(witness['path']),witness['sha256'])]:
            key=qpre+name;qpins[key+('_NPZ' if name.startswith('CARRIERS') else '_CSV')]=str(file);qpins[key+'_SHA256']=h
        proof[role]=dict(edges=len(w1),affected_old_edges=len(affected),candidate_overlay_edges=len(dist),maximum_relative_carrier_mass_change=rel,maximum_carrier_mass_error_kg=err,maximum_cell_mass_error_kg=cerr,maximum_proposed_distance_m=float(dist.max()) if dist.size else None)
        print(role,proof[role],flush=True)
    if len(proof)!=2:raise ValueError('See bounded local infeasibility witnesses')
    f=vol/'current-child.labels.csv'
    np.savetxt(f,outlabels,delimiter=',',fmt='%d',header=previous_labels.open().readline().strip(),comments='')
    env[pre+'CHILD_LABELS']=str(f);env[pre+'CHILD_LABELS_SHA256']=digest(f)
    (flows/'result.json').write_text(json.dumps(role_meta,indent=2)+'\n')
    (packets/'result.json').write_text(json.dumps(dict(environment=qpins,roles=proof),indent=2)+'\n')
    (stage/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    cmd=rd(job/'deep-fat-native/result.json')['native_command'];cmd[1]=str(vol)
    with (vol/'native-import.json').open('w') as log,(vol/'native-import.stderr').open('w') as err:
        subprocess.run(cmd,stdout=log,stderr=err,check=True)
    owner=Path(rd(oldsrc/'native-replay/launch.json')['command'][0]).parent
    return replay(stage,out,flows,packets,owner,out/'native-replay')
if __name__=='__main__':sys.exit(run(*(Path(s).resolve() for s in sys.argv[1:])))
