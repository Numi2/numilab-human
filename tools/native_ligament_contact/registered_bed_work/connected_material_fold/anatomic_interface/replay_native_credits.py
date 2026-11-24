"""Trace the original quadrature credits through the connected repairs and call the native allocator."""
from pathlib import Path
from collections import defaultdict,deque
import json,os,sys,numpy as np
from compile_connected_joint import digest
from register_material_surfaces import pinned
from write_material_witnesses import write_role
from replay_connected_allocation import run as replay
def routed(p,c,w,prev,kept,removed,credit_from,credit_to,credit_mass):
    order=np.argsort(c,kind='stable');off=np.r_[0,np.cumsum(np.bincount(c,minlength=len(prev)))]
    mapping=np.full(len(prev),-1,'i4');mapping[kept]=np.arange(len(kept))
    if len(removed):mapping[removed]=-1
    good=mapping[c]>=0;qs=[p[good]];cs=[mapping[c[good]]];ws=[w[good]]
    for u,v,m in zip(credit_from,credit_to,credit_mass):
        rr=order[off[u]:off[u+1]]
        if not len(rr) or prev[u]<=0 or m<=0:raise ValueError('Invalid local credits')
        qs.append(p[rr]);cs.append(np.full(len(rr),v,'i4'));ws.append(w[rr]*(m/prev[u]))
    return np.concatenate(qs),np.concatenate(cs),np.concatenate(ws),mapping
def density_transfer(p,c,w,credit):
    src,dst,amt=(credit[k]for k in ['credit_from','credit_to','credit_kg'])
    count=defaultdict(int);out=defaultdict(list);nodes=np.unique(np.r_[src,dst])
    for a,b,v in zip(src,dst,amt):count[b]+=1;out[a].append((b,v))
    sel=np.isin(c,nodes);credit_at={k:defaultdict(float)for k in nodes}
    for a,b,v in zip(p[sel],c[sel],w[sel]):credit_at[b][a]+=v
    queue=deque(k for k in nodes if count[k]==0);seen=set()
    while queue:
        k=queue.popleft();seen.add(k);values=credit_at[k];total=sum(values.values());debit=sum(s for _,s in out[k])
        if debit>total*(1+1e-12):raise ValueError('Overdrawn physical source')
        for j,v in out[k]:
            for key,m in values.items():credit_at[j][key]+=m*v/total
            count[j]-=1
            if not count[j]:queue.append(j)
        for key in values:values[key]*=max(0.,1-debit/total)
    if len(seen)!=len(nodes):raise ValueError('Cyclic density transfer')
    local=[(i,k,v)for k,values in credit_at.items()for i,v in values.items()if v>0]
    a,b,m=map(np.asarray,zip(*local));return np.r_[p[~sel],a],np.r_[c[~sel],b],np.r_[w[~sel],m]
def run(work):
    q=work.parent;p=q.parent;r=p/'relative';vol=q/'qualified-material';nr=work/'recorded-mass';nr.mkdir(exist_ok=True)
    pr='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_';pref='NUMI_H706_FRACTIONAL_';rd=lambda f:json.loads(f.read_bytes())
    d=rd(q/'gravity-scene/launch.json');env=d['environment'];basis=rd(work/'current-p1/result.json');env.update(basis['environment'])
    if any(r['inadmissible_label_rows']for r in basis['roles'].values()):raise ValueError('Physical labels disagree with P1')
    for key,h,f in [('TETGEN_NODE','TETGEN_NODE_SHA256','mesh.node'),('TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256','mesh.ele'),('CHILD_LABELS','CHILD_LABELS_SHA256','current-child.labels.csv')]:
        env[pr+key]=str(vol/f);env[pr+h]=digest(vol/f)
    for key,path in [('material-p1',work/'current-p1'),('posed-app',p.parent/'posed-app')]:
        if not (nr/key).exists():(nr/key).symlink_to(path,target_is_directory=True)
    for k in ['flows','packets','scene']:(nr/k).mkdir(exist_ok=True)
    pts=np.loadtxt(vol/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8');cells=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');oldmat=np.loadtxt(r/'volume/mesh.ele',skiprows=1,usecols=5,dtype='i4')
    labels=np.loadtxt(pinned(env,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),skiprows=1,delimiter=',',dtype='i8')
    tet=np.loadtxt(env[pref+'SOURCE330_TETS'],skiprows=1,delimiter=',',usecols=(3,4,5,6),dtype='i4');x=np.loadtxt(env[pref+'SOURCE330_NODES'],skiprows=1,delimiter=',',usecols=(5,6,7))
    radius=float(env[pr+'MAXIMUM_MASS_TRANSPORT_DISTANCE_M']);m0=np.fromfile(r/'volume/cell-mass.f64','<f8');target=np.fromfile(vol/'cell-mass.f64','<f8');final={};pins={};priorflow=rd(r/'flows/result.json')
    def select(stage,mat,prev,gids):
        a=np.load(stage/'ancestry.npz');newids=len(a['unchanged_prior_cells']);which=mat==16
        return a['unchanged_prior_cells'],a['removed_prior_cells'] if which else np.array([],int),a['from_cell'] if which else np.array([],int),a['to_cell'] if which else np.array([],int),a['mass_kg'] if which else np.array([]),np.fromfile(stage/'cell-mass.f64','<f8')
    trace={}
    for tag,mat in [('SKIN',15),('FAT',16)]:
        orig=r/'flows'/(tag+'-physical-flow.npz')
        if digest(orig)!=priorflow[tag]['flow_sha256']:raise ValueError('Changed verified flow')
        old=np.load(orig);orig_id=np.flatnonzero(oldmat==mat);ids=np.flatnonzero(cells[:,4]==mat);aa=old['carrier'];cc=orig_id[old['role_cell']];mw=old['mass'];credit0=np.bincount(aa,weights=mw)
        packet=dict(np.load(env[pref+'CARRIERS_'+tag+'_NPZ']));base=orig_id[packet['carrier_target_child_rows']].copy();comp=packet.get('regional_compensation_child_rows',np.array([],int)).copy()
        a=np.load(q/'seam-volume/local-ancestry.npz');prev=m0
        stages=[(a['surviving_source_cells'],a['old_local_cells'],a['from_cell'],a['to_cell'],a['mass_kg'],np.fromfile(q/'seam-volume/cell-mass.f64','<f8'))]
        for folder in (q/'fold-volume',q/'remaining-fold/collar-2/fold-volume'):stages.append(select(folder,mat,prev,ids))
        for kept,remove,src,dst,kw,nextmass in stages:
            if mat==15:src=dst=remove=np.array([],int);kw=np.array([])
            aa,cc,mw,mapping=routed(aa,cc,mw,prev,kept,remove,src,dst,kw)
            ok=base>=0;base[ok]=mapping[base[ok]];comp=mapping[comp];prev=nextmass
            if (comp<0).any():raise ValueError('Historical exception affected by new material')
        if mat==16:aa,cc,mw=density_transfer(aa,cc,mw,np.load(q/'local-density-transfer/allocation.npz'))
        ci=np.searchsorted(ids,cc)
        if not np.array_equal(ids[ci],cc):raise ValueError('Cross-material flow')
        dest=np.full(len(cells),-1,'i4');dest[ids]=np.arange(len(ids))
        carrier=np.repeat(packet['global_source330_tet_rows'],np.diff(packet['parent_offsets']));Q=np.einsum('ij,ijk->ik',packet['carrier_barycentric'],x[tet[carrier]]);center=pts[cells[ids,:4]-1].mean(1)
        support=dict(np.load(env[pref+'P1_'+tag+'_NPZ']));off=support['cell_admissible_region_offsets'];regions=support['cell_admissible_region_ids'];region=labels[carrier,3]
        for key in np.flatnonzero(base<0):
            rc=np.unique(ci[aa==key]);rc=rc[[region[key] in regions[off[k]:off[k+1]]for k in rc]];dd=np.linalg.norm(center[rc]-Q[key],axis=1)
            if not len(dd) or min(dd)>radius:raise ValueError('No eligible retained carrier target')
            base[key]=ids[rc[np.argmin(dd)]]
        packet['carrier_target_child_rows']=dest[base].astype(packet['carrier_target_child_rows'].dtype)
        packet['carrier_target_distance_m']=np.linalg.norm(Q-center[dest[base]],axis=1)
        if len(comp):packet['regional_compensation_child_rows']=comp
        # Combine duplicate contributions before measuring closure.
        key=aa.astype('i8')*len(ids)+ci;order=np.argsort(key,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(key[order]))+1];un=key[order[starts]]
        aa=(un//len(ids)).astype('i4');ci=(un%len(ids)).astype('i4');mw=np.add.reduceat(mw[order],starts)
        csum=np.bincount(ci,weights=mw,minlength=len(ids));qsum=np.bincount(aa,weights=mw,minlength=len(credit0));er=abs(csum-target[ids]);es=abs(qsum-credit0)
        if np.any(er>np.maximum(3e-12*target[ids],3e-19)) or np.any(es>np.maximum(3e-12*credit0,1e-20)):raise ValueError('Credit closure failure '+str((max(er),max(es))))
        fp=nr/'flows'/orig.name;flow=dict(carrier=aa,role_cell=ci,mass=mw);np.savez_compressed(fp,**flow);priorflow[tag]['flow_sha256']=digest(fp)
        ff=nr/'packets'/(tag.lower()+'-carriers.npz');np.savez_compressed(ff,**packet)
        witness=write_role(tag,mat,env,cells,pts,labels[:,1:4],tet,x,packet,support,flow,radius,nr/'packets'/(tag.lower()+'-witness.csv.gz'))
        for label,file,h in [('CARRIERS_'+tag,ff,digest(ff)),('CARRIER_WITNESS_'+tag,Path(witness['path']),witness['sha256'])]:
            pins[pref+label+('_NPZ'if label.startswith('CARRIERS')else '_CSV')]=str(file);pins[pref+label+'_SHA256']=h
        trace[tag]=dict(carriers=len(credit0),positive_connections=len(mw),mass_kg=float(mw.sum()),max_cell_error_kg=float(max(er)),max_carrier_error_kg=float(max(es)))
        print(tag,trace[tag],flush=True)
    (nr/'flows/result.json').write_text(json.dumps(priorflow,indent=2)+'\n');(nr/'packets/result.json').write_text(json.dumps(dict(environment=pins,trace=trace),indent=2)+'\n')
    (nr/'scene/launch.json').write_text(json.dumps(d,indent=2)+'\n')
    exe=Path(rd(r/'native-replay/launch.json')['command'][0]);return replay(nr/'scene',nr,nr/'flows',nr/'packets',exe.parent,nr/'native-replay')
if __name__=='__main__':sys.exit(run(Path(__file__).resolve().parent))
