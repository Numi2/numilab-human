"""Check source intersections, then compile admitted skin/fat cavities through the existing importer."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
from prepare_shoulder import read
from compile_connected_joint import digest
from supported_material_admission import geom
from transfer_cavities import intersection_fraction
from repair_material_cavities import exposed


def save_nodes(path,x):
    with path.open('w') as h:
        h.write(f'{len(x)} 3 0 0\n')
        for chunk in np.array_split(np.arange(len(x)),32):
            np.savetxt(h,np.column_stack([chunk+1,x[chunk]]),fmt=('%d','%.17g','%.17g','%.17g'))


def run(j,candidate,subdiv,out):
    out.mkdir(exist_ok=True);v=j/'conditioned/witness-material/volume';case=j/'posterior-interface-082/native-pair';wi=read(case/'material-check.json')
    cfg=read(case/'launch.json');env=cfg['environment'];files=[v/'mesh.node',Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']),case/'physical-accepted.node']
    if [digest(f) for f in files]!=[wi[k]for k in ('reference','loaded','physical_frame')]:raise ValueError('Unmatched simulation')
    xx=[np.loadtxt(p,skiprows=1,usecols=(1,2,3))for p in files]
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1;r=raw[:,4]
    tags=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8');m=np.fromfile(v/'cell-mass.f64','<f8')
    if not np.array_equal(tags[:,0],np.arange(len(t))) or not np.array_equal(tags[:,1],r):raise ValueError('Changed material role labels')
    new=[];meta=[];patches=[];reject=[];track=[];used=set();n0=len(xx[0]);nodes=[]
    a=read(candidate/'result.json');d=read(subdiv/'result.json')
    for s in a['edits']:
        patches.append((s['witness'],np.array(s['source_cells']),np.array(s['cells']),np.array([s['coordinates']]),s['type']))
    for s in d['proposals']:
        patches.append((s['target'],np.array(s['source_cells']),np.array(s['children']),np.array([v for _,v in sorted(s['coordinates'].items(),key=lambda it:int(it[0]))]),'edge_partition'))
    for witness,ids,fresh,sites,method in patches:
        if used.intersection(ids):raise ValueError('Overlapping cavity')
        # Per-point rows are [constitutive, loaded, accepted]. Target numbering is local to each trial.
        ns=np.unique(t[ids]);local=np.r_[ns,np.arange(n0,n0+len(sites))];old=np.searchsorted(local,t[ids]);child=np.searchsorted(local,fresh)
        if not np.array_equal(local[child],fresh):raise ValueError('Unknown node')
        X=[np.vstack([x[ns],sites[:,k]])for k,x in enumerate(xx)]
        splits=np.asarray([[intersection_fraction(a,b,X[0]) for b in child]for a in old])
        vold=geom(X[0],old)[0];vn=geom(X[0],child)[0]
        rowerr=float(abs(splits.sum(1)-1).max());colerr=float(abs((splits.T@vold)/vn-1).max())
        role=r[ids];sp=splits*vold[:,None];label=role[sp.argmax(0)]
        cross=float(np.sum(sp*(role[:,None]!=label[None,:])))
        if max(rowerr,colerr)>3e-9 or cross>1e-24:
            reject.append(dict(target=witness,method=method,source_coverage_error=rowerr,child_coverage_error=colerr,cross_material_m3=cross));continue
        if len(set(map(tuple,tags[ids,2:])))!=1:raise ValueError('Ambiguous source provenance')
        # Refined boundary facets need an updated anatomical wall map before native installation.
        if {tuple(k)for k in np.sort(exposed(old),axis=1)}!={tuple(k)for k in np.sort(exposed(child),axis=1)}:
            reject.append(dict(target=witness,method=method,reason='Changes current anatomical boundary indexing'));continue
        for x in X:
            if abs(geom(x,old)[0].sum()-geom(x,child)[0].sum())>2e-15:raise ValueError('Cavity volume changed')
        # Close both sides of the overlap quadrature. Normalizing rows alone
        # perturbs otherwise uniform-density daughters at the material's upper bound.
        volume_transport=splits*vold[:,None]
        target_volume=vn*(vold.sum()/vn.sum())
        for _ in range(128):
            volume_transport*=target_volume/volume_transport.sum(0)
            volume_transport*=(vold/volume_transport.sum(1))[:,None]
            if np.max(abs(volume_transport.sum(0)/target_volume-1))<4e-15:break
        splits=volume_transport/vold[:,None]
        if np.max(abs(splits.sum(1)-1))>1e-13:raise ValueError('Unclosed mass transport')
        mass=splits.T@m[ids]
        relX=X[0]-X[0][0];moment=mass@relX[child].mean(1)-m[ids]@relX[old].mean(1)
        if np.linalg.norm(moment)>1e-11 or abs(mass.sum()-m[ids].sum())>1e-14:raise ValueError('Mass conservation failure')
        placed=fresh.copy();placed[placed>=n0]+=len(nodes)
        nodes.extend(sites);used.update(ids);new.append(placed);meta.append((mass,label,np.repeat(tags[ids[:1],2:],len(placed),axis=0)))
        track.append(dict(target=witness,source=ids.tolist(),new=placed.tolist(),source_fractions=splits.tolist(),
                          conservative_mass_kg=mass.tolist(),first_moment_error_kg_m=float(np.linalg.norm(moment)),row_error=rowerr,column_error=colerr))
    if not track:raise ValueError('No conservatively transferable tissue cavity')
    keep=np.array([k for k in range(len(t))if k not in used]);T=np.vstack([t[keep],*new]);mass=np.concatenate([m[keep],*[a for a,r,s in meta]])
    role=np.concatenate([r[keep],*[s for a,s,f in meta]]);source=np.vstack([tags[keep,2:],*[s for a,r,s in meta]])
    # Keep each untouched row at its original runtime index. The native material
    # table contains interleaved roles and looks them up per element.
    assigned=np.full(len(T),-1,dtype=np.int64)
    assigned[keep]=np.arange(len(keep))
    end=len(keep);extra=len(t);source_rows=[];target_rows=[];fractions=[]
    for rec,cs,mdata in zip(track,new,meta):
        old_ids=np.array(rec['source']);weights=np.asarray(rec['source_fractions'])
        target=np.full(len(cs),-1,dtype=np.int64);available=set(range(len(cs)))
        for j_,oid in enumerate(old_ids):
            options=[k for k in available if mdata[1][k]==r[oid] and weights[j_,k]>0]
            if not options:raise ValueError('Cannot preserve native parent row')
            k=max(options,key=lambda i:weights[j_,i]);target[k]=oid;available.remove(k)
        for k in sorted(available):target[k]=extra;extra+=1
        assigned[target]=end+np.arange(len(cs));end+=len(cs)
        a,b=np.nonzero(weights);source_rows.extend(old_ids[a]);target_rows.extend(target[b]);fractions.extend(weights[a,b])
        rec['destination_rows']=target.tolist()
    if len(np.unique(assigned))!=len(T) or assigned.min()!=0:raise ValueError('Incomplete row registry')
    T,mass,role,source=T[assigned],mass[assigned],role[assigned],source[assigned]
    points=[np.vstack([x,np.asarray(nodes)[:,k]]) for k,x in enumerate(xx)]
    dest=out/'volume';dest.mkdir(exist_ok=True)
    for name,x in zip(['mesh.node','current.node'],points[:2]):save_nodes(dest/name,x)
    with (dest/'mesh.ele').open('w')as h:
        h.write(f'{len(T)} 4 1\n')
        for z in np.array_split(np.arange(len(T)),32):np.savetxt(h,np.column_stack([z+1,T[z]+1,role[z]]),fmt='%d')
    with (dest/'current-child.labels.csv').open('w')as h:
        h.write('child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n')
        for z in np.array_split(np.arange(len(T)),32):np.savetxt(h,np.column_stack([z,role[z],source[z]]),fmt='%d',delimiter=',')
    wall=dest/'material-walls.npz'
    if not wall.exists():wall.symlink_to(v/'material-walls.npz')
    if wall.resolve()!=(v/'material-walls.npz').resolve():raise ValueError('Changed wall identity')
    np.savez_compressed(dest/'source-overlay.npz',from_cell=np.asarray(source_rows),to_cell=np.asarray(target_rows),fraction=np.asarray(fractions),mass_kg=m[np.asarray(source_rows)]*np.asarray(fractions),old_cells=np.unique(source_rows),new_cells=np.unique(target_rows))
    mass.astype('<f8').tofile(dest/'cell-mass.f64');nodal=sum(np.bincount(T[:,k],weights=mass/4,minlength=len(points[0]))for k in range(4))
    nodal.astype('<f8').tofile(dest/'node-mass.f64')
    if nodal.min()<=0 or abs(nodal.sum()-m.sum())>1e-11:raise ValueError('Invalid nodal mass map')
    qa={}
    for name,x in zip(['reference','loaded','accepted_replay'],points):
        counts={k:dict(inverted=0,low_shape=0,low_volume=0)for k in (15,16)}
        for z in np.array_split(np.arange(len(T)),64):
            a,b=geom(x,T[z])
            for k in counts:
                mask=role[z]==k
                for tag,cond in [('inverted',a<=0),('low_volume',a<=1e-18),('low_shape',b<1e-5)]:counts[k][tag]+=int(np.count_nonzero(cond&mask))
        qa[name]=counts
    cmd=read(j.parent/'posterior-load-path-077/admitted-transfer/result.json')['native_command'];cmd[1]=str(dest)
    with (dest/'native-import.json').open('w')as log,(dest/'native-import.stderr').open('w')as err:
        result=subprocess.run(cmd,stdout=log,stderr=err)
    rec=read(dest/'native-import.json')if (dest/'native-import.json').stat().st_size else {}
    answer=dict(source=wi['full_scene_launch'],input_mass=digest(v/'cell-mass.f64'),native_import_command=cmd,exit=result.returncode,native=rec,quality=qa,
                admitted=track,excluded=reject,regional_mass_kg={str(k):float(mass[role==k].sum())for k in (15,16)},
                mass_change_kg=float(mass.sum()-m.sum()),pins={f:digest(dest/f)for f in ['mesh.node','mesh.ele','current.node','cell-mass.f64','node-mass.f64','current-child.labels.csv','material-walls.npz']})
    (out/'result.json').write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
    print('physical import',result.returncode,'new',len(nodes),'quality',qa,'excluded',reject,flush=True)
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
