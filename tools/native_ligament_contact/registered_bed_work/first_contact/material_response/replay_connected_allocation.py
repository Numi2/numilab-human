"""Invoke the native source/density/role replay on the actual full-cell credit ledger."""
import os,sys,json,subprocess
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from register_material_surfaces import pinned

def run(base,job,flows,packets,owner,out):
    out.mkdir();e=json.loads((base/'launch.json').read_bytes())['environment'];e.update(json.loads((packets/'result.json').read_bytes())['environment'])
    p='NUMI_H706_FRACTIONAL_';prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    read=lambda name,hashkey: str(pinned(e,name,hashkey))
    def typed(path,expected):
        if digest(Path(path))!=expected:raise ValueError('Cached input identity changed '+str(path))
        return str(path)
    state=json.loads((job/'material-p1/current-paired-volume-state.json').read_bytes())
    typed(job/'material-p1/current-paired-volume-state.json',e[p+'STATE_SHA256'])
    src=state['source330_identity'];typed_source=lambda k:typed(src['source330_'+k]['path'],src['source330_'+k]['sha256'])
    elements=read('TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256');points=read('TETGEN_NODE','TETGEN_NODE_SHA256')
    mat=np.loadtxt(elements,skiprows=1,usecols=5,dtype='i4');n=len(mat);total=np.zeros(n);start=0;qq=[];cc=[];kg=[];parent=[];aug=0
    r=json.loads((flows/'result.json').read_bytes());provenance={}
    for tag,m in [('SKIN',15),('FAT',16)]:
        f=np.load(typed(flows/(tag+'-physical-flow.npz'),r[tag]['flow_sha256']));qs=f['carrier'];dest=f['role_cell'];w=f['mass']
        if min(w)<=0 or not np.isfinite(w).all() or r[tag]['unsupported_edges_after']!=0:raise ValueError('Incomplete flow eligibility')
        ids=np.flatnonzero(mat==m)
        # A wrist or shoulder route may reach the same recipient through several old cells.
        # The native packet stores one amount for each (source quadrature, receiver) pair.
        raw_count=len(w);key=qs.astype('i8')*len(ids)+dest;order=np.argsort(key,kind='stable')
        starts=np.r_[0,np.flatnonzero(np.diff(key[order]))+1];u=key[order[starts]]
        w=np.add.reduceat(w[order],starts);qs=(u//len(ids)).astype('i4');dest=(u%len(ids)).astype('i4')
        total[ids]=np.bincount(dest,weights=w,minlength=len(ids))
        key=p+'CARRIERS_'+tag;v=np.load(typed(e[key+'_NPZ'],e[key+'_SHA256']))
        par=np.repeat(v['global_source330_tet_rows'],np.diff(v['parent_offsets']));parent.append(par)
        qq.append(qs+start);start+=len(par);cc.append(ids[dest]);kg.append(w)
        solved=r[tag]['solves'][-1];active=solved.get('solves',[{}])[-1];aug+=active.get('augmentations',0)
        provenance[tag]=dict(final_credit_solver=solved,input_edges=raw_count,unique_edges=len(w),source=typed(flows/(tag+'-physical-flow.npz'),r[tag]['flow_sha256']))
    mass=np.fromfile(Path(elements).with_name('cell-mass.f64'),'<f8');error=abs(total-mass)
    if np.any(total<=0) or np.any(error>np.maximum(3e-12*mass,3e-19)) or not aug:raise ValueError('Unclosed receiver or missing solver count')
    grouped=out/'grouped.npz';np.savez_compressed(grouped,carrier_row=np.concatenate(qq).astype('i4'),group=np.concatenate(cc).astype('i4'),group_for_cell=np.arange(n,dtype='i4'),source_parent=np.concatenate(parent).astype('i8'),mass_flow_kg=np.concatenate(kg),element_mass_kg=total)
    exe=owner/'validate_complete_fine_p1';cmd=[str(exe),typed_source('mass'),typed_source('labels'),elements,e[p+'STATE_SHA256'],src['source330_mass']['sha256']]
    for tag in ('SKIN','FAT'):cmd.extend((typed(e[p+'P1_'+tag+'_NPZ'],e[p+'P1_'+tag+'_SHA256']),e[p+'P1_'+tag+'_SHA256'],e[p+'GEOMETRY_'+tag+'_SHA256']))
    cmd.extend((str(n),points,typed_source('tets'),typed_source('nodes')))
    for key,suffix in [('CARRIERS_','NPZ'),('CARRIER_WITNESS_','CSV')]:
        for tag in ('SKIN','FAT'):
            name=p+key+tag;cmd.extend((typed(e[name+'_'+suffix],e[name+'_SHA256']),e[name+'_SHA256']))
    count=len(json.loads((job/'posed-app/scene/registered-initial-bodies.json').read_bytes())['all_bodies'])
    cmd.extend((e[prefix+'MAXIMUM_MASS_TRANSPORT_DISTANCE_M'],src['source330_tets']['sha256'],src['source330_nodes']['sha256'],read('CHILD_LABELS','CHILD_LABELS_SHA256'),e[prefix+'CHILD_LABELS_SHA256'],str(count),e[prefix+'COORDINATE_FRAME_SHA256'],str(grouped),digest(grouped),str(aug)))
    ledger=out/'role-flows';ledger.mkdir();env=dict(os.environ,NUMI_HUMAN_NATIVE_ROLE_FLOW_OUTPUT=str(ledger),NUMI_HUMAN_NATIVE_REFERENCE_VOLUME_OUTPUT=str(out/'native-volume.f64'))
    launch=dict(command=cmd,executable_sha256=digest(exe),role_credits=provenance,native_geometry=elements,maximum_cell_error_kg=float(error.max()),counted_final_fixed_point_augmentations=aug)
    (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    with (out/'stdout.log').open('w') as f,(out/'stderr.log').open('w') as g:result=subprocess.run(cmd,stdout=f,stderr=g,env=env)
    pins={};rec={}
    if not result.returncode:
        for tag in ('SKIN','FAT'):
            key='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_'+tag+'_';k=tag.lower()
            for label,name in [('RECEIPT_JSON','recorded-flow-receipt.json'),('TRANSPORT_FLOW_CSV','transport-flow.csv'),('SOURCE_CARRIER_FLOW_CSV','source-carrier-flow.csv'),('CHILD_SOURCE_DONOR_CSV','child-source-donor.csv'),('SOURCE_PARENT_CSV','source-parent.csv'),('SOURCE_PARENT_DONOR_CSV','source-parent-donor.csv'),('CHILD_MASS_CSV','child-mass.csv')]:
                file=ledger/(k+'-'+name);pins[key+label]=str(file);pins[key+('RECEIPT_SHA256' if label=='RECEIPT_JSON' else label+'_SHA256')]=digest(file)
            rec[tag]=json.loads(Path(pins[key+'RECEIPT_JSON']).read_bytes())['status']
    (out/'result.json').write_text(json.dumps(dict(exit=result.returncode,environment=pins,receipts=rec),indent=2)+'\n')
    return result.returncode
if __name__=='__main__':sys.exit(run(*(Path(z).resolve() for z in sys.argv[1:])))
