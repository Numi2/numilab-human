"""Transfer the existing quadrature point identities onto current physically reachable recipients."""
import json,sys
from pathlib import Path
import numpy as np
from register_material_surfaces import PREFIX,pinned,sha,tetgen_positions


def signatures(tets):
    return np.ascontiguousarray(np.column_stack([np.sort(tets[:,:4],axis=1),tets[:,4]]),dtype='i4').view('V20').ravel()


def project_role(tag,mat,env,old,new,mapping,world,source_faces,source_xyz,source_labels,mesh,routes,support,out,radius):
    src=np.flatnonzero(old[:,4]==mat);dest=np.flatnonzero(new[:,4]==mat)
    role_map=np.full(len(new),-1,dtype='i4');role_map[dest]=np.arange(len(dest))
    pf=Path(env['NUMI_H706_FRACTIONAL_CARRIERS_'+tag+'_NPZ'])
    if sha(pf)!=env['NUMI_H706_FRACTIONAL_CARRIERS_'+tag+'_SHA256']:raise ValueError('Source quadrature differs')
    packet=dict(np.load(pf));qcount=len(packet['carrier_weights'])
    parent=np.repeat(packet['global_source330_tet_rows'],np.diff(packet['parent_offsets']))
    bary=packet['carrier_barycentric'];source=np.einsum('ij,ijk->ik',bary,source_xyz[source_faces[parent]])
    typed=source_labels[parent]
    if not np.all(typed[:,0]==mat) or np.max(abs(bary.sum(1)-1))>2e-12:raise ValueError('Source quadrature ownership differs')
    src_flow=Path(env[PREFIX.replace('PREFLIGHT_','RECORDED_ALLOCATION_')+tag+'_SOURCE_CARRIER_FLOW_CSV'])
    if sha(src_flow)!=env[PREFIX.replace('PREFLIGHT_','RECORDED_ALLOCATION_')+tag+'_SOURCE_CARRIER_FLOW_CSV_SHA256']:raise ValueError('Unmatched active source allocation')
    # Preserve existing positive material routes over exact TetGen cells. They
    # are proposals until their distance, current admissibility and mass pass.
    f=np.loadtxt(src_flow,delimiter=',',skiprows=1,usecols=(0,3,4))
    target=mapping[src[f[:,0].astype('i4')]]
    k=target>=0
    qc=f[k,1].astype('i4');gc=target[k];w=f[k,2]
    interval=next(s for s in mesh['regions'] if s['material']==mat)
    selected=(gc < min(s['first_cell'] for s in mesh['regions']))
    qc=qc[selected];gc=gc[selected];w=w[selected]
    a=np.load(routes/(tag+'-ancestry.npz'));fp=routes/(tag+'-physical-flow.npz')
    if sha(fp)!=interval['route_sha256']:raise ValueError('Unmatched regional carrier allocation')
    r=np.load(fp);rq=a['carrier_source_rows'][r['carrier']];rg=interval['first_cell']+r['target']
    q=np.r_[qc,rq].astype('i4');g=np.r_[gc,rg].astype('i4');mass=np.r_[w,r['mass']]
    local=role_map[g]
    if np.any(local<0) or np.any(mass<=0):raise ValueError('Invalid proposed carrier-to-cell edge')
    centroids=world[new[dest,:4]-1].mean(axis=1)
    distances=np.linalg.norm(centroids[local]-source[q],axis=1)
    receipt=json.loads((support/(tag.lower()+'-p1-receipt.json')).read_bytes())
    sf=Path(receipt['p1_one_hop_path'])
    if sha(sf)!=receipt['p1_one_hop_sha256']:raise ValueError('Current P1 changed')
    p1=np.load(sf)
    off=p1['cell_admissible_region_offsets'];regions=p1['cell_admissible_region_ids']
    if len(off)!=len(dest)+1:raise ValueError('Current P1 rows differ')
    # Group by source region for a bounded vectorized lookup in the existing
    # per-cell region table. No geometric or anatomical radius is widened.
    neighbor=np.zeros(len(q),bool)
    for region in np.unique(typed[:,2]):
        rows=np.flatnonzero(typed[q,2]==region)
        if not len(rows):continue
        owner=np.searchsorted(off[1:],np.flatnonzero(regions==region),side='right')
        neighbor[rows]=np.isin(local[rows],owner)
    viable=neighbor & (distances<=radius)
    old_primary=mapping[src[packet['carrier_target_child_rows']]]
    retained_primary=(g==old_primary[q])
    order=np.lexsort((g,distances,~retained_primary,q))
    order=order[viable[order]]
    _,first=np.unique(q[order],return_index=True)
    selected=order[first]
    primary=np.full(qcount,-1,dtype='i4');primary[q[selected]]=local[selected]
    # A primary is an eligibility witness, not a debit of mass. Preserve its
    # exact source cell wherever the rebuilt volume still represents it.
    original_local=role_map[np.maximum(old_primary,0)]
    keep=(old_primary>=0) & (original_local>=0)
    orig_distance=np.linalg.norm(source-centroids[np.maximum(original_local,0)],axis=1)
    keep &= orig_distance<=radius
    for region in np.unique(typed[:,2]):
        rows=np.flatnonzero((typed[:,2]==region)&keep)
        candidates=np.searchsorted(off[1:],np.flatnonzero(regions==region),side='right')
        keep[rows]=np.isin(original_local[rows],candidates)
    primary[keep]=original_local[keep]
    missing=np.flatnonzero(primary<0)
    stats=dict(material=mat,carriers=qcount,unchanged_primary=int(keep.sum()),
        rebound_to_recorded_positive_edges=int(np.count_nonzero((primary>=0)&~keep)),
        missing_carriers=len(missing),source_edges=len(q),
        beyond_radius=int(np.count_nonzero(distances>radius)),outside_admissible_region=int(np.count_nonzero(~neighbor)))
    print(tag,stats,flush=True)
    np.savez_compressed(out/(tag+'-candidate-flow.npz'),carrier=q,global_cell=g,role_cell=local,mass=mass,
        distance_m=distances,eligible=viable,missing=missing,primary=primary)
    if len(missing):
        raise ValueError(f'{tag} lacks a physical primary candidate for {missing.tolist()[:25]}')
    result=dict(packet)
    result['carrier_target_child_rows']=primary.astype(packet['carrier_target_child_rows'].dtype)
    result['carrier_target_distance_m']=np.linalg.norm(source-centroids[primary],axis=1)
    if 'regional_compensation_child_rows' in result:
        old_extras=packet['regional_compensation_child_rows']
        mapped=mapping[old_extras]
        if np.any((mapped>=0)&(new[np.maximum(mapped,0),4]!=mat)):raise ValueError('Changed regional compensation material')
        result['regional_compensation_child_rows']=np.sort(mapped[mapped>=0]).astype(old_extras.dtype)
        stats.update(compensation_original=len(mapped),compensation_retained=len(result['regional_compensation_child_rows']),
                     compensation_lost=int(np.sum(mapped<0)),compensation_added=0)
    final=out/(tag.lower()+'-carriers.npz')
    np.savez_compressed(final,**result);stats.update(path=str(final),sha256=sha(final),max_distance_m=float(result['carrier_target_distance_m'].max()))
    return stats


def run(launch,material,routes,support,out):
    e=json.loads(launch.read_bytes())['environment'];mr=json.loads((material/'receipt.json').read_bytes())
    old=np.loadtxt(pinned(e,'TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256'),skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    new=np.loadtxt(material/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    for entry in mr['entries']:
        if sha(material/entry['file'])!=entry['sha256']:raise ValueError('Changed material')
    x=tetgen_positions(material/'mesh.node')
    qo=signatures(old);qn=signatures(new);order=np.argsort(qn);k=np.searchsorted(qn[order],qo)
    if np.any(qn[order][1:]==qn[order][:-1]) or len(np.unique(qo))!=len(qo):raise ValueError('Ambiguous direct cell identity')
    valid=k<len(qn);valid[valid]&=qn[order[k[valid]]]==qo[valid]
    mapping=np.full(len(qo),-1,dtype='i4');mapping[valid]=order[k[valid]]
    src=[]
    for name,cols,dtype in [('TETS',(3,4,5,6),'i4'),('NODES',(5,6,7),'f8')]:
        p=Path(e['NUMI_H706_FRACTIONAL_SOURCE330_'+name])
        if sha(p)!=e['NUMI_H706_FRACTIONAL_SOURCE330_'+name+'_SHA256']:raise ValueError('Changed source quadrature mesh')
        src.append(np.loadtxt(p,delimiter=',',skiprows=1,usecols=cols,dtype=dtype))
    source_labels=np.loadtxt(pinned(e,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    out.mkdir()
    result={}
    for tag,mat in [('SKIN',15),('FAT',16)]:
        result[tag]=project_role(tag,mat,e,old,new,mapping,x,*src,source_labels,mr,routes,support,out,mr['radius_m'])
        (out/'carriers.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
