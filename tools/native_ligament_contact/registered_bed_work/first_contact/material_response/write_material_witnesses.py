"""Reconstruct native carrier evidence for positive transfers on the current emitted mesh."""
import csv
import gzip
import json
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import pinned, sha, tetgen_positions


def write_role(tag, mat, env, cells, x, labels, ref, vertices, packet, support, f, radius, path):
    rows = np.flatnonzero(cells[:,4] == mat)
    centers = x[cells[rows,:4]-1].mean(axis=1)
    parent_idx = np.repeat(np.arange(len(packet['parent_offsets'])-1), np.diff(packet['parent_offsets']))
    local_parent = packet['source_parent_tet_rows'][parent_idx]
    parent = packet['global_source330_tet_rows'][parent_idx]
    bary = packet['carrier_barycentric']
    weights = packet['carrier_weights']
    primary = packet['carrier_target_child_rows']
    points = np.einsum('ij,ijk->ik',bary,vertices[ref[parent]])
    groups = support['group_tuples']
    table = {tuple(s):i for i,s in enumerate(groups)}
    parent_group = np.array([table[tuple(v)] for v in labels[packet['global_source330_tet_rows']]],dtype='i4')
    qgroup = parent_group[parent_idx]
    width = len(groups)
    # The native verifier adds exact tuples to each recipient's admissible group list.
    # Retain a physical positive-flow witness per missing tuple and all primary carriers.
    qr, rr = f['carrier'], f['role_cell']
    keys = rr.astype('i8')*width + qgroup[qr]
    base = np.repeat(np.arange(len(rows),dtype='i8'),np.diff(support['cell_group_offsets']))*width + support['cell_group_ids']
    prim = primary.astype('i8')*width+qgroup
    unique, ix = np.unique(keys,return_index=True)
    supplemental = ix[~np.isin(unique,np.union1d(base,prim))]
    offsets = support['cell_admissible_region_offsets']
    region = support['cell_admissible_region_ids']
    extraq, extrac = qr[supplemental], rr[supplemental]
    allow = np.ones(len(extraq),bool)
    for r in np.unique(labels[parent[extraq],2]):
        which = np.flatnonzero(labels[parent[extraq],2]==r)
        good = np.searchsorted(offsets[1:],np.flatnonzero(region==r),side='right')
        allow[which] = np.isin(extrac[which],good)
    # The existing separately bounded regional exception cannot serve as evidence
    # for normal one-hop eligibility.
    extraq, extrac = extraq[allow], extrac[allow]
    old = Path(env['NUMI_H706_FRACTIONAL_CARRIER_WITNESS_'+tag+'_CSV'])
    if sha(old)!=env['NUMI_H706_FRACTIONAL_CARRIER_WITNESS_'+tag+'_SHA256']:
        raise ValueError('Original carrier construction has changed')
    original = {}
    with gzip.open(old,'rt',newline='') as stream:
        reader=csv.DictReader(stream)
        columns=reader.fieldnames
        for record in reader:
            if record['record_kind']=='carrier':
                i=int(record['packet_carrier_row'])
                if i in original or int(record['global_source330_tet_row'])!=parent[i]:
                    raise ValueError('Invalid source carrier identity')
                original[i]=record['carrier_construction']
                if len(original)==len(primary):break
    if len(original)!=len(primary):raise ValueError('Missing quadrature construction')
    radmax=0.
    with gzip.open(path,'wt',newline='',compresslevel=2) as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(columns)
        for primary_row, qs, cs in ((True,np.arange(len(primary)),primary),(False,extraq,extrac)):
            distances=np.linalg.norm(points[qs]-centers[cs],axis=1)
            if np.any(distances>radius) or not np.isfinite(distances).all():
                raise ValueError('Witness beyond material reach')
            radmax=max(radmax,float(np.max(distances,initial=0)))
            for i,c,d in zip(qs,cs,distances):
                construction=original[i] if primary_row else 'current_source_carrier_edge_support'
                mode='source_tet_canonical_quadrature' if construction=='source_tet_canonical_degree1_quadrature_1046e' else 'CurrentCarrierPointProjection'
                writer.writerow(['carrier' if primary_row else 'carrier_child_slot',tag.title(),construction,i,local_parent[i],parent[i],
                                 c,rows[c],*labels[parent[i]],mode,-1,-1,*bary[i],weights[i],*points[i],*centers[c],d])
    return dict(path=str(path),sha256=sha(path),primary=len(primary),additional_exact_tuples=len(extraq),
                historical_exception_tuples=int(np.count_nonzero(~allow)),max_distance_m=radmax)


def run(launch,mesh,packets,support,flows,out):
    env=json.loads(launch.read_bytes())['environment']
    receipt=json.loads((mesh/'receipt.json').read_bytes())
    for entry in receipt['entries']:
        if sha(mesh/entry['file'])!=entry['sha256']:raise ValueError('Changed input mesh')
    cells=np.loadtxt(mesh/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    x=tetgen_positions(mesh/'mesh.node').astype('f4').astype('f8')
    labels=np.loadtxt(pinned(env,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    src=[]
    for name,cols,dtype in [('TETS',(3,4,5,6),'i4'),('NODES',(5,6,7),'f8')]:
        p=Path(env['NUMI_H706_FRACTIONAL_SOURCE330_'+name])
        if sha(p)!=env['NUMI_H706_FRACTIONAL_SOURCE330_'+name+'_SHA256']:raise ValueError('Changed reference')
        src.append(np.loadtxt(p,delimiter=',',skiprows=1,usecols=cols,dtype=dtype))
    out.mkdir()
    report={}
    for tag,mat in (('SKIN',15),('FAT',16)):
        r=json.loads((support/(tag.lower()+'-p1-receipt.json')).read_bytes())
        p=Path(r['p1_one_hop_path'])
        if sha(p)!=r['p1_one_hop_sha256']:raise ValueError('Changed support')
        report[tag]=write_role(tag,mat,env,cells,x,labels,*src,np.load(packets/(tag.lower()+'-carriers.npz')),
                      np.load(p),np.load(flows/(tag+'-physical-flow.npz')),receipt['radius_m'],out/(tag.lower()+'-witness.csv.gz'))
        print(tag,report[tag],flush=True)
    (out/'witnesses.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
