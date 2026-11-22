"""Audit mechanical nodal connectivity of the installed two-material skin/fat solid."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from fit_dorsal_material import inputs
from material_context import sha
from torso_cell_geometry import geom


def components(t,n):
    sources=np.tile(t[:,0],3)
    targets=np.concatenate([t[:,i] for i in (1,2,3)])
    graph=coo_matrix((np.ones(len(sources),dtype=bool),(sources,targets)),shape=(n,n)).tocsr()
    _, labels=connected_components(graph,directed=False)
    if np.any(labels[t]!=labels[t[:,0],None]):
        raise ValueError('A cell spans graph components')
    return labels

def audit(native, proposal, ledger, owner, folds, interface, output):
    x,y,t,tag,pins=inputs(native)
    env=json.loads((native/'launch.json').read_text())['environment']
    masses=ledger/'current-density-closed-flow.npz'
    if sha(masses)!=json.loads((ledger/'mass-transfer.json').read_text())['flow_sha256'] or sha(proposal)!=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:
        raise ValueError('Wrong material topology or mass map')
    with np.load(masses) as f:
        nodal=f['nodal_mass_kg'].copy()
        em=f['element_mass_kg']
        source_group=f['group_for_cell']
    with np.load(proposal) as q:
        keep=q['retained_input_cell_rows']
        new=q['replacement_tetrahedra']
        parent=q['replacement_parent_cell_rows'][q['replacement_parent_offsets'][:-1]]
        tag=np.r_[tag[keep],tag[parent]]
        retired=np.setdiff1d(np.arange(len(t)),keep)
        for c in range(4):
            np.add.at(nodal,t[retired,c],-em[retired]/4)
            np.add.at(nodal,new[:,c],q['replacement_mass_kg']/4)
        em=np.r_[em[keep],q['replacement_mass_kg']]
        t=np.vstack((t[keep],new))
    labels=components(t,len(y))
    cells=labels[t[:,0]]
    size=np.bincount(cells,minlength=labels.max()+1)
    mass=np.bincount(cells,weights=em,minlength=len(size))
    nodal_total=np.bincount(labels,weights=nodal)
    if np.max(abs(nodal_total-mass))>1e-8:
        raise ValueError('Material element and lumped mass disagree')
    biggest=int(np.argmax(mass))
    mask=labels!=biggest
    anchors=re.search(r'constexpr Anchor anchors\[\]=\{(.*?)\};',(owner/'NumiHumanLoadedSkinPoseData.inc').read_text(),re.S)
    anchor_nodes=[int(n) for n in re.findall(r'\{\s*\d+u?\s*,\s*(\d+)u?\s*,',anchors.group(1))]
    if len(anchor_nodes)!=14:raise ValueError('Changed enthesis table')
    mapping=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if sha(mapping)!=env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_SHA256']:
        raise ValueError('Wrong nodal registration')
    layers=np.full(len(y),-1,dtype='i1')
    donors=np.full(len(y),-1,dtype='i4')
    dmap={}
    with mapping.open() as src:
        for r in csv.DictReader(src):
            n=int(r['joint_node_id'])-1
            code={'':0,'O':1,'I':2,'D':3}[r['registered_surface_layer']]
            donor=int(r['donor_body'])
            if layers[n] not in (-1,code) or donors[n] not in (-1,donor):
                raise ValueError('Inconsistent nodal ownership')
            layers[n],donors[n]=code,donor
            if code==3:dmap[int(r['registered_surface_vertex_row'])]=n
    if (layers<0).any() or (size==0).any():raise ValueError('Unregistered or unused physical nodes')
    fold=json.loads(folds.read_text())
    fold_nodes=np.unique([n for f in fold['faces'].values() for n in f])
    fold_labels=np.unique(labels[fold_nodes])
    if len(fold_labels)!=1:raise ValueError('Fold witnesses now span multiple components')
    rows=[w['source_row'] for g in json.loads(interface.read_text())['joined_interface_rejections']
          for c in g['failed_components'] for w in c['witnesses']]
    ids=np.searchsorted(keep,rows)
    if np.any(keep[ids]!=rows):raise ValueError('A material failure witness is retired')
    def describe(c):
        where=labels==c
        sel=cells==c
        pts=y[where].astype(float)
        return dict(graph_component=int(c),nodes=int(where.sum()),tetrahedra=int(sel.sum()),
                    mass_kg=float(mass[c]),skin_tets=int((tag[sel]==15).sum()),fat_tets=int((tag[sel]==16).sum()),
                    layer_node_counts=dict(zip(('interior','outer_skin','bonded_skin_fat','deep_fat'),np.bincount(layers[where],minlength=4).tolist())),
                    direct_entheses=int(np.count_nonzero(labels[anchor_nodes]==c)),
                    donor_bodies=np.unique(donors[where]).tolist(),bounding_dimensions_m=np.ptp(pts,axis=0).tolist())
    fi=int(fold_labels[0])
    detail=describe(fi)
    if fi!=biggest:
        main_fat_nodes=np.unique(t[(cells==biggest)&(tag==16)])
        same=main_fat_nodes[np.isin(donors[main_fat_nodes],detail['donor_bodies'])]
        if len(same):
            dist,_=cKDTree(y[same]).query(y[labels==fi])
            detail['nearest_same_donor_main_fat_vertex_distance_m']=[float(dist.min()),float(dist.max())]
    d=dict(source=pins,proposal_sha256=sha(proposal),mass_sha256=sha(masses),map_sha256=sha(mapping),
           connected_components=len(size),main=describe(biggest),other_components=int(len(size)-1),
           other_nodes=int(mask.sum()),other_tets=int(size.sum()-size[biggest]),other_mass_kg=float(mass[np.arange(len(size))!=biggest].sum()),
           components_with_direct_entheses=int(len(np.unique(labels[anchor_nodes]))),
           deep_fold_component=detail,newton_witnesses=[dict(source_row=int(row),graph_component=int(cells[idx]),in_main=bool(cells[idx]==biggest)) for row,idx in zip(rows,ids)],
           largest_other_by_mass=[describe(int(c)) for c in np.argsort(mass)[::-1][1:11]],
           scope='Nodal continuity of the installed material mesh. Unilateral contacts and independent constraints may still exchange force between distinct components.')
    # A source group shared with contiguous material is necessary for an
    # in-group reroute. It does not establish a physical tissue attachment.
    if np.any(cells[len(keep):]!=biggest):
        raise ValueError('A remeshed cell belongs to a detached region')
    source_rows=keep[cells[:len(keep)]!=biggest]
    main_groups=np.unique(source_group[keep[cells[:len(keep)]==biggest]])
    can_share_group=np.isin(source_group[source_rows],main_groups)
    d['detached_material']=dict(donors=np.unique(donors[mask]).tolist(),
        layer_node_counts=dict(zip(('interior','outer_skin','bonded_skin_fat','deep_fat'),np.bincount(layers[mask],minlength=4).tolist())),
        cell_materials=dict(zip(*[a.tolist() for a in np.unique(tag[cells!=biggest],return_counts=True)])),
        source_groups=len(np.unique(source_group[source_rows])),
        cells_with_group_also_in_contiguous_material=int(can_share_group.sum()),
        cells_with_exclusive_source_group=int((~can_share_group).sum()))
    accepted=native/'physical-accepted.node'
    if accepted.exists():
        layer_report=json.loads((native/'accepted-layers.json').read_text())
        if sha(accepted)!=layer_report['inputs']['NATIVE_ACCEPTED_NODE_SHA256']:
            raise ValueError('Wrong physical frame')
        face_src=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_D_FACES_CSV'])
        if sha(face_src)!=env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_D_FACES_SHA256']:
            raise ValueError('Unknown D surface')
        df=np.loadtxt(face_src,skiprows=1,delimiter=',',usecols=(1,2,3),dtype=int)
        if set(dmap)!=set(range(len(dmap))):raise ValueError('Incomplete deep surface')
        df=np.array([dmap[i] for i in range(len(dmap))])[df]
        end=np.loadtxt(accepted,skiprows=1,usecols=(1,2,3))
        qa=[]
        for off in range(0,len(t),48000):
            block=t[off:off+48000]
            volume,shape=geom(end,block)
            for i in np.flatnonzero((volume<=1e-18)|(shape<1e-5)):
                row=off+i
                qa.append(dict(native_material_row=int(row),source_row=int(keep[row]) if row<len(keep) else None,
                               material=int(tag[row]),component=int(cells[row]),in_main=bool(cells[row]==biggest),
                               volume_m3=float(volume[i]),shape=float(shape[i]),nodes=t[row].tolist()))
        failures=[]
        for a,b in layer_report['witnesses'].get('2-2',[]):
            if max(a,b)>=len(df):
                raise ValueError('Headwall witness needs its source facet map')
            ids=np.r_[df[a],df[b]]
            group_ids=np.unique(labels[ids])
            failures.append(dict(faces=[a,b],components=group_ids.tolist(),involves_main=bool(biggest in group_ids),donors=np.unique(donors[ids]).tolist()))
        d['matched_motion']=dict(physical_nodes_sha256=sha(accepted),quality_witnesses=qa,deep_self_pairs=failures,
                                 all_material_layer_pair_counts=layer_report['pair_counts'])
    from locate_tissue_components import locate
    d['adipose_host_location']=locate(y,t,tag,labels,biggest,fi)
    output.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    print({k:d[k] for k in ('connected_components','other_nodes','other_tets','other_mass_kg','deep_fold_component','newton_witnesses')})

if __name__=='__main__':audit(*(Path(arg).resolve() for arg in sys.argv[1:]))
