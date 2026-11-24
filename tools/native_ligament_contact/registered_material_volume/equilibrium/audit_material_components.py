"""Trace registered FEM connectivity and solid mass from outer skin to anatomical attachment sites."""
import json,re,sys
from scipy.spatial import cKDTree
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from compile_connected_joint import digest
def audit(stage,prepared,candidate,case,out):
    mat=stage/'witness-material';volume=mat/'volume';meta=json.loads((volume/'result.json').read_bytes())
    for k in ('mesh.ele','cell-mass.f64','node-mass.f64','material-walls.npz'):
        if digest(volume/k)!=meta['inputs'][k]:raise ValueError('Unmatched '+k)
    raw=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1
    mass=np.fromfile(volume/'cell-mass.f64','<f8');nodes=np.fromfile(volume/'node-mass.f64','<f8')
    g=coo_matrix((np.ones(len(t)*3,dtype=np.int8),(np.tile(t[:,0],3),t[:,1:].T.ravel())),shape=(len(nodes),len(nodes))).tocsr()
    # Only connectivity is used. Explicit stored entries also represent edges when duplicate sums wrap.
    g.data.fill(1)
    nc,cc=connected_components(g,directed=False)
    ids=cc[t[:,0]]
    if not np.all(cc[t]==ids[:,None]):raise ValueError('Invalid connectivity partition')
    w=np.load(volume/'material-walls.npz')
    p=json.loads((prepared/'fit.json').read_text());bed=p['contact_nodes']
    hdr=(mat/'binding/app/NumiHumanLoadedSkinPoseData.inc').read_text()
    anchors=np.array([int(i)for i in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',hdr,re.M)])
    cortical=np.load(mat/'binding/contact/material-contact.npz')['facets'][:,5:8]
    patch=json.loads((candidate/'result.json').read_text())['modified_nodes']
    marks={'outer_skin':w['O'],'skin_fat_interface':w['I'],'deep_fat':w['D'],'cortical_surface':w['C'],'cortical_contact_facets':cortical,'anatomical_attachment':anchors,'registered_bed_nodes':np.asarray(bed)}
    dist={k:np.bincount(cc[np.unique(v)],minlength=nc) for k,v in marks.items()}
    count=np.bincount(ids,minlength=nc); nn=np.bincount(cc,minlength=nc);m=np.bincount(ids,weights=mass,minlength=nc)
    if np.any(count==0) or not len(anchors):raise ValueError('Incomplete node or anchor coverage')
    def component(i):
        return dict(minimum_global_node=int(np.flatnonzero(cc==i)[0]),nodes=int(nn[i]),tetrahedra=int(count[i]),mass_kg=float(m[i]),material_ids=np.unique(raw[ids==i,4]).tolist(),**{k:int(v[i])for k,v in dist.items()})
    touched=np.unique(cc[patch]);no_bed=(dist['registered_bed_nodes']==0);no_anchor=(dist['anatomical_attachment']==0);no_cort=(dist['cortical_contact_facets']==0)
    result=dict(total_material_components=nc,largest=component(np.argmax(m)),deep_fold_components=[component(i) for i in touched],bed_components=[component(i)for i in np.flatnonzero(~no_bed)],total_mass_kg=float(mass.sum()),material_components_without_explicit_cortical_or_attachment_rows=int((no_anchor&no_cort).sum()),material_mass_without_those_rows_kg=float(m[no_anchor&no_cort].sum()),nodes=int(len(nodes)),tetrahedra=len(t),reference=digest(volume/'mesh.node'),topology=meta['inputs']['mesh.ele'],scope='Connected components of the conforming material node graph. Dynamic contacts, soft-tissue anchors outside this table, and nonconforming constraints can still couple separate components.')
    cfg=json.loads((case/'launch.json').read_text())
    motion=json.loads((candidate/'motion.json').read_text())
    if digest(case/'launch.json')!=motion['full_scene_launch'] or digest(case/'physical-accepted.node')!=motion['physical_frame']:raise ValueError('Inconsistent physical frame')
    env=cfg['environment'];pref='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    labels=np.loadtxt(volume/'current-child.labels.csv',skiprows=1,delimiter=',',dtype=np.int64)[:,1:]
    source=np.loadtxt(env['NUMI_H706_FRACTIONAL_SOURCE330_TETS'],skiprows=1,delimiter=',',usecols=(1,2,3,4,5,6),dtype='i4')
    parentlabels=np.loadtxt(env[pref+'PARENT_LABELS_CSV'],skiprows=1,delimiter=',',dtype=np.int64)[:,1:]
    if not np.array_equal(source[:,:2],parentlabels[:,:2]):raise ValueError('Historical anatomy label mismatch')
    C=source[:,2:];G=coo_matrix((np.ones(3*len(C)),(np.tile(C[:,0],3),C[:,1:].T.ravel())),shape=(C.max()+1,C.max()+1)).tocsr()
    _,ps=connected_components(G,directed=False);groups={}
    for j,v in enumerate(parentlabels):groups.setdefault(tuple(v),set()).add(int(ps[C[j,0]]))
    main=int(np.argmax(m));dominant=ids==main;mainlabels=set(map(tuple,labels[dominant]))
    sources=set().union(*(groups.get(tag,set())for tag in mainlabels))
    off=np.flatnonzero(~dominant);tagset=set(map(tuple,labels[off]))
    miss=tagset-mainlabels
    # Mass per source label, aggregated independently of how many fragments carry it.
    shared=[tuple(labels[i]) in mainlabels for i in off]
    x=np.loadtxt(volume/'mesh.node',usecols=(1,2,3),skiprows=1); mainD=np.unique(w['D']);mainD=mainD[cc[mainD]==main]
    dist,_=cKDTree(x[mainD]).query(x[cc!=main])
    failures={}
    for role,vv in motion['roles'].items():
        xx=ids[np.array(vv['failing_cells'],int)];failures[role]=dict(total=len(xx),on_main_component=int((xx==main).sum()),detached_components=np.unique(xx[xx!=main]).size)
        failures[role]['detached_components']=int(failures[role]['detached_components'])
    result['source_trace']=dict(detached_cells=len(off),detached_labels=len(tagset),labels_also_in_main=len(tagset&mainlabels),labels_absent_from_main=[list(map(int,z)) for z in sorted(miss)],detached_cells_with_label_in_main=int(np.count_nonzero(shared)),mass_with_label_in_main_kg=float(mass[off][shared].sum()),unmapped_source_labels=[list(map(int,z))for z in tagset if z not in groups],detached_labels_with_parent_component_on_main=sum(bool(groups.get(z,set())&sources)for z in tagset),distance_from_detached_vertex_to_nearest_main_deep_surface_vertex_m=dict(min=float(min(dist)),median=float(np.median(dist)),max=float(max(dist))),accepted_shape_failures=failures)
    out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':audit(*(Path(s).resolve() for s in sys.argv[1:]))
