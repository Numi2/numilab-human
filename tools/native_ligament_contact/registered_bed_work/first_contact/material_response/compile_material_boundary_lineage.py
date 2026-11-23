"""Compose both physical cavity ancestries onto the historical O/I/D boundary rows."""
import csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from classify_connected_boundary import pack

def volume_generation(folder,overlaps):
    """Surviving child rows retain order. A cut child references all positive parent intersections."""
    native=json.loads((folder/('receipt.json' if (folder/'receipt.json').exists() else 'result.json')).read_bytes())
    source=folder/'material-source-indices.npz';a=np.load(source);edges=np.load(overlaps/'overlap.npz')
    receipt=json.loads((overlaps/'result.json').read_bytes())
    if (digest(source)!=native['output_hashes'][source.name] or
        digest(overlaps/'overlap.npz')!=receipt['mapping'] or receipt['mesh']!=native['domain_sha256']):raise ValueError('Mismatched physical cavity lineage')
    retired=a['old_cavity_tets'];active=int(a['child_begin']);count=len(a['child_geometric_parent'])
    stay=np.ones(active+len(retired),bool);stay[retired]=False;old=np.flatnonzero(stay)
    if len(old)!=active or not np.array_equal(retired,edges['parent_ids']):raise ValueError('Changed cavity incidence')
    parents=edges['parent'];children=edges['child'];order=np.argsort(children,kind='stable')
    offsets=np.r_[0,np.cumsum(np.bincount(children,minlength=count))]
    if len(offsets)!=count+1 or np.any(np.diff(offsets)==0) or not np.all(edges['volume_mm3']>0):raise ValueError('Invalid material overlap lineage')
    def ancestry(cell):
        if cell<active:return old[cell:cell+1]
        k=cell-active;return parents[order[offsets[k]:offsets[k+1]]]
    return ancestry

def closest(tri,q):
    # Closest point on each closed triangle including edges and vertices.
    a,b,c=tri.transpose(1,0,2);u=b-a;v=c-a;w=q-a
    cross=np.cross(u,v);area2=(cross*cross).sum(1)
    if np.any(area2<=0):raise ValueError('Degenerate historical source facet')
    beta=(np.cross(w,v)*cross).sum(1)/area2;gamma=(np.cross(u,w)*cross).sum(1)/area2
    weights=np.column_stack((1-beta-gamma,beta,gamma));best=np.full(len(tri),np.inf)
    inside=(weights>=0).all(1);best[inside]=np.linalg.norm(np.einsum('ij,ijk->ik',weights[inside],tri[inside])-q,axis=1)
    for i,j in [(0,1),(1,2),(2,0)]:
        edge=tri[:,j]-tri[:,i];t=np.clip(((q-tri[:,i])*edge).sum(1)/(edge*edge).sum(1),0,1)
        distance=np.linalg.norm(tri[:,i]+t[:,None]*edge-q,axis=1);hit=distance<best
        best[hit]=distance[hit];weights[hit]=0;weights[hit,i]=1-t[hit];weights[hit,j]=t[hit]
    k=best.argmin();return int(k),weights[k],float(best[k])

def run(h,wrist,job,out):
    env=json.loads((job/'posed-app/scene/launch.json').read_bytes())['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def asset(k):
        path=Path(env[pre+k+'_CSV']);expected=env[pre+k+'_SHA256']
        if digest(path)!=expected:raise ValueError('Stale '+k)
        return path
    src=h/'closed-native/source';current=h/'closed-native/native'
    old=volume_generation(src,wrist/'child-overlap');new=volume_generation(current,h/'needle-source/ancestry')
    capture=np.load(job/'boundaries/faces.npz');carpal=np.load(job/'wrist/cortical-facets.npz')
    xyz=np.loadtxt(current/'mesh.node',skiprows=1,usecols=(1,2,3))
    F=capture['face'];kind=capture['inherited_kind'].copy();ancestor=capture['ancestor'];role=capture['layer'];cells=capture['cell'];keys=pack(F)
    carpal_idx=np.searchsorted(keys,pack(carpal['nodes']))
    if not np.array_equal(F[carpal_idx].shape,carpal['nodes'].shape) or np.any(kind[carpal_idx]!=-1) or not np.array_equal(keys[carpal_idx],pack(carpal['nodes'])):raise ValueError('Cortical ownership conflict')
    kind[carpal_idx]=5
    source={k:{} for k in 'ID'}
    with asset('FACET_ANCESTRY').open() as f:
        for r in csv.DictReader(f):
            k={'inner_I':'I','inner_D':'D'}.get(r['face_source_role'])
            if k:
                face=int(r['face_source_row']);cell=int(r['child_tet_row'])
                if face in source[k]:raise ValueError('Duplicate historical surface parent')
                source[k][face]=cell
    emit={};stats={};out.mkdir()
    for code,k in enumerate('OID'):
        fs=asset('SURFACE_'+k+'_FACES');ns=asset('SURFACE_'+k+'_NODES')
        raw=np.loadtxt(fs,skiprows=1,delimiter=',',usecols=(0,1,2,3),dtype='i4')
        X=np.loadtxt(ns,skiprows=1,delimiter=',',usecols=(2,3,4));src_faces=raw[:,1:4]
        if not np.array_equal(raw[:,0],np.arange(len(raw))):raise ValueError('Noncanonical historical face ordering')
        kept=np.flatnonzero(kind==code);changed=np.flatnonzero((kind<0)&(role==code));combined=np.r_[kept,changed]
        # Historical triangle orientation is preserved even on the shared interface.
        nodes=np.unique(F[combined]);emit[k+'_boundary_node']=nodes;emit[k+'_material_face']=combined
        emit[k+'_joint_nodes']=F[combined];emit[k+'_source_rows']=ancestor[combined].copy();emit[k+'_child_tet']=cells[combined]
        emit[k+'_child_side']=capture['side'][combined]
        if not len(changed):
            stats[k]=dict(inherited=len(kept),retriangulated=0,rows=len(nodes));continue
        cell_faces=defaultdict(list);vfaces=defaultdict(list)
        if set(source[k])!=set(range(len(raw))):raise ValueError('Incomplete previous '+k+' triangle lineage')
        for i,c in source[k].items():cell_faces[c].append(i)
        missing=np.ones(len(raw),bool);missing[ancestor[kept]]=False
        for face in np.flatnonzero(missing):
            for v in src_faces[face]:vfaces[int(v)].append(int(face))
        source_ancestry=[];offset=[0];dep=[];quad_face=[];quad_bary=[];dist=[];no_lineage=[]
        surviving=np.load(h/'scene-registration/node-map.npz')[k]
        old_id=np.full(len(xyz),-1,'i4');valid=surviving>=0;old_id[surviving[valid]]=np.flatnonzero(valid)
        triangle_weights=np.vstack((np.eye(3),np.full(3,1/3),np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])))
        for j in changed:
            parent=set(int(p) for mid in new(cells[j]) for p in old(mid))
            direct=sorted({f for p in parent for f in cell_faces.get(p,())});source_ancestry.extend(direct);offset.append(len(source_ancestry))
            # Exact shared node or shared edge continuity also supports adjacent interface faces.
            vv=set(int(v) for v in old_id[F[j]] if v>=0)
            vv.update(int(v) for f in direct for v in src_faces[f])
            adjacent=sorted(set(direct)|{face for v in vv for face in vfaces[v]})
            if not adjacent:
                no_lineage.append(int(j));quad_face.append([-1]*7);quad_bary.append(np.zeros((7,3)));dist.append([np.inf]*7);dep.append(-1);continue
            tris=X[src_faces[adjacent]];picked=[];weights=[];distance=[]
            for point in triangle_weights@xyz[F[j]]:
                idx,bc,dd=closest(tris,point);picked.append(adjacent[idx]);weights.append(bc);distance.append(dd)
            dep.append(0 if set(picked)<=set(direct) else 1);quad_face.append(picked);quad_bary.append(weights);dist.append(distance)
        emit[k+'_ancestry_offsets']=np.array(offset);emit[k+'_volume_overlap_source_faces']=np.array(source_ancestry);emit[k+'_registration_method']=np.array(dep)
        emit[k+'_sample_source_face']=np.array(quad_face);emit[k+'_sample_barycentric']=np.array(quad_bary);emit[k+'_sample_gap_m']=np.array(dist)
        ok=np.array(dep)>=0;d=np.array(dist)[ok];stats[k]=dict(inherited=len(kept),retriangulated=len(changed),rows=len(nodes),direct=int((np.array(dep)==0).sum()),shared_vertex=int((np.array(dep)==1).sum()),missing=no_lineage,point_to_source_m=np.quantile(d,[0,.5,.95,.99,1]).tolist() if d.size else [])
    if sum(len(emit[k+'_material_face']) for k in 'OID')+int(np.isin(kind,[3,4,5]).sum())!=len(F):raise ValueError('Lost a tissue boundary')
    result=dict(reference=digest(current/'mesh.node'),elements=digest(current/'mesh.ele'),cortical_shoulders=int((kind==4).sum()),cortical_wrist=int((kind==5).sum()),authored_walls=int((kind==3).sum()),layers=stats)
    np.savez_compressed(out/'boundary-registration.npz',**emit)
    result['mapping']=digest(out/'boundary-registration.npz');(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:{a:(len(v) if a=='missing' else v) for a,v in r.items()} for k,r in stats.items()},flush=True)
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
