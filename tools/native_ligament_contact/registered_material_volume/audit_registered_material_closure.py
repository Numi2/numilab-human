"""Check emitted facet ancestry against every boundary of the skin and fat tetrahedra."""
import json,csv,sys
from collections import Counter
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha
launch,out=map(Path,sys.argv[1:]);env=json.loads(launch.read_text())['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
def source(name,hash_key):
    p=Path(env[prefix+name]);assert sha(p)==env[prefix+hash_key];return p
mesh=source('TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256')
ancestry=source('FACET_ANCESTRY_CSV','FACET_ANCESTRY_SHA256')
node=source('TETGEN_NODE','TETGEN_NODE_SHA256')
with node.open() as f:
    node_count,dim,_,_=map(int,next(f).split());assert dim==3
    nd=np.loadtxt(f, max_rows=node_count)
if not np.array_equal(nd[:,0],np.arange(1,node_count+1)):raise ValueError('Native node IDs differ')
xyz=nd[:,1:4].astype('f4').astype('f8')
del nd
with mesh.open() as f:
    count,corners,attr=map(int,next(f).split());assert (corners,attr)==(4,1)
    data=np.loadtxt(f,dtype='i4',max_rows=count)
assert np.array_equal(data[:,0],np.arange(1,count+1))
n=int(data[:,1:5].max())+1
assert n**3<2**63 and data[:,1:5].min()==1
def codes(f):
    a=np.sort(f,axis=1).astype('int64')
    return (a[:,0]*n+a[:,1])*n+a[:,2]
mats=np.unique(data[:,5]);physical={};roles={};labels=Counter();quality={}
if not np.array_equal(mats,[15,16]) or n!=node_count+1 or not np.isfinite(xyz).all():raise ValueError('Invalid skin/fat mesh')
for mat in mats:
    tet=data[data[:,5]==mat,1:5]
    det_min=float("inf");quality_min=float("inf");low=0;inverted=0;near=0;twice=0
    for offset in range(0,len(tet),32768):
        vertices=xyz[tet[offset:offset+32768]-1];ed=vertices[:,1:]-vertices[:,:1]
        det=np.einsum('ij,ij->i',ed[:,0],np.cross(ed[:,1],ed[:,2]))
        e=vertices[:,[0,0,0,1,1,2]]-vertices[:,[1,2,3,2,3,3]]
        q=det/(np.einsum('nij,nij->ni',e,e).max(axis=1)**1.5)
        if not np.isfinite(q).all():raise ValueError('Degenerate material tetrahedron')
        inverted+=int((det<=0).sum());low+=int((q<1e-5).sum())
        near+=int((q<=1.1e-5).sum());twice+=int((q<=2e-5).sum())
        det_min=min(det_min,float(det.min()));quality_min=min(quality_min,float(q.min()))
    quality[int(mat)]=dict(nonpositive_volume=inverted,below_normalized_determinant_floor=low,normalized_determinant_floor=1e-5,min_normalized_determinant=quality_min,cells_within_ten_percent_of_floor=near,cells_at_or_below_twice_floor=twice,min_signed_volume_m3=det_min/6)
    allkeys=np.concatenate([codes(tet[:,s]) for s in [(0,1,2),(1,2,3),(2,3,0),(3,0,1)]])
    keys,inc=np.unique(allkeys,return_counts=True);del allkeys
    if np.any(inc>2):raise ValueError('Nonmanifold cell faces for material '+str(mat))
    physical[int(mat)]=keys[inc==1]
seen={'skin':[],'fat':[]};mat_of={}
with ancestry.open() as f:
    for row in csv.DictReader(f):
        part=row['source_partition'];labels[part,row['face_source_role']]+=1
        ci=int(row['joint_element_id'])-1
        if not 0<=ci<count:raise ValueError('Unknown owning cell')
        mat=int(data[ci,5])
        if mat_of.setdefault(part,mat)!=mat:raise ValueError('Ambiguous role')
        a,b,c=sorted(int(row[k]) for k in ['joint_node0','joint_node1','joint_node2'])
        seen[part].append((a*n+b)*n+c)
for part,keys in seen.items():
    key,inc=np.unique(keys,return_counts=True)
    target=physical[mat_of[part]]
    tri=np.column_stack((target//(n*n),(target//n)%n,target%n))
    edges=np.concatenate([tri[:,idx] for idx in [(0,1),(1,2),(2,0)]])
    ei=np.sort(edges,axis=1);_,edge_inc=np.unique(ei[:,0]*n+ei[:,1],return_counts=True)
    roles[part]=dict(quality=quality[mat_of[part]],material=mat_of[part],tetrahedra=int(np.sum(data[:,5]==mat_of[part])),boundary_triangles=len(target),missing_registered_faces=len(np.setdiff1d(target,key)),extra_registered_faces=len(np.setdiff1d(key,target)),duplicate_registered_faces=int(np.sum(inc-1)),open_boundary_edges=int(np.sum(edge_inc==1)),nonmanifold_boundary_edges=int(np.sum(edge_inc>2)),facet_roles={r:int(v) for (k,r),v in labels.items() if k==part})
out.write_text(json.dumps(dict(roles=roles,node_sha256=sha(node),tetgen_sha256=sha(mesh),facet_ancestry_sha256=sha(ancestry)),indent=2)+'\n')
