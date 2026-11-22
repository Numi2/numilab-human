"""Replay imported O/I/D facets in their exact shared physical node indexing."""
import json,sys,csv,hashlib
from pathlib import Path
import numpy as np
from layer_contacts import run

def sha(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
launch,lib,out = map(Path,sys.argv[1:4])
if len(sys.argv) not in (4,5,7):
    raise ValueError("Expected an optional connected patch or --accepted-nodes REFERENCE ACCEPTED")
accepted = len(sys.argv) == 7
constructed=accepted and sys.argv[4]=="--constructed-nodes"
if accepted and sys.argv[4] not in ("--accepted-nodes","--constructed-nodes"):
    raise ValueError("Unknown endpoint input")
env = json.loads(launch.read_text())['environment']
prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
inputs={}
def pinned(name,hash_key):
    p=Path(env[prefix+name]);digest=sha(p)
    if digest != env[prefix+hash_key]: raise ValueError('Input drift '+name)
    inputs[name]=digest
    return p
p=pinned('TETGEN_NODE','TETGEN_NODE_SHA256')
with p.open() as f:
    count,dim,_,_=map(int,next(f).split()); assert dim==3
    data=np.loadtxt(f,max_rows=count)
if not np.array_equal(data[:,0],np.arange(1,count+1)):raise ValueError('Unknown TetGen row indexing')
x=data[:,1:4].astype('f4').astype('f8')
if not np.isfinite(x).all():raise ValueError('Nonfinite node')
del data
maps={};groups=[];topology={}
for layer in 'OID':
    base='SURFACE_'+layer+'_'
    p=pinned(base+'NODES_CSV',base+'NODES_SHA256')
    with p.open() as f: rows=sum(1 for _ in f)-1
    maps[layer]=np.full(rows,-1,dtype='i4')
p=pinned('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
current=np.empty_like(x); present=np.zeros(count,dtype=bool); role_bits=np.zeros(count,dtype="uint8"); seen_inner=0
with p.open() as f:
    for row in csv.DictReader(f):
        n=int(row['joint_node_id'])-1
        if not 0<=n<count:raise ValueError('Invalid joined volume node')
        point=[float(row['current_'+a+'_m']) for a in 'xyz']
        layer=row['registered_surface_layer'];part={'skin':1,'fat':2}[row['source_partition']]
        if present[n]:
            if layer!='I' or role_bits[n]&part or not np.array_equal(current[n],point) or maps['I'][int(row['registered_surface_vertex_row'])]!=n:
                raise ValueError('Unmatched skin/fat interface')
            role_bits[n]|=part;seen_inner+=1
            continue
        present[n]=True;role_bits[n]=part
        current[n]=point
        if layer not in maps:continue
        idx=int(row['registered_surface_vertex_row']);node=int(row['joint_node_id'])-1
        if maps[layer][idx]!=-1 or not (0<=node<count):raise ValueError('Invalid physical surface map')
        maps[layer][idx]=node
if seen_inner!=len(maps['I']) or not np.all(role_bits[maps['I']]==3):raise ValueError('Incomplete bonded interface')
if not present.all() or not np.isfinite(current).all():raise ValueError('Incomplete joined volume')
current=current.astype('f4').astype('f8')
coordinate_delta=float(np.linalg.norm(x-current,axis=1).max())
for layer,m in maps.items():
    if (m<0).any() or np.unique(m).size!=m.size:raise ValueError('Unresolved boundary '+layer)
    base='SURFACE_'+layer+'_'
    f=pinned(base+'FACES_CSV',base+'FACES_SHA256')
    local=np.loadtxt(f,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    if local.min()<0 or local.max()>=len(m):raise ValueError('Invalid facet')
    tri=m[local];groups.append(tri)
    e=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]])
    lo=e.min(1).astype('int64')
    hi=e.max(1).astype('int64')
    _,ix,counts=np.unique(lo*count+hi,return_inverse=True,return_counts=True)
    orientation=np.bincount(ix,weights=np.where(e[:,0]<e[:,1],1.,-1.))
    topo=dict(faces=len(tri),boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),same_direction_two_face_edges=int(((counts==2)&(orientation!=0)).sum()))
    topology[layer]=topo
walls={0:[],2:[]}
headwall_roles={'donor_skin_OI_headwall':0,'donor_fat_ID_headwall':2}
facet=pinned('FACET_ANCESTRY_CSV','FACET_ANCESTRY_SHA256')
with facet.open() as f:
    for row in csv.DictReader(f):
        role=row['face_source_role']
        if role in headwall_roles:
            walls[headwall_roles[role]].append(tuple(int(row['joint_node'+str(k)])-1 for k in range(3)))
        elif role not in ('outer_O','inner_I','outer_I','inner_D'):
            raise ValueError('Unknown material boundary '+role)
for layer,triangles in walls.items():
    if not triangles:raise ValueError('No registered headwall')
    groups[layer]=np.vstack((groups[layer],np.asarray(triangles,dtype='i4')))
if len(sys.argv)==5:
    patch_path=Path(sys.argv[4]);patch=np.load(patch_path);nodes=patch['nodes']
    assert nodes.ndim==1 and len(nodes)==len(np.unique(nodes)) and int(nodes.min())>=0 and int(nodes.max())<len(current)
    assert patch['x'].dtype==np.dtype('float32') and np.isfinite(patch['x']).all()
    current[nodes]=patch['x'];inputs['OFFLINE_CONNECTED_VOLUME_SHA256']=sha(patch_path)
if accepted:
    def snapshot(path):
        with path.open() as f:
            if list(map(int, next(f).split())) != [count,3,0,0]:
                raise ValueError("Unexpected registered native node layout")
            values=np.loadtxt(f,dtype='f8')
        if (values.shape!=(count,4) or not np.array_equal(values[:,0],np.arange(1,count+1))
                or not np.isfinite(values[:,1:]).all()):
            raise ValueError("Incomplete native cell capture")
        return np.ascontiguousarray(values[:,1:])
    reference_path,accepted_path=map(Path,sys.argv[5:7])
    expected_initial=current
    pose=env.get("NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE")
    if pose and not constructed:
        pose=Path(pose)
        identity=sha(pose)
        pin=json.loads(launch.read_text())["pins"].get(str(pose))
        installed=any(line.startswith("registered_loaded_material_install=PASS ") and
                      ("pose_sha256="+identity) in line.split() for line in
                      (launch.parent/"stdout.log").read_text().splitlines())
        if identity!=pin or not installed:
            raise ValueError("Initial tissue pose lacks a matched native installation receipt")
        expected_initial=snapshot(pose)
        inputs["NATIVE_LOADED_START_NODE_SHA256"]=identity
    if not np.array_equal(snapshot(reference_path),expected_initial):
        raise ValueError("Endpoint capture and installed initial material have different coordinates")
    current=snapshot(accepted_path)
    inputs['NATIVE_REFERENCE_NODE_SHA256']=sha(reference_path)
    inputs['NATIVE_ACCEPTED_NODE_SHA256']=sha(accepted_path)
run(current,groups,out,lib)
d=json.loads(out.read_text());d.update(group_membership=['O plus skin O-I headwall','I counted once at the bonded interface','D plus fat I-D headwall'],headwall_counts={str(k):len(v) for k,v in walls.items()},topology=topology,reference_to_registered_float32_max_m=coordinate_delta,scope='registered initial Float32 geometry, O/I/D distinct facets, shared material features excluded',physical_nodes=count,physical_coordinate_sha256=hashlib.sha256(np.ascontiguousarray(current,dtype='<f4').tobytes()).hexdigest(),inputs=inputs,library_sha256=sha(lib))
if accepted:
    d['scope']='accepted compensated native coordinates, O/I/D distinct facets, shared material features excluded'
    d['accepted_coordinate_f64_sha256']=hashlib.sha256(np.ascontiguousarray(current,dtype='<f8').tobytes()).hexdigest()
if constructed:
    launch_info=json.loads(launch.read_text());current_pin=launch_info['pins'].get(str(accepted_path))
    if not current_pin or current_pin!=sha(accepted_path):raise ValueError("unmatched constructed state")
    d['scope']='constructed loaded current positions before motion, all registered O/I/D distinct facets and headwalls, shared material features excluded'
    d['inputs']['CONSTRUCTED_NODE_SHA256']=d['inputs'].pop('NATIVE_ACCEPTED_NODE_SHA256')
    d['constructed_coordinate_f64_sha256']=d.pop('accepted_coordinate_f64_sha256')
    d['witness_node_rows']={}
    for key,hits in d['witnesses'].items():
        i,j=map(int,key.split('-'))
        if len(hits)!=d['pair_counts'][i][j]:raise ValueError('incomplete constructed triangle witnesses')
        d['witness_node_rows'][key]=[[groups[i][a].tolist(),groups[j][b].tolist()] for a,b in hits]
out.write_text(json.dumps(d,indent=2)+'\n')
