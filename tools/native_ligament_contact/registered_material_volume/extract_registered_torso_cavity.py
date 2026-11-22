"""Extract the connected released skin/fat cells carrying the measured torso conflicts."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from emit_skin_source_registration import sha
launch,envout,fit,mass,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
env=json.load(launch.open())['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
def get(k,h):
    p=Path(env[prefix+k])
    if sha(p)!=env[prefix+h]:raise ValueError('Source changed '+k)
    return p
points=get('TETGEN_NODE','TETGEN_NODE_SHA256');elements=get('TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256')
with points.open() as f:
    n=int(next(f).split()[0]);x=np.loadtxt(f,max_rows=n,usecols=(1,2,3)).astype('f4')
with elements.open() as f:
    nt=int(next(f).split()[0]);e=np.loadtxt(f,max_rows=nt,dtype='i4')
t=e[:,1:5]-1;tag=e[:,5];del e
sid={r['stable_id'] for r in json.load((envout/'attribution.json').open())['source_comparison'] if r['source']['intersection_pairs']==0}
hits={'O':np.load(fit/'targets.npz')['skin_face_rows']}
iz=np.load(envout/'torso-inner-interfaces.npz')
for layer in 'ID':hits[layer]=np.unique(np.concatenate([iz[f'{layer}_{n}'] for n in sorted(sid)]))
def keys(tri):
    a=np.sort(tri,axis=1).astype('i8')
    return (a[:,0]*len(x)+a[:,1])*len(x)+a[:,2]
ns={};surface={};fs={}
for layer in 'OID':
    pt='SURFACE_'+layer+'_';pc=get(pt+'NODES_CSV',pt+'NODES_SHA256');q=get(pt+'FACES_CSV',pt+'FACES_SHA256')
    with pc.open() as f:nv=sum(1 for _ in f)-1
    ns[layer]=np.full(nv,-1,dtype='i4')
    fs[layer]=np.loadtxt(q,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
donor=np.full(len(x),-1,dtype='i4')
with get('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256').open() as f:
    for r in csv.DictReader(f):
        k=r['registered_surface_layer'];node=int(r['joint_node_id'])-1;d=int(r['donor_body'])
        if donor[node]>=0 and donor[node]!=d:raise ValueError('Conflicting donor')
        donor[node]=d
        if k in ns:
            s=int(r['registered_surface_vertex_row'])
            if ns[k][s]>=0 and ns[k][s]!=node:raise ValueError('Conflicting material surface')
            ns[k][s]=node
for k in 'OID':
    assert (ns[k]>=0).all()
    surface[k]=ns[k][fs[k]]
witness={k:np.unique(keys(surface[k][hits[k]])) for k in hits}
all_faces=np.concatenate([surface[k][hits[k]] for k in 'OID'])
lo=x[all_faces].min((0,1));hi=x[all_faces].max((0,1))
# Include all cells touching the actual witness envelope and a complete vertex ring.
# Later reconstruction must keep the exact cut and its Dirichlet boundary.
candidate=np.zeros(len(t),dtype=bool)
for start in range(0,len(t),32768):
    v=x[t[start:start+32768]];candidate[start:start+len(v)]=np.all(v.min(1)<=hi,1)&np.all(v.max(1)>=lo,1)
ring=np.unique(t[candidate]);candidate=np.flatnonzero(np.isin(t,ring).any(1));cells=t[candidate]
orders=[(1,2,3),(0,3,2),(0,1,3),(0,2,1)]
def incidence(cells):
    count=len(cells);faces=np.concatenate([cells[:,r] for r in orders]);code=keys(faces);order=np.argsort(code,kind='stable');k=code[order];same=k[1:]==k[:-1]
    if np.any(same[:-1]&same[1:]):raise ValueError('Nonmanifold cell face')
    sides=np.flatnonzero(same);return faces,code,order[sides]%count,order[sides+1]%count,order[np.r_[True,~same]&np.r_[~same,True]]
fac,code,u,v,single=incidence(cells)
graph=coo_matrix((np.ones(len(u),dtype='u1'),(u,v)),shape=(len(cells),len(cells))).tocsr()
nc,labels=connected_components(graph,directed=False)
keep=set()
for k in 'OID':
    q=np.flatnonzero(np.isin(code,witness[k]));unique,counts=np.unique(code[q],return_counts=True)
    if not np.array_equal(unique,witness[k]) or not np.all(counts==(2 if k=='I' else 1)):raise ValueError('Physical witness ownership incomplete at '+k)
    keep.update(labels[q%len(cells)].tolist())
candidate=candidate[np.isin(labels,list(keep))]
cells=t[candidate];fac,code,u,v,single=incidence(cells)
component_count,_=connected_components(coo_matrix((np.ones(len(u),dtype='u1'),(u,v)),shape=(len(cells),len(cells))).tocsr(),directed=False)
actual_boundary=fac[single];all_surface=np.unique(np.concatenate([keys(surface[k]) for k in ('O','D')]))
known=np.isin(code[single],all_surface)
# Only physically joined facets not on either original exterior are eligible cut boundaries.
ledger=get('FACET_ANCESTRY_CSV','FACET_ANCESTRY_SHA256');endcaps=[]
with ledger.open() as f:
    for r in csv.DictReader(f):
        if 'headwall' in r['face_source_role']:endcaps.append([int(r['joint_node'+str(i)])-1 for i in range(3)])
if np.isin(code[single],keys(np.asarray(endcaps))).any():raise ValueError('Cavity touches headwall')
cut=actual_boundary[~known];fkeys=keys(cut)
rest=t[np.setdiff1d(np.arange(len(t)),candidate)]
neighbor=np.zeros(len(cut),dtype='i4');sort=np.argsort(fkeys)
for o in orders:
    rest_keys=keys(rest[:,o]);hit=np.isin(rest_keys,fkeys)
    np.add.at(neighbor,sort[np.searchsorted(fkeys[sort],rest_keys[hit])],1)
if not np.all(neighbor==1):raise ValueError('Cut ring does not have exactly one retained neighbor per face')
g=np.load(mass);qm=g['element_mass_kg'];group=g['group_for_cell'];vols={}
physical=np.unique(cells);fixed=np.unique(cut)
for role in (15,16):
    ids=candidate[tag[candidate]==role]
    vols[str(role)]=dict(cells=len(ids),mass_kg=float(qm[ids].sum()),recipient_groups=len(np.unique(group[ids])))
data=dict(cell=candidate.astype('i4'),fixed_boundary=cut,fixed_nodes=fixed,affected_nodes=physical)
# Identify bonded facets whose two owning cells are both in the cavity.
layer_keys=keys(surface['I']);allk,inc=np.unique(code,return_counts=True)
data['skin_fat_bonded_faces']=surface['I'][np.isin(layer_keys,allk[inc==2])]
for k in hits:data[k+'_witness_faces']=hits[k]
np.savez_compressed(out/'physical-cavity.npz',**data)
result=dict(scope='exact connected source region for reconstruction; current scene unmodified',connected_components=component_count,witness_faces={k:len(v) for k,v in hits.items()},cells=len(cells),nodes=len(physical),paired_skin_fat_facets=len(data['skin_fat_bonded_faces']),fixed_cut_faces=len(cut),fixed_cut_nodes=len(fixed),retained_neighbors_per_cut_face=1,body_donors=sorted(map(int,np.unique(donor[physical]))),minimum_node_height_m=float(x[physical,2].min()),witness_bounds_m=[lo.tolist(),hi.tolist()],material=vols,current_mass_map_sha256=sha(mass),mesh_nodes=sha(points),mesh_elements=sha(elements),cavity_sha256=sha(out/'physical-cavity.npz'))
(out/'cavity.json').write_text(json.dumps(result,indent=2)+'\n')
