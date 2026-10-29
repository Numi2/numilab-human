from __future__ import annotations
import collections, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

PT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
BASE=PT.parent
RUN2=PT/'bank32-f32-conditioning-001/run-002-ulp-and-exact-gates'
RUN_DIR=PT/'bank32-f32-conditioning-001/run-003-local-topology-repair'
OUT=RUN_DIR/'result'
PARENT=RUN2/'candidate-f32.npz'
PAIRNPZ=RUN2/'candidate-self-intersections.npz'
BANK=PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/bank-envelopes/bank-023-032-cube2um.npz'
OPERAND=PT/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/batch-007/bank-operands/bank-23-32.npz'
REPLAY=PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'
REPLAY_REPORT=PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/report.json'
SOURCE_NODES=PT/'bank32-f32-conditioning-001/input/D-conditioner-resume-f64.npz'
MAX_DISPLACEMENT=200e-9
MAX_OPERATIONS=20
MAX_SECONDS=600.0
EXPECTED={
 'parent':'da83ed13621e25624625cabaf15452748c31b4a129a265874c28f13afc437b92',
 'pairnpz':'fe16f3f07e0037eb58022390110dfd3ae4f26e2266df913dea648a9958d0c849',
 'bank':'4cb25b1387af6caabb3479ff990ac27b49d1592817af508e8e3d78d9d6f29067',
 'operand':'509d823330bf404ebaf01ae1a95fd5966097f906f4600f4a8cf4d82161041179',
 'replay':'e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5',
 'source_nodes':'853ce71fe5be6bd0afa24be683c5901412686c30fbf8a6715da63d71991623c3',
 'self_module':'677e9b1ec585722d78685ad5e00e091ceb5ecef2ba2eecb877c5c8eb153fa1b5'
}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def require_hash(name,path):
    actual=sha(path)
    expected=EXPECTED[name]
    if expected and actual!=expected: raise RuntimeError(f'{name} SHA mismatch: {actual}')
    return actual

if OUT.exists() and any(OUT.iterdir()): raise RuntimeError(f'output directory is not empty: {OUT}')
OUT.mkdir(parents=True,exist_ok=True)
input_hashes={name:require_hash(name,path) for name,path in [('parent',PARENT),('bank',BANK),('operand',OPERAND),('replay',REPLAY),('source_nodes',SOURCE_NODES)]}
input_hashes['pairnpz']=sha(PAIRNPZ)
if EXPECTED['pairnpz'] and input_hashes['pairnpz']!=EXPECTED['pairnpz']: raise RuntimeError('exact-pair witness pin mismatch')
parent=np.load(PARENT,allow_pickle=False)
V=np.asarray(parent['points_world_m'],np.float32).astype(np.float64)
F=np.asarray(parent['faces'],np.int64).copy()
alive=np.ones(len(F),bool)
if V.shape!=(15821,3) or F.shape!=(31638,3): raise RuntimeError('unexpected frozen pass-2 candidate dimensions')
if not np.array_equal(V.astype(np.float32),parent['points_world_m']): raise RuntimeError('candidate is not canonical F32')

# Preserve all prior source ancestry as per-face sets while the local operations are explored.
def csr_sets(offsets,rows,count):
    return [set(map(int,rows[int(offsets[i]):int(offsets[i+1])])) for i in range(count)]
face_anc=csr_sets(parent['candidate_face_ancestry_offsets'],parent['candidate_face_ancestry_rows'],len(F))
union_anc=[set(zip(map(int,parent['union_face_ancestry_run_original_ids'][int(parent['union_face_ancestry_offsets'][i]):int(parent['union_face_ancestry_offsets'][i+1])]),
                    map(int,parent['union_face_ancestry_face_ids'][int(parent['union_face_ancestry_offsets'][i]):int(parent['union_face_ancestry_offsets'][i+1])]),
                    map(int,parent['union_face_ancestry_rows'][int(parent['union_face_ancestry_offsets'][i]):int(parent['union_face_ancestry_offsets'][i+1])]))) for i in range(len(F))]
d289_anc=csr_sets(parent['d289_parent_row_indptr'],parent['d289_parent_rows'],len(F))
node_anc=csr_sets(parent['output_vertex_input_parent_offsets'],parent['output_vertex_input_parent_rows'],len(V))
source_nodes=np.asarray(np.load(SOURCE_NODES,allow_pickle=False)['points_world_m'],np.float64)
if len(source_nodes)!=16008: raise RuntimeError('unexpected original source-node domain')
attrs={k:np.asarray(parent[k]).copy() for k in ('face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1')}
union_input={k:np.asarray(parent[k]).copy() for k in ('union_input_run_original_id','union_input_face_id','union_input_face_rows')}
schema=json.loads(str(parent['face_source_identity_schema_by_original_id_json'][0]))
if not {'100000064','126000032'}.issubset(schema): raise RuntimeError('source identity schema lost expected body23/bone32 owners')

pairnpz=np.load(PAIRNPZ,allow_pickle=False)
base_pairs=np.asarray(pairnpz['face_pairs'],np.int64).reshape(-1,2)
if len(base_pairs)!=8: raise RuntimeError(f'expected eight pinned self witnesses, found {len(base_pairs)}')
# These two source-ancestry patches come directly from the eight exact pair rows.
seed_face_rows=[{4594,4593,4612,4615},{19725,19893,19891,19726,19890,19896}]
seed_source_rows=[set().union(*(face_anc[r] for r in rows)) for rows in seed_face_rows]
if len(seed_source_rows[0])==0 or len(seed_source_rows[1])==0 or seed_source_rows[0]&seed_source_rows[1]: raise RuntimeError('self-witness ancestry clusters are empty or overlap unexpectedly')

sys.path.insert(0,str(BASE/'authoring-deps'))
import igl.copyleft.cgal as cgal
if sha(Path(cgal.__file__))!=EXPECTED['self_module']: raise RuntimeError('CGAL exact owner module changed')

# The native replay logical report index 48 is array row 64 for this exact bank.
rr=json.loads(REPLAY_REPORT.read_text())
rb=next((x for x in rr['banks'] if int(x.get('index',-1))==48),None)
rz=np.load(REPLAY,allow_pickle=False)
if rb is None or (int(rb['body']),int(rb['bone']),int(rb['source_id']))!=(23,32,80000048): raise RuntimeError('logical replay bank pin mismatch')
idx=64
if (int(rz['bank_body'][idx]),int(rz['bank_bone'][idx]),int(rz['bank_source_id'][idx]))!=(23,32,80000048): raise RuntimeError('array replay row64 pin mismatch')
vo=np.asarray(rz['bank_vertex_offsets'],np.int64); fo=np.asarray(rz['bank_face_offsets'],np.int64)
v0,v1=int(vo[idx]),int(vo[idx+1]); f0,f1=int(fo[idx]),int(fo[idx+1])
NB=np.asarray(rz['points_world_f32'][v0:v1],np.float32).astype(np.float64)
NBF=np.asarray(rz['faces'][f0:f1],np.int64)-v0
if (len(NB),len(NBF))!=(7257,14510): raise RuntimeError('native bank dimensions differ from pinned source')


def active_geometry(Fx,ax):
    rows=np.flatnonzero(ax)
    used=np.unique(Fx[rows].reshape(-1))
    remap=np.full(len(V),-1,np.int64); remap[used]=np.arange(len(used),dtype=np.int64)
    return V[used],remap[Fx[rows]],used,rows

def exact_self(Fx,ax):
    vv,ff,used,rows=active_geometry(Fx,ax)
    ans=cgal.remesh_self_intersections(vv,ff,detect_only=True,first_only=False)
    pp=np.asarray(ans[2],np.int64).reshape(-1,2)
    if len(pp): pp=rows[pp]
    return pp

def exact_native(vv,ff):
    combined_v=np.vstack([NB,vv])
    combined_f=np.vstack([NBF,ff+len(NB)])
    cross=cgal.remesh_self_intersections(combined_v,combined_f,detect_only=True,first_only=False)
    cp=np.asarray(cross[2],np.int64).reshape(-1,2)
    native_count=len(NBF)
    is_cross=((cp[:,0]<native_count)&(cp[:,1]>=native_count))|((cp[:,1]<native_count)&(cp[:,0]>=native_count))
    actual=cp[is_cross]
    dv,df,dj=cgal.mesh_boolean(NB,NBF,vv,ff,'difference')
    return actual,np.asarray(df,np.int64).reshape(-1,3),np.asarray(dv,np.float64).reshape(-1,3)

def edge_map(Fx,ax):
    out=collections.defaultdict(list)
    for fi in np.flatnonzero(ax):
        tri=Fx[fi]
        for j,k in ((0,1),(1,2),(2,0)):
            out[tuple(sorted((int(tri[j]),int(tri[k]))))].append(int(fi))
    return out

def structural(Fx,ax):
    rows=np.flatnonzero(ax); active=Fx[rows]
    if not len(active): return {'valid':False,'reason':'empty_surface'}
    if np.any(active<0) or np.any(active>=len(V)): return {'valid':False,'reason':'node_index_range'}
    if np.any((active[:,0]==active[:,1])|(active[:,1]==active[:,2])|(active[:,2]==active[:,0])): return {'valid':False,'reason':'repeated_corner'}
    p=V[active]
    n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    area=np.linalg.norm(n,axis=1)
    if not np.isfinite(area).all() or np.any(area==0): return {'valid':False,'reason':'degenerate_or_nonfinite'}
    edges=np.concatenate([active[:,[0,1]],active[:,[1,2]],active[:,[2,0]]],axis=0)
    ue,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    if np.any(counts!=2): return {'valid':False,'reason':'edge_incidence','incidence_counts':{str(int(x)):int(np.sum(counts==x)) for x in np.unique(counts)}}
    keys=np.sort(active,axis=1)
    if len(np.unique(keys,axis=0))!=len(active): return {'valid':False,'reason':'duplicate_face'}
    used=np.unique(active)
    euler=int(len(used)-len(ue)+len(active))
    em=edge_map(Fx,ax)
    parent=np.arange(len(rows),dtype=np.int64)
    row_index={int(r):i for i,r in enumerate(rows)}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=int(parent[x])
        return x
    def join(a,b):
        a=find(a); b=find(b)
        if a!=b: parent[b]=a
    for inc in em.values():
        if len(inc)==2: join(row_index[inc[0]],row_index[inc[1]])
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows): groups[find(i)].append(int(r))
    comps=[]
    for members in groups.values():
        tri=Fx[np.asarray(members,np.int64)]
        xyz=V[tri]
        vol=float(np.sum(np.einsum('ij,ij->i',xyz[:,0],np.cross(xyz[:,1],xyz[:,2])))/6.0)
        comps.append({'face_count':len(members),'signed_volume_m3':vol})
    comps.sort(key=lambda x:abs(x['signed_volume_m3']),reverse=True)
    totalvol=float(sum(c['signed_volume_m3'] for c in comps))
    return {'valid':True,'face_count':len(active),'used_vertex_count':len(used),'edge_count':len(ue),'euler_characteristic':euler,'component_count':len(comps),'component_signed_volumes_m3':[c['signed_volume_m3'] for c in comps],'signed_volume_m3':totalvol,'boundary_edge_count':0}

base_struct=structural(F,alive)
if not base_struct.get('valid'): raise RuntimeError(f'frozen candidate fails structural baseline: {base_struct}')
base_node_universe=set().union(*(node_anc[v] for v in np.unique(F[alive].reshape(-1))))
base_face_universe=set().union(*(face_anc[r] for r in np.flatnonzero(alive)))
base_union_universe=set().union(*(union_anc[r] for r in np.flatnonzero(alive)))
base_d289_universe=set().union(*(d289_anc[r] for r in np.flatnonzero(alive)))
source_node_shift={}
def max_source_displacement(verts, node_sets):
    mx=0.0; witness=None
    for v in verts:
        prs=node_sets[int(v)]
        if not prs: return math.inf,{'vertex':int(v),'reason':'empty_node_ancestry'}
        ids=np.fromiter(prs,dtype=np.int64)
        if ids.min()<0 or ids.max()>=len(source_nodes): return math.inf,{'vertex':int(v),'reason':'parent_index_range'}
        ds=np.linalg.norm(source_nodes[ids]-V[int(v)],axis=1)
        j=int(np.argmax(ds))
        if float(ds[j])>mx: mx=float(ds[j]); witness={'output_vertex_row':int(v),'source_node_row':int(ids[j]),'distance_m':mx}
    return mx,witness
base_shift,base_shift_witness=max_source_displacement(np.unique(F[alive].reshape(-1)),node_anc)
if base_shift>MAX_DISPLACEMENT: raise RuntimeError(f'pass-2 candidate exceeds local 200nm source bound: {base_shift}')

base_self=exact_self(F,alive)
if sorted(map(tuple,base_self.tolist()))!=sorted(map(tuple,base_pairs.tolist())): raise RuntimeError('fresh exact self output does not match frozen eight-pair witness')
current_pairs=base_self
current_count=len(current_pairs)
if current_count!=8: raise RuntimeError('unexpected frozen self pair count')


def build_local_candidates(Fx,ax,fa,ua,pair_rows,flip_bound_used):
    em=edge_map(Fx,ax)
    face_adj=collections.defaultdict(set)
    for inc in em.values():
        if len(inc)==2:
            a,b=inc; face_adj[a].add(b); face_adj[b].add(a)
    cluster_dist={}
    for ci,source_rows_set in enumerate(seed_source_rows):
        seeds=[int(r) for r in np.flatnonzero(ax) if fa[int(r)]&source_rows_set]
        if not seeds: continue
        dist={r:0 for r in seeds}; frontier=list(seeds)
        for depth in (1,2):
            nxt=[]
            for r in frontier:
                for n in face_adj[r]:
                    if n not in dist: dist[n]=depth; nxt.append(n)
            frontier=nxt
        for r,d in dist.items(): cluster_dist[r]=min(cluster_dist.get(r,99),d)
    patch={r for r,d in cluster_dist.items() if d<=2}
    used_patch=set(map(int,Fx[np.asarray(sorted(patch),np.int64)].reshape(-1))) if patch else set()
    boundary_vertices=set()
    for e,inc in em.items():
        if len(inc)==1: boundary_vertices.update(e)
    # Current intersection rows keep priority; the surviving face ancestry patches expand up to 2 edge rings.
    pairfaces=set(map(int,pair_rows.reshape(-1)))
    lineage_ids={}
    for r in patch:
        ids={int(t[0]) for t in ua[r]}
        lineage_ids[r]=ids
    def domain_ok(rows):
        if not rows: return False
        if len({int(attrs['face_source_layer'][r]) for r in rows})!=1 or len({int(attrs['face_source_body'][r]) for r in rows})!=1 or len({int(attrs['face_source_bone'][r]) for r in rows})!=1: return False
        ids=set()
        for r in rows:
            ids.update(int(t[0]) for t in ua[r])
        return bool(ids) and -1 not in ids and len(ids)==1
    candidates=[]
    # Prefer legal flips that directly replace a self-witness face, then 1- and 2-ring edges.
    for edge,inc in em.items():
        if len(inc)!=2: continue
        fi,fj=map(int,inc)
        if fi not in patch or fj not in patch or not domain_ok((fi,fj)): continue
        a,b=edge
        c=None; d=None
        t1=Fx[fi].tolist(); t2=Fx[fj].tolist()
        for k in range(3):
            if t1[k]==a and t1[(k+1)%3]==b: c=int(t1[(k+2)%3]); break
            if t1[k]==b and t1[(k+1)%3]==a: a,b=b,a; c=int(t1[(k+2)%3]); break
        if c is None: continue
        # Adjacent face must traverse the shared edge in the opposite direction.
        if not any(t2[k]==b and t2[(k+1)%3]==a for k in range(3)): continue
        dval=[int(x) for x in t2 if int(x) not in (a,b)]
        if len(dval)!=1: continue
        d=dval[0]
        if len({a,b,c,d})!=4: continue
        diag=tuple(sorted((c,d)))
        if diag in em: continue
        nold=np.cross(V[Fx[fi,1]]-V[Fx[fi,0]],V[Fx[fi,2]]-V[Fx[fi,0]])+np.cross(V[Fx[fj,1]]-V[Fx[fj,0]],V[Fx[fj,2]]-V[Fx[fj,0]])
        if not np.isfinite(nold).all() or float(np.linalg.norm(nold))==0: continue
        new=np.asarray([[c,d,b],[d,c,a]],np.int64)
        nn=np.cross(V[new[:,1]]-V[new[:,0]],V[new[:,2]]-V[new[:,0]])
        if np.any(np.linalg.norm(nn,axis=1)==0) or np.any(nn@nold<=0): continue
        nplane=np.cross(V[b]-V[a],V[c]-V[a]); denom=float(np.linalg.norm(nplane))
        if denom==0: continue
        plane_dev=abs(float((V[d]-V[a])@nplane))/denom
        flip_bound=2.0*plane_dev
        if base_shift+flip_bound_used+flip_bound>MAX_DISPLACEMENT: continue
        keys={tuple(sorted(map(int,x))) for x in new}
        if len(keys)!=2: continue
        current_keys={tuple(sorted(map(int,x))) for x in Fx[np.flatnonzero(ax)]}
        oldkeys={tuple(sorted(map(int,Fx[fi]))),tuple(sorted(map(int,Fx[fj])))}
        if any(k in current_keys-oldkeys for k in keys): continue
        rank=min(cluster_dist.get(fi,99),cluster_dist.get(fj,99))
        direct=0 if fi in pairfaces or fj in pairfaces else 1
        candidates.append({'kind':'flip','edge':edge,'faces':[fi,fj],'new_faces':new.tolist(),'source_surface_bound_m':flip_bound,'priority':[direct,rank,0,flip_bound,fi,fj,edge[0],edge[1]]})
    # Link-safe contractions are considered only for source-coincident nearby edges in the two-ring witness patches.
    neighbors=[set() for _ in range(len(V))]
    incident=[set() for _ in range(len(V))]
    for fi in np.flatnonzero(ax):
        tri=Fx[fi]
        for v in tri:
            incident[int(v)].add(int(fi))
            neighbors[int(v)].update(int(w) for w in tri if int(w)!=int(v))
    for edge,inc in em.items():
        if len(inc)!=2: continue
        a,b=edge; fi,fj=map(int,inc)
        if fi not in patch or fj not in patch or a not in used_patch or b not in used_patch: continue
        dist=float(np.linalg.norm(V[a]-V[b]))
        if dist==0 or dist>MAX_DISPLACEMENT: continue
        if a in boundary_vertices or b in boundary_vertices: continue
        opp=set(int(v) for r in (fi,fj) for v in Fx[r] if int(v) not in (a,b))
        if neighbors[a]&neighbors[b] != opp: continue
        changed=sorted((incident[b] if len(node_anc[a])<=len(node_anc[b]) else incident[a])-set((fi,fj)))
        # Select keeper below, then re-evaluate changed rows after selection.
        choices=[]
        for keep,drop in ((a,b),(b,a)):
            combined=node_anc[keep]|node_anc[drop]
            if not combined: continue
            ids=np.fromiter(combined,dtype=np.int64)
            displacement=float(np.max(np.linalg.norm(source_nodes[ids]-V[keep],axis=1),initial=0.0))
            if displacement<=MAX_DISPLACEMENT: choices.append((displacement,keep,drop,combined))
        if not choices: continue
        displacement,keep,drop,combined=min(choices,key=lambda x:(x[0],x[1]))
        changed=sorted(incident[drop]-set((fi,fj)))
        if not changed or any(r not in patch for r in changed): continue
        if not domain_ok(set(incident[keep])|set(incident[drop])): continue
        old=Fx[np.asarray(changed,np.int64)]
        new=old.copy();new[new==drop]=keep
        if np.any((new[:,0]==new[:,1])|(new[:,1]==new[:,2])|(new[:,2]==new[:,0])): continue
        oldn=np.cross(V[old[:,1]]-V[old[:,0]],V[old[:,2]]-V[old[:,0]])
        newn=np.cross(V[new[:,1]]-V[new[:,0]],V[new[:,2]]-V[new[:,0]])
        if np.any(np.linalg.norm(newn,axis=1)==0) or np.any(np.einsum('ij,ij->i',oldn,newn)<=0): continue
        current_keys={tuple(sorted(map(int,x))) for x in Fx[np.flatnonzero(ax)]}
        remove_keys={tuple(sorted(map(int,Fx[r]))) for r in (fi,fj,*changed)}
        newkeys=[tuple(sorted(map(int,x))) for x in new]
        if len(set(newkeys))!=len(newkeys) or any(k in current_keys-remove_keys for k in newkeys): continue
        rank=min(cluster_dist.get(fi,99),cluster_dist.get(fj,99))
        direct=0 if fi in pairfaces or fj in pairfaces or set(changed)&pairfaces else 1
        candidates.append({'kind':'contract','edge':edge,'faces':[fi,fj],'keep':int(keep),'drop':int(drop),'changed_faces':changed,'new_faces':new.tolist(),'source_surface_bound_m':displacement,'priority':[direct,rank,1,dist,fi,fj,a,b]})
    candidates.sort(key=lambda c:tuple(c['priority']))
    return candidates,{'patch_face_count':len(patch),'patch_vertex_count':len(used_patch),'seed_current_face_counts':[sum(1 for r in patch if cluster_dist.get(r)==0)],'legal_candidate_count':len(candidates),'flip_candidate_count':sum(c['kind']=='flip' for c in candidates),'contract_candidate_count':sum(c['kind']=='contract' for c in candidates),'boundary_edges':int(sum(1 for inc in em.values() if len(inc)==1))}

def trial_state(Fx,ax,op):
    ft=Fx.copy(); at=ax.copy()
    if op['kind']=='flip':
        i,j=op['faces']; ft[i]=op['new_faces'][0]; ft[j]=op['new_faces'][1]
    else:
        i,j=op['faces']; keep=op['keep']; drop=op['drop']
        changed=np.asarray(op['changed_faces'],np.int64)
        ft[changed]=np.asarray(op['new_faces'],np.int64)
        at[i]=False; at[j]=False
    return ft,at

def lineage_domain_ok(rows,ua,attrs_):
    ids=set()
    for r in rows: ids.update(int(t[0]) for t in ua[r])
    return ids=={100000064} and len({int(attrs_['face_source_layer'][r]) for r in rows})==1 and len({int(attrs_['face_source_body'][r]) for r in rows})==1 and len({int(attrs_['face_source_bone'][r]) for r in rows})==1

start=time.monotonic()
operations=[]; attempts=[]; flip_surface_bound=0.0
changed_rows=set(); changed_node_rows=set(); vertex_representative=np.arange(len(V),dtype=np.int64)
current_struct=base_struct
current_pairs=base_self.copy()
current_count=len(current_pairs)
trial_count=0
stop_reason='operation_limit_or_no_improvement'
while trial_count<MAX_OPERATIONS and time.monotonic()-start<MAX_SECONDS and current_count>0:
    candidates,inv=build_local_candidates(F,alive,face_anc,union_anc,current_pairs,flip_surface_bound)
    if not candidates:
        stop_reason='no_legal_local_operations'
        attempts.append({'stage':len(operations),'status':'NO_LEGAL_OPERATIONS','inventory':inv})
        break
    attempts.append({'stage':len(operations),'status':'CANDIDATE_INVENTORY','inventory':inv,'top_candidates':[{'kind':c['kind'],'edge':c['edge'],'faces':c['faces'],'source_surface_bound_m':c['source_surface_bound_m'],'priority':c['priority']} for c in candidates[:20]]})
    accepted=False
    for op in candidates:
        if trial_count>=MAX_OPERATIONS or time.monotonic()-start>=MAX_SECONDS: break
        trial_count+=1
        ft,at=trial_state(F,alive,op)
        st=structural(ft,at)
        if not st.get('valid') or st['euler_characteristic']!=base_struct['euler_characteristic'] or st['component_count']!=base_struct['component_count'] or [math.copysign(1.0,x) for x in st['component_signed_volumes_m3']] != [math.copysign(1.0,x) for x in base_struct['component_signed_volumes_m3']]:
            attempts.append({'trial':trial_count,'op':{k:op[k] for k in ('kind','edge','faces','source_surface_bound_m')},'status':'REJECT_STRUCTURE','structural':st}); continue
        active_rows=np.flatnonzero(at)
        # Validate the source displacement bound on every active node after endpoint ancestry is merged.
        test_node_anc=node_anc
        if op['kind']=='contract':
            test_node_anc=list(node_anc)
            keep,drop=op['keep'],op['drop']
            test_node_anc[keep]=set(node_anc[keep])|set(node_anc[drop])
        used=np.unique(ft[active_rows].reshape(-1))
        node_shift,wit=max_source_displacement(used,test_node_anc)
        total_bound=node_shift+flip_surface_bound+(op['source_surface_bound_m'] if op['kind']=='flip' else 0.0)
        if node_shift>MAX_DISPLACEMENT or total_bound>MAX_DISPLACEMENT:
            attempts.append({'trial':trial_count,'op':{k:op[k] for k in ('kind','edge','faces','source_surface_bound_m')},'status':'REJECT_SOURCE_DISPLACEMENT','max_node_source_displacement_m':node_shift,'witness':wit,'accumulated_bound_m':total_bound}); continue
        # Complete local orientation and role/source-label checks before the exact owner is called.
        if op['kind']=='flip':
            i,j=op['faces']; new=np.asarray(op['new_faces'],np.int64)
            if not lineage_domain_ok((i,j),union_anc,attrs):
                attempts.append({'trial':trial_count,'op':{'kind':'flip','edge':op['edge'],'faces':[i,j]},'status':'REJECT_TYPED_DOMAIN'}); continue
            meta_keys=('face_source_layer','face_source_body','face_source_bone')
            if any(attrs[k][i]!=attrs[k][j] for k in meta_keys):
                attempts.append({'trial':trial_count,'op':{'kind':'flip','edge':op['edge'],'faces':[i,j]},'status':'REJECT_TYPED_LABEL_MISMATCH'}); continue
        else:
            i,j=op['faces']; changed=op['changed_faces']; allrows=set(changed)|{i,j}
            if not lineage_domain_ok(allrows,union_anc,attrs):
                attempts.append({'trial':trial_count,'op':{'kind':'contract','edge':op['edge'],'faces':[i,j]},'status':'REJECT_TYPED_DOMAIN'}); continue
        pp=exact_self(ft,at)
        if len(pp)>=current_count:
            attempts.append({'trial':trial_count,'op':{k:op[k] for k in ('kind','edge','faces','source_surface_bound_m')},'status':'REJECT_SELF_COUNT_NOT_STRICTLY_LOWER','before':current_count,'after':len(pp),'witness_pairs_internal':pp.tolist()[:24]}); continue
        vv,ff,used,active_rows=active_geometry(ft,at)
        cross,diff_faces,diff_verts=exact_native(vv,ff)
        if len(cross):
            attempts.append({'trial':trial_count,'op':{k:op[k] for k in ('kind','edge','faces','source_surface_bound_m')},'status':'REJECT_NATIVE_BONE_CROSS','self_pairs_after':len(pp),'native_cross_pairs':len(cross)}); continue
        if len(diff_faces):
            attempts.append({'trial':trial_count,'op':{k:op[k] for k in ('kind','edge','faces','source_surface_bound_m')},'status':'REJECT_NATIVE_BONE_CONTAINMENT','self_pairs_after':len(pp),'difference_faces':len(diff_faces)}); continue
        # Commit only after strict exact self improvement AND both exact native-bone gates pass.
        old_count=current_count
        old_face_rows=sorted(set(op['faces'] if op['kind']=='flip' else op['faces']+op['changed_faces']))
        old_face_snapshot=F[np.asarray(old_face_rows,np.int64)].astype(int).tolist()
        if op['kind']=='flip':
            i,j=op['faces']; merged_fa=face_anc[i]|face_anc[j]; merged_u=union_anc[i]|union_anc[j]; merged_d=d289_anc[i]|d289_anc[j]
            F=ft; alive=at
            for r in (i,j): face_anc[r]=set(merged_fa); union_anc[r]=set(merged_u); d289_anc[r]=set(merged_d); changed_rows.add(r)
            flip_surface_bound+=op['source_surface_bound_m']
        else:
            i,j=op['faces']; keep,drop=op['keep'],op['drop']; removed={i,j}
            removed_fa=set().union(*(face_anc[r] for r in removed)); removed_u=set().union(*(union_anc[r] for r in removed)); removed_d=set().union(*(d289_anc[r] for r in removed))
            F=ft; alive=at
            for r in op['changed_faces']:
                face_anc[r]|=removed_fa; union_anc[r]|=removed_u; d289_anc[r]|=removed_d; changed_rows.add(int(r))
            node_anc[keep]|=node_anc[drop]
            vertex_representative[drop]=keep
            vertex_representative[vertex_representative==drop]=keep
            changed_node_rows.add(keep)
            operations.append({'accepted_index':len(operations),'kind':'link_safe_edge_contraction','edge':list(op['edge']),'kept_vertex_row':int(keep),'retired_vertex_row':int(drop),'removed_face_rows':sorted(removed),'changed_face_rows':op['changed_faces'],'old_face_rows':old_face_rows,'old_faces':old_face_snapshot,'new_changed_faces':F[op['changed_faces']].astype(int).tolist(),'vertex_source_displacement_m':float(op['source_surface_bound_m']),'propagated_face_ancestry_rows':sorted(removed_fa),'propagated_union_ancestry':sorted([list(x) for x in removed_u])})
        if op['kind']=='flip':
            operations.append({'accepted_index':len(operations),'kind':'interior_diagonal_flip','edge':list(op['edge']),'face_rows':op['faces'],'old_faces':old_face_snapshot,'new_faces':op['new_faces'],'source_surface_deviation_bound_m':float(op['source_surface_bound_m']),'face_ancestry_union':sorted(set().union(*(face_anc[r] for r in op['faces']))),'union_ancestry_union':sorted([list(x) for x in set().union(*(union_anc[r] for r in op['faces']))])})
        current_pairs=pp; current_count=len(pp); current_struct=st; accepted=True
        attempts.append({'trial':trial_count,'operation_accepted_index':len(operations)-1,'status':'ACCEPT_STRICT_SELF_IMPROVEMENT_AND_NATIVE_BONE_GATES','before':old_count,'after':current_count,'native_cross_pairs':0,'native_difference_faces':0,'structural':st,'max_node_source_displacement_m':node_shift,'accumulated_surface_bound_m':flip_surface_bound})
        break
    if not accepted:
        stop_reason='no_strictly_improving_operation_within_local_candidate_set'
        break
if current_count==0: stop_reason='exact_self_zero'
elapsed=time.monotonic()-start

# Confirm every retained face/node ancestry row is still represented.
active_rows=np.flatnonzero(alive)
active_vertices=np.unique(F[active_rows].reshape(-1))
face_universe=set().union(*(face_anc[r] for r in active_rows))
union_universe=set().union(*(union_anc[r] for r in active_rows))
node_universe=set().union(*(node_anc[v] for v in active_vertices))
lineage_checks={
 'face_parent_rows_complete':face_universe==base_face_universe,
 'union_parent_tuples_complete':union_universe==base_union_universe,
 'd289_parent_rows_complete':set().union(*(d289_anc[r] for r in active_rows))==base_d289_universe,
 'source_node_rows_complete':node_universe==base_node_universe,
 'active_face_parent_count':len(face_universe),
 'baseline_face_parent_count':len(base_face_universe),
 'active_union_parent_count':len(union_universe),
 'baseline_union_parent_count':len(base_union_universe),
 'active_source_node_parent_count':len(node_universe),
 'baseline_source_node_parent_count':len(base_node_universe),
 'unreferenced_output_vertex_rows':int(len(V)-len(active_vertices))
}
if not all(lineage_checks[k] for k in ('face_parent_rows_complete','union_parent_tuples_complete','d289_parent_rows_complete','source_node_rows_complete')):
    raise RuntimeError(f'local repair lost source ancestry: {lineage_checks}')
final_shift,final_shift_witness=max_source_displacement(active_vertices,node_anc)
if final_shift+flip_surface_bound>MAX_DISPLACEMENT: raise RuntimeError('final source geometry bound exceeded')

final_gates={'self':{'status':'PASS' if current_count==0 else 'FAIL','pair_count':int(current_count)},'native_bone_cross':{'status':'PASS' if operations else 'NOT_RUN'},'native_bone_containment':{'status':'PASS' if operations else 'NOT_RUN'}}
if operations:
    # Last accepted state already passed these exact gates; repeat them on the final retained state.
    vv,ff,used,rows=active_geometry(F,alive)
    cross,diff_faces,diff_verts=exact_native(vv,ff)
    final_gates['native_bone_cross']={'status':'PASS' if len(cross)==0 else 'FAIL','pair_count':int(len(cross))}
    final_gates['native_bone_containment']={'status':'PASS' if len(diff_faces)==0 else 'FAIL','difference_faces':int(len(diff_faces)),'difference_vertices':int(len(diff_verts))}
    if len(cross): np.savez_compressed(OUT/'final-native-bone-cross-witnesses.npz',face_pairs=cross)
    if len(diff_faces): np.savez_compressed(OUT/'final-native-bone-difference.npz',points_world_m_f64=diff_verts,faces=diff_faces)

result={
 'schema':'numi.bank32-local-exact-self-repair.v1',
 'status':'NO_IMPROVING_LOCAL_OPERATION' if not operations else ('PASS_LOCAL_GATES_SELF_ZERO' if current_count==0 and final_gates['native_bone_cross']['status']=='PASS' and final_gates['native_bone_containment']['status']=='PASS' else 'IMPROVED_BUT_UNQUALIFIED'),
 'parent_candidate_path':str(PARENT),'parent_candidate_sha256':input_hashes['parent'],'parent_exact_pair_path':str(PAIRNPZ),'parent_exact_pair_sha256':input_hashes['pairnpz'],
 'source_envelope_path':str(BANK),'source_envelope_sha256':input_hashes['bank'],'native_bone_replay_sha256':input_hashes['replay'],'source_node_geometry_sha256':input_hashes['source_nodes'],
 'limits':{'max_source_geometric_displacement_m':MAX_DISPLACEMENT,'max_local_operations':MAX_OPERATIONS,'max_elapsed_s':MAX_SECONDS,'max_face_ring':2},
 'baseline':{'exact_self_pair_count':8,'structural':base_struct,'max_source_node_displacement_m':base_shift,'max_source_node_displacement_witness':base_shift_witness},
 'final':{'exact_self_pair_count':int(current_count),'structural':current_struct,'max_source_node_displacement_m':final_shift,'max_source_node_displacement_witness':final_shift_witness,'accumulated_flip_surface_deviation_bound_m':flip_surface_bound,'exact_gates':final_gates,'lineage':lineage_checks},
 'attempt_count':trial_count,'accepted_operation_count':len(operations),'operations':operations,'attempts':attempts,'stop_reason':stop_reason,'elapsed_s':elapsed,
 'physical_admission':False,'runtime':0,'accepted_physical_steps':0
}

if operations:
    # Emit a new, compact candidate only after the retained state passes all ancestry/structure checks.
    out_rows=np.flatnonzero(alive); Fout=F[out_rows].copy()
    out_to_internal={int(r):int(i) for i,r in enumerate(out_rows)}
    new_changed=sorted(out_to_internal[r] for r in changed_rows if r in out_to_internal)
    if not new_changed: raise RuntimeError('accepted operation has no changed output faces')
    new_id=126000033
    if str(new_id) in schema: raise RuntimeError('derived source ID already exists')
    # Compact per-face lineage for the final row domain.
    fa_off=[0];fa_rows=[];ua_off=[0];ua_run=[];ua_face=[];ua_row=[];d_off=[0];d_rows=[]
    for oldr in out_rows:
        ar=sorted(face_anc[int(oldr)]); fa_rows.extend(ar); fa_off.append(len(fa_rows))
        uu=sorted(union_anc[int(oldr)])
        if any(rid<0 or fid<0 or row<0 for rid,fid,row in uu): raise RuntimeError('negative/missing union parent identity in retained source ancestry')
        for rid,fid,row in uu: ua_run.append(rid); ua_face.append(fid); ua_row.append(row)
        ua_off.append(len(ua_row))
        dd=sorted(d289_anc[int(oldr)]); d_rows.extend(dd); d_off.append(len(d_rows))
    attrs_out={k:v[out_rows].copy() for k,v in attrs.items()}
    uin_out={k:v[out_rows].copy() for k,v in union_input.items()}
    for localr in new_changed:
        attrs_out['face_source_layer'][localr]=2; attrs_out['face_source_body'][localr]=23; attrs_out['face_source_bone'][localr]=32
        attrs_out['face_source_original_id'][localr]=new_id; attrs_out['face_source_face_id'][localr]=new_changed.index(localr); attrs_out['face_source_identity0'][localr]=new_changed.index(localr); attrs_out['face_source_identity1'][localr]=-1
        for k in uin_out: uin_out[k][localr]=-1
    schema[str(new_id)]={'role':'BONE','body':23,'bone':32,'layer':2,'path':str(OUT/'BONE-derived-126000033-source.npz'),'identity_keys':['face_source_identity0','face_source_identity1'],'identity_face_id_key':'face_source_face_id','face_count':len(new_changed),'derived_operation':'bounded_local_exact_self_intersection_flip_or_link_safe_contraction','derived_parent_source_ids':[100000064],'derived_parent_assets':[{'asset_role':'parent_conditioned_bank32_envelope','path':str(PARENT),'sha256':input_hashes['parent']},{'asset_role':'original_bank32_cube_envelope','path':str(BANK),'sha256':input_hashes['bank']},{'asset_role':'native_replay_bank32','path':str(REPLAY),'sha256':input_hashes['replay']}],'source_geometry_scope':'only locally altered output faces with full input face ancestry CSR','maximum_source_node_displacement_m':final_shift,'maximum_local_surface_deviation_bound_m':flip_surface_bound}
    out_data={k:np.asarray(parent[k]).copy() for k in parent.files}
    out_data['points_world_m']=V.astype(np.float32)
    out_data['faces']=Fout
    out_data.update(attrs_out);out_data.update(uin_out)
    out_data['candidate_face_ancestry_offsets']=np.asarray(fa_off,np.int64);out_data['candidate_face_ancestry_rows']=np.asarray(fa_rows,np.int64)
    out_data['union_face_ancestry_offsets']=np.asarray(ua_off,np.int64);out_data['union_face_ancestry_run_original_ids']=np.asarray(ua_run,np.int64);out_data['union_face_ancestry_face_ids']=np.asarray(ua_face,np.int64);out_data['union_face_ancestry_rows']=np.asarray(ua_row,np.int64)
    out_data['d289_parent_row_indptr']=np.asarray(d_off,np.int64);out_data['d289_parent_rows']=np.asarray(d_rows,np.int64)
    pno=[0];pnr=[]
    for s in node_anc:
        pnr.extend(sorted(s));pno.append(len(pnr))
    out_data['output_vertex_input_parent_offsets']=np.asarray(pno,np.int64);out_data['output_vertex_input_parent_rows']=np.asarray(pnr,np.int64)
    out_data['bank32_local_vertex_representative_rows']=vertex_representative.copy()
    out_data['bank32_local_topology_repair_json']=np.asarray([json.dumps({'schema':'numi.bank32-local-exact-self-repair.v1','parent_candidate_sha256':input_hashes['parent'],'limits':result['limits'],'operations':operations,'max_source_node_displacement_m':final_shift,'accumulated_flip_surface_deviation_bound_m':flip_surface_bound},sort_keys=True,separators=(',',':'))])
    out_data['bank32_local_topology_parent_candidate_sha256']=np.asarray([input_hashes['parent']])
    out_data['face_source_identity_schema_by_original_id_json']=np.asarray([json.dumps(schema,sort_keys=True)])
    candidate_path=OUT/'candidate-f32.npz'; np.savez_compressed(candidate_path,**out_data)
    wide=dict(out_data); wide['points_world_m_f64']=V.copy(); wide.pop('points_world_m',None); wide['faces_f64']=Fout.copy(); wide.pop('faces',None)
    f64_path=OUT/'candidate-f64.npz'; np.savez_compressed(f64_path,**wide)
    # Build the registered source view for altered faces; all original face/node parents remain explicit.
    patch_faces=Fout[np.asarray(new_changed,np.int64)]; patch_used,patch_inv=np.unique(patch_faces.reshape(-1),return_inverse=True); patch_faces_local=patch_inv.reshape(-1,3)
    source_fa_off=[0];source_fa_rows=[];source_ua_orig=[];source_ua_fid=[];source_ua_i0=[];source_ua_i1=[];source_ua_rows=[]
    for localr in new_changed:
        oldr=int(out_rows[localr]); source_fa_rows.extend(sorted(face_anc[oldr])); source_fa_off.append(len(source_fa_rows))
        for rid,fid,row in sorted(union_anc[oldr]):
            source_ua_orig.append(rid);source_ua_fid.append(fid);source_ua_i0.append(fid);source_ua_i1.append(-1);source_ua_rows.append(row)
    source_ua_off=[0]
    for localr in new_changed: source_ua_off.append(source_ua_off[-1]+len(union_anc[int(out_rows[localr])]))
    sourceview=OUT/'BONE-derived-126000033-source.npz'
    np.savez_compressed(sourceview,points_world_m=V[patch_used].astype(np.float32),faces=patch_faces_local,
      source_candidate_face_rows=np.asarray(new_changed,np.int64),source_candidate_vertex_rows=patch_used,
      face_source_layer=np.full(len(new_changed),2,np.int64),face_source_body=np.full(len(new_changed),23,np.int64),face_source_bone=np.full(len(new_changed),32,np.int64),
      face_source_original_id=np.full(len(new_changed),new_id,np.int64),face_source_face_id=np.arange(len(new_changed),dtype=np.int64),face_source_identity0=np.arange(len(new_changed),dtype=np.int64),face_source_identity1=np.full(len(new_changed),-1,np.int64),
      face_ancestry_offsets=np.asarray(source_fa_off,np.int64),face_ancestry_input_cache_rows=np.asarray(source_fa_rows,np.int64),
      face_ancestry_original_ids=np.asarray(source_ua_orig,np.int64),face_ancestry_face_ids=np.asarray(source_ua_fid,np.int64),face_ancestry_identity0=np.asarray(source_ua_i0,np.int64),face_ancestry_identity1=np.asarray(source_ua_i1,np.int64),face_ancestry_union_offsets=np.asarray(source_ua_off,np.int64),face_ancestry_union_input_rows=np.asarray(source_ua_rows,np.int64),
      parent_source_original_ids=np.asarray([100000064],np.int64),parent_assets_json=np.asarray([json.dumps(schema[str(new_id)]['derived_parent_assets'],sort_keys=True)]),
      source_operation=np.asarray(['bounded_local_exact_self_intersection_topology_repair']),source_candidate_sha256=np.asarray([input_hashes['parent']]),source_f64_candidate_sha256=np.asarray([sha(RUN2/'candidate-f64.npz')]),source_cache_sha256=np.asarray([input_hashes['bank']]),operation_log_json=np.asarray([json.dumps(operations,sort_keys=True,separators=(',',':'))]))
    result['candidate_f32_path']=str(candidate_path);result['candidate_f32_sha256']=sha(candidate_path);result['candidate_f64_path']=str(f64_path);result['candidate_f64_sha256']=sha(f64_path);result['derived_source_view_path']=str(sourceview);result['derived_source_view_sha256']=sha(sourceview)
    old_to_new=np.full(len(F),-1,np.int64);old_to_new[out_rows]=np.arange(len(out_rows),dtype=np.int64)
    mapped_pairs=old_to_new[current_pairs] if len(current_pairs) else np.empty((0,2),np.int64)
    np.savez_compressed(OUT/'candidate-self-intersections.npz',face_pairs=mapped_pairs)
else:
    result['candidate_f32_path']=str(PARENT);result['candidate_f32_sha256']=input_hashes['parent'];result['candidate_parent_retained']=True
    if current_count:
        np.savez_compressed(OUT/'candidate-self-intersections.npz',face_pairs=current_pairs)

(OUT/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
(OUT/'operations.json').write_text(json.dumps({'schema':'numi.bank32-local-topology-operation-log.v1','input_candidate_sha256':input_hashes['parent'],'attempts':attempts,'accepted_operations':operations},sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':result['status'],'attempts':trial_count,'accepted':len(operations),'self_pairs_before':8,'self_pairs_after':current_count,'final_structural':current_struct,'final_exact_gates':final_gates,'max_source_node_displacement_m':final_shift,'accumulated_flip_surface_deviation_bound_m':flip_surface_bound,'stop_reason':stop_reason,'elapsed_s':elapsed,'result_path':str(OUT/'result.json')},sort_keys=True,indent=2),flush=True)
if result['status'] in ('IMPROVED_BUT_UNQUALIFIED','NO_IMPROVING_LOCAL_OPERATION'): raise SystemExit(2)
