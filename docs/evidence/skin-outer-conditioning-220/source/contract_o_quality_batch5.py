#!/usr/bin/env python3
from pathlib import Path
import collections, hashlib, importlib.util, json, time, sys
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
W=D/'connected-obstacle-207/physical-transfer-209/geometry-worker/D-connectivity-218/limiting-O-edge-audit-001/quality-failure-inventory-001'
BASE=W/'contraction-batch-001/shape-repair-batch-004/candidate.npz'
OUT=W/'contraction-batch-001/shape-repair-batch-005/contraction-001'
HELPER=D/'connected-obstacle-207/physical-transfer-209/root-edge-flips/local_float32_edge_flip.py'
THRESHOLD_Q=1e-3
MAX_EDGE_M=25e-6
MAX_OPERATIONS=250
MAX_SECONDS=1200.0

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_helper():
    spec=importlib.util.spec_from_file_location('local_edge_flip_owner_for_contraction',HELPER)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod
def edgekey(a,b): return (min(int(a),int(b)),max(int(a),int(b)))
def area_cross(P,F):
    q=P[F]
    return np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
def row_altitudes(P,F):
    q=P[F]
    e=np.stack([np.linalg.norm(q[:,1]-q[:,0],axis=1),np.linalg.norm(q[:,2]-q[:,1],axis=1),np.linalg.norm(q[:,0]-q[:,2],axis=1)],axis=1)
    n=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)
    return n/np.max(e,axis=1)
def row_qualities(P,F):
    q=P[F]
    e=np.stack([np.linalg.norm(q[:,1]-q[:,0],axis=1),np.linalg.norm(q[:,2]-q[:,1],axis=1),np.linalg.norm(q[:,0]-q[:,2],axis=1)],axis=1)
    n=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)
    return n/(np.max(e,axis=1)**2)
def loops_from_faces(F):
    em=collections.defaultdict(list)
    for fi,(a,b,c) in enumerate(F):
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))): em[edgekey(u,v)].append((fi,u,v))
    directed=[x[0][1:] for e,x in em.items() if len(x)==1]
    out=collections.defaultdict(list);inc=collections.defaultdict(list)
    for a,b in directed:out[a].append(b);inc[b].append(a)
    if not all(len(v)==1 for v in out.values()) or not all(len(v)==1 for v in inc.values()): raise RuntimeError('boundary directed degree changed')
    nxt={a:v[0] for a,v in out.items()};todo=set(nxt);lens=[];vertices=set()
    while todo:
        start=min(todo);cur=start;loop=[]
        for _ in range(len(nxt)+1):
            if cur in loop:raise RuntimeError('boundary loop repeated')
            loop.append(cur);vertices.add(cur);todo.discard(cur);cur=nxt[cur]
            if cur==start:break
        if cur!=start:raise RuntimeError('boundary loop did not close')
        lens.append(len(loop))
    return sorted(lens),vertices,em
def edge_add(em,inc,F,row):
    a,b,c=map(int,F[row])
    for v in (a,b,c):inc[v].add(row)
    for u,v in ((a,b),(b,c),(c,a)):em[edgekey(u,v)].add(row)
def edge_remove(em,inc,F,row):
    a,b,c=map(int,F[row])
    for v in (a,b,c):inc[v].discard(row)
    for u,v in ((a,b),(b,c),(c,a)):
        k=edgekey(u,v);em[k].discard(row)
        if not em[k]:del em[k]
def topology(F):
    loops,bverts,em=loops_from_faces(F)
    used=set(map(int,F.ravel()))
    adj={v:set() for v in used}
    for u,v in em:adj[u].add(v);adj[v].add(u)
    comps=0;seen=set()
    for v in sorted(used):
        if v in seen:continue
        comps+=1;stack=[v];seen.add(v)
        while stack:
            q=stack.pop()
            for n in adj[q]:
                if n not in seen:seen.add(n);stack.append(n)
    chi=len(used)-len(em)+len(F)
    genus_numer=2*comps-len(loops)-chi
    if genus_numer<0 or genus_numer%2: raise RuntimeError(f'inconsistent orientable topology c={comps} b={len(loops)} chi={chi}')
    return {'used_vertices':len(used),'edges':len(em),'faces':len(F),'components':comps,'boundary_loops':loops,'boundary_edges':sum(len(x)==1 for x in em.values()),'nonmanifold_edges':sum(len(x)>2 for x in em.values()),'euler_characteristic':chi,'genus':genus_numer//2}

OUT.mkdir(parents=True,exist_ok=False)
z=np.load(BASE,allow_pickle=False)
data={k:z[k].copy() for k in z.files}
P32=np.asarray(data['points_world_m'],dtype=np.float32)
P=P32.astype(np.float64)
F=np.asarray(data['source_O_faces'],dtype=np.int64).copy()
if F.ndim!=2 or F.shape[1]!=3 or P.shape!=(32218,3): raise ValueError(f'unexpected base mesh shape {F.shape} {P.shape}')
if 'faces' in data and not np.array_equal(data['faces'],F): raise ValueError('input faces/source_O_faces mismatch')
base_face_count=len(F);nverts=len(P);input_used_vertex_count=int(len(np.unique(F)))
parent_offsets=np.asarray(data['edge_flip_parent_offsets'],np.int64)
parent_rows=np.asarray(data['edge_flip_parent_face_rows'],np.int64)
parent_orig=np.asarray(data['edge_flip_parent_envelope_original_ids'],np.int64)
parent_src=np.asarray(data['edge_flip_parent_envelope_source_face_rows'],np.int64)
if len(parent_offsets)!=base_face_count+1 or parent_offsets[0]!=0 or parent_offsets[-1]!=len(parent_rows): raise ValueError('input parent-face CSR invalid')
if len(parent_orig)!=len(parent_rows) or len(parent_src)!=len(parent_rows): raise ValueError('input parent metadata CSR length mismatch')
parents=[set(map(int,parent_rows[parent_offsets[i]:parent_offsets[i+1]])) for i in range(base_face_count)]
parent_meta={}
for i,original in enumerate(parent_rows):
    key=int(original); val=(int(parent_orig[i]),int(parent_src[i]))
    if key in parent_meta and parent_meta[key]!=val: raise ValueError(f'conflicting source metadata for parent face {key}')
    parent_meta[key]=val
env_orig=np.asarray(data['envelope_source_original_id'],np.int64).copy()
env_src=np.asarray(data['envelope_source_face_row'],np.int64).copy()
alive=np.ones(base_face_count,dtype=bool)
vertex_map=np.asarray(data.get('O033_to_candidate_vertex_row',np.arange(nverts)),dtype=np.int64).copy() # explicit O033->retained endpoint map
candidate_to_O033=np.asarray(data.get('candidate_vertex_to_O033_row',np.arange(nverts)),dtype=np.int64).copy()
current_face_map=np.asarray(data.get('output_face_to_precontraction_candidate_face_row',np.arange(base_face_count)),dtype=np.int64).copy()
if len(current_face_map)!=base_face_count:raise ValueError('current face map shape mismatch')
changed_face_rows=set()
helper=load_helper()
exact=helper.Flips(P,F)
if any(r is None for r in exact.recs):raise ValueError('base candidate contains an exact Float32 collapsed face')
loops0,boundary_vertices,_boundary_edges=loops_from_faces(F)
edge_map=collections.defaultdict(set)
inc=[set() for _ in range(nverts)]
for row in range(len(F)):
    a,b,c=map(int,F[row])
    for v in (a,b,c):inc[v].add(row)
    for u,v in ((a,b),(b,c),(c,a)):edge_map[edgekey(u,v)].add(row)
# Build a duplicate-triangle index and baseline quality.
face_keys={}
for row,tri in enumerate(F):
    key=tuple(sorted(map(int,tri)))
    if key in face_keys:raise ValueError(f'duplicate base face rows {row}/{face_keys[key]}')
    face_keys[key]=row
alts=np.full(base_face_count,np.nan)
alts[:]=row_qualities(P32,F)
if not np.isfinite(alts).all() or np.any(alts<=0):raise ValueError('base mesh has non-positive Float32 area')
bad={int(r) for r in np.flatnonzero(alts<THRESHOLD_Q)}
focus_base_rows={51204,39551}
focus_rows={int(i) for i,v in enumerate(current_face_map) if int(v) in focus_base_rows}
start=time.monotonic();ops=[];rejects=collections.Counter();max_local_pair_count=0
def link_ok(u,v,fs):
    nu=set()
    for r in inc[u]: nu.update(int(x) for x in F[r] if int(x)!=u)
    nv=set()
    for r in inc[v]: nv.update(int(x) for x in F[r] if int(x)!=v)
    opposite={int(x) for r in fs for x in F[r] if int(x) not in (u,v)}
    return (nu&nv)==opposite
def alt_for_faces(face_rows):
    if len(face_rows)==0:return np.empty(0)
    tris=np.asarray(face_rows,dtype=np.float64)
    e=np.stack([np.linalg.norm(tris[:,1]-tris[:,0],axis=1),np.linalg.norm(tris[:,2]-tris[:,1],axis=1),np.linalg.norm(tris[:,0]-tris[:,2],axis=1)],axis=1)
    cross=np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)
    return cross/(e.max(1)**2)
while bad and len(ops)<MAX_OPERATIONS and time.monotonic()-start<MAX_SECONDS:
    edge_rows={}
    for row in sorted(bad,key=lambda r:(0 if r in focus_rows else 1,alts[r],r)):
        target_vertices=set(map(int,F[row]))
        patch_rows=set().union(*(inc[v] for v in target_vertices))
        for patch_row in sorted(patch_rows):
            tri=F[patch_row]
            for k in range(3):
                e=edgekey(tri[k],tri[(k+1)%3]); fs=tuple(sorted(edge_map.get(e,())))
                if len(fs)==2 and e not in edge_rows:
                    length=float(np.linalg.norm(P[e[0]]-P[e[1]]))
                    if length<=MAX_EDGE_M and e[0] not in boundary_vertices and e[1] not in boundary_vertices:
                        edge_rows[e]=(length,float(alts[row]),int(row),fs)
    candidates=[]
    for e,(length,target_alt,target_row,fs) in edge_rows.items():
        if not link_ok(e[0],e[1],fs):
            rejects['link_condition']+=1;continue
        for keep,drop in ((e[0],e[1]),(e[1],e[0])):
            affected=sorted(inc[drop])
            newfaces={}
            dead=[]
            malformed=False
            for r in affected:
                tri=F[r].copy();tri[tri==drop]=keep
                if len(set(map(int,tri)))<3:dead.append(r)
                else:newfaces[r]=tri
            if set(dead)!=set(fs) or len(dead)!=2:
                rejects['not_exact_two_edge_faces_removed']+=1;continue
            survive=sorted(newfaces)
            if not survive:
                rejects['empty_surviving_star']+=1;continue
            # Reject duplicate topology and any face that becomes zero-area in exact Float32.
            seen=set()
            dup=False
            for r in survive:
                tri=newfaces[r];key=tuple(sorted(map(int,tri)))
                other=face_keys.get(key)
                if (other is not None and other not in affected) or key in seen:dup=True;break
                seen.add(key)
                if exact.record(r,tri) is None:
                    malformed=True;break
            if dup:rejects['duplicate_face']+=1;continue
            if malformed:rejects['exact_f32_degenerate_survivor']+=1;continue
            oldn=np.cross(P[F[survive]][:,1]-P[F[survive]][:,0],P[F[survive]][:,2]-P[F[survive]][:,0])
            newtri=np.asarray([newfaces[r] for r in survive],dtype=np.int64)
            newn=np.cross(P[newtri][:,1]-P[newtri][:,0],P[newtri][:,2]-P[newtri][:,0])
            dots=np.einsum('ij,ij->i',oldn,newn)
            if np.any(dots<=0):
                rejects['orientation_flip']+=1;continue
            old_alt=alts[affected]
            new_alt=alt_for_faces(P[newtri])
            oldbad=int(np.count_nonzero(old_alt<THRESHOLD_Q))
            newbad=int(np.count_nonzero(new_alt<THRESHOLD_Q))
            if newbad>oldbad:
                rejects['sliver_count_increased']+=1;continue
            oldmin=float(np.min(old_alt));newmin=float(np.min(new_alt))
            if newbad==oldbad and newmin<oldmin*1.2:
                rejects['no_local_quality_progress']+=1;continue
            # Score strictly by fewer slivers, then local minimum and mean altitude, then shorter displacement.
            ratio=newmin/oldmin if oldmin>0 else float('inf')
            meanratio=float(np.mean(new_alt)/np.mean(old_alt[np.isfinite(old_alt)])) if len(old_alt) and np.mean(old_alt)>0 else 1.
            score=(oldbad-newbad,ratio,meanratio,-length,1 if keep<drop else 0)
            candidates.append({'edge':e,'length':length,'target_row':target_row,'target_alt':target_alt,'fs':fs,'keep':keep,'drop':drop,'affected':affected,'dead':dead,'survive':survive,'newfaces':newfaces,'oldbad':oldbad,'newbad':newbad,'oldmin':oldmin,'newmin':newmin,'score':score,'newalt':new_alt,'oldn':oldn,'newn':newn})
    candidates.sort(key=lambda x:x['score'],reverse=True)
    chosen=None
    for cand in candidates:
        affected=cand['affected'];survive=cand['survive'];dead=cand['dead'];newfaces=cand['newfaces']
        saved={}
        for r in affected:
            saved[r]=(exact.recs[r],exact.lo[r].copy(),exact.hi[r].copy())
            if r in dead: exact.recs[r]=None
            else:
                q=P32[newfaces[r]].astype(np.float64)
                exact.lo[r]=q.min(0);exact.hi[r]=q.max(0)
        try:
            pair_count=exact.pairs(survive,newfaces=newfaces)
        finally:
            for r,(rec,lo,hi) in saved.items():
                exact.recs[r]=rec;exact.lo[r]=lo;exact.hi[r]=hi
        if pair_count is None or pair_count>0:
            rejects['exact_local_self_pairs']+=1
            max_local_pair_count=max(max_local_pair_count,int(pair_count or 0))
            continue
        cand['exact_local_self_pairs_after']=int(pair_count)
        chosen=cand;break
    if chosen is None:
        break
    e=chosen['edge'];keep=chosen['keep'];drop=chosen['drop'];affected=chosen['affected'];dead=chosen['dead'];survive=chosen['survive'];newfaces=chosen['newfaces']
    old_face_keys={r:tuple(sorted(map(int,F[r]))) for r in affected}
    union_parents=set().union(*(parents[r] for r in affected))
    before_values={'affected':affected[:],'bad_rows_before':sorted(r for r in affected if r in bad),'source_markers_before':sorted(set(map(int,env_orig[affected]))),'source_face_rows_before':sorted(set(map(int,env_src[affected])))}
    for r in affected: edge_remove(edge_map,inc,F,r)
    for r in affected: face_keys.pop(old_face_keys[r],None)
    for r in dead:
        alive[r]=False;F[r]=np.array([keep,keep,keep],dtype=np.int64);alts[r]=np.nan;bad.discard(r);parents[r]=set()
    for r in survive:
        F[r]=newfaces[r];edge_add(edge_map,inc,F,r);face_keys[tuple(sorted(map(int,F[r])))]=r
        exact.F[r]=F[r];exact.recs[r]=exact.record(r,F[r]);q=P32[F[r]].astype(np.float64);exact.lo[r]=q.min(0);exact.hi[r]=q.max(0)
        parents[r]=set(union_parents);env_orig[r]=-1;env_src[r]=-1;changed_face_rows.add(r)
    for r in dead:
        exact.F[r]=F[r];exact.recs[r]=None
        exact.lo[r]=P32[F[r]].astype(np.float64).min(0);exact.hi[r]=P32[F[r]].astype(np.float64).max(0)
    # Remap every original O033 support row through the explicit contraction endpoint, preserving full point array.
    vertex_map[vertex_map==drop]=keep;vertex_map[drop]=keep
    for r in affected:
        bad.discard(r)
        if alive[r]:
            alts[r]=float(row_qualities(P32,F[r:r+1])[0])
            if alts[r]<THRESHOLD_Q:bad.add(r)
    op={'operation_index':len(ops),'edge_vertex_rows':[int(e[0]),int(e[1])],'kept_vertex_row':int(keep),'collapsed_vertex_row':int(drop),'collapse_length_um':float(chosen['length']*1e6),'max_source_coordinate_change_um':float(chosen['length']*1e6),'face_rows_affected_input_candidate':affected,'face_rows_removed_input_candidate':dead,'face_rows_surviving_changed_input_candidate':survive,'target_sliver_face_row':int(chosen['target_row']),'target_face_quality_before':float(chosen['target_alt']),'local_bad_faces_before_after':[int(chosen['oldbad']),int(chosen['newbad'])],'local_min_quality_before_after':[float(chosen['oldmin']),float(chosen['newmin'])],'local_exact_self_pairs_after':int(chosen['exact_local_self_pairs_after']),'link_condition_pass':True,'boundary_vertices_touched':False,'source_marker_union_before':before_values['source_markers_before'],'source_face_rows_union_before':before_values['source_face_rows_before'],'parent_source_face_rows_after':sorted(union_parents),'max_normal_rotation_rad':float(np.max(np.arccos(np.clip(np.einsum('ij,ij->i',chosen['oldn'],chosen['newn'])/(np.linalg.norm(chosen['oldn'],axis=1)*np.linalg.norm(chosen['newn'],axis=1)),-1,1)))) if len(survive) else 0.0,'score':list(map(float,chosen['score'][:4]))+[int(chosen['score'][4])]}
    ops.append(op)
    (OUT/'progress.json').write_text(json.dumps({'accepted_operations':len(ops),'remaining_q_below_0p001':len(bad),'elapsed_seconds':time.monotonic()-start,'last_operation':op},separators=(',',':'))+'\n')
    if len(ops)%5==0 or not bad:
        print(json.dumps({'accepted_operations':len(ops),'remaining_q_below_0p001':len(bad),'last_face_quality':op['target_face_quality_before'],'elapsed_seconds':time.monotonic()-start}),flush=True)

out_rows=np.flatnonzero(alive).astype(np.int64);Fout=F[out_rows]
# Full topology check: each contraction passed the interior link condition; now recompute loops, components and genus.
topo=topology(Fout)
if topo['boundary_loops']!=loops0 or topo['nonmanifold_edges']!=0 or topo['components']!=1 or topo['genus']!=0: raise RuntimeError(f'topology gate failed: {topo}')
all_alt=row_altitudes(P32,Fout)
all_quality=row_qualities(P32,Fout)
if not np.isfinite(all_alt).all() or np.any(all_alt<=0) or not np.isfinite(all_quality).all() or np.any(all_quality<=0): raise RuntimeError('final emitted Float32 mesh has non-positive area or quality')
remaining=np.flatnonzero(all_quality<THRESHOLD_Q).astype(np.int64)
before_bad_count=int(np.count_nonzero(row_qualities(P32,np.asarray(data['source_O_faces'],np.int64))<THRESHOLD_Q))
# Check every surviving face has ancestry and build complete per-face CSR.
groups=[sorted(parents[int(r)]) for r in out_rows]
if any(not g for g in groups): raise RuntimeError('surviving face lacks source ancestry')
ancestry_offsets=np.r_[0,np.cumsum([len(g) for g in groups])].astype(np.int64)
ancestry_rows=np.asarray([x for g in groups for x in g],np.int64)
ancestry_orig=np.asarray([parent_meta[int(x)][0] for x in ancestry_rows],np.int64)
ancestry_src=np.asarray([parent_meta[int(x)][1] for x in ancestry_rows],np.int64)
changed_in=sorted(changed_face_rows)
# Preserve the full point table and explicit maps: support row -> candidate row is never inferred by proximity.
data['points_world_m']=P32.copy()
data['source_O_faces']=Fout.copy();data['faces']=Fout.copy()
data['output_face_to_precontraction_candidate_face_row']=current_face_map[out_rows]
data['output_face_to_shapequality_contraction_input_face_row']=out_rows.copy()
data['candidate_vertex_to_O033_row']=candidate_to_O033
data['O033_to_candidate_vertex_row']=vertex_map.copy()
data['contracted_O033_vertex_rows']=np.flatnonzero(vertex_map!=np.arange(nverts)).astype(np.int64)
data['face_ancestry_offsets']=ancestry_offsets
data['face_ancestry_parent_face_rows']=ancestry_rows
data['face_ancestry_source_original_ids']=ancestry_orig
data['face_ancestry_source_face_rows']=ancestry_src
data['edge_flip_parent_offsets']=ancestry_offsets
data['edge_flip_parent_face_rows']=ancestry_rows
data['edge_flip_parent_envelope_original_ids']=ancestry_orig
data['edge_flip_parent_envelope_source_face_rows']=ancestry_src
data['envelope_source_original_id']=env_orig[out_rows]
data['envelope_source_face_row']=env_src[out_rows]
data['edge_flip_parent_semantics']=np.asarray(['inherited edge-flip parent union plus link-condition edge-contraction star union; see face_ancestry_* CSR'])
data['contraction_parent_candidate_sha256']=np.asarray([sha(BASE)])
data['contraction_helper_sha256']=np.asarray([sha(Path(__file__))])
np.savez_compressed(OUT/'candidate.npz',**data)
# Verify fields and position/map integrity as saved.
check=np.load(OUT/'candidate.npz',allow_pickle=False)
if not np.array_equal(check['faces'],check['source_O_faces']):raise RuntimeError('faces/source_O_faces mismatch in final')
if not np.array_equal(check['points_world_m'].view(np.uint32),P32.view(np.uint32)):raise RuntimeError('point positions changed')
if not np.array_equal(check['candidate_vertex_to_O033_row'],candidate_to_O033):raise RuntimeError('candidate vertex map mismatch')
rep={'schema':'numi.connectedfit-O-link-condition-q-quality-contraction.v1',
 'status':'CANDIDATE_REQUIRES_FINAL_CHANGED_FACE_SELF_AND_185_BONE_GATES' if len(remaining)==0 else 'BOUNDED_CONTRACTION_STOPPED_WITH_Q_POOR_FACES',
 'input_path':str(BASE),'input_sha256':sha(BASE),'output_path':str(OUT/'candidate.npz'),'output_sha256':sha(OUT/'candidate.npz'),
 'helper_path':str(HELPER),'helper_sha256':sha(HELPER),'script_path':str(Path(__file__)),'script_sha256':sha(Path(__file__)),
 'quality_threshold':THRESHOLD_Q,'quality_definition':'Float32 cross-product norm divided by squared longest edge; dimensionless.',
 'max_edge_um':MAX_EDGE_M*1e6,'accepted_operations':len(ops),'operations':ops,'rejections':dict(rejects),
 'before_topology':{'used_vertices':input_used_vertex_count,'faces':base_face_count,'boundary_loops':loops0,'genus':0},'after_topology':topo,
 'before_q_below_0p001':before_bad_count,'after_q_below_0p001':len(remaining),'remaining_q_below_0p001_face_rows':remaining.tolist(),
 'after_min_quality':float(all_quality.min()),'after_min_altitude_um':float(all_alt.min()*1e6),
 'changed_input_candidate_face_rows':changed_in,'removed_input_candidate_face_rows':sorted(set(range(base_face_count))-set(map(int,out_rows))),
 'positions_bitwise_unchanged':True,'source_node_count_preserved':nverts,
 'O033_to_candidate_vertex_row':'explicit contraction endpoint remap; full point table retained; no nearest neighbor',
 'faces_and_source_O_faces_equal':True,'face_ancestry_csr_rows':int(len(ancestry_rows)),'all_surviving_faces_have_ancestry':True,
 'all_float32_faces_positive_area':True,'all_positive_area_exact_float32':True,'local_exact_self_pairs_after_each_operation':0,
 'local_exact_self_gate_scope':'Every trial was exact-checked against all AABB-near O faces before commit; exact failures were backed out. Final changed-face self and all-185 bone audit pending.',
 'geometry_admission':False,'elapsed_seconds':time.monotonic()-start}
(OUT/'receipt.json').write_text(json.dumps(rep,indent=2)+'\n')
(OUT/'progress.json').unlink(missing_ok=True)
print(json.dumps({k:v for k,v in rep.items() if k not in ('operations','remaining_q_below_0p001_face_rows','changed_input_candidate_face_rows','removed_input_candidate_face_rows')}),flush=True)
