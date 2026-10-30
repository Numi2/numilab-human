#!/Users/numi/numi-human-prep-venv-20261005/bin/python
from pathlib import Path
import collections, hashlib, json, math, time
import numpy as np

P = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
OUT = P / 'fine-physical-volume-706/local-microedge-condition-001'
PLAN = P / 'fine-physical-volume-706/plan-003/manifest.json'
INPUTS = {
    'O': (P/'paired-head-reference-fit-684/run-049-current-witness-refit-signed-id/output/candidate-O-f32.npz', '1e828ea593d941eeb0e071fa84d10d8499c6eca3d680e43d9d0aaf0331a46afa'),
    'I': (P/'paired-head-reference-fit-684/run-050-I-self-child-reference-refinement/attempt-002/candidate-I-f32-local-child-refined.npz', 'a702f368961cae10aeb2e9be38ac05cfdd422907d2beeab1d953d597a0d73469'),
    'D': (P/'paired-head-reference-fit-684/run-055-D-seam-disk-remesh/d010-ring1-composition-002/compact-unused-storage-001/candidate-D-d010-ring1-composed-schema62000204-compact-unused.npz', 'f18ea8a8f38afe6a724b8fb840ebe97efb7568fbbf98ea8f0c308eef1bf538d0'),
}
MAX_DISPLACEMENT_M = 1e-6
MAX_ITERATIONS = 4
SKIN_SKIPPED_FACE_ROWS = {'I': [113265, 113266]}
LEGACY_COLLAPSE_OWNER = P/'paired-head-reference-fit-684/run-010-two-link-valid-collapses/source/collapse_and_field_684.py'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def f32_cross(points, tri):
    q=np.asarray(points[np.asarray(tri,np.int64)],dtype=np.float32).astype(np.float64)
    return np.cross(q[1]-q[0],q[2]-q[0])

def edge_tables(faces):
    inc=collections.defaultdict(list)
    neigh=[set() for _ in range(int(np.max(faces))+1)]
    vfaces=[[] for _ in range(int(np.max(faces))+1)]
    directed=collections.defaultdict(list)
    for r,t in enumerate(np.asarray(faces,np.int64)):
        for v in map(int,t):vfaces[v].append(r)
        for k in range(3):
            a,b=map(int,(t[k],t[(k+1)%3]));e=(min(a,b),max(a,b))
            inc[e].append(r);directed[e].append((a,b));neigh[a].add(b);neigh[b].add(a)
    boundary={e for e,rr in inc.items() if len(rr)==1}
    return inc,directed,neigh,vfaces,boundary

def topology(points, faces, expected_boundary=None, expected_euler=None, expected_components=None):
    F=np.asarray(faces,np.int64); n=len(points)
    inc,directed,_,_,boundary=edge_tables(F)
    bad=[]
    for e,rr in inc.items():
        if len(rr)>2: bad.append((e,'incidence',len(rr)))
        elif len(rr)==2 and directed[e][0]==directed[e][1]: bad.append((e,'same_direction'))
        elif len(rr)<1: bad.append((e,'empty'))
    dups=collections.Counter(tuple(sorted(map(int,t))) for t in F)
    duprows=[k for k,v in dups.items() if v>1]
    used=np.unique(F)
    # Exact F32 nonzero triangle check.
    q=np.asarray(points,np.float32)[F].astype(np.float64)
    cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area2=np.linalg.norm(cross,axis=1)
    zero=int(np.count_nonzero((area2==0)|(~np.isfinite(area2))))
    # Vertex links must be one cycle for interior vertices or one path for boundary vertices.
    links={int(v):{} for v in used}
    for t in F:
        a,b,c=map(int,t)
        for v,x,y in ((a,b,c),(b,c,a),(c,a,b)):
            links[v].setdefault(x,set()).add(y);links[v].setdefault(y,set()).add(x)
    bverts={v for e in boundary for v in e};badlinks=[]
    for v,adj in links.items():
        deg=[len(q) for q in adj.values()]
        if not deg: badlinks.append((v,'empty'));continue
        want=2 if v in bverts else 0
        if ((want and deg.count(1)!=2) or (not want and any(d!=2 for d in deg)) or any(d>2 or d==0 for d in deg)):
            badlinks.append((v,'degree',deg.count(1),max(deg)))
        seen=set();stack=[next(iter(adj))]
        while stack:
            u=stack.pop()
            if u in seen:continue
            seen.add(u);stack.extend(adj[u]-seen)
        if len(seen)!=len(adj):badlinks.append((v,'disconnected',len(seen),len(adj)))
    # Components over used vertices only.
    parent={int(v):int(v) for v in used}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    for t in F:
        a=int(t[0])
        for b in map(int,t[1:]):parent[find(a)]=find(b)
    comps=len({find(v) for v in parent})
    euler=int(len(used)-len(inc)+len(F))
    report={'point_rows':n,'used_point_rows':int(len(used)),'face_rows':int(len(F)),'edge_rows':int(len(inc)),'euler':euler,'components':comps,'boundary_edges':len(boundary),'nonmanifold_or_orientation_count':len(bad),'nonmanifold_or_orientation_examples':bad[:12],'duplicate_face_count':len(duprows),'bad_vertex_link_count':len(badlinks),'bad_vertex_link_examples':badlinks[:12],'f32_zero_or_nonfinite_faces':zero,'unused_point_rows':int(n-len(used))}
    if expected_boundary is not None:
        report['boundary_delta_edges']=len(boundary ^ set(expected_boundary))
    if expected_euler is not None:report['euler_delta']=euler-int(expected_euler)
    if expected_components is not None:report['component_delta']=comps-int(expected_components)
    return report,boundary

def load_asset(role, path, expected_sha, plan):
    if sha(path)!=expected_sha: raise ValueError(f'{role} input SHA mismatch')
    with np.load(path,allow_pickle=False) as z: data={k:np.array(z[k],copy=True) for k in z.files}
    pts=np.asarray(data['points_world_m'],np.float32)
    faces=np.asarray(data['faces'],np.int64)
    if pts.ndim!=2 or pts.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3:raise ValueError(f'{role} geometry shape')
    if faces.min()<0 or faces.max()>=len(pts):raise ValueError(f'{role} face index out of bounds')
    n=len(pts);nf=len(faces)
    unused=set(range(n))-set(map(int,np.unique(faces)))
    if unused:raise ValueError(f'{role} input unexpectedly contains unused storage rows {sorted(unused)[:8]}')
    pins=set()
    for r in plan['protected149']['surface_routes']:
        if r.get('role')==role:pins.add(int(r['surface_vertex_row']))
    if role=='D':
        routes=plan['canonical_D_attachment_proof']['attachment_routes']
        if isinstance(routes,dict):routes=routes.values()
        for r in routes:
            if isinstance(r,dict) and int(r.get('surface_vertex_row',-1))>=0:pins.add(int(r['surface_vertex_row']))
    if any(v<0 or v>=n for v in pins):raise ValueError(f'{role} protected row outside point domain')
    _,_,_,_,boundary=edge_tables(faces)
    bverts={x for e in boundary for x in e}
    if not bverts.issubset(pins):raise ValueError(f'{role} has unprotected open-boundary vertices: {sorted(bverts-pins)[:8]}')
    base_topo,_=topology(pts,faces)
    # Parent ancestry starts with each input face row; each later collapse propagates the whole touched star.
    parent_faces=[{r} for r in range(nf)]
    point_members=[{r} for r in range(n)]
    point_rep=list(range(n));face_rep=list(range(nf))
    protected=set(pins);original_pins=set(pins)
    current_points=pts.copy();current_faces=faces.copy()
    final_point_map=np.arange(n,dtype=np.int64);final_face_map=np.arange(nf,dtype=np.int64)
    iteration_receipts=[];ops_all=[];skipped_counts=[]
    for iteration in range(1,MAX_ITERATIONS+1):
        inc,directed,neighbors,vertex_faces,boundary=edge_tables(current_faces)
        current_bverts={v for e in boundary for v in e}
        # Pin protection follows the unique preserved target rows.
        if any(len(point_members[v])!=1 for v in protected):raise ValueError(f'{role} protected target was merged')
        lengths=[];eligible=[];reason=collections.Counter()
        for e,rr in inc.items():
            d=float(np.linalg.norm(current_points[e[0]].astype(np.float64)-current_points[e[1]].astype(np.float64)))
            if not d<MAX_DISPLACEMENT_M:continue
            lengths.append(d)
            if len(rr)!=2:reason['not_two_face_edge']+=1;continue
            u,v=e
            if u in protected or v in protected or u in current_bverts or v in current_bverts:reason['pinned_or_boundary_endpoint']+=1;continue
            opposite={int(x) for r in rr for x in current_faces[r] if int(x) not in e}
            if (neighbors[u]&neighbors[v])!=opposite or len(opposite)!=2:reason['link_condition']+=1;continue
            # Keep the collapse on a single typed surface/source stratum.
            semantic=[]
            for r in rr:
                semantic.append(tuple(int(data[k][face_rep[r]]) for k in ('face_source_layer','face_source_body','face_source_bone') if k in data))
            if len(set(semantic))>1:reason['typed_stratum_boundary']+=1;continue
            parent_members=point_members[u]|point_members[v]
            best=None
            for keep,remove in ((u,v),(v,u)):
                maxmove=max((float(np.linalg.norm(np.asarray(data['points_world_m'][orig],np.float32).astype(np.float64)-np.asarray(data['points_world_m'][point_rep[keep]],np.float32).astype(np.float64))) for orig in parent_members),default=0.)
                if maxmove>MAX_DISPLACEMENT_M:continue
                star=set(vertex_faces[keep])|set(vertex_faces[remove])
                dropped=[];areas=[];min_alt_before=float('inf');min_alt_after=float('inf');orient_bad=False
                for r in star:
                    tri=list(map(int,current_faces[r]));oldcross=f32_cross(current_points,tri)
                    mapped=[keep if x==remove else x for x in tri]
                    if len(set(mapped))<3:dropped.append(r);continue
                    newcross=f32_cross(current_points,mapped);n2=float(np.linalg.norm(newcross));o2=float(np.linalg.norm(oldcross))
                    if not np.isfinite(n2) or n2==0 or float(np.dot(oldcross,newcross))<=0:orient_bad=True;break
                    oldp=current_points[tri].astype(np.float32).astype(np.float64);newp=current_points[np.asarray(mapped)].astype(np.float32).astype(np.float64)
                    oldedges=[np.linalg.norm(oldp[1]-oldp[0]),np.linalg.norm(oldp[2]-oldp[1]),np.linalg.norm(oldp[0]-oldp[2])]
                    newedges=[np.linalg.norm(newp[1]-newp[0]),np.linalg.norm(newp[2]-newp[1]),np.linalg.norm(newp[0]-newp[2])]
                    min_alt_before=min(min_alt_before,o2/max(oldedges or [1.],default=1.))
                    min_alt_after=min(min_alt_after,n2/max(newedges or [1.],default=1.))
                if orient_bad:continue
                score=(min_alt_after, min_alt_after/max(min_alt_before,1e-300), -maxmove, -keep)
                candidate=(score,keep,remove,d,maxmove,sorted(star),sorted(dropped),sorted(opposite))
                if best is None or candidate[0]>best[0]:best=candidate
            if best is None:reason['no_orientation_and_displacement_direction']+=1;continue
            eligible.append(best)
        # Known TetGen witness stars are scheduled first, then all remaining eligible submicron edges by length.
        priority=set()
        current_row_for_input={orig:r for r,members in enumerate(point_members) for orig in members}
        if role=='I':
            # Keep the actual skipped-face and smallest-edge witnesses at the front of each round.
            for r,parents in enumerate(parent_faces):
                if parents & set(SKIN_SKIPPED_FACE_ROWS['I']):
                    for k in range(3):priority.add(tuple(sorted(map(int,(current_faces[r][k],current_faces[r][(k+1)%3])))))
            if iteration==1:
                for u,v in ((49125,49128),(49122,49126),(52161,52162)):
                    if u in current_row_for_input and v in current_row_for_input:
                        eu,ev=current_row_for_input[u],current_row_for_input[v]
                        if tuple(sorted((eu,ev))) in inc:priority.add(tuple(sorted((eu,ev))))
        if role=='O' and iteration==1 and 13530 in current_row_for_input and 13531 in current_row_for_input:
            priority.add(tuple(sorted((current_row_for_input[13530],current_row_for_input[13531]))))
        eligible.sort(key=lambda c:(0 if tuple(sorted((c[1],c[2]))) in priority else 1,c[3],c[1],c[2]))
        selected=[];reserved=set()
        for cand in eligible:
            _,keep,remove,d,maxmove,star,dropped,opp=cand
            ring={keep,remove}|neighbors[keep]|neighbors[remove]
            if ring & reserved:reason['overlapping_local_stars_this_iteration']+=1;continue
            selected.append(cand);reserved|=ring
        if not selected:
            iteration_receipts.append({'iteration':iteration,'selected_collapses':0,'eligible_link_safe_edges':len(eligible),'reason_counts':dict(reason),'stalled':True})
            skipped_counts.append(dict(reason));break
        # Batch mapping is valid because selected 1-ring stars are disjoint.
        rep=np.arange(len(current_points),dtype=np.int64)
        touched_parent={};ops=[]
        for _,keep,remove,d,maxmove,star,dropped,opp in selected:
            if rep[keep]!=keep or rep[remove]!=remove:raise ValueError('operation endpoints overlap')
            rep[remove]=keep
            allparents=set().union(*(parent_faces[r] for r in star))
            touched_parent.update({r:allparents for r in star})
            cluster=point_members[keep]|point_members[remove]
            for orig in cluster:
                disp=float(np.linalg.norm(np.asarray(data['points_world_m'][orig],np.float32).astype(np.float64)-np.asarray(data['points_world_m'][point_rep[keep]],np.float32).astype(np.float64)))
                if disp>MAX_DISPLACEMENT_M:raise ValueError('per-original point displacement exceeded 1 um')
            ops.append({'role':role,'iteration':iteration,'keep_current_row':keep,'retire_current_row':remove,'edge_length_m':d,'max_original_point_displacement_m':maxmove,'star_current_face_rows':star,'dropped_current_face_rows':dropped,'link_opposite_vertices':opp,'parent_candidate_input_face_rows':sorted(allparents)})
        mapped=rep[current_faces]
        valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
        kept_current_faces=np.flatnonzero(valid);dropped_current_faces=np.flatnonzero(~valid)
        used=np.unique(mapped[valid]);curr_to_new=np.full(len(current_points),-1,np.int64);curr_to_new[used]=np.arange(len(used),dtype=np.int64)
        new_faces=curr_to_new[mapped[valid]]
        new_points=np.asarray(current_points[used],np.float32)
        new_point_members=[];new_point_rep=[]
        # Merge original-input vertex clusters under the selected endpoint map.
        member_by_rep=collections.defaultdict(set)
        for oldv in range(len(current_points)):member_by_rep[int(rep[oldv])].update(point_members[oldv])
        for oldv in used:
            new_point_members.append(member_by_rep[int(oldv)])
            new_point_rep.append(point_rep[int(oldv)])
        new_face_parents=[];new_face_rep=[]
        for oldface in kept_current_faces:
            new_face_parents.append(set(touched_parent.get(int(oldface),parent_faces[int(oldface)])))
            new_face_rep.append(face_rep[int(oldface)])
        expected_boundary=set()
        for a,b in boundary:
            na,nb=int(curr_to_new[int(rep[a])]),int(curr_to_new[int(rep[b])])
            if na<0 or nb<0:raise ValueError('boundary point retired')
            expected_boundary.add(tuple(sorted((na,nb))))
        before_topo,_=topology(current_points,current_faces)
        after_topo,new_boundary=topology(new_points,new_faces,expected_boundary,before_topo['euler'],before_topo['components'])
        if after_topo['nonmanifold_or_orientation_count'] or after_topo['duplicate_face_count'] or after_topo['bad_vertex_link_count'] or after_topo['f32_zero_or_nonfinite_faces'] or after_topo.get('boundary_delta_edges') or after_topo.get('euler_delta') or after_topo.get('component_delta'):
            raise ValueError(f'{role} iteration {iteration} topology failed: {after_topo}')
        # Update original point/face maps through this batch.
        # Build final maps after each round from memberships and representative rows.
        original_to_new=np.full(len(data['points_world_m']),-1,np.int64)
        for outrow,members in enumerate(new_point_members):
            for orig in members: original_to_new[int(orig)]=outrow
        face_input_to_new=np.full(len(data['faces']),-1,np.int64)
        for outrow,rep_input_face in enumerate(new_face_rep):
            face_input_to_new[int(rep_input_face)]=outrow
        opcount=len(ops);ops_all.extend(ops)
        iteration_receipts.append({'iteration':iteration,'selected_collapses':opcount,'eligible_link_safe_edges':len(eligible),'changed_current_faces':int(sum(len(q['star_current_face_rows']) for q in ops)),'dropped_current_faces':int(len(dropped_current_faces)),'reason_counts':dict(reason),'max_input_point_displacement_m':float(max(np.linalg.norm(np.asarray(data['points_world_m'][orig],np.float32).astype(np.float64)-np.asarray(data['points_world_m'][new_point_rep[outrow]],np.float32).astype(np.float64)) for outrow,members in enumerate(new_point_members) for orig in members)),'topology':after_topo})
        current_points=new_points;current_faces=new_faces;point_members=new_point_members;point_rep=new_point_rep;parent_faces=new_face_parents;face_rep=new_face_rep;protected={int(curr_to_new[int(rep[v])]) for v in protected}
        # Keep the row mapping to the original candidate through every iteration.
        final_point_map=original_to_new;final_face_map=face_input_to_new
        if after_topo['unused_point_rows']!=0:raise ValueError(f'{role} iteration left unused point storage')
    # Build output arrays from original input row representatives.
    final_orig_rows=np.asarray(point_rep,np.int64);final_face_orig_rows=np.asarray(face_rep,np.int64)
    out={}
    point_csr_keys={
      'reference_original_input_vertex_offsets':['reference_original_input_vertex_rows'],
      'reference_vertex_parent_face_offsets':['reference_vertex_parent_face_rows','reference_vertex_parent_face_barycentric'],
      'output_vertex_input_parent_offsets':['output_vertex_input_parent_rows'],
    }
    face_csr_keys={
      'face_parent_offsets':['face_parent_rows'],
      'edge_flip_parent_offsets':['edge_flip_parent_face_rows'],
      'face_ancestry_offsets':['face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows'],
      'candidate_face_ancestry_offsets':['candidate_face_ancestry_rows'],
      'union_face_ancestry_offsets':['union_face_ancestry_rows','union_face_ancestry_run_original_ids','union_face_ancestry_face_ids'],
      'operational_output_face_parent_offsets':['operational_output_face_parent_rows'],
    }
    skipkeys=set(point_csr_keys)|set(sum((v for v in point_csr_keys.values()),[]))|set(face_csr_keys)|set(sum((v for v in face_csr_keys.values()),[]))
    n0=len(data['points_world_m']);nf0=len(data['faces'])
    for k,v in data.items():
        if k in skipkeys or k in ('points_world_m','faces'):continue
        arr=np.asarray(v)
        if arr.ndim and arr.shape[0]==n0:out[k]=arr[final_orig_rows]
        elif arr.ndim and arr.shape[0]==nf0:out[k]=arr[final_face_orig_rows]
        else:out[k]=v
    out['points_world_m']=np.asarray(data['points_world_m'])[final_orig_rows]
    # Build final faces from original candidate triangles and the original→output point map.
    out['faces']=np.asarray(current_faces,np.int64)
    final_face_map=np.full(nf0,-1,np.int64)
    for outrow,rep_input_face in enumerate(final_face_orig_rows):final_face_map[int(rep_input_face)]=outrow
    # Preserve transitive vertex lineage through the original input point domain.
    point_offsets=[0];point_rows=[]
    for members in point_members:
        point_rows.extend(sorted(map(int,members)));point_offsets.append(len(point_rows))
    out['microedge_parent_input_point_offsets']=np.asarray(point_offsets,np.int64)
    out['microedge_parent_input_point_rows']=np.asarray(point_rows,np.int64)
    out['microedge_input_point_to_output_row']=np.asarray(final_point_map,np.int64)
    # Preserve transitive face lineage through the original input candidate face domain.
    face_offsets=[0];face_rows=[]
    for members in parent_faces:
        face_rows.extend(sorted(map(int,members)));face_offsets.append(len(face_rows))
    out['microedge_parent_input_face_offsets']=np.asarray(face_offsets,np.int64)
    out['microedge_parent_input_face_rows']=np.asarray(face_rows,np.int64)
    out['microedge_input_face_to_representative_output_row']=np.asarray(final_face_map,np.int64)
    out['microedge_parent_candidate_sha256']=np.asarray([expected_sha],dtype='<U64')
    out['microedge_schema']=np.asarray(['numi.fine-physical-volume.microedge-contracted-surface.v1'],dtype='<U96')
    out['microedge_max_displacement_m']=np.asarray([MAX_DISPLACEMENT_M],np.float64)
    # Rebuild each existing source CSR by unioning its full original records across each output point/face cluster.
    for offsets_key,row_keys,clusters in [(k,v,point_members) for k,v in point_csr_keys.items() if k in data] + [(k,v,parent_faces) for k,v in face_csr_keys.items() if k in data]:
        offsets=np.asarray(data[offsets_key],np.int64)
        if len(offsets)==n0+1:
            domain='point'
        elif len(offsets)==nf0+1:
            domain='face'
        else:raise ValueError(f'{role} unexpected source CSR {offsets_key} length')
        rebuilt=[[] for _ in clusters]
        aligned=[np.asarray(data[k]) for k in row_keys]
        for outrow,members in enumerate(clusters):
            records=set()
            for orig in members:
                for j in range(int(offsets[int(orig)]),int(offsets[int(orig)+1])):
                    records.add(tuple(tuple(np.asarray(a[j]).reshape(-1).tolist()) for a in aligned))
            rebuilt[outrow]=sorted(records,key=repr)
        new_offsets=[0];flat=[[] for _ in row_keys]
        for records in rebuilt:
            for rec in records:
                for j,val in enumerate(rec):
                    flat[j].append(val)
            new_offsets.append(len(flat[0]))
        out[offsets_key]=np.asarray(new_offsets,np.int64)
        for j,k in enumerate(row_keys):
            old=np.asarray(data[k]);tail=old.shape[1:]
            if tail:
                out[k]=np.asarray(flat[j],dtype=old.dtype).reshape((-1,)+tail)
            else:out[k]=np.asarray(flat[j],dtype=old.dtype)
    # Verify ancestry closure and pin preservation.
    if len(out['microedge_input_point_to_output_row'])!=n0 or np.any(out['microedge_input_point_to_output_row']<0):raise ValueError(f'{role} lost a source point parent')
    if set(map(int,out['microedge_parent_input_point_rows']))!=set(range(n0)):
        raise ValueError(f'{role} output point CSR does not cover every input point')
    if set(map(int,out['microedge_parent_input_face_rows']))!=set(range(nf0)):raise ValueError(f'{role} output face CSR does not cover every input face')
    route_rows=sorted(original_pins)
    pinproof=[]
    for old in route_rows:
        new=int(final_point_map[old])
        if new<0 or not np.array_equal(np.asarray(data['points_world_m'][old],np.float32).view(np.uint32),np.asarray(out['points_world_m'][new],np.float32).view(np.uint32)):
            raise ValueError(f'{role} protected target moved or disappeared row {old}')
        pinproof.append({'input_point_row':int(old),'output_point_row':new,'coordinate_f32_hex':np.asarray(data['points_world_m'][old],np.float32).view(np.uint32).tolist()})
    final_topo,_=topology(out['points_world_m'],out['faces'])
    result={'role':role,'input_path':str(path),'input_sha256':expected_sha,'output_point_rows':len(out['points_world_m']),'output_face_rows':len(out['faces']),'initial_topology':base_topo,'final_topology':final_topo,'iterations':iteration_receipts,'operations':len(ops_all),'dropped_face_rows':int(nf0-len(out['faces'])),'retired_point_rows':int(n0-len(out['points_world_m'])),'max_per_original_point_displacement_m':float(max(np.linalg.norm(np.asarray(data['points_world_m'][orig],np.float32).astype(np.float64)-np.asarray(data['points_world_m'][point_rep[int(final_point_map[orig])]],np.float32).astype(np.float64)) for orig in range(n0))),'protected_targets_preserved':len(pinproof),'protected_target_rows':pinproof,'operation_log':ops_all,'skip_reason_counts_by_iteration':skipped_counts,'source_ancestry_covers_all_input_face_rows':True,'source_ancestry_covers_all_input_point_rows':True,'exact_self_cross_bone_gates':'NOT_RUN','tetgen':'NOT_RUN'}
    return out,result


def main():
    if OUT.exists():raise FileExistsError(OUT)
    plan=json.loads(PLAN.read_text())
    OUT.mkdir(parents=True)
    (OUT/'source').mkdir()
    source_path=Path(__file__);(OUT/'source'/source_path.name).write_bytes(source_path.read_bytes())
    results={}
    all_hashes={}
    for role,(path,expected) in INPUTS.items():
        out,result=load_asset(role,path,expected,plan)
        target=OUT/f'candidate-{role}-microcollapsed.npz'
        np.savez_compressed(target,**out)
        result['output_path']=str(target);result['output_sha256']=sha(target)
        results[role]=result;all_hashes[str(path)]=expected
        print(json.dumps({'role':role,'status':'SOURCE_ONLY_STRUCTURAL_PASS','output':str(target),'sha256':result['output_sha256'],'iterations':[{'iteration':x['iteration'],'selected':x['selected_collapses'],'eligible':x['eligible_link_safe_edges'],'dropped':x.get('dropped_current_faces',0),'stalled':x.get('stalled',False)} for x in result['iterations']],'operations':result['operations'],'points':[result['initial_topology']['point_rows'],result['output_point_rows']],'faces':[result['initial_topology']['face_rows'],result['output_face_rows']],'max_displacement_m':result['max_per_original_point_displacement_m'],'final_topology':result['final_topology']},indent=2),flush=True)
    plan_sha=sha(PLAN)
    receipt={'schema':'numi.fine-physical-volume.microedge-condition-batch.v1','physical_admission':False,'runtime_created':0,'accepted_physical_steps':0,'input_manifest_path':str(PLAN),'input_manifest_sha256':plan_sha,'inputs':all_hashes,'limits':{'iterations':MAX_ITERATIONS,'per_original_point_displacement_m':MAX_DISPLACEMENT_M,'only_link_safe_interior_edges':True,'preserved_all_149_protected_targets':True,'outside_surviving_vertex_coordinates_unchanged':True},'legacy_collapse_method_path':str(LEGACY_COLLAPSE_OWNER),'legacy_collapse_method_sha256':sha(LEGACY_COLLAPSE_OWNER),'roles':results,'exact_self_cross_bone_gates':'NOT_RUN','tetgen':'NOT_RUN','scope':'Bounded topology-only endpoint contractions of submicron edges; every retired point maps to an output point within 1 um, all source ancestry is reachable through pinned original candidate row CSR, no accepted physical/runtime claim.'}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(OUT),'receipt_sha256':sha(OUT/'receipt.json'),'status':'SOURCE_ONLY_STRUCTURAL_PASS','exact_gates':'NOT_RUN','tetgen':'NOT_RUN'},indent=2),flush=True)

if __name__=='__main__':main()
