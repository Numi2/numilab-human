"""Construct a boundary cut tree along certified edges of an existing patch.

Retains physical boundary cycles and source vertices. Interior graph-path
vertices permit curved routes when no straight boundary-to-boundary bridge is
admissible. Exact chord and triangle gates remain mandatory.
"""
from collections import Counter,defaultdict
import heapq,itertools,time,numpy as np

def validate_patch(tris,loops,allowed,external_edges):
    tris=[tuple(map(int,t))for t in tris];boundary_vertices=set(v for l in loops for v in l)
    if any(len(set(t))!=3 or not set(t)<=allowed for t in tris):raise ValueError("invalid patch face")
    if len({tuple(sorted(t))for t in tris})!=len(tris):raise ValueError("duplicate physical face")
    dirs=Counter((u,v)for t in tris for u,v in zip(t,t[1:]+t[:1]));owners=defaultdict(list)
    for i,t in enumerate(tris):
        for u,v in zip(t,t[1:]+t[:1]):owners[tuple(sorted((u,v)))].append(i)
    expected=Counter((u,v)for l in loops for u,v in zip(l,l[1:]+l[:1]))
    boundary=Counter();adj=[set()for _ in tris]
    for (u,v),rows in owners.items():
        f,r=dirs[(u,v)],dirs[(v,u)]
        if len(rows)==1 and f+r==1:boundary[(u,v)if f else(v,u)]=1
        elif len(rows)==2 and f==r==1:
            if (u,v)in external_edges:raise ValueError("interior retained diagonal reused")
            a,b=rows;adj[a].add(b);adj[b].add(a)
        else:raise ValueError("edge multiplicity/orientation failed")
    if boundary!=expected:raise ValueError("directed boundary changed")
    reached={0};todo=[0]
    while todo:
        for j in adj[todo.pop()]:
            if j not in reached:reached.add(j);todo.append(j)
    used=set(v for t in tris for v in t);chi=len(used)-len(owners)+len(tris)
    if len(reached)!=len(tris)or not boundary_vertices<=used or chi!=2-len(loops):raise ValueError("connectivity/Euler failed")
    for v in used:
        link=defaultdict(set)
        for t in tris:
            if v in t:
                a,b=[x for x in t if x!=v];link[a].add(b);link[b].add(a)
        degrees=sorted(map(len,link.values()))
        if any(d not in (1,2)for d in degrees):raise ValueError("branched vertex link")
        ones=degrees.count(1)
        if ones!=(2 if v in boundary_vertices else 0):raise ValueError("vertex link boundary/cycle mismatch")
        seen={next(iter(link))};stack=list(seen)
        while stack:
            for x in link[stack.pop()]:
                if x not in seen:seen.add(x);stack.append(x)
        if len(seen)!=len(link):raise ValueError("disconnected vertex link")
    return {"vertices":len(used),"edges":len(owners),"faces":len(tris),"euler_characteristic":chi,
        "boundary_components":len(loops),"interior_vertices":len(used-boundary_vertices),"vertex_links":"PASS"}

def triangulate_with_interior_paths(loops,oldfaces,points,external_edges,index,pred,lattice,
        first_candidate_id,*,chord_pruning,dp_disk,identity_proxy,max_seconds=36):
    started=time.monotonic();loops=[list(map(int,l))for l in loops]
    allowed=set(map(int,np.asarray(oldfaces).ravel()))
    trueboundary={tuple(sorted((u,v)))for l in loops for u,v in zip(l,l[1:]+l[:1])}
    oldedges={tuple(sorted((int(u),int(v))))for t in oldfaces for u,v in zip(t,np.roll(t,-1))}
    adj=defaultdict(list);cache={};details=[];blocked=[];nodes=sorted(allowed)
    for u,v in sorted(oldedges):
        if time.monotonic()-started>max_seconds:raise TimeoutError("path edge certification budget exhausted")
        if (u,v)in external_edges-trueboundary:continue
        third=next(x for x in nodes if x not in (u,v))
        bad,info=chord_pruning.prune_chords([u,v,third],lattice,index,pred,max_seconds=4.,cache=cache,max_witnesses=4)
        if (u,v)in bad:blocked.append([u,v]);continue
        weight=float(np.linalg.norm(np.asarray(points[u],float)-points[v]))
        adj[u].append((v,weight));adj[v].append((u,weight))
    first=max(range(len(loops)),key=lambda i:(len(loops[i]),-i))
    cut=list(loops[first]);remaining=set(range(len(loops)))-{first};paths=[]
    while remaining:
        targets={v:i for i in remaining for v in loops[i]}
        dist={v:0. for v in cut};parent={v:None for v in cut};queue=[(0.,v)for v in sorted(set(cut))];heapq.heapify(queue)
        found=None
        while queue:
            d,u=heapq.heappop(queue)
            if d!=dist[u]:continue
            if u in targets:found=u;break
            for v,w in adj[u]:
                nd=d+w
                if nd<dist.get(v,float("inf")):
                    dist[v]=nd;parent[v]=u;heapq.heappush(queue,(nd,v))
        if found is None:raise ValueError("certified original-edge graph cannot connect physical boundary components; blocked="+repr(blocked[:16]))
        path=[found]
        while parent[path[-1]]is not None:path.append(parent[path[-1]])
        path.reverse();anchor=path[0];hole=targets[found];h=loops[hole];j=h.index(found);h=h[j:]+h[:j];i=cut.index(anchor)
        cut=cut[:i+1]+path[1:-1]+h+[found]+list(reversed(path[1:-1]))+[anchor]+cut[i+1:]
        paths.append(path);remaining.remove(hole)
    vp=np.asarray(points,np.float32)[cut];proxy=identity_proxy(pred,cut)
    keys=tuple(pred.float32_point_lattice_key(v)for v in vp)
    bad,pruneinfo=chord_pruning.prune_chords(list(range(len(cut))),keys,index,proxy,max_seconds=max(2.,max_seconds-(time.monotonic()-started)),cache=cache,max_witnesses=16)
    virtualboundary={tuple(sorted((i,(i+1)%len(cut))))for i in range(len(cut))}
    if bad&virtualboundary:raise ValueError("constructed graph cut contains an invalid exact chord")
    vex={(i,j)for i in range(len(cut))for j in range(i+1,len(cut))
      if tuple(sorted((cut[i],cut[j])))in external_edges-trueboundary}|bad
    remaining_time=max_seconds-(time.monotonic()-started)
    if remaining_time<=0:raise TimeoutError("path preprocessing consumed budget")
    tris,recs,dpinfo=dp_disk(list(range(len(cut))),vp.astype(float),vp,vex,index,proxy,keys,[],first_candidate_id,max_seconds=remaining_time,max_branches=2048)
    actual=[tuple(cut[v]for v in t)for t in tris]
    topology=validate_patch(actual,loops,allowed,external_edges)
    check=pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)
    if check["count"]:raise ValueError("physical path result self-intersects")
    return actual,recs,{"method":"certified_original_edge_cut_tree_then_exact_DP","paths":paths,"old_edges":len(oldedges),
        "blocked_old_edges":blocked,"chord_pruning":pruneinfo,"dp":dpinfo,"topology":topology,
        "coordinate_changes":0,"elapsed_s":time.monotonic()-started}
