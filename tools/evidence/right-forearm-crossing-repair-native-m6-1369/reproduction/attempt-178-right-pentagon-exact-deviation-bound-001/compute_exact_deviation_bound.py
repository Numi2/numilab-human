from __future__ import annotations
import hashlib,json,pickle,pathlib
from fractions import Fraction as F
BASE=pathlib.Path('/Users/n/numi-human-fds132-m6-1350')
CAND=BASE/'attempt-177-right-pentagon-option2-checkpoint-001/result-002/candidate.pkl'
CAND_SHA='2be23c705be2a6589a709481f1dcf57a09b87481e6524909ea95c6048818e5e9'
PARENT=BASE/'attempt-166-right-capB-ancestry-correction-001/result-001/candidate.pkl'
PARENT_SHA='25d759377d4861660f12c6458cd5bde4d48e152462048f15aea6410b70b2c17d'
OUT=BASE/'attempt-178-right-pentagon-exact-deviation-bound-001/result-003'
OUT.mkdir(parents=True,exist_ok=False)
raw=CAND.read_bytes(); assert hashlib.sha256(raw).hexdigest()==CAND_SHA
praw=PARENT.read_bytes(); assert hashlib.sha256(praw).hexdigest()==PARENT_SHA
cand=pickle.loads(raw); parent=pickle.loads(praw)
cycle=[4632,4661,4668,4671,4675]
old_ids=[3304,3323,3599]
new_ids=old_ids
old_tris=[tuple(map(int,parent['surfaces'][133]['faces_global'][i])) for i in old_ids]
new_tris=[tuple(map(int,cand['surfaces'][133]['faces_global'][i])) for i in new_ids]
s=parent['surfaces'][133]
g2p={int(g):tuple(F(x) for x in xyz) for g,xyz in zip(s['global_vertex_ids'],s['vertices_exact_m'])}
assert all(v in g2p for t in old_tris+new_tris for v in t)

# Choose the coordinate projection that maximizes the absolute exact polygon normal component.
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def cross3(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
q0,q1,q2=(g2p[v] for v in cycle[:3])
normal=cross3(sub(q1,q0),sub(q2,q0))
axis=max(range(3),key=lambda k:abs(normal[k]))
uv=[i for i in range(3) if i!=axis]
def proj(pt): return (pt[uv[0]],pt[uv[1]])
def hgt(pt): return pt[axis]
def c2(a,b): return a[0]*b[1]-a[1]*b[0]
def sub2(a,b): return (a[0]-b[0],a[1]-b[1])
poly=[proj(g2p[g]) for g in cycle]
turns=[c2(sub2(poly[(i+1)%5],poly[i]),sub2(poly[(i+2)%5],poly[(i+1)%5])) for i in range(5)]
poly_area2=sum(c2(poly[i],poly[(i+1)%5]) for i in range(5))
assert poly_area2 != 0
orientation=1 if poly_area2>0 else -1
# Prove the ordered boundary is simple; nonadjacent segments may not intersect.
def orient(a,b,c): return c2(sub2(b,a),sub2(c,a))
def between(x,a,b): return min(a,b)<=x<=max(a,b)
def onseg(p,a,b): return orient(a,b,p)==0 and between(p[0],a[0],b[0]) and between(p[1],a[1],b[1])
def seg_intersects(a,b,c,d):
    o1,o2,o3,o4=orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)
    if o1==0 and onseg(c,a,b): return True
    if o2==0 and onseg(d,a,b): return True
    if o3==0 and onseg(a,c,d): return True
    if o4==0 and onseg(b,c,d): return True
    return (o1>0)!=(o2>0) and (o3>0)!=(o4>0)
for i in range(5):
  for j in range(i+1,5):
    if j==i or j==(i+1)%5 or i==(j+1)%5: continue
    assert not seg_intersects(poly[i],poly[(i+1)%5],poly[j],poly[(j+1)%5]),(i,j)

def area2(tri):
    a,b,c=(proj(g2p[v]) for v in tri)
    return c2(sub2(b,a),sub2(c,a))
def boundary(tris):
    edges={}
    for t in tris:
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            edges[(a,b)]=edges.get((a,b),0)+1
    # cancel opposite directed edges
    rem={}
    for (a,b),n in edges.items():
        m=edges.get((b,a),0)
        if n>m: rem[(a,b)]=n-m
    return rem
for label,tris in [('old',old_tris),('new',new_tris)]:
    ars=[area2(t) for t in tris]
    assert all(x*orientation>0 for x in ars),(label,ars,orientation)
    assert sum(ars)==poly_area2, (label,sum(ars),poly_area2)
    assert len(boundary(tris))==5 and all(n==1 for n in boundary(tris).values()),(label,boundary(tris))
    assert set(boundary(tris))==set(((cycle[i],cycle[(i+1)%5]) for i in range(5))),(label,boundary(tris))
    uses={}
    for ti,t in enumerate(tris):
      for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
        uses.setdefault(tuple(sorted((a,b))),[]).append((ti,a,b))
    assert sum(len(v)==1 for v in uses.values())==5 and sum(len(v)==2 for v in uses.values())==2,(label,uses)
    assert all(len(v) in (1,2) for v in uses.values())
    assert all(v[0][1:]==(v[1][2],v[1][1]) for v in uses.values() if len(v)==2)
    adj={i:set() for i in range(3)}
    for v in uses.values():
      if len(v)==2: adj[v[0][0]].add(v[1][0]); adj[v[1][0]].add(v[0][0])
    seen={0}; todo=[0]
    while todo:
      u=todo.pop()
      for v in adj[u]-seen: seen.add(v); todo.append(v)
    assert len(seen)==3,(label,adj)

# Exact convex polygon clipping in the chosen 2D chart.
def clip(subject, cliptri):
    out=list(subject)
    cp=[proj(g2p[g]) for g in cliptri]
    a2=c2(sub2(cp[1],cp[0]),sub2(cp[2],cp[0]))
    assert a2*orientation>0
    for i in range(3):
        a,b=cp[i],cp[(i+1)%3]
        inp=out; out=[]
        if not inp: break
        def side(p): return orientation*c2(sub2(b,a),sub2(p,a))
        S=inp[-1]; ds=side(S)
        for E in inp:
            de=side(E)
            if de>=0:
                if ds<0:
                    t=ds/(ds-de)
                    out.append((S[0]+t*(E[0]-S[0]),S[1]+t*(E[1]-S[1])))
                out.append(E)
            elif ds>=0:
                t=ds/(ds-de)
                out.append((S[0]+t*(E[0]-S[0]),S[1]+t*(E[1]-S[1])))
            S,ds=E,de
    # remove successive duplicates and closing duplicate
    clean=[]
    for p in out:
        if not clean or p!=clean[-1]: clean.append(p)
    if len(clean)>1 and clean[0]==clean[-1]: clean.pop()
    return clean

def height_on(tri,q):
    p0,p1,p2=(proj(g2p[g]) for g in tri)
    z0,z1,z2=(hgt(g2p[g]) for g in tri)
    den=c2(sub2(p1,p0),sub2(p2,p0))
    alpha=c2(sub2(q,p0),sub2(p2,p0))/den
    beta=c2(sub2(p1,p0),sub2(q,p0))/den
    return z0+alpha*(z1-z0)+beta*(z2-z0)

maxdiff=F(0); maxw=None; intersections=0; overlay_points=0
for oi,ot in enumerate(old_tris):
    for ni,nt in enumerate(new_tris):
        polygon=clip([proj(g2p[g]) for g in ot],nt)
        if len(polygon)<3: continue
        intersections+=1
        for q in polygon:
            overlay_points+=1
            d=abs(height_on(ot,q)-height_on(nt,q))
            if d>maxdiff:
                maxdiff=d; maxw={'old_face':old_ids[oi],'new_face':new_ids[ni],'projected_point':[str(v) for v in q]}
assert intersections>=3
def polygon_area2_2d(poly2):
    if len(poly2)<3: return F(0)
    return sum(c2(poly2[i],poly2[(i+1)%len(poly2)]) for i in range(len(poly2)))
# Independently prove no positive-area overlap inside either triangulation.
intra=[]
for label,tris in [('old',old_tris),('new',new_tris)]:
  for i in range(3):
    for j in range(i+1,3):
      q=clip([proj(g2p[g]) for g in tris[i]],tris[j])
      a=abs(polygon_area2_2d(q)) if len(q)>=3 else F(0)
      intra.append({'triangulation':label,'faces':[i,j],'intersection_doubled_area':str(a)})
      assert a==0,(label,i,j,a)
overlay_area2=sum(abs(polygon_area2_2d(clip([proj(g2p[g]) for g in ot],nt))) for ot in old_tris for nt in new_tris)
assert overlay_area2==abs(poly_area2),(overlay_area2,abs(poly_area2))
report={
 'schema':'numi.human.right-pentagon-exact-graph-deviation-bound-178.v1',
 'candidate_path':str(CAND),'candidate_sha256':CAND_SHA,
 'parent_candidate_path':str(PARENT),'parent_candidate_sha256':PARENT_SHA,
 'patch_cycle_global_ids':cycle,'old_patch_faces':old_ids,'old_triangles_global':old_tris,
 'new_triangles_global':new_tris,'coordinate_units':'metres','exact_coordinate_source':'Fraction vertices_exact_m from pinned candidate166; candidate177 XYZ unchanged',
 'projection_omitted_axis':axis,'projection_axes':uv,'polygon_simple':True,'polygon_convexity':'concave with exact turn signs '+','.join('+' if v>0 else '-' if v<0 else '0' for v in turns),
 'old_and_new_are_same_domain_oriented_triangulations':True,
 'polygon_projected_doubled_area_fraction':str(poly_area2),
 'old_projected_doubled_area_fraction':str(sum(area2(t) for t in old_tris)),
 'new_projected_doubled_area_fraction':str(sum(area2(t) for t in new_tris)),
 'exact_overlay_triangle_intersections':intersections,'overlay_polygon_vertices_evaluated':overlay_points,
 'intra_triangulation_pairwise_overlap_checks':intra,'old_new_overlay_doubled_area_fraction':str(overlay_area2),
 'topology_embedding_and_domain_checks':'Both three-face patches are connected oriented disks with the same simple oriented boundary, exactly two internal opposite-directed edges, all projected triangle areas positive, zero pairwise interior overlap, and exact coverage area equal to boundary area.',
 'max_exact_vertical_graph_separation_m_fraction':str(maxdiff),
 'max_exact_vertical_graph_separation_m':float(maxdiff),
 'conservative_symmetric_hausdorff_upper_bound_m':float(maxdiff),
 'witness':maxw,
 'bound_scope':'Exact upper bound for symmetric Hausdorff distance between the two source-space piecewise-linear patches, under point correspondence with identical projected coordinates. Both patches are verified nonoverlapping oriented graphs over the same simple concave projected pentagon, and each overlay-cell height difference is affine; its maximum absolute value occurs at an overlay polygon vertex.',
 'native_deformed_pose_bound':'not implied; this is a rest/source-space tessellation difference bound only',
 'candidate_admission':False}
path=OUT/'report.json'; path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))
