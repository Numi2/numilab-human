#!/usr/bin/env python3
"""Direct boundary recapping for edge-connected surface deltas; diagnostic only."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

H = Path(__file__).resolve().parent
ASSET=H/"current-volume-boundary-remesh-032"
CORE = H / "fixed_boundary_volume_flip_198_root.py"
TRANSFER = H / "transfer_surface_contractions_to_volume_209_root.py"
def load(name, path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load("cap198",CORE);tr=load("cap209",TRANSFER)
def sha(path):
    with open(path,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def facekey(f):return tuple(sorted(map(int,f)))
def facesign(f):
    a,b,c=map(int,f);inv=(a>b)+(a>c)+(b>c);return 1 if inv%2==0 else -1
def oriented_counter(faces):
    c=Counter()
    for f in faces:c[(facekey(f),facesign(f))]+=1
    return c
def changed_components(extra,missing):
    faces=np.vstack([extra,missing]).astype(np.int64,copy=False);kind=np.r_[np.zeros(len(extra),np.int8),np.ones(len(missing),np.int8)]
    edge_rows=defaultdict(list)
    for i,(a,b,c) in enumerate(faces):
        for u,v in ((a,b),(b,c),(c,a)):edge_rows[tuple(sorted((int(u),int(v))))].append(i)
    adj=[set() for _ in range(len(faces))]
    for rows in edge_rows.values():
        if len(rows)>1:
            for r in rows[1:]:adj[rows[0]].add(r);adj[r].add(rows[0])
    seen=set();out=[]
    for seed in range(len(faces)):
        if seed in seen:continue
        todo=[seed];seen.add(seed);ids=[]
        while todo:
            r=todo.pop();ids.append(r)
            for q in adj[r]:
                if q not in seen:seen.add(q);todo.append(q)
        ids=np.asarray(sorted(ids),np.int64)
        out.append({"extra":faces[ids[kind[ids]==0]],"missing":faces[ids[kind[ids]==1]],"changed_face_rows":ids})
    return out
def node_incidence(t,npoints):
    flat=np.asarray(t,np.int64).ravel();order=np.argsort(flat,kind="stable");offsets=np.r_[0,np.cumsum(np.bincount(flat,minlength=npoints))]
    return order,offsets
def face_owner(face,t,inc_order,inc_offsets):
    verts=list(map(int,face));v=min(verts,key=lambda x:int(inc_offsets[x+1]-inc_offsets[x]))
    cand=np.unique(inc_order[inc_offsets[v]:inc_offsets[v+1]]//4);rows=t[cand];mask=np.ones(len(cand),bool)
    for x in verts:mask &= np.any(rows==x,axis=1)
    hit=cand[mask]
    if len(hit)!=1:return None,None,int(len(hit))
    row=int(hit[0]);opp=[int(x) for x in t[row] if int(x) not in verts]
    return row,(opp[0] if len(opp)==1 else None),int(len(hit))
def _box_overlap(alo,ahi,blo,bhi):return bool(np.all(alo<=bhi) and np.all(blo<=ahi))
def _sat_positive_overlap(a,b):
    """Exact Float32-lattice SAT. True means positive-volume tetra overlap."""
    ea=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    va=[tuple(a[j][k]-a[i][k] for k in range(3)) for i,j in ea]
    vb=[tuple(b[j][k]-b[i][k] for k in range(3)) for i,j in ea];axes=[]
    for tet in (a,b):
        for i,j,k in ((1,2,3),(0,3,2),(0,1,3),(0,2,1)):
            u=tuple(tet[j][q]-tet[i][q] for q in range(3));v=tuple(tet[k][q]-tet[i][q] for q in range(3))
            axes.append((u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]))
    for u in va:
        for v in vb:axes.append((u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]))
    for axis in axes:
        if axis==(0,0,0):continue
        pa=[sum(int(x[q])*axis[q] for q in range(3)) for x in a];pb=[sum(int(x[q])*axis[q] for q in range(3)) for x in b]
        if max(pa)<=min(pb) or max(pb)<=min(pa):return False
    return True
class TetBVH:
    def __init__(self,p,t,leaf=24):
        self.t=t;xyz=p[t];self.lo=xyz.min(axis=1);self.hi=xyz.max(axis=1);self.cent=(self.lo+self.hi)*.5
        self.order=np.arange(len(t),dtype=np.int64);self.nodes=[];self.leaf=leaf;self.root=self._build(0,len(t))
    def _build(self,start,end):
        ids=self.order[start:end];lo=self.lo[ids].min(axis=0);hi=self.hi[ids].max(axis=0);idx=len(self.nodes);self.nodes.append(None)
        if end-start<=self.leaf:self.nodes[idx]=(lo,hi,-1,-1,start,end);return idx
        axis=int(np.argmax(np.ptp(self.cent[ids],axis=0)));perm=np.argpartition(self.cent[ids,axis],(end-start)//2)
        self.order[start:end]=ids[perm];mid=(start+end)//2;left=self._build(start,mid);right=self._build(mid,end)
        self.nodes[idx]=(lo,hi,left,right,0,0);return idx
    def query(self,lo,hi):
        todo=[self.root]
        while todo:
            ni=todo.pop();nl,nh,left,right,start,end=self.nodes[ni]
            if not _box_overlap(lo,hi,nl,nh):continue
            if left<0:
                rows=self.order[start:end];keep=np.all((self.lo[rows]<=hi)&(self.hi[rows]>=lo),axis=1)
                yield from map(int,rows[keep])
            else:todo.extend((left,right))
def local_chain_after_recap(old,new,extra,missing):
    expected=Counter(m.chain(old))
    for f in extra:
        k=facekey(f);sgn=facesign(f)
        if expected.get(k,0)!=sgn:return False,"retired_face_not_single_oriented_local_boundary"
        del expected[k]
    for f in missing:
        expected[facekey(f)]+=facesign(f)
        if expected[facekey(f)]==0:del expected[facekey(f)]
    got=Counter(m.chain(new));return expected==got,("PASS" if expected==got else "local_oriented_chain_mismatch")
def exact_volume_int(p,t):
    if not len(t):return 0
    used=np.unique(np.asarray(t,np.int64).ravel());lookup=np.full(len(p),-1,np.int64);lookup[used]=np.arange(len(used));ip=m.integer_points(p[used])
    return sum(m.exact_det(ip,[int(lookup[int(v)]) for v in tet]) for tet in t)
def volume_units_to_m3(x):return float(x)/(6.0*(2.0**447))
def check_intersections(p,t,new,deleted_rows,previous_new,tree,max_pairs=2000000):
    deleted=set(map(int,deleted_rows));cache={};checked=0
    def ipoint(row):
        row=int(row)
        if row not in cache:
            vals=[]
            for x in p[row].astype(np.float32):
                n,d=float(x).as_integer_ratio();vals.append(n<<(149-(d.bit_length()-1)))
            cache[row]=tuple(vals)
        return cache[row]
    all_new=np.asarray(new,np.int64).reshape(-1,4);prev=np.vstack(previous_new) if len(previous_new) else np.empty((0,4),np.int64)
    prior_all=np.vstack([prev,all_new]);prior_tree=TetBVH(p,prior_all) if len(prior_all) else None
    for ni,tet in enumerate(all_new):
        xyz=p[tet];lo=xyz.min(axis=0);hi=xyz.max(axis=0);a=[ipoint(x) for x in tet]
        for local_r in tree.query(lo,hi):
            orig=int(local_r)
            if orig in deleted:continue
            checked+=1
            if checked>max_pairs:return {"pass":False,"reason":"aabb_pair_budget","checked_pairs":checked}
            if _sat_positive_overlap(a,[ipoint(x) for x in t[orig]]):
                return {"pass":False,"reason":"new_vs_retained_positive_volume_overlap","new_tet":ni,"retained_tet":orig,"checked_pairs":checked}
        if prior_tree is not None:
            limit=len(prev)+ni
            for pj in prior_tree.query(lo,hi):
                if pj>=limit:continue
                ptet=prior_all[pj];checked+=1
                if checked>max_pairs:return {"pass":False,"reason":"aabb_pair_budget","checked_pairs":checked}
                if _sat_positive_overlap(a,[ipoint(x) for x in ptet]):
                    return {"pass":False,"reason":"new_tet_positive_volume_overlap","new_tet":ni,"other_new_tet":pj,"checked_pairs":checked}
    return {"pass":True,"reason":"PASS_EXACT_F32_SAT_NO_POSITIVE_VOLUME_OVERLAP","checked_pairs":checked,"exact_point_cache_rows":len(cache)}
def boundary_union_options(p,old,extra,missing,mat):
    """Retetrahedralize exactly the changed owner union; no point movement."""
    expected=Counter(m.chain(old))
    for f in extra:
        k=facekey(f);sgn=facesign(f)
        if expected.get(k,0)!=sgn:return [],'retired_face_not_single_oriented_local_boundary'
        del expected[k]
    for f in missing:
        expected[facekey(f)]+=facesign(f)
        if expected[facekey(f)]==0:del expected[facekey(f)]
    if any(abs(v)!=1 for v in expected.values()):return [],'nonmanifold_expected_shell_multiplicity'
    if not expected:
        return [(np.empty((0,4),np.int64),None,'empty_target_shell_remove_wedge')],None
    faces=np.asarray([k if sg>0 else (k[0],k[2],k[1]) for k,sg in sorted(expected.items())],np.int64)
    directed=Counter()
    for f in faces:
        for i,j in ((0,1),(1,2),(2,0)):
            a,b=int(f[i]),int(f[j]);directed[(min(a,b),max(a,b))]+=1 if a<b else -1
    if any(directed.values()):return [],'expected_shell_not_closed'
    candidates=[]
    for apex in sorted(map(int,np.unique(old))):
        ff=faces[~np.any(faces==apex,axis=1)]
        new=np.c_[np.full(len(ff),apex,np.int64),ff]
        if not len(new):continue
        det,q=m.measurements(p,new)
        if np.any(det<=0):continue
        if dict(m.chain(new))!=dict(expected):continue
        if len({tuple(sorted(map(int,x))) for x in new})!=len(new):continue
        if exact_volume_int(p,new)<=0:continue
        candidates.append((new,apex,'connected_owner_union_boundary_vertex_fan'))
    candidates.sort(key=lambda x:(int(np.sum(~m.passes(*m.measurements(p,x[0])))),len(x[0]),x[1]))
    return candidates,None if candidates else 'no_positive_existing_vertex_fan'

def face_connected_expansion_options(p,t,mat,base,extra,missing,order,offsets,max_cells=256,max_rings=3,deadline=None):
    """Grow through actual shared faces; do not take whole vertex stars."""
    domain=set(map(int,base));material=int(mat[next(iter(domain))]);visited=[]
    for ring in range(max_rings+1):
        rows=np.asarray(sorted(domain),np.int64)
        options,why=boundary_union_options(p,t[rows],extra,missing,material)
        visited.append({'ring':ring,'cells':len(rows),'reason':why,'positive_options':len(options)})
        if options:return rows,options,visited
        if ring>=max_rings or (deadline is not None and time.monotonic()>=deadline):break
        shell=m.chain(t[rows]);newset=set(domain)
        for f in shell:
            v=min(f,key=lambda x:int(offsets[x+1]-offsets[x]))
            cand=np.unique(order[offsets[v]:offsets[v+1]]//4);cells=t[cand]
            hit=np.ones(len(cand),bool)
            for x in f:hit &= np.any(cells==x,axis=1)
            hits=cand[hit]
            if len(hits)>2:return rows,[],visited+[{'reason':'nonmanifold_full_mesh_face','face':list(f)}]
            newset.update(int(c) for c in hits if int(mat[c])==material)
        if newset==domain:break
        if len(newset)>max_cells:
            visited.append({'reason':'connected_cavity_cell_cap','cells':len(newset)});break
        domain=newset
    return np.asarray(sorted(domain),np.int64),[],visited

def direct_recap(input_path,out,max_operations=5000,seconds=300,max_q_debt=5000,max_pair_checks=2000000):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    with np.load(input_path) as z:v={k:z[k] for k in z.files}
    p=np.asarray(v["points_world_m"],np.float32);t=np.asarray(v["tetrahedra"],np.int64);mat=np.asarray(v["material_indices"],np.int32)
    desired=np.asarray(v["desired_boundary_faces"],np.int64);bb=np.asarray(v["combinatorial_boundary_faces"],np.int64)
    extra=np.asarray(v["current_extra_boundary_faces"],np.int64);missing=np.asarray(v["desired_missing_faces"],np.int64)
    actual=tr.boundary(t)
    if oriented_counter(actual)!=oriented_counter(bb):raise ValueError("pinned combinatorial boundary differs from recomputed input boundary")
    ak={facekey(f):facesign(f) for f in actual};dk={facekey(f):facesign(f) for f in desired}
    if len(ak)!=len(actual) or len(dk)!=len(desired):raise ValueError("duplicate oriented boundary face")
    if {k for k in dk if k not in ak}!={facekey(f) for f in missing}:raise ValueError("missing rows do not match target")
    if {k for k in ak if k not in dk}!={facekey(f) for f in extra}:raise ValueError("extra rows do not match target")
    if any(ak[k]!=dk[k] for k in set(ak)&set(dk)):raise ValueError("common boundary orientation mismatch")
    comps=changed_components(extra,missing);inc_order,inc_offsets=node_incidence(t,len(p));candidates=[];candidate_rejects=Counter();candidate_witnesses=[]
    def reject_component(ci,c,reason,**detail):
        candidate_rejects[reason]+=1
        candidate_witnesses.append({"changed_component_id":int(ci),"reason":reason,"extra_face_count":int(len(c["extra"])),
            "missing_face_count":int(len(c["missing"])),"extra_faces":c["extra"].tolist(),"missing_faces":c["missing"].tolist(),**detail})
    for ci,c in enumerate(comps):
        ex,mi=c["extra"],c["missing"]
        if not len(ex) or not len(mi):reject_component(ci,c,"component_lacks_extra_or_missing");continue
        owners=[];apices=[]
        for f in ex:
            r,a,n=face_owner(f,t,inc_order,inc_offsets)
            if r is None or a is None:owners=[];candidate_rejects["extra_facet_owner_not_unique"]+=1;candidate_witnesses.append({"changed_component_id":int(ci),"reason":"extra_facet_owner_not_unique","extra_face":f.tolist(),"owner_count":int(n),"extra_face_count":int(len(ex)),"missing_face_count":int(len(mi))});break
            owners.append(r);apices.append(a)
        if not owners:continue
        owners_a=np.asarray(sorted(set(owners)),np.int64);old=t[owners_a]
        if len(np.unique(mat[owners_a]))!=1:
            reject_component(ci,c,"mixed_owner_material",owner_cell_rows=owners_a.tolist());continue
        owners_a,options,growth=face_connected_expansion_options(p,t,mat,owners_a,ex,mi,inc_order,inc_offsets,deadline=started+seconds)
        old=t[owners_a];why=growth[-1].get("reason") if growth else "no_connected_growth_option"
        if not options:
            reject_component(ci,c,why,owner_cell_rows=owners_a.tolist(),opposite_apex_rows=apices,connected_growth=growth);continue
        # Every proposed option has the exact expected local oriented chain.
        # Keep the best quality existing-vertex fan; exact overlap is checked
        # against retained cells and earlier accepted patches below.
        new,apex,construction=options[0]
        det,qual=m.measurements(p,new) if len(new) else (np.empty(0),np.empty(0))
        used=np.unique(new.ravel());lookup=np.full(len(p),-1,np.int64);lookup[used]=np.arange(len(used));ip=m.integer_points(p[used])
        exact_new=[m.exact_det(ip,[int(lookup[int(v)]) for v in tet]) for tet in new]
        if any(x<=0 for x in exact_new):reject_component(ci,c,"nonpositive_exact_f32_tet",common_apex_row=apex,minimum_exact_determinant=str(min(exact_new)));continue
        ok,why=local_chain_after_recap(old,new,ex,mi)
        if not ok:reject_component(ci,c,why,owner_cell_rows=owners,common_apex_row=apex);continue
        sorted_new={tuple(sorted(map(int,x))) for x in new}
        if len(sorted_new)!=len(new):reject_component(ci,c,"duplicate_new_tets",common_apex_row=apex);continue
        owner_set=set(owners)
        dup=False
        for tet in new:
            v0=min(map(int,tet));rows=np.unique(inc_order[inc_offsets[v0]:inc_offsets[v0+1]]//4);cand=t[rows]
            key=tuple(sorted(map(int,tet)))
            if any(int(rr) not in owner_set and tuple(sorted(map(int,cell)))==key for rr,cell in zip(rows,cand)):dup=True;break
        if dup:reject_component(ci,c,"duplicate_against_input_tet",common_apex_row=apex);continue
        od,oq=m.measurements(p,old);oldbad=int(np.sum(~m.passes(od,oq)));newbad=int(np.sum(~m.passes(det,qual)))
        candidates.append({"id":ci,"extra":ex,"missing":mi,"owners":owners_a,"apex":apex,"construction":construction,"connected_growth":growth,"old":old,"new":new,"material":int(mat[owners_a[0]],
            ),"old_bad":oldbad,"new_bad":newbad,"debt":newbad-oldbad})
    candidates.sort(key=lambda c:(c["debt"],len(c["owners"]),c["id"]))
    accepted=[];rejects=Counter();consumed=set();totaldebt=0;tree=TetBVH(p,t) if candidates else None
    for c in candidates:
        if time.monotonic()-started>seconds or len(accepted)>=max_operations:rejects["global_budget"]+=1;continue
        own=set(map(int,c["owners"]))
        if own&consumed:rejects["consumed_owner_conflict"]+=1;continue
        if totaldebt+c["debt"]>max_q_debt:rejects["construction_q_debt_budget"]+=1;continue
        prior=[x["new"] for x in accepted];deleted=np.asarray(sorted(consumed|own),np.int64)
        inter=check_intersections(p,t,c["new"],deleted,prior,tree,max_pair_checks)
        if not inter["pass"]:rejects[inter["reason"]]+=1;continue
        c["intersection"]=inter;accepted.append(c);consumed.update(own);totaldebt+=c["debt"]
    keep=np.ones(len(t),bool)
    if consumed:keep[np.asarray(sorted(consumed),np.int64)]=False
    retained=np.flatnonzero(keep);newt=np.vstack([x["new"] for x in accepted]) if accepted else np.empty((0,4),np.int64)
    newmat=np.concatenate([np.full(len(x["new"]),x["material"],np.int32) for x in accepted]) if accepted else np.empty(0,np.int32)
    candidate=np.vstack([t[keep],newt]);candidate_mat=np.r_[mat[keep],newmat]
    out_boundary=tr.boundary(candidate);out_counter=oriented_counter(out_boundary);target_counter=oriented_counter(desired)
    resmiss=[k for k,vv in target_counter.items() if out_counter.get(k,0)!=vv];resextra=[k for k,vv in out_counter.items() if target_counter.get(k,0)!=vv]
    target_match=not resmiss and not resextra
    allnewdet,allnewq=m.measurements(p,newt) if len(newt) else (np.empty(0),np.empty(0))
    if len(newt) and np.any(allnewdet<=0):raise ValueError("accepted direct cap lost positivity")
    exactdelta=sum(exact_volume_int(p,x["new"])-exact_volume_int(p,x["old"]) for x in accepted)
    ops=[];row=0
    for x in accepted:
        n=len(x["new"]);ops.append({"changed_component_id":x["id"],"old_extra_boundary_faces":x["extra"].tolist(),"new_target_boundary_faces":x["missing"].tolist(),
            "owner_input_cell_rows":x["owners"].tolist(),"common_opposite_apex_input_point_row":x["apex"],"construction":x["construction"],"connected_growth":x["connected_growth"],"material_index":x["material"],
            "old_cells":len(x["old"]),"new_cells":n,"old_native_quality_failures":x["old_bad"],"new_native_quality_failures":x["new_bad"],
            "construction_quality_debt":x["debt"],"physical_volume_delta_m3":volume_units_to_m3(exact_volume_int(p,x["new"])-exact_volume_int(p,x["old"])),
            "replacement_output_tet_row_start":len(retained)+row,"replacement_output_tet_row_end":len(retained)+row+n,
            "local_oriented_chain":"PASS","intersection_check":x["intersection"]});row+=n
    report={"schema":"h706.direct-boundary-recap-diagnostic.v1","status":"PASS_DIRECT_RECAP_TARGET_BOUNDARY" if target_match and accepted else ("PASS_PARTIAL_DIRECT_RECAP" if accepted else "NO_DIRECT_RECAP_COMPONENTS"),
        "physical_admission":False,"native_steps":0,"input_path":str(input_path),"input_sha256":sha(input_path),"producer_path":str(Path(__file__)),"producer_sha256":sha(Path(__file__)),
        "policy":{"edge_connected_changed_facets":True,"exact_connected_owner_union_or_empty_shell_required":True,"maximum_face_growth_rings":3,"maximum_cavity_cells":256,"positive_exact_f32_required":True,"local_oriented_chain_required":True,
                  "max_construction_quality_debt":max_q_debt,"native_quality_threshold_remains_final_zero":True},
        "input_points":len(p),"input_cells":len(t),"input_boundary_faces":len(bb),"desired_boundary_faces":len(desired),"missing_target_faces":len(missing),"extra_current_faces":len(extra),
        "edge_connected_changed_components":len(comps),"eligible_components":len(candidates),"accepted_components":len(accepted),"accepted_new_tets":len(newt),
        "consumed_input_cells":len(consumed),"output_cells":len(candidate),"output_boundary_faces":len(out_boundary),"output_boundary_exactly_target":target_match,
        "residual_missing_face_count":len(resmiss),"residual_extra_face_count":len(resextra),"residual_face_key_examples":{"missing":resmiss[:50],"extra":resextra[:50]},
        "candidate_filter_rejections":dict(candidate_rejects),"candidate_reject_witnesses":candidate_witnesses,"operation_rejections":dict(rejects),"aggregate_construction_quality_debt":totaldebt,
        "old_native_quality_failures_in_consumed_cells":sum(x["old_bad"] for x in accepted),"new_native_quality_failures_in_replacements":int(np.sum(~m.passes(allnewdet,allnewq))) if len(newt) else 0,
        "exact_determinant_sum_delta_lattice_2m447":str(exactdelta),"physical_volume_delta_m3":volume_units_to_m3(exactdelta),
        "target_shell_volume_identity":"candidate oriented boundary exactly equals target shell" if target_match else "each accepted local shell identity is exact; remaining boundary delta is listed",
        "new_cap_internal_intersection":"exact SAT on F32 integer lattice against every AABB-overlapping retained tet and prior cap tet",
        "elapsed_seconds":time.monotonic()-started,"operations":ops,
        "limitations":["no TetGen","bounded face-connected owner-union fans and exact empty-shell wedge retirement","source mesh qualification inherited","construction diagnostic only","native zero-quality-defect gate not waived","no mass/native admission"]}
    outmap=np.r_[np.where(keep)[0],np.full(len(newt),-1,np.int64)]
    compids=np.r_[np.full(len(retained),-1,np.int32),np.concatenate([np.full(len(x["new"]),x["id"],np.int32) for x in accepted]) if accepted else np.empty(0,np.int32)]
    rr=[];cur=len(retained)
    for x in accepted:rr.append([cur,cur+len(x["new"])]);cur+=len(x["new"])
    np.savez_compressed(out/"direct-recap-candidate.npz",points_world_m=p,tetrahedra=candidate,material_indices=candidate_mat,
        output_tetrahedron_to_input_cell_row=outmap,output_new_component_id=compids,
        replacement_parent_input_cell_rows=np.concatenate([x["owners"] for x in accepted]) if accepted else np.empty(0,np.int64),
        replacement_output_tet_rows=np.asarray(rr,np.int64).reshape(-1,2),current_target_boundary_faces=desired,combinatorial_output_boundary_faces=out_boundary)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"report.sha256").write_text(sha(out/"report.json")+"  report.json\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("operations","residual_face_key_examples")},sort_keys=True))
    return report
def fixture():
    p=np.array([[.5,.5,1],[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32)
    old=m.orient(p,np.array([[0,1,2,3],[0,1,3,4]],np.int64));oldb=tr.boundary(old)
    newref=m.orient(p,np.array([[0,1,2,4],[0,2,3,4]],np.int64));newb=tr.boundary(newref)
    oldkeys={facekey(x) for x in oldb};newkeys={facekey(x) for x in newb}
    extra=np.asarray([x for x in oldb if facekey(x) not in newkeys],np.int64);missing=np.asarray([x for x in newb if facekey(x) not in oldkeys],np.int64)
    assert len(extra)==2 and len(missing)==2 and len(changed_components(extra,missing))==1
    order,offs=node_incidence(old,len(p));own=[];apex=[]
    for f in extra:
        r,a,n=face_owner(f,old,order,offs);assert n==1;own.append(r);apex.append(a)
    assert len(set(apex))==1;new=np.c_[np.full(len(missing),apex[0]),missing]
    assert np.all(m.measurements(p,new)[0]>0);assert local_chain_after_recap(old[own],new,extra,missing)[0]
    ip=m.integer_points(p);a=[ip[i] for i in (0,1,2,3)];assert _sat_positive_overlap(a,a)
    p2=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],np.float32);ip2=m.integer_points(p2)
    assert not _sat_positive_overlap([ip2[i] for i in (0,1,2,3)],[ip2[i] for i in (0,2,1,4)])
    # A protruding tetra has three exterior old faces; its fourth face
    # becomes the target exterior of the retained neighbor. No replacement
    # tetra is appropriate: the exact target local shell cancels to zero.
    p3=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],np.float32)
    pair=m.orient(p3,[[0,1,2,3],[0,2,1,4]])
    ab=tr.boundary(pair);desired=tr.boundary(pair[1:])
    ak={facekey(f):f for f in ab};dk={facekey(f):f for f in desired}
    ex=np.asarray([ak[k] for k in ak.keys()-dk.keys()],np.int64)
    mi=np.asarray([dk[k] for k in dk.keys()-ak.keys()],np.int64)
    assert len(ex)==3 and len(mi)==1
    options,why=boundary_union_options(p3,pair[:1],ex,mi,15)
    assert len(options)==1 and len(options[0][0])==0,why
    assert local_chain_after_recap(pair[:1],options[0][0],ex,mi)[0]
    options,why=boundary_union_options(p,old,extra,missing,15)
    assert options and local_chain_after_recap(old,options[0][0],extra,missing)[0],why
    return {"status":"PASS_DIRECT_RECAP_FIXTURES","controls":["shared_apex_square_flip","connected_owner_union_vertex_fan","three_exterior_faces_one_owner_empty_shell_wedge_retirement","exact_oriented_local_chain","positive_orientation","exact_SAT_overlap","shared_face_touch_allowed"],"physical_admission":False,"native_steps":0}
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--out",type=Path)
    ap.add_argument("--max-operations",type=int,default=5000);ap.add_argument("--seconds",type=float,default=300);ap.add_argument("--max-construction-quality-debt",type=int,default=5000);ap.add_argument("--max-pair-checks",type=int,default=2000000)
    a=ap.parse_args()
    if a.fixture:print(json.dumps(fixture()))
    else:
        if not a.input or not a.out:ap.error("--input and --out are required")
        direct_recap(a.input,a.out,a.max_operations,a.seconds,a.max_construction_quality_debt,a.max_pair_checks)
