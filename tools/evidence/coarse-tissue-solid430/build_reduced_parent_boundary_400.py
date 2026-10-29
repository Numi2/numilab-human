#!/usr/bin/env python3
"""Build a bounded parent330 mechanical boundary candidate from registered O/I/D."""
from __future__ import annotations
import csv, hashlib, json, math, pathlib, sys, time
from collections import Counter, defaultdict, deque
import numpy as np
from scipy.spatial import cKDTree

P = pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
CAP = P / "native-physical-parent-mass-330/native-source-identity-control-003"
SID = P / "full-current-surface-registration-399/sidecars"
OUT = P / "reduced-physical-boundary-400/attempt-002-parent-ancestry-walk"
OUT.mkdir(parents=True, exist_ok=True)
ROLES = ("O", "I", "D")
CANDIDATES = {
    "O": P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/minimum-thickness-minkowski-328/attempt-006-anchored-local-topology/run-030-global-derived-patch-remesh/attempt-002/connected-witness-repair-001/result-003/candidate-O-open-f32-typed.npz",
    "I": P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz",
    "D": P / "deep-core-connected-patch-filter-375/candidate-f32-typed.npz",
}


def sha(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def read_csv(path: pathlib.Path) -> list[dict[str,str]]:
    with path.open(newline="") as f:return list(csv.DictReader(f))
def write_csv(path: pathlib.Path, header: list[str], rows) -> None:
    with path.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def closest_batch(Pt, A, B, C):
    """Closest point and barycentrics for row-aligned points and candidate triangles."""
    ab=B-A;ac=C-A;ap=Pt[:,None,:]-A
    d1=np.einsum("mki,mki->mk",ab,ap);d2=np.einsum("mki,mki->mk",ac,ap)
    m,k=d1.shape; bary=np.zeros((m,k,3),dtype=np.float64);done=np.zeros((m,k),bool)
    mask=(d1<=0)&(d2<=0);bary[:,:,0][mask]=1;done|=mask
    bp=Pt[:,None,:]-B;d3=np.einsum("mki,mki->mk",ab,bp);d4=np.einsum("mki,mki->mk",ac,bp)
    mask=(~done)&(d3>=0)&(d4<=d3);bary[:,:,1][mask]=1;done|=mask
    vc=d1*d4-d3*d2;den=d1-d3
    mask=(~done)&(vc<=0)&(d1>=0)&(d3<=0)&(np.abs(den)>0)
    v=np.divide(d1,den,out=np.zeros_like(d1),where=np.abs(den)>0)
    bary[:,:,0][mask]=1-v[mask];bary[:,:,1][mask]=v[mask];done|=mask
    cp=Pt[:,None,:]-C;d5=np.einsum("mki,mki->mk",ab,cp);d6=np.einsum("mki,mki->mk",ac,cp)
    mask=(~done)&(d6>=0)&(d5<=d6);bary[:,:,2][mask]=1;done|=mask
    vb=d5*d2-d1*d6;den=d2-d6
    mask=(~done)&(vb<=0)&(d2>=0)&(d6<=0)&(np.abs(den)>0)
    w=np.divide(d2,den,out=np.zeros_like(d2),where=np.abs(den)>0)
    bary[:,:,0][mask]=1-w[mask];bary[:,:,2][mask]=w[mask];done|=mask
    va=d3*d6-d5*d4;den=(d4-d3)+(d5-d6)
    mask=(~done)&(va<=0)&((d4-d3)>=0)&((d5-d6)>=0)&(np.abs(den)>0)
    w=np.divide((d4-d3),den,out=np.zeros_like(d4),where=np.abs(den)>0)
    bary[:,:,1][mask]=1-w[mask];bary[:,:,2][mask]=w[mask];done|=mask
    mask=~done
    den=va+vb+vc
    vv=np.divide(vb,den,out=np.zeros_like(vb),where=np.abs(den)>1e-300)
    ww=np.divide(vc,den,out=np.zeros_like(vc),where=np.abs(den)>1e-300)
    bary[:,:,0][mask]=(1-vv-ww)[mask];bary[:,:,1][mask]=vv[mask];bary[:,:,2][mask]=ww[mask]
    q=bary[:,:,0,None]*A+bary[:,:,1,None]*B+bary[:,:,2,None]*C
    dist=np.linalg.norm(Pt[:,None,:]-q,axis=2)
    return q,bary,dist

def signed_volumes(points,tets):
    x=points[tets[:,0]]
    a=points[tets[:,1]]-x;b=points[tets[:,2]]-x;c=points[tets[:,3]]-x
    return np.einsum("ij,ij->i",a,np.cross(b,c))/6.0

def main():
    t0=time.time();inputs={}
    # Parent physical topology and row domains.
    layer_rows={r:[] for r in ROLES};node_rows=[None]*7233
    for x in read_csv(CAP/"parent-layer-rows.csv"):
        role=x["role"];row=int(x["volume_node_row"]);layer_rows[role].append(x);node_rows[row]=x
    for role in ROLES:layer_rows[role].sort(key=lambda x:int(x["role_row"]))
    if any(x is None for x in node_rows):raise ValueError("parent node row domain is not complete")
    if [len(layer_rows[r]) for r in ROLES] != [2411,2411,2411]:raise ValueError("parent role row counts changed")
    for role in ROLES:
        src=[int(x["source_vertex_row"]) for x in layer_rows[role]]
        if src != [int(x["source_vertex_row"]) for x in layer_rows["O"]]:
            raise ValueError("O/I/D parent source vertex row sequences are not identical")
    current0=np.array([[float(x[f"current_{a}_m"]) for a in "xyz"] for x in node_rows],dtype=np.float64)
    reference0=np.array([[float(x[f"reference_{a}_m"]) for a in "xyz"] for x in node_rows],dtype=np.float64)
    local0=np.array([[float(x[f"donor_local_{a}_m"]) for a in "xyz"] for x in node_rows],dtype=np.float64)
    owner=np.array([int(x["frame_owner_body"]) for x in node_rows],dtype=np.int64)
    source_vertex=np.array([int(x["source_vertex_row"]) for x in node_rows],dtype=np.int64)
    weights=[]
    wr=read_csv(CAP/"compiled-parent-cell-mass.csv.weights.csv")
    if len(wr)!=7233:raise ValueError("parent full86 weight row count differs")
    for i,x in enumerate(wr):
        if int(x["vertex"])!=i:raise ValueError("parent full86 weights are not row-identity ordered")
        weights.append([float(x[f"w{j}"]) for j in range(86)])
    weights=np.asarray(weights,dtype=np.float64)
    tets=[];materials=[];donors=[];regions=[];stable=[];masses=[];parent_volumes=[]
    tr=read_csv(CAP/"compiled-parent-cell-mass.csv.tets.csv");lr=read_csv(CAP/"compiled-parent-cell-mass.csv.labels.csv")
    if len(tr)!=28632 or len(lr)!=len(tr):raise ValueError("parent tet/label row domain changed")
    for i,(x,y) in enumerate(zip(tr,lr)):
        if int(x["parent_tet_row"])!=i or int(y["parent_tet_row"])!=i:raise ValueError("parent tet rows are not identity ordered")
        tets.append([int(x[f"parent_node{k}"]) for k in range(4)])
        materials.append(int(x["material_index"]));donors.append(int(y["geometric_donor_body"]))
        regions.append(int(y["anatomical_region_id"]));stable.append(int(y["stable_source_face_id"]))
        masses.append(float(x["parent_mass_kg"]));parent_volumes.append(float(x["parent_reference_volume_m3"]))
    tets=np.asarray(tets,dtype=np.int64);materials=np.asarray(materials);donors=np.asarray(donors);regions=np.asarray(regions);stable=np.asarray(stable);masses=np.asarray(masses);parent_volumes=np.asarray(parent_volumes)
    role_nodes={r:set(int(x["volume_node_row"]) for x in layer_rows[r]) for r in ROLES}
    # Surface face incidence from the original parent tet topology.
    incident=defaultdict(list)
    for ti,t in enumerate(tets):
        for omit in range(4):incident[tuple(sorted(int(t[j]) for j in range(4) if j!=omit))].append((ti,omit))
    tet_adjacency=[set() for _ in range(len(tets))]
    for uses in incident.values():
        if len(uses)==2:
            ta,tb=uses[0][0],uses[1][0]
            if materials[ta]==materials[tb] and regions[ta]==regions[tb]:tet_adjacency[ta].add(tb);tet_adjacency[tb].add(ta)
    surface_faces={};surface_face_tets={};surface_face_other_tets={};surface_face_pairs={};ring_nodes={};face_out=[];walls=[]
    for role in ROLES:
        selected=[];sel_tet=[];sel_other_tet=[];sel_pair=[]
        for face,uses in incident.items():
            if role=="O":ok=len(uses)==1 and materials[uses[0][0]]==15 and all(v in role_nodes["O"] for v in face)
            elif role=="D":ok=len(uses)==1 and materials[uses[0][0]]==16 and all(v in role_nodes["D"] for v in face)
            else:ok=len(uses)==2 and {int(materials[z[0]]) for z in uses}=={15,16} and all(v in role_nodes["I"] for v in face)
            if not ok:continue
            if role=="I":
                ti=next(z[0] for z in uses if materials[z[0]]==15)
                oti=next(z[0] for z in uses if materials[z[0]]==16)
            else:ti=uses[0][0];oti=-1
            omit=next(z[1] for z in uses if z[0]==ti)
            tet=tets[ti];oriented=[int(tet[j]) for j in range(4) if j!=omit]
            opposite=int(tet[omit]);a,b,c=current0[oriented]
            if np.dot(np.cross(b-a,c-a),current0[opposite]-(a+b+c)/3)>0:oriented[1],oriented[2]=oriented[2],oriented[1]
            selected.append(oriented);sel_tet.append(ti);sel_other_tet.append(oti);sel_pair.append((int(donors[ti]),int(regions[ti])))
        if len(selected)!=4777:raise ValueError(f"parent coarse {role} surface face count changed: {len(selected)}")
        selected=np.asarray(selected,dtype=np.int64);surf_edges=Counter()
        for f in selected:
            for j in range(3):surf_edges[tuple(sorted((int(f[j]),int(f[(j+1)%3]))))]+=1
        ring=set(v for e,n in surf_edges.items() if n==1 for v in e)
        if len(ring)!=45 or any(n not in (1,2) for n in surf_edges.values()):raise ValueError(f"parent {role} boundary loop contract changed")
        surface_faces[role]=selected;surface_face_tets[role]=np.asarray(sel_tet,dtype=np.int64);surface_face_other_tets[role]=np.asarray(sel_other_tet,dtype=np.int64);surface_face_pairs[role]=np.asarray(sel_pair,dtype=np.int64);ring_nodes[role]=ring
        for fi,(f,ti,oti,pair) in enumerate(zip(selected,sel_tet,sel_other_tet,sel_pair)):
            face_out.append([role,fi,int(ti),*map(int,f),int(materials[ti]),int(donors[ti]),int(regions[ti]),int(stable[ti]),int(oti),int(materials[oti]) if oti>=0 else -1,int(donors[oti]) if oti>=0 else -1,int(regions[oti]) if oti>=0 else -1,int(stable[oti]) if oti>=0 else -1])
    # Preserve the original two-layer head sidewall facets as a separate physical wall domain.
    for role,mat,a_role,b_role in (("skin-wall",15,"O","I"),("fat-wall",16,"I","D")):
        allowed=role_nodes[a_role]|role_nodes[b_role]
        count=0
        for face,uses in incident.items():
            if len(uses)!=1:continue
            ti,omit=uses[0]
            if materials[ti]!=mat or not all(v in allowed for v in face):continue
            if not any(v in role_nodes[a_role] for v in face) or not any(v in role_nodes[b_role] for v in face):continue
            tet=tets[ti];f=[int(tet[j]) for j in range(4) if j!=omit];op=int(tet[omit]);a,b,c=current0[f]
            if np.dot(np.cross(b-a,c-a),current0[op]-(a+b+c)/3)>0:f[1],f[2]=f[2],f[1]
            walls.append([role,count,ti,*f,mat,int(donors[ti]),int(regions[ti]),int(stable[ti])]);count+=1
        if count!=90:raise ValueError(f"parent {role} head wall expected 90 facets, got {count}")
    # Native pose matrices own per-node local coordinates; full86 weights remain separately pinned.
    pose=json.loads((CAP/"registered-skin-poses-100.json").read_text());body={int(x["body_index"]):x for x in pose["all_bodies"]}
    Rc=np.zeros((7233,3,3));Rr=np.zeros((7233,3,3));tc=np.zeros((7233,3));trst=np.zeros((7233,3));invRc=np.zeros((7233,3,3))
    for i,bid in enumerate(owner):
        x=body[int(bid)];Rc[i]=np.asarray(x["current"]["local_to_world_row_major_f32"],dtype=np.float64).reshape(3,3)
        Rr[i]=np.asarray(x["rest"]["local_to_world_row_major_f32"],dtype=np.float64).reshape(3,3)
        tc[i]=np.asarray(x["current"]["position_world_m"],dtype=np.float64);trst[i]=np.asarray(x["rest"]["position_world_m"],dtype=np.float64)
        invRc[i]=np.linalg.inv(Rc[i])
    owner_replay=tc+np.einsum("nij,nj->ni",Rc,local0)
    owner_replay_error=np.linalg.norm(owner_replay-current0,axis=1)
    # Candidate current surfaces and full399 source-domain labels.
    fine={};fine_labels={};fine_material={};fine_regs={};fine_ancestry={}
    input_paths=[CAP/"parent-layer-rows.csv",CAP/"compiled-parent-cell-mass.csv.tets.csv",CAP/"compiled-parent-cell-mass.csv.labels.csv",CAP/"compiled-parent-cell-mass.csv.weights.csv",CAP/"parent-deep-attachments.csv",CAP/"registered-skin-poses-100.json",SID/"receipt.json",SID/"node-parent-registration.csv"]
    input_paths+=list(CANDIDATES.values())
    input_paths += [SID/f"registered-{r}.csv" for r in ROLES] + [SID/f"registered-{r}-faces.csv" for r in ROLES]
    for role in ROLES:
        with np.load(CANDIDATES[role],allow_pickle=False) as z:
            pts=np.asarray(z["points_world_m"],dtype=np.float32).astype(np.float64);faces=np.asarray(z["faces"],dtype=np.int64)
            if "face_source_face_id" in z:fsid=np.asarray(z["face_source_face_id"],dtype=np.int64)
            else:fsid=np.full(len(faces),-1,dtype=np.int64)
            for key in ("face_source_original_id","face_source_identity0","face_source_identity1"):
                if key in z:fine_ancestry.setdefault(role,{})[key]=np.asarray(z[key],dtype=np.int64)
            fine[role]={"points":pts,"faces":faces,"fsid":fsid}
        nr=read_csv(SID/f"registered-{role}.csv");fr=read_csv(SID/f"registered-{role}-faces.csv")
        if len(nr)!=len(fine[role]["points"]) or len(fr)!=len(fine[role]["faces"]):raise ValueError(f"{role} sidecar row counts disagree with typed candidate")
        if [int(x["vertex"]) for x in nr]!=list(range(len(nr))) or [int(x["face"]) for x in fr]!=list(range(len(fr))):raise ValueError(f"{role} sidecar row identities are not contiguous")
        ptscsv=np.asarray([[float(x["w"+a]) for a in "xyz"] for x in nr],dtype=np.float32).astype(np.float64)
        if not np.array_equal(ptscsv,fine[role]["points"]):raise ValueError(f"{role} full399 point rows do not exactly match the typed F32 candidate")
        fc=np.asarray([[int(x[f"v{i}"]) for i in range(3)] for x in fr],dtype=np.int64)
        if not np.array_equal(fc,fine[role]["faces"]):raise ValueError(f"{role} full399 face rows do not exactly match typed candidate face rows")
        labels=np.asarray([[int(x["parent_geometric_donor_body"]),int(x["parent_anatomical_region_id"])] for x in nr],dtype=np.int64)
        material=np.asarray([int(x["parent_material_index"]) for x in nr],dtype=np.int64)
        if np.any(labels<0):raise ValueError(f"{role} full399 has an unresolved region/donor node registration")
        fine_labels[role]=labels;fine_material[role]=material;fine_regs[role]=nr
        fpairs=[]
        for tri in fine[role]["faces"]:
            cnt=Counter(tuple(map(int,fine_labels[role][v])) for v in tri)
            fpairs.append(sorted(cnt.items(),key=lambda z:(-z[1],z[0]))[0][0])
        fine[role]["face_pairs"]=np.asarray(fpairs,dtype=np.int64)
        fine[role]["incident"]=[[] for _ in range(len(pts))]
        for fi,tri in enumerate(fine[role]["faces"]):
            for v in tri:fine[role]["incident"][int(v)].append(fi)
        trees={}
        for pair in sorted(set(map(tuple,fine_labels[role].tolist()))):
            ids=np.flatnonzero(np.all(fine_labels[role]==pair,axis=1))
            trees[pair]=(ids,cKDTree(pts[ids]))
        fine[role]["vertex_trees"]=trees
        # Verify open fine surface rings; the map itself remains face-row based.
        edgecnt=Counter()
        for tri in faces:
            for j in range(3):edgecnt[tuple(sorted((int(tri[j]),int(tri[(j+1)%3]))))]+=1
        f_ring=set(v for e,n in edgecnt.items() if n==1 for v in e)
        if len(f_ring)!=45:raise ValueError(f"{role} typed surface ring changed: {len(f_ring)}")
        fine[role]["ring_vertices"]=np.asarray(sorted(f_ring),dtype=np.int64)
    # Node-local allowed source regions come only from incident parent boundary faces.
    allowed={r:defaultdict(set) for r in ROLES}
    for role in ROLES:
        for fi,f in enumerate(surface_faces[role]):
            pair=tuple(map(int,surface_face_pairs[role][fi]))
            for v in f:allowed[role][int(v)].add(pair)
    # Forward source-connected projection: coarse parent node -> nearby face in same region, starting in a fine vertex one-ring.
    target=np.zeros_like(current0);target_face={r:np.full(2411,-1,dtype=np.int64) for r in ROLES};target_bary={r:np.zeros((2411,3)) for r in ROLES};target_dist={r:np.full(2411,np.nan) for r in ROLES};target_pair={r:np.full((2411,2),-1,dtype=np.int64) for r in ROLES};target_fallback={r:np.zeros(2411,dtype=np.uint8) for r in ROLES}
    for role in ROLES:
        pts=fine[role]["points"];faces=fine[role]["faces"];face_pairs=fine[role]["face_pairs"]
        for rr,x in enumerate(layer_rows[role]):
            node=int(x["volume_node_row"]);q=current0[node];best=None
            for pair in sorted(allowed[role][node]):
                entry=fine[role]["vertex_trees"].get(pair)
                if entry is None:continue
                ids,tree=entry;_,localrow=tree.query(q,k=1);v=int(ids[int(localrow)])
                inc=fine[role]["incident"][v]
                cand=[fi for fi in inc if tuple(map(int,face_pairs[fi]))==pair]
                fallback=0
                if not cand:
                    cand=[fi for fi in inc if any(tuple(map(int,fine_labels[role][u]))==pair for u in faces[fi])];fallback=1
                if not cand:cand=inc;fallback=2
                tri=faces[np.asarray(cand,dtype=np.int64)]
                A=pts[tri[:,0]][None,:,:];B=pts[tri[:,1]][None,:,:];C3=pts[tri[:,2]][None,:,:]
                Q=q[None,:]
                qp,ba,ds=closest_batch(Q,A,B,C3);j=int(np.argmin(ds[0]));rec=(float(ds[0,j]),int(cand[j]),ba[0,j].copy(),qp[0,j].copy(),pair,fallback)
                if best is None or (rec[0],rec[1])<(best[0],best[1]):best=rec
            if best is None:raise ValueError(f"no same-region local fine surface projection for {role} parent node {node}")
            target[node]=best[3];target_face[role][rr]=best[1];target_bary[role][rr]=best[2];target_dist[role][rr]=best[0];target_pair[role][rr]=best[4];target_fallback[role][rr]=best[5]
    # Preserve all existing head rings and the 14 declared deep attachment endpoints.
    pinned=set(v for r in ROLES for v in ring_nodes[r]);anchor_matches=[];anchor_unresolved=[]
    attach_path=CAP/"parent-deep-attachments.csv"
    for a in read_csv(attach_path):
        sv=int(a["source_vertex_row"]);cand=np.flatnonzero(source_vertex[np.asarray(sorted(role_nodes["D"]),dtype=np.int64)]==sv)
        drows=np.asarray(sorted(role_nodes["D"]),dtype=np.int64)
        if len(cand)==0:
            anchor_unresolved.append({"site_row":a.get("site_row"),"source_vertex_row":sv,"reason":"no parent D source row"});continue
        wanted=None
        for keyset in (("current_x_m","current_y_m","current_z_m"),("reference_x_m","reference_y_m","reference_z_m")):
            if all(k in a for k in keyset):wanted=np.array([float(a[k]) for k in keyset]);break
        rows=drows[cand]
        if wanted is not None:
            ix=int(np.argmin(np.linalg.norm(current0[rows]-wanted,axis=1)));dist=float(np.linalg.norm(current0[rows[ix]]-wanted));node=int(rows[ix])
        else:dist=float("nan");node=int(rows[0])
        if len(rows)>1 and (wanted is None or dist>1e-5):
            anchor_unresolved.append({"site_row":a.get("site_row"),"source_vertex_row":sv,"candidate_volume_rows":rows.tolist(),"nearest_distance_m":dist});continue
        pinned.add(node);anchor_matches.append({"site_row":a.get("site_row"),"source_vertex_row":sv,"volume_node_row":node,"distance_m":dist,"donor_body":int(a.get("donor_body",-1))})
    if len(anchor_matches)!=14:write_json(OUT/"anchor-preflight-failure.json",{"matched":anchor_matches,"unresolved":anchor_unresolved});raise ValueError(f"only {len(anchor_matches)}/14 parent deep attachment anchors mapped")
    for node in pinned:target[node]=current0[node]
    delta=target-current0
    # Apply local displacement through the recorded frame owner; do not reinterpret full86 skin weights as a volume-node LBS map.
    local_delta=np.einsum("nij,nj->ni",invRc,delta)
    ref_delta=np.einsum("nij,nj->ni",Rr,local_delta)
    # Joint backtracking enforces emitted F32 positive cells, positive paired thickness, and inherited per-cell mass-density bounds.
    v0cur=signed_volumes(current0.astype(np.float32).astype(np.float64),tets);v0ref=signed_volumes(reference0.astype(np.float32).astype(np.float64),tets)
    if np.any(v0cur<=0) or np.any(v0ref<=0):raise ValueError("parent330 baseline has nonpositive current/reference tet orientation")
    alpha=np.ones(7233,dtype=np.float64);alpha[list(pinned)]=0.0
    pair_rows=np.arange(2411,dtype=np.int64);role_global={r:np.asarray([int(x["volume_node_row"]) for x in layer_rows[r]],dtype=np.int64) for r in ROLES}
    mass_lower={15:900.0,16:850.0};mass_upper={15:1100.0,16:950.0};constraint_log=[]
    def material_gates(cand_ref):
        ref32=cand_ref.astype(np.float32).astype(np.float64)
        vv=signed_volumes(ref32,tets);dens=masses/np.abs(vv)
        bad=(vv<=0)|(~np.isfinite(dens))
        for mat,lo in mass_lower.items():bad|=(materials==mat)&((dens<lo)|(dens>mass_upper[mat]))
        return vv,dens,bad
    def thickness_bad(cand_cur):
        bad=[]
        for a,b,thick_name in (("O","I","skin_thickness_m"),("I","D","fat_thickness_m")):
            aa=role_global[a];bb=role_global[b];v0=current0[aa]-current0[bb];v1=cand_cur[aa]-cand_cur[bb]
            dist=np.linalg.norm(v1,axis=1);orient=np.einsum("ij,ij->i",v0,v1)
            idx=np.flatnonzero((dist<=0)|(orient<=0))
            if len(idx):bad.append((a,b,idx))
        return bad
    maxbt=80;done=False
    for it in range(maxbt):
        cand_cur=(current0+alpha[:,None]*delta).astype(np.float32).astype(np.float64)
        cand_ref=(reference0+alpha[:,None]*ref_delta).astype(np.float32).astype(np.float64)
        vcur=signed_volumes(cand_cur,tets);vref,dens,badref=material_gates(cand_ref)
        badcur=vcur<=0
        badidx=np.flatnonzero(badcur|badref)
        thick=thickness_bad(cand_cur)
        constraint_log.append({"iteration":it,"bad_current_tets":int(badcur.sum()),"bad_reference_or_density_tets":int(badref.sum()),"bad_thickness_pairs":int(sum(len(x[2]) for x in thick)),"min_alpha":float(alpha.min()),"moved_nodes":int(np.count_nonzero(alpha>0))})
        if not len(badidx) and not thick:done=True;break
        changed=False
        for ti in badidx:
            nodes=tets[int(ti)]
            movable=nodes[alpha[nodes]>0]
            if len(movable):alpha[movable]*=0.5;changed=True
        for a,b,ids in thick:
            nodes=np.concatenate((role_global[a][ids],role_global[b][ids]))
            movable=nodes[alpha[nodes]>0]
            if len(movable):alpha[movable]*=0.5;changed=True
        if not changed:break
    if not done:raise ValueError("joint local backtracking could not find a positive-cell, positive-thickness, mass-bounded candidate")
    current1=(current0+alpha[:,None]*delta).astype(np.float32).astype(np.float64)
    reference1=(reference0+alpha[:,None]*ref_delta).astype(np.float32).astype(np.float64)
    local1=local0+alpha[:,None]*local_delta
    vcur=signed_volumes(current1,tets);vref=signed_volumes(reference1,tets);dens=masses/vref
    # Candidate output is durable before the more expensive all-detail reverse map.
    np.savez_compressed(OUT/"candidate.npz",points_current_world_m_f32=current1.astype(np.float32),points_reference_world_m_f64=reference1,
        points_donor_local_m_f64=local1,volume_tetrahedra=tets,material_index=materials,geometric_donor_body=donors,
        anatomical_region_id=regions,stable_source_face_id=stable,parent_volume_node_row=np.arange(7233,dtype=np.int64),
        child_parent_tet_row=np.arange(28632,dtype=np.int64),full86_weights=weights,
        coarse_boundary_faces_O=surface_faces["O"],coarse_boundary_faces_I=surface_faces["I"],coarse_boundary_faces_D=surface_faces["D"],
        coarse_boundary_other_parent_tets_O=surface_face_other_tets["O"],coarse_boundary_other_parent_tets_I=surface_face_other_tets["I"],coarse_boundary_other_parent_tets_D=surface_face_other_tets["D"],
        coarse_to_fine_face_O=target_face["O"],coarse_to_fine_bary_O=target_bary["O"],
        coarse_to_fine_face_I=target_face["I"],coarse_to_fine_bary_I=target_bary["I"],
        coarse_to_fine_face_D=target_face["D"],coarse_to_fine_bary_D=target_bary["D"],
        target_current_world_m=target,pinned_volume_node_rows=np.asarray(sorted(pinned),dtype=np.int64))
    node_path=OUT/"parent-reduced-boundary.node";ele_path=OUT/"parent-reduced-boundary.ele"
    with node_path.open("w") as f:
        f.write("7233 3 0 0\n")
        for i,pnt in enumerate(reference1):f.write(f"{i+1} {pnt[0]:.17g} {pnt[1]:.17g} {pnt[2]:.17g}\n")
    with ele_path.open("w") as f:
        f.write("28632 4 1\n")
        for i,(tet,mat) in enumerate(zip(tets,materials)):f.write(f"{i+1} {tet[0]+1} {tet[1]+1} {tet[2]+1} {tet[3]+1} {mat}\n")
    labels_path=OUT/"parent-reduced-boundary.labels.csv"
    write_csv(labels_path,["child_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"],
              ([i,int(materials[i]),int(donors[i]),int(regions[i]),int(stable[i])] for i in range(len(tets))))
    parent_map=OUT/"child-parent-identity.csv"
    write_csv(parent_map,["child_tet_row","parent_tet_row","match_kind"],([i,i,"exact_parent_tet_identity"] for i in range(len(tets))))
    # Per-node target and source-frame receipt.
    node_header=["volume_node_row","role","role_row","source_vertex_row","source_face_row","frame_owner_body","source_payload_sha256","source_layer_coordinates_sha256",
      "parent_current_x_m","parent_current_y_m","parent_current_z_m","parent_reference_x_m","parent_reference_y_m","parent_reference_z_m",
      "candidate_current_x_m","candidate_current_y_m","candidate_current_z_m","candidate_reference_x_m","candidate_reference_y_m","candidate_reference_z_m",
      "candidate_donor_local_x_m","candidate_donor_local_y_m","candidate_donor_local_z_m","accepted_alpha",
      "fine_surface_face_row","fine_bary0","fine_bary1","fine_bary2","fine_projection_residual_x_m","fine_projection_residual_y_m","fine_projection_residual_z_m","fine_projection_distance_m",
      "fine_parent_donor_body","fine_parent_region_id","embedding_kind","jacobian_current_R00","jacobian_current_R01","jacobian_current_R02","jacobian_current_R10","jacobian_current_R11","jacobian_current_R12","jacobian_current_R20","jacobian_current_R21","jacobian_current_R22","jacobian_det","jacobian_condition"]
    node_out=[]
    for role in ROLES:
        for rr,x in enumerate(layer_rows[role]):
            node=int(x["volume_node_row"]);frow=int(target_face[role][rr]);bary=target_bary[role][rr];pr=fine[role]["points"][fine[role]["faces"][frow]];q=bary@pr;res=current1[node]-q
            J=Rc[node];cond=float(np.linalg.cond(J));det=float(np.linalg.det(J))
            node_out.append([node,role,rr,source_vertex[node],int(x["source_face_row"]),owner[node],x["source_payload_sha256"],x["source_layer_coordinates_sha256"],
              *current0[node],*reference0[node],*current1[node],*reference1[node],*local1[node],alpha[node],frow,*bary,*res,float(np.linalg.norm(res)),int(target_pair[role][rr,0]),int(target_pair[role][rr,1]),"same_region_local_one_ring_surface_projection",*J.reshape(-1),det,cond])
    write_csv(OUT/"coarse-node-to-fine-surface.csv",node_header,node_out)
    write_csv(OUT/"coarse-boundary-faces.csv",["role","face_row","parent_tet_row","volume_node0","volume_node1","volume_node2","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id","opposite_parent_tet_row","opposite_material_index","opposite_geometric_donor_body","opposite_anatomical_region_id","opposite_stable_source_face_id"],face_out)
    write_csv(OUT/"preserved-parent-head-sidewall-faces.csv",["wall_role","wall_face_row","parent_tet_row","volume_node0","volume_node1","volume_node2","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"],walls)
    write_csv(OUT/"preserved-attachment-anchors.csv",["site_row","source_vertex_row","volume_node_row","distance_m","donor_body"],([x["site_row"],x["source_vertex_row"],x["volume_node_row"],x["distance_m"],x["donor_body"]] for x in anchor_matches))
    # Reverse detail-to-coarse mapping requires exact registered stable-face identity and a bounded
    # same-material/region path through the original parent tetra adjacency graph. No donor-wide fallback.
    reverse_all=[];region_stats={};refinement=defaultdict(lambda:{"max":0.0,"n":0,"dist":[]});fallback_counts=Counter();unmapped=[]
    max_boundary_walk_hops=6
    for role in ROLES:
        fpoints=fine[role]["points"];ffaces=fine[role]["faces"];coarse=surface_faces[role];cpoints=current1
        cpairs=surface_face_pairs[role];ctets=surface_face_tets[role];other_tets=surface_face_other_tets[role]
        c_tri=cpoints[coarse]
        tet_face=defaultdict(list);face_side_meta=[{} for _ in range(len(coarse))]
        group_faces=defaultdict(list);group_faces_material_region=defaultdict(list)
        for fi,(tetrow,othertet,pair) in enumerate(zip(ctets,other_tets,cpairs)):
            primary=int(tetrow);group_faces[tuple(map(int,pair))].append(fi)
            group_faces_material_region[(int(materials[primary]),int(regions[primary]))].append(fi)
            sides=[primary]+([int(othertet)] if int(othertet)>=0 else [])
            for side_tet in sides:
                face_side_meta[fi][side_tet]=(int(materials[side_tet]),int(regions[side_tet]),int(stable[side_tet]),int(donors[side_tet]))
                tet_face[side_tet].append(fi)
        regs=fine_regs[role];node_parent_tets=np.full(len(regs),-1,dtype=np.int64)
        for j,x in enumerate(regs):
            try:node_parent_tets[j]=int(x["parent_tet_row"])
            except (KeyError,ValueError):pass
        nodepairs=fine_labels[role];node_material=fine_material[role]
        best_face=np.full(len(fpoints),-1,dtype=np.int64);best_bary=np.zeros((len(fpoints),3));best_dist=np.full(len(fpoints),np.nan)
        kind=np.full(len(fpoints),"",dtype=object);direct_mask=np.zeros(len(fpoints),dtype=bool)
        best_side_tet=np.full(len(fpoints),-1,dtype=np.int64);best_start_tet=np.full(len(fpoints),-1,dtype=np.int64)
        best_hops=np.full(len(fpoints),-1,dtype=np.int64);best_path=np.full(len(fpoints),"",dtype=object)
        for vi,q in enumerate(fpoints):
            x=regs[vi];fine_mat=int(node_material[vi]);fine_donor=int(x["parent_geometric_donor_body"])
            fine_region=int(x["parent_anatomical_region_id"]);fine_stable=int(x["parent_stable_source_face_id"])
            primary=int(node_parent_tets[vi])
            try:tied=[int(t) for t in json.loads(x.get("parent_tied_tet_rows_json","[]"))]
            except (TypeError,ValueError):tied=[]
            start_rows=[]
            for t in [primary]+tied:
                if 0<=t<len(tets) and materials[t]==fine_mat and regions[t]==fine_region and t not in start_rows:start_rows.append(t)
            if fine_stable<0 or not start_rows:
                unmapped.append({"role":role,"fine_vertex_row":vi,"fine_material_index":fine_mat,"fine_donor_body":fine_donor,"fine_region_id":fine_region,"fine_stable_source_face_id":fine_stable,"fine_parent_tet_row":primary,"reason":"missing stable source-face identity or same-material/region parent-tet seed"})
                continue
            queue=deque((t,0,t,(t,)) for t in start_rows);seen=set(start_rows);found=[];found_hops=None
            while queue:
                ti,hops,start_tet,path=queue.popleft()
                if found_hops is not None and hops>found_hops:break
                for fi in tet_face.get(ti,[]):
                    meta=face_side_meta[fi].get(ti)
                    if meta is None:continue
                    if meta[0]==fine_mat and meta[1]==fine_region and meta[2]==fine_stable:
                        found_hops=hops;found.append((fi,ti,start_tet,path))
                if found_hops is not None:continue
                if hops>=max_boundary_walk_hops:continue
                for nb in sorted(tet_adjacency[ti]):
                    if nb not in seen and materials[nb]==fine_mat and regions[nb]==fine_region:
                        seen.add(nb);queue.append((nb,hops+1,start_tet,path+(nb,)))
            if not found:
                unmapped.append({"role":role,"fine_vertex_row":vi,"fine_material_index":fine_mat,"fine_donor_body":fine_donor,"fine_region_id":fine_region,"fine_stable_source_face_id":fine_stable,"fine_parent_tet_row":primary,"registered_tied_tet_rows_json":x.get("parent_tied_tet_rows_json","[]"),"reason":f"no boundary face with same material/region/stable source-face ID within {max_boundary_walk_hops} parent-tet hops"})
                continue
            face_ids=np.asarray(sorted(set(int(r[0]) for r in found)),dtype=np.int64);tri=c_tri[face_ids]
            Q=q[None,:];qp,ba,ds=closest_batch(Q,tri[:,0][None,:,:],tri[:,1][None,:,:],tri[:,2][None,:,:])
            order=np.lexsort((face_ids,ds[0]));j=int(order[0]);fi=int(face_ids[j]);dist=float(ds[0,j]);bary=ba[0,j]
            choices=[r for r in found if int(r[0])==fi]
            choice=min(choices,key=lambda r:(len(r[3]),r[2],r[1],r[3]))
            _,side_tet,start_tet,path=choice;hops=len(path)-1
            method=("primary_registered_parent_tet_boundary_stable_face" if hops==0 and start_tet==primary else
                    "registered_parent_tie_boundary_stable_face" if hops==0 else
                    "bounded_same_material_region_parent_tet_walk_to_stable_boundary_face")
            best_face[vi]=fi;best_bary[vi]=bary;best_dist[vi]=dist;kind[vi]=method
            best_side_tet[vi]=side_tet;best_start_tet[vi]=start_tet;best_hops[vi]=hops;best_path[vi]=json.dumps(list(path),separators=(",",":"))
            direct_mask[vi]=hops==0 and start_tet==primary
            fallback_counts[(role,method)]+=1
        if unmapped:
            write_csv(OUT/"unmapped-fine-vertices.csv",["role","fine_vertex_row","fine_material_index","fine_donor_body","fine_region_id","fine_stable_source_face_id","fine_parent_tet_row","registered_tied_tet_rows_json","reason"],([u.get(k,"") for k in ["role","fine_vertex_row","fine_material_index","fine_donor_body","fine_region_id","fine_stable_source_face_id","fine_parent_tet_row","registered_tied_tet_rows_json","reason"]] for u in unmapped))
        if np.any(best_face<0):raise ValueError(f"{role} reverse detail-to-coarse map has {int(np.sum(best_face<0))} unresolved vertices; no region-only or donor-wide fallback used")
        # Keep inferred parent registration separate from the typed detail surface ancestry.
        outrows=[];face_max=np.zeros(len(coarse));face_count=np.zeros(len(coarse),dtype=np.int64)
        for vi,x in enumerate(regs):
            fi=int(best_face[vi]);bary=best_bary[vi];qproj=bary@c_tri[fi];res=fpoints[vi]-qproj;dist=float(np.linalg.norm(res));face_max[fi]=max(face_max[fi],dist);face_count[fi]+=1
            srcface=coarse[fi];primary_tet=int(ctets[fi]);mapped_tet=int(best_side_tet[vi]);mapped_meta=face_side_meta[fi][mapped_tet]
            incident_face_rows=fine[role]["incident"][vi]
            outrows.append([vi,role,fi,primary_tet,mapped_tet,int(best_start_tet[vi]),int(best_hops[vi]),best_path[vi],
                *map(int,srcface),int(materials[primary_tet]),int(donors[primary_tet]),int(regions[primary_tet]),int(stable[primary_tet]),
                mapped_meta[0],mapped_meta[3],mapped_meta[1],mapped_meta[2],
                *bary,*res,dist,kind[vi],fine_material[role][vi],x["parent_geometric_donor_body"],x["parent_anatomical_region_id"],
                x["parent_stable_source_face_id"],x["parent_registration_kind"],x["parent_tet_row"],x.get("parent_tied_tet_rows_json","[]"),
                x["parent_registration_distance_m"],x["parent_registration_tie_count"],x["frame_owner_method"],
                x["source_domain"],x["source_mapping_kind"],x["source_vertex_row"],x["source_face_row"],x["source_bary0"],x["source_bary1"],x["source_bary2"],
                json.dumps(incident_face_rows,separators=(",",":"))])
        header=["fine_vertex_row","role","coarse_face_row","coarse_boundary_primary_parent_tet_row","mapped_boundary_side_parent_tet_row","boundary_walk_start_parent_tet_row","boundary_walk_hop_count","boundary_walk_parent_tet_rows_json",
                "volume_node0","volume_node1","volume_node2","coarse_primary_material_index","coarse_primary_geometric_donor_body","coarse_primary_region_id","coarse_primary_stable_source_face_id",
                "mapped_side_material_index","mapped_side_geometric_donor_body","mapped_side_region_id","mapped_side_stable_source_face_id",
                "bary0","bary1","bary2","residual_x_m","residual_y_m","residual_z_m","projection_distance_m","mapping_kind",
                "fine_parent_material_index","fine_parent_geometric_donor_body","fine_parent_region_id","fine_parent_stable_source_face_id","fine_parent_registration_kind","fine_parent_tet_row","fine_parent_tied_tet_rows_json",
                "fine_parent_registration_distance_m","fine_parent_registration_tie_count","fine_frame_owner_body","fine_source_domain","fine_source_mapping_kind","fine_source_vertex_row","fine_source_face_row","fine_source_bary0","fine_source_bary1","fine_source_bary2","fine_incident_detail_face_rows_json"]
        write_csv(OUT/f"fine-{role}-to-coarse-boundary.csv",header,outrows)
        dd=best_dist
        groups=defaultdict(list)
        for vi,pair in enumerate(nodepairs):groups[(int(node_material[vi]),int(pair[0]),int(pair[1]))].append(vi)
        for key,ids0 in sorted(groups.items()):
            ids=np.asarray(ids0,dtype=np.int64);v=dd[ids];w=int(ids[np.argmax(v)])
            region_stats[f"{role}:{key[0]}:{key[1]}:{key[2]}"]={"role":role,"material_index":key[0],"fine_geometric_donor_body":key[1],"anatomical_region_id":key[2],"fine_vertex_count":int(len(ids)),"distance_m_quantiles":{"p50":float(np.quantile(v,.5)),"p90":float(np.quantile(v,.9)),"p95":float(np.quantile(v,.95)),"p99":float(np.quantile(v,.99)),"max":float(v.max())},"worst_fine_vertex_row":w,"worst_coarse_face_row":int(best_face[w]),"mapped_by_primary_parent_tet_boundary_fraction":float(direct_mask[ids].mean()),"max_parent_tet_walk_hops":int(best_hops[ids].max()),"mapping_kind_counts":dict(Counter(str(kind[j]) for j in ids))}
        # Refine only face regions with a measured fine-to-coarse residual tail.
        for key,ids0 in sorted(group_faces_material_region.items()):
            face_ids=np.asarray(ids0,dtype=np.int64);values=face_max[face_ids]
            if not len(values):continue
            cutoff=float(np.quantile(values,.95));chosen=face_ids[values>=cutoff]
            for fi in chosen:
                if face_max[fi]<=0:continue
                tri=coarse[int(fi)];pair=tuple(map(int,cpairs[int(fi)]));mat_face=int(materials[int(ctets[int(fi)])])
                for j in range(3):
                    e=tuple(sorted((int(tri[j]),int(tri[(j+1)%3]))))
                    refinement[(role,mat_face,int(pair[0]),int(pair[1]),e)]["max"]=max(refinement[(role,mat_face,int(pair[0]),int(pair[1]),e)]["max"],float(face_max[fi]))
                    refinement[(role,mat_face,int(pair[0]),int(pair[1]),e)]["n"]+=1
                    refinement[(role,mat_face,int(pair[0]),int(pair[1]),e)]["dist"].append(float(face_max[fi]))
    if unmapped:raise ValueError(f"reverse maps have {len(unmapped)} unresolved rows")
    refinement_rows=[]
    for (role,material,donor,region,u,v),r in sorted(refinement.items()):refinement_rows.append([role,material,donor,region,u,v,r["max"],r["n"],"top_5_percent_parent_face_projection_error_within_material_region"])
    write_csv(OUT/"measured-local-refinement-edges.csv",["role","material_index","geometric_donor_body","anatomical_region_id","parent_volume_node0","parent_volume_node1","max_fine_to_coarse_error_m","selected_parent_faces","selection_basis"],refinement_rows)
    # Cell, density, thickness, topology, and frame summaries.
    thickness={}
    for name,a,b,field in (("skin","O","I","skin_thickness_m"),("fat","I","D","fat_thickness_m")):
        aa=role_global[a];bb=role_global[b];old=current0[aa]-current0[bb];new=current1[aa]-current1[bb];d=np.linalg.norm(new,axis=1);declared=np.asarray([float(x[field]) for x in layer_rows[a]])
        thickness[name]={"paired_role_row_count":len(d),"min_distance_m":float(d.min()),"distance_quantiles_m":{"p10":float(np.quantile(d,.1)),"p50":float(np.quantile(d,.5)),"p90":float(np.quantile(d,.9))},"declared_field_m_quantiles":{"p10":float(np.quantile(declared,.1)),"p50":float(np.quantile(declared,.5)),"p90":float(np.quantile(declared,.9))},"distance_to_declared_ratio_quantiles":{"p10":float(np.quantile(d/declared,.1)),"p50":float(np.quantile(d/declared,.5)),"p90":float(np.quantile(d/declared,.9))},"nonpositive_or_reversed_pairs":int(np.sum((d<=0)|(np.einsum("ij,ij->i",old,new)<=0)))}
    regional_mass={}
    for mat in sorted(set(map(int,materials))):
        mask=materials==mat;vol0=np.abs(signed_volumes(reference0,tets[mask]));vol1=np.abs(signed_volumes(reference1,tets[mask]));rho=masses[mask]/vol1
        regional_mass[str(mat)]={"parent_tet_count":int(mask.sum()),"volume_before_m3":float(vol0.sum()),"volume_after_m3":float(vol1.sum()),"relative_volume_change":float(vol1.sum()/vol0.sum()-1),"mass_kg":float(masses[mask].sum()),"density_kg_m3_min":float(rho.min()),"density_kg_m3_max":float(rho.max()),"density_bounds_kg_m3":[mass_lower[mat],mass_upper[mat]],"density_out_of_bounds_count":int(np.sum((rho<mass_lower[mat])|(rho>mass_upper[mat]))),"min_current_signed_tet_volume_m3":float(vcur[mask].min()),"min_reference_signed_tet_volume_m3":float(vref[mask].min())}
    forward={}
    for role in ROLES:
        ds=target_dist[role];accepted=np.linalg.norm(current1[role_global[role]]-np.asarray([target[ni] for ni in role_global[role]]),axis=1)
        forward[role]={"coarse_node_count":2411,"projection_error_m_quantiles":{"p50":float(np.quantile(ds,.5)),"p90":float(np.quantile(ds,.9)),"p95":float(np.quantile(ds,.95)),"p99":float(np.quantile(ds,.99)),"max":float(ds.max())},"accepted_residual_to_target_m_quantiles":{"p50":float(np.quantile(accepted,.5)),"p90":float(np.quantile(accepted,.9)),"p95":float(np.quantile(accepted,.95)),"p99":float(np.quantile(accepted,.99)),"max":float(accepted.max())},"projection_fallback_count":int(np.count_nonzero(target_fallback[role]))}
    receipt={"schema":"NumiHuman.reduced-parent-boundary-candidate.v1","status":"bounded_reduced_boundary_candidate_not_physical_admission","physical_admission":False,
        "input_files":{str(x):sha(x) for x in input_paths},"parent_node_count":7233,"parent_tet_count":28632,"parent_tet_identity_preserved":True,
        "role_node_counts":{r:len(layer_rows[r]) for r in ROLES},"coarse_role_face_counts":{r:len(surface_faces[r]) for r in ROLES},"open_boundary_vertex_counts":{r:len(ring_nodes[r]) for r in ROLES},"separate_parent_head_wall_facets":len(walls),
        "exact_candidate_source_hashes":{r:sha(CANDIDATES[r]) for r in ROLES},"source_projection":"face+bary rows restricted to same parent material/donor/region and local incident fine-surface one-rings; reverse map prefers the registered parent tet's role-boundary face and otherwise uses a certified nearest triangle in the same material/donor/region",
        "source_ancestry":"existing parent volume node/tet identities and detailed typed face rows remain separate domains; geometric projection is explicitly not topological ancestry",
        "fit":"simultaneous O/I/D local target projection with per-node backtracking against current/reference positive tetrahedra, positive paired layer order, and original parent cell mass density bounds; no new refinement nodes in this initial candidate",
        "fit_backtracking":constraint_log,"minimum_accepted_alpha":float(alpha.min()),"median_accepted_alpha":float(np.median(alpha)),"moved_parent_node_count":int(np.count_nonzero(np.linalg.norm(current1-current0,axis=1)>0)),
        "pinned_head_ring_volume_node_rows":sorted(set.union(*(ring_nodes[r] for r in ROLES))),"pinned_attachment_count":len(anchor_matches),"attachment_matches":anchor_matches,
        "owner_frame_baseline_current_replay_error_m":{"max":float(owner_replay_error.max()),"median":float(np.median(owner_replay_error))},"frame_jacobian":"exact captured current donor-frame R per volume node; local target delta uses inverse current R; reference delta uses captured rest R; full86 weights are copied unchanged and not treated as volume-node current LBS",
        "positive_cell_checks":{"current_float32_nonpositive":int(np.sum(vcur<=0)),"reference_float64_nonpositive":int(np.sum(vref<=0)),"parent_tet_row_identity":True},"paired_thickness":thickness,"per_material_mass_and_volume":regional_mass,
        "forward_coarse_to_fine_error":forward,"reverse_fine_to_coarse_region_error":region_stats,"reverse_map_method_counts":{f"{r[0]}:{r[1]}":n for r,n in fallback_counts.items()},"refinement_edge_count":len(refinement_rows),"output_sha256":{},"elapsed_seconds_before_receipt":time.time()-t0}
    outputs=[OUT/"candidate.npz",node_path,ele_path,labels_path,parent_map,OUT/"coarse-node-to-fine-surface.csv",OUT/"coarse-boundary-faces.csv",OUT/"preserved-parent-head-sidewall-faces.csv",OUT/"preserved-attachment-anchors.csv",OUT/"measured-local-refinement-edges.csv"]
    outputs += [OUT/f"fine-{r}-to-coarse-boundary.csv" for r in ROLES]
    receipt["output_sha256"]={x.name:sha(x) for x in outputs}
    write_json(OUT/"receipt.json",receipt)
    print(json.dumps({"candidate":str(OUT/"candidate.npz"),"receipt":str(OUT/"receipt.json"),"counts":{"nodes":7233,"tets":28632,"faces_by_role":{r:len(surface_faces[r]) for r in ROLES}},"moved_nodes":receipt["moved_parent_node_count"],"alpha_min":receipt["minimum_accepted_alpha"],"unresolved":len(unmapped),"volume":regional_mass,"elapsed_s":time.time()-t0},indent=2))

if __name__=="__main__":
    try:main()
    except Exception as e:
        import traceback
        print("ERROR",repr(e),file=sys.stderr);traceback.print_exc();sys.exit(2)
