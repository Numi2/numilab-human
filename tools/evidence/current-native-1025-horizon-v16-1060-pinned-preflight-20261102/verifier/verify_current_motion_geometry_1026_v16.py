#!/usr/bin/env python3
"""Verify accepted terminal O/I/D MRV geometry against current 217 FEM nodes."""
import argparse,csv,gzip,hashlib,importlib,json,math,re,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
# Set only from explicit CLI pins in this version. A refreshed source export must
# bind its own current orientation audit and validator; no 1044 baseline default.
ORIENTATION_AUDIT=None
ORIENTATION_AUDIT_SHA=None
ORIENTATION_OWNER=None
ORIENTATION_OWNER_SHA=None
EXACT_ROOT=H.parent/"deep-boundary-checkpoint-349/payload/exact-owner"
sys.path.insert(0,str(EXACT_ROOT))
EXACT=importlib.import_module("numilab_human.cardiac_cavity_intersections")
def need(x,m):
    if not x: raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def npz(p,keys=None):
    with np.load(p,allow_pickle=False) as z:
        selected=z.files if keys is None else keys
        return {k:np.array(z[k],copy=True) for k in selected}
def fields(line):
    d={}
    if line.startswith("resting_FEM_endpoint_capture=begin"):
        tokens=line.rstrip("\n").split()
        tokens=tokens[1:]
    else:
        tokens=line.rstrip("\n").split(",")[1:]
    for x in tokens:
        if "=" in x:
            k,v=x.split("=",1);d[k]=v
    return d
def capture(path):
    nodes={};tets={};begin=None;ends=0;boundary=None
    for line in Path(path).open(errors="strict"):
        if line.startswith("accepted_fem_boundary_render=prepared") and boundary is None:
            boundary=line.rstrip("\n")
        elif line.startswith("resting_FEM_endpoint_capture=begin"):
            need(begin is None,"duplicate endpoint capture");begin=fields(line)
        elif line.startswith("resting_FEM_endpoint_capture=end"):ends+=1
        elif line.startswith("resting_FEM_node,"):
            d=fields(line);i=int(d["global"]);need(i not in nodes,"duplicate native node")
            nodes[i]=(np.array([float(d["initial_"+a]) for a in "xyz"],np.float32),
                      np.array([float(d["accepted_"+a]) for a in "xyz"],np.float32))
        elif line.startswith("resting_FEM_tet,"):
            d=fields(line);i=int(d["global"]);need(i not in tets,"duplicate native tet")
            tets[i]=(np.array([int(d["n"+str(k)]) for k in range(4)],np.int64),float(d["J"]))
    if begin is None:
        need(ends==0 and not nodes and not tets,"partial terminal FEM endpoint dump")
        return None,{}, {},boundary
    need(ends==1,"terminal capture section incomplete")
    return begin,nodes,tets,boundary

def native_install_summary(path,mesh):
    install=[];prepared=[]
    with Path(path).open(errors="strict") as f:
        for line in f:
            if line.startswith("whole_body_skin_fat_registered_volume_install="):
                toks=line.strip().split();status=toks[0].split("=",1)[1];d={}
                for token in toks[1:]:
                    if "=" in token:
                        k,v=token.split("=",1);d[k]=v
                install.append((status,d))
            elif line.startswith("resting_skin_fat_whole_body_preparation="):
                toks=line.strip().split();status=toks[0].split("=",1)[1];d={}
                for token in toks[1:]:
                    if "=" in token:
                        k,v=token.split("=",1);d[k]=v
                prepared.append((status,d))
    need(len(install)==1 and install[0][0]=="PASS","registered-volume install pass is missing, failed, or duplicated")
    need(len(prepared)==1 and prepared[0][0]=="accepted","whole-body Skin/Fat preparation record is missing, rejected, or duplicated")
    ins=install[0][1];prep=prepared[0][1]
    expected={"O_rows":int(np.count_nonzero(mesh["R"]["O"]["row"]>=0)),
              "I_rows":int(np.count_nonzero(mesh["R"]["I"]["row"]>=0)),
              "D_rows":int(np.count_nonzero(mesh["R"]["D"]["row"]>=0)),
              "volume_nodes":len(mesh["joint"]),"child_tets":len(mesh["expected"]),
              "shared_fem_nodes":len(mesh["joint"]),"shared_fem_tets":len(mesh["expected"]),
              "skin_tets":int(np.count_nonzero(mesh["byrole"]=="Skin")),
              "fat_tets":int(np.count_nonzero(mesh["byrole"]=="Fat"))}
    for k in ("O_rows","I_rows","D_rows","volume_nodes","child_tets"):
        need(int(ins.get(k,-1))==expected[k],"native registered-volume install count mismatch: "+k)
    for k in ("shared_fem_nodes","shared_fem_tets","skin_tets","fat_tets"):
        need(int(prep.get(k,-1))==expected[k],"native whole-body FEM preparation count mismatch: "+k)
    return {"status":"PASS_NATIVE_INSTALL_COUNTS","counts":expected}

def compact_terminal_audit(path):
    rows=[]
    with Path(path).open(errors="strict") as f:
        for line in f:
            if line.startswith("resting_FEM_terminal_audit="):
                tokens=line.strip().split();status=tokens[0].split("=",1)[1];d={}
                for token in tokens[1:]:
                    if "=" in token:
                        k,v=token.split("=",1);d[k]=v
                rows.append((status,d))
    need(len(rows)==1,"native compact terminal audit is missing or duplicated")
    status,d=rows[0]
    need(status=="passed","native compact terminal audit did not pass")
    required=("expected_steps","accepted_steps","min_J","max_J","max_displacement_m","capture")
    need(all(k in d for k in required),"native compact terminal audit fields incomplete")
    expected=int(d["expected_steps"]);accepted=int(d["accepted_steps"])
    minj=float(d["min_J"]);maxj=float(d["max_J"]);disp=float(d["max_displacement_m"])
    need(expected>0 and accepted==expected,"native compact accepted-step count mismatch")
    need(all(math.isfinite(v) for v in (minj,maxj,disp)) and minj>0.35 and maxj>=minj and disp>=0.0,
         "native compact terminal audit has invalid J/displacement")
    need(d["capture"]=="terminal_only","native compact audit capture scope is not terminal-only")
    return {"expected_steps":expected,"accepted_steps":accepted,"minimum_J_global":minj,
            "maximum_J_global":maxj,"maximum_node_displacement_m":disp,
            "capture_scope":d["capture"],"status":"PASS_NATIVE_COMPACT_TERMINAL_AUDIT"}
def closed(f):
    f=np.asarray(f,np.int64);e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    s=np.sort(e,axis=1);u,inv,c=np.unique(s,axis=0,return_inverse=True,return_counts=True)
    orient=np.bincount(inv,weights=np.where(e[:,0]==s[:,0],1,-1),minlength=len(u))
    return {"closed_oriented":bool(np.all(c==2)&np.all(orient==0)),"edges":len(u),
            "boundary_edges":int(np.count_nonzero(c==1)),"nonmanifold_edges":int(np.count_nonzero(c>2)),
            "orientation_conflict_edges":int(np.count_nonzero((c==2)&(orient!=0))),
            "bad_edges":int(np.count_nonzero((c!=2)|(orient!=0)))}
def node_export(p):
    rr=[]
    with Path(p).open(newline="") as f:
        r=csv.DictReader(f)
        need(r.fieldnames==["source_vertex_row","x_m","y_m","z_m","active"],"MRV node CSV schema")
        for x in r:rr.append((int(x["source_vertex_row"]),[float(x[a+"_m"]) for a in "xyz"],x["active"]=="1"))
    need([x[0] for x in rr]==list(range(len(rr))),"MRV node rows are not ordered complete interval")
    return np.asarray([x[1] for x in rr],np.float32),np.asarray([x[2] for x in rr],bool)
def face_export(p):
    out=[]
    with Path(p).open(newline="") as f:
        r=csv.DictReader(f);need(r.fieldnames==["source_face_row","node0","node1","node2"],"MRV face CSV schema")
        for x in r:out.append([int(x[f"node{k}"]) for k in range(3)])
    return np.asarray(out,np.int64)
def exact_records(points,faces,joint):
    groups={}
    for i,x in enumerate(np.asarray(joint,np.int64)):
        if x>=0:groups.setdefault(int(x),[]).append(i)
    seams=[v for v in groups.values() if len(v)>1]
    rec,_=EXACT.float32_surface_records(np.asarray(points,np.float32),np.asarray(faces,np.int64),source_seam_groups=seams)
    return rec
def run_export(exe,pack,first,count,prefix,bones=None):
    argv=[str(exe),str(pack),str(first),str(count),str(prefix)]
    if bones:argv.append(str(bones))
    r=subprocess.run(argv,text=True,capture_output=True)
    need(r.returncode==0,"MRV exporter failed: "+r.stderr.strip())
    return Path(str(prefix)+".nodes.csv"),Path(str(prefix)+".faces.csv")
def parse_bones(p):
    pts=[]
    with Path(p).open(newline="") as f:
        for r in csv.DictReader(f):
            for c in range(3):
                pts.append([struct.unpack("<f",struct.pack("<I",int(r[f"v{c}{a}_bits_u32"],16)))[0] for a in "xyz"])
    need(pts,"bone CSV empty")
    pts=np.asarray(pts,np.float32)
    return pts,np.arange(len(pts),dtype=np.int64).reshape(-1,3)
def build_mesh(state,handoff):
    cross=npz(handoff/"skin-fat-combined-registered-surface-node-map.npz",
              [k for r in ("O","I","D") for k in (r+"_source_row_to_joint_node",r+"_active_source_vertex_rows",r+"_inactive_source_vertex_rows")])
    R={}
    for role in ("O","I","D"):
        ent=state["source_surfaces"][role];need(sha(ent["path"])==ent["sha256"],role+" source surface hash mismatch")
        surf=npz(ent["path"],["points_world_m_f32","faces"]); row_full=np.asarray(cross[role+"_source_row_to_joint_node"],np.int64)
        points_full=np.asarray(surf["points_world_m_f32"],np.float32);faces_full=np.asarray(surf["faces"],np.int64)
        used=np.unique(faces_full.ravel())
        active=np.asarray(cross[role+"_active_source_vertex_rows"],np.int64)
        inactive=np.asarray(cross[role+"_inactive_source_vertex_rows"],np.int64)
        need(np.array_equal(np.sort(np.concatenate((active,inactive))),np.arange(len(row_full))),role+" active/inactive map partition")
        need(np.array_equal(active,used),role+" renderer compact row order is not sorted face-active source rows")
        need(np.all(row_full[active]>=0) and np.all(row_full[inactive]==-1),"current active "+role+" sentinels")
        need(len(row_full)==len(points_full) and np.all(row_full[used]>=0),"current active "+role+" map mismatch")
        local=np.full(len(row_full),-1,np.int64);local[active]=np.arange(len(active),dtype=np.int64)
        faces=local[faces_full]
        need(np.all(faces>=0),role+" active typed faces do not map into compact renderer range")
        points=points_full[active];row=row_full[active]
        R[role]={"row":row,"points":points,"faces":faces,"used":np.unique(faces.ravel()),"source_rows":active}
    V={}
    for role in ("skin","fat"):
        n="Skin_current" if role=="skin" else "Fat_current";ent=state[n];need(sha(ent["path"])==ent["sha256"],n+" volume hash mismatch");z=npz(ent["path"],["points_world_m_f32","tetrahedra"])
        V[role]={"points":np.asarray(z["points_world_m_f32"],np.float32),"tets":np.asarray(z["tetrahedra"],np.int64)}
    node_map=handoff/"skin-fat-combined-node-map.csv.gz"
    local={r:np.full(len(V[r]["points"]),-1,np.int64) for r in V}; coords={}
    with gzip.open(node_map,"rt",newline="") as f:
        for x in csv.DictReader(f):
            joint=int(x["joint_node_id"])-1;pairs=[(x["source_partition"].lower(),int(x["source_tetgen_node_id"])-1)]
            peer=x.get("merge_peer_partition","").strip().lower()
            if peer:pairs.append((peer,int(x["merge_peer_tetgen_node_id"])-1))
            for role,row in pairs:
                if role not in local:continue
                need(0<=row<len(local[role]) and local[role][row] in (-1,joint),"bad 217 point crosswalk")
                local[role][row]=joint;point=V[role]["points"][row]
                need(joint not in coords or np.array_equal(coords[joint],point),"merged points differ")
                coords[joint]=point.copy()
    unmapped_unused={}
    for r in local:
        used_nodes=np.unique(V[r]["tets"].ravel())
        need(np.all(local[r][used_nodes]>=0),"217 node crosswalk omits a node referenced by child tets: "+r)
        unmapped_unused[r]=int(np.count_nonzero(local[r]<0))
    joint=np.empty((max(coords)+1,3),np.float32)
    for i,p in coords.items():joint[i]=p
    need(len(coords)==len(joint),"joint node ids have gaps")
    for role,d in R.items():
        for i in d["used"]:need(np.array_equal(d["points"][i],joint[d["row"][i]]),role+" source points differ from current FEM nodes")
    ncell=sum(len(v["tets"]) for v in V.values());expected=np.full((ncell,4),-1,np.int64);byrole=np.full(ncell,"",dtype="U4")
    with gzip.open(handoff/"skin-fat-combined-element-map.csv.gz","rt",newline="") as f:
        for x in csv.DictReader(f):
            i=int(x["joint_element_id"])-1;role=x["source_partition"].lower();child=int(x["source_tetgen_element_id"])-1
            need(role in V and 0<=i<ncell and byrole[i]=="","bad 217 element crosswalk")
            need(int(x["child_tet_row"])==i and 0<=child<len(V[role]["tets"]),"217 global/local element row identity mismatch")
            expected[i]=local[role][V[role]["tets"][child]];byrole[i]="Skin" if role=="skin" else "Fat"
    need(np.all(byrole!="") and np.all(expected>=0),"217 tet crosswalk incomplete")
    return {"R":R,"V":V,"joint":joint,"local":local,"expected":expected,"byrole":byrole,"cross":cross,"unmapped_unused_volume_point_rows":unmapped_unused}
def terminal_j(capture_data,mesh):
    begin,nodes,tets,_boundary=capture_data;nn=len(mesh["joint"]);nt=len(mesh["expected"])
    need(int(begin["nodes"])==nn and len(nodes)==nn,"terminal node count differs from 217")
    need(int(begin["tetrahedra"])==nt and len(tets)==nt,"terminal tet count differs from 217")
    p0=np.stack([nodes[i][0] for i in range(nn)]);p1=np.stack([nodes[i][1] for i in range(nn)])
    need(np.array_equal(p0,mesh["joint"]),"native initial nodes differ from exact 217 F32 input")
    logged=np.empty(nt,np.float64)
    for i in range(nt):
        corners,j=tets[i];need(np.array_equal(corners,mesh["expected"][i]),"native tet connectivity/order differs from 217 element map")
        logged[i]=j
    mins={};err=0.0
    for st in range(0,nt,65536):
        en=min(nt,st+65536);t=mesh["expected"][st:en]
        a=p0[t].astype(np.float64);b=p1[t].astype(np.float64)
        d0=np.einsum("ij,ij->i",a[:,1]-a[:,0],np.cross(a[:,2]-a[:,0],a[:,3]-a[:,0]))
        d1=np.einsum("ij,ij->i",b[:,1]-b[:,0],np.cross(b[:,2]-b[:,0],b[:,3]-b[:,0]))
        j=d1/d0;need(np.all(np.isfinite(j)) and np.all(j>0),"accepted motion contains nonpositive/nonfinite tet Jacobian")
        err=max(err,float(np.max(np.abs(j-logged[st:en])/np.maximum(1.0,np.abs(j)))))
        for role in ("Skin","Fat"):
            ix=np.flatnonzero(mesh["byrole"][st:en]==role)
            if len(ix):
                k=int(ix[np.argmin(j[ix])])+st;cur=mins.get(role)
                if cur is None or j[k-st]<cur["minimum_J"]:mins[role]={"minimum_J":float(j[k-st]),"joint_tet_row":k}
    need(err<2e-8,"replayed J differs from native terminal endpoint output")
    return {"minimum_J_by_material":mins,"max_native_J_replay_relative_error":err,"terminal_only":True,"step":int(begin["steps"]),"time_s":float(begin["endpoint_time_s"])}
def _oriented_triangle_key(tri):
    tri=tuple(map(int,tri)); inv=sum(tri[i]>tri[j] for i in range(3) for j in range(i+1,3))
    return tuple(sorted(tri)),(-1 if inv%2 else 1)
def _interface_signature(faces,mask):
    from collections import Counter
    selected=np.asarray(faces,np.int64)[np.asarray(mask,bool)]
    need(len(selected)>0,"shared interface has no faces")
    signs={};counts=Counter()
    for tri in selected:
        key,sign=_oriented_triangle_key(tri);counts[key]+=1;signs[key]=sign
    need(all(v==1 for v in counts.values()),"duplicate interface face identity")
    return counts,signs
def paired_interface_report(skin_faces,fat_faces,skin_mask,fat_mask):
    sc,ss=_interface_signature(skin_faces,skin_mask);fc,fs=_interface_signature(fat_faces,fat_mask)
    need(sc==fc,"Skin/Fat I interface face sets differ")
    same=sum(ss[k]==fs[k] for k in sc);opposite=sum(ss[k]==-fs[k] for k in sc)
    need(opposite==len(sc) and same==0,"Skin/Fat I interface winding is not exactly opposite")
    return {"status":"PASS_EXACT_SHARED_I_INTERFACE_OPPOSITE_WINDING","faces":len(sc),"same_winding_faces":same,"opposite_winding_faces":opposite,"coincident_interface_faces_excluded_from_external_intersection_count":len(sc)}
def remap_shell_to_joint(shell_points,shell_faces,shell_to_joint,joint_points):
    shell_points=np.asarray(shell_points,np.float32);shell_faces=np.asarray(shell_faces,np.int64);mp=np.asarray(shell_to_joint,np.int64);joint_points=np.asarray(joint_points,np.float32)
    need(mp.shape==(len(shell_points),),"shell point-to-joint map shape mismatch")
    used=np.unique(shell_faces.ravel());need(np.all((mp[used]>=0)&(mp[used]<len(joint_points))),"face-used shell point lacks accepted joint binding")
    jf=mp[shell_faces]
    need(np.array_equal(shell_points[used],joint_points[mp[used]]),"input shell F32 coordinates differ from exact joint reference")
    return jf

def load_native_orientation_contract(checkpoint,shell_receipt):
    need(sha(ORIENTATION_AUDIT)==ORIENTATION_AUDIT_SHA,"1022 signed orientation audit pin mismatch")
    need(sha(ORIENTATION_OWNER)==ORIENTATION_OWNER_SHA,"1022 signed orientation validator owner pin mismatch")
    audit=json.loads(ORIENTATION_AUDIT.read_text())
    need(audit.get("schema")=="numi.h706.1022-role-surface-orientation-audit.v1" and audit.get("status")=="PASS_EXACT_1022_SIGNED_ROLE_CONVENTIONS","1022 signed orientation audit did not pass")
    need(audit.get("physical_admission") is False and audit.get("native_steps")==0,"1022 orientation audit overclaims physical/native evidence")
    shellent=checkpoint.get("paired_shell_receipt",{})
    need(audit.get("receipt")==shellent,"1022 signed orientation audit receipt pin differs from motion checkpoint")
    need(audit.get("receipt")==shell_receipt["receipt_pin"],"1022 signed orientation audit does not bind the loaded shell receipt")
    sign={"O":1,"I":-1,"D":-1}
    need(audit.get("source_to_material_outward_sign")==sign,"1022 source-to-rendered outward sign contract mismatch")
    expected_roles={"O":{"same":152341,"reverse":0,"unmatched":0},"I":{"same":0,"reverse":236389,"unmatched":0},"D":{"same":0,"reverse":207901,"unmatched":0}}
    need(set(audit.get("roles",{}))==set(sign),"1022 orientation audit role roster mismatch")
    for r in ("O","I","D"):
        role=audit["roles"][r];asset=role["asset"];pin=checkpoint["source_state"]
        # Match the current state’s typed role pin, not merely the old audit’s claimed path.
        state=json.loads(Path(pin["path"]).read_text());current=state["source_surfaces"][r]
        need(asset["path"]==current["path"] and asset["sha256"]==current["sha256"] and sha(current["path"])==current["sha256"],"1022 orientation audit typed source pin mismatch: "+r)
        need(role.get("source_triangle_row_order_exact") is True and role.get("source_face_rows_identity") is True,"1022 orientation audit does not preserve typed face row identity: "+r)
        lanes=role.get("identity_lanes_equal",{});need(lanes and all(v is True for v in lanes.values()),"1022 orientation audit identity lanes fail: "+r)
        need(role.get("orientation_counts")==expected_roles[r] and role.get("shell_face_rows")==asset.get("faces"),"1022 orientation audit per-face sign counts mismatch: "+r)
        need(role.get("orientation_counts",{}).get("unmatched")==0,"1022 orientation audit has unmatched face: "+r)
    for key,filekey in (("skin","skin-shell.npz"),("fat","fat-shell.npz")):
        ae=audit["shell_assets"][key];se=shell_receipt["outputs"][filekey]
        need(ae["path"]==se["path"] and ae["sha256"]==se["sha256"],"1022 orientation audit shell asset pin mismatch: "+key)
    need(audit.get("1022_orientation_contract")==shell_receipt["assembly"]["orientation"],"1022 orientation contract differs from paired shell receipt")
    return {"report_path":str(ORIENTATION_AUDIT),"report_sha256":ORIENTATION_AUDIT_SHA,"validator_path":str(ORIENTATION_OWNER),"validator_sha256":ORIENTATION_OWNER_SHA,
            "schema":audit["schema"],"status":audit["status"],"source_to_accepted_role_winding_sign":sign,"face_set":"exact oriented bijection by vertex IDs; exporter face-row permutation and cyclic corner rotation are allowed; only pinned role orientation sign may reverse winding","roles":audit["roles"]}
def _oriented_cyclic_key(tri):
    tri=tuple(map(int,tri))
    return min((tri,tri[1:]+tri[:1],tri[2:]+tri[:2]))
def role_face_bijection(source_faces,accepted_faces,sign):
    source=np.asarray(source_faces,np.int64);actual=np.asarray(accepted_faces,np.int64)
    base={"face_count_source":int(len(source)) if source.ndim else 0,"face_count_accepted":int(len(actual)) if actual.ndim else 0,"source_to_accepted_role_winding_sign":int(sign)}
    if sign not in (-1,1) or source.ndim!=2 or actual.ndim!=2 or source.shape[1:]!=(3,) or actual.shape[1:]!=(3,):
        return {**base,"status":"FAIL_INVALID_FACE_SHAPE_OR_SIGN","matched_faces":0,"missing_source_faces":base["face_count_source"],"unexpected_or_duplicate_accepted_faces":base["face_count_accepted"]}
    expected=source if sign==1 else source[:,[0,2,1]]
    source_by_key={};source_unoriented=set();duplicate_source_oriented=0;duplicate_source_geometric=0
    for i,tri in enumerate(expected):
        itri=tuple(map(int,tri))
        if len(set(itri))!=3:
            duplicate_source_oriented+=1;continue
        uk=tuple(sorted(itri))
        if uk in source_unoriented:duplicate_source_geometric+=1
        else:source_unoriented.add(uk)
        k=_oriented_cyclic_key(itri)
        if k in source_by_key:duplicate_source_oriented+=1
        else:source_by_key[k]=i
    rowmap=np.full(len(actual),-1,np.int64);rotations=np.full(len(actual),-1,np.int8);corner_perms=np.full((len(actual),3),-1,np.int8);used=set();accepted_unoriented=set();duplicate_accepted=0;duplicate_accepted_geometric=0;unmatched=0
    for i,tri0 in enumerate(actual):
        tri=tuple(map(int,tri0))
        if len(set(tri))!=3:
            unmatched+=1;continue
        uk=tuple(sorted(tri))
        if uk in accepted_unoriented:
            duplicate_accepted_geometric+=1;continue
        accepted_unoriented.add(uk)
        j=source_by_key.get(_oriented_cyclic_key(tri))
        if j is None:
            unmatched+=1;continue
        if j in used:
            duplicate_accepted+=1;continue
        used.add(j);rowmap[i]=j;et=tuple(map(int,expected[j]));rotation=None
        for shift in range(3):
            perm=tuple(range(shift,3))+tuple(range(shift))
            if tuple(et[k0] for k0 in perm)==tri:
                rotation=shift
                baseperm=(0,1,2) if sign==1 else (0,2,1)
                corner_perms[i]=np.asarray(baseperm[shift:]+baseperm[:shift],np.int8)
                break
        if rotation is None:
            rowmap[i]=-1;used.remove(j);unmatched+=1
        else:rotations[i]=rotation
    missing=len(source)-len(used)
    passed=(len(source)==len(actual) and duplicate_source_oriented==0 and duplicate_source_geometric==0 and duplicate_accepted==0 and duplicate_accepted_geometric==0 and unmatched==0 and missing==0 and np.all(rowmap>=0) and np.all(rotations>=0))
    return {**base,"status":"PASS_EXACT_ORIENTED_FACE_BIJECTION" if passed else "FAIL_ORIENTED_FACE_BIJECTION","matched_faces":int(np.count_nonzero(rowmap>=0)),
            "missing_source_faces":int(missing),"unexpected_accepted_faces":int(unmatched),"duplicate_source_oriented_keys":int(duplicate_source_oriented),"duplicate_source_geometric_faces":int(duplicate_source_geometric),"duplicate_accepted_faces":int(duplicate_accepted),"duplicate_accepted_geometric_faces":int(duplicate_accepted_geometric),
            "cyclic_corner_rotation_counts":{str(k):int(np.count_nonzero(rotations==k)) for k in (0,1,2)},
            "accepted_face_row_to_typed_source_face_row_sha256":hashlib.sha256(np.asarray(rowmap,dtype='<i8').tobytes()).hexdigest(),
            "accepted_corner_permutation_to_typed_source_sha256":hashlib.sha256(np.asarray(corner_perms,dtype='i1').tobytes()).hexdigest(),
            "comparison":"exact vertex IDs and oriented face-set bijection; arbitrary face-row permutation and cyclic corner rotation allowed; reflection only follows pinned role sign"}

def load_paired_shells(checkpoint,state,mesh,hand):
    ent=checkpoint.get("paired_shell_receipt");need(isinstance(ent,dict),"checkpoint lacks paired-shell receipt pin")
    rp=Path(ent["path"]);need(sha(rp)==ent["sha256"],"paired-shell receipt pin mismatch")
    receipt=json.loads(rp.read_text());need(receipt.get("status")=="PASS_CURRENT_938V2_PAIRED_SHELLS_SOURCE_ONLY","paired-shell receipt status mismatch")
    need(receipt.get("physical_admission") is False,"paired-shell receipt overclaims physical admission")
    orient_contract=load_native_orientation_contract(checkpoint,{"receipt_pin":ent,"outputs":receipt["outputs"],"assembly":receipt["assembly"]})
    for role in ("O","I","D"):
        a=state["source_surfaces"][role];b=receipt["inputs"]["source_surfaces"][role]
        need(a["path"]==b["path"] and a["sha256"]==b["sha256"] and sha(a["path"])==a["sha256"],"paired-shell typed source pin differs from current checkpoint: "+role)
    need(receipt.get("shared_current_I",{}).get("both_shells_registered_same_I_faces") is True and receipt.get("shared_current_I",{}).get("exact_f32_coordinates") is True,"paired-shell I interface source receipt is incomplete")
    mapent=checkpoint["registered_active_source_crosswalk"];mappath=Path(mapent["path"]);need(sha(mappath)==mapent["sha256"],"paired-shell 217 crosswalk pin mismatch")
    mpz=npz(mappath,["skin_shell_point_row_to_joint_node","fat_shell_point_row_to_joint_node"])
    expected={"skin":{"markers":{"O":1,"I":2,"wall":3},"used_source_roles":("O","I"),"interface_marker":2,"wall_marker":3,"out":"skin-shell.npz","map":"skin_shell_point_row_to_joint_node"},
              "fat":{"markers":{"I":4,"D":5,"wall":6},"used_source_roles":("I","D"),"interface_marker":4,"wall_marker":6,"out":"fat-shell.npz","map":"fat_shell_point_row_to_joint_node"}}
    shells={};active_joint=np.unique(np.concatenate([mesh["R"][r]["row"] for r in ("O","I","D")])).astype(np.int64)
    active_set=np.zeros(len(mesh["joint"]),bool);active_set[active_joint]=True
    marker_roles=receipt["assembly"]["marker_roles"]
    need(marker_roles=={"skin_inner_I":2,"skin_outer_O":1,"skin_wall_OI":3,"fat_outer_I":4,"fat_inner_D":5,"fat_wall_ID":6},"paired-shell marker role contract mismatch")
    shell_files={}
    for mat,d in expected.items():
        oe=receipt["outputs"][d["out"]];op=Path(oe["path"]);need(sha(op)==oe["sha256"],mat+" shell NPZ pin mismatch")
        z=npz(op,["points_world_m","faces","face_marker"]+[f"registered_{r}_{suffix}" for r in d["used_source_roles"] for suffix in ("source_vertex_rows","shell_node_rows","source_faces","shell_face_rows")])
        points=np.asarray(z["points_world_m"],np.float32);faces=np.asarray(z["faces"],np.int64);marks=np.asarray(z["face_marker"],np.int64);jmap=np.asarray(mpz[d["map"]],np.int64)
        need(points.ndim==2 and points.shape[1]==3 and faces.ndim==2 and faces.shape[1]==3 and marks.shape==(len(faces),),mat+" shell array shapes invalid")
        need(jmap.shape==(len(points),),mat+" shell-to-joint map shape mismatch")
        used=np.unique(faces.ravel());need(np.all((jmap[used]>=0)&(jmap[used]<len(mesh["joint"]))),mat+" shell face point not mapped into current joint node range")
        need(np.array_equal(points[used],mesh["joint"][jmap[used]]),mat+" shell reference coordinates differ from pinned current FEM F32 nodes")
        mapped_used=np.unique(jmap[used]);need(np.all(active_set[mapped_used]),mat+" shell face vertex is not covered by an active O/I/D MRV vertex")
        counts={int(k):int(np.count_nonzero(marks==k)) for k in np.unique(marks)}
        expected_counts={int(d["markers"][r]):len(npz(state["source_surfaces"][r]["path"],["faces"])["faces"]) for r in d["used_source_roles"]}
        expected_wall=len(faces)-sum(expected_counts.values());need(expected_wall>0,mat+" shell wall/collar face count is not positive")
        expected_counts[int(d["wall_marker"])]=expected_wall
        need(counts==expected_counts,mat+" shell marker face inventory differs from registered O/I/D/wall source faces")
        jf=jmap[faces];topo=closed(jf);need(topo["closed_oriented"],mat+" shell is not a closed oriented complex after 217 joint remap")
        srfaces={}
        for r in d["used_source_roles"]:
            src=state["source_surfaces"][r];surf=npz(src["path"],["points_world_m_f32","faces"]);rpoints=np.asarray(surf["points_world_m_f32"],np.float32);rfaces=np.asarray(surf["faces"],np.int64)
            sv=np.asarray(z[f"registered_{r}_source_vertex_rows"],np.int64);sn=np.asarray(z[f"registered_{r}_shell_node_rows"],np.int64);sf=np.asarray(z[f"registered_{r}_source_faces"],np.int64);sfr=np.asarray(z[f"registered_{r}_shell_face_rows"],np.int64)
            need(sv.shape==sn.shape and len(np.unique(sv))==len(sv) and np.all((sv>=0)&(sv<len(rpoints))) and np.all((sn>=0)&(sn<len(points))),mat+"/"+r+" registered vertex maps invalid")
            rmap=np.asarray(mesh["cross"][r+"_source_row_to_joint_node"],np.int64)
            need(np.array_equal(sf,rfaces),mat+"/"+r+" registered source faces differ from checkpoint typed faces")
            need(len(sfr)==len(sf) and np.all((sfr>=0)&(sfr<len(faces))),mat+"/"+r+" registered shell face rows invalid")
            need(np.all(marks[sfr]==d["markers"][r]),mat+"/"+r+" shell face marker mismatch")
            node_j=jmap[sn]; mapped=rmap[sv]>=0
            need(np.all(node_j[mapped]==rmap[sv[mapped]]),mat+"/"+r+" typed source-to-shell joint identity mismatch")
            used_shell=np.zeros(len(points),bool);used_shell[used]=True
            need(not np.any((~mapped)&used_shell[sn]),mat+"/"+r+" shell face uses an inactive/unmapped typed source row")
            need(np.array_equal(points[sn[mapped]],mesh["joint"][node_j[mapped]]) and np.array_equal(points[sn[mapped]],rpoints[sv[mapped]]),mat+"/"+r+" registered typed source coordinate mismatch")
            src_j=rmap[sf];need(np.all(src_j>=0),mat+"/"+r+" typed face vertex lacks active joint map")
            actual_jf=jf[sfr]
            src_keys=np.sort(src_j,axis=1);actual_keys=np.sort(actual_jf,axis=1)
            src_keys=src_keys[np.lexsort(src_keys.T[::-1])];actual_keys=actual_keys[np.lexsort(actual_keys.T[::-1])]
            need(np.array_equal(src_keys,actual_keys),mat+"/"+r+" shell registered faces do not reproduce current typed source face identities")
            srfaces[r]={"source_faces":len(sf),"registered_shell_faces":len(sfr),"mapped_source_vertex_rows":int(np.count_nonzero(mapped)),"unused_unmapped_source_vertex_rows":int(np.count_nonzero(~mapped))}
        iface=marks==d["interface_marker"];wall=marks==d["wall_marker"]
        need(np.count_nonzero(wall)>0,mat+" retained wall/collar faces missing")
        shells[mat]={"points":points,"faces":faces,"face_marker":marks,"joint_map":jmap,"joint_faces":jf,"interface_mask":iface,"external_faces":jf[~iface],"topology":topo}
        shell_files[mat]={"path":str(op),"sha256":oe["sha256"],"vertices":len(points),"faces":len(faces),"face_used_vertices":len(used),"unused_or_unmapped_point_rows":int(len(points)-len(used)),"wall_faces":int(np.count_nonzero(wall)),"joint_topology":topo,"registered_typed_faces":srfaces}
    interface=paired_interface_report(shells["skin"]["joint_faces"],shells["fat"]["joint_faces"],shells["skin"]["interface_mask"],shells["fat"]["interface_mask"])
    external=np.concatenate((shells["skin"]["external_faces"],shells["fat"]["external_faces"]),axis=0);external_topo=closed(external);need(external_topo["closed_oriented"],"paired external boundary without shared I interface is not closed/oriented")
    return {"receipt_path":str(rp),"receipt_sha256":ent["sha256"],"crosswalk_path":str(mappath),"crosswalk_sha256":mapent["sha256"],"orientation_contract":orient_contract,"shells":shells,
            "report":{"status":"PASS_INPUT_SHELL_TO_JOINT_COVERAGE","shells":shell_files,"shared_I_interface":interface,"combined_external_boundary_topology":external_topo,
                      "all_face_used_shell_vertices_map_to_active_OID_MRV_vertices":True,"shell_reference_F32_matches_joint_reference_F32":True,"native_role_orientation_contract":orient_contract,"heavy_triangle_predicates_run":False,"native_motion_run":False,"physical_admission":False}}
def accepted_joint_positions(mesh,role_geometry,required_joint_ids,step):
    n=len(mesh["joint"]);out=mesh["joint"].copy();assigned=np.zeros(n,bool);allids=[];allpts=[]
    for r in ("O","I","D"):
        g=role_geometry[r];ids=np.asarray(g["row"],np.int64);pts=np.asarray(g["points"],np.float32)
        need(len(ids)==len(pts) and np.all((ids>=0)&(ids<n)),r+" accepted MRV point map invalid")
        allids.append(ids);allpts.append(pts)
    ids=np.concatenate(allids);pts=np.concatenate(allpts);order=np.argsort(ids,kind="stable");sid=ids[order];spts=pts[order]
    starts=np.r_[0,np.flatnonzero(sid[1:]!=sid[:-1])+1];ends=np.r_[starts[1:],len(sid)]
    for a,b in zip(starts,ends):
        if b-a>1:need(np.all(spts[a:b]==spts[a]),"duplicate O/I/D accepted MRV joint rows disagree")
    uniq=sid[starts];values=spts[starts];out[uniq]=values;assigned[uniq]=True
    req=np.asarray(required_joint_ids,np.int64);need(np.all(assigned[req]),"accepted MRV frames do not cover every shell-boundary joint node")
    if step==0:need(np.array_equal(out[uniq],mesh["joint"][uniq]),"initial accepted MRV source positions differ from pinned 217 coordinates")
    return out

def paired_shell_frame_audit(shell_ctx,joint_points,bone_records):
    shells=shell_ctx["shells"];n=len(joint_points);ident=np.arange(n,dtype=np.int64);result={}
    for mat in ("skin","fat"):
        sh=shells[mat];need(np.all(np.isfinite(joint_points[np.unique(sh["joint_faces"].ravel())])),mat+" accepted shell point is nonfinite")
        rec=exact_records(joint_points,sh["joint_faces"],ident)
        selfhit=EXACT._audit_pair(rec,rec,same_surface=True)
        bonehit=EXACT._audit_pair(rec,bone_records,same_surface=False)
        result[mat]={"self_intersections":selfhit,"bone_intersections_including_retained_wall_faces":bonehit,"closed_oriented":closed(sh["joint_faces"])["closed_oriented"],"triangles":len(sh["joint_faces"])}
    ext=np.concatenate((shells["skin"]["external_faces"],shells["fat"]["external_faces"]),axis=0)
    extrec=exact_records(joint_points,ext,ident);external_hit=EXACT._audit_pair(extrec,extrec,same_surface=True)
    iface=paired_interface_report(shells["skin"]["joint_faces"],shells["fat"]["joint_faces"],shells["skin"]["interface_mask"],shells["fat"]["interface_mask"])
    required=np.unique(np.concatenate([shells[m]["joint_faces"].ravel() for m in ("skin","fat")]))
    skin_i=np.unique(shells["skin"]["joint_faces"][shells["skin"]["interface_mask"]].ravel())
    fat_i=np.unique(shells["fat"]["joint_faces"][shells["fat"]["interface_mask"]].ravel())
    need(np.array_equal(skin_i,fat_i) and np.all(np.isfinite(joint_points[skin_i])),"accepted shared I interface does not resolve to the same finite joint rows")
    return {"skin_and_fat_shells":result,"combined_external_boundary_self_intersections":external_hit,"shared_I_interface":iface,
            "all_shell_faces_including_walls_audited":True,"coverage":"ACCEPTED_MRV_OID_JOINTS"}

def input_preflight(checkpoint_path):
    ck=json.loads(Path(checkpoint_path).read_text())
    stent=ck["source_state"];statepath=Path(stent["path"]);need(sha(statepath)==stent["sha256"],"current source state pin mismatch")
    state=json.loads(statepath.read_text())
    hent=ck["217_handoff"];hpath=Path(hent["path"]);need(sha(hpath)==hent["sha256"],"217 receipt pin mismatch")
    receipt=json.loads(hpath.read_text())
    need(receipt.get("status")=="PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION","217 handoff receipt status mismatch")
    child=receipt.get("combined_skin_fat_child_mesh",{});need(child.get("status")=="COMBINED_POSITIVE_F32_CHILD_MESH_NOT_PHYSICAL_ADMISSION","217 child mesh status mismatch")
    mapent=ck["registered_active_source_crosswalk"];mappath=Path(mapent["path"]);need(sha(mappath)==mapent["sha256"],"217 source-map pin mismatch")
    mesh=build_mesh(state,hpath.parent)
    need(int(child["nodes"])==len(mesh["joint"]) and int(child["combined_tetrahedra"])==len(mesh["expected"]),"217 receipt node/cell counts differ from reconstructed source map")
    need(int(child["skin_tetrahedra"])==int(np.count_nonzero(mesh["byrole"]=="Skin")) and int(child["fat_tetrahedra"])==int(np.count_nonzero(mesh["byrole"]=="Fat")),"217 per-material cell counts differ from reconstructed element map")
    active={r:int(np.count_nonzero(mesh["R"][r]["row"]>=0)) for r in ("O","I","D")}
    for role in ("O","I","D"):
        need(active[role]==int(child["registered_surface_active_node_rows"][role]),"217 active source-row count differs: "+role)
    topo={r:closed(mesh["R"][r]["faces"]) for r in ("O","I","D")}
    shell_ctx=load_paired_shells(ck,state,mesh,hpath.parent)
    return {"status":"PASS_INPUT_MAP_AND_CLOSED_PAIRED_SHELL_COVERAGE_ONLY","state_sha256":stent["sha256"],"handoff_sha256":hent["sha256"],"crosswalk_sha256":mapent["sha256"],"nodes":len(mesh["joint"]),"cells":len(mesh["expected"]),"skin_cells":int(np.count_nonzero(mesh["byrole"]=="Skin")),"fat_cells":int(np.count_nonzero(mesh["byrole"]=="Fat")),"active_source_rows":active,"source_surface_vertices_full":{r:len(mesh["R"][r]["source_rows"])+int(np.count_nonzero(mesh["cross"][r+"_inactive_source_vertex_rows"])) for r in ("O","I","D")},"source_surface_vertices_rendered":{r:len(mesh["R"][r]["points"]) for r in ("O","I","D")},"typed_role_sheet_topology":{r:{**topo[r],"class":"CLOSED_ORIENTED" if topo[r]["closed_oriented"] else "OPEN_SOURCE_SHEET"} for r in topo},"unmapped_unused_volume_point_rows":mesh["unmapped_unused_volume_point_rows"],"paired_shell_input_coverage":shell_ctx["report"],"heavy_surface_predicates_run":False,"native_motion_run":False,"physical_admission":False}

def _motion_stats(base,current,predicted):
    b=np.asarray(base,np.float32);c=np.asarray(current,np.float32);q=np.asarray(predicted,np.float64)
    need(b.shape==c.shape and b.ndim==2 and b.shape[1]==3 and q.shape==b.shape,"motion metric point array shape mismatch")
    need(np.all(np.isfinite(b)) and np.all(np.isfinite(c)) and np.all(np.isfinite(q)),"motion metric contains nonfinite coordinates")
    raw=c.astype(np.float64)-b.astype(np.float64);res=c.astype(np.float64)-q
    raw_n=np.linalg.norm(raw,axis=1);res_n=np.linalg.norm(res,axis=1)
    ulp=np.maximum(np.abs(np.spacing(b)),np.abs(np.spacing(c))).astype(np.float64)
    cutoff=2.0*np.max(ulp,axis=1)
    def one(x):
        return {"maximum_m":float(np.max(x)),"mean_m":float(np.mean(x)),"rms_m":float(np.sqrt(np.mean(x*x))),
                "p95_m":float(np.percentile(x,95)),"p99_m":float(np.percentile(x,99)),"nonzero_exact_count":int(np.count_nonzero(x>0)),
                "points_above_2x_max_component_f32_ulp":int(np.count_nonzero(x>cutoff)),
                "fraction_above_2x_max_component_f32_ulp":float(np.count_nonzero(x>cutoff)/len(x)),
                "max_pointwise_2x_f32_ulp_cutoff_m":float(np.max(cutoff)),"p99_pointwise_2x_f32_ulp_cutoff_m":float(np.percentile(cutoff,99))}
    return {"point_count":len(b),"world_displacement_from_step0":one(raw_n),"residual_after_matched_rigid_body_transform":one(res_n),
            "f32_noise_rule":"Per point, require displacement norm > 2 times the largest coordinate ULP across step-0/current float32 coordinates; reported as a resolution filter only, not physical qualification."}
def _fit_rigid_body(x0,x1):
    x0=np.asarray(x0,np.float64);x1=np.asarray(x1,np.float64)
    need(x0.shape==x1.shape and x0.ndim==2 and x0.shape[1]==3 and len(x0)>=3,"matched rigid-body landmark arrays invalid")
    need(np.all(np.isfinite(x0)) and np.all(np.isfinite(x1)),"rigid-body landmarks contain nonfinite coordinates")
    c0=x0.mean(axis=0);c1=x1.mean(axis=0);a=x0-c0;b=x1-c1
    need(np.linalg.matrix_rank(a,tol=max(np.linalg.norm(a,ord=2),1.0)*1e-12)>=2,"matched rigid-body landmarks are collinear")
    u,_,vt=np.linalg.svd(a.T@b,full_matrices=False);fix=np.eye(3);fix[2,2]=1.0 if np.linalg.det(u@vt)>=0 else -1.0
    R=u@fix@vt;t=c1-c0@R;pred=x0@R+t;err=np.linalg.norm(x1-pred,axis=1)
    return R,t,{"landmark_count":len(x0),"fit_rms_m":float(np.sqrt(np.mean(err*err))),"fit_max_m":float(np.max(err)),"rotation_determinant":float(np.linalg.det(R))}
def _reference_body(body_ids):
    ids,counts=np.unique(np.asarray(body_ids,np.int64),return_counts=True);need(len(ids)>0,"bone body roster is empty")
    ix=int(np.argmax(counts));return int(ids[ix]),int(counts[ix])
def accepted_motion_measurement(step,terminal_step,mesh,role_geometry,accepted_joint,bones,body_ids,shell_ctx,baseline,frame_receipt):
    bones=np.asarray(bones,np.float32);body_ids=np.asarray(body_ids,np.int64)
    need(bones.ndim==2 and bones.shape[1]==3 and len(bones)==3*len(body_ids),"accepted bone motion roster shape mismatch")
    if baseline is None:
        need(step==0,"accepted motion baseline is not MRV step zero")
        ref_id,tri_count=_reference_body(body_ids)
        baseline={"role_points":{r:np.asarray(role_geometry[r]["points"],np.float32).copy() for r in ("O","I","D")},
                  "role_rows":{r:np.asarray(role_geometry[r]["row"],np.int64).copy() for r in ("O","I","D")},
                  "role_source_rows":{r:np.asarray(role_geometry[r]["source_rows"],np.int64).copy() for r in ("O","I","D")},
                  "joint_points":np.asarray(accepted_joint,np.float32).copy(),"bones":bones.copy(),"bone_body_ids":body_ids.copy(),
                  "reference_body_index":ref_id,"reference_body_triangle_count":tri_count}
        R=np.eye(3);t=np.zeros(3);fit={"landmark_count":int(3*tri_count),"fit_rms_m":0.0,"fit_max_m":0.0,"rotation_determinant":1.0}
        mode="REFERENCE_STEP_IDENTITY"
    else:
        need(step>0,"step zero was encountered more than once")
        need(np.array_equal(body_ids,baseline["bone_body_ids"]),"accepted bone body/triangle row roster changed across frames")
        ref_id=baseline["reference_body_index"];tri_count=baseline["reference_body_triangle_count"]
        sel=np.repeat(body_ids==ref_id,3);need(np.count_nonzero(sel)>=9,"matched reference body has too few bone vertices")
        R,t,fit=_fit_rigid_body(baseline["bones"][sel],bones[sel]);mode="MATCHED_ACCEPTED_BONE_BODY_KABSCH"
    roles={}
    for r in ("O","I","D"):
        g=role_geometry[r]
        if step>0:
            need(np.array_equal(np.asarray(g["row"],np.int64),baseline["role_rows"][r]) and np.array_equal(np.asarray(g["source_rows"],np.int64),baseline["role_source_rows"][r]),r+" MRV point identity/order changed across accepted frames")
        p0=baseline["role_points"][r];p1=np.asarray(g["points"],np.float32);pred=p0.astype(np.float64)@R+t
        roles[r]=_motion_stats(p0,p1,pred)
    materials={}
    for mat in ("skin","fat"):
        ids=np.unique(shell_ctx["shells"][mat]["joint_faces"].ravel());p0=baseline["joint_points"][ids];p1=np.asarray(accepted_joint,np.float32)[ids];pred=p0.astype(np.float64)@R+t
        materials[mat]=_motion_stats(p0,p1,pred)
    time_s=None
    for key in ("accepted_time_s","simulation_time_s","time_s","elapsed_s"):
        if key in frame_receipt:
            try:
                val=float(frame_receipt[key])
                if math.isfinite(val):time_s=val;break
            except (TypeError,ValueError):pass
    return {"accepted_step":int(step),"terminal_accepted_step":int(terminal_step),"step_fraction_of_observed_horizon":float(step/terminal_step) if terminal_step else 0.0,
            "frame_time_s_if_exported":time_s,"reference_step":0,"coordinate_frame":"world-space accepted MRV render vertices, mapped through pinned 217 joint-node IDs","measurement_status":"ACCEPTED_MRV_GEOMETRY_MEASURED_NOT_PHYSICAL_QUALIFICATION",
            "matched_body_rigid_transform":{"method":mode,"reference_body_index":ref_id,"reference_body_triangle_count":tri_count,
                "rotation_row_vector_convention":R.tolist(),"translation_m":t.tolist(),"matched_bone_fit":fit},
            "per_role_active_surface_motion":roles,"per_material_closed_shell_motion_including_walls":materials},baseline

def actual(a):
    ck=json.loads(a.checkpoint.read_text());stent=ck["source_state"];statepath=Path(stent["path"])
    need(sha(statepath)==stent["sha256"],"current source state pin mismatch");state=json.loads(statepath.read_text())
    hent=ck["217_handoff"];hpath=Path(hent["path"]);need(sha(hpath)==hent["sha256"],"217 receipt pin mismatch")
    receipt=json.loads(hpath.read_text());need(receipt.get("status")=="PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION","217 handoff receipt status mismatch")
    child_receipt=receipt.get("combined_skin_fat_child_mesh",{});need(child_receipt.get("status")=="COMBINED_POSITIVE_F32_CHILD_MESH_NOT_PHYSICAL_ADMISSION","217 child mesh status mismatch")
    mapent=ck["registered_active_source_crosswalk"];mappath=Path(mapent["path"]);need(sha(mappath)==mapent["sha256"],"217 source-map pin mismatch")
    hand=hpath.parent;mesh=build_mesh(state,hand);shell_ctx=load_paired_shells(ck,state,mesh,hand)
    child=receipt["combined_skin_fat_child_mesh"]
    need(int(child["nodes"])==len(mesh["joint"]) and int(child["combined_tetrahedra"])==len(mesh["expected"]),"217 receipt node/cell counts differ from reconstructed source map")
    need(int(child["skin_tetrahedra"])==int(np.count_nonzero(mesh["byrole"]=="Skin")) and int(child["fat_tetrahedra"])==int(np.count_nonzero(mesh["byrole"]=="Fat")),"217 receipt per-material cell counts differ from reconstructed element map")
    for role,key in (("O","O"),("I","I"),("D","D")):
        active_count=int(np.count_nonzero(mesh["R"][role]["row"]>=0))
        need(active_count==int(child["registered_surface_active_node_rows"][key]),"217 receipt active source-row count differs: "+role)
    install=native_install_summary(a.stdout,mesh);cap=capture(a.stdout);audit=compact_terminal_audit(a.stdout)
    full_dump=cap[0] is not None
    if full_dump:
        J=terminal_j(cap,mesh);terminal_step=int(cap[0]["steps"])
        need(terminal_step==audit["expected_steps"],"full endpoint and compact audit step counts differ")
        replay_min=min(v["minimum_J"] for v in J["minimum_J_by_material"].values())
        need(abs(replay_min-audit["minimum_J_global"])<=1e-5*max(1.0,abs(replay_min)),"independent terminal J minimum differs from native compact audit")
        terminal_j_record={"mode":"INDEPENDENT_FULL_NODE_AND_TET_REPLAY","minimum_J_by_material":J["minimum_J_by_material"],"max_native_J_replay_relative_error":J["max_native_J_replay_relative_error"],"global_minimum_J_native_audit":audit["minimum_J_global"]}
    else:
        J=None;terminal_step=audit["expected_steps"]
        terminal_j_record={"mode":"NATIVE_COMPACT_ALL_TET_AUDIT","minimum_J_global":audit["minimum_J_global"],"maximum_J_global":audit["maximum_J_global"],"minimum_J_by_material":"NOT_EMITTED_BY_COMPACT_AUDIT","independent_node_level_replay":False}
    obs=json.loads(a.observer.read_text()) if a.observer else None
    if obs:
        validate_motion_observer(obs)
        need(obs.get("terminal_fem_audit_verified") is True,"measurement observer did not record terminal FEM audit")
        need(int(obs.get("requested_accepted_steps",-1))==terminal_step,"native observer step count differs from terminal audit")
        need(obs.get("stdout_sha256")==sha(a.stdout),"native stdout differs from measurement observer pin")
        if a.stderr:
            need(obs.get("stderr_sha256")==sha(a.stderr),"native stderr differs from measurement observer pin")
    else:
        need(full_dump,"compact mode requires the native measurement observer receipt")
    if not full_dump and obs:
        need(a.stderr is not None and a.stderr.is_file(),"compact mode requires stderr log for observer pin")
        need(obs.get("full_fem_endpoint_state_dump_requested") is False,"compact mode observer unexpectedly claims full endpoint dump")
        route=obs.get("install_route_receipt",{});route_path=Path(route.get("path",""))
        need(route_path.is_file() and sha(route_path)==route.get("sha256"),"native install-motion route receipt is missing or stale")
        route_receipt=json.loads(route_path.read_text())
        validate_motion_route(obs,route,route_receipt,terminal_step,install["counts"])
    prep=cap[3]
    need(prep,"native log lacks accepted I/D render ranges")
    im=re.search(r"inner_vertex_range=\[(\d+),(\d+)\)",prep);dm=re.search(r"deep_fat_vertex_range=\[(\d+),(\d+)\)",prep)
    need(im and dm,"I/D accepted MRV ranges absent")
    spans={"I":(int(im.group(1)),int(im.group(2))),"D":(int(dm.group(1)),int(dm.group(2)))}
    geomdir=a.motion_dir/"accepted-geometry"
    available={}
    for pack0 in geomdir.glob("step-*.mrvpack"):
        m=re.fullmatch(r"step-(\d+)\.mrvpack",pack0.name)
        if not m:continue
        step0=int(m.group(1));need(step0 not in available,"duplicate accepted MRV frame ID")
        available[step0]=pack0
    steps=sorted(available)
    need(0 in available and terminal_step in available,"accepted MRV captures must include initial and terminal motion frames")
    need(2<=len(steps)<=8 and all(0<=x<=terminal_step for x in steps),"accepted MRV frame roster is empty, oversized, or outside motion")
    if obs:
        reported=obs.get("accepted_geometry_frames")
        need(isinstance(reported,dict) and {0,terminal_step}<={int(k) for k in reported},"observer does not pin both initial and terminal MRV frames")
        for key,val in reported.items():
            frame=int(key);need(frame in available,"observer pins an MRV frame absent from output directory")
            pp=Path(val["pack_path"]);rp=Path(val["receipt_path"])
            need(pp.resolve()==available[frame].resolve() and rp.resolve()==available[frame].with_name(f"step-{frame}.receipt.json").resolve(),"observer MRV frame path differs from output directory")
            need(val["pack_sha256"]==sha(pp) and val["receipt_sha256"]==sha(rp),"observer MRV frame hash mismatch")
    initial_nodes=np.stack([cap[1][i][0] for i in range(len(mesh["joint"]))]).astype(np.float32,copy=False) if full_dump else mesh["joint"]
    terminal_nodes=np.stack([cap[1][i][1] for i in range(len(mesh["joint"]))]).astype(np.float32,copy=False) if full_dump else None
    exports=a.out.parent/(a.out.stem+"-exports");need(not exports.exists(),"refusing export overwrite");exports.mkdir(parents=True)
    frames=[];motion_baseline=None
    for step in steps:
        pack=available[step];rp=pack.with_name(f"step-{step}.receipt.json")
        need(pack.is_file() and rp.is_file(),"accepted MRV frame receipt missing")
        rec=json.loads(rp.read_text());need(rec.get("accepted_step")==step and rec.get("physical_endpoint")=="accepted" and rec.get("surface_audit_endpoint")=="passed" and rec.get("pack_file_sha256")==sha(pack),"MRV frame receipt/hash invalid")
        need(rec.get("position_normal_tangent_source")=="accepted-state renderer mesh buffer copied on the same Metal command buffer","MRV geometry is not pinned to the same-command-buffer accepted render state")
        poses=rec.get("accepted_body_poses")
        need(isinstance(poses,list) and len(poses)==int(rec.get("accepted_body_pose_count",-1)) and len(poses)>0,"MRV accepted body-pose roster missing/incomplete")
        for pose in poses:
            pos=np.asarray(pose.get("position_m",[]),np.float64);quat=np.asarray(pose.get("quaternion_xyzw",[]),np.float64)
            need(pos.shape==(3,) and quat.shape==(4,) and np.all(np.isfinite(pos)) and np.all(np.isfinite(quat)),"invalid accepted MRV body pose")
            need(abs(float(np.linalg.norm(quat))-1.0)<2e-3,"accepted MRV body quaternion is not normalized")
        om=rec.get("skin_source_mapping",{});need(isinstance(om,dict) and om,"MRV O source mapping missing")
        spans["O"]=(int(om["pack_first_vertex"]),int(om["pack_first_vertex"])+int(om["source_vertex_count"]))
        need(spans["O"][1]-spans["O"][0]==len(mesh["R"]["O"]["points"]),"O MRV range differs from current source rows")
        stepdir=exports/f"step-{step}";stepdir.mkdir()
        rolegeom={};bone_csv=stepdir/"bones.csv"
        endpoint_nodes=(initial_nodes if step==0 else (terminal_nodes if step==terminal_step else None)) if full_dump else None
        for role in ("O","I","D"):
            lo,hi=spans[role];d=mesh["R"][role];need(hi-lo==len(d["points"]),role+" MRV range mismatch")
            prefix=stepdir/role;argv=[str(a.exporter),str(pack),str(lo),str(hi-lo),str(prefix)]
            if role=="O":argv.append(str(bone_csv))
            rr=subprocess.run(argv,text=True,capture_output=True)
            need(rr.returncode==0,"native accepted geometry exporter failed: "+rr.stderr.strip())
            points,active=node_export(str(prefix)+".nodes.csv");faces=face_export(str(prefix)+".faces.csv")
            used=np.zeros(len(points),bool);used[np.unique(d["faces"].ravel())]=True
            sign=shell_ctx["orientation_contract"]["source_to_accepted_role_winding_sign"][role]
            face_bijection=role_face_bijection(d["faces"],faces,sign)
            need(np.array_equal(active,used) and face_bijection["status"]=="PASS_EXACT_ORIENTED_FACE_BIJECTION",role+" MRV topology/active rows/1022-authenticated oriented face bijection differs from typed source")
            if endpoint_nodes is not None:
                joint_rows=d["row"][used]
                need(np.all(joint_rows>=0) and np.array_equal(points[used],endpoint_nodes[joint_rows]),role+" MRV positions differ from mapped initial/terminal accepted FEM nodes")
            cl=closed(faces);src_cl=closed(d["faces"])
            need(cl==src_cl,role+" accepted role-sheet edge/orientation signature differs from typed source baseline")
            rolegeom[role]={"points":points,"faces":faces,"row":d["row"],"source_rows":d["source_rows"],"topology":cl,"face_bijection":face_bijection,"topology_class":"CLOSED_ORIENTED" if cl["closed_oriented"] else "OPEN_SOURCE_SHEET_PRESERVED"}
        # Rebuild complete Skin/Fat shells, including wall/collar faces, from the accepted O/I/D MRV vertices.
        shell_required=np.unique(np.concatenate([shell_ctx["shells"][m]["joint_faces"].ravel() for m in ("skin","fat")]))
        accepted_joint=accepted_joint_positions(mesh,rolegeom,shell_required,step)
        # Exact self, O/I/D pair, and accepted-pose bone-surface predicates from the existing owner.
        records={r:exact_records(d["points"],d["faces"],d["row"]) for r,d in rolegeom.items()}
        surf={}
        for r in records:surf[r]=EXACT._audit_pair(records[r],records[r],same_surface=True)
        cross={}
        rr=list(records)
        for i,x in enumerate(rr):
            for y in rr[i+1:]:cross[x+":"+y]=EXACT._audit_pair(records[x],records[y],same_surface=False)
        bones,bfaces,bone_body_ids=parse_bones(bone_csv);pose_match=validate_bone_pose_membership(bone_body_ids,poses);brec=EXACT._records(tuple(tuple(EXACT.float32_point_lattice_key(p)) for p in bones),[tuple(x) for x in bfaces])
        accepted_motion,motion_baseline=accepted_motion_measurement(step,terminal_step,mesh,rolegeom,accepted_joint,bones,bone_body_ids,shell_ctx,motion_baseline,rec)
        bone={r:EXACT._audit_pair(records[r],brec,same_surface=False) for r in records}
        paired_shell_audit=paired_shell_frame_audit(shell_ctx,accepted_joint,brec)
        frames.append({"step":step,"pack_sha256":sha(pack),"receipt_sha256":sha(rp),"accepted_body_pose_count":len(poses),"accepted_body_state_sha256":rec.get("accepted_body_state_sha256"),"accepted_body_poses_sha256":hashlib.sha256(json.dumps(poses,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"bone_pose_match":pose_match,"role_sheet_topology":{r:{"signature":d["topology"],"class":d["topology_class"]} for r,d in rolegeom.items()},"role_face_winding_vs_typed_source":shell_ctx["orientation_contract"]["source_to_accepted_role_winding_sign"],"role_oriented_face_bijection":{r:d["face_bijection"] for r,d in rolegeom.items()},"fem_node_binding":"EXACT_INITIAL" if step==0 else ("EXACT_TERMINAL" if step==terminal_step else "MRV_ACCEPTED_FRAME_ONLY_NO_PER_STEP_FULL_NODE_DUMP"),"surface_closure":"PASS","self_intersections":surf,"inter_role_intersections":cross,"bone_surface_intersections":bone,"paired_material_shell_geometry":paired_shell_audit,"accepted_motion_from_step0":accepted_motion,"bone_triangles":int(len(bfaces))})
    clean=all(
        all(v["count"]==0 for v in x.get("self_intersections",{}).values()) and
        all(v["count"]==0 for v in x.get("inter_role_intersections",{}).values()) and
        all(v["count"]==0 for v in x.get("bone_surface_intersections",{}).values()) and
        all(v["self_intersections"]["count"]==0 and v["bone_intersections_including_retained_wall_faces"]["count"]==0 for v in x.get("paired_material_shell_geometry",{}).get("skin_and_fat_shells",{}).values()) and
        x.get("paired_material_shell_geometry",{}).get("combined_external_boundary_self_intersections",{}).get("count")==0 for x in frames)
    report={"schema":"numi.human.current-motion-geometry-verification.v8","status":"PASS" if clean else "FAIL_OR_INCOMPLETE",
        "motion_coverage":"ALL_ACCEPTED_STEP_IDS" if len(steps)==terminal_step+1 else "SAMPLED_ACCEPTED_MRV_FRAMES",
        "accepted_frame_ids":steps,"omitted_accepted_step_id_count":terminal_step+1-len(steps),
        "checkpoint_sha256":sha(a.checkpoint),"source_state_sha256":stent["sha256"],"217_receipt_sha256":hent["sha256"],"217_map_sha256":mapent["sha256"],"stdout_sha256":sha(a.stdout),"stderr_sha256":sha(a.stderr) if a.stderr else None,
        "frames":frames,"accepted_motion_measurement":{"reference_frame":0,"terminal_step":terminal_step,"accepted_step_horizon":terminal_step,"sampled_frame_ids":steps,"coordinate_frame":"world-space accepted MRV render vertices mapped through the pinned 217 joint-node crosswalk","per_role_and_closed_material_shell_displacements":"reported relative to accepted MRV step zero, with matched dominant bone-body rigid transform residuals and pointwise F32 ULP sensitivity cutoff","movement_is_qualification":False},"native_registered_volume_install":install,"terminal_cell_J":terminal_j_record,"native_terminal_audit":audit,
        "fem_endpoint_state_dump_available":full_dump,
        "geometry_coverage":{"exact_role_sheets":["O","I","D"],"role_sheet_faces_match_typed_source_oriented_face_set_after_authenticated_signed_orientation":True,
            "per_role_closure_is_compared_to_typed_source_baseline":True,
            "open_role_sheets_are_not_claimed_as_closed_material_boundaries":True,
            "unrendered_material_wall_or_collar_faces":"reconstructed by exact shell-point-to-217-joint mapping and included in closed Skin/Fat shell tests",
            "paired_shell_receipt_path":shell_ctx["receipt_path"],"paired_shell_receipt_sha256":shell_ctx["receipt_sha256"],
            "paired_shell_input_coverage":shell_ctx["report"],
            "shared_I_interface":"verified as exact same joint-ID triangles with opposite winding; deliberately excluded from Skin-vs-Fat external self-intersection set",
            "bone_surface_comparison":"accepted world-space frozen-instance bone triangles versus complete reconstructed closed Skin/Fat shells, including retained wall/collar faces"},
        "per_step_cell_J":"not present in MRV; only terminal J is available from native audit; intermediate MRV frames have surface geometry only",
        "physical_admission":False,"mass_mapping":ck.get("mass_mapping","NOT_RUN"),
        "verifier_source":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__))},
        "accepted_geometry_exporter":{"path":str(a.exporter),"sha256":sha(a.exporter)},
        "exact_crossing_owner":{"path":str(Path(EXACT.__file__).resolve()),"sha256":sha(EXACT.__file__)},
        "observer":{"path":str(a.observer) if a.observer else None,"sha256":sha(a.observer) if a.observer else None}}
    need(not a.out.exists(),"refusing to overwrite output report");a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report
def parse_bones(path):
    p=[];body_ids=[]
    with Path(path).open(newline="") as f:
        for x in csv.DictReader(f):
            body_ids.append(int(x["body_index"]))
            for c in range(3):p.append([struct.unpack("<f",struct.pack("<I",int(x[f"v{c}{a}_bits_u32"],16)))[0] for a in "xyz"])
    need(p and len(body_ids)*3==len(p),"bone export is empty/incomplete")
    p=np.asarray(p,np.float32);return p,np.arange(len(p),dtype=np.int64).reshape(-1,3),np.asarray(body_ids,np.int64)
def validate_bone_pose_membership(body_ids,poses):
    pose_ids={int(x["body_index"]) for x in poses}
    bone_ids=set(map(int,np.asarray(body_ids,np.int64)))
    need(bool(bone_ids) and bone_ids<=pose_ids,"accepted bone body index lacks same-frame accepted pose")
    return {"matched_bone_body_count":len(bone_ids),"accepted_pose_body_count":len(pose_ids),"all_bone_instances_have_accepted_pose":True}
def validate_motion_observer(obs):
    accepted = {"PASS_BOUNDED_ACCEPTED_STATE_TRACE_AND_MRV_CHECKS", "PASS_BOUNDED_DIRECT_O_TRACES_AND_MRV"}
    need(obs.get("status") in accepted and obs.get("native_runtime_executed") is True and obs.get("gpu_executed") is True,
         "native observer is not a completed accepted GPU run")

def validate_motion_route(obs, route, receipt, terminal_step, install_counts=None):
    pairs = {"PASS_BOUNDED_ACCEPTED_STATE_TRACE_AND_MRV_CHECKS": "PASS_INSTALL_AND_ACCEPTED_STATE_MOTION",
             "PASS_BOUNDED_DIRECT_O_TRACES_AND_MRV": "PASS_DIRECT_O_INSTALL_AND_ACCEPTED_STATE_MOTION"}
    expected = pairs.get(obs.get("status"))
    need(expected is not None and route.get("status") == expected and receipt.get("status") == expected,
         "native install-motion observer/route status contract mismatch")
    need(receipt.get("requested_accepted_steps") == terminal_step and receipt.get("observed_accepted_steps") == terminal_step
         and receipt.get("registered_volume_install_pass") is True, "route receipt step/install claims mismatch")
    if expected == "PASS_DIRECT_O_INSTALL_AND_ACCEPTED_STATE_MOTION":
        need(obs.get("schema") == "numi.human.current-native-direct-o-measurement-observer-receipt.v1"
             and receipt.get("schema") == "numi.human.current-native-direct-o-motion-receipt.v1",
             "direct-O receipt schema mismatch")
        need(receipt.get("legacy_detail_bridge_enabled") is False and receipt.get("native_runtime_executed") is True
             and receipt.get("gpu_executed") is True and receipt.get("native_exit_code") == 0,
             "direct-O route lacks successful native GPU execution or enables the legacy bridge")
        direct = receipt.get("direct_O")
        need(isinstance(direct, dict) and direct == obs.get("direct_O"), "direct-O observer/route readback mismatch")
        need(all(type(direct.get(k)) is int and direct[k] > 0 for k in ("direct_O_rows", "direct_O_volume_nodes"))
             and direct["direct_O_rows"] <= direct["direct_O_volume_nodes"], "direct-O rows are not a positive subset of volume nodes")
        if install_counts is not None:
            need(direct["direct_O_rows"] == install_counts["O_rows"] and direct["direct_O_volume_nodes"] == install_counts["volume_nodes"],
                 "direct-O readback counts differ from installed typed domains")

def selftest():
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);f=np.asarray([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int64)
    need(closed(f)["closed_oriented"],"closed tetra fixture failed")
    rec=exact_records(p,f,np.arange(4));need(EXACT._audit_pair(rec,rec,same_surface=True)["count"]==0,"self-free tetra fixture failed")
    a=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32);b=np.asarray([[.25,.25,-1],[.25,.25,1],[.75,.1,0]],np.float32)
    ra=EXACT._records(tuple(tuple(EXACT.float32_point_lattice_key(x)) for x in a),[(0,1,2)])
    rb=EXACT._records(tuple(tuple(EXACT.float32_point_lattice_key(x)) for x in b),[(0,1,2)])
    need(EXACT._audit_pair(ra,rb,same_surface=False)["count"]==1,"crossing negative control failed")
    p0=p.copy(); p1=p.copy(); p1[1,0]=1.5; p1[2,1]=1.5; p1[3,2]=1.5
    mini={"joint":p0,"expected":np.asarray([[0,1,2,3]],np.int64),"byrole":np.asarray(["Skin"],dtype="U4")}
    begin={"nodes":"4","tetrahedra":"1","steps":"3","endpoint_time_s":"0.3"}
    nodes={i:(p0[i],p1[i]) for i in range(4)}
    j=3.375
    tets={0:(np.asarray([0,1,2,3],np.int64),j)}
    jr=terminal_j((begin,nodes,tets,"boundary"),mini)
    need(abs(jr["minimum_J_by_material"]["Skin"]["minimum_J"]-j)<1e-14,"terminal J replay control failed")
    inverted={i:(p0[i],p1[i].copy()) for i in range(4)};inverted[3][1][2]*=-1
    rejected=False
    try:terminal_j((begin,inverted,tets,"boundary"),mini)
    except ValueError:rejected=True
    need(rejected,"inverted accepted-node control not rejected")
    pose=[{"body_index":4,"position_m":[0.,0.,0.],"quaternion_xyzw":[0.,0.,0.,1.]}]
    need(validate_bone_pose_membership([4,4],pose)["all_bone_instances_have_accepted_pose"],"same-frame bone pose match control failed")
    rejected=False
    try:validate_bone_pose_membership([4,5],pose)
    except ValueError:rejected=True
    need(rejected,"unmatched accepted bone pose control not rejected")
    with tempfile.TemporaryDirectory(prefix="current-motion-verify-") as td:
        log=Path(td)/"compact.stdout"
        log.write_text("resting_FEM_terminal_audit=passed expected_steps=2 initial_control=0 final_control=1 accepted_steps=2 adapter_passes=2 expected_adapter_passes=2 min_J=0.8 max_J=1.1 max_displacement_m=0.01 max_anchor_reaction_N=0.0 capture=terminal_only\n")
        audit=compact_terminal_audit(log)
        need(audit["expected_steps"]==2 and audit["minimum_J_global"]==0.8,"compact audit parse control failed")
        need(capture(log)[0] is None,"compact capture incorrectly required full node dump")
        install_log=Path(td)/"install.stdout"
        install_log.write_text("resting_skin_fat_whole_body_preparation=accepted shared_fem_nodes=4 shared_fem_tets=2 skin_tets=1 fat_tets=1\nwhole_body_skin_fat_registered_volume_install=PASS O_rows=3 I_rows=3 D_rows=3 volume_nodes=4 child_tets=2\n")
        m={"R":{r:{"row":np.asarray([0,1,2],np.int64)} for r in ("O","I","D")},"joint":np.zeros((4,3)),"expected":np.zeros((2,4),np.int64),"byrole":np.asarray(["Skin","Fat"],dtype="U4")}
        need(native_install_summary(install_log,m)["counts"]["child_tets"]==2,"native install count replay control failed")
        bad_install=Path(td)/"bad-install.stdout"
        bad_install.write_text("resting_skin_fat_whole_body_preparation=accepted shared_fem_nodes=4 shared_fem_tets=2 skin_tets=1 fat_tets=1\nwhole_body_skin_fat_registered_volume_install=PASS O_rows=3 I_rows=3 D_rows=3 volume_nodes=4 child_tets=3\n")
        rejected=False
        try:native_install_summary(bad_install,m)
        except ValueError:rejected=True
        need(rejected,"native install count mismatch control not rejected")
        negatives={
            "failed": "resting_FEM_terminal_audit=failed expected_steps=2 accepted_steps=2 min_J=0.8 max_J=1.1 max_displacement_m=0.01 capture=terminal_only\n",
            "step_mismatch": "resting_FEM_terminal_audit=passed expected_steps=2 accepted_steps=1 min_J=0.8 max_J=1.1 max_displacement_m=0.01 capture=terminal_only\n",
            "nan": "resting_FEM_terminal_audit=passed expected_steps=2 accepted_steps=2 min_J=NaN max_J=1.1 max_displacement_m=0.01 capture=terminal_only\n",
            "subthreshold": "resting_FEM_terminal_audit=passed expected_steps=2 accepted_steps=2 min_J=0.34 max_J=1.1 max_displacement_m=0.01 capture=terminal_only\n",
            "missing": "no terminal audit\n",
        }
        for name,contents in negatives.items():
            bad=Path(td)/(name+".stdout");bad.write_text(contents)
            rejected=False
            try:compact_terminal_audit(bad)
            except (ValueError,KeyError):rejected=True
            need(rejected,f"compact audit negative control not rejected: {name}")
    def tet_faces_local(points,tet):
        out=[];p=np.asarray(points,np.float32);tet=list(map(int,tet));center=p[tet].astype(np.float64).mean(axis=0)
        for omit in range(4):
            tri=[tet[k] for k in range(4) if k!=omit];a,b,c=tri;normal=np.cross(p[b].astype(np.float64)-p[a],p[c].astype(np.float64)-p[a]);fc=p[tri].astype(np.float64).mean(axis=0)
            if float(np.dot(normal,center-fc))>0:tri[1],tri[2]=tri[2],tri[1]
            out.append(tri)
        return np.asarray(out,np.int64)
    bip=np.asarray([[0,0,0],[1,0,0],[0,1,0],[.1,.1,-1],[.1,.1,1]],np.float32)
    sf=tet_faces_local(bip,[0,1,2,3]);ff=tet_faces_local(bip,[0,1,2,4]);sm=np.asarray([0,1,2,3]);fm=np.asarray([2,4,0,1]);inv=np.full(len(bip),-1,np.int64);inv[fm]=np.arange(len(fm));fflocal=inv[ff]
    sfj=remap_shell_to_joint(bip[sm],sf,sm,bip);ffj=remap_shell_to_joint(bip[fm],fflocal,fm,bip)
    si=np.all(np.sort(sfj,axis=1)==np.asarray([0,1,2]),axis=1);fi=np.all(np.sort(ffj,axis=1)==np.asarray([0,1,2]),axis=1)
    iface_test=paired_interface_report(sfj,ffj,si,fi);need(iface_test["faces"]==1 and iface_test["opposite_winding_faces"]==1,"reordered shared-interface fixture failed")
    ext=np.concatenate((sfj[~si],ffj[~fi]));need(closed(ext)["closed_oriented"],"paired external shell closure fixture failed")
    srec=exact_records(bip,sfj,np.arange(len(bip)));frec=exact_records(bip,ffj,np.arange(len(bip)))
    need(EXACT._audit_pair(srec,srec,same_surface=True)["count"]==0 and EXACT._audit_pair(frec,frec,same_surface=True)["count"]==0,"reordered closed shell self-audit fixture failed")
    need(EXACT._audit_pair(exact_records(bip,ext,np.arange(len(bip))),exact_records(bip,ext,np.arange(len(bip))),same_surface=True)["count"]==0,"shared-interface-stripped external shell fixture failed")
    miss=fm.copy();miss[0]=-1;rejected=False
    try:remap_shell_to_joint(bip[fm],fflocal,miss,bip)
    except ValueError:rejected=True
    need(rejected,"face-used shell point with missing joint map was accepted")
    samew=ffj.copy();samew[fi]=sfj[si];rejected=False
    try:paired_interface_report(sfj,samew,si,fi)
    except ValueError:rejected=True
    need(rejected,"same-winding I interface was accepted")
    orient_src=np.asarray([[0,1,2],[2,3,4],[5,6,7]],np.int64);orient_rev=orient_src[:,[0,2,1]]
    reordered=np.asarray([orient_src[2,[1,2,0]],orient_src[0,[2,0,1]],orient_src[1]],np.int64)
    ob=role_face_bijection(orient_src,reordered,1);need(ob["status"]=="PASS_EXACT_ORIENTED_FACE_BIJECTION" and ob["cyclic_corner_rotation_counts"]=={"0":1,"1":1,"2":1},"face row permutation/cyclic rotation control failed")
    reordered_rev=np.asarray([orient_rev[1,[1,2,0]],orient_rev[2,[2,0,1]],orient_rev[0]],np.int64)
    ib=role_face_bijection(orient_src,reordered_rev,-1);need(ib["status"]=="PASS_EXACT_ORIENTED_FACE_BIJECTION","authenticated I/D signed oriented-set control failed")
    wrong=reordered_rev.copy();wrong[1]=wrong[1,[0,2,1]];need(role_face_bijection(orient_src,wrong,-1)["status"]!="PASS_EXACT_ORIENTED_FACE_BIJECTION","one wrongly oriented face was accepted")
    duplicate=reordered.copy();duplicate[2]=duplicate[0];need(role_face_bijection(orient_src,duplicate,1)["status"]!="PASS_EXACT_ORIENTED_FACE_BIJECTION","duplicate/missing face bijection was accepted")
    opposite_duplicate=np.concatenate((orient_src,orient_src[:1,[0,2,1]]),axis=0);need(role_face_bijection(opposite_duplicate,opposite_duplicate,1)["status"]!="PASS_EXACT_ORIENTED_FACE_BIJECTION","opposite-winding duplicate geometric source face was accepted")
    x0=np.asarray([[-1.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]],np.float64)
    ang=.37;R0=np.asarray([[np.cos(ang),-np.sin(ang),0.],[np.sin(ang),np.cos(ang),0.],[0.,0.,1.]],np.float64);t0=np.asarray([.12,-.07,.03]);x1=x0@R0+t0
    R1,t1,fit1=_fit_rigid_body(x0,x1);need(np.max(np.abs(x0@R1+t1-x1))<1e-12 and fit1["rotation_determinant"]>0.999999,"matched body rigid fit control failed")
    rolebase=np.asarray([[.1,.2,.3],[.4,.2,.3],[.2,.6,.3],[.2,.3,.8]],np.float32);rolecur=np.asarray(rolebase.astype(np.float64)@R0+t0,np.float32);pure=_motion_stats(rolebase,rolecur,rolebase.astype(np.float64)@R0+t0)
    need(pure["world_displacement_from_step0"]["points_above_2x_max_component_f32_ulp"]>0 and pure["residual_after_matched_rigid_body_transform"]["maximum_m"]<2e-7,"rigid-following motion metric control failed")
    local=rolecur.copy();local[0,0]+=np.float32(.01);deformed=_motion_stats(rolebase,local,rolebase.astype(np.float64)@R0+t0)
    need(deformed["residual_after_matched_rigid_body_transform"]["maximum_m"]>.009,"deformation residual metric failed to detect local motion")
    unchanged=_motion_stats(rolebase,rolebase,rolebase.astype(np.float64));need(unchanged["world_displacement_from_step0"]["points_above_2x_max_component_f32_ulp"]==0,"F32 stationary resolution control failed")
    triids=np.asarray([7,7,7,7],np.int64);bone0=np.tile(x0[:3],(4,1)).astype(np.float32);bone1=np.asarray(bone0.astype(np.float64)@R0+t0,np.float32)
    testmesh={"joint":rolebase};testroles0={r:{"points":rolebase.copy(),"row":np.arange(4,dtype=np.int64),"source_rows":np.arange(4,dtype=np.int64)} for r in ("O","I","D")}
    testroles1={r:{"points":rolecur.copy(),"row":np.arange(4,dtype=np.int64),"source_rows":np.arange(4,dtype=np.int64)} for r in ("O","I","D")}
    test_shell={"shells":{"skin":{"joint_faces":np.asarray([[0,1,2]],np.int64)},"fat":{"joint_faces":np.asarray([[0,2,3]],np.int64)}}}
    m0,b0=accepted_motion_measurement(0,1,testmesh,testroles0,rolebase,bone0,triids,test_shell,None,{})
    m1,b1=accepted_motion_measurement(1,1,testmesh,testroles1,rolecur,bone1,triids,test_shell,b0,{})
    need(m1["per_role_active_surface_motion"]["O"]["world_displacement_from_step0"]["maximum_m"]>0.1 and m1["per_role_active_surface_motion"]["O"]["residual_after_matched_rigid_body_transform"]["maximum_m"]<2e-7,"end-to-end accepted-motion rigid-following control failed")
    moved=rolecur.copy();moved[0,0]+=np.float32(.01);testroles2={r:{"points":moved,"row":np.arange(4,dtype=np.int64),"source_rows":np.arange(4,dtype=np.int64)} for r in ("O","I","D")}
    m2,_=accepted_motion_measurement(2,2,testmesh,testroles2,moved,bone1,triids,test_shell,b0,{})
    need(m2["per_role_active_surface_motion"]["I"]["residual_after_matched_rigid_body_transform"]["maximum_m"]>.009,"end-to-end accepted-motion deformation control failed")
    print("current motion verifier controls PASS: exact self/crossing, terminal replay, compact audit negatives, bone-pose binding, authenticated signed oriented face bijection, row/cyclic reorder and one-face/duplicate rejection, matched-body motion fit, F32 resolution cutoff, end-to-end rigid/deformation metrics, reordered paired-shell joint mapping, closed external union, opposite shared-I winding, and missing-map/same-winding rejection")
def main():
    global ORIENTATION_AUDIT, ORIENTATION_AUDIT_SHA, ORIENTATION_OWNER, ORIENTATION_OWNER_SHA
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,default=H/"current-native-1024-root/integration-checkpoint.json")
    p.add_argument("--motion-dir",type=Path);p.add_argument("--stdout",type=Path);p.add_argument("--stderr",type=Path)
    p.add_argument("--observer",type=Path)
    p.add_argument("--exporter",type=Path,default=Path("/Users/numi/Work/skin-fat-native-integration-209/build-native/numi-human-accepted-skin-export"))
    p.add_argument("--out",type=Path)
    p.add_argument("--orientation-audit",type=Path)
    p.add_argument("--orientation-audit-sha256")
    p.add_argument("--orientation-owner",type=Path)
    p.add_argument("--orientation-owner-sha256")
    p.add_argument("--self-test",action="store_true");p.add_argument("--input-preflight",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest();return
    need(a.orientation_audit and a.orientation_audit_sha256 and a.orientation_owner and a.orientation_owner_sha256,
         "v16 requires explicit --orientation-audit PATH SHA and --orientation-owner PATH SHA pins")
    ORIENTATION_AUDIT=a.orientation_audit.resolve()
    ORIENTATION_AUDIT_SHA=a.orientation_audit_sha256
    ORIENTATION_OWNER=a.orientation_owner.resolve()
    ORIENTATION_OWNER_SHA=a.orientation_owner_sha256
    if a.input_preflight:
        need(a.out,"--out is required for --input-preflight")
        result=input_preflight(a.checkpoint);need(not a.out.exists(),"refusing to overwrite preflight report")
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        print(json.dumps({"status":result["status"],"out":str(a.out),"nodes":result["nodes"],"cells":result["cells"],"heavy_surface_predicates_run":False},sort_keys=True));return
    need(a.motion_dir and a.stdout and a.out,"--motion-dir --stdout --out are required")
    r=actual(a);print(json.dumps({"status":r["status"],"out":str(a.out),"terminal_J":r["terminal_cell_J"]["minimum_J_by_material"]},sort_keys=True))
if __name__=="__main__":main()
