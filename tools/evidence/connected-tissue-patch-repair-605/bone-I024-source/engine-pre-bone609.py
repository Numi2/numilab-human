#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
DEFAULT_G=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype"
REMESH=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py"
BOUNDARY=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/remesh_d_witness_stars_rerun022.py"
ANNULUS=DEFAULT_G/"pair-geometry-013-connected-remesh/annulus_helper.py"
BRIDGE=P/"general-annulus-repair-607/annulus_bridge.py"
NEW_ID_BASE=62010000

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.array(z[k],copy=True) for k in z.files}

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def edge_map(faces):
    out=collections.defaultdict(list)
    for r,(a,b,c) in enumerate(faces):
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            out[tuple(sorted((u,v)))].append(int(r))
    return out

def face_components(rows,em):
    todo=set(map(int,rows)); comps=[]
    while todo:
        stack=[min(todo)]; comp=set()
        while stack:
            r=stack.pop()
            if r in comp: continue
            comp.add(r);todo.discard(r)
            a,b,c=map(int, CURRENT_FACES[r])
            for u,v in ((a,b),(b,c),(c,a)):
                stack.extend(set(em[tuple(sorted((u,v)))]) & todo)
        comps.append(comp)
    return comps

def build_plan(faces,pairs,boundary_mod):
    global CURRENT_FACES
    CURRENT_FACES=faces
    em=edge_map(faces);seed=set(map(int,pairs.reshape(-1)))
    # Connected components use both exact intersection-pair links and shared edges.
    parent={r:r for r in seed}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for a,b in pairs: union(int(a),int(b))
    seed_edge_first={}
    for r in seed:
        a,b,c=map(int,faces[r])
        for u,v in ((a,b),(b,c),(c,a)):
            e=tuple(sorted((u,v)))
            if e in seed_edge_first: union(r,seed_edge_first[e])
            else: seed_edge_first[e]=r
    seed_groups=collections.defaultdict(set)
    for r in seed: seed_groups[find(r)].add(r)
    collars=[]
    for group in seed_groups.values():
        col=set(group)
        for r in group:
            a,b,c=map(int,faces[r])
            for u,v in ((a,b),(b,c),(c,a)): col.update(em[tuple(sorted((u,v)))])
        collars.append(col)
    # Merge collared repair regions that overlap by any source face.
    cp=list(range(len(collars)))
    def cf(x):
        while cp[x]!=x: cp[x]=cp[cp[x]];x=cp[x]
        return x
    def cu(a,b):
        a=cf(a);b=cf(b)
        if a!=b:cp[max(a,b)]=min(a,b)
    owner={}
    for i,col in enumerate(collars):
        for r in col:
            if r in owner:cu(i,owner[r])
            else:owner[r]=i
    merged=collections.defaultdict(set)
    for i,col in enumerate(collars):merged[cf(i)].update(col)
    # Stable transaction IDs are sorted by minimum candidate face row.
    transactions=[]
    for rows in sorted(merged.values(),key=lambda q:min(q)):
        # Split each transaction into edge-connected surface patch pieces.
        pieces=[]
        for comp in face_components(rows,em):
            local_edge_counts=collections.Counter()
            for r in comp:
                a,b,c=map(int,faces[r])
                for u,v in ((a,b),(b,c),(c,a)):local_edge_counts[tuple(sorted((u,v)))]+=1
            bad_internal=sum(n>2 for n in local_edge_counts.values())
            try:
                loops,bd=boundary_mod.boundary_cycles(faces,comp)
                verts=set(map(int,faces[np.asarray(sorted(comp),np.int64)].reshape(-1)))
                undirected=set()
                for rr in comp:
                    aa,bb,cc=map(int,faces[rr])
                    undirected.update((tuple(sorted((aa,bb))),tuple(sorted((bb,cc))),tuple(sorted((cc,aa)))))
                chi=len(verts)-len(undirected)+len(comp)
                if bad_internal: kind="branching"
                elif len(loops)==1 and chi==1: kind="disk"
                elif len(loops)==2 and chi==0: kind="annulus"
                else: kind="closed_or_complex"
                err=""
            except Exception as e:
                loops=[];bd=[];kind="branching";err=str(e)
            pieces.append({"face_rows":sorted(map(int,comp)),
                           "boundary_loops":[list(map(int,x)) for x in loops],
                           "kind":kind,"error":err,"nonmanifold_internal_edge_count":int(bad_internal),
                           "euler_characteristic":int(chi) if not err else None,
                           "seed_face_rows":sorted(map(int,set(comp)&seed))})
        seed_rows=sorted(set().union(*(set(x["seed_face_rows"]) for x in pieces)))
        pair_indices=[i for i,(a,b) in enumerate(pairs) if int(a) in seed_rows or int(b) in seed_rows]
        if not pair_indices: raise ValueError("transaction lost exact-pair witness links")
        transactions.append({"transaction_id":len(transactions),"face_rows":sorted(map(int,rows)),
                             "seed_face_rows":seed_rows,"pair_indices":pair_indices,
                             "pair_count":len(pair_indices),"pieces":pieces})
    # Every exact pair must be contained in exactly one transaction.
    face_to_tx={}
    for tx in transactions:
        for r in tx["face_rows"]: face_to_tx[r]=tx["transaction_id"]
    for i,(a,b) in enumerate(pairs):
        if face_to_tx.get(int(a))!=face_to_tx.get(int(b)):
            raise ValueError(f"pair {i} crosses transaction partition")
    # Large collision components first; stable IDs remain unchanged.
    priority=sorted(transactions,key=lambda t:(-t["pair_count"],-len(t["seed_face_rows"]),-len(t["face_rows"]),t["transaction_id"]))
    priority_rank={t["transaction_id"]:i for i,t in enumerate(priority)}
    for tx in transactions: tx["priority_rank"]=priority_rank[tx["transaction_id"]]
    return transactions

def save_plan(plan_path,base_path,self_path,transactions):
    obj={"schema":"candidate007-pair-linked-connected-remesh-plan.v1",
         "candidate_path":str(base_path),"candidate_sha256":sha(base_path),
         "exact_self_report_path":str(self_path),"exact_self_report_sha256":sha(self_path),
         "transactions":transactions,
         "counts":{"transaction_count":len(transactions),
                   "pair_count":sum(t["pair_count"] for t in transactions),
                   "seed_face_count":len(set(r for t in transactions for r in t["seed_face_rows"])),
                   "patch_piece_count":sum(len(t["pieces"]) for t in transactions),
                   "disk_pieces":sum(p["kind"]=="disk" for t in transactions for p in t["pieces"]),
                   "annulus_pieces":sum(p["kind"]=="annulus" for t in transactions for p in t["pieces"]),
                   "exceptional_pieces":sum(p["kind"] not in ("disk","annulus") for t in transactions for p in t["pieces"])}}
    plan_path.parent.mkdir(parents=True,exist_ok=True)
    plan_path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    return obj

def csr_rows(offsets,flat):
    o=np.asarray(offsets,np.int64);v=np.asarray(flat,np.int64)
    return [v[int(o[i]):int(o[i+1])].tolist() for i in range(len(o)-1)]

def atomic_npz(path,**arrays):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

def lattice_bounds(lattice,vertex_ids):
    ids=list(map(int,vertex_ids))
    if not ids: raise ValueError("cannot build an AABB for an empty vertex set")
    return (tuple(min(int(lattice[v][k]) for v in ids) for k in range(3)),
            tuple(max(int(lattice[v][k]) for v in ids) for k in range(3)))

def record_aabb_overlaps(rec,lo,hi):
    return all(rec[2][k]>=lo[k] and hi[k]>=rec[1][k] for k in range(3))

def verify_lattice_aabb_scale(pred):
    pts=np.asarray([[-0.03,-0.08,0.10],[-0.02,-0.08,0.10],[-0.03,-0.07,0.10]],np.float32)
    lattice=[pred.float32_point_lattice_key(p) for p in pts]
    rec=pred._records(lattice,[(0,1,2)])[0]
    lo,hi=lattice_bounds(lattice,[0,1,2])
    world_lo=tuple(float(x) for x in pts.min(axis=0))
    world_hi=tuple(float(x) for x in pts.max(axis=0))
    if not record_aabb_overlaps(rec,lo,hi):
        raise ValueError("Float32 lattice AABB regression: lattice-space record was excluded")
    if record_aabb_overlaps(rec,world_lo,world_hi):
        raise ValueError("Float32 lattice AABB regression fixture did not distinguish world and lattice units")
    return {"status":"PASS","record_domain":"exact integer lattice at denominator 2^149",
            "filter_domain":"same exact integer lattice","old_world_bounds_would_miss":True}

def validate_tx227_topology_repair(tx,piece,rows,loops,chi,d,base_sha,self_sha,plan_sha):
    repair=tx.get("topology_repair")
    expected_plan_sha="b1131e56a98e800c11fcb6fee842dbb79b8fa0ce9a28d25a6486d3d6f47b705c"
    if int(tx.get("transaction_id",-1))!=227 or not isinstance(repair,dict):
        raise ValueError("unauthorized topology repair metadata")
    if plan_sha!=expected_plan_sha:
        raise ValueError("tx227 topology exception is not bound to approved plan003")
    if repair.get("schema")!="candidate007-local-constructed-handle-fill.v1" or repair.get("kind")!="constructed_handle_fill":
        raise ValueError("tx227 topology repair schema/kind mismatch")
    if repair.get("transaction_id")!=227 or repair.get("candidate_sha256")!=base_sha or repair.get("exact_self_report_sha256")!=self_sha:
        raise ValueError("tx227 topology repair candidate/exact pins mismatch")
    loop_hash=hashlib.sha256(json.dumps(loops,separators=(",",":")).encode()).hexdigest()
    rows_hash=hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest()
    if repair.get("boundary_loops_sha256")!=loop_hash or repair.get("selected_face_rows_sha256")!=rows_hash:
        raise ValueError("tx227 topology repair row/boundary witness mismatch")
    seeds=sorted(map(int,tx.get("seed_face_rows",[])))
    if seeds!=sorted(map(int,repair.get("exact_seed_face_rows",[]))) or seeds!=sorted(map(int,piece.get("seed_face_rows",[]))):
        raise ValueError("tx227 topology repair seed-face mismatch")
    if sorted(map(int,tx.get("pair_indices",[])))!=sorted(map(int,repair.get("exact_pair_indices",[]))):
        raise ValueError("tx227 topology repair exact-pair set mismatch")
    sources=sorted(set(map(int,np.asarray(d["face_source_original_id"])[np.asarray(rows,np.int64)])))
    if sources!=sorted(map(int,repair.get("source_original_ids",[]))):
        raise ValueError("tx227 topology repair source-origin set mismatch")
    before=repair.get("before",{});after=repair.get("after",{});delta=repair.get("delta",{})
    if (int(chi)!=-2 or int(before.get("euler_characteristic",-999))!=-2 or int(before.get("genus",-999))!=1
        or int(before.get("boundary_component_count",-999))!=2
        or sorted(map(int,before.get("boundary_loop_sizes",[])))!=sorted(len(x) for x in loops)):
        raise ValueError("tx227 input topology does not match the reviewed genus-one handle")
    if (piece.get("kind")!="annulus" or int(piece.get("euler_characteristic_after",-999))!=0
        or int(after.get("euler_characteristic",-999))!=0 or int(after.get("genus",-999))!=0
        or int(after.get("boundary_component_count",-999))!=2
        or sorted(map(int,after.get("boundary_loop_sizes",[])))!=sorted(len(x) for x in loops)
        or int(delta.get("euler_characteristic",-999))!=2 or int(delta.get("genus",999))!=-1):
        raise ValueError("tx227 output topology/declared delta mismatch")
    if (not repair.get("same_ordered_boundary_loops") or repair.get("coordinate_change") is not False
        or not repair.get("all_exact_seed_faces_retained_in_selected_region")
        or not repair.get("outside_face_rows_unchanged")):
        raise ValueError("tx227 topology repair preservation claims failed")
    prior=repair.get("prior_preserving_handle_attempt",{})
    if (int(prior.get("accepted_transaction_count",-1))!=0
        or prior.get("interpretation_sha256")!="f1dd3cca20e190dcb6aeff1c3c6d59ff7db5352ff65dcabbcd206e1ec166d501"):
        raise ValueError("tx227 prior-handle failure pin mismatch")
    return {"schema":repair["schema"],"plan_sha256":plan_sha,"input_euler":-2,"input_genus":1,
            "output_euler":0,"output_genus":0,"delta_chi":2,"delta_genus":-1,
            "source_original_ids":sources,"seed_face_rows":seeds,"pair_indices":sorted(map(int,tx["pair_indices"])),
            "boundary_loops_sha256":loop_hash,"prior_preserving_attempt_sha256":prior["interpretation_sha256"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--exact-report",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--expected-candidate-sha",default="")
    ap.add_argument("--plan",type=Path,default=None,help="prebuilt plan/selection JSON; if absent, build deterministic pair+edge+collar plan")
    ap.add_argument("--transactions",default="all-disks",help="all-disks, all-eligible, or comma-separated stable transaction IDs")
    ap.add_argument("--plan-only",action="store_true")
    ap.add_argument("--replay-operations",type=Path,default=None,help="revalidate saved compact operations against this pinned input")
    ap.add_argument("--max-seconds",type=float,default=180.0)
    ap.add_argument("--per-patch-seconds",type=float,default=1.0)
    ap.add_argument("--max-nodes",type=int,default=5000)
    ap.add_argument("--checkpoint-every",type=int,default=25)
    args=ap.parse_args()
    started=time.monotonic()
    base=Path(args.candidate); selfp=Path(args.exact_report); outdir=Path(args.out)
    if not base.is_file() or not selfp.is_file(): raise FileNotFoundError("candidate or exact report missing")
    base_sha=sha(base); self_sha=sha(selfp)
    if args.expected_candidate_sha and base_sha!=args.expected_candidate_sha: raise ValueError("candidate SHA does not match expected")
    d=load_npz(base)
    points32=np.asarray(d["points_world_m"],np.float32)
    faces=np.asarray(d["faces"],np.int64)
    witness=json.loads(selfp.read_text())
    if witness.get("candidate_sha256")!=base_sha: raise ValueError("exact report is not pinned to candidate")
    pairs=np.asarray(witness.get("triangle_pairs",[]),np.int64)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or np.any(pairs<0) or np.any(pairs>=len(faces)): raise ValueError("invalid exact witness pair rows")
    outdir.mkdir(parents=True,exist_ok=False)
    boundary_mod=loadmod("shared_boundary_helper_014",BOUNDARY)
    remesh_mod=loadmod("shared_collision_ear_014",REMESH)
    annulus_mod=loadmod("shared_annulus_014",ANNULUS)
    bridge_mod=loadmod("shared_annulus_bridge_607",BRIDGE)
    pred=remesh_mod.load_exact_owner()
    aabb_regression=verify_lattice_aabb_scale(pred)
    if args.plan:
        plan_obj=json.loads(Path(args.plan).read_text())
        if plan_obj.get("candidate_sha256")!=base_sha or plan_obj.get("exact_self_report_sha256")!=self_sha:
            raise ValueError("selection plan is not pinned to input candidate and exact report")
        transactions=plan_obj["transactions"]
    else:
        transactions=build_plan(faces,pairs,boundary_mod)
        plan_path=outdir/"selection-plan.json"
        plan_obj=save_plan(plan_path,base,selfp,transactions)
    if not args.plan:
        plan_path=outdir/"selection-plan.json"
    else:
        plan_path=Path(args.plan)
    if args.plan_only:
        print(json.dumps({"status":"PLAN_READY","plan":str(plan_path),"plan_sha256":sha(plan_path),"counts":plan_obj.get("counts",{}),"elapsed_s":time.monotonic()-started},sort_keys=True),flush=True)
        return
    replay_by_patch={}
    replay_sha=None
    replay_source_plan_sha=None
    replay_plan_matches_current=None
    if args.replay_operations is not None:
        replay_obj=json.loads(args.replay_operations.read_text())
        if replay_obj.get("candidate_sha256")!=base_sha or replay_obj.get("exact_report_sha256")!=self_sha:
            raise ValueError("saved operations are not pinned to this candidate and exact report")
        replay_sha=sha(args.replay_operations)
        replay_source_plan_sha=replay_obj.get("plan_sha256")
        replay_plan_matches_current=(replay_source_plan_sha==sha(plan_path))
        for saved_op in replay_obj.get("operations",[]):
            key=(int(saved_op["transaction_id"]),int(saved_op["patch_index"]))
            if key in replay_by_patch: raise ValueError("duplicate saved operation key: "+repr(key))
            replay_by_patch[key]=saved_op
    if args.transactions=="all-disks":
        wanted={int(t["transaction_id"]) for t in transactions if all(p["kind"]=="disk" for p in t["pieces"])}
    elif args.transactions=="all-eligible":
        wanted={int(t["transaction_id"]) for t in transactions if all(p["kind"] in ("disk","annulus") for p in t["pieces"])}
    else:
        wanted=set(map(int,args.transactions.split(",")))
    txmap={int(t["transaction_id"]):t for t in transactions}
    if not wanted or not wanted<=set(txmap): raise ValueError("transaction selection is empty or out of range")
    selected=[txmap[i] for i in sorted(wanted,key=lambda q:(txmap[q]["priority_rank"],q))]
    bad=[(t["transaction_id"],p["kind"]) for t in selected for p in t["pieces"] if p["kind"] not in ("disk","annulus")]
    if bad: raise ValueError("selected transactions include unexpanded exceptional patch: "+repr(bad[:10]))
    selected_rows=set().union(*(set(map(int,t["face_rows"])) for t in selected))
    if len(selected_rows)!=sum(len(t["face_rows"]) for t in selected): raise ValueError("selected transaction face rows overlap")
    # Full candidate exact records are made once; selected rows are excluded from the stable obstacle index.
    stable_mask=np.ones(len(faces),bool);stable_mask[np.asarray(sorted(selected_rows),np.int64)]=False
    stable_faces=faces[stable_mask]
    P64=points32.astype(np.float64)
    base_records,_=pred.float32_surface_records(P64,stable_faces,source_seam_groups=())
    base_index=pred._prepare_surface_aabb(base_records)
    selected_sorted=np.asarray(sorted(selected_rows),np.int64)
    selected_raw,_=pred.float32_surface_records(P64,faces[selected_sorted],source_seam_groups=())
    if len(selected_raw)!=len(selected_sorted): raise ValueError("selected-original record row-domain mismatch")
    selected_original_records={int(row):(rec[0],rec[1],rec[2],int(row),rec[4]) for row,rec in zip(selected_sorted,selected_raw)}
    stable_edges=set()
    for a,b,c in stable_faces:
        stable_edges.update((tuple(sorted((int(a),int(b)))),tuple(sorted((int(b),int(c)))),tuple(sorted((int(c),int(a))))))
    lattice=tuple(pred.float32_point_lattice_key(p) for p in points32)
    accepted=[]; failed=[]; ops=[];accepted_pairs=[];accepted_faces=[];removed_original_rows=set();stale_replay=[]
    em=edge_map(faces); deadline=time.monotonic()+args.max_seconds
    topology_repair_receipts=[]
    # Transactional processing: either every disconnected patch piece in a pair-linked transaction succeeds, or none is applied.
    for ti,tx in enumerate(selected):
        if time.monotonic()>=deadline:
            failed.append({"transaction_id":int(tx["transaction_id"]),"reason":"global_time_bound"})
            continue
        tx_id=int(tx["transaction_id"]);tx_faces=set(map(int,tx["face_rows"]))
        active_selected_rows=selected_rows-removed_original_rows-tx_faces
        active_external_edges=set(stable_edges)
        for r in active_selected_rows:
            a,b,c=map(int,faces[r])
            active_external_edges.update((tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))))
        for tri in accepted_faces:
            a,b,c=map(int,tri)
            active_external_edges.update((tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))))
        local_new=[];local_records=[];local_ops=[];tx_fail=None
        for pi,piece in enumerate(tx["pieces"]):
            rows=sorted(map(int,piece["face_rows"]));loops=[list(map(int,x)) for x in piece["boundary_loops"]]
            # Recheck the packet boundary against this exact candidate row domain.
            loops_now,bd=boundary_mod.boundary_cycles(faces,rows)
            if loops_now!=loops: tx_fail={"reason":"boundary_loop_mismatch","piece":pi};break
            local_edge_count=collections.Counter()
            for r in rows:
                a,b,c=map(int,faces[r])
                for u,v in ((a,b),(b,c),(c,a)): local_edge_count[tuple(sorted((u,v)))]+=1
            if any(v>2 for v in local_edge_count.values()):
                tx_fail={"reason":"nonmanifold_internal_patch_edge","piece":pi};break
            external_edges=set(active_external_edges)
            for tri in local_new:
                a,b,c=map(int,tri)
                external_edges.update((tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))))
            patch_verts=np.unique(faces[np.asarray(rows,np.int64)].reshape(-1))
            # Every prepared record, including previous replacements, is in
            # the exact 2^149 integer lattice. Keep broad-phase bounds in that
            # same domain; comparing lattice integers to world metres silently
            # drops positive-z replacement records.
            qlo,qhi=lattice_bounds(lattice,patch_verts)
            excluded_rows=removed_original_rows|tx_faces
            near_records=[]
            for r,rec in selected_original_records.items():
                if r not in excluded_rows and all(rec[2][k]>=qlo[k] and qhi[k]>=rec[1][k] for k in range(3)):
                    near_records.append((rec,tuple(map(int,faces[r]))))
            for rec,tri in accepted_pairs+local_records:
                if all(rec[2][k]>=qlo[k] and qhi[k]>=rec[1][k] for k in range(3)):
                    near_records.append((rec,tri))
            # Reserve disjoint predicate face-ID ranges per transaction/piece;
            # failed ear candidates consume IDs inside the helper as well.
            if tx_id<0 or tx_id>=1_000_000 or pi<0 or pi>=1000:
                raise ValueError("transaction/piece exceeds reserved exact face-ID domain")
            first_id=1_000_000_000_000_000+tx_id*1_000_000_000_000+pi*1_000_000_000
            try:
                verts=set(map(int,faces[np.asarray(rows,np.int64)].reshape(-1)))
                chi=len(verts)-len(local_edge_count)+len(rows)
                expected_chi=1 if piece["kind"]=="disk" else 0
                topology_receipt=None
                if chi!=expected_chi:
                    if tx_id!=227:
                        raise ValueError(f"patch Euler characteristic {chi} does not match {piece['kind']}")
                    topology_receipt=validate_tx227_topology_repair(tx,piece,rows,loops,chi,d,base_sha,self_sha,sha(plan_path))
                saved=replay_by_patch.get((tx_id,pi))
                if saved is not None and (sorted(map(int,saved["removed_face_rows"]))!=rows or saved["boundary_loops"]!=loops):
                    stale_replay.append({"transaction_id":tx_id,"patch_index":int(pi),
                                         "saved_rows_match":sorted(map(int,saved["removed_face_rows"]))==rows,
                                         "saved_boundary_matches":saved["boundary_loops"]==loops})
                    saved=None
                if saved is not None:
                    try:
                        tris=[tuple(map(int,x)) for x in saved["new_faces"]]
                        allowed_boundary_edges={tuple(sorted((int(loop[i]),int(loop[(i+1)%len(loop)]))))
                                                for loop in loops for i in range(len(loop))}
                        forbidden_edges={tuple(sorted(map(int,e))) for e in external_edges}-allowed_boundary_edges
                        for tri in tris:
                            for u,v in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                                if tuple(sorted((int(u),int(v)))) in forbidden_edges:
                                    raise ValueError("saved proposal uses forbidden interior mesh edge")
                        fresh=[];prior_pairs=list(near_records)
                        for j,tri in enumerate(tris):
                            if len(set(tri))!=3 or any(v<0 or v>=len(points32) for v in tri):
                                raise ValueError("saved operation has invalid face indices")
                            raw=pred._records(lattice,[tri])[0]
                            rec=(raw[0],raw[1],raw[2],first_id+j,raw[4])
                            ext=pred._audit_pair_prepared_first(base_index,(rec,),same_surface=True)
                            if int(ext["count"]): raise ValueError("saved proposal intersects stable original geometry: "+repr(ext["triangle_pairs"][:8]))
                            for other,other_tri in prior_pairs:
                                hit=pred._audit_record_pairs(((rec,other),),same_surface=True)
                                if int(hit["count"]): raise ValueError("saved proposal intersects retained/accepted geometry: "+repr(hit["triangle_pairs"][:8]))
                            fresh.append(rec);prior_pairs.append((rec,tri))
                        details={"method":"replayed and exactly revalidated compact proposal","source_operations_sha256":replay_sha,"saved_details":saved.get("details",{})}
                    except Exception as replay_error:
                        stale_replay.append({"transaction_id":tx_id,"patch_index":int(pi),
                                             "reason":"saved_proposal_failed_current_exact_revalidation",
                                             "error":repr(replay_error)})
                        saved=None
                if saved is None and piece["kind"]=="disk":
                    tris,recs,details=remesh_mod.collision_aware_ear_clip(
                        loops[0],P64,points32,external_edges,base_index,pred,lattice,
                        near_records,first_id,max_nodes=args.max_nodes,max_seconds=args.per_patch_seconds)
                    if len(tris)!=len(loops[0])-2: raise ValueError("disk triangulation face count mismatch")
                    if len(recs)<len(near_records): raise ValueError("disk helper returned fewer records than previous-pair prefix")
                    fresh=recs[len(near_records):]
                elif saved is None:
                    try:
                        tris,fresh,details=annulus_mod.collision_aware_annulus(
                            loops,P64,points32,external_edges,base_index,pred,lattice,
                            near_records,first_id,max_nodes=args.max_nodes,max_seconds=args.per_patch_seconds,max_offsets=len(loops[1]))
                    except (ValueError,TimeoutError) as zipper_error:
                        tris,fresh,details=bridge_mod.collision_aware_bridged_annulus(
                            loops,P64,points32,external_edges,base_index,pred,lattice,
                            near_records,first_id,ear_clip=remesh_mod.collision_aware_ear_clip,
                            max_nodes=args.max_nodes,max_seconds=4.0,max_bridges=16)
                        details["zipper_failure"] = repr(zipper_error)
                        details["fallback_after_all_loop_offsets"] = len(loops[1])
                if len(fresh)!=len(tris): raise ValueError("remesher fresh-record/face count mismatch")
                if any(int(rec[3])>=first_id+1_000_000_000 for rec in fresh): raise ValueError("remesher exhausted reserved face-ID range")
                old_boundary=collections.Counter(tuple(map(int,e)) for e in bd)
                if any(v!=1 for v in old_boundary.values()): raise ValueError("input patch boundary is not a simple directed cycle set")
                tri_tuples=[tuple(map(int,x)) for x in tris]
                canonical_faces=[tuple(sorted(x)) for x in tri_tuples]
                if len(set(canonical_faces))!=len(canonical_faces): raise ValueError("replacement patch contains duplicate unoriented triangles")
                new_dir=collections.Counter()
                edge_faces=collections.defaultdict(list)
                for fi,(a,b,c) in enumerate(tri_tuples):
                    for u,v in ((a,b),(b,c),(c,a)):
                        new_dir[(int(u),int(v))]+=1
                        edge_faces[tuple(sorted((int(u),int(v))))].append(fi)
                new_boundary=collections.Counter()
                adjacency=[set() for _ in tri_tuples]
                for edge,faces_on_edge in edge_faces.items():
                    u,v=edge;forward=new_dir[(u,v)];reverse=new_dir[(v,u)];total=forward+reverse
                    if len(faces_on_edge)==1 and total==1 and (forward,reverse) in ((1,0),(0,1)):
                        directed=(u,v) if forward else (v,u)
                        new_boundary[directed]=1
                    elif len(faces_on_edge)==2 and total==2 and forward==1 and reverse==1:
                        a,b=faces_on_edge;adjacency[a].add(b);adjacency[b].add(a)
                    else: raise ValueError("replacement patch has duplicate directed edge or invalid edge incidence")
                if new_boundary!=old_boundary: raise ValueError("directed patch boundary changed")
                if tri_tuples:
                    reached={0};stack=[0]
                    while stack:
                        q=stack.pop()
                        for other in adjacency[q]:
                            if other not in reached: reached.add(other);stack.append(other)
                    if len(reached)!=len(tri_tuples): raise ValueError("replacement patch has multiple edge-connected components")
                out_verts=set(v for tri in tri_tuples for v in tri)
                out_edges=set(edge_faces)
                out_chi=len(out_verts)-len(out_edges)+len(tri_tuples)
                if out_chi!=expected_chi: raise ValueError(f"replacement Euler characteristic {out_chi} does not match {piece['kind']}")
                local_new.extend(tri_tuples);local_records.extend((rec,tri) for rec,tri in zip(fresh,tri_tuples))
                local_ops.append({"transaction_id":tx_id,"patch_index":int(pi),"removed_face_rows":rows,"boundary_loops":loops,
                                  "new_faces":[list(map(int,x)) for x in tris],"details":details,
                                  "parent_face_rows":rows,"topology_repair":topology_receipt})
                if topology_receipt is not None: topology_repair_receipts.append({"transaction_id":tx_id,**topology_receipt})
            except Exception as exc:
                tx_fail={"reason":"bounded_patch_search_failed","piece":pi,"error":repr(exc),
                         "face_rows":rows,"boundary_loops":loops}
                break
        if tx_fail:
            failed.append({"transaction_id":tx_id,**tx_fail})
            continue
        accepted.append(tx_id);ops.extend({"transaction_id":tx_id,**x} for x in local_ops)
        accepted_pairs.extend(local_records);accepted_faces.extend(local_new);removed_original_rows.update(tx_faces)
        # Compact operation log is durable after each accepted pair-linked transaction.
        op_path=outdir/"accepted-patch-operations.json"
        progress={"schema":"pair-linked-remesh-operations.v1","candidate_sha256":base_sha,
                  "exact_report_sha256":self_sha,"plan_sha256":sha(plan_path),
                  "accepted_transactions":accepted,"operations":ops,"failed_transactions":failed}
        tmp=op_path.with_suffix(".json.tmp");tmp.write_text(json.dumps(progress,separators=(",",":"),sort_keys=True)+"\n");os.replace(tmp,op_path)
        if len(accepted)%args.checkpoint_every==0:
            removed=set(r for op in ops for r in op["removed_face_rows"])
            newf=np.asarray([tri for op in ops for tri in op["new_faces"]],np.int64).reshape(-1,3)
            keep=np.ones(len(faces),bool)
            if removed:keep[np.asarray(sorted(removed),np.int64)]=False
            checkpoint_faces=np.concatenate((faces[keep],newf),axis=0)
            ck=outdir/"candidate-geometry-checkpoint.npz"
            atomic_npz(ck,points_world_m=points32,faces=checkpoint_faces,
                      retained_input_face_rows=np.flatnonzero(keep).astype(np.int64),
                      removed_input_face_rows=np.asarray(sorted(removed),np.int64),
                      new_faces=np.asarray(newf,np.int64),candidate_input_sha256=np.asarray([base_sha]),
                      exact_report_sha256=np.asarray([self_sha]))
            print(json.dumps({"status":"CHECKPOINT","accepted_transactions":len(accepted),"failed_transactions":len(failed),
                              "output":str(ck),"faces":len(checkpoint_faces),"elapsed_s":time.monotonic()-started},sort_keys=True),flush=True)
        if len(accepted)%10==0 or ti==0:
            print(json.dumps({"status":"PROGRESS","transaction_id":tx_id,"accepted":len(accepted),
                              "failed":len(failed),"new_faces":sum(len(x["new_faces"]) for x in ops),
                              "elapsed_s":time.monotonic()-started},sort_keys=True),flush=True)
    removed=set(r for op in ops for r in op["removed_face_rows"])
    new_faces=np.asarray([tri for op in ops for tri in op["new_faces"]],np.int64).reshape(-1,3)
    keep=np.ones(len(faces),bool)
    if removed:keep[np.asarray(sorted(removed),np.int64)]=False
    fgeom=np.concatenate((faces[keep],new_faces),axis=0)
    # Save raw geometry before provenance or gate code.
    raw_path=outdir/"candidate-geometry-final-checkpoint.npz"
    atomic_npz(raw_path,points_world_m=points32,faces=fgeom,
               retained_input_face_rows=np.flatnonzero(keep).astype(np.int64),
               removed_input_face_rows=np.asarray(sorted(removed),np.int64),new_faces=new_faces,
               candidate_input_sha256=np.asarray([base_sha]),exact_report_sha256=np.asarray([self_sha]))
    # Cache the candidate lineage arrays once before any per-patch packaging.
    n_old=len(faces);retained=np.flatnonzero(keep).astype(np.int64)
    candidate_old_recs=csr_rows(d["candidate_face_ancestry_offsets"],d["candidate_face_ancestry_rows"])
    union_run_old=csr_rows(d["union_face_ancestry_offsets"],d["union_face_ancestry_run_original_ids"])
    union_fid_old=csr_rows(d["union_face_ancestry_offsets"],d["union_face_ancestry_face_ids"])
    union_row_old=csr_rows(d["union_face_ancestry_offsets"],d["union_face_ancestry_rows"])
    union_old_recs=[list(zip(union_run_old[r],union_fid_old[r],union_row_old[r])) for r in range(n_old)]
    if "d289_parent_row_indptr" in d:
        d289_old_recs=csr_rows(d["d289_parent_row_indptr"],d.get("d289_parent_rows",np.empty(0,np.int64)))
    else:d289_old_recs=[[] for _ in range(n_old)]
    if len(candidate_old_recs)!=n_old or len(union_old_recs)!=n_old or len(d289_old_recs)!=n_old:
        raise ValueError("input lineage CSR row-domain mismatch")
    op_by_patch=ops
    patch_source={}
    next_source=NEW_ID_BASE
    source_used=set(map(int,np.asarray(d["face_source_original_id"]).tolist()))
    schema=json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))
    new_face_meta=[]
    new_sidecars=[]
    side_dir=outdir/"derived-source-views";side_dir.mkdir(parents=True,exist_ok=True)
    # For every output patch, generate a distinct source view with actual current coordinates and explicit many-to-many ancestry.
    for opidx,op in enumerate(ops):
        source_tx=int(op["transaction_id"]);source_patch=int(op["patch_index"])
        if source_tx<0 or source_tx>=1_000_000 or source_patch<0 or source_patch>=100:
            raise ValueError("derived source ID exceeds the reserved transaction/piece namespace")
        source_id=NEW_ID_BASE+source_tx*100+source_patch
        if source_id in source_used: raise ValueError(f"derived source id collision: {source_id}")
        source_used.add(source_id);next_source=max(next_source,source_id+1)
        parent_rows=sorted(set(map(int,op["parent_face_rows"])))
        src_set=set(map(int,np.asarray(d["face_source_original_id"])[parent_rows]))
        layer_vals=set(map(int,np.asarray(d["face_source_layer"])[parent_rows]))
        body_vals=set(map(int,np.asarray(d["face_source_body"])[parent_rows]))
        bone_vals=set(map(int,np.asarray(d["face_source_bone"])[parent_rows]))
        layer=next(iter(layer_vals)) if len(layer_vals)==1 else -1
        body=next(iter(body_vals)) if len(body_vals)==1 else -1
        bone=next(iter(bone_vals)) if len(bone_vals)==1 else -1
        tris=np.asarray(op["new_faces"],np.int64).reshape(-1,3)
        fids=np.arange(len(tris),dtype=np.int64)
        used_v=np.unique(tris.reshape(-1));mp=np.full(len(points32),-1,np.int64);mp[used_v]=np.arange(len(used_v))
        source_faces=mp[tris]
        parents_cand=sorted(set(int(x) for r in parent_rows for x in candidate_old_recs[r]))
        parents_union=sorted(set(tuple(map(int,x)) for r in parent_rows for x in union_old_recs[r]))
        parents_d289=sorted(set(int(x) for r in parent_rows for x in d289_old_recs[r]))
        side=side_dir/f"D-derived-source-{source_id}.npz"
        atomic_npz(side,points_world_m=points32[used_v],source_faces=source_faces,
                   face_source_original_id=np.full(len(tris),source_id,np.int64),
                   face_source_face_id=fids,face_source_identity0=fids.copy(),
                   face_source_identity1=np.full(len(tris),-1,np.int64),
                   parent_candidate_sha256=np.asarray([base_sha]),
                   parent_candidate_face_offsets=np.r_[0,np.cumsum([len(parent_rows)]*len(tris))].astype(np.int64),
                   parent_candidate_face_rows=np.tile(np.asarray(parent_rows,np.int64),len(tris)),
                   parent_candidate_face_support_input_rows=np.asarray(parent_rows,np.int64),
                   parent_union_source_offsets=np.r_[0,np.cumsum([len(parents_union)]*len(tris))].astype(np.int64),
                   parent_union_source_run_original_ids=np.tile(np.asarray([x[0] for x in parents_union],np.int64),len(tris)),
                   parent_union_source_face_ids=np.tile(np.asarray([x[1] for x in parents_union],np.int64),len(tris)),
                   parent_union_source_rows=np.tile(np.asarray([x[2] for x in parents_union],np.int64),len(tris)),
                   parent_d289_offsets=np.r_[0,np.cumsum([len(parents_d289)]*len(tris))].astype(np.int64),
                   parent_d289_rows=np.tile(np.asarray(parents_d289,np.int64),len(tris)),
                   operation=np.asarray(["pair-linked exact-witness connected disk/annulus remesh; full parent support in candidate CSR; no unique source face asserted"]))
        side_sha=sha(side)
        schema[str(source_id)]={"role":"D","layer":layer,"body":body,"bone":bone,
            "identity_face_id_key":"face_source_face_id",
            "identity_keys":["face_source_identity0","face_source_identity1"],
            "derived_operation":"pair_linked_exact_witness_connected_patch_retriangulation",
            "derived_parent_assets":[{"asset_role":"candidate007_input","path":str(base),"sha256":base_sha},
                                     {"asset_role":"exact_self_witnesses","path":str(selfp),"sha256":self_sha},
                                     {"asset_role":"pair_linked_patch_plan","path":str(plan_path),"sha256":sha(plan_path)}],
            "derived_parent_source_ids":sorted(src_set),"face_count":len(tris),
            "source_geometry_scope":"retriangulated one-ring patch; every replacement face carries complete union ancestry of all removed patch faces",
            "path":str(side),"sha256":side_sha}
        new_face_meta.append({"source_id":source_id,"layer":layer,"body":body,"bone":bone,"fids":fids,
                              "tris":tris,"parent_rows":parent_rows,"parents_cand":parents_cand,
                              "parents_union":parents_union,"parents_d289":parents_d289,"sidecar":str(side)})
    # Concatenate one new source per patch; source IDs are unique across transaction subsets.
    new_n=len(new_faces)
    outdata=dict(d);outdata["faces"]=fgeom;outdata["points_world_m"]=points32
    def append_field(name,values):
        arr=np.asarray(d[name]);return np.concatenate((arr[retained],np.asarray(values,dtype=arr.dtype)),axis=0)
    if new_n:
        outdata["face_source_original_id"]=append_field("face_source_original_id",np.concatenate([np.full(len(x["tris"]),x["source_id"],np.int64) for x in new_face_meta]))
        outdata["face_source_face_id"]=append_field("face_source_face_id",np.concatenate([x["fids"] for x in new_face_meta]))
        outdata["face_source_identity0"]=append_field("face_source_identity0",np.concatenate([x["fids"] for x in new_face_meta]))
        outdata["face_source_identity1"]=append_field("face_source_identity1",np.full(new_n,-1,np.int64))
        for name in ("face_source_layer","face_source_body","face_source_bone"):
            values=np.concatenate([np.full(len(x["tris"]),x[name.removeprefix("face_source_")],np.int64) for x in new_face_meta])
            outdata[name]=append_field(name,values)
        for name in ("union_input_run_original_id","union_input_face_id","union_input_face_rows"):
            outdata[name]=append_field(name,np.full(new_n,-1,np.int64))
    else:
        for name in ("face_source_original_id","face_source_face_id","face_source_identity0","face_source_identity1",
                     "face_source_layer","face_source_body","face_source_bone","union_input_run_original_id","union_input_face_id","union_input_face_rows"):
            outdata[name]=np.asarray(d[name])[retained]
    # Preserve full original ancestry, union all removed patch parent support into every replacement face.
    candidate_ancestry=[candidate_old_recs[int(r)] for r in retained]
    union_ancestry=[union_old_recs[int(r)] for r in retained]
    d289_ancestry=[d289_old_recs[int(r)] for r in retained]
    new_candidate_lists=[];new_union_lists=[];new_d289_lists=[]
    for op in ops:
        parent_rows=sorted(set(map(int,op["parent_face_rows"])))
        cand=sorted(set(int(x) for r in parent_rows for x in candidate_old_recs[r]))
        un=sorted(set(tuple(map(int,x)) for r in parent_rows for x in union_old_recs[r]))
        d289=sorted(set(int(x) for r in parent_rows for x in d289_old_recs[r]))
        for _ in op["new_faces"]:
            new_candidate_lists.append(cand);new_union_lists.append(un);new_d289_lists.append(d289)
    candidate_ancestry+=new_candidate_lists;union_ancestry+=new_union_lists;d289_ancestry+=new_d289_lists
    def flatten_csr(rows):
        offs=[0];flat=[]
        for row in rows:flat.extend(row);offs.append(len(flat))
        return np.asarray(offs,np.int64),flat
    co,cf=flatten_csr(candidate_ancestry);outdata["candidate_face_ancestry_offsets"]=co;outdata["candidate_face_ancestry_rows"]=np.asarray(cf,np.int64)
    uo=[0];ur=[];uf=[];ui=[]
    for row in union_ancestry:
        uniq=sorted(set(tuple(map(int,x)) for x in row))
        ui.extend(x[0] for x in uniq);uf.extend(x[1] for x in uniq);ur.extend(x[2] for x in uniq);uo.append(len(ui))
    outdata["union_face_ancestry_offsets"]=np.asarray(uo,np.int64)
    outdata["union_face_ancestry_run_original_ids"]=np.asarray(ui,np.int64)
    outdata["union_face_ancestry_face_ids"]=np.asarray(uf,np.int64)
    outdata["union_face_ancestry_rows"]=np.asarray(ur,np.int64)
    do,dr=flatten_csr(d289_ancestry);outdata["d289_parent_row_indptr"]=do;outdata["d289_parent_rows"]=np.asarray(dr,np.int64)
    schemas=json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))
    schemas.update({k:v for k,v in schema.items() if k not in json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))})
    outdata["face_source_identity_schema_by_original_id_json"]=np.asarray([json.dumps(schemas,sort_keys=True,separators=(",",":"))])
    op_obj={"schema":"pair-linked-connected-remesh-operation.v1","input_candidate_sha256":base_sha,
            "exact_report_sha256":self_sha,"plan_path":str(plan_path),"plan_sha256":sha(plan_path),
            "selected_transaction_ids":sorted(wanted),"accepted_transaction_ids":accepted,
            "failed_transactions":failed,"accepted_patch_count":len(ops),"new_face_count":new_n,
            "replayed_operations_sha256":replay_sha,"replayed_source_plan_sha256":replay_source_plan_sha,
            "replay_source_plan_matches_current":replay_plan_matches_current,
            "stale_replay_proposals_discarded":stale_replay,
            "aabb_lattice_regression":aabb_regression,
            "topology_repair_exceptions":topology_repair_receipts,
            "removed_face_count":len(removed),"new_original_ids":[x["source_id"] for x in new_face_meta],
            "full_source_ancestry":"candidate_face_ancestry and union_face_ancestry are union support of every removed patch face; no single source-face identity is asserted",
            "physical_admission":False}
    outdata["pair_linked_remesh_operation_json"]=np.asarray([json.dumps(op_obj,sort_keys=True,separators=(",",":"))])
    typed_path=outdir/"candidate-f32-linked-remesh-typed.npz"
    atomic_npz(typed_path,**outdata)
    final_receipt={"schema":"pair-linked-connected-remesh-result.v1","candidate_input":str(base),"candidate_input_sha256":base_sha,
        "exact_report":str(selfp),"exact_report_sha256":self_sha,"plan_path":str(plan_path),"plan_sha256":sha(plan_path),
        "typed_candidate":str(typed_path),"typed_candidate_sha256":sha(typed_path),"raw_geometry_checkpoint":str(raw_path),"raw_geometry_sha256":sha(raw_path),
        "input_vertices":len(points32),"input_faces":len(faces),"output_vertices":len(points32),"output_faces":len(fgeom),
        "selected_transaction_count":len(wanted),"accepted_transaction_count":len(accepted),"failed_transaction_count":len(failed),
        "accepted_patch_count":len(ops),"new_faces":new_n,"removed_faces":len(removed),"failed_transactions":failed,
        "replay_operations_sha256":replay_sha,"replay_source_plan_sha256":replay_source_plan_sha,
        "replay_source_plan_matches_current":replay_plan_matches_current,
        "stale_replay_proposal_count":len(stale_replay),"stale_replay_proposals":stale_replay,
        "aabb_lattice_regression":aabb_regression,"topology_repair_exceptions":topology_repair_receipts,
        "exact_self_gate":"NOT_RUN","all185_gate":"NOT_RUN","D_minus_I_gate":"NOT_RUN","physical_admission":False,
        "elapsed_s":time.monotonic()-started}
    rp=outdir/"result.json";rp.write_text(json.dumps(final_receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"CANDIDATE_SAVED","candidate":str(typed_path),"candidate_sha256":sha(typed_path),
        "accepted_transactions":len(accepted),"failed_transactions":len(failed),"new_faces":new_n,"faces":len(fgeom),
        "receipt":str(rp),"elapsed_s":time.monotonic()-started},sort_keys=True),flush=True)

if __name__=="__main__":
    # build_plan uses a module-level face array only for the tiny BFS helper.
    global CURRENT_FACES
    main()

