"""Strict reader primitives for v2 current-row reciprocal lung maps."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib
import struct

PAIRS = tuple((a, b) for i, a in enumerate((305,306,307,308,309))
              for b in (305,306,307,308,309)[i+1:])
EXPECTED_OPERATIONS = {
    "operation_1083": {"ledger":("/Users/n/numi-human-resting-evidence-20261005/native-lung-307-sliver-reciprocal-flip-1083/candidate-307-309-parent-ledger.json","df98cf7961a3f791d16d4f7eb7d5b6d63f1ac0b01df72aa22c360b70a34e34e0"),"npz":("/Users/n/numi-human-resting-evidence-20261005/native-lung-307-sliver-reciprocal-flip-1083/candidate-307-309-rows.npz","83898f5257a9c2cbb61ab567ed4af2106d761a2db20f7bd160f75fc80a8cec89")},
    "operation_306": {"npz":("/Users/n/numi-human-resting-evidence-20261005/native-lung-306-interior-seam-collapse-1099/candidate-rows.npz","25812e48e0f4bdeccfa13ffd6b2392b6de5008fc2f593c7c1b2a094ce3e7b325"),"ledger":("/Users/n/numi-human-resting-evidence-20261005/native-lung-306-interior-seam-collapse-1099/parent-ledger.json","de646ea2ffc7f015aa43a3ce4d9291c01b15276e5e56d0fb526e58a2189c4fb4")},
    "operation_308_first": {"npz":("/Users/n/numi-human-resting-evidence-20261005/native-lung-305-308-sequential-collapse-trial-1105/candidate-rows.npz","c94ee31676bbb76d3ac697bab3af93b4225cbf10f900c6f586346a4441f8781e"),"report":("/Users/n/numi-human-resting-evidence-20261005/native-lung-305-308-sequential-collapse-trial-1105/report.json","a658e3abe1cf93bba9928f31a5e465d1a2c0cb42deb2370c32c7f177fa077057")},
    "operation_308_second": {"npz":("/Users/n/numi-human-resting-evidence-20261005/native-lung-left-second-cluster-star-trials-1105/trial-02.npz","74d8f052f3847725769b571ca99b8dd9b5cf74984a5d8f1ea573a83e5511cba9"),"report":("/Users/n/numi-human-resting-evidence-20261005/native-lung-left-second-cluster-star-trials-1105/report.json","4f7769e02fe0665feeb0941c945673dcb96f24d041675e58533790938642f1be"),"owner":("/Users/n/numi-human-final-lung-composition-001/src/numilab_human/resting_respiratory_mesh_quality.py","774b49fc5cc97518f035ad4434e1cf64b3dceb037b91dfd0a37aff89a81c8951")},
}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _float_bits(x):
    # Accept packed Float32 hex words (1055/tests) or four byte hex octets (1113).
    if isinstance(x, str):
        word=x.lower()
        if len(word)!=8: raise ValueError("Float32 word must have eight hex digits")
        bytes.fromhex(word)
        return word
    if isinstance(x,(list,tuple)) and len(x)==4 and all(isinstance(q,str) and len(q)==2 for q in x):
        word="".join(q.lower() for q in x); bytes.fromhex(word); return word
    raise ValueError("invalid Float32 bit word")

def _bits(p):
    if not isinstance(p,(list,tuple)):
        raise ValueError("coordinate must contain three Float32 bit words")
    # Some producers serialize a point as 12 little-endian byte octets (xyz
    # concatenated); others preserve the 3x4 byte grouping. Normalize both.
    if len(p)==12 and all(isinstance(x,str) and len(x)==2 for x in p):
        p=[p[i:i+4] for i in (0,4,8)]
    if len(p)!=3:
        raise ValueError("coordinate must contain three Float32 bit words")
    try:
        out=tuple(_float_bits(x) for x in p)
        for x in out: struct.unpack("<f",bytes.fromhex(x))
    except (ValueError,struct.error) as exc:
        raise ValueError("invalid Float32 coordinate bits") from exc
    return out

def _scalar_ids(v,label,*,unique=True):
    xs=[v] if isinstance(v,int) else v
    if not isinstance(xs,(list,tuple)) or not xs or any(not isinstance(x,int) or x<0 for x in xs) or (unique and len(set(xs))!=len(xs)):
        raise ValueError(label+" must be nonnegative IDs" + (" without duplicates" if unique else ""))
    return tuple(int(x) for x in xs)


def _tri_bits(row, fi):
    fi=int(fi)
    if fi<0 or fi>=len(row["faces"]): raise ValueError("current face row out of bounds")
    return tuple(tuple(struct.pack("<f",float(c)).hex() for c in row["vertices6"][int(i),:3])
                 for i in row["faces"][fi])


def _ids(v,label):
    xs=[v] if isinstance(v,int) else v
    if not isinstance(xs,list) or not xs or any(not isinstance(x,int) or x<0 for x in xs) or len(set(xs))!=len(xs):
        raise ValueError(label+" must be unique nonnegative IDs")
    return tuple(xs)


def _sig(tri,point_key):
    ks=[point_key(tuple(struct.unpack("<f",bytes.fromhex(c))[0] for c in p)) for p in tri]
    if len(set(ks))!=3: raise ValueError("degenerate mapped triangle")
    ordered=sorted(ks); perm=[ordered.index(k) for k in ks]
    parity=sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))%2
    return tuple(ordered),(-1 if parity else 1)


def _norm_inc(i):
    d=i.get("directed_endpoint_bits")
    if not isinstance(d,list) or len(d)!=2: raise ValueError("edge incidence lacks endpoints")
    sid=i.get("source_face_ids",i.get("source_face_id"))
    kind=i.get("patch_kinds",i.get("patch_kind"))
    return (int(i["face_map_row"]),int(i["face_row"]),_scalar_ids(sid,"edge source IDs"),
            _scalar_ids(kind,"edge patch kinds",unique=False),tuple(_bits(p) for p in d))


def validate_v2_face_edge_maps(*,face_rows,edge_rows,pair_summaries,final_rows,point_key):
    """Validate exact mapped triangles, reciprocal winding, and full edge incidence."""
    summaries={}
    for s in pair_summaries:
        p=tuple(int(x) for x in s.get("pair",()))
        if p not in PAIRS or p in summaries: raise ValueError("invalid/duplicate pair declaration")
        summaries[p]=s
    if set(summaries)!=set(PAIRS): raise ValueError("all ten lobe pairs, including zeros, must be declared")
    by=defaultdict(list); used=defaultdict(lambda:[set(),set()]); edges=defaultdict(lambda:defaultdict(lambda:[[],[]]))
    for mi,r in enumerate(face_rows):
        p=tuple(int(x) for x in r.get("pair",()))
        if p not in PAIRS: raise ValueError("undeclared face-map pair")
        fr=r.get("face_rows"); ob=r.get("mapped_output_triangle_vertices_bits")
        src=r.get("source_face_ids"); kind=r.get("patch_kinds")
        if not all(isinstance(x,list) and len(x)==2 for x in (fr,ob,src,kind)):
            raise ValueError("face-map row must provide two aligned sides")
        fr=[int(x) for x in fr]; ob=[tuple(_bits(q) for q in t) for t in ob]
        if any(len(t)!=3 for t in ob): raise ValueError("mapped face must have three vertices")
        if r.get("orientation_relation")!="opposite_winding": raise ValueError("missing opposite-winding declaration")
        if r.get("source_parent_relation") not in ("identical_source_parent_triangle","paired_two_face_union","parent_face_union","two_parent_inferred_reference_diagonal_flip"):
            raise ValueError("unsupported source-parent relation")
        for s,sid in enumerate(p):
            _ids(src[s],"source_face_ids"); _scalar_ids(kind[s],"patch_kinds",unique=False)
            if len(_ids(src[s],"source_face_ids"))!=len(_scalar_ids(kind[s],"patch_kinds",unique=False)):
                raise ValueError("source IDs/kinds parent arity mismatch")
            if sid not in final_rows: raise ValueError("current NHA row missing")
            if fr[s] in used[p][s]: raise ValueError("face mapped more than once within pair")
            used[p][s].add(fr[s])
            if _tri_bits(final_rows[sid],fr[s])!=ob[s]:
                raise ValueError("mapped output triangle differs from exact current NHA face")
        sigs=[_sig(ob[s],point_key) for s in (0,1)]
        if sigs[0][0]!=sigs[1][0] or sigs[0][1]!=-sigs[1][1]:
            raise ValueError("current mapped triangles are not opposite-wound")
        for s in (0,1):
            for k in range(3):
                u,v=ob[s][k],ob[s][(k+1)%3]
                edges[p][frozenset((tuple(u),tuple(v)))][s].append({
                    "face_map_row":mi,"face_row":fr[s],"source_face_ids":src[s],
                    "patch_kinds":kind[s],"directed_endpoint_bits":[u,v]})
        by[p].append(dict(r,_map_row=mi))
    seen=defaultdict(dict)
    for e in edge_rows:
        p=tuple(int(x) for x in e.get("pair",()))
        if p not in PAIRS: raise ValueError("undeclared edge-map pair")
        ep=e.get("endpoint_coordinate_bits")
        if not isinstance(ep,list) or len(ep)!=2: raise ValueError("edge endpoint arity mismatch")
        ek=frozenset(tuple(_bits(x)) for x in ep)
        if ek in seen[p]: raise ValueError("duplicate edge map row")
        want=edges[p].get(ek)
        if want is None: raise ValueError("edge map contains nonmapped edge")
        sides=[]
        for name,s in (("incidences_a",0),("incidences_b",1)):
            got=e.get(name)
            if not isinstance(got,list) or sorted(_norm_inc(x) for x in got)!=sorted(_norm_inc(x) for x in want[s]):
                raise ValueError("edge incidence differs from face map")
            sides.append(got)
        if len(sides[0]) not in (1,2) or len(sides[0])!=len(sides[1]): raise ValueError("edge incidence not reciprocal")
        cls="boundary" if len(sides[0])==1 else "interior"
        if e.get("classification")!=cls or e.get("paired_face_edge_directions_opposed") is not True:
            raise ValueError("edge class/orientation mismatch")
        for a in sides[0]:
            b=next((x for x in sides[1] if int(x["face_map_row"])==int(a["face_map_row"])),None)
            if b is None or a["directed_endpoint_bits"]!=list(reversed(b["directed_endpoint_bits"])):
                raise ValueError("paired directions are not opposed")
        seen[p][ek]=e
    result={}
    for p in PAIRS:
        s=summaries[p]; entries=by[p]
        n=int(s.get("full_reciprocal_face_map_count",s.get("face_pair_count",-1)))
        if n!=len(entries): raise ValueError("per-pair map count mismatch")
        status=s.get("map_status",s.get("status"))
        if n==0:
            if status not in ("explicit_zero_no_map","zero_no_map") or seen[p]:
                raise ValueError("zero pair is not explicitly zero/no-map")
            es=s.get("edge_summary",{})
            if es and (int(es.get("edge_count",-1))!=0 or int(es.get("boundary_edge_count_a",-1))!=0
                       or int(es.get("boundary_edge_count_b",-1))!=0
                       or es.get("explicit_zero_no_map") is not True):
                raise ValueError("zero pair edge summary is not explicitly empty")
            result[p]={"face_pairs":set(),"shared_vertices":set(),"shared_edges":set(),"map_count":0,"edge_count":0}
            continue
        if status not in ("exact_source_supported_reciprocal_surface","exact_current_reciprocal_surface","mapped"):
            raise ValueError("nonzero pair lacks source/current reciprocal status")
        if set(seen[p])!=set(edges[p]): raise ValueError("edge map incomplete or extended")
        counts={"boundary":0,"interior":0}
        for e in seen[p].values(): counts[e["classification"]]+=1
        ec=s.get("edge_counts",s.get("edge_map_counts",{}))
        es=s.get("edge_summary",{})
        if es and (es.get("boundary_sets_equal") is not True
                   or int(es.get("boundary_edge_count_a",-1))!=int(es.get("boundary_edge_count_b",-1))):
            raise ValueError("edge summary does not prove reciprocal boundary sets")
        if not ec and es:
            boundary=int(es.get("boundary_edge_count_a",-1))
            ec={"boundary":boundary,"interior":int(es.get("edge_count",-1))-boundary}
        if ec and (counts["boundary"]!=int(ec.get("boundary",-1)) or counts["interior"]!=int(ec.get("interior",-1))):
            raise ValueError("edge summary count mismatch")
        fp=set(); verts=set(); es=set()
        for r in entries:
            fa,fb=map(int,r["face_rows"]); fp.add((fa,fb))
            tri=r["mapped_output_triangle_vertices_bits"][0]
            ks=[point_key(tuple(struct.unpack("<f",bytes.fromhex(c))[0] for c in _bits(q))) for q in tri]
            verts.update(ks)
            es.update(tuple(sorted((ks[i],ks[(i+1)%3]))) for i in range(3))
        result[p]={"face_pairs":fp,"shared_vertices":verts,"shared_edges":es,"map_count":n,"edge_count":len(edges[p])}
    return {"pairs":result,"declared_pairs":[list(p) for p in PAIRS],
            "schema":"numi.human.lung-reciprocal-interface-map.v2"}


def validate_paired_union(*,stable_id,current_face_row,parent_face_rows,parent_row,candidate_row,ledger):
    """Strictly replay one child of an exact two-face diagonal-flip ledger."""
    import numpy as np
    sid=str(int(stable_id)); f=ledger.get("modified_faces",{}).get(sid)
    if not isinstance(f,dict): raise ValueError("ledger lacks modified owner")
    rows=sorted(int(x) for x in f.get("face_rows",[]))
    if len(rows)!=2 or int(current_face_row) not in rows: raise ValueError("child face absent from paired operation")
    parents=sorted(int(x) for x in parent_face_rows)
    assigned=sorted(int(x) for x in f.get("parent_sets_after",{}).get(str(int(current_face_row)),[]))
    if parents!=rows or assigned!=rows: raise ValueError("paired-union parent set missing/incorrect")
    before,after=f.get("before_faces",[]),f.get("after_faces",[])
    if len(before)!=2 or len(after)!=2: raise ValueError("ledger must retain two parent and child faces")
    for i,row in enumerate(rows):
        if list(map(int,parent_row["faces"][row]))!=list(map(int,before[i])): raise ValueError("parent face differs from source")
        if list(map(int,candidate_row["faces"][row]))!=list(map(int,after[i])): raise ValueError("child face differs from candidate")
    if not f.get("all_xyz_unchanged") or not np.array_equal(parent_row["vertices6"][:,:3],candidate_row["vertices6"][:,:3]):
        raise ValueError("paired flip changed XYZ")
    old,new=defaultdict(int),defaultdict(int)
    for counter,triangles in ((old,[parent_row["faces"][r] for r in rows]),
                              (new,[candidate_row["faces"][r] for r in rows])):
        for tri in triangles:
            for i in range(3): counter[tuple(sorted((int(tri[i]),int(tri[(i+1)%3]))))]+=1
    if {e for e,n in old.items() if n==1}!={e for e,n in new.items() if n==1}:
        raise ValueError("paired flip changed exact patch boundary")
    if any(n not in (1,2) for n in old.values()) or any(n not in (1,2) for n in new.values()):
        raise ValueError("paired patch edge incidence invalid")
    return {"status":"PASS_exact_paired_parent_union","stable_id":int(stable_id),
            "child_face_row":int(current_face_row),"parent_face_rows":rows}


def _side_parent_lists(value,label):
    if not isinstance(value,list) or len(value)!=2:
        raise ValueError(label+" must have two sides")
    out=[]
    for x in value:
        vals=[x] if isinstance(x,int) else x
        if not isinstance(vals,list) or not vals or any(not isinstance(i,int) or i<0 for i in vals):
            raise ValueError(label+" contains invalid parent IDs")
        out.append(tuple(int(i) for i in vals))
    if len(out[0])!=len(out[1]):
        raise ValueError(label+" parent arity differs by side")
    return out


def _tri_key(tri):
    return tuple(sorted(_bits(p) for p in tri))


def build_1055_source_row_lookup(source_records):
    lookup=defaultdict(list)
    for ordinal,x in enumerate(source_records):
        pair=tuple(int(v) for v in x.get("pair",()))
        if pair not in PAIRS: raise ValueError("1055 source map has invalid pair")
        for side in (0,1):
            identity=(pair,side,int(x["face_rows"][side]),int(x["source_face_ids"][side]))
            lookup[identity].append(ordinal)
            lookup[identity+(int(x["patch_kinds"][side]),)].append(ordinal)
    return lookup


def resolve_1055_source_parents(*,source_records,lookup,pair,parent_rows,source_ids,patch_kinds=None):
    """Resolve overlay rows to reciprocal 1055 records; derive ordinary-row kinds when omitted."""
    pair=tuple(int(x) for x in pair)
    rows=_side_parent_lists(parent_rows,"1055 parent rows")
    ids=_side_parent_lists(source_ids,"1055 source IDs")
    kinds=None if patch_kinds is None else _side_parent_lists(patch_kinds,"1055 patch kinds")
    if kinds is not None and not (len(rows[0])==len(ids[0])==len(kinds[0])==len(rows[1])==len(ids[1])==len(kinds[1])):
        raise ValueError("1055 parent/source arrays are not aligned")
    if any(len(rows[s])!=len(ids[s]) for s in (0,1)):
        raise ValueError("1055 parent/source arrays are not aligned")
    ordinals=[];resolved=[]
    for side in (0,1):
        side_ord=[]
        triples=zip(rows[side],ids[side],kinds[side]) if kinds is not None else ((r,i,None) for r,i in zip(rows[side],ids[side]))
        for row,sid,kind in triples:
            identity=(pair,side,int(row),int(sid))
            key=identity if kind is None else identity+(int(kind),)
            hits=lookup.get(key,[])
            if len(hits)!=1: raise ValueError("1055 face-row/source identity is missing or non-unique")
            rec=source_records[hits[0]]
            if kind is not None and int(rec["patch_kinds"][side])!=int(kind):
                raise ValueError("1055 patch kind differs from source-map record")
            side_ord.append(hits[0])
        ordinals.append(side_ord)
        resolved.append([source_records[i] for i in side_ord])
    if set(ordinals[0])!=set(ordinals[1]):
        raise ValueError("reciprocal sides resolve to different 1055 source-map rows")
    return resolved,ordinals


def _parent_triangles(raw, arity, label):
    if arity==1:
        value=[raw]
    else:
        value=raw
    if not isinstance(value,list) or len(value)!=arity:
        raise ValueError(label+" parent coordinate arity mismatch")
    return [tuple(_bits(p) for p in tri) for tri in value]


def _verify_meta(meta,label):
    if not isinstance(meta,dict) or not isinstance(meta.get("path"),str) or not isinstance(meta.get("sha256"),str):
        raise ValueError(label+" lacks path/hash")
    path=Path(meta["path"]).resolve()
    if not path.is_file() or sha(path)!=meta["sha256"]:
        raise ValueError(label+" hash mismatch: "+str(path))
    return path


def load_v2_bridge(*,base,report_path,final_nha_path,final_nha_sha,final_rows):
    """Verify a v2 current-row bridge and return the 1084 native lobe-map shape."""
    import collections, json
    from pathlib import Path
    rp=Path(report_path).resolve()
    if not rp.is_file(): raise ValueError("v2 bridge report missing")
    report=json.loads(rp.read_text())
    if report.get("schema")!="numi.human.lobe-lobe-source-proven-reciprocal-interface-map.v2":
        raise ValueError("unsupported current lobe bridge schema")
    final_path=Path(final_nha_path).resolve()
    final_meta=report.get("final_nha",{})
    if Path(final_meta.get("path","")).resolve()!=final_path or final_meta.get("sha256")!=final_nha_sha or sha(final_path)!=final_nha_sha:
        raise ValueError("v2 bridge is not bound to exact current NHA")
    parent_meta=report.get("parent_1078",{})
    parent_path=_verify_meta(parent_meta,"1078 parent NHA")
    expected_parent=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-conditioned-final-compose-1078/final/resting-thorax.nhanatomy").resolve()
    if parent_path!=expected_parent or parent_meta["sha256"]!="7f6a8175e3cadb93a6dfc43535467b6896cf1b8bacfd0334df7b5bb03c414e92":
        raise ValueError("v2 bridge parent is not the pinned 1078 NHA")

    src1055=report.get("source_1055",{})
    srmeta=src1055.get("report",{}); smeta=src1055.get("map",{})
    source_report=_verify_meta(srmeta,"1055 source report")
    source_map=_verify_meta(smeta,"1055 source map")
    expected_sr=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-lobe-source-lineage-1055-verified/source-proven-reciprocal-interface-report.json").resolve()
    expected_sm=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-lobe-source-lineage-1055-verified/reciprocal-coincident-face-map.jsonl").resolve()
    if source_report!=expected_sr or source_map!=expected_sm:
        raise ValueError("v2 bridge does not use the pinned 1055 source lineage")
    sr=json.loads(source_report.read_text())
    src_payload=sr.get("source_payload_revisions",{}).get("947",{}).get("payload",{})
    src_nha=Path(src_payload.get("path","")).resolve(); src_sha=src_payload.get("sha256")
    if not src_nha.is_file() or not src_sha or sha(src_nha)!=src_sha:
        raise ValueError("1055 source NHA reference missing/hash mismatch")
    parser=base.load(base.PARSER,"v2_parent_source_parser")
    source_rows=parser.parse_payload(src_nha)[1]
    source_lineage=base.load_lobe_lineage(source_report,src_nha,src_sha,source_rows)
    if Path(source_lineage["map_path"]).resolve()!=source_map:
        raise ValueError("1055 map path differs from validated source lineage")
    source_records=[json.loads(x) for x in source_map.read_text().splitlines() if x.strip()]
    if len(source_records)!=29226: raise ValueError("1055 source map row count mismatch")

    oprefs={}
    for name in ("operation_1083","operation_306","operation_308_first","operation_308_second"):
        ent=report.get(name)
        if not isinstance(ent,dict): raise ValueError("v2 bridge missing "+name)
        expected=EXPECTED_OPERATIONS[name]
        for k,v in ent.items():
            if k not in expected:
                if isinstance(v,dict) and "path" in v: oprefs[(name,k)]=_verify_meta(v,name+" "+k)
                continue
            path=_verify_meta(v,name+" "+k)
            exp_path,exp_sha=expected[k]
            if path!=Path(exp_path).resolve() or v.get("sha256")!=exp_sha:
                raise ValueError(name+" "+k+" differs from the pinned selected operation")
            oprefs[(name,k)]=path
        if set(ent).intersection(expected)!=set(expected):
            raise ValueError(name+" omits a required selected operation artifact")
    op1083=report["operation_1083"]
    led1083=Path(op1083["ledger"]["path"]).resolve(); npz1083=Path(op1083["npz"]["path"]).resolve()
    ledger=json.loads(led1083.read_text())
    with base.np.load(npz1083,allow_pickle=False) as z:
        flip_candidate={sid:{"vertices6":base.np.asarray(z[f"row{sid}_vertices6"],dtype="<f4"),
                             "faces":base.np.asarray(z[f"row{sid}_faces"],dtype=base.np.int64)} for sid in (307,309)}
    parent_rows=parser.parse_payload(parent_path)[1]

    fmap=report.get("face_map",{}); emap=report.get("edge_map",{})
    face_path=_verify_meta(fmap,"v2 face map"); edge_path=_verify_meta(emap,"v2 edge map")
    face_raw=[json.loads(x) for x in face_path.read_text().splitlines() if x.strip()]
    edge_raw=[json.loads(x) for x in edge_path.read_text().splitlines() if x.strip()]
    if int(fmap.get("rows",-1))!=len(face_raw) or int(emap.get("rows",-1))!=len(edge_raw):
        raise ValueError("v2 map row count metadata mismatch")
    if len(source_records)==0: raise ValueError("1055 source map is empty")
    source_row_lookup=build_1055_source_row_lookup(source_records)

    # Rebind exact old source face rows and verify each final/current triangle's
    # immediate 1078 parent. Diagonal-flip children must replay the specific 1083 ledger.
    for mi,r in enumerate(face_raw):
        pair=tuple(int(x) for x in r.get("pair",()))
        if pair not in PAIRS: raise ValueError("v2 face map has invalid pair")
        p1055_raw=r.get("parent_face_rows_1055")
        ids1055_raw=r.get("source_face_ids_1055")
        kinds1055_raw=r.get("patch_kinds_1055")
        p1078=_side_parent_lists(r.get("parent_face_rows_1078"),"1078 parent rows")
        srcids=_side_parent_lists(r.get("source_face_ids"),"current source IDs")
        kinds=_side_parent_lists(r.get("patch_kinds"),"current patch kinds")
        map_parent_records_by_side,map_ordinals=resolve_1055_source_parents(
            source_records=source_records,lookup=source_row_lookup,pair=pair,
            parent_rows=p1055_raw,source_ids=ids1055_raw,patch_kinds=kinds1055_raw)
        p1055=_side_parent_lists(p1055_raw,"1055 parent rows")
        ids1055=_side_parent_lists(ids1055_raw,"1055 source IDs")
        kinds1055=[[int(rec["patch_kinds"][side]) for rec in map_parent_records_by_side[side]] for side in (0,1)]
        if kinds1055_raw is not None:
            declared_kinds=_side_parent_lists(kinds1055_raw,"1055 patch kinds")
            if [list(x) for x in declared_kinds]!=kinds1055: raise ValueError("1055 patch kinds differ from source-map records")
        if not (len(map_ordinals[0])==len(p1078[0])==len(srcids[0])==len(kinds[0])):
            raise ValueError("v2 parent/source arrays are not aligned")
        for side in (0,1):
            if tuple(int(x) for x in srcids[side])!=tuple(int(x) for x in ids1055[side]) or tuple(int(x) for x in kinds[side])!=tuple(int(x) for x in kinds1055[side]):
                raise ValueError("current map source identity changed from 1055")
        relation=r.get("source_parent_relation")
        if relation=="identical_source_parent_triangle":
            if len(p1055[0])!=1 or len(p1078[0])!=1: raise ValueError("identity parent must have one source and one 1078 parent")
            for side,sid in enumerate(pair):
                oldtri=map_parent_records_by_side[side][0]["mapped_output_triangle_vertices_bits"][side]
                if _tri_bits(parent_rows[sid],p1078[side][0])!=tuple(_bits(point) for point in r["parent_face_vertices_bits_1078"][side]):
                    raise ValueError("1078 parent face bits disagree with parent NHA")
                if _tri_bits(parent_rows[sid],p1078[side][0])!=_tri_bits(final_rows[sid],int(r["face_rows"][side])):
                    raise ValueError("current mapped face differs from immediate 1078 parent")
                if _tri_bits(final_rows[sid],int(r["face_rows"][side]))!=tuple(_bits(point) for point in oldtri):
                    raise ValueError("current mapped face differs from 1055 source face")
        elif relation=="two_parent_inferred_reference_diagonal_flip":
            if pair!=(307,309) or len(p1055[0])!=2 or len(p1078[0])!=2:
                raise ValueError("paired flip relation used outside exact 1083 case")
            opref=r.get("operation_ref",{})
            if (Path(opref.get("path","")).resolve()!=led1083
                or Path(opref.get("npz_path","")).resolve()!=npz1083
                or opref.get("sha256")!=op1083["ledger"]["sha256"]
                or opref.get("npz_sha256")!=op1083["npz"]["sha256"]
                or opref.get("label")!=ledger.get("label")
                or opref.get("method")!=ledger.get("operation",{}).get("method")):
                raise ValueError("paired flip row is not bound to top-level 1083 operation")
            bits1078=r.get("parent_face_vertices_bits_1078")
            bits1055=r.get("source_parent_face_vertices_bits_1055")
            if not isinstance(bits1078,list) or len(bits1078)!=2 or not isinstance(bits1055,list) or len(bits1055)!=2:
                raise ValueError("paired flip omits immediate/source parent coordinate witnesses")
            for side,sid in enumerate(pair):
                for j,oldrow in enumerate(p1055[side]):
                    src_record=map_parent_records_by_side[side][j]
                    oldtri=src_record["mapped_output_triangle_vertices_bits"][side]
                    srcparent=src_record["source_parent_face_vertices_bits"][side]
                    if _tri_key(oldtri)!=_tri_key(srcparent):
                        raise ValueError("1055 reciprocal parent is not exact source geometry")
                    if _tri_key(oldtri)!=_tri_key(bits1055[side][j]):
                        raise ValueError("declared 1055 parent bits differ from source map")
                    if _tri_key(oldtri)!=_tri_key(_tri_bits(parent_rows[sid],p1078[side][j])):
                        raise ValueError("1078 parent row differs from 1055 parent map")
                    if _tri_key(oldtri)!=_tri_key(bits1078[side][j]):
                        raise ValueError("declared 1078 parent bits differ from parent NHA")
                child=int(r["face_rows"][side])
                if child not in p1078[side]: raise ValueError("flip child not in paired 1078 parent set")
                if _tri_bits(final_rows[sid],child)!=_tri_bits(flip_candidate[sid],child):
                    raise ValueError("final flip child differs from 1083 candidate")
                validate_paired_union(stable_id=sid,current_face_row=child,parent_face_rows=p1078[side],
                    parent_row=parent_rows[sid],candidate_row=flip_candidate[sid],ledger=ledger)
                parent_xyz={_bits(pt) for x in map_parent_records_by_side[side] for pt in x["mapped_output_triangle_vertices_bits"][side]}
                if any(_bits(pt) not in parent_xyz for pt in _tri_bits(final_rows[sid],child)):
                    raise ValueError("flip child has a vertex outside its exact 1055 parent union")
        else:
            raise ValueError("unsupported current-face source ancestry relation")

    pair_summaries=report.get("pairs",[])
    edge_summary=report.get("edge_summary",{})
    # Attach per-pair edge statistics so the validator can compare declared counts.
    normalized=[]
    for s in pair_summaries:
        q=dict(s); pair=tuple(int(x) for x in q["pair"])
        if pair in edge_summary: q["edge_summary"]=edge_summary[pair] if pair in edge_summary else edge_summary.get("%d-%d"%pair,{})
        if not q.get("edge_summary"): q["edge_summary"]=edge_summary.get("%d-%d"%pair,{})
        normalized.append(q)
    parsed=validate_v2_face_edge_maps(face_rows=face_raw,edge_rows=edge_raw,pair_summaries=normalized,
                                      final_rows=final_rows,point_key=base.pkey)
    inputs=dict(source_lineage.get("inputs",{}))
    inputs.update({str(rp):sha(rp),str(final_path):sha(final_path),str(parent_path):sha(parent_path),
            str(source_report):sha(source_report),str(source_map):sha(source_map),
            str(src_nha):sha(src_nha),str(led1083):sha(led1083),str(npz1083):sha(npz1083),
            str(face_path):sha(face_path),str(edge_path):sha(edge_path)})
    for name,k in oprefs:
        path=oprefs[(name,k)]; inputs[str(path)]=sha(path)
    lineage=dict(source_lineage)
    lineage.update({"path":str(rp),"sha256":sha(rp),"map_path":str(face_path),"map_sha256":sha(face_path),
        "edge_path":str(edge_path),"edge_sha256":sha(edge_path),"pairs":parsed["pairs"],
        "declared_pairs":[list(x) for x in PAIRS],"inputs":inputs,
        "manifest_products":sorted(set(source_lineage.get("manifest_products",[])+list(inputs))),
        "source_simplex_validation":"1055 original source map rows are rebound through pinned operation ancestry; current triangles and edge incidences are exact Float32 validated. Only the two declared 1083 diagonal-flip children use paired parent unions."})
    return lineage
