"""Retriangulate connected skin witnesses without moving source vertices."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from register_tissue_bellies import B, CASES, HERE, LATE, READER, capture, clear, exact, hash_file, install_for_float32
sys.path.insert(0, str(B.parent/"resting-clearance-and-atrium-004"))
from fit_skin_source import decode
from trim_skin_witnesses import positioned_records


def edges(face):
    a,b,c=map(int,face)
    return [(a,b,c),(b,c,a),(c,a,b)]


def topology(faces):
    uses=defaultdict(dict)
    for i,face in enumerate(faces):
        for a,b,c in edges(face):
            uses[min(a,b),max(a,b)][i]=(a,b,c)
    return uses


class Geometry:
    def __init__(self, points, faces, baseline, movable, targets=None):
        self.points=points
        self.tree=exact._prepare_surface_aabb(positioned_records(points,faces,np.arange(len(faces))))
        self.movable=set(movable)
        self.edits={}
        self.baseline=set(map(tuple,baseline))
        self.self_hits=self.baseline.copy()
        self.targets=targets

    def inspect(self, ids, rows):
        points=self.points
        new=positioned_records(points,rows,np.arange(len(rows)))
        records={int(i):(a,b,c,int(i),d) for i,(a,b,c,_,d) in zip(ids,new)}
        ignored=self.edits.keys()|records.keys()
        nearby={}
        for x,y in exact._query_surface_aabb(records.values(),self.tree):
            if y[3] not in ignored:
                nearby[tuple(sorted((x[3],y[3])))]=(x,y)
        for x in records.values():
            for y in (*self.edits.values(),*records.values()):
                if x[3]!=y[3] and y[3] not in records:
                    nearby[tuple(sorted((x[3],y[3])))]=(x,y)
        a,b=records.values()
        nearby[tuple(sorted((a[3],b[3])))]=(a,b)
        # The owner reads face IDs in input order. Stabilize even when a query
        # against either old or newly created records discovers the same pair.
        encountered=exact._audit_record_pairs(
            (tuple(sorted(v,key=lambda r:r[3])) for v in nearby.values()),same_surface=True)
        changed=set(map(tuple,encountered["triangle_pairs"]))
        if changed-self.self_hits:
            return None
        previously={pair for pair in self.self_hits if any(i in records for i in pair)}
        move=previously-changed
        tissue_removed=0
        if self.targets is not None:
            target_tree,by_face=self.targets
            pairs=exact._query_surface_aabb(records.values(),target_tree)
            found=exact._audit_record_pairs(pairs,same_surface=False)["triangle_pairs"]
            prior={(i,j) for i in records for j in by_face.get(i,())}
            if set(map(tuple,found))-prior:
                return None
            tissue_removed=len(prior)-len(found)
        return records,move,tissue_removed

    def accept(self, result):
        records,removed,tissue_removed=result
        self.edits.update(records)
        self.self_hits.difference_update(removed)
        if self.targets is not None:
            tree,by_face=self.targets
            for i in records:
                by_face[i]=set()
            hits=exact._audit_record_pairs(exact._query_surface_aabb(records.values(),tree),
                                            same_surface=False)["triangle_pairs"]
            for i,j in hits:
                by_face[i].add(j)


def prepare_case(path, audit, faces, movable):
    records=[]
    existing=defaultdict(set)
    tables={(r["semantic"],r["stable_id"]):r for r in audit["surfaces"]}
    witnesses={int(i) for i in movable}
    with capture(path) as (m,bodies,receipt):
        assert receipt["pack_file_sha256"]==audit["accepted_pack_sha256"]
        skin=[(p,f) for key,_,p,f in READER["primitives"](m,bodies,skin_only=True)]
        assert len(skin)==1
        compact,emitted=skin[0]
        source_rows=np.unique(faces)
        assert len(compact)==len(source_rows)
        assert np.array_equal(np.searchsorted(source_rows, faces),emitted)
        points=np.zeros((receipt["skin_source_mapping"]["source_vertex_count"],3),dtype="<f4")
        points[source_rows]=compact
        # A valid flip uses vertices of two old adjacent triangles. Both new
        # triangles therefore stay in the union of that quad's AABB.
        tri=points[faces]
        one=clear._exact_surface_records(points,faces)
        boxes=exact._prepare_surface_aabb(one)
        adjacency=topology(faces)
        keys=exact.float32_point_lattice_key
        envelope_all=[]
        for i in movable:
            neighbors={i}
            for a,b,_ in edges(faces[i]):
                neighbors.update(adjacency[min(a,b),max(a,b)].keys() & witnesses)
            xyz=tri[sorted(neighbors)].reshape(-1,3)
            envelope_all.append((None,keys(xyz.min(axis=0)),keys(xyz.max(axis=0)),int(i),()))
        envelope=clear._float32_skin_aabb_index(envelope_all)
        offset=0
        for key,_,p,f in READER["primitives"](m,bodies):
            if key==READER["SKIN"]:
                continue
            chosen,_=clear._candidate_target_rows_for_skin_aabb(f,p,envelope)
            rows=positioned_records(p,f,chosen)
            records.extend((a,b,c,offset+i,d) for a,b,c,i,d in rows)
            original=tables.pop(key)
            for a,b in original["triangle_pairs"]:
                if a in witnesses:
                    existing[a].add(b+offset)
            offset+=len(f)
    assert not tables
    geometry=Geometry.__new__(Geometry)
    geometry.points=points
    geometry.tree=boxes
    geometry.movable=witnesses
    geometry.edits={}
    geometry.baseline=set(map(tuple,audit["self_intersections"]["triangle_pairs"]))
    geometry.self_hits=geometry.baseline.copy()
    geometry.targets=(exact._prepare_surface_aabb(records),existing)
    baseline_by_face=exact._audit_record_pairs(
        exact._query_surface_aabb((boxes.records[i] for i in movable),geometry.targets[0]),
        same_surface=False)["triangle_pairs"]
    assert set(map(tuple,baseline_by_face))=={(a,b) for a,bs in existing.items() for b in bs}
    return geometry


def quality(points, old, new):
    old,new=points[old].astype(float),points[new].astype(float)
    norm=lambda x: np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])
    left,right=norm(old),norm(new)
    base=left.sum(axis=0)
    n=np.linalg.norm(base)
    areas=np.linalg.norm(left,axis=1)
    if not n or np.min(areas)<=0:
        return False
    proj=right@base/n
    if np.min(proj)<=.42*np.min(areas):
        return False
    ratio=np.linalg.norm(right,axis=1).sum()/areas.sum()
    angle=(right@base)/(n*np.linalg.norm(right,axis=1))
    return .92<=ratio<=1.08 and np.min(angle)>.58


def main():
    install_for_float32()
    launch=json.loads((LATE/"launch.json").read_text())
    path=Path(launch["argv"][launch["argv"].index("--skin-payload")+1])
    assert hash_file(path)==launch["pins"][str(path)]
    model=decode(path.read_bytes())
    source=model["verts"][:,:3].copy()
    original=model["inds"]
    faces=original.copy()
    protected=set(np.load(B.parent/"resting-clearance-and-atrium-004/bed-section-pins.npz")["vertices"])
    cases=[json.loads(r.read_text()) for _,r in CASES]
    uses=topology(faces)
    for (a,b),half in uses.items():
        if len(half)!=2:
            protected.update((a,b))
    protected.update(clear._exact_coordinate_tied_source_vertex_ids(source))
    seed_ids=set()
    for report in cases:
        seed_ids.update(x for pair in report["self_intersections"]["triangle_pairs"] for x in pair)
        for bank in report["surfaces"]:
            seed_ids.update(a for a,_ in bank["triangle_pairs"])
    active=set(seed_ids)
    for i in seed_ids:
        for a,b,_ in edges(faces[i]):
            active.update(uses[min(a,b),max(a,b)])
    active=np.array(sorted(active),dtype=np.int64)
    source_records=positioned_records(source,faces,np.arange(len(faces)))
    source_tree=exact._prepare_surface_aabb(source_records)
    source_hits=set()
    seen=set()
    for a,b in exact._query_surface_aabb((source_tree.records[i] for i in active),source_tree):
        i,j=sorted((a[3],b[3]))
        if i!=j:
            seen.add((i,j))
    report=exact._audit_record_pairs(((source_tree.records[i],source_tree.records[j])
                    for i,j in sorted(seen)),same_surface=True)
    source_hits.update(map(tuple,report["triangle_pairs"]))
    controls=[Geometry(source,faces,source_hits,active)]
    for (pack,_),audit in zip(CASES,cases):
        geom=prepare_case(pack,audit,faces,active)
        controls.append(geom)
        print("prepared",audit["accepted_step"],"self",len(geom.self_hits),
              "tissue",sum(map(len,geom.targets[1].values())),flush=True)
    fixed=set()
    edits=[]
    base_self=[len(g.self_hits) for g in controls]
    for i in sorted(seed_ids):
        if i in fixed or protected.intersection(faces[i]):
            continue
        best=None
        for a,b,c in edges(faces[i]):
            pair=uses[min(a,b),max(a,b)]
            if len(pair)!=2:
                continue
            j=next(k for k in pair if k!=i)
            if j in fixed or protected.intersection(faces[j]):
                continue
            u,v,d=pair[j]
            if (u,v)!=(b,a) or (min(c,d),max(c,d)) in uses:
                continue
            proposed=np.asarray([(c,a,d),(d,b,c)],dtype=faces.dtype)
            ids=np.asarray((i,j))
            old=faces[ids]
            if not all(quality(g.points,old,proposed) for g in controls):
                continue
            tested=[g.inspect(ids,proposed) for g in controls]
            if any(r is None for r in tested):
                continue
            result=[(len(r[1]),r[2]) for r in tested]
            rank=(sum(x for x,_ in result[1:]),sum(y for _,y in result[1:]))
            if rank==(0,0):
                continue
            if best is None or rank>best[0]:
                best=(rank,(a,b,c,d),ids,proposed.copy(),tested,result)
        if best is None:
            continue
        _,(a,b,c,d),ids,proposed,tested,result=best
        for k in ids:
            for x,y,_ in edges(faces[k]):
                uses[min(x,y),max(x,y)].pop(k)
        for k,face in zip(ids,proposed):
            for x,y,z in edges(face):
                uses[min(x,y),max(x,y)][k]=(x,y,z)
        faces[ids]=proposed
        for g,r in zip(controls,tested):
            g.accept(r)
        fixed.update(ids.tolist())
        edits.append({"faces":ids.tolist(),"old_diagonal":[a,b],"new_diagonal":[c,d],
                      "removals_source_and_frames":result})
    assert len(fixed)==2*len(edits)
    assert not protected.intersection(faces[np.flatnonzero(np.any(faces!=original,axis=1))].ravel())
    out=dict(source_sha256=hash_file(path),face_ancestry="The two original rows retain their connected quad identity and directed boundary",
             pose_sha256=[r["accepted_pack_sha256"] for r in cases],
             protected_source_vertices=len(protected),candidate_steps=len(edits),edits=edits,
             self_before=base_self,self_after=[len(g.self_hits) for g in controls],
             remaining_pose_self_pairs=[sorted(g.self_hits) for g in controls[1:]],
             introduced_pairs=0,native_qualified=False)
    if edits:
        candidate=HERE/"connected-skin-diagonals.nhskin"
        raw=model["raw"]
        start=model["indexoff"]
        new_indices=faces.astype("<u4").tobytes()
        assert len(faces)==len(original)
        payload=raw[:start]+new_indices+raw[start+len(new_indices):]
        decoded=decode(payload)
        assert np.array_equal(decoded["inds"],faces)
        assert np.array_equal(decoded["verts"],model["verts"])
        assert np.array_equal(decoded["weights"],model["weights"])
        candidate.write_bytes(payload)
        out["candidate_sha256"]=hash_file(candidate)
    (HERE/"connected-skin-diagonals.json").write_text(json.dumps(out,indent=2)+"\n")
    print({k:v for k,v in out.items() if k not in ("edits","remaining_pose_self_pairs")},flush=True)


if __name__=="__main__":
    main()
