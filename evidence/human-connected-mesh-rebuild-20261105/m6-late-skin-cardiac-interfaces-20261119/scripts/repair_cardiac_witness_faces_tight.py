"""Bounded 2-to-2 face repair on the original chamber/wall source rows.

No vertex, binding, or owner is changed. Only the author's exact witness
poses are optimized here. Full range, wall containment, polynomial parity,
and runtime gates still apply to any proposed asset.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import json
import ast
import itertools
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from numilab_human import cardiac_cavity_intersections as exact

from prepare_cardiac_interval import load, prove_closed, sha

HERE=Path(__file__).resolve().parent

def edge(a,b):
    return tuple(sorted((int(a),int(b))))

class AuditPose:
    def __init__(self, points, baseline):
        self.points=points.astype(float)
        self.centers=self.points[baseline].mean(axis=1)
        self.tree=cKDTree(self.centers)
        self.max_radius=float(np.linalg.norm(self.points[baseline]-self.centers[:,None,:],axis=2).max())
        self.exact_vertex=lru_cache(maxsize=65536)(lambda i: exact.float32_point_lattice_key(self.points[i]))
        self.record=lru_cache(maxsize=100000)(self._record)

    def _record(self, row, ids):
        tri=tuple(self.exact_vertex(i) for i in ids)
        return tri,tuple(map(min,zip(*tri))),tuple(map(max,zip(*tri))),row,ids

    def touching(self, faces, changed, replacements):
        # Include every moved row separately, since the tree indexes original
        # centers. The global maximum original radius makes the query conservative.
        all_changed=set(changed)|set(replacements)
        pairs=[]
        candidates=set()
        for i,tri in replacements.items():
            p=self.points[tri]
            center=p.mean(axis=0)
            radius=float(np.linalg.norm(p-center,axis=1).max())
            assert np.isfinite(radius)
            pool=set(self.tree.query_ball_point(center, radius+self.max_radius+1.e-12))
            pool.update(all_changed)
            candidates.update(edge(i,k) for k in pool if k!=i)
        def at(i):
            return self.record(i,tuple(map(int,replacements.get(i,faces[i]))))
        for i,j in candidates:
            first,second=at(i),at(j)
            if all(a<=b and c<=d for a,b,c,d in zip(first[1],second[2],second[1],first[2])):
                pairs.append((first,second))
        found=exact._audit_record_pairs(pairs,same_surface=True)
        return {tuple(p) for p in found["triangle_pairs"]}

    def valid_triangles(self, old, new):
        a,b=self.points[old],self.points[new]
        def measure(x):
            ab=x[:,1]-x[:,0]
            ac=x[:,2]-x[:,0]
            n=np.cross(ab,ac)
            width=np.linalg.norm(x-np.roll(x,1,axis=1),axis=2).max(axis=1)
            return n,np.linalg.norm(n,axis=1)/width
        normal,prev=measure(a)
        nxt,height=measure(b)
        normal_sum=normal.sum(axis=0)
        return (np.all(nxt @ normal_sum > 0) and
                bool(np.all(height>=max(1.1e-7, .5*prev.min()))))

# Reuse the established determinant polynomial and precision reserve without
# importing its full NHANATOMY CLI dependency tree.
owner = HERE/"retained_audit_cardiac_flip_volume.py"
parsed = ast.parse(owner.read_bytes(), str(owner))
selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and
            node.name in ("tetra_volume_delta_polynomial","relative_volume_bounds")]
assert len(selected)==2
exec(compile(ast.Module(body=selected,type_ignores=[]),str(owner),"exec"))

def conservative_volume_bound(raw, ids):
    coefficient,products=tetra_volume_delta_polynomial(*raw[list(ids)].astype(float))
    return relative_volume_bounds(coefficient,products,1.0)[0]

def main():
    source=(HERE/"common-cardiac-input.bin")
    n,nf,nw=map(int,np.fromfile(source,"<u4",count=3))
    raw=np.fromfile(source,"<f4",offset=12,count=24*n).reshape(n,8,3)
    faces=np.fromfile(source,"<u4",offset=12+96*n,count=nf*3).reshape(nf,3)
    ancestry=np.load(HERE/"common-cardiac-face-owners.npz")
    owner=ancestry["owner"]
    face_row=ancestry["face_row"]
    baseline=json.loads((HERE/"common-cardiac-metal-witness-audit.json").read_text())
    original_hits=[{tuple(pair) for pair in r["compact_witnesses"]["triangle_pairs"]} for r in baseline["states"]]
    current=[set(p) for p in original_hits]

    surfaces,coeff,field,_=load()
    regions=[r["stable_id"] for r in field["vertex_ranges"]]
    reference=dict(zip(regions,field["source_reference_volumes_m3"]))
    # Determine whether any source face survived the quotient as a shared
    # facet. Such a facet never flips unilaterally.
    all_positions=np.concatenate([surfaces[k]["vertices6"][:,:3] for k in regions])
    recovered, inverse=np.unique(all_positions,axis=0,return_inverse=True)
    assert np.array_equal(recovered,raw[:,0])
    rows={}
    for spec in field["vertex_ranges"]:
        k,start,count=(spec[s] for s in ("stable_id","first_vertex","vertex_count"))
        rows[k]=inverse[start:start+count][surfaces[k]["faces"]]
    joined=np.concatenate([rows[k] for k in regions])
    _,i,c=np.unique(np.sort(joined,axis=1),axis=0,return_inverse=True,return_counts=True)
    facet_set={tuple(row) for row in np.sort(joined[c[i]>1],axis=1)}
    witnessed=np.unique(np.asarray([p for region in current for pair in region for p in pair],dtype=int))
    index_by_owner=defaultdict(dict)
    for index in range(len(faces)):
        index_by_owner[int(owner[index])][int(face_row[index])]=index
    initial=dict()
    source_faces_by_vertex=defaultdict(set)
    for i in witnessed:
        for vertex_id in faces[i]:
            source_faces_by_vertex[(int(owner[i]),int(vertex_id))].add(int(i))

    # Build adjacency only around the measured patch vertices, retaining
    # ownership boundaries and links to the original stable source rows.
    for k,f in rows.items():
        targets={v for a,v in source_faces_by_vertex if a==k}
        nearby=np.flatnonzero(np.isin(f,list(targets)).any(axis=1))
        mentions=defaultdict(list)
        for source_row in nearby:
            for i,j in ((0,1),(1,2),(2,0)):
                mentions[edge(f[source_row,i],f[source_row,j])].append(source_row)
        touched=[int(i) for i in witnessed if owner[i]==k]
        for i in touched:
            tri=faces[i]
            if tuple(sorted(tri)) in facet_set:
                continue
            for u,v in zip(tri,np.roll(tri,-1)):
                e=edge(u,v)
                around=mentions[e]
                assert len(around)==2
                adj=next(int(a) for a in around if a!=face_row[i])
                j=index_by_owner[k].get(adj)
                if j is None or tuple(sorted(faces[j])) in facet_set:
                    continue
                x=int(next(a for a in tri if a not in e))
                y=int(next(a for a in faces[j] if a not in e))
                diagonal=edge(x,y)
                if len(set((int(u),int(v),x,y)))!=4 or mentions.get(diagonal):
                    continue
                initial[edge(i,j)]=({i:np.asarray((x,u,y)),j:np.asarray((y,v,x))},
                                    k,tuple(map(int,(u,v,x,y))))
    shared_triangles=np.asarray(sorted(facet_set),dtype=np.uint32)
    assert len(shared_triangles)>0
    del all_positions,recovered,joined,i,c
    probes=[]
    metal=HERE/"common-cardiac-metal.bin"
    with metal.open("rb") as file:
        for index in range(nw):
            record=int(np.frombuffer(file.read(4),"<u4")[0])
            assert record==index
            xyz=np.frombuffer(file.read(12*n),"<f4").reshape(n,3)
            probes.append(AuditPose(xyz,faces))
    probes.append(AuditPose(raw[:,0],faces))
    changed={}
    vbound_by_proposal={k:conservative_volume_bound(raw,v[2]) for k,v in initial.items()}
    volume_budgets=Counter()
    decisions=[]
    selected_edges=set()
    started=time.monotonic()
    for count in range(32):
        best=None
        for key,(replacements,region,vertex_ids) in initial.items():
            if changed.keys() & replacements.keys() or any(v in selected_edges for v in vertex_ids):
                continue
            if time.monotonic()-started>150:
                break
            vbound=vbound_by_proposal[key]
            if volume_budgets[region]+vbound>reference[region]*2e-7:
                continue
            keys=list(replacements)
            baseline_tri=faces[keys]
            trial=np.asarray([replacements[k] for k in keys])
            if not all(pose.valid_triangles(baseline_tri,trial) for pose in probes):
                continue
            # Existing witnesses are kept in exact published row IDs.
            evidence=[]
            eligible=True
            effective=faces
            # The unchanged table is copied once per accepted operation below.
            for pose,pairs in zip(probes,current):
                old={p for p in pairs if set(p) & replacements.keys()}
                new=pose.touching(effective,changed,replacements)
                if new-old:
                    eligible=False
                    break
                evidence.append((old,new))
            if not eligible:
                continue
            if probes[-1].touching(effective,changed,replacements)-probes[-1].touching(
                    effective,changed,{i:effective[i] for i in replacements}):
                continue
            gain=sum(len(old)-len(new) for old,new in evidence)
            if gain>0 and (best is None or gain>best[0]):
                best=(gain,replacements,region,vertex_ids,vbound,evidence)
        if best is None:
            break
        gain,replacements,region,vertex_ids,vbound,evidence=best
        if not changed:
            faces=faces.copy()
        for i,face in replacements.items():
            faces[i]=face
            changed[i]=face.tolist()
        for pairs,(old,new) in zip(current,evidence):
            pairs.difference_update(old)
            pairs.update(new)
        selected_edges.update(vertex_ids)
        volume_budgets[region]+=vbound
        operation=dict(removed_across_probes=gain,stable_id=region,
            source_rows={int(face_row[i]):v.tolist() for i,v in replacements.items()},
            volume_bound_m3=vbound)
        decisions.append(operation)
        print(operation,flush=True)
    result=dict(source_sha256=sha(source),gpu_positions_sha256=sha(metal),
        proposals=len(initial),pose_pair_counts_before=list(map(len,original_hits)),
        pose_pair_counts_after_predicted=list(map(len,current)),
        source_replacement_rows=decisions,
        relative_volume_upper_bounds={k:v/reference[k] for k,v in volume_budgets.items()},
        adopted=False,scope="Three Metal local pose checks and source at zero. Full continuous geometry, wall containment and native load qualification pending.")
    (HERE/"cardiac-interface-repair-tight.json").write_text(json.dumps(result,indent=2)+"\n")
    if changed:
        raw_file=(HERE/"common-cardiac-input.bin").read_bytes()
        begin=12+96*n
        (HERE/"common-cardiac-repaired-tight-probe.bin").write_bytes(raw_file[:begin]+faces.tobytes()+raw_file[begin+12*nf:])
    print({k:v for k,v in result.items() if k!="source_replacement_rows"},flush=True)
if __name__=="__main__":
    main()
