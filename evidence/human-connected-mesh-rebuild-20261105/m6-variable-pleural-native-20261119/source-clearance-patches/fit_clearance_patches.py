"""Backtrack source-patch clearance against all typed boundaries on retained poses."""
from collections import defaultdict
from functools import lru_cache
import json

import numpy as np
from scipy.sparse import coo_matrix, csgraph, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from fit_joint_surface_registration import axes, decode, mesh_edges
from register_tissue_bellies import CASES, HERE, READER, SKIN, capture, clear, hash_file
from classify_native_crossing_geometry import audit_integer
from numilab_human import cardiac_cavity_intersections as owner

SELF = (0, 0)
MAX_MOVE = .004
MARGIN = .00006


def lattice(points):
    return tuple(owner.float32_point_lattice_key(p) for p in points)


def exact_intersection(left, right, same, shared):
    normals = [owner._cross(owner._sub(t[1], t[0]), owner._sub(t[2], t[0])) for t in (left, right)]
    if any(not any(n) for n in normals):
        raise ValueError("degenerate authored face")
    for a, b, n in ((left, right, normals[0]), (right, left, normals[1])):
        z = [owner._dot(n, owner._sub(p, a[0])) for p in b]
        nonzero = [v for v in z if v]
        if len(nonzero) == 3-len(shared) and nonzero and (min(nonzero)>0 or max(nonzero)<0):
            return False
    ps = audit_integer(left, right)
    if not ps:
        return False
    return not (same and len(shared) in (1,2) and all(owner._allowed_shared_point(p,shared) for p in ps))


def area(p):
    return np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])


def oriented(initial, trial):
    n0,n1=area(initial.astype(float)),area(trial.astype(float))
    return bool(np.all(np.einsum("ij,ij->i",n0,n1)>.675*np.einsum("ij,ij->i",n0,n0)))


def indexed_triangles(points, faces):
    tri=points[faces]
    lo=tri.min(axis=1)
    hi=tri.max(axis=1)
    center=(lo.astype(float)+hi)*.5
    radii=np.linalg.norm(hi.astype(float)-center,axis=1)
    return tri,center,radii,cKDTree(center)


class Pose:
    def __init__(self, path, skin, ref_faces, banks):
        faces=skin["inds"]
        used=np.unique(faces)
        with capture(path) as (m,bodies,receipt):
            self.step=receipt["accepted_step"]
            self.sha=receipt["pack_file_sha256"]
            self.j=np.einsum("vb,bce->vce",skin["weights"],axes(skin,receipt)[0])
            self.skin=np.zeros((len(skin["verts"]),3),dtype="<f4")
            primitive=list(READER["primitives"](m,bodies))
            self.banks=[(key,p[f]) for key,_,p,f in primitive if key!=SKIN]
            emitted,ef=next((p,f) for k,_,p,f in primitive if k==SKIN)
            assert np.array_equal(ef,np.searchsorted(used,ref_faces))
            self.skin[used]=emitted
        self.faces=faces
        self.init=self.skin.copy()
        self.seeds=banks
        self.tissue=np.concatenate([p for _,p in self.banks]).astype("<f4",copy=False)
        self.bank_keys=[k for k,_ in self.banks]
        self.bank_base=np.r_[0,np.cumsum([len(p) for _,p in self.banks])]
        c=(self.tissue.min(axis=1).astype(float)+self.tissue.max(axis=1))*.5
        radius=np.linalg.norm(self.tissue.max(axis=1).astype(float)-c,axis=1)
        self.terrain_bins=[]
        for bucket in np.unique(np.frexp(radius)[1]):
            rows=np.flatnonzero(np.frexp(radius)[1]==bucket)
            self.terrain_bins.append((cKDTree(c[rows]),rows,radius[rows].max()))
        self.tissue_low=self.tissue.min(axis=1)
        self.tissue_high=self.tissue.max(axis=1)
        _,_,r,self.skin_tree=indexed_triangles(self.skin,self.faces)
        self.skin_radius=r.max()
        self.lattice_target=lru_cache(maxsize=12000)(lambda j:lattice(self.tissue[j]))

    @classmethod
    def source_pose(cls, points, faces):
        pose=cls.__new__(cls)
        pose.skin=points.copy()
        pose.init=points.copy()
        pose.faces=faces
        pose.terrain_bins=[]
        _,_,r,pose.skin_tree=indexed_triangles(points,faces)
        pose.skin_radius=r.max()
        return pose

    def targets_for_patch(self, face_ids, tri):
        # Surface triangles indexed at build remain within MAX_MOVE per vertex.
        # Radius padding accounts for all source and transformed displacement.
        exact_set=defaultdict(set)
        lows,highs=tri.min(1),tri.max(1)
        c=(lows.astype(float)+highs)*.5
        radius=np.linalg.norm(highs.astype(float)-c,axis=1)
        tb=[[] for _ in range(len(c))]
        for index,ids,radius_cap in self.terrain_bins:
            for dest,rows in zip(tb,index.query_ball_point(c,radius+radius_cap+1.e-9)):
                if rows:
                    dest.extend(ids[rows])
        sb=self.skin_tree.query_ball_point(c,radius+self.skin_radius+MAX_MOVE+1.e-9)
        row_index={i:k for k,i in enumerate(face_ids)}
        for q,i in enumerate(face_ids):
            lo,hi=lows[q],highs[q]
            left=None
            for ids,kind in ((tb[q],"tissue"),(sb[q],"skin")):
                ids=np.asarray(ids,dtype=int)
                if kind=="tissue":
                    if not len(ids):
                        continue
                    keep=np.all(self.tissue_high[ids]>=lo,axis=1)&np.all(self.tissue_low[ids]<=hi,axis=1)
                    ids=ids[keep]
                    x=self.tissue[ids]
                else:
                    ids=ids[ids!=i]
                    x=self.skin[self.faces[ids]]
                    rows=[(jj,row_index[j]) for jj,j in enumerate(ids) if j in row_index]
                    if rows:
                        a,b=zip(*rows); x[list(a)]=tri[list(b)]
                mask=np.all(x.max(axis=1)>=lo,axis=1)&np.all(x.min(axis=1)<=hi,axis=1)
                for j,b in zip(ids[mask],x[mask]):
                    if left is None:
                        left=lattice(tri[q])
                    if kind=="skin":
                        shared_ids=np.intersect1d(self.faces[i],self.faces[j])
                        shared={left[np.flatnonzero(self.faces[i]==idx)[0]] for idx in shared_ids}
                        if exact_intersection(left,lattice(b),True,shared):
                            exact_set[SELF].add(tuple(sorted((int(i),int(j)))))
                    elif exact_intersection(left,self.lattice_target(int(j)),False,set()):
                        k=np.searchsorted(self.bank_base,j,side="right")-1
                        exact_set[self.bank_keys[k]].add((int(i),int(j-self.bank_base[k])))
        return exact_set

    def witness_support(self, key, pairs, ids, directions):
        if not pairs:
            return []
        bank=self.bank_keys.index(key)
        s,t=np.asarray(sorted(pairs),dtype=int).T
        triangles=self.faces[s]
        data=[]
        for a,b in zip(triangles,self.tissue[self.bank_base[bank]+t]):
            n=area(self.skin[a][None].astype(float))[0]
            n/=np.linalg.norm(n)
            slope=np.einsum("vr,r->v", np.einsum("vij,vj->vi",self.j[a],directions[a]),n)
            gap=((b-self.skin[a[0]])@n).max()-((self.skin[a]-self.skin[a[0]])@n)
            for vertex,limit,m in zip(a,gap,slope):
                if m>.25 and limit>0. and ids[vertex]:
                    data.append((vertex,min((limit+MARGIN)/m,MAX_MOVE)))
        return data


def main():
    path=HERE/"connected-skin-diagonals.nhskin"
    skin=decode(path.read_bytes())
    p=skin["verts"][:,:3]
    f=skin["inds"]
    ref=decode((HERE.parent/"resting-clearance-and-atrium-005/trimmed-guarded-pressure-skin.nhskin").read_bytes())["inds"]
    edges,counts=mesh_edges(f)
    pinned=np.zeros(len(p),dtype=bool)
    pinned[edges[counts!=2]]=True
    pinned[clear._exact_coordinate_tied_source_vertex_ids(p)]=True
    pinned[np.load(HERE.parent/"resting-clearance-and-atrium-004/bed-section-pins.npz")["vertices"]]=True
    normals=np.zeros_like(p,dtype=float)
    for i in range(3):
        np.add.at(normals,f[:,i],area(p[f].astype(float)))
    norm=np.linalg.norm(normals,axis=1)
    assert np.all(norm[np.unique(f)]>0)
    normals/=np.maximum(norm[:,None],1.e-25)
    direction=(normals + skin["verts"][:,3:6])*.5
    direction/=np.maximum(1.e-25,np.linalg.norm(direction,axis=1)[:,None])
    raw=HERE/"connected-skin-native"
    terminal=raw/"native/accepted-geometry/step-8000.mrvpack"
    poses=[]
    keys=set()
    for pack,report in (*CASES[:2],(terminal,raw/"skin-exact.json")):
        frozen=json.loads(report.read_text())
        pairs={(b["semantic"],b["stable_id"]):set(map(tuple,b["triangle_pairs"]))
               for b in frozen["surfaces"] if b["count"]}
        pose=Pose(pack,skin,f if pack==terminal else ref,pairs)
        assert pose.sha==frozen["accepted_pack_sha256"]
        keys.update(pairs)
        poses.append(pose)
        print("pose",pose.step, "typed banks", len(pose.banks), flush=True)
    # For the 0 and 4 second samples only index unchanged old-face witnesses.
    # Local narrow phase always measures the actual candidate connectivity.
    swapped=np.any(f!=ref,axis=1)
    for pose in poses[:-1]:
        pose.seeds={k:{(i,j) for i,j in pairs if not swapped[i]} for k,pairs in pose.seeds.items()}
    weights=1./np.maximum(np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)*1000,.1)
    a,b=edges.T
    graph=coo_matrix((np.r_[weights,weights],(np.r_[a,b],np.r_[b,a])),shape=(len(p),len(p))).tocsr()
    lap=csgraph.laplacian(graph).tocsr()
    all_delta=np.zeros_like(p,dtype=float)
    frozen_keys=sorted(keys,key=lambda k:(len(poses[-1].seeds.get(k,()))==0,
                                            len(poses[-1].seeds.get(k,())),k))
    accepted=[]
    source_pose=Pose.source_pose(p,f)
    previous=HERE/"source-clearance-patches-frozen-only.json"
    if previous.exists():
        saved={tuple(r["bank"]) for r in json.loads(previous.read_text())["accepted"]}
        frozen_keys=[key for key in frozen_keys if key in saved]
    for key in frozen_keys:
        touched=np.zeros(len(p),bool)
        target=[pose.seeds.get(key,set()) for pose in poses]
        for pairs in target:
            touched[f[[i for i,_ in pairs]]]=True
        touched[pinned]=False
        if not touched.any():
            continue
        local=touched.copy()
        for _ in range(3):
            local[edges[local[edges].any(axis=1)].ravel()]=True
        local[pinned]=False
        rows=np.flatnonzero(local)
        affected=np.flatnonzero(local[f].any(axis=1))
        req=np.zeros(len(p),dtype=float)
        for pose,pairs in zip(poses,target):
            for v,d in pose.witness_support(key,pairs,local,direction):
                req[v]=max(req[v],d)
        if not np.any(req>0.):
            continue
        anchors=(req[rows]>0.)*15.
        smooth=lap[rows][:,rows]+diags(anchors+.35)
        amplitude=spsolve(smooth,anchors*req[rows])
        d=direction[rows]*amplitude[:,None]
        if not np.all(np.isfinite(d)):
            raise ValueError("invalid local displacement")
        old=[pose.targets_for_patch(affected,pose.skin[f[affected]]) for pose in poses]
        if not any(a.get(key) for a in old):
            continue
        initial_source=source_pose.targets_for_patch(affected,source_pose.skin[f[affected]])[SELF]
        for scale in (1.25,1.,.75,.4,.2):
            proposal=all_delta.copy()
            proposal[rows]+=d*scale
            if np.linalg.norm(proposal[rows],axis=1).max()>=MAX_MOVE*.99:
                continue
            posed_source=(p.astype(float)+proposal).astype("<f4")
            src=posed_source[f[affected]]
            if not oriented(p[f[affected]],src):
                continue
            new_source=source_pose.targets_for_patch(affected,src)[SELF]
            if new_source-initial_source:
                print("source guard",key,scale,sorted(new_source-initial_source)[:6],flush=True)
                continue
            staged=[]
            verdict=[]
            for pose,before in zip(poses,old):
                updated=pose.skin.copy()
                world_delta=np.einsum("vij,vj->vi",pose.j[rows],proposal[rows])
                assert np.linalg.norm(world_delta,axis=1).max()<MAX_MOVE
                updated[rows]=(pose.init[rows].astype(float)+world_delta).astype("<f4")
                mesh=updated[f[affected]]
                if not oriented(pose.init[f[affected]],mesh):
                    break
                after=pose.targets_for_patch(affected,mesh)
                if any(not a.issubset(before[k]) for k,a in after.items()):
                    break
                staged.append(updated)
                verdict.append((len(before[key]),len(after[key])))
            if len(staged)!=len(poses) or not any(a>b for a,b in verdict):
                continue
            source_pose.skin=posed_source
            all_delta=proposal
            for pose,updated in zip(poses,staged):
                pose.skin=updated
            accepted.append(dict(bank=key,scale=scale,witness_counts=verdict))
            print("accepted",accepted[-1],flush=True)
            break
    assert np.all(all_delta[pinned]==0.)
    dst=HERE/"source-clearance-patches.npz"
    np.savez_compressed(dst,displacement_m=all_delta)
    (HERE/"source-clearance-patches.json").write_text(json.dumps(dict(
        skin_sha256=hash_file(path),pose_pack_sha256=[p.sha for p in poses],
        displacement_sha256=hash_file(dst),accepted=accepted,
        maximum_displacement_m=float(np.linalg.norm(all_delta,axis=1).max()),
        protected_rows=int(pinned.sum())),indent=2)+"\n")
    print("done",len(accepted),flush=True)


if __name__=="__main__":
    main()
