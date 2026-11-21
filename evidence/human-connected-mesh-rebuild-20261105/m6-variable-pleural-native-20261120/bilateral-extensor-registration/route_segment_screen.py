"""Compare node weights to branch-local authored site-interval interpolation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

from screen import (
    banks, bindings, bind_transforms, exact, hits, indices, linear, manifest,
    neighborhood, pair_index, poses, records, registered, sha, shape, skin,
    skin_faces, vertices,
)

from numilab_human.model import _myosim_surface_route_context


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def value_weights(verts, size):
    return np.array([
        np.sum(np.where((verts["binding"]==i)&(verts["weight"]>2e-6),verts["weight"],0),axis=1)
        for i in range(size)
    ])

def retained_point_weights(x, nodes, owners, size):
    by_owner=np.full((size,len(x)),np.inf)
    for point, owner in zip(nodes,owners,strict=True):
        sq=np.sum((x-point)**2,axis=1)
        np.minimum(by_owner[owner],sq,out=by_owner[owner])
    order=np.argsort(by_owner,axis=0,kind="stable")[:4]
    d=np.take_along_axis(by_owner,order,axis=0)
    w=np.reciprocal(d+9e-6)
    close=d[0]<=1e-12
    w[:,close]=np.array([[1],[0],[0],[0]])
    w/=w.sum(axis=0)
    dense=np.zeros_like(by_owner)
    np.put_along_axis(dense,order,w,axis=0)
    return dense,np.sqrt(d[0])

def segment_weights(x, matches, by_name):
    best=np.full(len(x),np.inf)
    owner_a=np.zeros(len(x),dtype=int)
    owner_b=owner_a.copy()
    fraction=np.zeros(len(x))
    branch=np.full(len(x),-1)
    available=0
    for k,route in enumerate(matches):
        points=route["route_points"]
        # A wrap's centre is not a point on the tendon. Score only site-site
        # spans with no wrap node intervening, using the source sequence.
        for left,right in zip(points,points[1:]):
            if left["kind"]!="site" or right["kind"]!="site":
                continue
            a=np.asarray(left["world_m"])
            b=np.asarray(right["world_m"])
            d=b-a
            length=d@d
            if length<=1e-18:
                continue
            available+=1
            u=np.clip((x-a)@d/length,0,1)
            error=x-a-u[:,None]*d
            value=np.sum(error*error,axis=1)
            take=value<best
            best[take]=value[take]
            owner_a[take]=by_name[left["body"]]
            owner_b[take]=by_name[right["body"]]
            branch[take]=k
            fraction[take]=u[take]
    assert available>0 and np.all(branch>=0)
    w=np.zeros((len(by_name),len(x)))
    i=np.arange(len(x))
    w[owner_a,i]=1-fraction
    w[owner_b,i]+=fraction
    return w,np.sqrt(best),branch,available

def run_for(stable, global_map, route_context, skin_index):
    bank=manifest["source"]["surfaces"]
    descriptor=next(row for row in bank if row["stable_id"]==stable)
    match=[route_context[m["name"]] for m in descriptor["matched_muscles"]]
    assert all(m["source_actuator_index"]==r["source_actuator_index"] and
               m["source_route_node_count"]==r["source_route_node_count"]
               for m,r in zip(descriptor["matched_muscles"],match,strict=True))
    assert len(match)==4
    b0,nb,v0,nv,i0,ni,_,_=map(int,records[records[:,6]==stable][0])
    bs=bindings[b0:b0+nb]
    raw=(indices[i0:i0+ni]-v0).reshape(-1,3)
    used=np.unique(raw)
    faces=np.searchsorted(used,raw)
    assert np.all(used<nv)
    vs=vertices[v0:v0+nv][used]
    x=vs["position"].astype(float)
    w=value_weights(vs,nb)
    by_name={entry["myosim_body"]:i for i,entry in enumerate(descriptor["body_bindings"])}
    assert [b["core_body_index"] for b in descriptor["body_bindings"]]==bs["body"].tolist()
    world=x @ (1000*global_map[:3,:3]).T + global_map[:3,3]
    nodes=[point for route in match for point in route["route_points"]]
    old_w,d_node=retained_point_weights(world,np.array([p["world_m"] for p in nodes]),
                                       [by_name[p["body"]] for p in nodes],nb)
    recipe_diff=np.abs(w-old_w).sum(axis=0)
    route_w,d_seg,branch,num_spans=segment_weights(world,match,by_name)
    terminals=np.array([p["world_m"] for r in match for p in (r["route_points"][0],r["route_points"][-1])])
    terminal_distance=cKDTree(terminals).query(world)[0]
    pinned=(w.max(axis=0)>1-2e-6)|(terminal_distance<=.003)
    # Preserve each surface vertex closest to an authored terminal, even if
    # the asset's mixed-source geometry is outside the 3 mm end band.
    pinned[cKDTree(world).query(terminals)[1]]=True
    protection=np.clip((terminal_distance-.003)/.009,0,1)
    protection=protection**2*(3-2*protection)
    protection[pinned]=0
    native,native_faces=banks[51005,stable]
    assert np.array_equal(faces,native_faces)
    transforms=[]
    for use_registered,p in ((True,registered),(False,poses)):
        if not all(int(b["body"]) in p for b in bs):
            continue
        scale,q,t=bind_transforms(bs,use_registered=use_registered)
        reconstructed=linear(x,w,scale,q,t)
        error=float(np.linalg.norm(reconstructed-native,axis=1).max())
        transforms.append((error,scale,q,t,reconstructed))
    error,scale,q,t,reconstructed=min(transforms,key=lambda row:row[0])
    assert error<=2e-6
    control=hits(skin_index,native,faces)
    subjects=np.unique(faces[np.array(sorted(f for _,f in control),dtype=int)])
    original_index=pair_index(native,faces)
    old_self=set(map(tuple,exact._audit_pair_prepared_first(original_index[0],original_index[2],same_surface=True)["triangle_pairs"]))
    result={"stable_id":stable,"matched_routes":[m["name"] for m in descriptor["matched_muscles"]],
            "native_pairs":len(control),"native_self_pairs":len(old_self),
            "source_vertex_count":nv,"referenced_vertex_count":len(used),
            "compiler_recipe_max_weight_difference_l1":float(recipe_diff.max()),
            "compiler_recipe_vertices_within_1e_3":int((recipe_diff<.001).sum()),
            "protected_vertex_count":int(pinned.sum()),"witness_vertex_count":len(subjects),
            "maximum_roundtrip_error_m":error,"authored_site_intervals":num_spans,
            "max_node_distance_mm":float(1000*d_node.max()),
            "max_interval_distance_mm":float(1000*d_seg.max()),
            "median_witness_node_distance_mm":float(1000*np.median(d_node[subjects])),
            "median_witness_interval_distance_mm":float(1000*np.median(d_seg[subjects])),
            "witness_interval_branch_counts":np.bincount(branch[subjects],minlength=len(match)).tolist(),
            "trials":[]}
    for fraction in (1.,.5,.25,.125,.0625):
        blend=protection*fraction
        new_weights=w*(1-blend)+route_w*blend
        order=np.argsort(-new_weights,axis=0,kind="stable")[4:]
        np.put_along_axis(new_weights,order,0,axis=0)
        mask=blend>0
        new_weights[:,mask]/=new_weights[:,mask].sum(axis=0)
        delta=linear(x,new_weights,scale,q,t)-reconstructed
        assert np.all(delta[pinned]==0)
        proposal=(native.astype(float)+delta).astype("<f4")
        new=hits(skin_index,proposal,faces)
        metrics=shape(native,proposal,faces)
        displacement=float(1000*np.linalg.norm(delta,axis=1).max())
        quality=(displacement<=3 and metrics["min_area_ratio"]>=.8 and
                 metrics["max_area_ratio"]<=1.25 and metrics["minimum_face_normal_cosine"]>0 and
                 metrics["relative_signed_volume_change"] is not None and
                 metrics["relative_signed_volume_change"]<=.01)
        stage=dict(scale=fraction,pairs=len(new),cleared=len(control-new),added=len(new-control),
                   max_delta_mm=displacement,passes_existing_shape_limits=bool(quality),**metrics)
        if quality and len(new)<len(control):
            surface=pair_index(proposal,faces)
            self_pairs=set(map(tuple,exact._audit_pair_prepared_first(surface[0],surface[2],same_surface=True)["triangle_pairs"]))
            stage["new_self_pairs"]=len(self_pairs-old_self)
            stage["neighbor_changes"]=neighborhood(native,proposal,faces,stable)
        result["trials"].append(stage)
    return result

if __name__=="__main__":
    source=myosim=Path(sys.argv[2]).resolve()
    config=Path(sys.argv[3]).resolve()
    expected=manifest["source"]
    assert digest(myosim/expected["myosim_manifest"]["file"])==expected["myosim_manifest"]["sha256"]
    assert digest(config)==expected["registration"]["sha256"]
    _, routes, _=_myosim_surface_route_context(source,expected["myosim_source_archive_sha256"])
    matrix=np.array(json.loads(config.read_text())["coordinate_system"]["global_source_mm_to_myosim_world_m"])
    index=pair_index(skin,skin_faces)
    results=[run_for(i,matrix,routes,index) for i in (137,138)]
    print(json.dumps({"nhtissue_sha256":sha,"myosim_artifact":str(source),
                      "registration":str(config),"surfaces":results},indent=2))
