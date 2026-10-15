#!/usr/bin/env python3
"""Offline bounded skin contact preparation; existing exact geometry owns gates."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from numilab_human import model as h
from numilab_human import resting_anatomy_verify as v
from numilab_human import skin_source_payload_preflight as sp
from numilab_human import gate_preparation_boundary as gp

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def barycentric(triangle, point):
    a, b, c = triangle
    u = [b[i]-a[i] for i in range(3)]
    z = [c[i]-a[i] for i in range(3)]
    p = [point[i]-a[i] for i in range(3)]
    n = [u[1]*z[2]-u[2]*z[1], u[2]*z[0]-u[0]*z[2], u[0]*z[1]-u[1]*z[0]]
    axis = max(range(3), key=lambda i: abs(n[i]))
    i, j = [i for i in range(3) if i != axis]
    det = u[i]*z[j]-u[j]*z[i]
    if not det:
        raise ValueError("exact degenerate contact face")
    wb = (p[i]*z[j]-p[j]*z[i])/det
    wc = (u[i]*p[j]-u[j]*p[i])/det
    result = [1-wb-wc, wb, wc]
    if not all(Fraction(0) <= x <= Fraction(1) for x in result):
        raise ValueError("exact witness outside its source face")
    return np.array([float(x) for x in result])

def projected_area_gradient(triangle, normal):
    e1=triangle[1]-triangle[0]
    e2=triangle[2]-triangle[0]
    gb=np.cross(e2,normal)
    gc=np.cross(normal,e1)
    return np.array([-gb-gc,gb,gc])

def ray_box(point,direction,lower,upper):
 entry=Fraction(0);exit=None
 for axis,component in enumerate(direction):
  if not component:
   if not Fraction(lower[axis])<=point[axis]<=Fraction(upper[axis]):return False
   continue
  first=(Fraction(lower[axis])-point[axis])/component
  last=(Fraction(upper[axis])-point[axis])/component
  if first>last:first,last=last,first
  if last<0:return False
  entry=max(entry,first);exit=last if exit is None else min(exit,last)
  if entry>exit:return False
 return True
def signed_candidates(point,index,direction):
 stack=[index.aabb_root];result=[]
 while stack:
  lower,upper,indices,left,right=stack.pop()
  if not ray_box(point,direction,lower,upper):continue
  if indices is None:stack.extend([left,right])
  else:result.extend(indices)
 return sorted(result)
def exact_forward_exit(surface,index,point,direction):
    exact_point=tuple(Fraction(x) for x in point)
    exact_direction=tuple(Fraction(x) for x in direction)
    def cross(a,b):
        return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def dot(a,b):
        return sum((x*y for x,y in zip(a,b)),Fraction(0))
    nearest_t=None
    nearest_face=None
    for fi in signed_candidates(exact_point,index,exact_direction):
        a,b,c=surface.triangle_meters(int(fi))
        u=tuple(b[j]-a[j] for j in range(3));z=tuple(c[j]-a[j] for j in range(3));n=cross(u,z)
        den=dot(n,exact_direction)
        if not den:continue
        t=dot(n,tuple(a[j]-exact_point[j] for j in range(3)))/den
        if t<=0 or (nearest_t is not None and t>=nearest_t):continue
        hit=tuple(exact_point[j]+t*exact_direction[j] for j in range(3))
        axis=max(range(3),key=lambda j:abs(n[j]));i,j=[j for j in range(3) if j!=axis]
        det=u[i]*z[j]-u[j]*z[i]
        if not det:raise ValueError("degenerate exact ray face")
        w=tuple(hit[k]-a[k] for k in range(3))
        wb=(w[i]*z[j]-w[j]*z[i])/det;wc=(u[i]*w[j]-u[j]*w[i])/det;wa=1-wb-wc
        if min(wa,wb,wc)<0:continue
        nearest_t=t;nearest_face=int(fi)
    if nearest_t is None:raise ValueError("oriented ray has no positive target exit")
    return float(nearest_t),nearest_face

def contact_forward_exit(surface,index,point,direction):
    try:
        return exact_forward_exit(surface,index,point,direction)
    except ValueError as error:
        if str(error)!="oriented ray has no positive target exit":
            raise
        location=v.ci.point_location_prepared_indexed(tuple(Fraction(x) for x in point),index)["location"]
        if location!="boundary":
            raise ValueError(f"no forward exit from non-boundary contact: {location}; point={point}") from error
        # A ray from an edge/corner may immediately leave through its other
        # incident face even when its dot with this witness face is negative.
        return 0.,None


def self_gap_constraint(base_points,faces,first,first_bary,second,second_bary,margin_m):
    """Preserve the reference side at an actual self-contact witness only."""
    reference_first=first_bary@base_points[faces[first]].astype(float)
    reference_second=second_bary@base_points[faces[second]].astype(float)
    gap=reference_first-reference_second
    length=float(np.linalg.norm(gap))
    if not np.isfinite(length) or length<=margin_m:
        raise ValueError("self-contact reference has no unambiguous separation margin")
    normal=gap/length
    return (int(first),first_bary.copy(),int(second),second_bary.copy(),
            normal,float(margin_m-length))



def face_support_constraints(base_points,faces,first,second,margin_m):
    """Preserve a complete reference triangle separator, not just witness points.

    Exact rational source projections choose and verify the axis before its
    dimensionless normalized coefficients enter the offline authoring LP.
    """
    triangles=[[[Fraction(float(x)) for x in point]
                for point in base_points[faces[index]]] for index in (first,second)]
    def sub(a,b):return [x-y for x,y in zip(a,b)]
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    a,b=triangles
    ea=[sub(a[(i+1)%3],a[i]) for i in range(3)]
    eb=[sub(b[(i+1)%3],b[i]) for i in range(3)]
    na,nb=cross(ea[0],ea[1]),cross(eb[0],eb[1])
    axes=[na,nb,*[cross(x,y) for x in ea for y in eb],
          *[cross(x,na) for x in ea],*[cross(x,nb) for x in eb]]
    shared=set(map(int,faces[first]))&set(map(int,faces[second]))
    if len(shared)==3:
        raise ValueError("self-contact reference triangles have no complete separating support plane")
    candidates=[]
    for axis in axes:
        length2=dot(axis,axis)
        if not length2:continue
        for sign in (1,-1):
            oriented=[sign*x for x in axis]
            exact_gaps=[dot(oriented,sub(x,y)) for x in a for y in b]
            gap=min(exact_gaps)
            # A shared vertex/edge is an intended zero gap. Keep all nine
            # corner inequalities, including zero bounds at that simplex.
            if gap<0 or (gap==0 and not shared):continue
            positive=[x for x in exact_gaps if x>0]
            if not positive:continue
            candidates.append((gap>0,min(positive)**2/length2,oriented))
    candidates.sort(key=lambda row:(row[0],row[1]),reverse=True)
    normal=None
    for _,_,axis in candidates:
        axis=np.array([float(x) for x in axis])
        trial=axis/np.linalg.norm(axis)
        if not np.isfinite(trial).all():continue
        # The rounded LP axis itself must have exact, nonnegative source
        # projections. Zero is allowed only for shared topological simplices.
        rational=[Fraction(float(x)) for x in trial]
        trial_gaps=[dot(rational,sub(x,y)) for x in a for y in b]
        if min(trial_gaps)<0 or (min(trial_gaps)==0 and not shared):continue
        if not any(x>0 for x in trial_gaps):continue
        normal=trial;gaps=trial_gaps;break
    if normal is None:
        raise ValueError("self-contact reference triangles have no complete separating support plane")
    floors=[min(float(margin_m),1.e-6,float(gap)/4.) for gap in gaps]
    if not all(np.isfinite(x) and x>=0 for x in floors):
        raise ValueError("invalid face-support floor")
    eye=np.eye(3)
    return [(int(first),eye[i].copy(),int(second),eye[j].copy(),
             normal.copy(),floors[3*i+j]-float(gaps[3*i+j]))
            for i in range(3) for j in range(3)]


def compartment_direction(skin_normal,weights,binding_body_ids,torso_body_id):
    """Offline direction proposal from full retained skin material support.

    Arm skin encloses its limb bone; torso skin must leave that limb bone
    outside the torso branch. This selects an exit, never waives a crossing.
    """
    weights=np.asarray(weights,dtype=float);owners=np.asarray(binding_body_ids)
    normal=np.asarray(skin_normal,dtype=float)
    if (weights.ndim!=1 or owners.shape!=weights.shape or owners.dtype.kind not in "iu"
            or len(np.unique(owners))!=len(owners) or normal.shape!=(3,)
            or not np.isfinite(weights).all() or np.any(weights<0)
            or abs(float(weights.sum())-1.)>1e-4
            or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-12):
        raise ValueError("invalid full-owner compartment direction inputs")
    match=np.flatnonzero(owners==torso_body_id)
    if len(match)!=1:raise ValueError("torso owner is absent or duplicated")
    torso_weight=float(weights[int(match[0])])
    direction=normal/np.linalg.norm(normal)
    region="arm_skin_enclosing_limb_bone"
    if torso_weight>.5:
        direction=-direction
        region="torso_skin_excluding_limb_bone"
    return direction,dict(region=region,torso_weight=torso_weight,
                          torso_core_body_index=int(torso_body_id))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ["source", "authoring-base", "initial-source-points", "contact-cache", "capture", "receipt", "output", "bone-artifact", "body-manifest"]:
        ap.add_argument("--"+flag, type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=4, choices=range(1,5))
    ap.add_argument("--step-bound-m", type=float, default=.003)
    ap.add_argument("--total-bound-m", type=float, default=.006)
    ap.add_argument("--margin-m", type=float, default=.0005)
    ap.add_argument("--solid-depth", action="store_true")
    ap.add_argument("--repair-self-contacts", action="store_true",
        help="bounded joint solve from actual self-contact witnesses; never a clearance waiver")
    ap.add_argument("--self-contact-separator", choices=["witness","face-support"],default="witness")
    ap.add_argument("--exit-policy",choices=["nearest","skin-normal","compartment-skin-normal"],default="nearest")
    ap.add_argument("--orientation-ratio", type=float, default=0.)
    args = ap.parse_args()
    if not 0 < args.margin_m < args.step_bound_m <= args.total_bound_m <= .03:
        ap.error("require 0 < margin < step <= total <= 0.03 m")
    if not 0 <= args.orientation_ratio < 1:
        ap.error("orientation ratio must be in [0,1)")
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    pins = {str(p): sha(p) for p in [Path(__file__), args.source, args.authoring_base,
        args.initial_source_points, args.contact_cache, args.capture, args.receipt,
        *[Path(m.__file__) for m in [h,v,sp,gp]]]}
    source = sp.decode_payload(args.source.read_bytes())
    bone_ids, bone_identity_proof = v.load_compiled_bone_identities(args.bone_artifact, args.body_manifest, capture=args.capture)
    if bone_identity_proof["source_archive_sha256"] != source["source_archive_sha256"]:
        raise ValueError("bone and skin source archives differ")
    for item in ("bone_payload", "bone_manifest", "runtime_manifest", "native_invocation"):
        entry=bone_identity_proof[item]
        pins[entry["path"]]=entry["sha256"]
    for sid in [10,12,14,16]:
        if sid not in bone_ids:
            raise ValueError(f"required compiled bone identity absent: {sid}")

    torso_body_id=None
    if args.exit_policy=="compartment-skin-normal":
        torso_ids={row["core_body_index"] for row in bone_ids.values()
                   if row["core_body_name"]=="torso"}
        if len(torso_ids)!=1:raise ValueError("compiled bone owner does not uniquely identify torso")
        torso_body_id=next(iter(torso_ids))

    base = np.load(args.authoring_base, allow_pickle=False)
    initial = np.load(args.initial_source_points, allow_pickle=False)
    cache = np.load(args.contact_cache, allow_pickle=False)
    full_faces = source["indices"].reshape(-1,3)
    used = np.unique(full_faces)
    faces = np.asarray(base["capture_faces"], dtype=np.int64)
    gids = base["capture_global_vertex_ids"]
    base_points = base["capture_positions_m"]
    if not np.array_equal(faces, np.searchsorted(used, full_faces)):
        raise ValueError("source and authoring face ancestry differ")
    if gids.dtype.kind not in "iu" or len(np.unique(gids)) != len(gids):
        raise ValueError("ordered integer vertex identity required")
    if not np.array_equal(initial["full_weights"], source["full_weights"]):
        raise ValueError("initial material weights differ")
    points_source = initial["source_positions_m"].copy()
    if points_source.shape != source["vertices_f"][:,:3].shape or not np.isfinite(points_source).all():
        raise ValueError("source point schema differs")
    receipt = json.loads(args.receipt.read_text())
    poses = {r["body_index"]: (np.array(r["position_m"]),
        np.array(h._myosim_matrix_from_quaternion_xyzw(r["quaternion_xyzw"])))
        for r in receipt["accepted_body_poses"]}
    affine = np.zeros((len(used),3,3))
    for j,k in enumerate(source["bindings_u"][:,0]):
        _, rot = poses[int(k)]
        bind = source["bindings_f"][j]
        rest = np.array(h._myosim_matrix_from_quaternion_xyzw(bind[4:8].tolist()))*float(bind[8])
        affine += source["full_weights"][used,j,None,None].astype(float)*(rot@rest)
    if not np.isfinite(affine).all() or np.min(abs(np.linalg.det(affine))) < 1e-8:
        raise ValueError("invalid full-owner rest map")
    base_source = source["vertices_f"][:,:3].astype(float)
    skin_normals=base["capture_normals"].astype(float)
    skin_normals/=np.linalg.norm(skin_normals,axis=1)[:,None]
    if args.exit_policy in ("skin-normal","compartment-skin-normal") and not args.solid_depth:
        ap.error("skin-normal exit policy requires solid-depth classification")
    base_triangles=base_points[faces].astype(float)
    base_area_vectors=np.cross(base_triangles[:,1]-base_triangles[:,0],base_triangles[:,2]-base_triangles[:,0])
    base_areas=np.linalg.norm(base_area_vectors,axis=1)
    if not np.all(base_areas>0):raise ValueError("source has degenerate faces")
    reference_normals=base_area_vectors/base_areas[:,None]
    edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    proof = dict(status="PASS_COMPLETE_AUTHORING_PREPARATION", predicate_phase_started=False,
        argv=sys.argv, source_vertex_count=len(base_source), referenced_vertex_count=len(used),
        face_count=len(faces), positive_material_support_unchanged=True,
        units="m; linear coefficients dimensionless", exact_emission=False,
        bounds=dict(step_m=args.step_bound_m,total_m=args.total_bound_m,margin_m=args.margin_m),
        bone_target_identities=[bone_ids[sid] for sid in [10,12,14,16]],
        compiled_bone_identity=bone_identity_proof,
        scope="Offline authoring approximation only; no anatomical/native admission.",
        construction_exit_policy=args.exit_policy,torso_core_body_index=torso_body_id)
    frozen = gp.freeze_preparation(args.output/"preparation.json", proof,
        runner_argv=[sys.executable,*sys.argv],input_pins=pins)
    gp.verify_frozen_preparation(frozen)
    cap, identity = v.load_capture(args.capture,args.receipt)
    bones = {}
    for sid in [10,12,14,16]:
        key=(51004,sid); bp,bf,bn,bg=cap[key]
        bone=v.prepare_surface(key,bp,bf,normals=bn,source_vertex_ids=bg)
        if not bone.closed:
            raise ValueError("target must be a closed solid")
        tri=bp[bf].astype(float)
        volume=np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
        if not volume:
            raise ValueError("zero target signed volume")
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals*=np.sign(volume)/np.linalg.norm(normals,axis=1)[:,None]
        bones[sid]=(bone,normals,bone.meter_index() if args.solid_depth else None)
    constraints=[]
    if args.exit_policy=="nearest":
        for sf,bary,normal in zip(cache["skin_face_ids"],cache["barycentrics"],cache["outward_normals"]):
            constraints.append((int(sf),bary.copy(),normal.copy(),args.margin_m))
    previous=None
    self_constraints=[]
    history=[]
    for iteration in range(args.iterations+1):
        out=args.output/f"iteration-{iteration:02d}";out.mkdir()
        delta=np.einsum("nij,nj->ni",affine,(points_source.astype(float)-base_source)[used])
        world=(base_points.astype(float)+delta).astype("<f4")
        surf=v.prepare_surface((51007,1),world,faces,source_vertex_ids=gids)
        self_result=surf.self_audit() if previous is None else surf.self_audit_since(previous)
        (out/"self.json").write_text(json.dumps(self_result,indent=2,allow_nan=False)+"\n")
        np.savez_compressed(out/"diagnostic.npz",source_positions_m=points_source,full_weights=source["full_weights"])
        np.savez_compressed(out/"authoring-overlay.npz",capture_positions_m=world,capture_faces=faces,capture_global_vertex_ids=gids)
        self_failed=bool(self_result["count"])
        if self_failed and not args.repair_self_contacts:
            status="FAIL_SELF"
            history.append(dict(iteration=iteration,status=status,pair_count=self_result["count"]))
            break
        if self_failed:
            if self_result["count"]>256:
                raise ValueError("self-contact repair exceeds the bounded local patch")
            witnesses=[]
            for first,second in self_result["triangle_pairs"]:
                if args.self_contact_separator=="face-support":
                    derived=face_support_constraints(base_points,faces,first,second,args.margin_m)
                else:
                    derived=[]
                    for point in surf.intersection_points_meters(surf,int(first),int(second)):
                        ba=barycentric(surf.triangle_meters(int(first)),point)
                        bb=barycentric(surf.triangle_meters(int(second)),point)
                        derived.append(self_gap_constraint(base_points,faces,first,ba,second,bb,
                                                           min(args.margin_m,.0001)))
                for constraint in derived:
                    self_constraints.append(constraint)
                    a,wa,b,wb,n,bound=constraint
                    witnesses.append(dict(first_face=a,first_barycentric=wa.tolist(),
                        second_face=b,second_barycentric=wb.tolist(),normal=n.tolist(),
                        relative_displacement_lower_bound_m=bound))
            (out/"self-gap-constraints.json").write_text(json.dumps(witnesses,indent=2,allow_nan=False)+"\n")
        else:
            previous=surf
        new_constraints=[]
        inventory=[]
        solid_witnesses=[]
        compartment_witnesses=[]
        for sid,(bone,normals,solid_index) in ([] if self_failed else bones.items()):
            result=v.audit_surface_pair(surf,bone)
            (out/f"bone-{sid}.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
            pairs=result.get("unexpected_triangle_pairs",[])
            inventory.append(dict(bone=sid,triangle_pair_count=len(pairs)))
            for sf,bi in pairs:
                for point in surf.intersection_points_meters(bone,int(sf),int(bi)):
                    bary=barycentric(surf.triangle_meters(int(sf)),point)
                    normal=normals[bi]
                    distance=0.
                    if args.exit_policy in ("skin-normal","compartment-skin-normal"):
                        normal=bary@skin_normals[faces[sf]]
                        normal/=np.linalg.norm(normal)
                        if args.exit_policy=="compartment-skin-normal" and bone_ids[sid]["core_body_name"] in {"humerus_r","ulna_r","radius_r"}:
                            normal,region=compartment_direction(normal,
                                bary@source["full_weights"][used[faces[sf]]],
                                source["bindings_u"][:,0],torso_body_id)
                            compartment_witnesses.append(dict(skin_face=int(sf),
                                bone=sid,kind="triangle_contact",**region))
                        if normal@normals[bi]<0:
                            distance,_=contact_forward_exit(bone,solid_index,point,normal)
                    current=float(bary@delta[faces[sf]]@normal)
                    new_constraints.append((int(sf),bary,normal,current+distance+args.margin_m))
            if args.solid_depth and pairs:
                point_faces={}
                for sf,_ in pairs:
                    for corner,vi in enumerate(faces[sf]):
                        point_faces.setdefault(int(vi),(int(sf),corner))
                tri=bone.positions[bone.faces].astype(float)
                lower=tri.min(1);upper=tri.max(1)
                for vi,(sf,corner) in point_faces.items():
                    point=world[vi].astype(float)
                    exact_point=tuple(Fraction.from_float(float(x)) for x in point)
                    location=v.ci.point_location_prepared_indexed(exact_point,solid_index)["location"]
                    if location!="inside":continue
                    nearest=None
                    lower_distance=np.linalg.norm(np.maximum(np.maximum(lower-point,point-upper),0),axis=1)
                    for bi in np.argsort(lower_distance):
                        if nearest is not None and lower_distance[bi]>nearest[0]:break
                        near=np.array(h._tendon_closest_point_on_triangle(point.tolist(),tri[bi].tolist())[0])
                        distance=float(np.linalg.norm(near-point))
                        if nearest is None or distance<nearest[0]:nearest=(distance,int(bi),near)
                    distance,bi,near=nearest
                    if not distance:raise ValueError("inside point has zero nearest-solid distance")
                    normal=(near-point)/distance
                    if args.exit_policy in ("skin-normal","compartment-skin-normal"):
                        normal=skin_normals[vi]
                        if args.exit_policy=="compartment-skin-normal" and bone_ids[sid]["core_body_name"] in {"humerus_r","ulna_r","radius_r"}:
                            normal,region=compartment_direction(normal,
                                source["full_weights"][used[vi]],
                                source["bindings_u"][:,0],torso_body_id)
                            compartment_witnesses.append(dict(skin_vertex=int(vi),
                                bone=sid,kind="inside_vertex",**region))
                        distance,bi=exact_forward_exit(bone,solid_index,point,normal)
                    bary=np.zeros(3);bary[corner]=1
                    current=float(delta[vi]@normal)
                    new_constraints.append((sf,bary,normal,current+distance+args.margin_m))
                    solid_witnesses.append(dict(bone=sid,vertex=vi,face=sf,nearest_bone_face=bi,depth_m=distance,direction=normal.tolist()))
        (out/"compartment-direction-witnesses.json").write_text(json.dumps(compartment_witnesses,indent=2,allow_nan=False)+"\n")
        (out/"solid-depth-witnesses.json").write_text(json.dumps(solid_witnesses,indent=2,allow_nan=False)+"\n")
        history.append(dict(iteration=iteration,status="SELF_CONTACT_REPAIR" if self_failed else "CONTACT_INVENTORY",targets=inventory,
            new_constraints=len(new_constraints)))
        print(json.dumps(history[-1]),flush=True)
        if not new_constraints and not self_failed:
            status="PASS_AUTHORING_FOUR_BONES_ONLY"
            break
        if iteration==args.iterations or time.monotonic()-started>1200:
            status="FAIL_BOUNDED_METHOD_EXHAUSTED"
            break
        constraints.extend(new_constraints)
        seed_faces=[x[0] for x in constraints]+[f for a,_,b,_,_,_ in self_constraints for f in (a,b)]
        seed=np.unique(faces[np.unique(seed_faces)])
        active=seed
        for _ in range(2):
            active=np.unique(faces[np.any(np.isin(faces,active),axis=1)])
        lookup={int(g):i for i,g in enumerate(active)}
        local_edges=edges[np.any(np.isin(edges,active),axis=1)]
        nd=len(active)*3;ne=len(local_edges)*3;nv=nd*2+ne
        rr=[];cc=[];vv=[];rhs=[]
        def add(terms,bound):
            row=len(rhs)
            for col,value in terms:
                if value:rr.append(row);cc.append(col);vv.append(float(value))
            rhs.append(float(bound))
        for sf,bary,normal,bound in constraints:
            terms=[]
            for corner,g in enumerate(faces[sf]):
                if int(g) in lookup:
                    terms.extend((lookup[int(g)]*3+a,-bary[corner]*normal[a]) for a in range(3))
            if not terms:raise ValueError("unowned movable contact")
            add(terms,-bound)
        for first,wa,second,wb,normal,bound in self_constraints:
            terms=[]
            for fi,bary,sign in [(first,wa,-1),(second,wb,1)]:
                for corner,g in enumerate(faces[fi]):
                    if int(g) in lookup:
                        terms.extend((lookup[int(g)]*3+a,sign*bary[corner]*normal[a]) for a in range(3))
            if not terms:raise ValueError("unowned movable self-contact witness")
            add(terms,-bound)
        if args.orientation_ratio:
            affected=np.flatnonzero(np.any(np.isin(faces,active),axis=1))
            for fi in affected:
                ids=faces[fi];t=world[ids].astype(float);normal=reference_normals[fi]
                e1=t[1]-t[0];e2=t[2]-t[0]
                gradient=projected_area_gradient(t,normal)
                scale=float(np.linalg.norm(gradient,axis=1).max())
                if not scale:raise ValueError("orientation gradient degenerate")
                current_area=float(np.cross(e1,e2)@normal)
                bound=(args.orientation_ratio*base_areas[fi]-current_area+
                       float(np.sum(gradient*delta[ids])))/scale
                terms=[]
                for corner,g in enumerate(ids):
                    if int(g) in lookup:
                        terms.extend((lookup[int(g)]*3+a,-gradient[corner,a]/scale) for a in range(3))
                add(terms,-bound)
        for j in range(nd):
            add([(j,1),(nd+j,-1)],0);add([(j,-1),(nd+j,-1)],0)
        for ei,(a,b) in enumerate(local_edges):
            for axis in range(3):
                terms=[]
                if int(a) in lookup:terms.append((lookup[int(a)]*3+axis,1))
                if int(b) in lookup:terms.append((lookup[int(b)]*3+axis,-1))
                aux=nd*2+ei*3+axis
                add(terms+[(aux,-1)],0);add([(j,-z) for j,z in terms]+[(aux,-1)],0)
        prior=delta[active].ravel()
        bounds=[(max(-args.total_bound_m,float(x)-args.step_bound_m),
                 min(args.total_bound_m,float(x)+args.step_bound_m)) for x in prior]
        matrix=coo_matrix((vv,(rr,cc)),shape=(len(rhs),nv)).tocsr()
        cost=np.concatenate([np.zeros(nd),np.ones(nd),np.ones(ne)])
        all_bounds=bounds+[(0,None)]*(nd+ne)
        np.savez_compressed(out/"linear-program.npz",cost=cost,rhs=np.array(rhs),
            matrix_data=matrix.data,matrix_indices=matrix.indices,matrix_indptr=matrix.indptr,
            matrix_shape=np.array(matrix.shape),bounds=np.array([(a,np.inf if b is None else b) for a,b in all_bounds]),
            active_vertex_ids=active,previous_displacement_m=prior)
        solved=linprog(cost,A_ub=matrix,b_ub=np.array(rhs),bounds=all_bounds,
            method="highs",options={"time_limit":60})
        (out/"solve.json").write_text(json.dumps(dict(success=bool(solved.success),
            message=solved.message,status=int(solved.status),constraints=len(constraints),
            active_vertices=len(active)),indent=2,allow_nan=False)+"\n")
        if not solved.success:
            status="FAIL_LINEAR_CONSTRUCTION"
            break
        requested=solved.x[:nd].reshape(-1,3)
        accepted=False
        for backtrack in range(11):
            fraction=2.**(-backtrack)
            trial=delta[active]+fraction*(requested-delta[active])
            candidate_source=source["vertices_f"][:,:3].copy()
            candidate_source[used[active]]=(base_source[used[active]]+
                np.linalg.solve(affine[active],trial[:,:,None])[...,0]).astype("<f4")
            actual_delta=np.einsum("nij,nj->ni",affine,(candidate_source.astype(float)-base_source)[used])
            trial_world=(base_points.astype(float)+actual_delta).astype("<f4")
            t=trial_world[faces].astype(float)
            areas=np.einsum("ij,ij->i",np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),reference_normals)
            if not args.orientation_ratio or np.all(areas>=args.orientation_ratio*base_areas):
                points_source=candidate_source;accepted=True;break
        (out/"orientation-line-search.json").write_text(json.dumps(dict(accepted=accepted,fraction=fraction,minimum_ratio=float(np.min(areas/base_areas)),required_ratio=args.orientation_ratio),indent=2,allow_nan=False)+"\n")
        if not accepted:
            status="FAIL_ORIENTATION_CONSTRUCTION"
            break
    report=dict(status=status,source_pins=pins,capture_identity=identity,history=history,
        elapsed_seconds=time.monotonic()-started,anatomical_admission=False,native_admission=False,
        next_change="Existing-owner export and all-near/native gates if authoring passes; otherwise retire this bounded contact construction and repair the specific retained witness.")
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=status,elapsed_seconds=report["elapsed_seconds"])),flush=True)
    return 0 if status.startswith("PASS_") else 1

if __name__=="__main__":
    try:
        result=main()
    except Exception as error:
        import traceback
        if "--output" in sys.argv:
            failure_dir=Path(sys.argv[sys.argv.index("--output")+1])
            if failure_dir.is_dir():
                failure=dict(status="FAILED_EXECUTION",error=repr(error),traceback=traceback.format_exc(),
                    runner_argv=[sys.executable,*sys.argv],script_sha256=sha(Path(__file__)),
                    anatomical_admission=False,native_admission=False)
                with (failure_dir/"failed-execution.json").open("x") as stream:
                    stream.write(json.dumps(failure,indent=2,allow_nan=False)+"\n")
        raise
    raise SystemExit(result)
