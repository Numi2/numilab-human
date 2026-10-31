"""Sparse coupled tet-quality LP with row-equilibrated constraints, exact shared-feature SAT, and baseline Float64 ULP allowance; construction only."""
from pathlib import Path
import hashlib, importlib.util, sys
import numpy as np, time
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.csgraph import connected_components

_H=Path(__file__).resolve().parent.parent
_Q=_H/"fixed_boundary_volume_flip_198_root.py"
_L=_H/"condition_fixed_skin_interior_LP_455_root.py"
_QSHA="650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
_LSHA="5d6e7a22c49ec907a881e01d76118877f67cc95f2bb7c5302a09ee2e26469687"
def _sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def _load():
 q,l=_sha(_Q),_sha(_L)
 if q!=_QSHA or l!=_LSHA:raise RuntimeError("pinned quality/LP owner changed")
 s=importlib.util.spec_from_file_location("quality_owner_198_for_629",_Q)
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
 return m,q,l
_M,QUALITY_OWNER_SHA256,LP_REFERENCE_SHA256=_load()

def determinant_gradients(points,tetrahedra):
 """Analytic signed det6 gradients, same vectorized formula as LP owner 455."""
 xyz=np.asarray(points,dtype=np.float64)[np.asarray(tetrahedra,dtype=np.int64)]
 v=xyz[:,1:,:]-xyz[:,:1,:]
 return np.stack((-np.cross(v[:,1],v[:,2])-np.cross(v[:,2],v[:,0])-np.cross(v[:,0],v[:,1]),
                  np.cross(v[:,1],v[:,2]),np.cross(v[:,2],v[:,0]),np.cross(v[:,0],v[:,1])),axis=1)

def _targets(points,tets):
 xyz=np.asarray(points,dtype=np.float64)[tets];ell=np.zeros(len(tets),dtype=np.float64)
 for i in range(4):
  for j in range(i+1,4):ell=np.maximum(ell,np.linalg.norm(xyz[:,i]-xyz[:,j],axis=1))
 # Native gates: det6/ell^3 >= 1e-5 and det6/6 > 1e-18 m^3.
 return ell,np.maximum(1.e-5*ell**3,6.e-18)

def _log_deficit(qratio,volratio):
 """Sum positive normalized log deficits for quality and minimum volume."""
 q=np.asarray(qratio,dtype=np.float64);v=np.asarray(volratio,dtype=np.float64)
 if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError("nonfinite normalized quality ratio")
 floor=np.finfo(np.float64).tiny
 return float(np.maximum(0.,-np.log(np.maximum(q,floor))).sum(dtype=np.float64)
              +np.maximum(0.,-np.log(np.maximum(v,floor))).sum(dtype=np.float64))

def _exact_positive(points,tets):
 used=np.unique(tets);ip=_M.integer_points(np.asarray(points,dtype=np.float32)[used])
 local=np.full(len(points),-1,dtype=np.int64);local[used]=np.arange(len(used),dtype=np.int64)
 return [_M.exact_det(ip,[int(local[int(v)]) for v in tet]) for tet in tets]


def _linear_program_matrix_stats(A,b):
 """Compact coefficient, RHS, and row-norm ranges for numerical diagnosis."""
 A=A.tocsr();b=np.asarray(b,dtype=np.float64)
 ad=np.asarray(A.data,dtype=np.float64);nz=np.abs(ad[ad!=0.])
 rowmax=np.asarray(np.abs(A).max(axis=1).toarray(),dtype=np.float64).reshape(-1)
 rhsabs=np.abs(b);rnz=rhsabs[rhsabs>0.]
 combined=np.maximum(rowmax,rhsabs);cnz=combined[combined>0.]
 def span(values):
  if not len(values):return {"min":None,"max":None,"ratio":None}
  lo=float(np.min(values));hi=float(np.max(values))
  return {"min":lo,"max":hi,"ratio":(hi/lo if lo>0 else None)}
 return {
  "all_finite":bool(np.isfinite(ad).all() and np.isfinite(b).all()),
  "rows":int(A.shape[0]),"columns":int(A.shape[1]),"stored_nnz":int(A.nnz),
  "explicit_zero_count":int(np.count_nonzero(ad==0.)),
  "coefficient_abs_nonzero":span(nz),"rhs_abs_nonzero":span(rnz),
  "row_coefficient_linf_nonzero":span(rowmax[rowmax>0.]),
  "row_max_coefficient_or_rhs_nonzero":span(cnz),
  "rhs_signed_min":float(np.min(b)) if len(b) else None,
  "rhs_signed_max":float(np.max(b)) if len(b) else None,
  "zero_constraint_rows":int(np.count_nonzero(np.diff(A.indptr)==0)),
 }


def _equilibrate_inequality_rows(A,b):
 """Multiply each A_ub row and its RHS by the same positive finite factor."""
 A=A.tocsr().astype(np.float64);b=np.asarray(b,dtype=np.float64)
 if A.ndim!=2 or b.shape!=(A.shape[0],):raise ValueError("LP inequality matrix/RHS shape mismatch")
 if not np.isfinite(A.data).all() or not np.isfinite(b).all():raise ValueError("nonfinite LP inequality matrix/RHS")
 before=_linear_program_matrix_stats(A,b)
 rowmax=np.asarray(np.abs(A).max(axis=1).toarray(),dtype=np.float64).reshape(-1)
 magnitude=np.maximum(rowmax,np.abs(b))
 factors=np.ones(len(b),dtype=np.float64)
 nz=magnitude>0.
 factors[nz]=1./np.maximum(magnitude[nz],np.finfo(np.float64).tiny)
 if not np.isfinite(factors).all() or np.any(factors<=0.):raise ValueError("invalid positive LP row-equilibration factor")
 Ae=A.multiply(factors[:,None]).tocsr();be=b*factors
 if not np.isfinite(Ae.data).all() or not np.isfinite(be).all():raise ValueError("row equilibration produced nonfinite LP system")
 after=_linear_program_matrix_stats(Ae,be)
 return Ae,be,{"method":"positive full-row infinity equilibration over all A_ub coefficients including slack columns and matching RHS",
               "zero_rows_left_unscaled":int(np.count_nonzero(~nz)),
               "scaled_rows":int(np.count_nonzero(nz)),
               "positive_factor_abs":{ "min":float(factors.min()) if len(factors) else None,
                                        "max":float(factors.max()) if len(factors) else None},
               "pre_equilibration":before,"post_equilibration":after}


def _solve_highs_with_status4_fallback(cost,A,b,bounds,time_limit_s):
 """Try dual simplex, then one IPM retry within the same total LP time budget."""
 started=time.monotonic();limit=float(time_limit_s)
 res=linprog(cost,A_ub=A,b_ub=b,bounds=bounds,method="highs-ds",options={"time_limit":limit})
 attempts=[{"method":"highs-ds","status":int(res.status),"message":str(res.message),
            "elapsed_s":time.monotonic()-started,"time_limit_s":limit}]
 primary_status=int(res.status)
 if primary_status==4:
  remaining=max(0.,limit-(time.monotonic()-started))
  if remaining>0.01:
   fallback=linprog(cost,A_ub=A,b_ub=b,bounds=bounds,method="highs-ipm",options={"time_limit":remaining})
   attempts.append({"method":"highs-ipm","status":int(fallback.status),"message":str(fallback.message),
                    "elapsed_s":time.monotonic()-started,"time_limit_s":remaining,
                    "trigger":"bounded fallback after highs-ds status 4; same row-equilibrated LP, objective and bounds"})
   res=fallback
 return res,attempts,primary_status,time.monotonic()-started



def _closest_point_triangle(p,tri):
    p=np.asarray(p,float);a,b,c=np.asarray(tri,float);ab=b-a;ac=c-a;ap=p-a
    d1=float(ab@ap);d2=float(ac@ap)
    if d1<=0 and d2<=0:return a.copy()
    bp=p-b;d3=float(ab@bp);d4=float(ac@bp)
    if d3>=0 and d4<=d3:return b.copy()
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:return a+(d1/(d1-d3))*ab
    cp=p-c;d5=float(ab@cp);d6=float(ac@cp)
    if d6>=0 and d5<=d6:return c.copy()
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:return a+(d2/(d2-d6))*ac
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
    den=va+vb+vc
    if den<=0 or not np.isfinite(den):raise ValueError("degenerate triangle")
    v=vb/den;w=vc/den
    return a+v*ab+w*ac

def _closest_segments(p1,q1,p2,q2):
    p1=np.asarray(p1,float);q1=np.asarray(q1,float);p2=np.asarray(p2,float);q2=np.asarray(q2,float)
    d1=q1-p1;d2=q2-p2;r=p1-p2;a=float(d1@d1);e=float(d2@d2);f=float(d2@r);tiny=np.finfo(float).tiny
    if a<=tiny and e<=tiny:return p1.copy(),p2.copy()
    if a<=tiny:s=0.;t=float(np.clip(f/e,0,1))
    else:
        c=float(d1@r)
        if e<=tiny:t=0.;s=float(np.clip(-c/a,0,1))
        else:
            b=float(d1@d2);den=a*e-b*b
            s=float(np.clip((b*f-c*e)/den,0,1)) if den>tiny*a*e else 0.
            t=(b*s+f)/e
            if t<0:t=0.;s=float(np.clip(-c/a,0,1))
            elif t>1:t=1.;s=float(np.clip((b-c)/a,0,1))
    return p1+s*d1,p2+t*d2

def _closest_triangle_pair(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=(3,3) or b.shape!=(3,3) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("invalid triangle pair")
    if np.linalg.norm(np.cross(a[1]-a[0],a[2]-a[0]))==0 or np.linalg.norm(np.cross(b[1]-b[0],b[2]-b[0]))==0:raise ValueError("degenerate triangle")
    c=[(v.copy(),_closest_point_triangle(v,b)) for v in a]
    c += [(_closest_point_triangle(v,a),v.copy()) for v in b]
    c += [_closest_segments(a[i],a[(i+1)%3],b[j],b[(j+1)%3]) for i in range(3) for j in range(3)]
    q0,q1=min(c,key=lambda z:float((z[1]-z[0])@(z[1]-z[0])))
    return q0,q1,float(np.linalg.norm(q1-q0))

def _prepare_triangle_separation_rows_v15(points,faces_a,faces_b,pair_rows,*,points_b=None,retained_gap_fraction=.5):
    """Build fixed-normal halfspaces only from supplied exact failed triangle pairs."""
    p=np.asarray(points);fa=np.asarray(faces_a,np.int64);fb=np.asarray(faces_b,np.int64)
    pairs=np.asarray(pair_rows,np.int64).reshape((-1,2));pb=p if points_b is None else np.asarray(points_b)
    if p.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError("source baseline must be finite Float32")
    if pb.ndim!=2 or pb.shape[1]!=3 or not np.isfinite(pb).all():raise ValueError("invalid second surface points")
    if fa.ndim!=2 or fa.shape[1]!=3 or fb.ndim!=2 or fb.shape[1]!=3:raise ValueError("face arrays must be [F,3]")
    if (fa.size and (fa.min()<0 or fa.max()>=len(p))) or (fb.size and (fb.min()<0 or fb.max()>=len(pb))):raise ValueError("face index outside point array")
    if len(pairs) and (pairs.min()<0 or pairs[:,0].max()>=len(fa) or pairs[:,1].max()>=len(fb)):raise ValueError("pair row outside faces")
    frac=float(retained_gap_fraction)
    if not np.isfinite(frac) or not 0<frac<1:raise ValueError("retained gap fraction must be in (0,1)")
    same=points_b is None;rows=[];reject=[];freeze=set();gaps=[]
    for pid,(ia,ib) in enumerate(pairs):
        va=fa[int(ia)];vb=fb[int(ib)];ta=p[va].astype(float);tb=pb[vb].astype(float)
        try:qa,qb,dist=_closest_triangle_pair(ta,tb)
        except ValueError as exc:
            reject.append({"pair_id":pid,"face_rows":[int(ia),int(ib)],"reason":"INVALID_BASELINE_TRIANGLE","detail":str(exc)})
            freeze.update(map(int,va))
            if same:freeze.update(map(int,vb))
            continue
        scale=max(float(np.linalg.norm(np.ptp(ta,axis=0))),float(np.linalg.norm(np.ptp(tb,axis=0))),1.e-12,float(np.max(np.abs(np.r_[ta.ravel(),tb.ravel()]))))
        floor=128*np.finfo(float).eps*max(scale,1.)
        if not np.isfinite(dist) or dist<=floor:
            reject.append({"pair_id":pid,"face_rows":[int(ia),int(ib)],"reason":"NO_POSITIVE_BASELINE_GAP","closest_distance_m":dist})
            freeze.update(map(int,va))
            if same:freeze.update(map(int,vb))
            continue
        n=(qb-qa)/dist;gap=float((tb@n).min()-(ta@n).max())
        if not np.isfinite(gap) or gap<=floor:
            reject.append({"pair_id":pid,"face_rows":[int(ia),int(ib)],"reason":"NO_POSITIVE_SUPPORT_GAP","closest_distance_m":dist,"support_gap_m":gap})
            freeze.update(map(int,va))
            if same:freeze.update(map(int,vb))
            continue
        margin=frac*gap;o=(qa+qb)*.5
        if same:
            for ai in range(3):
                for bj in range(3):
                    ns=np.asarray((int(va[ai]),int(vb[bj])),np.int64);vs=np.asarray((n,-n))
                    if ns[0]==ns[1]:ns=np.asarray((int(ns[0]),),np.int64);vs=np.zeros((1,3))
                    rows.append({"point_rows":ns,"vectors":vs,"origin":o.copy(),"upper_bound":-margin,
                        "face_pair":[int(ia),int(ib)],"pair_id":pid,"baseline_gap_m":gap,"retained_gap_m":margin,"axis":n.copy(),"kind":"source_source"})
        else:
            minb=float((tb@n).min())
            for ai in range(3):
                rows.append({"point_rows":np.asarray((int(va[ai]),),np.int64),"vectors":n.reshape(1,3).copy(),
                    "origin":o.copy(),"upper_bound":minb-float(n@o)-margin,"face_pair":[int(ia),int(ib)],
                    "pair_id":pid,"baseline_gap_m":gap,"retained_gap_m":margin,"axis":n.copy(),"kind":"source_fixed"})
        gaps.append(gap)
    by={}
    for ri,r in enumerate(rows):
        for node in np.unique(r["point_rows"]):by.setdefault(int(node),[]).append(ri)
    report={"exact_pair_witnesses":int(len(pairs)),"usable_separating_pairs":len(gaps),"rejected_pairs":len(reject),
        "linear_halfspace_rows":len(rows),"source_source_pair_count":sum(r["kind"]=="source_source" for r in rows)//9,
        "source_fixed_pair_count":sum(r["kind"]=="source_fixed" for r in rows)//3,
        "minimum_baseline_support_gap_m":float(min(gaps)) if gaps else None,"median_baseline_support_gap_m":float(np.median(gaps)) if gaps else None,
        "retained_gap_fraction":frac,"rejected_pair_details":reject[:64],"freeze_source_nodes":sorted(freeze),
        "baseline_source_coordinate_dtype":"Float32","fixed_context_coordinate_dtype":str(pb.dtype) if not same else None,
        "scope":"supplied exact face-pair witnesses only; final exact pair audit remains required"}
    return {"constraints":rows,"row_indices_by_point":by,"preparation":report}


# Add exact shared-feature SAT support to the pinned row-scaled v15 separator.
# The LP assembly and positive row equilibration remain byte-identical to v15.
import importlib.util as _importlib_util
_SHARED_PATH = Path(__file__).resolve().parent / "quality_lp_629_v14d_shared_feature_witness.py"
_SHARED_SHA256 = "d136b83b730e285b8bb5abad186920fb5c76a70eee877ad28aff67ae12787bcd"
def _load_shared_feature_owner():
    h=hashlib.sha256(_SHARED_PATH.read_bytes()).hexdigest()
    if h != _SHARED_SHA256:
        raise RuntimeError("pinned shared-feature witness owner changed")
    spec=_importlib_util.spec_from_file_location("quality_lp_v14d_shared_feature_owner",_SHARED_PATH)
    if spec is None or spec.loader is None:raise RuntimeError("cannot load pinned v14d shared-feature owner")
    module=_importlib_util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
_SHARED_FEATURE_OWNER=_load_shared_feature_owner()
SHARED_FEATURE_OWNER_SHA256=_SHARED_SHA256
ROW_SCALED_SHARED_BASE_SHA256="e471dde6049ecf5e722813dda41918a4b6a2202a45bd3d3dfc178bf18062cea8"

def prepare_triangle_separation_rows(points,faces_a,faces_b,pair_rows,*,points_b=None,retained_gap_fraction=.5):
    """Row-scaled v15 constraints plus v14d exact shared-feature weak SAT rows."""
    base=_prepare_triangle_separation_rows_v15(points,faces_a,faces_b,pair_rows,
                                               points_b=points_b,retained_gap_fraction=retained_gap_fraction)
    if points_b is not None:return base
    p=np.asarray(points);fa=np.asarray(faces_a,np.int64);fb=np.asarray(faces_b,np.int64)
    pairs=np.asarray(pair_rows,np.int64).reshape((-1,2))
    base_rows=list(base["constraints"])
    accepted_ids={int(r["pair_id"]) for r in base_rows}
    rejected_ids=sorted(set(range(len(pairs)))-accepted_ids)
    if not rejected_ids:return base
    base_prep=dict(base["preparation"])
    old_details={int(x["pair_id"]):dict(x) for x in base_prep.get("rejected_pair_details",[])}
    safe_rows=[];accepted_shared=[];unsafe_details=[];unsafe_frozen=set();numeric_ulp_rows=0;numeric_slacks=[]
    for pid in rejected_ids:
        ia,ib=map(int,pairs[pid])
        detail0=old_details.get(pid,{"pair_id":pid,"face_rows":[ia,ib],"reason":"REJECTED_BY_BASE_OWNER"})
        candidate=None
        if detail0.get("reason")=="NO_POSITIVE_BASELINE_GAP":
            candidate=_SHARED_FEATURE_OWNER._feature_rows_for_pair(p,fa[ia],fb[ib],pid)
        if candidate is None:
            unsafe_frozen.update(map(int,fa[ia]));unsafe_frozen.update(map(int,fb[ib]))
            unsafe_details.append(detail0);continue
        rows,feature_detail=candidate
        for row in rows:
            row["face_pair"]=[ia,ib]
            if not bool(row.get("shared_feature_strict_support",False)):
                nodes=np.asarray(row["point_rows"],np.int64)
                vectors=np.asarray(row["vectors"],np.float64)
                origin=np.asarray(row["origin"],np.float64)
                baseline_lhs=float(np.einsum("ij,ij->",vectors,
                    p[nodes].astype(np.float64)-origin[None,:]))
                old_upper=float(row["upper_bound"])
                if baseline_lhs>old_upper:
                    # Permit only the exact current Float64 evaluation plus one
                    # representable ULP. Final 805 source-pair audit is unchanged.
                    new_upper=float(np.nextafter(max(old_upper,baseline_lhs),np.inf))
                    row["upper_bound"]=new_upper
                    row["baseline_float64_lhs_m"]=baseline_lhs
                    row["baseline_float64_old_upper_m"]=old_upper
                    row["baseline_numeric_allowance_m"]=new_upper-old_upper
                    row["baseline_numeric_allowance_rule"]="nextafter(max(old_upper, same immutable Float64 LHS), +inf)"
                    numeric_ulp_rows+=1;numeric_slacks.append(new_upper-old_upper)
        safe_rows.extend(rows)
        accepted_shared.append({"pair_id":int(pid),"face_rows":[ia,ib],**feature_detail})
    constraints=base_rows+safe_rows;by={}
    for ri,row in enumerate(constraints):
        for node in np.unique(np.asarray(row["point_rows"],np.int64)):
            by.setdefault(int(node),[]).append(int(ri))
    base_prep.update({
        "usable_separating_pairs":int(base_prep.get("usable_separating_pairs",0)+len(accepted_shared)),
        "rejected_pairs":int(len(unsafe_details)),
        "linear_halfspace_rows":int(len(constraints)),
        "source_source_pair_count":int(base_prep.get("source_source_pair_count",0)+len(accepted_shared)),
        "rejected_pair_details":unsafe_details,
        "rejected_pair_details_omitted_by_owner_limit":0,
        "freeze_source_nodes":sorted(unsafe_frozen),
        "shared_feature_contact_pairs":int(len(accepted_shared)),
        "shared_feature_strict_support_pairs":int(sum(bool(x["strict_support"]) for x in accepted_shared)),
        "shared_feature_weak_support_pairs":int(sum(not bool(x["strict_support"]) for x in accepted_shared)),
        "shared_feature_contact_details":accepted_shared,
        "weak_feature_baseline_numeric_ulp_adjustments":int(numeric_ulp_rows),
        "weak_feature_max_baseline_numeric_allowance_m":float(max(numeric_slacks)) if numeric_slacks else 0.0,
        "weak_feature_baseline_numeric_allowance_rule":"only when same immutable Float64 row LHS exceeds zero upper bound; upper becomes nextafter(max(old upper,LHS),+inf); final exact 805 audit unchanged",
        "shared_feature_rule":"exact shared point/edge only; nonzero spread and support on both sides; nine relative halfspaces; final exact audit mandatory",
    })
    return {"constraints":constraints,"row_indices_by_point":by,"preparation":base_prep}

def _normalise_separation_rows(x):
    if x is None:return None
    if isinstance(x,dict) and "constraints" in x and "row_indices_by_point" in x:return x
    rows=list(x);by={}
    for ri,r in enumerate(rows):
        for n in np.unique(np.asarray(r["point_rows"],np.int64)):by.setdefault(int(n),[]).append(ri)
    return {"constraints":rows,"row_indices_by_point":by,"preparation":{"linear_halfspace_rows":len(rows)}}

def _separation_violation_rows(points,separation_rows,point_filter=None):
    if separation_rows is None:return []
    p=np.asarray(points,dtype=np.float64);obj=_normalise_separation_rows(separation_rows);rows=obj["constraints"]
    if point_filter is None:ids=range(len(rows))
    else:
        selected=set()
        for n in np.asarray(point_filter,np.int64).reshape(-1):selected.update(obj["row_indices_by_point"].get(int(n),()))
        ids=sorted(selected)
    bad=[]
    for ri in ids:
        r=rows[int(ri)];nodes=np.asarray(r["point_rows"],np.int64);v=np.asarray(r["vectors"],np.float64)
        lhs=float(np.einsum("ij,ij->",v,p[nodes]-np.asarray(r["origin"],np.float64)[None,:]))
        if lhs>float(r["upper_bound"]):bad.append(int(ri))
    return bad

_TRUST_ROUNDING_MAX_ULPS=16
def recover_f32_trust_rounding(baseline_nodes,rounded_target_nodes,trust_cap_m,*,max_ulps=_TRUST_ROUNDING_MAX_ULPS):
    """Correct only Float32 rounding overshoot with bounded nextafter steps toward baseline."""
    p0=np.asarray(baseline_nodes);trial=np.asarray(rounded_target_nodes)
    cap=float(trust_cap_m);limit=int(max_ulps)
    if p0.dtype!=np.float32 or trial.dtype!=np.float32 or p0.shape!=trial.shape or p0.ndim!=2 or p0.shape[1]!=3:
        raise ValueError("trust rounding expects equal Float32 [N,3] arrays")
    if not np.isfinite(cap) or cap<=0 or limit<0 or limit>_TRUST_ROUNDING_MAX_ULPS:
        raise ValueError("invalid trust cap or ULP recovery bound")
    out=trial.copy();initial_l1=np.abs(out.astype(np.float64)-p0.astype(np.float64)).sum(axis=1)
    initial_over=np.flatnonzero(initial_l1>cap);steps=np.zeros((len(p0),3),np.int16)
    for row in initial_over:
        for _ in range(limit):
            displacement=out[row].astype(np.float64)-p0[row].astype(np.float64)
            if float(np.abs(displacement).sum())<=cap:break
            axis=int(np.argmax(np.abs(displacement)))
            current=np.float32(out[row,axis]);target=np.float32(p0[row,axis])
            if current==target:break
            next_value=np.nextafter(current,target)
            if next_value==current:break
            out[row,axis]=next_value;steps[row,axis]+=1
    final_l1=np.abs(out.astype(np.float64)-p0.astype(np.float64)).sum(axis=1)
    final_over=np.flatnonzero(final_l1>cap)
    changed_rows=np.flatnonzero(np.any(out.view(np.uint32)!=trial.view(np.uint32),axis=1))
    return out,{"max_ulps_per_node":limit,"initial_overshoot_node_count":int(len(initial_over)),
        "recovered_node_count":int(len(changed_rows)),"ulp_steps_total":int(steps.sum()),
        "ulp_steps_per_node":steps.sum(axis=1).astype(int).tolist(),"coordinate_ulp_steps":steps.astype(int).tolist(),
        "remaining_overshoot_node_count":int(len(final_over)),"initial_max_l1_m":float(initial_l1.max(initial=0.)),
        "final_max_l1_m":float(final_l1.max(initial=0.)),"recovered_point_rows":changed_rows.astype(int).tolist()}

def componentwise_f32_recovery(points,tetrahedra,free_point_mask,active_point_rows,delta_m,trust_cap_m,
                               *,max_backtracks=12,recovery_time_limit_s=30.,separation_rows=None):
    # Backtrack each independent free-node/tet component separately.
    p0=np.asarray(points);t=np.asarray(tetrahedra,dtype=np.int64);free=np.asarray(free_point_mask,dtype=bool)
    active=np.asarray(active_point_rows,dtype=np.int64);delta=np.asarray(delta_m,dtype=np.float64);cap=float(trust_cap_m)
    if p0.dtype!=np.float32 or p0.ndim!=2 or p0.shape[1]!=3:raise ValueError("points must be Float32 [N,3]")
    if free.shape!=(len(p0),) or active.ndim!=1 or delta.shape!=(len(active),3):raise ValueError("component recovery shape mismatch")
    if np.any(active<0) or np.any(active>=len(p0)) or np.any(~free[active]):raise ValueError("active rows must be free global rows")
    affected=np.flatnonzero(np.any(free[t],axis=1));ta=t[affected]
    if not np.array_equal(np.unique(ta[free[ta]]),np.sort(active)):raise ValueError("active rows must exactly cover affected free vertices")
    if _separation_violation_rows(p0,separation_rows):raise ValueError("recovery baseline violates separation witness")
    det0,nd0=_M.measurements(p0,ta);good0=_M.passes(det0,nd0)
    qratio0=nd0/1.e-5;volratio0=det0/6.e-18
    local=np.full(len(p0),-1,dtype=np.int64);local[active]=np.arange(len(active),dtype=np.int64)
    ei=[];ej=[]
    for a in range(4):
        for b in range(a+1,4):
            rows=np.flatnonzero(free[ta[:,a]]&free[ta[:,b]])
            if len(rows):
                ei.append(local[ta[rows,a]]);ej.append(local[ta[rows,b]])
    if ei:
        u=np.concatenate(ei);v=np.concatenate(ej);ix=np.concatenate((u,v));iy=np.concatenate((v,u))
        graph=coo_matrix((np.ones(len(ix),np.uint8),(ix,iy)),shape=(len(active),len(active))).tocsr()
    else:graph=coo_matrix((len(active),len(active))).tocsr()
    # Witness halfspaces connect free point groups across otherwise separate tet components.
    if separation_rows is not None:
        si=[];sj=[]
        for row in _normalise_separation_rows(separation_rows)["constraints"]:
            locs=local[np.asarray(row["point_rows"],np.int64)];locs=locs[locs>=0]
            for i in range(len(locs)):
                for j in range(i+1,len(locs)):si.extend((int(locs[i]),int(locs[j])));sj.extend((int(locs[j]),int(locs[i])))
        if si:graph=graph+coo_matrix((np.ones(len(si),np.int32),(si,sj)),shape=(len(active),len(active))).tocsr()
    ncomp,labels=connected_components(graph,directed=False,return_labels=True)
    cell_component=np.full(len(ta),-1,dtype=np.int64)
    for c in range(4):
        rows=np.flatnonzero(free[ta[:,c]])
        if not len(rows):continue
        found=labels[local[ta[rows,c]]];old=cell_component[rows]
        if np.any((old>=0)&(old!=found)):raise AssertionError("one tet spans independent free-node components")
        cell_component[rows]=found
    if np.any(cell_component<0):raise AssertionError("affected tet lacks free-node component")
    node_counts=np.bincount(labels,minlength=ncomp);node_order=np.argsort(labels,kind="stable")
    node_offsets=np.r_[0,np.cumsum(node_counts)]
    cell_counts=np.bincount(cell_component,minlength=ncomp);cell_order=np.argsort(cell_component,kind="stable")
    cell_offsets=np.r_[0,np.cumsum(cell_counts)]
    first_rows=np.full(ncomp,len(active),dtype=np.int64);np.minimum.at(first_rows,labels,np.arange(len(active),dtype=np.int64))
    order=np.argsort(first_rows,kind="stable")
    working=p0.copy();reports=[];deadline=time.monotonic()+float(recovery_time_limit_s)
    accepted=0;full_bad=0;full_regressions=0;full_nonpositive=0;full_trust=0;full_l1=0.
    full_raw_trust=0;full_rounding_recovered=0;full_rounding_steps=0
    for cid in order:
        locnodes=node_order[node_offsets[cid]:node_offsets[cid+1]]
        nodes=active[locnodes];cellrows=cell_order[cell_offsets[cid]:cell_offsets[cid+1]]
        tcomp=ta[cellrows];before_good=good0[cellrows];before_bad=int(np.count_nonzero(~before_good))
        before_score=_log_deficit(qratio0[cellrows],volratio0[cellrows])
        merit_tolerance=max(1.e-8,1.e-6*before_score)
        old_nodes=working[nodes].copy();dcomp=delta[locnodes];attempts=[];accepted_alpha=None
        if time.monotonic()>=deadline:
            reports.append({"component":int(cid),"nodes":len(nodes),"affected_cells":len(cellrows),
                            "bad_before":before_bad,"status":"RECOVERY_TIME_LIMIT","attempts":[]})
            continue
        for bt in range(int(max_backtracks)+1):
            if time.monotonic()>=deadline:
                attempts.append({"alpha":float(.5**bt),"reason":"RECOVERY_TIME_LIMIT"});break
            alpha=.5**bt;raw_trial_nodes=(p0[nodes].astype(np.float64)+alpha*dcomp).astype(np.float32)
            raw_l1=np.abs(raw_trial_nodes.astype(np.float64)-p0[nodes].astype(np.float64)).sum(axis=1)
            trial_nodes,trust_rounding=recover_f32_trust_rounding(p0[nodes],raw_trial_nodes,cap)
            l1=np.abs(trial_nodes.astype(np.float64)-p0[nodes].astype(np.float64)).sum(axis=1)
            over=np.flatnonzero(l1>cap);working[nodes]=trial_nodes
            separation_bad=_separation_violation_rows(working,separation_rows,point_filter=nodes)
            d1,q1=_M.measurements(working,tcomp);good1=_M.passes(d1,q1)
            bad1=int(np.count_nonzero(~good1));nonpos=np.flatnonzero(~np.isfinite(d1)|(d1<=0));reg=np.flatnonzero(before_good&~good1)
            qratio1=q1/1.e-5;volratio1=d1/6.e-18
            score1=_log_deficit(qratio1,volratio1)
            rec={"alpha":alpha,"bad_before":before_bad,"bad_after":bad1,
                 "bad_score_before":before_score,"bad_score_after":score1,
                 "log_deficit_improvement":before_score-score1,"merit_tolerance":merit_tolerance,
                 "good_regression_count":int(len(reg)),
                 "nonpositive_count":int(len(nonpos)),"trust_overshoot_count":int(len(over)),
                 "raw_trust_overshoot_count":int(np.count_nonzero(raw_l1>cap)),
                 "trust_rounding_recovery":trust_rounding,
                 "separation_witness_violation_count":int(len(separation_bad)),
                 "max_node_l1_displacement_m":float(l1.max(initial=0.))}
            if bt==0:
                full_bad+=bad1;full_regressions+=len(reg);full_nonpositive+=len(nonpos);full_trust+=len(over)
                full_raw_trust+=int(np.count_nonzero(raw_l1>cap));full_rounding_recovered+=trust_rounding["recovered_node_count"]
                full_rounding_steps+=trust_rounding["ulp_steps_total"]
                full_l1=max(full_l1,float(l1.max(initial=0.)))
                rec["witness_tet_rows"]={"first_bad_after":[int(affected[cellrows[r]]) for r in np.flatnonzero(~good1)[:8]],
                    "first_good_regressions":[int(affected[cellrows[r]]) for r in reg[:8]],
                    "first_nonpositive":[int(affected[cellrows[r]]) for r in nonpos[:8]],
                    "first_trust_overshoot_nodes":[int(nodes[r]) for r in over[:8]]}
            if len(over):
                rec["reason"]="F32_TRUST_BOUND";rec["trust_overshoot_node_rows"]=[int(nodes[r]) for r in over[:8]]
            elif len(nonpos):rec["reason"]="F32_NONPOSITIVE"
            elif len(reg):rec["reason"]="GOOD_CELL_REGRESSION"
            elif len(separation_bad):rec["reason"]="SEPARATION_WITNESS_REGRESSION"
            elif bad1>before_bad:rec["reason"]="NATIVE_BAD_COUNT_REGRESSION"
            elif bad1==before_bad and before_score-score1<=merit_tolerance:
                rec["reason"]="NO_LEXICOGRAPHIC_PROGRESS"
            else:
                exact=_exact_positive(working,tcomp);badexact=[i for i,x in enumerate(exact) if x<=0]
                if badexact:
                    rec["reason"]="EXACT_F32_NONPOSITIVE"
                    rec["exact_witness_tet_rows"]=[int(affected[cellrows[i]]) for i in badexact[:8]]
                elif bad1<before_bad:
                    rec["reason"]="ACCEPTED_COUNT_REDUCTION";accepted_alpha=alpha;accepted+=1
                else:
                    rec["reason"]="ACCEPTED_LOG_DEFICIT_REDUCTION";accepted_alpha=alpha;accepted+=1
            if rec.get("reason") in ("GOOD_CELL_REGRESSION","F32_NONPOSITIVE","F32_TRUST_BOUND","EXACT_F32_NONPOSITIVE"):
                witness_local=np.unique(np.concatenate((reg,nonpos))).astype(np.int64)
                witness_rows=set(int(affected[cellrows[r]]) for r in witness_local)
                witness_rows.update(int(r) for r in rec.get("exact_witness_tet_rows",[]))
                if witness_rows:
                    rec["witness_tet_rows_full"]=sorted(witness_rows)
                    rec["witness_tet_node_rows"]={str(r):[int(x) for x in t[r]] for r in sorted(witness_rows)}
                if len(over):
                    rec["witness_node_rows"]=[int(nodes[r]) for r in over]
            attempts.append(rec)
            if accepted_alpha is not None:break
        if accepted_alpha is None:working[nodes]=old_nodes
        reports.append({"component":int(cid),"nodes":int(len(nodes)),"affected_cells":int(len(cellrows)),
            "bad_before":before_bad,"accepted_alpha":accepted_alpha,
            "status":"ACCEPTED" if accepted_alpha is not None else "UNCHANGED","attempts":attempts})
    if _separation_violation_rows(working,separation_rows):raise AssertionError("component merge violated coupled separation witness")
    d1,q1=_M.measurements(working,ta);good1=_M.passes(d1,q1);reg=np.flatnonzero(good0&~good1)
    if len(reg):raise AssertionError("component merge regressed previously-good cells")
    exact=_exact_positive(working,ta);badexact=[i for i,x in enumerate(exact) if x<=0]
    if badexact:raise AssertionError("component merge created exact-negative tet rows "+str([int(affected[i]) for i in badexact[:8]]))
    qratio1=q1/1.e-5;volratio1=d1/6.e-18
    score0=_log_deficit(qratio0,volratio0)
    score1=_log_deficit(qratio1,volratio1)
    bad0=int(np.count_nonzero(~good0));bad1=int(np.count_nonzero(~good1))
    count_progress=bad1<bad0
    merit_tolerance=max(1.e-8,1.e-6*score0)
    score_progress=(bad1==bad0 and score0-score1>merit_tolerance)
    lex_progress=count_progress or score_progress
    changed=np.flatnonzero(np.any(working.view(np.uint32)!=p0.view(np.uint32),axis=1))
    moved_l1=np.abs(working[changed].astype(np.float64)-p0[changed].astype(np.float64)).sum(axis=1)
    accepted_merit=sum(1 for comp in reports for attempt in comp.get("attempts",[]) if attempt.get("reason")=="ACCEPTED_LOG_DEFICIT_REDUCTION")
    report={"status":"PASS_COMPONENTWISE_LEX_PROGRESS" if lex_progress else "NO_COMPONENT_LEX_PROGRESS",
        "component_count":int(ncomp),"accepted_components":int(accepted),"accepted_log_deficit_components":int(accepted_merit),"affected_cells":int(len(ta)),
        "bad_before":bad0,"bad_after":bad1,
        "good_cell_regressions":0,"exact_F32_positive_cells":int(len(exact)),
        "bad_score_before":score0,"bad_score_after":score1,"bad_score_improvement":score0-score1,
        "merit_tolerance":merit_tolerance,"strict_quality_count_progress":bool(count_progress),
        "strict_bad_score_progress":bool(score_progress),"lexicographic_quality_progress":bool(lex_progress),
        "full_step_independent_component_bad_sum":int(full_bad),"full_step_good_regression_count":int(full_regressions),
        "full_step_nonpositive_count":int(full_nonpositive),"full_step_trust_overshoot_node_count":int(full_trust),
        "full_step_raw_trust_overshoot_node_count":int(full_raw_trust),
        "full_step_trust_rounding_recovered_node_count":int(full_rounding_recovered),
        "full_step_trust_rounding_ulp_steps_total":int(full_rounding_steps),
        "trust_rounding_policy":"bounded Float32 nextafter toward baseline, at most 16 ULP per node",
        "full_step_max_node_l1_displacement_m":float(full_l1),
        "max_free_node_l1_displacement_m":float(moved_l1.max(initial=0.)),
        "changed_point_rows":changed.astype(int).tolist(),"component_attempts":reports}
    return working,report

def _propose_quality_repair_once(points,tetrahedra,material_indices,free_point_mask,trust_cap_m,
                           *,linear_margin=1.e-3,time_limit_s=10.,max_backtracks=8,recovery_time_limit_s=30.,
                           separation_rows=None):
 """Return a Float32 point candidate and diagnostics for one coupled sparse LP.

 One global point row is one variable, including a shared Skin/Fat interface
 row. Every tet incident to a free row contributes a linearized quality row.
 Good cells are hard constrained with margin; bad cells have minimized,
 normalized slack. Final replay checks the owning Float32 predicate, exact
 determinant signs, and zero formerly-good-cell regressions. This is not
 physical or native admission.
 """
 p0=np.asarray(points);t=np.asarray(tetrahedra);mat=np.asarray(material_indices);free=np.asarray(free_point_mask)
 if p0.dtype!=np.float32 or p0.ndim!=2 or p0.shape[1]!=3:raise ValueError("points must be Float32 [N,3]")
 if t.ndim!=2 or t.shape[1]!=4 or t.dtype.kind not in "iu":raise ValueError("tetrahedra must be integer [M,4]")
 t=np.asarray(t,dtype=np.int64)
 if mat.shape!=(len(t),):raise ValueError("material_indices must have one row per tet")
 if free.shape!=(len(p0),) or free.dtype.kind!="b":raise ValueError("free_point_mask must be bool [N]")
 if not np.isfinite(p0).all() or not np.isfinite(float(trust_cap_m)):raise ValueError("nonfinite input")
 if trust_cap_m<=0 or linear_margin<0 or not np.isfinite(linear_margin):raise ValueError("invalid trust/margin")
 if time_limit_s<=0 or not np.isfinite(time_limit_s) or max_backtracks<0:raise ValueError("invalid solver budget")
 if np.any(t<0) or np.any(t>=len(p0)) or np.any(np.diff(np.sort(t,axis=1),axis=1)==0):raise ValueError("invalid tet connectivity")
 candidate0=p0.copy();affected=np.flatnonzero(np.any(free[t],axis=1))
 if not len(affected):
  return candidate0,{"status":"NO_AFFECTED_CELLS","physical_admission":False,"native_steps":0,
                     "quality_owner_sha256":QUALITY_OWNER_SHA256,"analytic_lp_reference_sha256":LP_REFERENCE_SHA256}
 ta=t[affected];det0,nd0=_M.measurements(p0,ta);good0=_M.passes(det0,nd0);_,target=_targets(p0,ta)
 if not np.isfinite(det0).all() or not np.isfinite(nd0).all() or np.any(target<=0):raise ValueError("invalid affected-cell metrics")
 nbad=int(np.count_nonzero(~good0));incident=np.unique(ta);active=incident[free[incident]]
 separation_rows=_normalise_separation_rows(separation_rows)
 sep_preparation=(separation_rows or {}).get("preparation",{})
 sep_baseline_bad=_separation_violation_rows(p0,separation_rows)
 common={"physical_admission":False,"native_steps":0,"quality_owner_sha256":QUALITY_OWNER_SHA256,
         "analytic_lp_reference_sha256":LP_REFERENCE_SHA256,"affected_cells":int(len(ta)),
         "separation_witness_constraints":int(len((separation_rows or {}).get("constraints",[]))),
         "separation_witness_preparation":{k:sep_preparation.get(k) for k in ("exact_pair_witnesses","usable_separating_pairs","rejected_pairs","linear_halfspace_rows","retained_gap_fraction")},
         "free_nodes_used":int(len(active)),"bad_before":nbad,"good_before":int(np.count_nonzero(good0)),
         "linear_margin":float(linear_margin),"trust_cap_m":float(trust_cap_m),
         "trust_norm":"per-node L1 displacement","all_affected_good_and_bad_rows_constrained":True,
         "frozen_input_longest_edge_target":True,"shared_global_point_rows_are_single_variables":True,
         "material_cell_counts":{str(int(m)):int(np.count_nonzero(mat[affected]==m)) for m in np.unique(mat[affected])}}
 if sep_baseline_bad:return candidate0,{"status":"SEPARATION_BASELINE_INFEASIBLE","separation_baseline_violation_count":len(sep_baseline_bad),"first_violating_constraint_rows":sep_baseline_bad[:16],**common}
 if nbad==0:return candidate0,{"status":"NO_AFFECTED_BAD_CELLS",**common}
 if not len(active):return candidate0,{"status":"NO_FREE_VARIABLES_FOR_AFFECTED_BAD_CELLS",**common}
 local=np.full(len(p0),-1,dtype=np.int64);local[active]=np.arange(len(active))
 grad_det=determinant_gradients(p0,ta)
 xyz=p0[ta].astype(np.float64)
 edge_pairs=np.asarray([(i,j) for i in range(4) for j in range(i+1,4)],dtype=np.int64)
 edge_vec=xyz[:,edge_pairs[:,0],:]-xyz[:,edge_pairs[:,1],:]
 edge_len=np.linalg.norm(edge_vec,axis=2);edge_choice=np.argmax(edge_len,axis=1)
 longest=edge_len[np.arange(len(ta)),edge_choice]
 if np.any(longest<=0):raise ValueError("degenerate longest edge")
 qratio0=nd0/1.e-5;volratio0=det0/6.e-18
 qpass=qratio0>=1.;vpass=det0/6.>1.e-18
 if not np.array_equal(good0,qpass&vpass):raise AssertionError("native predicate and separate quality/volume rows disagree")
 edge_i=edge_pairs[edge_choice,0];edge_j=edge_pairs[edge_choice,1]
 unit=edge_vec[np.arange(len(ta)),edge_choice]/longest[:,None]
 grad_long=np.zeros((len(ta),4,3),dtype=np.float64)
 rows0=np.arange(len(ta));grad_long[rows0,edge_i]=unit;grad_long[rows0,edge_j]=-unit
 qgrad=grad_det/(1.e-5*longest[:,None,None]**3)-3.*qratio0[:,None,None]*grad_long/longest[:,None,None]
 vgrad=grad_det/6.e-18
 margin=1.+float(linear_margin)
 qfloor=np.where(qpass,np.minimum(qratio0,margin),margin)
 vfloor=np.where(vpass,np.minimum(volratio0,margin),margin)
 rhs_q=qratio0-qfloor;rhs_v=volratio0-vfloor
 qbad=np.flatnonzero(~qpass);vbad=np.flatnonzero(~vpass)
 qbadlocal=np.full(len(ta),-1,dtype=np.int64);qbadlocal[qbad]=np.arange(len(qbad),dtype=np.int64)
 vbadlocal=np.full(len(ta),-1,dtype=np.int64);vbadlocal[vbad]=np.arange(len(vbad),dtype=np.int64)
 k=len(active);ndelta=3*k;nabs=3*k;slackbase=ndelta+nabs
 qslackbase=slackbase;vslackbase=qslackbase+len(qbad);nvar=vslackbase+len(vbad)
 ci,corner=np.nonzero(free[ta]);pl=local[ta[ci,corner]]
 if np.any(pl<0):raise AssertionError("free row lacks a variable")
 sgq=qgrad*float(trust_cap_m);sgv=vgrad*float(trust_cap_m)
 ncell=len(ta);rhs=np.concatenate((rhs_q,rhs_v))
 rowmax=np.maximum(1.,np.abs(rhs))
 if len(ci):
  np.maximum.at(rowmax,ci,np.max(np.abs(sgq[ci,corner]),axis=1))
  np.maximum.at(rowmax,ncell+ci,np.max(np.abs(sgv[ci,corner]),axis=1))
 scale=1./rowmax
 # The final full-row equilibration below includes all coefficients, including
 # failing-cell slack columns, and scales each RHS by the same positive factor.
 ar=[];ac=[];av=[]
 if len(ci):
  basecol=(3*pl[:,None]+np.arange(3)[None,:]).reshape(-1).tolist()
  ar.extend(np.repeat(ci,3).tolist());ac.extend(basecol)
  av.extend((-sgq[ci,corner]*scale[ci,None]).reshape(-1).tolist())
  ar.extend(np.repeat(ncell+ci,3).tolist());ac.extend(basecol)
  av.extend((-sgv[ci,corner]*scale[ncell+ci,None]).reshape(-1).tolist())
 for r in qbad:ar.append(int(r));ac.append(int(qslackbase+qbadlocal[r]));av.append(-float(scale[r]))
 for r in vbad:ar.append(int(ncell+r));ac.append(int(vslackbase+vbadlocal[r]));av.append(-float(scale[ncell+r]))
 bcell=np.concatenate((rhs_q*scale[:ncell],rhs_v*scale[ncell:]))
 row=2*ncell
 for j in range(ndelta):
  ar.extend((row,row,row+1,row+1));ac.extend((j,ndelta+j,j,ndelta+j));av.extend((1.,-1.,-1.,-1.));row+=2
 trustrow=row
 for i in range(k):
  ar.extend([trustrow+i]*3);ac.extend((nabs+3*i+c for c in range(3)));av.extend((1.,1.,1.))
 nrows=trustrow+k;b=np.concatenate((bcell,np.zeros(6*k),np.ones(k)))
 A=coo_matrix((np.asarray(av),(np.asarray(ar),np.asarray(ac))),shape=(nrows,nvar)).tocsr()
 sep_b=[];sep_r=[];sep_c=[];sep_v=[]
 for si,srow in enumerate((separation_rows or {}).get("constraints",[])):
  sn=np.asarray(srow["point_rows"],np.int64);sv=np.asarray(srow["vectors"],np.float64);origin=np.asarray(srow["origin"],np.float64)
  lhs=float(np.einsum("ij,ij->",sv,p0[sn].astype(np.float64)-origin[None,:]))
  rhs=float(srow["upper_bound"])-lhs;plane_scale=max(abs(rhs),float(np.max(np.abs(sv),initial=0.))*float(trust_cap_m),1.e-12)
  sep_b.append(rhs/plane_scale)
  for node,vec in zip(sn,sv):
   loc=int(local[int(node)])
   if loc>=0:
    for ax in range(3):
     val=float(vec[ax])*float(trust_cap_m)/plane_scale
     if val:sep_r.append(si);sep_c.append(3*loc+ax);sep_v.append(val)
 if sep_b:
  Asep=coo_matrix((np.asarray(sep_v),(np.asarray(sep_r),np.asarray(sep_c))),shape=(len(sep_b),nvar)).tocsr()
  A=vstack((A,Asep),format="csr");b=np.concatenate((b,np.asarray(sep_b,np.float64)));nrows=int(A.shape[0])
  common["separation_linear_rows_added"]=len(sep_b)
 else:common["separation_linear_rows_added"]=0
 # At zero displacement, each failing predicate receives precisely its required slack.
 baseline=np.zeros(nvar,dtype=np.float64)
 baseline[qslackbase:vslackbase]=np.maximum(0.,-rhs_q[qbad])
 baseline[vslackbase:]=np.maximum(0.,-rhs_v[vbad])
 cell_lhs=np.asarray(A[:2*ncell].dot(baseline)).ravel()
 baseline_violation=np.maximum(cell_lhs-bcell,0.)
 baseline_max_violation=float(baseline_violation.max(initial=0.))
 baseline_roundoff=64.*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(bcell),initial=0.)))
 if baseline_max_violation>baseline_roundoff:raise AssertionError(f"zero-displacement baseline infeasible by {baseline_max_violation:g}")
 A,b,row_equilibration=_equilibrate_inequality_rows(A,b)
 scaled_cell_lhs=np.asarray(A[:2*ncell].dot(baseline)).ravel()
 scaled_cell_violation=np.maximum(scaled_cell_lhs-b[:2*ncell],0.)
 scaled_baseline_max_violation=float(scaled_cell_violation.max(initial=0.))
 scaled_baseline_roundoff=64.*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(b[:2*ncell]),initial=0.)))
 if scaled_baseline_max_violation>scaled_baseline_roundoff:raise AssertionError(f"row-equilibrated zero-displacement baseline infeasible by {scaled_baseline_max_violation:g}")
 matrix_diag=dict(row_equilibration["post_equilibration"])
 matrix_diag.update({
  "row_equilibration":row_equilibration,
  "good_cell_rows":int(np.count_nonzero(good0)),"bad_cells":int(nbad),
  "quality_passing_rows":int(len(ta)-len(qbad)),"quality_slack_rows":int(len(qbad)),
  "volume_passing_rows":int(len(ta)-len(vbad)),"volume_slack_rows":int(len(vbad)),
  "quality_good_rows_preserved_at_current_ratio":int(np.count_nonzero(qpass&(qratio0<margin))),
  "quality_good_rows_receiving_margin":int(np.count_nonzero(qpass&(qratio0>=margin))),
  "volume_good_rows_preserved_at_current_ratio":int(np.count_nonzero(vpass&(volratio0<margin))),
  "volume_good_rows_receiving_margin":int(np.count_nonzero(vpass&(volratio0>=margin))),
  "longest_edge_gradient":"first lexicographic longest edge; normalized q=det6/(1e-5*L^3)",
  "longest_edge_tie_cell_count":int(np.count_nonzero((edge_len==longest[:,None]).sum(axis=1)>1)),
  "zero_displacement_baseline_feasible":True,"zero_displacement_max_cell_violation":baseline_max_violation,
  "row_equilibrated_zero_displacement_max_cell_violation":scaled_baseline_max_violation,
  "quality_bad_rows_pre_equilibration_unit_slack":bool(np.all(scale[qbad]==1.)),
  "volume_bad_rows_pre_equilibration_unit_slack":bool(np.all(scale[ncell+vbad]==1.)),
 })
 cost=np.zeros(nvar);cost[ndelta:slackbase]=1.e-6/max(k,1)
 cost[qslackbase:vslackbase]=.5/max(len(qbad),1)
 cost[vslackbase:]=.5/max(len(vbad),1)
 qbounds=[(0.,max(1.,float(max(0.,-rhs_q[r]))+1.)) for r in qbad]
 vbounds=[(0.,max(1.,float(max(0.,-rhs_v[r]))+1.)) for r in vbad]
 bounds=[(None,None)]*ndelta+[(0.,1.)]*nabs+qbounds+vbounds
 cost_nonzero=np.abs(cost[cost!=0.])
 objective_diag={"coefficient_abs_min_nonzero":float(cost_nonzero.min()) if len(cost_nonzero) else None,
                 "coefficient_abs_max":float(cost_nonzero.max()) if len(cost_nonzero) else None,
                 "coefficient_abs_ratio_max_to_min":float(cost_nonzero.max()/cost_nonzero.min()) if len(cost_nonzero) else None,
                 "objective_coefficients_rescaled":False,"solver_cost_vector_unchanged":True}
 res,solver_attempts,primary_status,solver_elapsed=_solve_highs_with_status4_fallback(cost,A,b,bounds,time_limit_s)
 common.update({"solver_status":int(res.status),"solver_message":str(res.message),
                "solver_objective":float(res.fun) if res.fun is not None else None,
                "solver_objective_original_units":float(res.fun) if res.fun is not None else None,
                "objective_policy":"bad-slack-dominant weighted L1 motion",
                "objective_diagnostics":objective_diag,
                "solver_attempts":solver_attempts,"primary_solver_status":primary_status,
                "solver_elapsed_s":solver_elapsed,
                "per_abs_displacement_objective_weight":1.e-6/max(k,1),
                "quality_bad_slack_objective_weight":.5/max(len(qbad),1),
                "volume_bad_slack_objective_weight":.5/max(len(vbad),1),
                "linear_bad_slack_sum":float(np.sum(res.x[slackbase:])) if res.x is not None else None,
                "linear_constraint_rows":int(nrows),"linear_constraint_nnz":int(A.nnz),
                "matrix_diagnostics":matrix_diag,"solver_method":solver_attempts[-1]["method"]})
 if not res.success or res.x is None:return candidate0,{"status":"LP_NO_CANDIDATE",**common}
 delta=res.x[:ndelta].reshape(k,3)*float(trust_cap_m)
 candidate,recovery=componentwise_f32_recovery(p0,t,free,active,delta,trust_cap_m,
                                                max_backtracks=max_backtracks,recovery_time_limit_s=recovery_time_limit_s,
                                                separation_rows=separation_rows)
 common.update({key:value for key,value in recovery.items() if key not in ("component_attempts","changed_point_rows","status")})
 common.update({"component_attempts":recovery["component_attempts"],
                "point_rows_changed":recovery["changed_point_rows"],
                "topology_unchanged":True,"candidate_is_construction_proposal_only":True,
                "componentwise_recovery":True,"linear_bad_slack_sum":float(np.sum(res.x[slackbase:]))})
 if recovery["status"]!="PASS_COMPONENTWISE_LEX_PROGRESS":
  return candidate0,{"status":recovery["status"],**common}
 final_det,final_quality=_M.measurements(candidate,ta);final_good=_M.passes(final_det,final_quality)
 common.update({"status":"PASS_F32_COUPLED_QUALITY_PROPOSAL","bad_after":int(np.count_nonzero(~final_good)),
                "good_after":int(np.count_nonzero(final_good)),
                "strict_quality_count_progress":bool(recovery["strict_quality_count_progress"]),
                "strict_bad_score_progress":bool(recovery["strict_bad_score_progress"]),
                "lexicographic_quality_progress":True,
                "per_step_trust_cap_m":float(trust_cap_m),
                "max_free_node_l1_displacement_bound_m":float(trust_cap_m),
                "accepted_component_count":recovery["accepted_components"],
                "exact_F32_positive_cells":recovery["exact_F32_positive_cells"],
                "bad_score_before":recovery["bad_score_before"],"bad_score_after":recovery["bad_score_after"]})
 return candidate,common



def propose_quality_repair(points,tetrahedra,material_indices,free_point_mask,trust_cap_m,
                           *,linear_margin=1.e-3,time_limit_s=30.,max_backtracks=12,
                           recovery_time_limit_s=30.,max_inner_iterations=10,separation_rows=None):
 """Run up to ten continuous SLP steps under one immutable total trust cap.

 Each LP and its Float32 recovery receive half the remaining total wall budget.
 Every step is limited to total_cap/10; all accumulated moves are rechecked
 against the immutable input points and native gates before returning.
 """
 p0=np.asarray(points);t=np.asarray(tetrahedra,dtype=np.int64);mat=np.asarray(material_indices)
 free=np.asarray(free_point_mask,dtype=bool);cap=float(trust_cap_m)
 budget=float(time_limit_s);niter=int(max_inner_iterations)
 if p0.dtype!=np.float32 or p0.ndim!=2 or p0.shape[1]!=3:raise ValueError("points must be Float32 [N,3]")
 if t.ndim!=2 or t.shape[1]!=4 or mat.shape!=(len(t),) or free.shape!=(len(p0),):raise ValueError("SLP input shape mismatch")
 if not np.isfinite(cap) or cap<=0 or niter<1 or niter>10:raise ValueError("invalid total trust cap or SLP iteration count")
 if not np.isfinite(budget) or budget<=0 or budget>60 or not np.isfinite(recovery_time_limit_s) or recovery_time_limit_s<=0 or recovery_time_limit_s>60:
  raise ValueError("total SLP/recovery budget must be in (0,60] seconds")
 if np.any(t<0) or np.any(t>=len(p0)) or np.any(np.diff(np.sort(t,axis=1),axis=1)==0):raise ValueError("invalid tetrahedron connectivity")
 start=time.monotonic();deadline=start+budget
 separation_rows=_normalise_separation_rows(separation_rows)
 if _separation_violation_rows(p0,separation_rows):raise ValueError("immutable SLP baseline violates separation witnesses")
 affected=np.flatnonzero(np.any(free[t],axis=1));ta=t[affected]
 if not len(affected):
  return p0.copy(),{"status":"NO_AFFECTED_CELLS","physical_admission":False,"native_steps":0,
                    "bad_before":0,"bad_after":0,"inner_iterations":[],"max_free_node_l1_displacement_bound_m":0.}
 base_det,base_quality=_M.measurements(p0,ta);base_good=np.asarray(_M.passes(base_det,base_quality),bool)
 if np.any(base_det<=0) or not np.isfinite(base_det).all() or not np.isfinite(base_quality).all():
  raise ValueError("SLP immutable baseline contains a nonpositive/nonfinite affected cell")
 base_bad=int(np.count_nonzero(~base_good))
 base_score=_log_deficit(base_quality/1.e-5,base_det/6.e-18)
 current=p0.copy();cur_det=base_det;cur_quality=base_quality;cur_good=base_good.copy()
 cur_bad=base_bad;cur_score=base_score
 step_cap=cap/10.
 iteration_rows=[];last_diag={}
 for iteration in range(niter):
  remaining=deadline-time.monotonic()
  if remaining<=.02:break
  inner_budget=remaining/2.
  candidate,diag=_propose_quality_repair_once(
      current,t,mat,free,step_cap,linear_margin=linear_margin,
      time_limit_s=inner_budget,max_backtracks=max_backtracks,
      recovery_time_limit_s=min(float(recovery_time_limit_s),inner_budget),
      separation_rows=separation_rows)
  last_diag=diag
  if not np.array_equal(candidate[~free].view(np.uint32),current[~free].view(np.uint32)):
   raise AssertionError("SLP iteration moved a frozen point row")
  d,q=_M.measurements(candidate,ta);good=np.asarray(_M.passes(d,q),bool)
  if np.any(d<=0) or not np.isfinite(d).all() or not np.isfinite(q).all():
   raise AssertionError("SLP iteration produced nonpositive/nonfinite Float32 cell")
  regress=np.flatnonzero(base_good&~good)
  if len(regress):raise AssertionError("SLP iteration regressed immutable-baseline good cells")
  sep_bad=_separation_violation_rows(candidate,separation_rows)
  bad=int(np.count_nonzero(~good));score=_log_deficit(q/1.e-5,d/6.e-18)
  moved=np.flatnonzero(np.any(candidate.view(np.uint32)!=p0.view(np.uint32),axis=1))
  l1=np.abs(candidate[moved].astype(np.float64)-p0[moved].astype(np.float64)).sum(axis=1)
  max_l1=float(l1.max(initial=0.))
  cap_ok=max_l1<=cap
  tol=max(1.e-8,1.e-6*cur_score)
  count_progress=bad<cur_bad
  merit_progress=(bad==cur_bad and cur_score-score>tol)
  accepted=bool(cap_ok and (count_progress or merit_progress))
  if not cap_ok:reason="IMMUTABLE_TOTAL_TRUST_CAP"
  elif sep_bad:reason="SEPARATION_WITNESS_REGRESSION"
  elif count_progress:reason="NATIVE_BAD_COUNT_REDUCTION"
  elif merit_progress:reason="TIED_BAD_COUNT_LOG_DEFICIT_REDUCTION"
  else:reason="NO_LEXICOGRAPHIC_PROGRESS"
  iterrow={
      "iteration":iteration+1,"proposal_status":diag.get("status"),
      "bad_before":cur_bad,"bad_after":bad,"bad_score_before":cur_score,
      "bad_score_after":score,"log_deficit_improvement":cur_score-score,
      "merit_tolerance":tol,"strict_quality_count_progress":count_progress,
      "strict_bad_score_progress":bool(bad==cur_bad and cur_score-score>tol),
      "accepted":accepted,"reason":reason,
      "solver_status":diag.get("solver_status"),"solver_method":diag.get("solver_method"),
      "solver_attempts":diag.get("solver_attempts"),"matrix_diagnostics":diag.get("matrix_diagnostics"),
      "separation_witness_violation_count":int(len(sep_bad)),
      "accepted_components":int(diag.get("accepted_components",0)),
      "accepted_log_deficit_components":int(diag.get("accepted_log_deficit_components",0)),
      "per_step_trust_cap_m":step_cap,"maximum_cumulative_L1_m":max_l1,
      "elapsed_s":budget-(deadline-time.monotonic()),
  }
  iteration_rows.append(iterrow)
  if not accepted:break
  current=candidate.copy();cur_det=d;cur_quality=q;cur_good=good;cur_bad=bad;cur_score=score
  if cur_bad==0:break
 final_sep_bad=_separation_violation_rows(current,separation_rows)
 if final_sep_bad:raise AssertionError("final SLP output violates actual-pair separation witnesses")
 final_det,final_quality=_M.measurements(current,ta);final_good=np.asarray(_M.passes(final_det,final_quality),bool)
 if np.any(base_good&~final_good):raise AssertionError("final SLP candidate regresses immutable-baseline good cells")
 if np.any(final_det<=0) or not np.isfinite(final_det).all():raise AssertionError("final SLP candidate is not Float32-positive")
 moved=np.flatnonzero(np.any(current.view(np.uint32)!=p0.view(np.uint32),axis=1))
 l1=np.abs(current[moved].astype(np.float64)-p0[moved].astype(np.float64)).sum(axis=1)
 max_l1=float(l1.max(initial=0.))
 if max_l1>cap:raise AssertionError("final SLP candidate exceeds immutable total trust cap")
 exact=_exact_positive(current,ta)
 if any(x<=0 for x in exact):raise AssertionError("final SLP candidate fails exact determinant positivity")
 final_bad=int(np.count_nonzero(~final_good))
 final_score=_log_deficit(final_quality/1.e-5,final_det/6.e-18)
 final_tol=max(1.e-8,1.e-6*base_score)
 count_progress=final_bad<base_bad
 score_progress=final_bad==base_bad and base_score-final_score>final_tol
 lex_progress=count_progress or score_progress
 if not lex_progress:current=p0.copy()
 changed=np.flatnonzero(np.any(current.view(np.uint32)!=p0.view(np.uint32),axis=1))
 # Preserve the final iteration's detailed component recovery; outer history is compact.
 component_attempts=last_diag.get("component_attempts",[])
 return current,{
   "status":"PASS_F32_COUPLED_QUALITY_PROPOSAL" if lex_progress else "NO_LEXICOGRAPHIC_PROGRESS",
   "physical_admission":False,"native_steps":0,
   "bad_before":base_bad,"bad_after":final_bad if lex_progress else base_bad,
   "good_before":int(np.count_nonzero(base_good)),
   "good_after":int(np.count_nonzero(final_good)) if lex_progress else int(np.count_nonzero(base_good)),
   "bad_score_before":base_score,"bad_score_after":final_score if lex_progress else base_score,
   "strict_quality_count_progress":bool(count_progress),
   "strict_bad_score_progress":bool(score_progress),
   "lexicographic_quality_progress":bool(lex_progress),
   "merit_tolerance":final_tol,
   "inner_iteration_count":len(iteration_rows),"inner_iterations":iteration_rows,
   "separation_witness_constraints":int(len((separation_rows or {}).get("constraints",[]))),
   "separation_witness_preparation":(separation_rows or {}).get("preparation",{}),
   "accepted_components":int(sum(int(x.get("accepted_components",0)) for x in iteration_rows if x.get("accepted"))),
   "accepted_log_deficit_components":int(sum(int(x.get("accepted_log_deficit_components",0)) for x in iteration_rows if x.get("accepted"))),
   "good_cell_regressions":0,"exact_F32_positive_cells":int(len(exact)),
   "max_free_node_l1_displacement_bound_m":max_l1 if lex_progress else 0.,
   "per_step_trust_cap_m":step_cap,"total_trust_cap_m":cap,
   "trust_accumulation_policy":"10 steps maximum; each step cap is caller total cap divided by 10; final cumulative per-node L1 rechecked against immutable input",
   "changed_point_rows":changed.astype(int).tolist(),
   "component_attempts":component_attempts,
   "last_inner_diagnostics":{k:last_diag.get(k) for k in ("solver_status","solver_message","linear_constraint_rows","linear_constraint_nnz","linear_bad_slack_sum","component_count","bad_score_before","bad_score_after","strict_quality_count_progress","strict_bad_score_progress")},
   "solver_status":last_diag.get("solver_status"),"solver_message":last_diag.get("solver_message"),
   "quality_owner_sha256":QUALITY_OWNER_SHA256,
   "analytic_lp_reference_sha256":LP_REFERENCE_SHA256,
   "elapsed_s":time.monotonic()-start,
   "physical_admission":False,"native_steps":0,
 }



def run_v12_controls():
 """Small real-LP control: tied bad-count merit steps accumulate to a pass."""
 p=np.asarray(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,2.e-5),
               (4.,0.,0.),(5.,0.,0.),(4.,1.,0.),(4.,0.,1.)),dtype=np.float32)
 t=np.asarray(((0,1,2,3),(4,5,6,7)),dtype=np.int64)
 mat=np.asarray((15,16),dtype=np.int32)
 free=np.zeros(len(p),dtype=bool);free[3]=True
 d0,q0=_M.measurements(p,t);g0=np.asarray(_M.passes(d0,q0),bool)
 if int(np.count_nonzero(~g0))!=1 or not bool(g0[1]):raise AssertionError("control fixture did not contain one bad and one good cell")
 cap=1.e-5
 candidate,diag=propose_quality_repair(p,t,mat,free,cap,time_limit_s=10.,max_backtracks=12,max_inner_iterations=10)
 d1,q1=_M.measurements(candidate,t);g1=np.asarray(_M.passes(d1,q1),bool)
 moved=np.flatnonzero(np.any(candidate.view(np.uint32)!=p.view(np.uint32),axis=1))
 max_l1=float(np.abs(candidate[moved].astype(np.float64)-p[moved].astype(np.float64)).sum(axis=1).max(initial=0.))
 rows=diag.get("inner_iterations",[])
 ties=[x for x in rows if x["bad_after"]==x["bad_before"] and x["bad_score_before"]-x["bad_score_after"]>x["merit_tolerance"]]
 crossings=[x for x in rows if x["bad_after"]<x["bad_before"]]
 if not ties:raise AssertionError("control failed to accept tied-count log-deficit improvement")
 if not crossings or int(np.count_nonzero(~g1))!=0:raise AssertionError("control failed to cross native quality threshold later")
 if np.any(g0&~g1):raise AssertionError("control regressed a formerly-good cell")
 if np.any(d1<=0) or not all(x>0 for x in _exact_positive(candidate,t)):raise AssertionError("control failed positive/exact native gates")
 if max_l1>cap:raise AssertionError("control exceeded immutable total trust cap")
 return {"status":"PASS_V12_TIED_DEFICIT_SLP_CONTROLS",
         "controls":["tied bad count with strict log-deficit progress accepted","later inner step crosses native threshold","formerly-good cell remains passing","Float32 and exact determinant positivity","immutable total per-node L1 cap"],
         "iterations":len(rows),"accepted_tied_merit_steps":len(ties),
         "threshold_crossing_iteration":crossings[0]["iteration"],
         "bad_before":int(np.count_nonzero(~g0)),"bad_after":int(np.count_nonzero(~g1)),
         "bad_score_before":diag["bad_score_before"],"bad_score_after":diag["bad_score_after"],
         "max_cumulative_L1_m":max_l1,"total_trust_cap_m":cap,
         "good_cell_regressions":0,"physical_admission":False,"native_steps":0}


def select_active_set_freezes(tetrahedra,free_point_mask,component_reports):
 """Freeze movable rows of one deterministic bad-cell witness per blocked component."""
 t=np.asarray(tetrahedra,dtype=np.int64);free=np.asarray(free_point_mask,dtype=bool)
 if t.ndim!=2 or t.shape[1]!=4 or free.ndim!=1:raise ValueError("active-set shape mismatch")
 frozen=set();records=[]
 failures={"GOOD_CELL_REGRESSION","F32_NONPOSITIVE","F32_TRUST_BOUND","EXACT_F32_NONPOSITIVE"}
 for comp in component_reports:
  if comp.get("status")!="UNCHANGED":continue
  candidates=[a for a in comp.get("attempts",[]) if a.get("reason") in failures
              and int(a.get("bad_after",a.get("bad_before",0)))<int(a.get("bad_before",0))]
  if not candidates:continue
  candidates.sort(key=lambda a:(-(int(a["bad_before"])-int(a["bad_after"])),
                                -float(a.get("alpha",0.)),str(a.get("reason",""))))
  attempt=candidates[0];rows=list(attempt.get("witness_tet_rows_full",[]))
  if not rows:
   w=attempt.get("witness_tet_rows",{})
   rows=list(w.get("first_good_regressions",[]))+list(w.get("first_nonpositive",[]))
  rows=sorted(set(int(r) for r in rows if 0<=int(r)<len(t)))
  before=set(frozen)
  tr=rows[0] if rows else None
  if tr is not None:frozen.update(int(n) for n in t[tr] if free[int(n)])
  if attempt.get("reason")=="F32_TRUST_BOUND":
   frozen.update(int(n) for n in attempt.get("witness_node_rows",[]) if 0<=int(n)<len(free) and free[int(n)])
  new=sorted(frozen-before)
  if new:
   records.append({"component":int(comp.get("component",-1)),
                   "bad_before":int(attempt["bad_before"]),"bad_after":int(attempt["bad_after"]),
                   "reason":str(attempt["reason"]),"witness_tet_row":tr,
                   "newly_frozen_global_point_rows":new})
 return np.asarray(sorted(frozen),dtype=np.int64),records

def propose_quality_repair_active_set(points,tetrahedra,material_indices,free_point_mask,trust_cap_m,
                                      *,max_active_rounds=3,linear_margin=1.e-3,
                                      time_limit_s=60.,max_backtracks=12,separation_rows=None):
 """Re-solve from one baseline, freezing only witnessed blocked-cell rows."""
 p0=np.asarray(points);t=np.asarray(tetrahedra,dtype=np.int64);mat=np.asarray(material_indices)
 free0=np.asarray(free_point_mask,dtype=bool)
 if p0.dtype!=np.float32 or p0.ndim!=2 or p0.shape[1]!=3:raise ValueError("points must be Float32 [N,3]")
 if t.ndim!=2 or t.shape[1]!=4 or mat.shape!=(len(t),) or free0.shape!=(len(p0),):raise ValueError("active-set input shape mismatch")
 if int(max_active_rounds)<1 or not np.isfinite(time_limit_s) or time_limit_s<=0:raise ValueError("invalid active-set budget")
 deadline=time.monotonic()+float(time_limit_s)
 base_det,base_q=_M.measurements(p0,t);base_good=_M.passes(base_det,base_q)
 base_bad=int(np.count_nonzero(~base_good));free=free0.copy();best=p0.copy();best_bad=base_bad;best_score=float("inf")
 rounds=[];all_frozen=set()
 for rid in range(int(max_active_rounds)):
  remaining=deadline-time.monotonic()
  if remaining<=.05:break
  cand,diag=propose_quality_repair(p0,t,mat,free,trust_cap_m,
       linear_margin=linear_margin,time_limit_s=remaining,max_backtracks=max_backtracks,
       recovery_time_limit_s=remaining,separation_rows=separation_rows)
  d,q=_M.measurements(cand,t);good=_M.passes(d,q);bad=int(np.count_nonzero(~good))
  regress=np.flatnonzero(base_good&~good);nonpositive=np.flatnonzero(~np.isfinite(d)|(d<=0))
  moved=np.flatnonzero(np.any(cand.view(np.uint32)!=p0.view(np.uint32),axis=1))
  l1=np.abs(cand[moved].astype(np.float64)-p0[moved].astype(np.float64)).sum(axis=1)
  if len(regress):raise AssertionError("active-set round regressed a formerly-good cell")
  if len(nonpositive):raise AssertionError("active-set round created a nonpositive cell")
  if np.any(l1>float(trust_cap_m)):raise AssertionError("active-set round exceeded baseline trust cap")
  score=float(diag.get("bad_score_after",float("inf")))
  if bad<best_bad or (bad==best_bad and score<best_score):
   best=cand.copy();best_bad=bad;best_score=score
  components=diag.get("component_attempts",[])
  frozen,frecords=select_active_set_freezes(t,free,components)
  new=frozen[free[frozen]] if len(frozen) else frozen
  rounds.append({"round":rid,"proposal_status":diag.get("status"),"global_bad_before":base_bad,
                 "global_bad_after":bad,"best_global_bad_after":best_bad,
                 "solver_status":diag.get("solver_status"),"solver_message":diag.get("solver_message"),
                 "free_nodes_before":int(free.sum()),"newly_frozen_node_count":int(len(new)),
                 "newly_frozen_node_rows":new.astype(int).tolist(),"freeze_witnesses":frecords,
                 "proposal_diagnostics":diag})
  if not len(new):break
  free[new]=False;all_frozen.update(int(x) for x in new)
 status="PASS_F32_ACTIVE_SET_QUALITY_COUNT_REDUCTION" if best_bad<base_bad else "ACTIVE_SET_NO_COUNT_REDUCTION"
 return best,{"schema":"numi.human.coupled-quality-active-set.v6","status":status,
       "physical_admission":False,"native_steps":0,"active_set_policy":"first-witness-per-blocked-component",
       "max_active_rounds":int(max_active_rounds),"round_count":len(rounds),
       "baseline_bad_cells":base_bad,"final_bad_cells":best_bad,
       "strict_quality_count_progress":bool(best_bad<base_bad),
       "frozen_global_point_rows":sorted(all_frozen),"trust_cap_m":float(trust_cap_m),
       "trust_reference":"same immutable Float32 baseline each round","rounds":rounds,
       "quality_owner_sha256":QUALITY_OWNER_SHA256,"analytic_lp_reference_sha256":LP_REFERENCE_SHA256}
