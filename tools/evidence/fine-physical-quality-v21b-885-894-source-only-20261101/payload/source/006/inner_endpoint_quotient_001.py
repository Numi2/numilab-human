"""Bounded topology-only endpoint quotient for a current I/D source edge.

Coordinates are never changed or pruned. A source row is redirected to an
existing endpoint; all affected volume tets are remapped through the same
F32-coordinate quotient. This module is a proposal primitive. Exact source
intersection audits, mass mapping, and native admission remain external.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
H = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
M_PATH = H / 'fixed_boundary_volume_flip_198_root.py'
M_SHA = '650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0'
QMIN = 1.0e-5
VMIN = 1.0e-18
MAX_EDGE_M = 50.0e-6
LANES = ('layer','body','bone','face_id','original_id','identity0','identity1')
ANCESTRY_LANES = ('source_face_rows','source_original_ids','parent_face_rows')


def sha(path: Path) -> str:
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def _load_owner():
    if not M_PATH.is_file():
        raise FileNotFoundError(M_PATH)
    actual = sha(M_PATH)
    if M_SHA != 'DERIVE_AT_FREEZE' and actual != M_SHA:
        raise ValueError(f'198 quality owner SHA changed: {actual}')
    spec = importlib.util.spec_from_file_location('endpoint_quotient_quality_198', M_PATH)
    if not spec or not spec.loader:
        raise RuntimeError(f'cannot import {M_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_owner()


def f32key(x: Any) -> bytes:
    a=np.ascontiguousarray(np.asarray(x,dtype=np.float32).reshape(3))
    if not np.isfinite(a).all():
        raise ValueError('nonfinite endpoint coordinate')
    return a.view(np.uint32).tobytes()


def face_sign(f) -> int:
    a,b,c=map(int,f)
    return -1 if (int(a>b)+int(a>c)+int(b>c))%2 else 1


def face_chain(faces):
    out=Counter()
    for f in np.asarray(faces,np.int64):
        out[tuple(sorted(map(int,f)))]+=face_sign(f)
    return {k:v for k,v in out.items() if v}


def boundary_faces(tets):
    """Return oriented boundary triangles; fail on nonmanifold or bad winding."""
    t=np.asarray(tets,np.int64)
    if t.ndim!=2 or t.shape[1]!=4:
        raise ValueError('malformed tetrahedra')
    f=t[:,M.PAT].reshape(-1,3)
    sf=np.sort(f,axis=1)
    order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]))
    ss=sf[order]
    starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))]
    counts=np.diff(np.r_[starts,len(ss)])
    if np.any(counts>2):
        raise ValueError('nonmanifold tetrahedral face')
    inv=((f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8))
    signs=1-2*(inv%2)
    sums=np.add.reduceat(signs[order],starts)
    if np.any(sums[counts==2]!=0):
        raise ValueError('inconsistent internal tetrahedral face winding')
    rows=order[starts[counts==1]]
    return f[rows]


def _edge_faces(faces):
    ef=defaultdict(list)
    for fi,f in enumerate(np.asarray(faces,np.int64)):
        a,b,c=map(int,f)
        for x,y in ((a,b),(b,c),(c,a)):
            ef[tuple(sorted((x,y)))].append(int(fi))
    return ef


def _surface_boundary_chain(faces):
    counts=Counter();signed=Counter()
    for a,b,c in np.asarray(faces,np.int64):
        for x,y in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            edge=tuple(sorted((x,y)));counts[edge]+=1;signed[edge]+=1 if x<y else -1
    if any(n>2 or (n==2 and signed[e]!=0) for e,n in counts.items()):
        raise ValueError('source surface is not a consistently oriented 2-manifold')
    return {e:int(signed[e]) for e,n in counts.items() if n==1}


def _vertex_neighbors(faces):
    out=defaultdict(set)
    for a,b,c in np.asarray(faces,np.int64):
        a,b,c=int(a),int(b),int(c)
        out[a].update((b,c));out[b].update((a,c));out[c].update((a,b))
    return out


def _face_qbound(points, faces):
    p=np.asarray(points,np.float64)[np.asarray(faces,np.int64)]
    area2=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    e=np.stack((np.linalg.norm(p[:,1]-p[:,0],axis=1),
                np.linalg.norm(p[:,2]-p[:,1],axis=1),
                np.linalg.norm(p[:,0]-p[:,2],axis=1)),axis=1)
    l=np.max(e,axis=1)
    return area2/np.maximum(l*l,np.finfo(float).tiny)


def collapse_surface(points_f32, faces, edge, *, drop, keep,
                     fixed_endpoint_rows=(), max_edge_m=MAX_EDGE_M,
                     qmin=QMIN):
    """Apply one link-valid two-face edge contraction without changing points.

    `fixed_endpoint_rows` includes immutable rim/route/O rows. Inner back rows
    may be contracted only within `max_edge_m`; their prior row remains in the
    point array but is recorded as retired by the quotient.
    """
    p=np.asarray(points_f32,np.float32)
    f=np.asarray(faces,np.int64)
    u,v=map(int,edge);drop=int(drop);keep=int(keep)
    if p.ndim!=2 or p.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3:
        raise ValueError('malformed source surface')
    if {u,v}!={drop,keep} or drop==keep or min(drop,keep)<0 or max(drop,keep)>=len(p):
        raise ValueError('edge/endpoints disagree or outside point domain')
    length=float(np.linalg.norm(p[drop].astype(np.float64)-p[keep].astype(np.float64)))
    if not np.isfinite(length) or length<=0 or length>max_edge_m:
        raise ValueError('endpoint displacement exceeds the declared inner-surface allowance')
    if drop in set(map(int,fixed_endpoint_rows)):
        raise ValueError('retired endpoint is an immutable rim/route/O anchor')
    ef=_edge_faces(f)
    inc=ef.get(tuple(sorted((u,v))),[])
    if len(inc)!=2:
        raise ValueError('only a two-face interior edge is eligible')
    # Exact simplicial link condition prevents duplicate faces after quotient.
    neighbors=_vertex_neighbors(f)
    common=(neighbors[drop]&neighbors[keep])-{drop,keep}
    opposite={int(x) for r in inc for x in f[int(r)] if int(x) not in (drop,keep)}
    if common!=opposite:
        raise ValueError('surface edge contraction fails the link condition')
    mapped=f.copy();mapped[mapped==drop]=keep
    deg=((mapped[:,0]==mapped[:,1])|(mapped[:,1]==mapped[:,2])|(mapped[:,2]==mapped[:,0]))
    # The source row order is retained for every nondegenerate parent face.
    parent=np.flatnonzero(~deg).astype(np.int64)
    out=mapped[~deg]
    keys=np.sort(out,axis=1)
    if len(np.unique(keys,axis=0))!=len(out):
        raise ValueError('edge quotient creates duplicate source faces')
    # Positive orientation and nonzero face measure are mandatory on changed rows.
    changed=(f[parent]==drop).any(axis=1)
    if np.any(changed):
        old=f[parent[changed]];new=out[changed]
        oldn=np.cross(p[old[:,1]].astype(np.float64)-p[old[:,0]],p[old[:,2]].astype(np.float64)-p[old[:,0]])
        newn=np.cross(p[new[:,1]].astype(np.float64)-p[new[:,0]],p[new[:,2]].astype(np.float64)-p[new[:,0]])
        if np.any(np.einsum('ij,ij->i',oldn,newn)<=0):
            raise ValueError('quotient changes source-face winding or collapses area')
    if np.any(_face_qbound(p,out)<=0):
        raise ValueError('source quotient produced zero-area face')
    # The oriented 1D boundary of the full source surface must transform only
    # by the same endpoint map; no extra open edge or rim is allowed.
    old_boundary=_surface_boundary_chain(f)
    expected=Counter()
    for (a,b),orientation in old_boundary.items():
        x=keep if a==drop else a;y=keep if b==drop else b
        if x!=y:
            edge=tuple(sorted((x,y)));expected[edge]+=orientation*(1 if x<y else -1)
    actual=_surface_boundary_chain(out)
    if dict(expected)!=actual:
        raise ValueError('source surface boundary/rim changed outside the endpoint quotient')
    q0=_face_qbound(p,f);q1=_face_qbound(p,out)
    before_bad=int(np.count_nonzero(q0<qmin));after_bad=int(np.count_nonzero(q1<qmin))
    if after_bad>before_bad:
        raise ValueError('source facet lower-bound failure count regressed')
    quotient=np.arange(len(p),dtype=np.int64);quotient[drop]=keep
    affected=np.flatnonzero(np.any((f==drop)|(f==keep),axis=1)).astype(np.int64)
    changed_parent=np.flatnonzero(changed).astype(np.int64)
    return {'points_world_m_f32':p,'faces':out,'face_output_to_parent_rows':parent,
            'changed_output_rows':np.flatnonzero(changed).astype(np.int64),
            'retired_parent_face_rows':np.flatnonzero(deg).astype(np.int64),
            'affected_parent_face_rows':affected,'complete_operation_parent_face_rows':affected,
            'endpoint_quotient_output_to_current_rows':quotient,
            'drop_row':drop,'keep_row':keep,'edge_rows':(u,v),'edge_length_m':length,
            'facet_bad_before':before_bad,'facet_bad_after':after_bad,
            'minimum_facet_q_before':float(q0.min()),'minimum_facet_q_after':float(q1.min()),
            'strict_face_orientation':True,'source_boundary_quotient_exact':True}


def _exact_positive(owner, points_f32, tets):
    t=np.asarray(tets,np.int64)
    used=np.unique(t.reshape(-1))
    look={int(x):i for i,x in enumerate(used)}
    ip=owner.integer_points(np.asarray(points_f32,np.float32)[used])
    dets=[owner.exact_det(ip,[look[int(x)] for x in cell]) for cell in t]
    return dets


def _log_deficit(q, qmin=QMIN):
    q=np.asarray(q,np.float64)
    bad=q<qmin
    return float(np.sum(np.log(qmin/np.maximum(q[bad],np.finfo(np.float64).tiny))))


def collapse_volume(owner, points_f32, tets, materials, desired_boundary,
                    source_points_f32, edge, *, drop, keep, max_edge_m=MAX_EDGE_M,
                    qmin=QMIN, vmin=VMIN):
    """Apply quotient to one volume and independently certify local/global gates."""
    p=np.asarray(points_f32,np.float32);t=np.asarray(tets,np.int64);mat=np.asarray(materials)
    target=np.asarray(desired_boundary,np.int64);sp=np.asarray(source_points_f32,np.float32)
    if p.ndim!=2 or p.shape[1]!=3 or t.shape[1]!=4 or mat.shape!=(len(t),):
        raise ValueError('malformed volume arrays')
    if len(np.unique(p.view(np.dtype((np.void,p.dtype.itemsize*p.shape[1]))).ravel()))!=len(p):
        raise ValueError('volume point F32 coordinates are ambiguous')
    if face_chain(boundary_faces(t))!=face_chain(target):
        raise ValueError('input volume is not already the exact oriented target boundary')
    lookup={f32key(x):i for i,x in enumerate(p)}
    su,sv=map(int,edge);du,kv=int(drop),int(keep)
    if f32key(sp[su]) not in lookup or f32key(sp[sv]) not in lookup:
        raise ValueError('source edge endpoint is not represented in the volume')
    dropv,keepv=lookup[f32key(sp[du])],lookup[f32key(sp[kv])]
    if dropv==keepv:raise ValueError('source endpoints alias one volume row before quotient')
    length=float(np.linalg.norm(p[dropv].astype(np.float64)-p[keepv].astype(np.float64)))
    if length>max_edge_m:raise ValueError('volume endpoint displacement exceeds cap')
    nodeq=np.arange(len(p),dtype=np.int64);nodeq[dropv]=keepv
    mapped=nodeq[t]
    sorted_mapped=np.sort(mapped,axis=1)
    deg=np.any(sorted_mapped[:,1:]==sorted_mapped[:,:-1],axis=1)
    keepcells=np.flatnonzero(~deg).astype(np.int64)
    out=mapped[~deg]
    outmat=mat[keepcells]
    affected=np.flatnonzero(np.any(t==dropv,axis=1)).astype(np.int64)
    changed_out_positions=np.flatnonzero(np.any(t[keepcells]==dropv,axis=1)).astype(np.int64)
    changed_input=keepcells[changed_out_positions]
    if not len(changed_input):raise ValueError('endpoint is unused by volume cells')
    # Ensure the quotient did not merge distinct tetrahedra.
    sorted_tets=np.sort(out,axis=1)
    if len(np.unique(sorted_tets,axis=0))!=len(out):
        raise ValueError('quotient creates duplicate volume cells')
    det0,q0=owner.measurements(p,t);det1,q1=owner.measurements(p,out)
    if np.any(~np.isfinite(det1)) or np.any(det1/6<=vmin) or np.any(q1<qmin):
        # Quality failures already present are allowed; this unconditional line
        # would overreject. The exact regression checks below decide quality.
        positive_bad=np.flatnonzero((~np.isfinite(det1))|(det1/6<=vmin))
        if len(positive_bad):raise ValueError('quotient produces nonpositive or sub-volume-gate tetrahedra')
    exact_new=_exact_positive(owner,p,out[changed_out_positions])
    if any(x<=0 for x in exact_new):raise ValueError('quotient has exact F32 nonpositive changed tetrahedron')
    oldgood=np.asarray(owner.passes(det0,q0),bool)
    newgood=np.asarray(owner.passes(det1,q1),bool)
    # Cell row mapping is one-to-one for survivors; compare each previous good
    # cell to its mapped counterpart and preserve material bad counts.
    if np.any(oldgood[keepcells]&~newgood):
        raise ValueError('quotient regresses a previously native-good cell')
    before_counts={int(m):int(np.count_nonzero((~oldgood)&(mat==m))) for m in np.unique(mat)}
    after_counts={int(m):int(np.count_nonzero((~newgood)&(outmat==m))) for m in np.unique(mat)}
    if any(after_counts[m]>before_counts[m] for m in before_counts):
        raise ValueError('quotient increases native bad count in a material')
    before_total=int(np.count_nonzero(~oldgood));after_total=int(np.count_nonzero(~newgood))
    before_score=_log_deficit(q0,qmin);after_score=_log_deficit(q1,qmin)
    if not (after_total<before_total or (after_total==before_total and after_score<before_score-1e-8)):
        raise ValueError('quotient has no strict native quality lexicographic progress')
    # Exact oriented boundary must equal the quotient of the parent target.
    mapped_target=nodeq[target]
    sorted_target=np.sort(mapped_target,axis=1)
    nondeg_target=~np.any(sorted_target[:,1:]==sorted_target[:,:-1],axis=1)
    target_out=mapped_target[nondeg_target]
    actual_boundary=boundary_faces(out)
    if face_chain(actual_boundary)!=face_chain(target_out):
        raise ValueError('whole output volume boundary differs from quotient target')
    # Verify input boundary was already the declared exact target.
    # Exact signed volume delta on affected cells only; topology-only quotient
    # need not conserve it automatically, so keep it explicit for later mapping.
    old_local=t[affected]
    new_local=mapped[affected]
    sorted_local=np.sort(new_local,axis=1)
    new_local=new_local[~np.any(sorted_local[:,1:]==sorted_local[:,:-1],axis=1)]
    old_exact=_exact_positive(owner,p,old_local)
    new_exact=_exact_positive(owner,p,new_local)
    delta=sum(new_exact)-sum(old_exact)
    return {'points_world_m_f32':p,'tetrahedra':out,'material_indices':outmat,
            'boundary_faces':actual_boundary,'desired_boundary_faces':target_out,
            'input_cell_to_output_cell_rows':_inverse_map(len(t),keepcells),
            'output_cell_to_input_cell_rows':keepcells,'retained_input_cell_rows':keepcells,
            'input_node_to_output_node_rows':nodeq,'output_node_to_input_node_rows':np.arange(len(p),dtype=np.int64),
            'drop_volume_row':int(dropv),'keep_volume_row':int(keepv),
            'changed_output_cell_rows':changed_out_positions,'changed_input_cell_rows':changed_input,
            'affected_input_cell_rows':affected,'drop_volume_row':int(dropv),'keep_volume_row':int(keepv),
            'edge_length_m':length,'native_bad_before':before_total,'native_bad_after':after_total,
            'native_deficit_before':before_score,'native_deficit_after':after_score,
            'native_bad_by_material_before':before_counts,'native_bad_by_material_after':after_counts,
            'minimum_normalized_det_before':float(q0.min()),'minimum_normalized_det_after':float(q1.min()),
            'minimum_volume_m3_after':float(np.min(det1)/6),
            'exact_positive_changed_cells':len(exact_new),'exact_oriented_boundary_quotient':True,
            'exact_signed_volume_delta_det6_integer':str(delta),
            'exact_signed_volume_delta_m3':float(delta)*2.0**(-447)/6.0,
            'point_array_unchanged':True}


def _inverse_map(n, rows):
    out=np.full(int(n),-1,np.int64)
    out[np.asarray(rows,np.int64)]=np.arange(len(rows),dtype=np.int64)
    return out


def propose_endpoint_quotient(role, surface, volumes, edge, *, drop, keep,
                              immutable_source_rows=(), max_edge_m=MAX_EDGE_M):
    """Validate a D->Fat or paired I->Skin/Fat quotient atomically."""
    role=str(role).upper()
    target_volumes={'D':('Fat',),'I':('Skin','Fat')}.get(role)
    if target_volumes is None:raise ValueError('only inner I or D endpoint quotient is allowed')
    sr=collapse_surface(surface['points_world_m_f32'],surface['faces'],edge,
                        drop=drop,keep=keep,fixed_endpoint_rows=immutable_source_rows,
                        max_edge_m=max_edge_m)
    results={}
    for vr in target_volumes:
        v=volumes[vr]
        results[vr]=collapse_volume(M,v['points_world_m'],v['tetrahedra'],v['material_indices'],
            v.get('desired_boundary_faces',v.get('boundary_faces')),
            surface['points_world_m_f32'],edge,drop=drop,keep=keep,max_edge_m=max_edge_m)
    before=sum(x['native_bad_before'] for x in results.values())
    after=sum(x['native_bad_after'] for x in results.values())
    bscore=sum(x['native_deficit_before'] for x in results.values())
    ascore=sum(x['native_deficit_after'] for x in results.values())
    if not (after<before or (after==before and ascore<bscore-1e-8)):
        raise ValueError('paired quotient has no strict combined native quality progress')
    return {'role':role,'surface':sr,'volumes':results,'combined_native_bad_before':before,
            'combined_native_bad_after':after,'combined_log_deficit_before':bscore,
            'combined_log_deficit_after':ascore,'paired':role=='I',
            'source_and_volume_points_unchanged':True,'physical_admission':False,'native_steps':0,
            'mass_mapping':'PENDING','exact_source_intersection_audit':'PENDING_EXTERNAL_775_805'}


def _csr_slice(offsets, values, row):
    o=np.asarray(offsets,np.int64);v=np.asarray(values,np.int64)
    return v[int(o[row]):int(o[row+1])]


def endpoint_surface_arrays(parent: dict, result: dict, *, parent_path: str, parent_sha: str, operation_id: int):
    """Construct fully typed current surface arrays with immediate ancestry."""
    out={k:np.array(v,copy=True) for k,v in parent.items()}
    pfaces=np.asarray(parent['faces'],np.int64);faces=np.asarray(result['faces'],np.int64)
    parent_rows=np.asarray(result['face_output_to_parent_rows'],np.int64)
    changed=np.asarray(result['changed_output_rows'],np.int64);changed_set=set(map(int,changed))
    out['faces']=faces.copy()
    # Filter all face-row-aligned fields through the immediate parent row map.
    # This includes primary typed IDs and historical operation annotations.
    for key,value in parent.items():
        a=np.asarray(value)
        if key not in ('faces','quality_star_patch_union_offsets','face_ancestry_offsets') and a.ndim>=1 and a.shape[0]==len(pfaces):
            out[key]=np.asarray(a[parent_rows]).copy()
    # All point arrays are preserved verbatim; only the explicit quotient maps
    # redirect use of the retired row.
    for key in ('points_world_m','points_world_m_f32'):
        if key not in parent:raise KeyError(key)
        out[key]=np.array(parent[key],copy=True)
    quotient=np.asarray(result['endpoint_quotient_output_to_current_rows'],np.int64)
    out['endpoint_quotient_output_to_current_parent_rows']=quotient
    for key in ('quality_star_point_output_to_input_row',):
        if key in parent:out[key]=np.asarray(parent[key],np.int64).copy()
    if 'source_input_vertex_quotient' in parent:
        q=np.asarray(parent['source_input_vertex_quotient'],np.int64).copy()
        if len(q)!=len(quotient):raise ValueError('source input point quotient shape mismatch')
        q[int(result['drop_row'])]=q[int(result['keep_row'])]
        out['source_input_vertex_quotient']=q
    # Immediate face ancestry is one-to-one for each surviving parent triangle;
    # rows changed by the quotient are explicitly marked as operation rows.
    for key in ('quality_star_face_output_to_input_row','source_input_face_rows'):
        if key in parent:out[key]=np.asarray(parent[key],np.int64)[parent_rows].copy()
    if 'source_collapse_operation_rows' in parent:
        op=np.asarray(parent['source_collapse_operation_rows'],np.int64)[parent_rows].copy()
        op[changed]=int(operation_id)
        out['source_collapse_operation_rows']=op
    # Changed faces inherit the complete edge-star union; unchanged rows retain
    # the exact prior CSR slices. Typed lane identities are never fabricated.
    uoff=np.asarray(parent['quality_star_patch_union_offsets'],np.int64)
    urows={k:np.asarray(parent['quality_star_patch_union_face_source_'+k],np.int64) for k in LANES}
    aoff=np.asarray(parent['face_ancestry_offsets'],np.int64)
    arows={k:np.asarray(parent['face_ancestry_'+k],np.int64) for k in ANCESTRY_LANES}
    star=np.asarray(result['complete_operation_parent_face_rows'],np.int64)
    star_union=set()
    for r in star:
        vals=[_csr_slice(uoff,urows[k],int(r)) for k in LANES]
        star_union.update(zip(*(x.tolist() for x in vals)))
    star_union_rows=sorted(star_union)
    star_anc=set()
    for r in star:
        vals=[_csr_slice(aoff,arows[k],int(r)) for k in ANCESTRY_LANES]
        star_anc.update(zip(*(x.tolist() for x in vals)))
    star_anc_rows=sorted(star_anc)
    new_u_offsets=[0];new_u={k:[] for k in LANES}
    new_a_offsets=[0];new_a={k:[] for k in ANCESTRY_LANES}
    for i,pr in enumerate(parent_rows):
        if i in changed_set:
            us=star_union_rows;ans=star_anc_rows
        else:
            us=list(zip(*[_csr_slice(uoff,urows[k],int(pr)).tolist() for k in LANES]))
            ans=list(zip(*[_csr_slice(aoff,arows[k],int(pr)).tolist() for k in ANCESTRY_LANES]))
        for k,j in zip(LANES,range(len(LANES))):new_u[k].extend(int(t[j]) for t in us)
        for k,j in zip(ANCESTRY_LANES,range(len(ANCESTRY_LANES))):new_a[k].extend(int(t[j]) for t in ans)
        new_u_offsets.append(len(new_u['layer']));new_a_offsets.append(len(new_a['source_face_rows']))
    out['quality_star_patch_union_offsets']=np.asarray(new_u_offsets,np.int64)
    for k in LANES:out['quality_star_patch_union_face_source_'+k]=np.asarray(new_u[k],np.int64)
    out['face_ancestry_offsets']=np.asarray(new_a_offsets,np.int64)
    for k in ANCESTRY_LANES:out['face_ancestry_'+k]=np.asarray(new_a[k],np.int64)
    # Parent path/hash are scalar strings in some generations and singleton
    # vectors in others; preserve their rank convention.
    for key,value in (('typed_parent_candidate_path',parent_path),('typed_parent_candidate_sha256',parent_sha),
                      ('endpoint_quotient_parent_path',parent_path),('endpoint_quotient_parent_sha256',parent_sha)):
        if key in parent and np.asarray(parent[key]).ndim==0:out[key]=np.asarray(value,dtype=f'<U{max(1,len(value))}')
        else:out[key]=np.asarray([value],dtype=f'<U{max(1,len(value))}')
    out['endpoint_quotient_operation_id']=np.asarray([int(operation_id)],np.int64)
    out['endpoint_quotient_drop_source_row']=np.asarray([int(result['drop_row'])],np.int64)
    out['endpoint_quotient_keep_source_row']=np.asarray([int(result['keep_row'])],np.int64)
    out['endpoint_quotient_retired_parent_face_rows']=np.asarray(result['retired_parent_face_rows'],np.int64)
    out['endpoint_quotient_changed_output_face_rows']=changed.copy()
    return out


def endpoint_volume_arrays(parent: dict, result: dict, *, parent_path: str, parent_sha: str):
    """Build current volume arrays; point domains/bytes are never compacted."""
    out={k:np.array(v,copy=True) for k,v in parent.items()}
    ninput=len(np.asarray(parent['tetrahedra']))
    keepcells=np.asarray(result['output_cell_to_input_cell_rows'],np.int64)
    special={'tetrahedra','boundary_faces','desired_boundary_faces','material_indices',
             'retained_input_cell_rows','output_cell_to_input_cell','output_cell_to_input_cell_rows',
             'input_cell_to_output_cell','input_cell_to_output_cell_rows',
             'replacement_tetrahedra','replacement_parent_offsets','replacement_parent_cell_rows'}
    # Filter any row-aligned cell metadata by the same retained-cell map.
    for key,value in parent.items():
        a=np.asarray(value)
        if key not in special and a.ndim>=1 and a.shape[0]==ninput:
            out[key]=np.asarray(a[keepcells]).copy()
    for k in ('points_world_m','points_world_m_f32'):
        if k in parent:out[k]=np.array(parent[k],copy=True)
    out['tetrahedra']=np.asarray(result['tetrahedra'],np.int64).copy()
    out['material_indices']=np.asarray(result['material_indices']).copy()
    out['boundary_faces']=np.asarray(result['boundary_faces'],np.int64).copy()
    out['desired_boundary_faces']=np.asarray(result['desired_boundary_faces'],np.int64).copy()
    for key in ('retained_input_cell_rows','output_cell_to_input_cell','output_cell_to_input_cell_rows'):
        if key in parent:out[key]=np.asarray(result['output_cell_to_input_cell_rows'],np.int64).copy()
    for key in ('input_cell_to_output_cell','input_cell_to_output_cell_rows'):
        if key in parent:out[key]=np.asarray(result['input_cell_to_output_cell_rows'],np.int64).copy()
    out['retained_input_cell_rows']=np.asarray(result['retained_input_cell_rows'],np.int64).copy()
    out['output_cell_to_input_cell']=np.asarray(result['output_cell_to_input_cell_rows'],np.int64).copy()
    out['output_cell_to_input_cell_rows']=np.asarray(result['output_cell_to_input_cell_rows'],np.int64).copy()
    out['input_cell_to_output_cell']=np.asarray(result['input_cell_to_output_cell_rows'],np.int64).copy()
    out['input_cell_to_output_cell_rows']=np.asarray(result['input_cell_to_output_cell_rows'],np.int64).copy()
    for key in ('input_node_to_output_node','input_node_to_output_node_rows','source_interior_vertex_quotient'):
        if key in parent and np.asarray(parent[key]).shape==(len(result['input_node_to_output_node_rows']),):
            old=np.asarray(parent[key],np.int64).copy()
            # Preserve historical aliases and compose only this operation.
            old[int(result['drop_volume_row'])]=old[int(result['keep_volume_row'])]
            out[key]=old
    out['input_node_to_output_node']=np.asarray(result['input_node_to_output_node_rows'],np.int64).copy()
    out['input_node_to_output_node_rows']=np.asarray(result['input_node_to_output_node_rows'],np.int64).copy()
    out['output_node_to_input_node']=np.arange(len(parent['points_world_m']),dtype=np.int64)
    out['output_node_to_input_node_rows']=np.arange(len(parent['points_world_m']),dtype=np.int64)
    for key in ('output_point_to_input_point_row','output_point_to_input_point_rows'):
        if key in parent:out[key]=np.arange(len(parent['points_world_m']),dtype=np.int64)
    for key in ('input_point_to_output_point_row','input_point_to_output_point_rows'):
        if key in parent:out[key]=np.asarray(result['input_node_to_output_node_rows'],np.int64).copy()
    out['input_point_to_output_point_row']=np.asarray(result['input_node_to_output_node_rows'],np.int64).copy()
    out['output_point_to_input_point_row']=np.arange(len(parent['points_world_m']),dtype=np.int64)
    changed=np.asarray(result['changed_output_cell_rows'],np.int64)
    out['endpoint_quotient_changed_output_cell_rows']=changed.copy()
    out['endpoint_quotient_changed_input_cell_rows']=np.asarray(result['changed_input_cell_rows'],np.int64).copy()
    out['endpoint_quotient_affected_input_cell_rows']=np.asarray(result['affected_input_cell_rows'],np.int64).copy()
    out['endpoint_quotient_input_node_map']=np.asarray(result['input_node_to_output_node_rows'],np.int64).copy()
    # Every changed surviving tet inherits the complete edge-star cell union;
    # deleted degenerate tets remain listed in the operation receipt.
    affected=np.asarray(result['affected_input_cell_rows'],np.int64)
    offsets=np.zeros(len(changed)+1,np.int64)
    if len(changed):offsets[1:]=np.arange(1,len(changed)+1,dtype=np.int64)*len(affected)
    out['replacement_tetrahedra']=np.asarray(result['tetrahedra'],np.int64)[changed].copy()
    out['replacement_parent_offsets']=offsets
    out['replacement_parent_cell_rows']=np.tile(affected,len(changed)).astype(np.int64)
    # Boundary row metadata is synchronized with the mapped target faces. No
    # endpoint is allowed to disappear from a protected wall/rim/route row.
    oldtarget=np.asarray(parent['desired_boundary_faces'],np.int64)
    q=np.asarray(result['input_node_to_output_node_rows'],np.int64)
    mapped=q[oldtarget]
    valid=((mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0]))
    for key,value in parent.items():
        a=np.asarray(value)
        if key not in ('desired_boundary_faces','boundary_faces') and a.ndim>=1 and a.shape[0]==len(oldtarget):
            out[key]=np.asarray(a[valid]).copy()
    out['desired_boundary_faces']=mapped[valid].copy()
    for key in ('mapped_current_boundary_faces','proposed_current_desired_boundary_faces'):
        if key in parent and np.asarray(parent[key]).shape==oldtarget.shape:
            out[key]=mapped[valid].copy()
    if 'desired_boundary_degenerate_source_rows' in parent:
        old=np.asarray(parent['desired_boundary_degenerate_source_rows'],np.int64).reshape(-1)
        removed=np.asarray([],np.int64)
        if 'desired_boundary_source_rows' in parent:
            removed=np.asarray(parent['desired_boundary_source_rows'],np.int64)[~valid]
        out['desired_boundary_degenerate_source_rows']=np.unique(np.r_[old,removed[removed>=0]]).astype(np.int64)
    for k,value in (('current_geometry_parent_path',parent_path),('current_geometry_parent_sha256',parent_sha),
                    ('endpoint_quotient_parent_path',parent_path),('endpoint_quotient_parent_sha256',parent_sha)):
        if k in parent and np.asarray(parent[k]).ndim==0:out[k]=np.asarray(value,dtype=f'<U{max(1,len(value))}')
        else:out[k]=np.asarray([value],dtype=f'<U{max(1,len(value))}')
    out['canonical_parent_union_physical_status']=np.asarray(['ENDPOINT_QUOTIENT_FULL_AFFECTED_CELL_STAR_MASS_MAPPING_PENDING'],dtype='U72')
    return out


if __name__=='__main__':
    raise SystemExit('This frozen core is imported by a state-bound owner; no standalone actual-mesh CLI is provided in this version.')
