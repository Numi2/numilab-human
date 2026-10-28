from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
WORK=ROOT/'connected-obstacle-207/physical-transfer-209'
BASE=WORK/'geometry-worker/D-connectivity-218'
CAND=BASE/'limiting-O-edge-audit-001/quality-failure-inventory-001/contraction-batch-001/shape-repair-batch-005/contraction-001/candidate.npz'
EXACT=BASE/'limiting-O-edge-audit-001/quality-failure-inventory-001/contraction-batch-001/shape-repair-batch-005/contraction-001/final-audit-006/report.json'
OUT=BASE/'limiting-O-edge-audit-001/quality-failure-inventory-001/contraction-batch-001/shape-repair-batch-005/contraction-001/final-audit-007'
OUT.mkdir(parents=True,exist_ok=True)
QMOD=BASE/'bone-distance-query-001/bone_distance_query.py'
sys.path.insert(0,str(QMOD.parent))
from bone_distance_query import BoneDistanceQuery

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def exact_vec(v): return tuple(Fraction(float(x)) for x in v)

def exact_ray_inside(p_float, tri_float):
    p=exact_vec(p_float)
    dirs=[(Fraction(1),Fraction(1,3),Fraction(1,7)),
          (Fraction(1),Fraction(2,9),Fraction(3,11)),
          (Fraction(1),Fraction(3,13),Fraction(5,17))]
    # Fast positive-ray AABB broadphase is conservative: any hit must have
    # triangle max coordinate >= point in each axis.
    for di,d in enumerate(dirs):
        hits=0; retry=False
        for tri0 in tri_float:
            if np.any(np.max(tri0,axis=0) < np.asarray(p_float)): continue
            a,b,c=(exact_vec(x) for x in tri0)
            e1=sub(b,a); e2=sub(c,a); h=cross(d,e2); det=dot(e1,h)
            if det==0: continue
            s=sub(p,a); un=dot(s,h); q=cross(s,e1); vn=dot(d,q); tn=dot(e2,q)
            if det<0:
                det=-det; un=-un; vn=-vn; tn=-tn
            if un<0 or vn<0 or un+vn>det: continue
            if tn<=0: continue
            # Avoid edge/vertex ray hits; retry with an independent exact ray.
            if un==0 or vn==0 or un+vn==det:
                retry=True; break
            hits+=1
        if not retry: return (hits%2)==1, di, hits, 'exact-ray-parity'
    return None, None, None, 'ray-edge-degenerate'

start=time.monotonic()
z=np.load(CAND)
points=np.asarray(z['points_world_m'],dtype=np.float32)
faces=np.asarray(z['faces'],dtype=np.int64)
report=json.load(open(EXACT))
changed=np.asarray(report['changed_final_face_rows'],dtype=np.int64)
tri=points[faces[changed]].astype(np.float64)
# For every changed triangle, test its centroid and all three edge midpoints.
probes=np.concatenate([tri.mean(axis=1,keepdims=True), (tri+np.roll(tri,-1,axis=1))*0.5],axis=1).astype(np.float32).reshape(-1,3)
probe_face=np.repeat(changed,4)
probe_kind=np.tile(np.asarray(['centroid','edge01_midpoint','edge12_midpoint','edge20_midpoint']),len(changed))
query=BoneDistanceQuery()
load_seconds=time.monotonic()-start
qt=time.monotonic()
violations=query.query(probes,0.0)
query_seconds=time.monotonic()-qt
banks={(b['body'],b['stable_bone_id']):b for b in query._banks}
ambiguous=[]; strict_inside=[]; boundary=[]; resolved=[]
for r in violations:
    row=int(r['query_row']); world=probes[row]
    bank=banks[(int(r['body']),int(r['stable_bone_id']))]
    local=(world.astype(np.float64)-bank['position_world_m'])@bank['world_to_body'].T-bank['com_offset_m']
    d=float(r['signed_distance_m']); kind=r['violation_kind']
    item={'probe_row':row,'face_row':int(probe_face[row]),'probe_kind':str(probe_kind[row]),'body':int(r['body']),'stable_bone_id':int(r['stable_bone_id']),'signed_distance_m':d,'unsigned_distance_m':float(r['unsigned_distance_m']),'winding_number':float(r['winding_number']),'violation_kind':kind}
    if kind=='inside-bone' and d<0:
        strict_inside.append(item); continue
    if kind=='winding-sign-ambiguous' or abs(d)==0.0:
        ambiguous.append(item)
        tris=bank['vertices_local_m'].reshape(-1,3,3)
        inside,di,hits,method=exact_ray_inside(local,tris)
        item.update({'exact_point_classification':('inside' if inside is True else 'outside' if inside is False else 'unresolved'),'exact_ray_direction_index':di,'exact_ray_crossings':hits,'exact_method':method})
        resolved.append(item)
        if inside is True: strict_inside.append(item)
        elif inside is None: boundary.append(item)
        continue
    # A nonambiguous row with nonpositive signed distance is an interior witness.
    if d<=0: strict_inside.append(item)
    else: boundary.append(item)

# Verify source topology/provenance maps to the pinned base candidate.
base=Path(report['baseline_candidate_path'])
b=np.load(base)
vm=np.asarray(z['O033_to_candidate_vertex_row'],dtype=np.int64)
inv=np.asarray(z['candidate_vertex_to_O033_row'],dtype=np.int64)
contracted=np.asarray(z['contracted_O033_vertex_rows'],dtype=np.int64)
face_equal=bool(np.array_equal(z['faces'],z['source_O_faces']))
map_ok=bool(vm.shape==(len(points),) and np.all((vm>=0)&(vm<len(points))) and np.array_equal(vm[vm],vm) and np.array_equal(inv,np.arange(len(points))) and np.array_equal(contracted,np.flatnonzero(vm!=np.arange(len(points)))))
csr_off=np.asarray(z['face_ancestry_offsets'],dtype=np.int64)
ancestry_ok=bool(len(csr_off)==len(faces)+1 and csr_off[0]==0 and csr_off[-1]==len(z['face_ancestry_parent_face_rows']) and np.all(np.diff(csr_off)>0))
res={
 'schema':'numi.connectedfit-O-final-shape-repair-sampled-solid-containment.v1',
 'candidate_path':str(CAND),'candidate_sha256':sha(CAND),
 'surface_audit_report':str(EXACT),'surface_audit_sha256':sha(EXACT),
 'bone_query_module':str(QMOD),'bone_query_module_sha256':sha(QMOD),
 'bone_query_bank_count':query.bank_count,'bone_query_bank_hashes':query.bank_hashes,
 'changed_face_count':int(len(changed)),'samples_per_changed_face':4,'sample_points':int(len(probes)),
 'sample_definition':'Float32 centroid and three edge midpoints of every changed Float32 face.',
 'query_margin_m':0.0,'query_load_seconds':load_seconds,'query_seconds':query_seconds,
 'returned_nonpositive_or_ambiguous_sample_bank_rows':int(len(violations)),
 'strict_inside_sample_bank_rows':strict_inside,'strict_inside_count':len(strict_inside),
 'ambiguous_rows':ambiguous,'ambiguous_resolved_rows':resolved,'ambiguous_resolved_inside_count':sum(x.get('exact_point_classification')=='inside' for x in resolved),'ambiguous_unresolved_count':sum(x.get('exact_point_classification')=='unresolved' for x in resolved),
 'boundary_rows':boundary,'all_185_typed_banks_queried':query.bank_count==185,
 'face_source_O_faces_equal':face_equal,'source_vertex_map_composed_idempotent':map_ok,
 'face_ancestry_csr_valid_nonempty':ancestry_ok,
 'limitations':['This is a sampled solid-containment gate over changed-face centroids and edge midpoints, with pinned signed-distance/Fast-Winding queries and exact rational ray-parity resolution for ambiguous samples. It is not exhaustive triangle-in-solid containment or physical admission.','Exact surface pair crossings are separately reported by the pinned final-audit-006 report.']
}
(OUT/'report.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({k:v for k,v in res.items() if k not in ('strict_inside_sample_bank_rows','ambiguous_rows','ambiguous_resolved_rows','boundary_rows','bone_query_bank_hashes')},indent=2))
if strict_inside or res['ambiguous_unresolved_count'] or not map_ok or not ancestry_ok or not face_equal or query.bank_count!=185:
    raise SystemExit(2)
