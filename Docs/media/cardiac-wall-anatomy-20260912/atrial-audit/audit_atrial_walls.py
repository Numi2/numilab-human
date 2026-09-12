#!/usr/bin/env python3
"""Bounded offline source geometry audit; no source edits or physical stepping.

Exact predicates are the existing Human owner. The auxiliary integer AABB tree
only prunes provably disjoint boxes; it never changes intersection semantics.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.root / 'src'))
from numilab_human import cardiac_cavity_geometry as geometry
from numilab_human import cardiac_cavity_intersections as exact
from numilab_human.cardiac_partition_certificate import _moment_vector, _moments_json

PINS = {
    'right_atrium_wall': ('FJ2439', 'FMA9457', 'wall of right atrium', 'e7c21cd1659eced056eb56e28f4fa9019ace451ea6e0333097846177f807bb5a'),
    'left_atrium_wall': ('FJ2438', 'FMA9531', 'wall of left atrium', '187235f3c3612ef27abde924d01c71579c2946435e64319164f43e5ad008284d'),
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def tree(records):
    lo = tuple(min(r[1][k] for r in records) for k in range(3))
    hi = tuple(max(r[2][k] for r in records) for k in range(3))
    if len(records) <= 8:
        return lo, hi, records, None
    axis = max(range(3), key=lambda k: hi[k]-lo[k])
    ordered = sorted(records, key=lambda r: (r[1][axis]+r[2][axis], r[3]))
    mid = len(ordered)//2
    return lo, hi, tree(ordered[:mid]), tree(ordered[mid:])

def query(node, lower, upper):
    lo, hi, a, b = node
    if any(upper[k] < lo[k] or hi[k] < lower[k] for k in range(3)):
        return
    if b is None:
        for row in a:
            if not any(upper[k] < row[1][k] or row[2][k] < lower[k] for k in range(3)):
                yield row
    else:
        yield from query(a, lower, upper)
        yield from query(b, lower, upper)

def audit_pair(first, second, same_surface):
    started=time.monotonic()
    index=tree(second)
    candidates=allowed=0
    pairs=[]
    for tri, lo, hi, i, ids in first:
        for other, lower, upper, j, other_ids in query(index, lo, hi):
            if same_surface and j <= i:
                continue
            candidates+=1
            points=exact.triangle_intersection_points(tri,other)
            if not points:
                continue
            shared_ids=set(ids)&set(other_ids) if same_surface else set()
            common={tri[ids.index(k)] for k in shared_ids}
            if same_surface and len(shared_ids) in (1,2) and all(exact._allowed_shared_point(p,common) for p in points):
                allowed+=1
            else:
                pairs.append([i,j])
    return {'triangle_pairs':sorted(pairs),'count':len(pairs),'aabb_candidate_pairs':candidates,
            'allowed_shared_vertex_or_edge_pairs':allowed,'elapsed_seconds':time.monotonic()-started}

args.output.mkdir(parents=True,exist_ok=True)
started=time.monotonic()
cavities=geometry.extract_cavity_surfaces(sources=args.root/'Sources',source_lock=args.root/'sources.lock.json')
anatomy=geometry.load_anatomy(args.root/'Sources',args.root/'sources.lock.json')
selected={x['source_id']:x for x in cavities['chambers'] if x['source_id'] in ('right_atrium','left_atrium')}
with zipfile.ZipFile(args.root/'Sources'/geometry.ARCHIVE) as archive:
    for name,(member,concept,label,sha) in PINS.items():
        for table in anatomy['tables'].values():
            assert table[(concept,label)]=={member}
        path=f'partof_BP3D_4.0_obj_99/{member}.obj'
        assert sum(x.filename==path for x in archive.infolist())==1
        data=archive.read(path)
        assert digest(data)==sha
        parsed=geometry.parse_obj(data,path)
        selected[name]={'source_id':name,'source':{'member':path,'sha256':sha,'bytes':len(data),
            'comments':parsed['comments']},'concept_id':concept,'source_name':label,
            'raw_topology':geometry.analyze_topology(parsed['vertices_mm'],parsed['triangles']),
            'exact_coordinate_quotient':geometry.exact_coordinate_quotient(parsed)}

denominator=max(Fraction.from_float(x).denominator for s in selected.values()
    for p in s['exact_coordinate_quotient']['vertices_m'] for x in p)
records={}
surfaces={}
for name,s in selected.items():
    q=s['exact_coordinate_quotient']
    rational=[tuple(Fraction.from_float(x) for x in p) for p in q['vertices_m']]
    integer=[tuple(x.numerator*(denominator//x.denominator) for x in p) for p in rational]
    r=exact._records(integer,q['triangles']); records[name]=r
    # Independent direct broadphase crosscheck on a deterministic bounded set.
    sample=r[::max(1,len(r)//256)]
    direct=exact._audit_pair(sample,sample,same_surface=True)
    accelerated=audit_pair(sample,sample,True)
    assert all(direct[k]==accelerated[k] for k in direct)
    print('self audit',name,len(r),flush=True)
    self_report=audit_pair(r,r,True)
    moments=_moment_vector([tuple(rational[i] for i in f) for f in q['triangles']])
    topology=q['topology']
    surfaces[name]={'source':s['source'],'concept_id':s['concept_id'],'source_name':s['source_name'],
        'raw_vertex_count':s['raw_topology']['vertex_count'],
        'raw_boundary_edge_count':s['raw_topology']['boundary_edge_count'],
        'identified_vertex_count':q['identified_vertex_count'],
        'topology':{k:v for k,v in topology.items() if k not in ('face_components','boundary_edges')},
        'self_intersections':self_report,
        'embedded_closed_surface':topology['closed_oriented_manifold_candidate'] and topology['face_component_count']==1 and self_report['count']==0,
        'geometry_sha256':digest(geometry.canonical({'vertices_m':q['vertices_m'],'triangles':q['triangles']})),
        'exact_oriented_moments':_moments_json(moments),'oriented_volume_m3':float(moments[0]),
        'centroid_m':[float(x/moments[0]) for x in moments[1:4]],
        'broadphase_crosscheck':{'sample_face_count':len(sample),'direct_owner_equal':True}}
    (args.output/(name+'.json')).write_text(json.dumps(surfaces[name],sort_keys=True,indent=2)+'\n')
    print(name,'self pairs',self_report['count'],'seconds',self_report['elapsed_seconds'],flush=True)

pairs=[]
for atrium in ('right_atrium','left_atrium'):
    wall=atrium+'_wall'
    print('wall-cavity audit',atrium,flush=True)
    a,b=records[atrium],records[wall]
    result=audit_pair(a,b,False)
    qa=selected[atrium]['exact_coordinate_quotient']; qb=selected[wall]['exact_coordinate_quotient']
    va={tuple(v) for v in qa['vertices_m']}; vb={tuple(v) for v in qb['vertices_m']}
    fa={tuple(sorted(tuple(qa['vertices_m'][i]) for i in f)) for f in qa['triangles']}
    fb={tuple(sorted(tuple(qb['vertices_m'][i]) for i in f)) for f in qb['triangles']}
    result.update(cavity=atrium,wall=wall,exact_shared_vertices=len(va&vb),
                  exact_shared_triangles=len(fa&fb),identical_boundary=fa==fb)
    if all(surfaces[n]['embedded_closed_surface'] for n in (atrium,wall)):
        if result['count']==0:
            result['containment']={'cavity_in_wall':exact.point_location(a[0][0][0],b),
                                   'wall_in_cavity':exact.point_location(b[0][0][0],a)}
        else:
            result['containment']={'status':'transverse_or_boundary_contacts_require_arrangement'}
        witnesses=[]
        for face in range(0,len(a),max(1,len(a)//16)):
            point=tuple(sum(p[k] for p in a[face][0])/Fraction(3) for k in range(3))
            location=exact.point_location(point,b)
            witnesses.append({'cavity_face':face,'centroid_location_in_wall':location})
            if {w['centroid_location_in_wall']['location'] for w in witnesses}>={'inside','outside'}:
                break
        result['cavity_boundary_witnesses']=witnesses
    else:
        result['containment']={'status':'not_admissible_until_source_embeddedness_defects_resolved'}
    pairs.append(result)
    print(atrium,'wall-cavity pairs',result['count'],'shared faces',result['exact_shared_triangles'],flush=True)

report={'schema':'NumiHuman.AtrialWallSourceAudit.v1','qualification':'offline_source_geometry_only',
        'archive':cavities['archive'],'source_lock_sha256':digest((args.root/'sources.lock.json').read_bytes()),
        'coordinate_semantics':'exact rational values of published binary64 metres',
        'intersection_owner_algorithm':exact.ALGORITHM,'broadphase':'exact_integer_AABB_tree_no_tolerance',
        'owners_sha256':{str(Path(m.__file__).relative_to(args.root)):digest(Path(m.__file__).read_bytes()) for m in (geometry,exact)},
        'script_sha256':digest(Path(__file__).read_bytes()),'surfaces':surfaces,'pairs':pairs,
        'source_coordinates_modified':False,'added_faces':0,'physical_stepping':False,
        'elapsed_seconds':time.monotonic()-started}
(args.output/'atrial-wall-audit.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print('audit complete',report['elapsed_seconds'],flush=True)
