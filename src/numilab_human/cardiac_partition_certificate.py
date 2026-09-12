"""Exact certificates for a source-bound two-cavity triangle arrangement.

The input geometry is the exact rational value of published binary64 metres.
Positive oriented coverage, complete cuts and independent parity classification
precede set composition. No physical state is advanced and no intersection is
repaired here. Historical source-surface admission remains unchanged.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import math

from . import cardiac_cavity_intersections as predicates
from .model import ImportError as HumanImportError
from .physiology import canonical

SOURCES = ('right_atrium', 'right_ventricle')
SCHEMA = 'HumanPack.cardiac-partition-certificate.v1'


def require(ok, message):
    if not ok:
        raise HumanImportError('cardiac partition certificate: ' + message)


def rational(value):
    require(type(value) in (int, Fraction), 'exact integer/Fraction coordinates required')
    return Fraction(value)


def _point(value):
    require(isinstance(value, (list, tuple)) and len(value) == 3, 'invalid point')
    p = tuple(rational(x) for x in value)
    require(all(abs(x) <= 10**6 and x.numerator.bit_length() <= 16384 and x.denominator.bit_length() <= 16384 for x in p),
            'unbounded exact coordinates')
    return p


def encode_rational(value):
    x = Fraction(value)
    # Hexadecimal avoids Python's decimal-string digit limit for exact moment
    # denominators without changing the process-wide safety setting.
    return [hex(x.numerator), hex(x.denominator)]


def encode_triangles(triangles):
    return [[[encode_rational(x) for x in p] for p in tri] for tri in triangles]


def source_geometry_sha256(surface):
    """Bind exact metre coordinates, original face IDs and source member SHA."""
    require(isinstance(surface, dict), 'invalid source surface')
    vertices = [_point(v) for v in surface.get('vertices', [])]
    faces = surface.get('triangles')
    require(vertices and isinstance(faces, list) and faces, 'empty source geometry')
    require(all(isinstance(t, (list, tuple)) and len(t) == 3 and all(type(i) is int and 0 <= i < len(vertices) for i in t)
                for t in faces), 'invalid source faces')
    member=surface.get('source_sha256')
    require(isinstance(member,str) and len(member)==64 and all(c in '0123456789abcdef' for c in member),'invalid source member identity')
    return hashlib.sha256(canonical({'source_sha256':member,'vertices_m': [[encode_rational(x) for x in p] for p in vertices],
                                     'triangles': [list(t) for t in faces]})).hexdigest()


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _cross(a, b):
    return predicates._cross(a, b)


def _dot(a, b):
    return predicates._dot(a, b)


def _normal(tri):
    return _cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _on_segment(point, a, b):
    return _cross(_sub(point, a), _sub(b, a)) == (0, 0, 0) and all(
        min(a[k], b[k]) <= point[k] <= max(a[k], b[k]) for k in range(3))


def _mesh(triangles):
    vertices, lookup, faces = [], {}, []
    for tri in triangles:
        ids = []
        for p in tri:
            if p not in lookup:
                lookup[p] = len(vertices); vertices.append(p)
            ids.append(lookup[p])
        faces.append(tuple(ids))
    return vertices, faces


def _topology(triangles, *, allow_empty=False):
    if not triangles:
        require(allow_empty, 'empty required surface')
        return {'closed': True, 'face_components': 0, 'vertex_count': 0, 'face_count': 0, 'edge_count': 0}
    vertices, faces = _mesh(triangles)
    require(len(set(tuple(sorted(f)) for f in faces)) == len(faces), 'duplicate face')
    edges, incident = defaultdict(list), defaultdict(list)
    for i, face in enumerate(faces):
        require(len(set(face)) == 3 and any(_normal(triangles[i])), 'degenerate triangle')
        for a, b in zip(face, face[1:]+face[:1]):
            edges[_edge_key(a,b)].append((a,b,i))
        for v in face:
            incident[v].append(i)
    require(all(len(rows)==2 and rows[0][:2] == tuple(reversed(rows[1][:2])) for rows in edges.values()),
            'open, nonmanifold or inconsistently oriented edge chain')
    adjacency = defaultdict(set)
    for rows in edges.values():
        a,b=rows[0][2],rows[1][2];adjacency[a].add(b);adjacency[b].add(a)
    unseen=set(range(len(faces)));components=0
    while unseen:
        components+=1;todo=[min(unseen)]
        while todo:
            i=todo.pop()
            if i in unseen:unseen.remove(i);todo.extend(adjacency[i] & unseen)
    for vertex, ids in incident.items():
        link=defaultdict(set);counts=Counter()
        for i in ids:
            a,b=[v for v in faces[i] if v!=vertex]
            link[a].add(b);link[b].add(a);counts[_edge_key(a,b)]+=1
        seen=set();todo=[next(iter(link))]
        while todo:
            i=todo.pop()
            if i not in seen:seen.add(i);todo.extend(link[i]-seen)
        require(len(seen)==len(link) and all(len(v)==2 for v in link.values()) and all(c==1 for c in counts.values()),
                'nonmanifold vertex link')
    return {'closed': True, 'face_components': components, 'vertex_count': len(vertices),
            'face_count': len(faces), 'edge_count': len(edges)}


def _face_coverage(parent, children):
    require(children, 'missing parent-face coverage')
    normal=_normal(parent)
    require(any(normal), 'degenerate source face')
    boundary=Counter()
    for tri in children:
        require(all(_dot(normal,_sub(p,parent[0]))==0 for p in tri), 'child is not on its source-face plane')
        require(all(predicates._inside(predicates._signs(p,1,parent,normal)) for p in tri), 'child lies outside source face')
        require(_dot(normal,_normal(tri))>0, 'child winding or area differs from source parent')
        for a,b in zip(tri,tri[1:]+tri[:1]):
            key=_edge_key(a,b);boundary[key]+=1 if (a,b)==key else -1
    intervals=[[] for _ in range(3)]
    for (a,b),count in boundary.items():
        if not count:continue
        require(abs(count)==1, 'duplicate or overlapping oriented face coverage')
        if count<0:a,b=b,a
        candidates=[i for i in range(3) if _on_segment(a,parent[i],parent[(i+1)%3]) and
                    _on_segment(b,parent[i],parent[(i+1)%3])]
        require(len(candidates)==1, 'uncancelled internal edge or incomplete face arrangement')
        i=candidates[0];p,q=parent[i],parent[(i+1)%3]
        axis=max(range(3),key=lambda k:abs(q[k]-p[k]))
        lo,hi=Fraction(a[axis]-p[axis],q[axis]-p[axis]),Fraction(b[axis]-p[axis],q[axis]-p[axis])
        require(0<=lo<hi<=1, 'parent boundary orientation differs')
        intervals[i].append((lo,hi))
    for spans in intervals:
        position=Fraction(0)
        for lo,hi in sorted(spans):
            require(lo==position, 'gap or overlap in source-face boundary coverage')
            position=hi
        require(position==1, 'incomplete source-face boundary coverage')


class _Prepared:
    def __init__(self, triangles):
        self.triangles=triangles
        self.vertices,self.faces=_mesh(triangles)
        self.records=predicates._records(self.vertices,self.faces)
        self.ordered=sorted(self.records,key=lambda row:row[1][0])
        self.starts=[r[1][0] for r in self.ordered]
        self.normals={r[3]:_normal(r[0]) for r in self.records}
        self.lo=tuple(min(p[k] for p in self.vertices) for k in range(3))
        self.hi=tuple(max(p[k] for p in self.vertices) for k in range(3))

    def candidates(self,tri):
        lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
        return [row for row in self.ordered[:bisect_right(self.starts,hi[0])]
                if all(lo[k]<=row[2][k] and row[1][k]<=hi[k] for k in range(3))]

    def certify_uncut(self,tri):
        for row in self.candidates(tri):
            points=predicates.triangle_intersection_points(tri,row[0])
            if not points:continue
            require(any(all(_on_segment(p,tri[i],tri[(i+1)%3]) for p in points) for i in range(3)),
                    'child interior crosses the other source or has unsupported coplanar overlap')

    def location(self,point):
        if any(point[k]<self.lo[k] or point[k]>self.hi[k] for k in range(3)):
            return 'outside'
        for row in self.records:
            if all(row[1][k]<=point[k]<=row[2][k] for k in range(3)):
                normal=self.normals[row[3]]
                if _dot(normal,_sub(point,row[0][0]))==0 and predicates._inside(predicates._signs(point,1,row[0],normal)):
                    return 'boundary'
        for attempt in range(1,65):
            direction=(1,attempt,attempt*attempt);crossings=0;ambiguous=False
            for row in self.records:
                # Positive-direction slab intersection avoids most exact plane
                # predicates. Fraction ratios keep this broadphase conservative.
                lo=max(Fraction(row[1][k]-point[k],direction[k]) for k in range(3))
                hi=min(Fraction(row[2][k]-point[k],direction[k]) for k in range(3))
                if hi<=0 or lo>hi:continue
                tri=row[0];normal=self.normals[row[3]]
                numerator,denominator=_dot(normal,_sub(tri[0],point)),_dot(normal,direction)
                if not denominator:
                    if not numerator:ambiguous=True;break
                    continue
                if numerator*denominator<=0:continue
                hit=tuple(point[k]*denominator+numerator*direction[k] for k in range(3))
                if denominator<0:denominator,hit=-denominator,tuple(-x for x in hit)
                signs=predicates._signs(hit,denominator,tri,normal)
                if predicates._inside(signs):
                    if 0 in signs:ambiguous=True;break
                    crossings+=1
            if not ambiguous:return 'inside' if crossings%2 else 'outside'
        return 'indeterminate'


def _moment_vector(triangles):
    # Numerators for integral(1,x,xx^T); local origins are unnecessary because
    # the arithmetic is rational rather than floating point.
    result=[0]*10
    for a,b,c in triangles:
        determinant=_dot(a,_cross(b,c));s=tuple(a[k]+b[k]+c[k] for k in range(3))
        result[0]+=Fraction(determinant,6)
        for i in range(3):result[1+i]+=Fraction(determinant*s[i],24)
        for k,(i,j) in enumerate(((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))):
            result[4+k]+=Fraction(determinant*(s[i]*s[j]+a[i]*a[j]+b[i]*b[j]+c[i]*c[j]),120)
    return tuple(Fraction(x) for x in result)


def _add(*vectors):
    return tuple(sum(x) for x in zip(*vectors))


def _neg(vector):
    return tuple(-x for x in vector)


def _moments_json(vector):
    q=vector[4:]
    return {'volume_m3':encode_rational(vector[0]),
            'first_volume_moment_m4':[encode_rational(x) for x in vector[1:4]],
            'second_volume_moment_m5':[[encode_rational(q[k]) for k in row] for row in ((0,1,2),(1,3,4),(2,4,5))]}


def build_triangle_sets(subtriangles):
    """Return exact triangle lists; caller may apply one common Float64 map."""
    groups={(s,loc):[] for s in SOURCES for loc in ('inside','outside')}
    for row in subtriangles:
        require(isinstance(row,dict) and row.get('source') in SOURCES and row.get('other_location') in ('inside','outside'),
                'invalid arrangement source or location')
        tri=tuple(_point(p) for p in row['vertices']);require(len(tri)==3,'invalid subtriangle')
        groups[(row['source'],row['other_location'])].append(tri)
    a,b=SOURCES;ai,ao,bi,bo=[groups[key] for key in ((a,'inside'),(a,'outside'),(b,'inside'),(b,'outside'))]
    reverse=lambda triangles:[(t[0],t[2],t[1]) for t in triangles]
    return {'sources':{a:ao+ai,b:bo+bi},'union':ao+bo,'intersection':ai+bi,
            'partitions':{a+'_priority':{'regions':{a:ao+ai,b:bo+reverse(ai)},'shared_interface':ai},
                          b+'_priority':{'regions':{a:ao+reverse(bi),b:bo+bi},'shared_interface':bi}}}


def certify_partition(source_surfaces,subtriangles,*,expected_geometry_sha256):
    require(isinstance(source_surfaces,dict) and set(source_surfaces)==set(SOURCES) and
            isinstance(expected_geometry_sha256,dict) and set(expected_geometry_sha256)==set(SOURCES),'source coverage differs')
    require(isinstance(subtriangles,list) and 1<=len(subtriangles)<=40000,'invalid arrangement size')
    originals,source_ids,scale={}, {}, 1
    for name in SOURCES:
        surface=source_surfaces[name]
        require(isinstance(surface,dict) and set(surface)=={'vertices','triangles','source_sha256'},'source fields differ')
        digest=source_geometry_sha256(surface)
        require(digest==expected_geometry_sha256[name],'source geometry identity differs')
        member=surface['source_sha256']
        require(isinstance(member,str) and len(member)==64 and all(c in '0123456789abcdef' for c in member),'invalid source member identity')
        vertices=[_point(p) for p in surface['vertices']]
        for p in vertices:
            for x in p:
                require(x.denominator & (x.denominator-1)==0 and Fraction.from_float(float(x))==x,
                        'source must be exact published binary64 metres')
                scale=max(scale,x.denominator)
        originals[name]=[tuple(vertices[i] for i in t) for t in surface['triangles']]
        source_ids[name]={'geometry_sha256':digest,'source_sha256':member}
    def scaled(p):
        values=[x*scale for x in p]
        return tuple(x.numerator if x.denominator==1 else x for x in values)
    prepared={}
    for name,triangles in originals.items():
        topology=_topology(triangles)
        require(topology['face_components']==1,'source must have one connected boundary')
        mesh=_Prepared([tuple(scaled(p) for p in t) for t in triangles])
        require(predicates._audit_pair(mesh.records,mesh.records,same_surface=True)['count']==0,'source self-intersection')
        prepared[name]=mesh
    coverage=defaultdict(list);classification=Counter();canonical_rows=[]
    for row in subtriangles:
        require(isinstance(row,dict) and set(row)=={'vertices','source','source_face','other_location'},'subtriangle fields differ')
        source,parent,location=row['source'],row['source_face'],row['other_location']
        require(source in SOURCES and type(parent) is int and 0<=parent<len(originals[source]),'forged source parent')
        require(location in ('inside','outside'),'unsupported classification')
        require(isinstance(row['vertices'],(list,tuple)) and len(row['vertices'])==3,'invalid child triangle')
        tri=tuple(_point(p) for p in row['vertices']);coverage[(source,parent)].append(tri)
        scaled_tri=tuple(scaled(p) for p in tri)
        other=prepared[SOURCES[1] if source==SOURCES[0] else SOURCES[0]]
        other.certify_uncut(scaled_tri)
        centroid=tuple(Fraction(sum(p[k] for p in scaled_tri),3) for k in range(3))
        observed=other.location(centroid)
        require(observed==location,'false or indeterminate other-source classification')
        classification[(source,location)]+=1
        canonical_rows.append({'source':source,'source_face':parent,'other_location':location,'vertices':encode_triangles([tri])[0]})
    for name,parents in originals.items():
        for i,parent in enumerate(parents):_face_coverage(parent,coverage[(name,i)])
    sets=build_triangle_sets(subtriangles)
    # Compute the four disjoint boundary classes once, then combine their exact
    # integrals. Coverage/classification established the set semantics above.
    group_moments={}
    for name in SOURCES:
        for location in ('inside','outside'):
            group_moments[(name,location)]=_moment_vector([tuple(_point(p) for p in r['vertices']) for r in subtriangles
                                                        if r['source']==name and r['other_location']==location])
    a,b=SOURCES;ai,ao,bi,bo=[group_moments[key] for key in ((a,'inside'),(a,'outside'),(b,'inside'),(b,'outside'))]
    source_moments={a:_add(ao,ai),b:_add(bo,bi)}
    for name in SOURCES:
        require(source_moments[name]==_moment_vector(originals[name]),'source-face moment coverage differs')
        require(source_moments[name][0]>0,'source winding must be outward')
    union,intersection=_add(ao,bo),_add(ai,bi)
    require(union==_add(source_moments[a],source_moments[b],_neg(intersection)),'union inclusion-exclusion failed')
    topology={'union':_topology(sets['union']),'intersection':_topology(sets['intersection'],allow_empty=True)}
    partitions={}
    for priority,moments in ((a,{a:source_moments[a],b:_add(bo,_neg(ai))}),
                              (b,{a:_add(ao,_neg(bi)),b:source_moments[b]})):
        name=priority+'_priority';regions=sets['partitions'][name]['regions'];interface=sets['partitions'][name]['shared_interface']
        require(all(moments[s][0]>0 for s in SOURCES),'priority partition removes an entire chamber')
        require(_add(*moments.values())==union,'partition moment conservation failed')
        region_topology={s:_topology(regions[s]) for s in SOURCES}
        # Opposite copies are constructed from one interface; prove that exact
        # oriented cancellation holds in the actual emitted triangle chains.
        _verify_interface(regions,interface)
        partitions[name]={'regions':{s:{'moments':_moments_json(moments[s]),'topology':region_topology[s],
                                       'triangle_sha256':hashlib.sha256(canonical(encode_triangles(regions[s]))).hexdigest()} for s in SOURCES},
                          'interface_face_count':len(interface),'shared_interface_opposite_chain':True,
                          'interiors_disjoint_by_source_membership':True,'union_moments_exact':True}
    return {'schema':SCHEMA,'coordinate_semantics':'exact_rational_value_of_published_binary64_metres',
            'source_identity':source_ids,'arrangement_sha256':hashlib.sha256(canonical(canonical_rows)).hexdigest(),
            'source_face_coverage_exact':True,'child_interiors_uncut':True,'independent_classification_exact':True,
            'classification_counts':{s:{loc:classification[(s,loc)] for loc in ('inside','outside')} for s in SOURCES},
            'source_moments':{s:_moments_json(v) for s,v in source_moments.items()},
            'intersection':{'moments':_moments_json(intersection),'topology':topology['intersection']},
            'union':{'moments':_moments_json(union),'topology':topology['union']},'partitions':partitions,
            'source_union_preserved':True,'source_exclusive_regions_preserved':True,
            'physical_stepping':False,'biological_valve_interface':False,'mechanical_mass_assigned':False}


def _oriented_key(tri):
    return min(tri,tri[1:]+tri[:1],tri[2:]+tri[:2])


def _verify_interface(regions,interface):
    require(interface,'empty shared interface')
    expected=Counter(_oriented_key(t) for t in interface)
    reverse=Counter(_oriented_key((t[0],t[2],t[1])) for t in interface)
    require(all(c==1 for c in expected.values()) and not (expected.keys() & reverse.keys()),'duplicate or contradictory interface')
    counts=[Counter(_oriented_key(t) for t in triangles) for triangles in regions.values()]
    require(len(counts)==2,'two regions required')
    valid=((all(counts[0][t]==1 for t in expected) and all(counts[1][t]==1 for t in reverse)) or
           (all(counts[1][t]==1 for t in expected) and all(counts[0][t]==1 for t in reverse)))
    require(valid,'interface is not present once with opposite winding')
    for tri in interface:
        for p in tri:require(len(p)==3,'invalid interface point')


def audit_shared_interface_partition(regions,shared_interface):
    """Audit final binary64 geometry independently of the rational construction.

    `regions` maps two IDs to triangle lists of float metre points. Interface
    triangles use the same globally converted coordinates. Contacts away from
    that declared interface complex are rejected, including nesting.
    """
    require(isinstance(regions,dict) and len(regions)==2,'two emitted regions required')
    def exact(triangles):
        require(isinstance(triangles,list) and triangles,'empty emitted triangles')
        result=[]
        for tri in triangles:
            require(isinstance(tri,(list,tuple)) and len(tri)==3,'invalid emitted triangle')
            row=[]
            for p in tri:
                require(isinstance(p,(list,tuple)) and len(p)==3 and all(type(x) is float and math.isfinite(x) for x in p),
                        'emitted coordinates must be finite Float64 values')
                row.append(tuple(Fraction.from_float(x) for x in p))
            result.append(tuple(row))
        return result
    precise={name:exact(triangles) for name,triangles in regions.items()};interface=exact(shared_interface)
    _verify_interface(precise,interface)
    scale=max(x.denominator for tris in list(precise.values())+[interface] for t in tris for p in t for x in p)
    integer=lambda triangles:[tuple(tuple(int(x*scale) for x in p) for p in tri) for tri in triangles]
    prepared={};reports={};moments={}
    for name,triangles in precise.items():
        topo=_topology(triangles)
        require(topo['face_components']==1,'emitted region must have one connected boundary')
        moments[name]=_moment_vector(triangles)
        require(moments[name][0]>0,'emitted region winding must be outward')
        mesh=_Prepared(integer(triangles))
        audit=predicates._audit_pair(mesh.records,mesh.records,same_surface=True)
        require(audit['count']==0,'emitted region self-intersects')
        prepared[name]=mesh;reports[name]={'topology':topo,'self_intersection_count':0,'moments':_moments_json(moments[name])}
    shared=integer(interface)
    shared_keys={tuple(sorted(t)) for t in shared}
    interface_vertices={p for t in shared for p in t}
    interface_edges={_edge_key(a,b) for t in shared for a,b in zip(t,t[1:]+t[:1])}
    a,b=list(prepared);contacts=0
    for row in prepared[a].records:
        for other in prepared[b].candidates(row[0]):
            points=predicates.triangle_intersection_points(row[0],other[0])
            if not points:continue
            contacts+=1
            if tuple(sorted(row[0]))==tuple(sorted(other[0])) and tuple(sorted(row[0])) in shared_keys:
                continue
            allowed=(all(p in interface_vertices for p in points) and len(set(points))==1) or any(
                all(_on_segment(p,x,y) for p in points) for x,y in interface_edges)
            uncut=all(any(all(_on_segment(p,t[i],t[(i+1)%3]) for p in points) for i in range(3))
                      for t in (row[0],other[0]))
            require(allowed and uncut,'emitted regions intersect outside their declared interface or through a face interior')
    # Opposite interface normals do not justify arbitrary crossings at shared
    # edges. Every other open boundary patch must independently be outside the
    # opposite solid after its uncut interior has been established above.
    for name,mesh in prepared.items():
        other=prepared[b if name==a else a];checked=0
        for tri in mesh.triangles:
            if tuple(sorted(tri)) in shared_keys:continue
            center=tuple(Fraction(sum(p[k] for p in tri),3) for k in range(3))
            require(other.location(center)=='outside','non-interface emitted boundary patch is inside the other region')
            checked+=1
        reports[name]['outside_noninterface_face_count']=checked
    # A strict interior witness prevents accepting nested regions with no
    # transverse boundary crossing. Inward steps use exact rational points.
    witnesses={}
    for name,mesh in prepared.items():
        other=prepared[b if name==a else a]
        found=None
        for tri in mesh.triangles[:32]:
            center=tuple(Fraction(sum(p[k] for p in tri),3) for k in range(3));normal=_normal(tri)
            norm=max(abs(x) for x in normal);extent=max(mesh.hi[k]-mesh.lo[k] for k in range(3))
            for power in range(4,132):
                p=tuple(center[k]-Fraction(normal[k]*extent,norm*(2**power)) for k in range(3))
                own=mesh.location(p)
                if own=='inside':
                    require(other.location(p)=='outside','emitted region interior lies in the other region')
                    found=p;break
            if found is not None:break
        require(found is not None,'could not establish strict emitted interior witness')
        witnesses[name]=[encode_rational(Fraction(x,scale)) for x in found]
    return {'schema':'HumanPack.cardiac-shared-interface-audit.v1','coordinate_semantics':'exact_binary64_metres',
            'per_region':reports,'cross_domain_contact_pairs':contacts,'shared_interface_face_count':len(interface),
            'shared_interface_opposite_chain':True,'interiors_disjoint':True,'strict_interior_witnesses_m':witnesses,
            'union_moments':_moments_json(_add(*moments.values())),
            'biological_valve_interface':False,'mechanical_mass_assigned':False}
