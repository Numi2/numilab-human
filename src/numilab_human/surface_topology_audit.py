"""Independent source-chain, FP32 embeddedness and native repair witnesses.

Closed topology is not anatomical or physical-volume admission. Exact triangle
predicates include adjacent triangles; only their common vertex/edge is exempt.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math

from . import lung_envelope as lung
from . import model as human
from .cardiac_cavity_geometry import analyze_topology
from . import cardiac_cavity_intersections as intersection
from . import surface_topology_repair as repair
from .torso_anatomy_audit import _pack_sections, anatomy_hidden_ids


def require(ok, message):
    if not ok:
        raise human.ImportError('surface topology audit: ' + message)


def verify_derivation(keys, source_faces, derived_faces, proof):
    """Prove an oriented source-chain identity without rerunning the repair."""
    require(all(len(p) == 3 and all(isinstance(x, Fraction) for x in p) for p in keys),
            'exact source coordinates')
    require(all(len(f) == 3 and all(type(x) is int and 0 <= x < len(keys) for x in f)
                for f in source_faces), 'source face ranges')
    vertex_ids = proof['source_vertex_ids']
    face_ids = proof['source_face_ids']
    require(all(type(x) is int and 0 <= x < len(keys) for x in vertex_ids)
            and len(set(vertex_ids)) == len(vertex_ids)
            and len({keys[i] for i in vertex_ids}) == len(vertex_ids), 'unique source vertex ancestry')
    require(len(face_ids) == len(derived_faces) and all(type(x) is int and 0 <= x < len(source_faces)
                                                    for x in face_ids), 'source face ancestry')
    require(all(len(f) == 3 and all(type(x) is int and 0 <= x < len(vertex_ids) for x in f)
                for f in derived_faces), 'derived face ranges')
    require(set(i for f in derived_faces for i in f) == set(range(len(vertex_ids))), 'complete used derived vertices')
    def normal(points):
        a,b,c = points
        return intersection._cross(intersection._sub(b,a), intersection._sub(c,a))
    zero = proof['removed_exact_zero_area_source_faces']
    pairs = proof['cancelled_opposite_source_face_pairs']
    dropped = list(zero)
    require(all(type(i) is int and 0 <= i < len(source_faces) for i in zero), 'zero-area face ranges')
    for i in zero:
        require(not any(normal([keys[k] for k in source_faces[i]])), 'false zero-area removal')
    for pair in pairs:
        require(len(pair) == 2 and all(type(i) is int and 0 <= i < len(source_faces) for i in pair),
                'cancelled face ranges')
        a,b = [[keys[k] for k in source_faces[i]] for i in pair]
        na,nb = normal(a),normal(b)
        require(len(set(a)) == 3 and set(a) == set(b) and any(na)
                and na == tuple(-x for x in nb), 'false opposite-face cancellation')
        dropped.extend(pair)
    require(len(dropped) == len(set(dropped)), 'source face removed more than once')
    weights = defaultdict(Fraction)
    for f,parent in zip(derived_faces,face_ids,strict=True):
        a,b,c = [keys[k] for k in source_faces[parent]]
        original = normal([a,b,c]); require(any(original), 'degenerate retained source face')
        axis = next(k for k in range(3) if original[k])
        points = [keys[vertex_ids[k]] for k in f]
        child = normal(points); weight = child[axis]/original[axis]
        require(weight > 0 and child == tuple(weight*x for x in original), 'changed face orientation or plane')
        for p in points:
            bary = [normal([p,b,c])[axis]/original[axis], normal([a,p,c])[axis]/original[axis],
                    normal([a,b,p])[axis]/original[axis]]
            require(all(x >= 0 for x in bary) and sum(bary) == 1 and
                    p == tuple(sum(w*v[k] for w,v in zip(bary,[a,b,c])) for k in range(3)),
                    'derived face outside its source triangle')
        weights[parent] += weight
    require(set(weights) == set(range(len(source_faces)))-set(dropped)
            and all(w == 1 for w in weights.values()), 'incomplete retained source face partition')
    # Direct rational triple product, distinct from the composer's integer grid.
    def integral(points, faces):
        return sum(sum(x*y for x,y in zip(points[a], intersection._cross(points[b],points[c])))
                   for a,b,c in faces)
    before = integral(keys, source_faces)
    after = integral([keys[i] for i in vertex_ids], derived_faces)
    require(before == after and proof['signed_six_integral'] ==
            {'numerator':before.numerator,'denominator':before.denominator}, 'signed-integral witness')
    require(proof['exact_signed_integral_preserved'] is True
            and proof['retained_source_support_preserved'] is True
            and proof['vertex_coordinates_modified'] is False and proof['hole_capping'] is False,
            'source-only repair boundary')
    return {'oriented_source_chain_identity':True,'source_triangles_accounted':len(source_faces),
            'removed_zero_area_faces':len(zero),'cancelled_opposite_face_pairs':len(pairs),
            'retained_source_face_partitions':len(weights),'exact_signed_integral_preserved':True}


def exact_embedding(vertices, faces):
    """Complete self-pair audit with an exact integer AABB tree broad phase."""
    require(all(len(p) == 3 and all(type(x) in (int,float) and math.isfinite(x) for x in p)
                for p in vertices), 'finite executed coordinates')
    require(all(len(f) == 3 and all(type(i) is int and 0 <= i < len(vertices) for i in f)
                for f in faces), 'executed face ranges')
    rational = [tuple(Fraction.from_float(float(x)) for x in p) for p in vertices]
    denominator = max(x.denominator for p in rational for x in p)
    points = [tuple(x.numerator*(denominator//x.denominator) for x in p) for p in rational]
    records = intersection._records(points, faces)
    require(bool(records), 'empty embedding surface')
    candidates = allowed = 0; defects = []
    for (tri,lo,hi,i,ids), (other,_,_,j,other_ids) in intersection._aabb_candidate_pairs(
            records,records,same_surface=True):
        candidates += 1
        hits = intersection.triangle_intersection_points(tri,other)
        if not hits: continue
        shared = set(ids)&set(other_ids)
        common = {tri[ids.index(k)] for k in shared}
        if len(shared) in (1,2) and all(intersection._allowed_shared_point(p,common) for p in hits):
            allowed += 1
        else: defects.append([i,j])
    topology = analyze_topology(vertices,faces)
    return {'triangle_pairs':sorted(defects),'count':len(defects),'aabb_candidate_pairs':candidates,
            'allowed_shared_vertex_or_edge_pairs':allowed,
            'self_intersection_free':not defects,'topology':repair.compact_topology(topology),
            'physical_volume':False,'intercomponent_containment':'not_assessed',
            'algorithm':'complete_exact_FP32_integer_triangle_predicates_with_AABB_tree_v1'}


def audit_source(sources, base_payload, payload):
    import numpy as np
    specs,provenance = repair.source_specs(base_payload)
    _,old_h,old_r,old_v,old_i = lung.decode(base_payload)
    _,h,records,vertices,indices = lung.decode(payload)
    manifest = human.read_json(payload.with_name(repair.MANIFEST))
    candidates = manifest['candidates']
    require(h[:2] == (5,579+len(candidates)) and h[4:] == old_h[4:], 'candidate header identity')
    require(records[:579].tobytes() == old_r.tobytes() and vertices[:len(old_v)].tobytes() == old_v.tobytes()
            and indices[:len(old_i)].tobytes() == old_i.tobytes(), 'raw geometry changed')
    require(manifest['base_payload_sha256'] == repair.BASE_SHA and manifest['predecessor_manifests'] == provenance
            and manifest['payload'] == {'abi':5,'surfaces':h[1],'vertices':h[2],'indices':h[3],
                                       'sha256':human.sha256(payload)}, 'candidate provenance')
    require(manifest['raw_579_geometry_byte_identical'] is True and manifest['new_anatomical_members'] == 0
            and manifest['physical_volume'] is False and manifest['mechanics'] is False
            and manifest['clinical_registration'] is False, 'candidate admission boundary')
    require(manifest['raw_parent_ids_for_candidate_inspection'] == [c['stable_id'] for c in candidates]
            and len({c['stable_id'] for c in candidates}) == len(candidates), 'inspection parent identities')
    rows = []; nv,ni = len(old_v),len(old_i)
    for i,(c,record) in enumerate(zip(candidates,records[579:],strict=True),580):
        spec = specs[c['stable_id']-1]
        require(all(c[k] == v for k,v in spec.items()) and c['candidate_stable_id'] == i,
                'source/candidate identity')
        require(record.tolist() == [spec['core_body_index'],nv,c['vertex_count'],ni,c['triangle_count']*3,
                                    i,spec['layer_code'],0], 'candidate owner/type/range')
        raw_parent = old_r[c['stable_id']-1]
        f = (indices[ni:ni+record[4]]-nv).reshape(-1,3).tolist()
        _,keys,source_faces = repair.source_mesh(spec,sources)
        proof = c['source_derivation']
        chain = verify_derivation(keys,source_faces,f,proof)
        require(np.array_equal(vertices[nv:nv+record[2],:3],
                               old_v[int(raw_parent[1])+np.asarray(proof['source_vertex_ids']),:3]),
                'candidate moved a compiled source position')
        local = vertices[nv:nv+record[2],:3].astype(float)
        # Scalar accumulation is an independent oracle for the vector composer.
        normals = np.zeros_like(local)
        for a,b,d in f:
            n = np.cross(local[b]-local[a],local[d]-local[a])
            for v in [a,b,d]: normals[v] += n
        lengths = np.linalg.norm(normals,axis=1)
        require((lengths > 0).all(), 'normal degeneracy')
        normals /= lengths[:,None]
        require(np.max(np.linalg.norm(vertices[nv:nv+record[2],3:]-normals,axis=1)) <= 2e-7,
                'candidate normal direction')
        embedded = exact_embedding(local.tolist(),f)
        require(embedded['topology'] == c['executed_FP32_topology']
                and embedded['topology']['closed_oriented_manifold_candidate'], 'executed topology receipt')
        rows.append({'source_stable_id':c['stable_id'],'candidate_stable_id':i,'member_id':c.get('member_id'),
                     'source_chain':chain,'executed_FP32_embedding':embedded,'passed_source_chain_and_topology':True})
        nv += int(record[2]); ni += int(record[4])
        print({'candidate':i,'source':c['stable_id'],'exact_intersection_pairs':embedded['count']},flush=True)
    require(nv == len(vertices) and ni == len(indices), 'complete candidate geometry ownership')
    return {'schema':'numi.human.source-topology-repair-audit.v1','passed':True,'rows':rows,
            'raw_579_geometry_byte_identical':True,'candidate_count':len(rows),
            'intersection_free_candidate_count':sum(r['executed_FP32_embedding']['self_intersection_free'] for r in rows),
            'payload_sha256':human.sha256(payload),'physical_volume':False,'clinical_anatomy':False,'mechanics':False}


def audit_native(payload,native_pack,native_poses,mask,hidden_ids):
    import numpy as np
    _,h,records,vertices,indices = lung.decode(payload)
    snapshot = human.read_json(native_poses)
    hidden = anatomy_hidden_ids(snapshot,hidden_ids,h[1])
    require(snapshot['surface_count'] == h[1] and snapshot['visible_layer_mask'] == mask
            and snapshot['registration_fingerprint32'] == h[4], 'native profile identity')
    owners = {r['body_index'] for r in snapshot['bodies']}
    require(len(owners) == len(snapshot['bodies']) and owners == set(map(int,records[:,0])), 'native owner coverage')
    sections = _pack_sections(native_pack)
    pv = np.frombuffer(sections[2][0],'<f4').reshape(-1,20)
    pi = np.frombuffer(sections[3][0],'<u4')
    primitives = np.frombuffer(sections[4][0],'<u4').reshape(-1,16)
    iu = np.frombuffer(sections[5][0],'<u4').reshape(-1,20)
    transform = np.frombuffer(sections[5][0],'<f4').reshape(-1,20)
    semantics = [51010,51011,51012,51020,51021,51022,51023,51024,51025,51026,51027,51028,51029,51030,51031]
    anatomy = primitives[np.isin(primitives[:,4],semantics)]
    require(len(anatomy) == h[1] and set(map(int,anatomy[:,5])) == set(range(1,h[1]+1)), 'native all surface identities')
    by_id = {int(p[5]):p for p in anatomy}; seen = set(); ranges = []
    for body,fv,nv,fi,ni,stable,layer,_ in records.tolist():
        p = by_id[stable];first,count,_,inst = map(int,p[:4])
        require(count == ni and first+ni <= len(pi) and inst < len(iu) and inst not in seen,
                'native independent primitive instance range')
        seen.add(inst)
        require(p[4] == semantics[layer-1] and p[6] == body
                and iu[inst,9] == body and iu[inst,10] == 3
                and np.array_equal(iu[inst,12:16],p[4:8]), 'native semantic/owner identity')
        require(iu[inst,11] == (11 if mask&(1<<(layer-1)) and stable not in hidden else 0), 'native visibility')
        require(np.array_equal(transform[inst,:8],[0,0,0,1,0,0,0,1]), 'native local transform')
        start = int(pi[first:first+ni].min())
        require(start+nv <= len(pv) and np.array_equal(pi[first:first+ni]-start,indices[fi:fi+ni]-fv),
                'native candidate topology differs')
        require(np.array_equal(pv[start:start+nv,:3],vertices[fv:fv+nv,:3])
                and np.array_equal(pv[start:start+nv,4:7],vertices[fv:fv+nv,3:]), 'native payload geometry differs')
        ranges.append((start,start+nv))
    ranges.sort()
    require(all(a[1] <= b[0] for a,b in zip(ranges,ranges[1:])), 'native surface vertex ranges overlap')
    return {'schema':'numi.human.native-topology-repair-audit.v1','passed':True,'surface_count':h[1],
            'candidate_count':h[1]-579,'hidden_parent_ids':list(hidden_ids),'visible_layer_mask':mask,
            'payload_sha256':human.sha256(payload),'native_pack_sha256':human.sha256(native_pack),
            'snapshot_sha256':human.sha256(native_poses),'native_local_geometry_bit_identical':True,
            'physical_volume':False,'mechanics':False,'clinical_anatomy':False}
