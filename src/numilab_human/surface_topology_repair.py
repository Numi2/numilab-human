"""Source-support-preserving surface candidates; never a physical volume owner.

Exact decimal (BodyParts3D) or binary64 (Z-Anatomy) coordinates permit oriented
face cancellation and conforming subdivision of exactly straight seams. The
raw published composite is retained byte for byte, followed by derived copies.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct

from . import model as human
from . import lung_envelope as lung
from .cardiac_cavity_geometry import parse_obj, exact_coordinate_quotient, analyze_topology

ROOT = human.REPOSITORY_ROOT
PAYLOAD = 'source-topology-repair-candidates.nhanatomy'
MANIFEST = 'source-topology-repair-candidates.manifest.json'
BASE_SHA = 'b2d2490a5fcf37c5229eca04e9c20da95e538f5dc6bcbbccdf2ab68da1f8f580'
MANIFEST_HASHES = [
    '37c0b02b473f1d9ec612a03e7cf425f254f8bf704d21d2f29457385821c56b7f',
    '20316ae7a690e2b184687c3349b35a8859dc1f03e5f11272827c108131802a08',
    '64ec90641612a63420fe29c5fb29fb98780fe71ef021890900af8e3e270b9be3',
    '3de6fd5f976d23b2607f42a70171c44ca58c1b87dfe5b695053afdef8727b0f8',
]


def require(ok, message):
    if not ok:
        raise human.ImportError('surface topology repair: ' + message)


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def compact_topology(t):
    return {k: len(v) if isinstance(v, list) else v for k, v in t.items()
            if k not in ['boundary_edges', 'boundary_loops', 'face_components']}


def signed_six_volume(keys, faces):
    # One exact integer grid per source; no tolerance or origin approximation.
    denominator = math.lcm(*(x.denominator for p in keys for x in p))
    v = [tuple(x.numerator*(denominator//x.denominator) for x in p) for p in keys]
    total = 0
    for a, b, c in faces:
        n = cross(v[b], v[c]); total += sum(x*y for x, y in zip(v[a], n))
    return Fraction(total, denominator**3)


def source_mesh(spec, sources):
    if spec['provider'] == 'bodyparts3d':
        _, member, obj = human._bodyparts_obj_member(sources, spec['hierarchy'], spec['member_id'])
        require(hashlib.sha256(obj).hexdigest() == spec['source_member_sha256'], 'source OBJ hash')
        p = parse_obj(obj, member)
        return p['vertices_mm'], p['coordinate_keys'], p['triangles']
    d = lung.source_data()
    require(spec['source_export_sha256'] == lung.configuration()['export']['sha256'], 'Z-Anatomy source identity')
    mesh = next(x for x in d['objects'] if x['object_name'] == spec['object_name'])
    v = mesh['vertices_world_m']
    return v, [tuple(Fraction.from_float(float(x)) for x in p) for p in v], mesh['triangles']


def source_specs(base):
    """Bind all 579 source records through their retained predecessor manifests."""
    base = base.resolve()
    _, header, records, _, _ = lung.decode(base)
    require(human.sha256(base) == BASE_SHA and header[:2] == (5, 579), 'published 579-surface identity')
    paths = [ROOT/'Build/lung-source-coverage-20260929/payload/bodyparts3d-myosim-torso-anatomy.manifest.json',
             ROOT/'Build/lung-envelope-20260929/payload/thorax-lung-envelope.manifest.json',
             ROOT/'Build/organ-family-coverage-20260929/payload.final/source-organ-family-anatomy.manifest.json',
             base.with_name('source-organ-family-anatomy.manifest.json')]
    require([human.sha256(p) for p in paths] == MANIFEST_HASHES, 'verified predecessor manifest identities')
    docs = [human.read_json(p) for p in paths]
    # Every predecessor's payload hash is independently pinned by its successor.
    require(docs[3]['payload']['sha256'] == BASE_SHA
            and docs[3]['base_payload_sha256'] == docs[2]['payload']['sha256']
            and docs[2]['base_payload_sha256'] == docs[1]['payload']['sha256']
            and docs[1]['base_payload_sha256'] == docs[0]['payload']['sha256'], 'predecessor provenance chain')
    result = []
    for i, s in enumerate(docs[0]['source']['surfaces'], 1):
        result.append({'stable_id': i, 'provider':'bodyparts3d', 'member_id':s['member_id'],
                       'hierarchy':s['hierarchy'], 'source_member_sha256':s['member_sha256']})
    for s in docs[1]['surfaces']:
        result.append({'stable_id':s['stable_id'], 'provider':'zanatomy', 'object_name':s['object_name'],
                       'source_export_sha256':docs[1]['source_export_sha256']})
    for doc in docs[2:]:
        for s in doc['surfaces']:
            result.append({'stable_id':s['stable_id'], 'provider':'bodyparts3d', 'member_id':s['member_id'],
                           'hierarchy':s['hierarchy'], 'source_member_sha256':s['source_member_sha256']})
    require(len(result) == 579 and [s['stable_id'] for s in result] == list(range(1,580)), 'complete source identities')
    for s, rec in zip(result, records, strict=True):
        s.update(core_body_index=int(rec[0]), layer_code=int(rec[6]), source_vertex_count=int(rec[2]), source_face_count=int(rec[4])//3)
    return result, {str(p.relative_to(ROOT)):human.sha256(p) for p in paths}


def repair(vertices, keys, source_faces):
    """Reduce a source-oriented chain and conform only exactly flat seam loops."""
    require(len(vertices) == len(keys) and len(vertices) > 0, 'coordinate/key coverage')
    unique, first, source_map, qkeys, qv = {}, [], [], [], []
    for i, (v, key) in enumerate(zip(vertices, keys, strict=True)):
        require(len(key) == 3 and all(isinstance(x, Fraction) for x in key), 'exact coordinate keys')
        if key not in unique:
            unique[key] = len(qkeys);qkeys.append(key);qv.append(v);first.append(i)
        source_map.append(unique[key])
    faces = [tuple(source_map[i] for i in f) for f in source_faces]
    before = analyze_topology(qv, [list(f) for f in faces])
    groups = defaultdict(lambda: [[], []]);removed_zero = [];removed_pairs = [];drop = set()
    for i, f in enumerate(faces):
        a,b,c = (qkeys[k] for k in f)
        if not any(cross(sub(b,a),sub(c,a))):
            drop.add(i);removed_zero.append(i);continue
        parity = sum(f[j]>f[k] for j in range(3) for k in range(j+1,3)) % 2
        groups[tuple(sorted(f))][parity].append(i)
    for g in groups.values():
        for a,b in zip(*g):
            drop.update([a,b]);removed_pairs.append([a,b])
    rows = [(f,i) for i,f in enumerate(faces) if i not in drop]
    subdivisions = []
    if rows:
        interim = analyze_topology(qv, [list(f) for f,_ in rows])
        edge_splits = {}
        for loop in interim['boundary_loops']:
            a,b = qkeys[loop[0]],qkeys[loop[1]];line = sub(b,a)
            if not any(line) or any(any(cross(line,sub(qkeys[k],a))) for k in loop[2:]):
                continue
            for a,b in zip(loop,loop[1:]+loop[:1]):
                delta = sub(qkeys[b],qkeys[a]);axis = next(k for k in range(3) if delta[k])
                points=[]
                for p in loop:
                    t=(qkeys[p][axis]-qkeys[a][axis])/delta[axis]
                    if 0<t<1:
                        require(qkeys[p] == tuple(qkeys[a][k]+t*delta[k] for k in range(3)), 'exact seam incidence')
                        points.append((t,p))
                if points:edge_splits[(a,b)] = [p for _,p in sorted(points)]
        refined=[]
        for face,parent in rows:
            hits=[j for j in range(3) if (face[j],face[(j+1)%3]) in edge_splits]
            require(len(hits) <= 1, 'ambiguous conforming source seam')
            if not hits:refined.append((face,parent));continue
            j=hits[0];a,b,c=face[j],face[(j+1)%3],face[(j+2)%3]
            chain=[a,*edge_splits[(a,b)],b]
            children=[(chain[k],chain[k+1],c) for k in range(len(chain)-1)]
            refined.extend((f,parent) for f in children)
            subdivisions.append({'source_face_id':parent,'source_quotient_edge':[a,b],'inserted_source_quotient_vertices':chain[1:-1], 'derived_faces':[list(f) for f in children]})
        rows=refined
    require(rows, 'repair erased entire source surface')
    # Every retained face is either unchanged or partitions its source triangle
    # with positive exact oriented area. This proves unchanged geometric support.
    weights=defaultdict(Fraction)
    for face,parent in rows:
        original=faces[parent];a,b,c=(qkeys[k] for k in original);normal=cross(sub(b,a),sub(c,a))
        axis=next(k for k in range(3) if normal[k]);p,q,r=(qkeys[k] for k in face)
        n=cross(sub(q,p),sub(r,p));weight=n[axis]/normal[axis]
        require(weight>0 and n == tuple(weight*x for x in normal), 'oriented face support')
        for point in [p,q,r]:
            bary=[cross(sub(b,point),sub(c,point))[axis]/normal[axis],
                  cross(sub(c,point),sub(a,point))[axis]/normal[axis],
                  cross(sub(a,point),sub(b,point))[axis]/normal[axis]]
            require(all(x>=0 for x in bary) and sum(bary)==1 and
                    point==tuple(sum(w*v[k] for w,v in zip(bary,[a,b,c])) for k in range(3)), 'source triangle barycentric support')
        weights[parent]+=weight
    require(set(weights) == set(range(len(faces)))-drop and all(w==1 for w in weights.values()), 'complete retained source support')
    used=sorted({j for f,_ in rows for j in f});index={old:new for new,old in enumerate(used)}
    v=[qv[j] for j in used];k=[qkeys[j] for j in used];f=[[index[j] for j in face] for face,_ in rows]
    after=analyze_topology(v,f);old_volume=signed_six_volume(qkeys,faces);new_volume=signed_six_volume(k,f)
    require(old_volume==new_volume,'exact signed-integral conservation')
    return {'vertices':v,'triangles':f,'source_vertex_ids':[first[j] for j in used],
            'source_face_ids':[parent for _,parent in rows],
            'source_vertex_to_quotient_vertex':source_map,'source_quotient_vertex_ids':used,
            'removed_exact_zero_area_source_faces':removed_zero,'cancelled_opposite_source_face_pairs':removed_pairs,
            'conforming_seam_subdivisions':subdivisions,
            'before':compact_topology(before),'after':compact_topology(after),
            'signed_six_integral':{'numerator':old_volume.numerator,'denominator':old_volume.denominator},
            'exact_signed_integral_preserved':True,'retained_source_support_preserved':True,
            'vertex_coordinates_modified':False,'hole_capping':False}


def compose(sources, base, output):
    import numpy as np
    lock = human.read_json(ROOT/'sources.lock.json')['sources']['bodyparts3d_4']['files']
    for filename in ['isa_BP3D_4.0_obj_99.zip','partof_BP3D_4.0_obj_99.zip']:
        path = sources/filename
        require(path.stat().st_size == lock[filename]['bytes'] and human.sha256(path) == lock[filename]['sha256'], 'source archive identity')
    specs, provenance=source_specs(base)
    _,h,records,vertices,indices=lung.decode(base)
    record_parts=[records.tobytes()];vertex_parts=[vertices.tobytes()];index_parts=[indices.tobytes()]
    nv,ni=h[2:4];candidate_count=0;outcomes=[];candidates=[]
    for spec,record in zip(specs,records,strict=True):
        v,keys,f=source_mesh(spec,sources)
        require(len(v)==record[2] and len(f)*3==record[4], 'source/raw geometry range')
        q=repair(v,keys,f)
        changed=bool(q['removed_exact_zero_area_source_faces'] or q['cancelled_opposite_source_face_pairs'] or q['conforming_seam_subdivisions'])
        outcome={**spec,'source_topology_before':q['before'],'source_topology_after':q['after'],
                 'source_support_preserved':True,'source_signed_integral_preserved':True,
                 'removed_zero_area_faces':len(q['removed_exact_zero_area_source_faces']),
                 'cancelled_opposite_face_pairs':len(q['cancelled_opposite_source_face_pairs']),
                 'conforming_seam_count':len(q['conforming_seam_subdivisions']),
                 'clinical_anatomy':False,'mechanics':False,'physical_volume':False}
        if not changed:outcome['status']='unchanged_source_closed_candidate' if q['after']['closed_oriented_manifold_candidate'] else 'unresolved_source_topology';outcomes.append(outcome);continue
        if not q['after']['closed_oriented_manifold_candidate']:
            outcome['status']='repair_refused_remaining_source_topology';outcomes.append(outcome);continue
        local=vertices[int(record[1])+np.asarray(q['source_vertex_ids']),:3]
        faces=np.asarray(q['triangles'],dtype='<u4')
        topology=analyze_topology(local.tolist(),faces.tolist())
        outcome['executed_FP32_topology']=compact_topology(topology)
        if not topology['closed_oriented_manifold_candidate']:
            outcome['status']='repair_refused_FP32_topology';outcomes.append(outcome);continue
        normals=np.zeros_like(local,dtype=np.float64)
        face_normals=np.cross(local[faces[:,1]].astype(float)-local[faces[:,0]],local[faces[:,2]].astype(float)-local[faces[:,0]])
        for corner in range(3):np.add.at(normals,faces[:,corner],face_normals)
        length=np.linalg.norm(normals,axis=1)
        require((length>0).all() and np.isfinite(length).all(),'derived normal degeneracy')
        normals/=length[:,None]
        stable=580+candidate_count;candidate_count+=1
        record_parts.append(struct.pack('<8I',int(record[0]),nv,len(local),ni,faces.size,stable,int(record[6]),0))
        vertex_parts.append(np.column_stack([local,normals]).astype('<f4').tobytes())
        index_parts.append((faces.ravel()+nv).astype('<u4').tobytes())
        detail={**spec,'candidate_stable_id':stable,'vertex_count':len(local),'triangle_count':len(faces),
                'source_derivation':{k:value for k,value in q.items() if k not in ['vertices','triangles']},
                'executed_FP32_topology':compact_topology(topology),'self_intersections':'pending_exact_audit',
                'intercomponent_containment':'not_assessed','physical_volume':False,'mechanics':False}
        candidates.append(detail);outcome.update(status='closed_source_and_FP32_candidate',candidate_stable_id=stable)
        outcomes.append(outcome);nv+=len(local);ni+=faces.size
        print(json.dumps({'source_stable_id':spec['stable_id'],'candidate':stable,'member':spec.get('member_id',spec.get('object_name')),'vertices':len(local),'triangles':len(faces)}),flush=True)
    require(len(outcomes)==579 and nv<=1000000 and ni<=6000000,'complete bounded native composition')
    raw=lung.HEADER.pack(b'NHANAT1\0',5,579+candidate_count,nv,ni,h[4],h[5])+b''.join(record_parts)+b''.join(vertex_parts)+b''.join(index_parts)
    output.mkdir(parents=True,exist_ok=True);path=output/PAYLOAD;path.write_bytes(raw)
    manifest={'schema':'numi.human.source-topology-repair-candidates.v1','payload':{'abi':5,'surfaces':579+candidate_count,'vertices':nv,'indices':ni,'sha256':human.sha256(path)},
              'base_payload_sha256':BASE_SHA,'predecessor_manifests':provenance,'source_outcomes':outcomes,'candidates':candidates,
              'raw_579_geometry_byte_identical':True,'new_anatomical_members':0,'raw_parent_ids_for_candidate_inspection':[s['stable_id'] for s in candidates],
              'physical_volume':False,'clinical_registration':False,'mechanics':False,
              'boundary':'Every raw source surface retained. Added copies are exact-support topology candidates, with source triangle ancestry and exact signed-integral preservation. No moved source points or capped openings. Source and executed FP32 topology are distinct gates. Self-intersection/containment, clinical anatomy, physical volume, materials, mechanics and whole-Human qualification remain separate.'}
    human.write_json(output/MANIFEST,manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['compose']);p.add_argument('--sources',type=Path,required=True);p.add_argument('--base-payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compose(a.sources,a.base_payload,a.output)
    print(json.dumps({'payload':result['payload'],'candidates':len(result['candidates']),'source_outcomes':len(result['source_outcomes'])}))

if __name__=='__main__':main()
