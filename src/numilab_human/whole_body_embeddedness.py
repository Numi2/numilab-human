"""Resumable exact self-intersection census of all compiled Human source meshes.

This audits individual source surfaces only. A closed, embedded surface here is
still not a clinical organ, disjoint tissue region, physical volume, or mass
owner. Each completed row is tied to the exact payload slice and predicate
program, so a stopped run can resume without losing completed measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

from . import model as human
from . import lung_envelope as lung
from . import surface_topology_repair as repair
from . import surface_topology_audit as audit
from . import cardiac_cavity_intersections as intersection
from .cardiac_cavity_geometry import analyze_topology

SCHEMA = 'numi.human.compiled-whole-body-embeddedness.v1'
ROW_SCHEMA = 'numi.human.compiled-surface-embeddedness.v1'
CODE_FILES = ('whole_body_embeddedness.py', 'surface_topology_audit.py',
              'cardiac_cavity_intersections.py', 'cardiac_cavity_geometry.py')


def require(ok, message):
    if not ok:
        raise human.ImportError('whole-body embeddedness: ' + message)


def geometry_hash(vertices, indices):
    # This binds the compiled Float32 positions and globally indexed faces,
    # including their exact byte order. Vertex normals are separately retained
    # in the immutable baseline payload but do not change triangle embedding.
    return hashlib.sha256(vertices[:, :3].tobytes() + indices.tobytes()).hexdigest()


def atomic_json(path, value):
    """Publish each checkpoint only after its complete JSON bytes are written."""
    pending=path.with_name(path.name+f'.{os.getpid()}.pending')
    raw=json.dumps(value,indent=2,sort_keys=True)+'\n'
    with pending.open('x',encoding='utf-8') as stream:
        stream.write(raw)
        stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def classify_surface(vertices, faces):
    """Check a surface without adopting a cached source-topology declaration."""
    require(len(vertices) >= 3 and len(faces) > 0 and all(len(p) == 3 for p in vertices),
            'empty or malformed compiled surface')
    require(all(len(f) == 3 and all(type(i) is int and 0 <= i < len(vertices) for i in f)
                for f in faces), 'invalid compiled face indices')
    require(all(all(type(x) in (int, float) and math.isfinite(x) for x in p)
                for p in vertices), 'nonfinite compiled vertex')
    topology = repair.compact_topology(analyze_topology(vertices, faces))
    try:
        exact = audit.exact_embedding(vertices, faces)
    except human.ImportError as error:
        require(str(error) == 'cardiac cavity intersections: exactly degenerate triangle',
                'unexpected exact-predicate failure: ' + str(error))
        return {'status':'exact_degenerate_face','topology':topology,
                'self_intersection':'not_checked_exact_degenerate_face',
                'aabb_candidate_pairs':None,'exact_intersection_pairs':None,
                'first_intersecting_face_pairs':[],
                'closed_embedded_surface_candidate':False}
    require(exact['topology'] == topology, 'independent topology replay disagrees')
    pairs = exact['count'];closed = topology['closed_oriented_manifold_candidate']
    status = ('closed_embedded_source_candidate' if closed and pairs == 0 else
              'closed_self_intersecting' if closed else
              'open_or_nonmanifold_intersecting' if pairs else 'open_or_nonmanifold')
    return {'status':status,'topology':topology,
            'self_intersection':'exact_checked','aabb_candidate_pairs':exact['aabb_candidate_pairs'],
            'exact_intersection_pairs':pairs,'allowed_shared_vertex_or_edge_pairs':exact['allowed_shared_vertex_or_edge_pairs'],
            'first_intersecting_face_pairs':exact['triangle_pairs'][:16],
            'intersection_pair_list_sha256':hashlib.sha256(json.dumps(exact['triangle_pairs'],separators=(',',':')).encode()).hexdigest(),
            'closed_embedded_surface_candidate':status=='closed_embedded_source_candidate'}


def scan(payload: Path, output: Path) -> dict:
    import numpy as np
    payload = payload.resolve()
    output = output.resolve()
    require(human.sha256(payload) == repair.BASE_SHA, 'published 579-surface payload identity')
    specs,manifest_hashes = repair.source_specs(payload)
    _,header,records,vertices,indices = lung.decode(payload)
    require(header[:2] == (5,579) and len(records) == 579, 'whole-source ABI/record identity')
    code={name:human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}
    identity={'schema':SCHEMA,'payload_sha256':repair.BASE_SHA,
              'source_manifest_hashes':manifest_hashes,'predicate_source_sha256':code,
              'python':__import__('sys').version.split()[0],
              'numpy':np.__version__,'surface_count':579,
              'boundary':'Exact individual compiled Float32 source-surface embeddedness only; no cross-surface overlap, containment, clinical anatomy, material volume or mechanics admission.'}
    output.mkdir(parents=True,exist_ok=True)
    identity_path=output/'identity.json'
    if identity_path.exists():
        require(human.read_json(identity_path) == identity, 'resume identity changed')
    else:
        atomic_json(identity_path,identity)
    rows_dir=output/'rows';rows_dir.mkdir(exist_ok=True)
    start=time.monotonic();new=0
    for spec,record in zip(specs,records,strict=True):
        owner,fv,nv,fi,ni,stable,layer,reserved=map(int,record)
        require(stable==spec['stable_id'] and owner==spec['core_body_index']
                and layer==spec['layer_code'] and reserved==0 and ni%3==0,
                'source record/semantic identity')
        require(0<=fv<fv+nv<=len(vertices) and 0<=fi<fi+ni<=len(indices),
                'source geometry range')
        v=vertices[fv:fv+nv,:3]
        global_faces=indices[fi:fi+ni]
        require(bool(np.isfinite(v).all()), 'nonfinite compiled positions')
        require(bool(((global_faces>=fv)&(global_faces<fv+nv)).all()),
                'source indices escape their record')
        mesh_hash=geometry_hash(vertices[fv:fv+nv],global_faces)
        row_path=rows_dir/f'{stable:03}.json'
        if row_path.exists():
            row=human.read_json(row_path)
            require(row['schema']==ROW_SCHEMA and row['stable_id']==stable
                    and row['geometry_sha256']==mesh_hash and row['predicate_source_sha256']==code,
                    'resume row provenance changed')
            continue
        f=(global_faces-fv).reshape(-1,3).tolist()
        t=time.monotonic()
        checked=classify_surface(v.astype(float).tolist(),f)
        row={'schema':ROW_SCHEMA,'stable_id':stable,'provider':spec['provider'],
             'source_member_id':spec.get('member_id'),
             'source_object_name':spec.get('object_name'),
             'source_body_index':owner,'source_layer_code':layer,
             'vertex_count':nv,'triangle_count':ni//3,
             'geometry_sha256':mesh_hash,'predicate_source_sha256':code,
             'wall_seconds':time.monotonic()-t,**checked,
             'physical_volume':False,'clinical_anatomy':False,'mechanics':False}
        atomic_json(row_path,row)
        new+=1
        print(json.dumps({'stable_id':stable,'faces':ni//3,'status':checked['status'],
                          'intersections':checked['exact_intersection_pairs'],
                          'seconds':row['wall_seconds']}),flush=True)
    paths=sorted(rows_dir.glob('*.json'))
    require(len(paths)==579 and [p.stem for p in paths]==[f'{i:03}' for i in range(1,580)],
            'incomplete or unexpected source rows')
    rows=[human.read_json(p) for p in paths]
    require({name:human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}==code,
            'predicate source changed during execution')
    counts=dict(sorted(Counter(row['status'] for row in rows).items()))
    result={'schema':SCHEMA,'identity_sha256':human.sha256(identity_path),
            'payload_sha256':repair.BASE_SHA,'surface_count':579,
            'triangles_audited':sum(r['triangle_count'] for r in rows),
            'exact_predicate_rows':sum(r['self_intersection']=='exact_checked' for r in rows),
            'status_counts':counts,'closed_embedded_surface_candidate_count':counts.get('closed_embedded_source_candidate',0),
            'total_exact_intersection_pairs':sum(r['exact_intersection_pairs'] or 0 for r in rows),
            'unqualified_source_ids':[r['stable_id'] for r in rows if not r['closed_embedded_surface_candidate']],
            'row_files':{p.name:human.sha256(p) for p in paths},
            'new_rows_this_execution':new,'wall_seconds_this_execution':time.monotonic()-start,
            'cross_surface_disjointness':False,'component_containment':False,
            'physical_volume':False,'clinical_anatomy':False,'mechanics':False,
            'boundary':'Source-local compiled FP32 triangle embeddedness only. Source members may overlap by hierarchy or pathology. No tissue or organ volume, clinical placement, interdomain disjointness, or physical ownership is inferred.'}
    atomic_json(output/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payload',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=scan(args.payload,args.output)
    print(json.dumps({'status_counts':result['status_counts'],'triangles_audited':result['triangles_audited'],
                      'exact_predicate_rows':result['exact_predicate_rows'],
                      'new_rows_this_execution':result['new_rows_this_execution']}),flush=True)


if __name__=='__main__':main()
