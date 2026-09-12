#!/usr/bin/env python3
"""Exact RA/LA boundary embedding audit, preserving source cell and face IDs."""
import argparse
from array import array
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--asset',type=Path,required=True)
p.add_argument('--owners-root',type=Path,required=True)
p.add_argument('--topology-report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.owners_root/'src'))
from numilab_human import cardiac_cavity_intersections as exact

def sha(data):return hashlib.sha256(data).hexdigest()
started=time.monotonic()
manifest_bytes=(a.asset/'manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
topology_bytes=a.topology_report.read_bytes();topology=json.loads(topology_bytes)
assert topology['asset_manifest_sha256']==sha(manifest_bytes)

def read(name,code):
    data=(a.asset/name).read_bytes();expected=manifest['buffers'][name]
    assert len(data)==expected['bytes'] and sha(data)==expected['sha256']
    result=array(code);result.frombytes(data)
    if sys.byteorder!='little':result.byteswap()
    return result

nodes=read('nodes.f64le','d');boundary=read('boundary.u32le','I')
owners=read('boundary_owners.u32le','I');components=read('boundary_components.u32le','I')
labels=read('labels.u32le','I');tets=read('tetrahedra.u32le','I')
denominator=max(x.as_integer_ratio()[1] for x in nodes)
points=[tuple(n*(denominator//d) for n,d in (x.as_integer_ratio() for x in nodes[i:i+3])) for i in range(0,len(nodes),3)]

def tree(records):
    lo=tuple(min(r[1][k] for r in records) for k in range(3));hi=tuple(max(r[2][k] for r in records) for k in range(3))
    if len(records)<=8:return lo,hi,records,None
    axis=max(range(3),key=lambda k:hi[k]-lo[k]);ordered=sorted(records,key=lambda r:(r[1][axis]+r[2][axis],r[3]));mid=len(ordered)//2
    return lo,hi,tree(ordered[:mid]),tree(ordered[mid:])

def query(node,lower,upper):
    lo,hi,left,right=node
    if any(upper[k]<lo[k] or hi[k]<lower[k] for k in range(3)):return
    if right is None:
        for row in left:
            if not any(upper[k]<row[1][k] or row[2][k]<lower[k] for k in range(3)):yield row
    else:
        yield from query(left,lower,upper);yield from query(right,lower,upper)

def audit(first,second,same_surface):
    begin=time.monotonic();index=tree(second);candidates=allowed=0;pairs=[]
    for tri,lo,hi,i,ids in first:
        for other,lower,upper,j,other_ids in query(index,lo,hi):
            if same_surface and j<=i:continue
            candidates+=1;hits=exact.triangle_intersection_points(tri,other)
            if not hits:continue
            shared=set(ids)&set(other_ids) if same_surface else set()
            common={tri[ids.index(k)] for k in shared}
            if same_surface and len(shared) in (1,2) and all(exact._allowed_shared_point(x,common) for x in hits):allowed+=1
            else:pairs.append([i,j])
    return {'triangle_pairs':sorted(pairs),'count':len(pairs),'aabb_candidate_pairs':candidates,
            'allowed_shared_vertex_or_edge_pairs':allowed,'elapsed_seconds':time.monotonic()-begin}

records={};surfaces={}
source_roles={row['id']:row for row in manifest['source_config']['labels']}
for component,name in ((1,'left_atrium'),(4,'right_atrium')):
    check=next(c for c in topology['components'] if c['component']==component)
    assert check['closed_oriented_vertex_manifold'] and check['independently_recomputed_edge_connected_components']==1
    ids=[i for i,c in enumerate(components) if c==component]
    faces=[tuple(boundary[3*i:3*i+3]) for i in ids]
    raw=exact._records(points,faces)
    r=[(tri,lo,hi,ids[i],source_nodes) for tri,lo,hi,i,source_nodes in raw]
    records[name]=r
    print('exact embeddedness',name,len(r),flush=True)
    sample=r[::max(1,len(r)//256)]
    direct=exact._audit_pair(sample,sample,same_surface=True);fast=audit(sample,sample,True)
    assert all(direct[k]==fast[k] for k in direct)
    result=audit(r,r,True)
    role_faces={};wrong_sides=[]
    for i,f in zip(ids,faces):
        role=source_roles[labels[owners[i]]]['role'];role_faces.setdefault(role,[]).append(i)
        tet=tets[4*owners[i]:4*owners[i]+4]
        opposite=set(tet)-set(f)
        assert len(opposite)==1
        x,y,z=(points[n] for n in f)
        normal=exact._cross(exact._sub(y,x),exact._sub(z,x))
        sign=exact._dot(normal,exact._sub(points[next(iter(opposite))],x))
        if sign>=0:wrong_sides.append(i)
    assert not wrong_sides
    result.update(component=component,source_boundary_face_ids=ids,
        closed_connected_oriented_vertex_manifold=True,
        embedded_closed_surface=result['count']==0,
        orientation='original material-outward; reverse face orientation for a positive lumen boundary',
        all_source_owner_opposite_vertices_on_material_side=True,
        source_owner_adjacency_scope='exact local oriented halfspace only; excludes no remote crossing tetrahedra',
        source_label_face_counts=dict(sorted(Counter(str(labels[owners[i]]) for i in ids).items())),
        face_ids_by_source_role=role_faces,
        source_cap_role_provenance='role mapping copied from pinned source configuration; no caps added or relabeled',
        source_oriented_volume_m3=check['signed_material_outward_volume_m3'],
        positive_lumen_reference_volume_m3=-check['signed_material_outward_volume_m3'],
        source_roundtrip_node_and_tetrahedron_indices_retained=True,
        native_pressure_boundary_admitted=False,
        broadphase_direct_owner_crosscheck={'sample_face_count':len(sample),'equal':True})
    surfaces[name]=result
    print(name,'self pairs',result['count'],'seconds',result['elapsed_seconds'],flush=True)

names=sorted(records);first,second=names
pair=audit(records[first],records[second],False)
pair.update(first=first,second=second)
if not pair['count'] and all(s['embedded_closed_surface'] for s in surfaces.values()):
    pair['containment']={'first_in_second':exact.point_location(records[first][0][0][0],records[second]),
                         'second_in_first':exact.point_location(records[second][0][0][0],records[first]),
                         'witness_source_nodes':[records[first][0][4][0],records[second][0][4][0]]}
    pair['disjoint_closed_domains']=all(pair['containment'][k]['location']=='outside' for k in ('first_in_second','second_in_first'))
else:
    pair['containment']={'status':'not_checked_intersections_or_invalid_surface'}
    pair['disjoint_closed_domains']=False

result={'schema':'NumiHuman.Rodero18AtrialBoundaryEmbeddingAudit.v1',
        'qualification':'source_conforming_atrial_geometry_only',
        'asset_manifest_sha256':sha(manifest_bytes),'topology_report_sha256':sha(topology_bytes),
        'source_config_sha256':manifest['source_config_sha256'],
        'input_buffers':{k:manifest['buffers'][k] for k in ('nodes.f64le','boundary.u32le','boundary_owners.u32le','boundary_components.u32le','labels.u32le','tetrahedra.u32le')},
        'script_sha256':sha(Path(__file__).read_bytes()),
        'predicate_owner':{'path':'src/numilab_human/cardiac_cavity_intersections.py','sha256':sha(Path(exact.__file__).read_bytes()),'algorithm':exact.ALGORITHM},
        'coordinate_semantics':'exact rational values of source asset binary64 metres',
        'source_coordinates_modified':False,'source_faces_added_or_removed':False,'physical_stepping':False,
        'surfaces':surfaces,'pair':pair,'global_wall_tetrahedral_embedding':'not_checked',
        'global_wall_excluded_from_atrial_interiors':'not_checked',
        'physiological_or_mechanical_calibration':False,'elapsed_seconds':time.monotonic()-started}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print('RA/LA pair',pair['count'],'disjoint',pair['disjoint_closed_domains'],'seconds',result['elapsed_seconds'],flush=True)
