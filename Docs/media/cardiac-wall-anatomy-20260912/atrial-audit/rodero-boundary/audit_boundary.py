#!/usr/bin/env python3
"""Independent source-index boundary topology audit; no repair or stepping."""
import argparse
from array import array
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--asset',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
started=time.monotonic()
manifest_bytes=(a.asset/'manifest.json').read_bytes()
manifest=json.loads(manifest_bytes)
assert manifest['schema']=='HumanPack.cardiac-wall-source-asset.v1'

def read(name,code):
    data=(a.asset/name).read_bytes()
    expected=manifest['buffers'][name]
    assert len(data)==expected['bytes']
    assert hashlib.sha256(data).hexdigest()==expected['sha256']
    result=array(code)
    result.frombytes(data)
    if sys.byteorder!='little':result.byteswap()
    return result

nodes=read('nodes.f64le','d')
boundary=read('boundary.u32le','I')
owners=read('boundary_owners.u32le','I')
components=read('boundary_components.u32le','I')
labels=read('labels.u32le','I')
tets=read('tetrahedra.u32le','I')
assert len(boundary)==3*len(owners)==3*len(components)
assert len(nodes)%3==0 and len(tets)==4*len(labels)
faces=[tuple(boundary[3*i:3*i+3]) for i in range(len(owners))]
assert len(set(tuple(sorted(f)) for f in faces))==len(faces)

FACE_NODES=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
def cycle(f):
    j=f.index(min(f));return f[j:]+f[:j]

def det(a,b,c,d):
    u=tuple(b[k]-a[k] for k in range(3));v=tuple(c[k]-a[k] for k in range(3));w=tuple(d[k]-a[k] for k in range(3))
    return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])

denominator=max(x.as_integer_ratio()[1] for x in nodes)
points=[tuple(n*(denominator//d) for n,d in (x.as_integer_ratio() for x in nodes[i:i+3])) for i in range(0,len(nodes),3)]
global_incident=defaultdict(list)
groups=defaultdict(list)
for i,f in enumerate(faces):
    assert len(set(f))==3 and min(f)>=0 and max(f)<len(points)
    tet=tuple(tets[4*owners[i]:4*owners[i]+4])
    assert len(tet)==4
    assert cycle(f) in {cycle(tuple(tet[j] for j in order)) for order in FACE_NODES}
    groups[components[i]].append(i)
    for n in f:global_incident[n].append(i)

def edge_details(edge,rows):
    return {'nodes':list(edge),'positions_m':[list(nodes[3*i:3*i+3]) for i in edge],
        'incidence_count':len(rows),'oriented_incidence':[
            {'boundary_face':i,'nodes':list(faces[i]),'source_tetrahedron':owners[i],
             'source_label':labels[owners[i]],'ascending_edge':orientation,'component':components[i]}
            for i,orientation in rows]}

def link_defect(node,ids):
    graph=defaultdict(set);counts=Counter()
    for i in ids:
        others=[n for n in faces[i] if n!=node]
        assert len(others)==2
        x,y=others; graph[x].add(y);graph[y].add(x);counts[tuple(sorted((x,y)))]+=1
    unseen=set(graph);parts=[]
    while unseen:
        todo=[min(unseen)];part=[]
        while todo:
            x=todo.pop()
            if x not in unseen:continue
            unseen.remove(x);part.append(x);todo.extend(graph[x]&unseen)
        parts.append(sorted(part))
    wrong={str(n):len(neighbours) for n,neighbours in graph.items() if len(neighbours)!=2}
    repeated=[list(edge) for edge,count in counts.items() if count!=1]
    if len(parts)==1 and not wrong and not repeated:return None
    return {'node':node,'position_m':list(nodes[3*node:3*node+3]),'link_component_count':len(parts),
            'link_components':parts,'noncycle_degree_by_node':wrong,'duplicate_link_edges':repeated,
            'boundary_faces':ids,'source_tetrahedra':[owners[i] for i in ids],
            'source_labels':dict(sorted(Counter(str(labels[owners[i]]) for i in ids).items())),
            'boundary_components':sorted(set(components[i] for i in ids))}

reports=[]
for component,ids in sorted(groups.items()):
    edges=defaultdict(list);incident=defaultdict(list)
    for i in ids:
        f=faces[i]
        for x,y in zip(f,f[1:]+f[:1]):edges[min(x,y),max(x,y)].append((i,x<y))
        for n in f:incident[n].append(i)
    defects=[edge_details(edge,rows) for edge,rows in sorted(edges.items()) if len(rows)!=2 or rows[0][1]==rows[1][1]]
    vertex_defects=[d for n,local in sorted(incident.items()) if (d:=link_defect(n,local)) is not None]
    parent={i:i for i in ids}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for rows in edges.values():
        for row in rows[1:]:parent[root(row[0])]=root(rows[0][0])
    connected_components=len({root(i) for i in ids})
    assert connected_components==1, 'published boundary component is not edge-connected'
    origin=points[min(incident)]
    volume_numerator=sum(det(origin,*(points[n] for n in faces[i])) for i in ids)
    euler=len(incident)-len(edges)+len(ids)
    closed=not defects and not vertex_defects and connected_components==1
    report={'component':component,'face_count':len(ids),'node_count':len(incident),'edge_count':len(edges),
        'euler':euler,'genus_if_closed_oriented_manifold':(2-euler)//2 if closed else None,
        'independently_recomputed_edge_connected_components':connected_components,
        'edge_incidence_histogram':dict(sorted(Counter(str(len(x)) for x in edges.values()).items())),
        'edge_defects':defects,'vertex_link_defects':vertex_defects,'edge_defect_count':len(defects),
        'vertex_link_defect_count':len(vertex_defects),'closed_oriented_vertex_manifold':closed,
        'source_label_face_counts':dict(sorted(Counter(str(labels[owners[i]]) for i in ids).items())),
        'signed_material_outward_volume_m3':volume_numerator/(6*denominator**3),
        'exact_volume_numerator_hex':hex(volume_numerator),'exact_volume_denominator_hex':hex(6*denominator**3),
        'embeddedness':'not_checked','native_cavity_admitted':False}
    reports.append(report)
    print(component,'faces',len(ids),'edge defects',len(defects),'vertex defects',len(vertex_defects),'euler',euler,flush=True)

global_vertex_defects=[d for n,ids in sorted(global_incident.items()) if (d:=link_defect(n,ids)) is not None]
cross_component_vertices=[{'node':n,'components':sorted(set(components[i] for i in ids))}
    for n,ids in sorted(global_incident.items()) if len(set(components[i] for i in ids))>1]
report={'schema':'NumiHuman.Rodero18BoundaryTopologyAudit.v1','qualification':'offline_source_boundary_topology_only',
    'asset_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
    'source_config_sha256':manifest['source_config_sha256'],
    'source_archive_sha256':manifest['source_config']['source']['archive']['sha256'],
    'input_buffers':{name:manifest['buffers'][name] for name in ('nodes.f64le','boundary.u32le','boundary_owners.u32le','boundary_components.u32le','labels.u32le','tetrahedra.u32le')},
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'geometry_edits':False,'source_faces_added_or_removed':False,'physical_stepping':False,
    'all_face_source_owner_orientations_match':True,'components':reports,
    'total_edge_defects':sum(r['edge_defect_count'] for r in reports),
    'global_vertex_link_defects':global_vertex_defects,
    'cross_component_shared_vertices':cross_component_vertices,
    'component3_identity':'unresolved in geometry; source labels 8/10 suggest RV under the documented source port interpretation, while tissue includes tags 1 and 2',
    'elapsed_seconds':time.monotonic()-started}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print('global vertex defects',len(global_vertex_defects),'shared component vertices',len(cross_component_vertices),'seconds',report['elapsed_seconds'],flush=True)
