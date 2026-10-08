"""Compile registered whole-body anatomy into the existing NHANAT1 native ABI.

This is a source-bound preparation step, not a tissue or fluid solver. Atlas
lung boundary loops are closed explicitly for the reduced gas-envelope model;
the generated receipt records every such repair and its source lineage.
"""
from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

from . import model as human
from . import lung_envelope as lung
from . import surface_topology_audit
from .cardiac_cavity_geometry import CAVITIES, analyze_topology, exact_coordinate_quotient, parse_obj


HEADER = struct.Struct('<8s5I32s')
RECORD = struct.Struct('<8I')
RIGID_PATH = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
BONE_DIR = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/current-bone-registration-b1b410ad')
REGISTRATION_PATH = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json')
SKIN_PATH = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/skin-boundaries-001/bodyparts3d-myosim-skinned-shell.nhskin')
SKIN_SOURCE_PATH = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-weight-heldout-20261004/inputs/base/bodyparts3d-myosim-skinned-shell.nhskin')
SOURCES = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/anatomy-complete-20261005/inputs/Sources')
NHTISS_PATH = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/current-anatomy-20261005/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
NHTISS_MANIFEST_PATH = NHTISS_PATH.with_suffix('.manifest.json')
NHTISS_SHA256 = '9340447e7874366033f544ee8e56e9c09bdf4efecf52b0fa3b16d7b4d5489c33'
NHTISS_MANIFEST_SHA256 = 'a254c4cd76927d5ed82348cba2a85a952df67f40cbf042ac86f5a49dcc925bbe'
TORSO_BODY_INDEX = 20
REFERENCE_TOTAL_MASS_KG = 72.0
TORSO_POSITION_WORLD_M = np.asarray([-0.02500000037, 0.19939999282, 1.35720002651], dtype=np.float64)
TORSO_QUATERNION_XYZW = np.asarray([0.5, -0.5, -0.5, 0.5], dtype=np.float64)
LUNG_SPECS = (
    (305, 'Inferior lobe of left lung'),
    (306, 'Inferior lobe of right lung'),
    (307, 'Middle lobe of right lung'),
    (308, 'Superior lobe of left lung'),
    (309, 'Superior lobe of right lung'),
)
MUSCLE_SPECS = (
    # NHANAT ABI5 reserves 10/11 for duct/neural-region semantics. These are
    # source muscle surfaces; the viewer promotes the exact stable IDs into
    # its muscle layer using functional_bindings.
    (311, 1, 'diaphragm', 'FJ3131'),
    (312, 1, 'external intercostal muscle', 'FJ1451'),
    (313, 1, 'external intercostal muscle', 'FJ1451M'),
    (314, 1, 'internal intercostal muscle', 'FJ1455'),
    (315, 1, 'internal intercostal muscle', 'FJ1455M'),
    (316, 1, 'innermost intercostal muscle', 'FJ1454'),
    (317, 1, 'innermost intercostal muscle', 'FJ1454M'),
)
CAVITY_STABLE_IDS = (318, 319, 320, 321)
CVSIM_CHAMBER_BINDINGS = (
    {'name':'right_atrium','source_member':'FJ2424','fma_id':'FMA11359','stable_id':318,'cvsim_compartment_id':16},
    {'name':'right_ventricle','source_member':'FJ2423','fma_id':'FMA9291','stable_id':319,'cvsim_compartment_id':17},
    {'name':'left_atrium','source_member':'FJ2425','fma_id':'FMA9465','stable_id':320,'cvsim_compartment_id':20},
    {'name':'left_ventricle','source_member':'FJ2422','fma_id':'FMA9466','stable_id':321,'cvsim_compartment_id':21},
)
RIB_MEMBERS = (
    'FJ3334','FJ3336','FJ3338','FJ3340','FJ3342','FJ3344','FJ3346','FJ3347','FJ3348','FJ3330','FJ3331','FJ3332',
    'FJ3228','FJ3229','FJ3230','FJ3231','FJ3232','FJ3233','FJ3234','FJ3235','FJ3236','FJ3225','FJ3226','FJ3227',
)
STERNUM_MEMBERS = ('FJ3178', 'FJ3290')
LUNG_306_SELF_PATCH_OUTPUT_FACE_IDS = (
    4494, 4496, 7108, 7109, 7116, 7640, 7641, 7643, 7644,
    7645, 7646, 7647, 7696, 7697, 7698, 7699, 7700, 7748,
)
LUNG_306_EXPECTED_SELF_PAIRS = (
    (7109, 7647), (7109, 7696), (7640, 7647), (7640, 7696),
    (7641, 7645), (7641, 7696), (7641, 7698), (7641, 7699),
)


def _mass_and_skin_volume_audit(*, skin_path: Path = SKIN_PATH, skin_rebind_manifest_path: Path | None = None) -> dict:
    """Bind reference mass and skin closure diagnostics to exact NHSKIN inputs."""
    from .resting_scene import load_rigid, load_skin, source_shell_world

    skin_path = Path(skin_path).resolve()
    rigid = load_rigid(RIGID_PATH)
    skin = load_skin(skin_path, rigid)
    rebind_manifest = None
    if skin_rebind_manifest_path is None:
        repair_path = SKIN_PATH.with_suffix('.boundary-repair.json')
        repair = json.loads(repair_path.read_text())
        if (repair.get('source_skin', {}).get('path') != str(SKIN_SOURCE_PATH)
                or repair.get('source_skin', {}).get('sha256') != hashlib.sha256(SKIN_SOURCE_PATH.read_bytes()).hexdigest()
                or repair.get('derived_skin', {}).get('path') != str(SKIN_PATH.resolve())
                or repair.get('derived_skin', {}).get('sha256') != skin['sha256']):
            raise ValueError('derived NHSKIN repair receipt does not bind its exact source and payload')
        source_path = SKIN_SOURCE_PATH
        source_sha = hashlib.sha256(SKIN_SOURCE_PATH.read_bytes()).hexdigest()
    else:
        skin_rebind_manifest_path = Path(skin_rebind_manifest_path).resolve()
        rebind_manifest = json.loads(skin_rebind_manifest_path.read_text())
        if rebind_manifest.get('schema') != 'numi.human.skin-lower-limb-anchor-rebind-candidate.v1':
            raise ValueError('unsupported NHSKIN binding-rebind manifest')
        emitted = rebind_manifest.get('output_payload', {})
        if (Path(emitted.get('path', '')).resolve() != skin_path
                or emitted.get('sha256') != skin['sha256']
                or emitted.get('bytes') != len(skin['raw'])):
            raise ValueError('NHSKIN binding-rebind manifest does not bind the candidate payload')
        inputs = rebind_manifest.get('inputs', {})
        source = inputs.get('source_payload', {})
        upstream = inputs.get('upstream_skin_provenance', {})
        if not isinstance(source.get('path'), str) or not isinstance(source.get('sha256'), str):
            raise ValueError('NHSKIN binding-rebind manifest lacks its source payload identity')
        repair_path = Path(upstream.get('path', '')).resolve()
        if (not repair_path.is_file()
                or hashlib.sha256(repair_path.read_bytes()).hexdigest() != upstream.get('sha256')):
            raise ValueError('NHSKIN binding-rebind upstream repair receipt identity differs')
        repair = json.loads(repair_path.read_text())
        derived = repair.get('derived_skin', {})
        if (derived.get('path') != source['path'] or derived.get('sha256') != source['sha256']
                or not Path(source['path']).is_file()
                or hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']):
            raise ValueError('NHSKIN binding-rebind source differs from its upstream repair receipt')
        source_record = repair.get('source_skin', {})
        source_path = Path(source_record.get('path', '')).resolve()
        source_sha = source_record.get('sha256')
        if (not source_path.is_file() or not isinstance(source_sha, str)
                or hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha):
            raise ValueError('NHSKIN binding-rebind original source identity differs')
        runtime = inputs.get('runtime_reference', {})
        rigid_record = runtime.get('rigid', {})
        if (rigid_record.get('sha256') != rigid['sha256']
                or Path(rigid_record.get('file', '')).resolve() != RIGID_PATH.resolve()):
            raise ValueError('NHSKIN binding-rebind rigid runtime identity differs')
        registration_sha = inputs.get('registration_sha256')
        if registration_sha != hashlib.sha256(REGISTRATION_PATH.read_bytes()).hexdigest():
            raise ValueError('NHSKIN binding-rebind registration identity differs')
        if rebind_manifest.get('payload_identity', {}).get('source_archive_sha256') != rigid['source_sha256']:
            raise ValueError('NHSKIN binding-rebind source archive differs from NHRIGID2')
    world_vertices, _, _ = source_shell_world(rigid, skin)
    raw = skin['raw']
    _, _, binding_count, vertex_count, index_count, _, _ = struct.unpack_from(
        '<8s5I32s', raw, 0,
    )
    index_offset = 60 + 36 * binding_count + 56 * vertex_count
    triangles = np.frombuffer(raw, '<u4', count=index_count, offset=index_offset).reshape(-1, 3)

    # Do not turn an open-surface divergence integral into a geometric volume.
    directed = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    undirected = np.sort(directed, axis=1)
    _, edge_inverse, edge_counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True,
    )
    orientation_sum = np.bincount(
        edge_inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1),
        minlength=len(edge_counts),
    )
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
    orientation_errors = int(np.count_nonzero((edge_counts == 2) & (orientation_sum != 0)))
    boundary_rows = undirected[edge_counts[edge_inverse] == 1]
    boundary_adjacency: dict[int, set[int]] = {}
    for first, second in boundary_rows:
        a, b = int(first), int(second)
        boundary_adjacency.setdefault(a, set()).add(b)
        boundary_adjacency.setdefault(b, set()).add(a)
    boundary_components = 0
    boundary_unseen = set(boundary_adjacency)
    while boundary_unseen:
        boundary_components += 1
        stack = [boundary_unseen.pop()]
        while stack:
            stack.extend(boundary_adjacency[stack.pop()] & boundary_unseen)
            for neighbor in stack:
                boundary_unseen.discard(neighbor)
    boundary_branches = sum(len(neighbors) != 2 for neighbors in boundary_adjacency.values())
    signed_surface_integral = _signed_volume(world_vertices, triangles.astype(np.int64))
    closed_volume = (abs(signed_surface_integral)
                     if not (boundary_edges or nonmanifold_edges or orientation_errors) else None)
    source_mass = float(np.sum(rigid['masses']))
    if not np.isfinite(source_mass) or source_mass <= 0:
        raise ValueError('source rigid mass is non-positive/non-finite')
    return {
        'source_rigid_mass_kg': source_mass,
        'reference_total_mass_kg': REFERENCE_TOTAL_MASS_KG,
        'uniform_mass_and_inertia_scale': REFERENCE_TOTAL_MASS_KG / source_mass,
        'source_rigid_body_count': int(rigid['body_count']),
        'source_dynamic_body_count': int(np.count_nonzero(rigid['masses'] > 0.0)),
        'closed_skin_volume_m3': closed_volume,
        'open_surface_signed_integral_m3_diagnostic_only': signed_surface_integral,
        'apparent_density_kg_m3': (
            REFERENCE_TOTAL_MASS_KG / closed_volume if closed_volume is not None else None
        ),
        'skin_geometry_status': (
            'closed_oriented_2_manifold_volume_candidate'
            if closed_volume is not None else 'open_surface_volume_rejected'
        ),
        'skin_boundary_edge_count': boundary_edges,
        'skin_boundary_loop_count': boundary_components,
        'skin_boundary_branch_vertex_count': boundary_branches,
        'skin_nonmanifold_edge_count': nonmanifold_edges,
        'skin_orientation_error_count': orientation_errors,
        'skin_self_intersection': 'not assessed by this closure/volume check',
        'skin_payload_path': str(skin_path),
        'skin_payload_sha256': skin['sha256'],
        'skin_source_path': str(source_path),
        'skin_source_sha256': source_sha,
        'skin_boundary_repair_receipt':str(repair_path),
        'skin_boundary_repair_receipt_sha256':hashlib.sha256(repair_path.read_bytes()).hexdigest(),
        'skin_boundary_repair':{'source_triangle_count':repair['source_triangle_count'],
            'added_triangle_count':repair['added_triangle_count'],
            'exact_coordinate_weight_identical_seam_vertices':repair['exact_coordinate_weight_identical_seam_vertices'],
            'boundary_edges_after_exact_weld':repair['boundary_edges_after_exact_weld'],
            'boundary_edges_after_small_caps':repair['boundary_edges_after_small_caps'],
            'coordinates_normals_weights_vertex_ids_unchanged':repair['coordinates_normals_weights_vertex_ids_unchanged'],
            'retained_anatomical_interfaces':repair['retained_anatomical_interfaces'],
            'qualification':repair['qualification']},
        'basis': (
            '72 kg is an explicitly declared generic adult reference mass, not a measured '
            'participant. The source rigid mass/inertia distribution is uniformly rebased. '
            'The current visual shell has open boundaries, so its signed surface integral '
            'is not admitted as a geometric volume or density.'
        ),
    }


def _qrotation(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    p = vertices[faces].astype(np.float64)
    return float(np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()/6.0)


def _split_disconnected_vertex_fans(vertices: list, faces: list, defect_ids: list[int]) -> tuple[list, list, list]:
    """Separate only source vertices whose incident face link is disconnected."""
    out_v = list(vertices)
    out_f = [list(f) for f in faces]
    repairs = []
    for vertex in defect_ids:
        incident = [i for i, f in enumerate(out_f) if vertex in f]
        adjacency = {i: set() for i in incident}
        for a_pos, a in enumerate(incident):
            edges_a = {tuple(sorted((vertex, x))) for x in out_f[a] if x != vertex}
            for b in incident[a_pos+1:]:
                edges_b = {tuple(sorted((vertex, x))) for x in out_f[b] if x != vertex}
                if edges_a & edges_b:
                    adjacency[a].add(b); adjacency[b].add(a)
        components=[];unseen=set(incident)
        while unseen:
            stack=[min(unseen)];component=[]
            while stack:
                current=stack.pop()
                if current not in unseen:continue
                unseen.remove(current);component.append(current);stack.extend(adjacency[current]&unseen)
            components.append(sorted(component))
        if len(components) <= 1:
            continue
        new_vertices=[]
        for component in components[1:]:
            new_id=len(out_v);out_v.append(tuple(out_v[vertex]));new_vertices.append(new_id)
            for face_id in component:
                out_f[face_id]=[new_id if x==vertex else x for x in out_f[face_id]]
        repairs.append({'source_vertex_id':int(vertex),'copied_vertex_ids':new_vertices,
                        'face_fan_components':[len(c) for c in components],
                        'coordinate_displacement_m':0.0})
    return out_v,out_f,repairs


def _orient2(a, b, c):
    return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))


def _point_in_triangle(p, a, b, c, sign, tol):
    return sign*_orient2(a,b,p)>=-tol and sign*_orient2(b,c,p)>=-tol and sign*_orient2(c,a,p)>=-tol


def _cap_loop(vertices: np.ndarray, loop: list[int]) -> tuple[list[list[int]], dict]:
    """Ear-clip one source boundary ring; caps use the opposite winding."""
    if len(loop) < 3:
        raise ValueError('boundary loop has fewer than three vertices')
    p3=vertices[np.asarray(loop,dtype=np.int64)].astype(np.float64)
    center=p3.mean(axis=0);_,_,basis=np.linalg.svd(p3-center,full_matrices=False)
    uv=(p3-center)@basis[:2].T
    area=sum(uv[i,0]*uv[(i+1)%len(loop),1]-uv[(i+1)%len(loop),0]*uv[i,1] for i in range(len(loop)))*0.5
    if abs(area)<1e-14:raise ValueError('boundary ring projection has zero area')
    sign=1.0 if area>0 else -1.0
    tol=max(1e-16,float(np.ptp(uv,axis=0).max())**2*1e-12)
    remaining=list(range(len(loop)));ears=[]
    guard=0
    while len(remaining)>3:
        found=False
        for j in range(len(remaining)):
            prev=remaining[(j-1)%len(remaining)];cur=remaining[j];nxt=remaining[(j+1)%len(remaining)]
            if sign*_orient2(uv[prev],uv[cur],uv[nxt])<=tol:continue
            if any(_point_in_triangle(uv[k],uv[prev],uv[cur],uv[nxt],sign,tol)
                   for k in remaining if k not in (prev,cur,nxt)):continue
            ears.append([loop[prev],loop[nxt],loop[cur]]) # reverse loop winding for the cap
            remaining.pop(j);found=True;break
        guard+=1
        if not found or guard>len(loop)*len(loop):
            raise ValueError('boundary loop is self-intersecting or cannot be ear-clipped safely')
    a,b,c=remaining
    ears.append([loop[a],loop[c],loop[b]])
    signed_distance=(p3-center)@basis[-1]
    return ears,{'vertex_ids':list(map(int,loop)),'cap_triangles':len(ears),
        'best_fit_plane_max_deviation_m':float(np.abs(signed_distance).max()),
        'best_fit_plane_rms_deviation_m':float(np.sqrt(np.mean(signed_distance**2))),
        'cap_method':'derived_constrained_boundary_loop_ear_clipping','vertex_coordinates_moved':False}


def _close_lung(stable_id: int, name: str, vertices: list, faces: list) -> tuple[np.ndarray,np.ndarray,dict]:
    original_topology=analyze_topology(vertices,faces)
    edits={'source_topology_before':{k:(len(v) if isinstance(v,list) else v) for k,v in original_topology.items()
              if k not in ('boundary_edges','boundary_loops','face_components')},
           'removed_source_face_ids':[],'vertex_fan_splits':[],'boundary_caps':[]}
    if stable_id == 306:
        # The pinned atlas lobe has eight three-face edges from an overlapping
        # source sheet. Removing its ten connected coarse faces retains the
        # two denser source sheets; the resulting exposed rim is capped below.
        removed=list(range(3290,3300));faces=[f for i,f in enumerate(faces) if i not in set(removed)]
        edits['removed_source_face_ids']=removed
        edits['repair_reason']='ten-face overlap patch included a detached two-triangle cap artifact and created eight three-incident edges; source coordinates retained'
    topo=analyze_topology(vertices,faces)
    if topo['vertex_manifold_defect_ids']:
        vertices,faces,edits['vertex_fan_splits']=_split_disconnected_vertex_fans(vertices,faces,topo['vertex_manifold_defect_ids'])
        topo=analyze_topology(vertices,faces)
    if topo['nonmanifold_edges'] or topo['boundary_branch_vertex_ids'] or topo['orientation_defect_edges']:
        raise ValueError(f'{name}: nonmanifold source remains after the narrow local repair')
    caps=[]
    for loop in topo['boundary_loops']:
        patch,detail=_cap_loop(np.asarray(vertices,dtype=np.float64),loop)
        faces.extend(patch);caps.append(detail)
    faces_array=np.asarray(faces,dtype=np.int64)
    vertices_array=np.asarray(vertices,dtype=np.float64)
    used=sorted(set(map(int,faces_array.ravel())))
    if len(used)!=len(vertices_array):
        remap={old:new for new,old in enumerate(used)}
        edits['compacted_unused_vertex_ids']=[i for i in range(len(vertices_array)) if i not in remap]
        edits['compaction_vertex_map_old_to_new']={str(old):new for old,new in remap.items() if old!=new}
        vertices_array=vertices_array[np.asarray(used,dtype=np.int64)]
        faces_array=np.vectorize(remap.__getitem__,otypes=[np.int64])(faces_array)
    closed=analyze_topology(vertices_array.tolist(),faces_array.tolist())
    if closed['vertex_manifold_defect_ids']:
        split_v,split_f,post_cap_splits=_split_disconnected_vertex_fans(
            vertices_array.tolist(),faces_array.tolist(),closed['vertex_manifold_defect_ids'])
        vertices_array=np.asarray(split_v,dtype=np.float64)
        faces_array=np.asarray(split_f,dtype=np.int64)
        edits['post_cap_vertex_fan_splits']=post_cap_splits
        closed=analyze_topology(vertices_array.tolist(),faces_array.tolist())
    if (closed['boundary_edges'] or closed['nonmanifold_edges'] or closed['orientation_defect_edges']
            or closed['boundary_branch_vertex_ids'] or closed['vertex_manifold_defect_ids']):
        raise ValueError(f'{name}: derived surface failed closed-manifold topology audit')
    # Normalize each disconnected shell's orientation independently.
    for comp in closed['face_components']:
        ids=np.asarray(comp,dtype=np.int64)
        volume=_signed_volume(vertices_array,faces_array[ids])
        if abs(volume)<1e-10:raise ValueError(f'{name}: closed shell has negligible enclosed volume')
        if volume<0:faces_array[ids]=faces_array[ids][:,[0,2,1]]
    edits['boundary_caps']=caps
    edits['source_topology_after']= {k:(len(v) if isinstance(v,list) else v) for k,v in analyze_topology(
        vertices_array.tolist(),faces_array.tolist()).items() if k not in ('boundary_edges','boundary_loops','face_components')}
    edits['closed_component_count']=len(closed['face_components'])
    edits['closed_component_volumes_m3']=[abs(_signed_volume(vertices_array,faces_array[np.asarray(c,dtype=np.int64)]))
                                         for c in closed['face_components']]
    edits['self_intersections']='not_assessed'
    edits['inter_lobe_intersections']='not_assessed'
    return vertices_array,faces_array,edits


def _repair_registered_lung_306_self_intersection(
    vertices: np.ndarray, faces: np.ndarray, prior_edits: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Replace one registered source patch that self-crosses in native Float32.

    The atlas surface is retained away from the explicitly mapped 18-face patch.
    A single source-bound boundary loop is ear-clipped using existing boundary
    vertices; no coordinate is moved, and unreferenced source vertices are
    removed during ordinary payload compaction.
    """
    packed_before = np.asarray(vertices, dtype='<f4')
    before = surface_topology_audit.exact_embedding(
        packed_before.astype(np.float64).tolist(), np.asarray(faces, dtype=np.int64).tolist())
    pairs = tuple(tuple(map(int, pair)) for pair in before['triangle_pairs'])
    if pairs != LUNG_306_EXPECTED_SELF_PAIRS:
        raise ValueError(
            'registered right inferior lung changed its pinned eight-pair self-intersection signature')
    if max(LUNG_306_SELF_PATCH_OUTPUT_FACE_IDS) >= len(faces):
        raise ValueError('registered lung self-repair patch exceeds the pinned surface')

    removed_output_faces = set(LUNG_306_SELF_PATCH_OUTPUT_FACE_IDS)
    retained_faces = [i for i in range(len(faces)) if i not in removed_output_faces]
    candidate_faces = np.asarray(faces[retained_faces], dtype=np.int64)
    boundary = analyze_topology(vertices.tolist(), candidate_faces.tolist())
    if (boundary['boundary_edge_count'] != 16 or len(boundary['boundary_loops']) != 1
            or boundary['boundary_branch_vertex_ids'] or boundary['nonmanifold_edges']
            or boundary['orientation_defect_edges'] or boundary['vertex_manifold_defect_ids']):
        raise ValueError('pinned right inferior lung repair no longer creates one simple 16-edge patch boundary')
    boundary_loop = boundary['boundary_loops'][0]
    cap_faces, cap_detail = _cap_loop(vertices, boundary_loop)
    joined = np.concatenate([candidate_faces, np.asarray(cap_faces, dtype=np.int64)], axis=0)

    used_vertex_ids = sorted(set(map(int, joined.ravel())))
    removed_vertex_ids = sorted(set(range(len(vertices))) - set(used_vertex_ids))
    remap = {old: new for new, old in enumerate(used_vertex_ids)}
    compact_vertices = np.asarray(vertices, dtype=np.float64)[np.asarray(used_vertex_ids, dtype=np.int64)]
    compact_faces = np.asarray([[remap[int(v)] for v in face] for face in joined], dtype=np.int64)
    topology = analyze_topology(compact_vertices.tolist(), compact_faces.tolist())
    if not topology['closed_oriented_manifold_candidate']:
        raise ValueError('right inferior lung replacement patch failed closed oriented manifold audit')
    for component in topology['face_components']:
        face_ids = np.asarray(component, dtype=np.int64)
        if _signed_volume(compact_vertices, compact_faces[face_ids]) < 0.0:
            compact_faces[face_ids] = compact_faces[face_ids][:, [0, 2, 1]]
    topology = analyze_topology(compact_vertices.tolist(), compact_faces.tolist())
    if not topology['closed_oriented_manifold_candidate']:
        raise ValueError('right inferior lung replacement patch winding is inconsistent')

    packed_after = np.asarray(compact_vertices, dtype='<f4')
    retained_packed_before = packed_before[np.asarray(used_vertex_ids, dtype=np.int64)]
    coordinate_bytes_before = retained_packed_before.tobytes()
    coordinate_bytes_after = packed_after.tobytes()
    if coordinate_bytes_before != coordinate_bytes_after:
        raise ValueError('right inferior lung repair changed one or more retained Float32 source coordinates')
    after = surface_topology_audit.exact_embedding(
        packed_after.astype(np.float64).tolist(), compact_faces.tolist())
    if after['count'] != 0:
        raise ValueError('right inferior lung replacement still self-intersects in native Float32 geometry')

    source_face_count = int(prior_edits['source_topology_before']['face_count'])
    previous_source_removals = set(map(int, prior_edits.get('removed_source_face_ids', [])))
    surviving_source_faces = [i for i in range(source_face_count) if i not in previous_source_removals]
    if any(i >= len(surviving_source_faces) for i in removed_output_faces):
        raise ValueError('self-intersection repair would remove a derived cap instead of an atlas source face')
    source_face_ids = [surviving_source_faces[i] for i in sorted(removed_output_faces)]
    prior_vertex_remap = {int(k): int(v) for k, v in prior_edits.get(
        'compaction_vertex_map_old_to_new', {}).items()}
    source_vertex_for_output = {new: old for old, new in prior_vertex_remap.items()}
    for output_id in range(len(vertices)):
        source_vertex_for_output.setdefault(output_id, output_id)
    replacement_source_triangles = [
        [source_vertex_for_output[int(v)] for v in face]
        for face in cap_faces
    ]
    removed_source_vertices = [source_vertex_for_output[v] for v in removed_vertex_ids]
    area_before = float(np.linalg.norm(np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],), axis=1).sum() * 0.5)
    area_after = float(np.linalg.norm(np.cross(
        compact_vertices[compact_faces[:, 1]] - compact_vertices[compact_faces[:, 0]],
        compact_vertices[compact_faces[:, 2]] - compact_vertices[compact_faces[:, 0]],), axis=1).sum() * 0.5)
    repair = {
        'method': 'registered_local_source_patch_replacement_with_boundary_loop_ear_clipping',
        'source_member': 'Inferior lobe of right lung',
        'source_face_ids_removed': source_face_ids,
        'prepatch_output_face_ids_removed': sorted(removed_output_faces),
        'native_float32_self_intersection_pairs_before': [list(pair) for pair in pairs],
        'native_float32_self_intersection_pair_count_after': int(after['count']),
        'boundary_loop_output_vertex_ids': list(map(int, boundary_loop)),
        'replacement_triangles_source_vertex_ids': replacement_source_triangles,
        'replacement_triangle_count': len(cap_faces),
        'removed_unreferenced_source_vertex_ids': removed_source_vertices,
        'source_coordinate_sha256_retained_before': hashlib.sha256(coordinate_bytes_before).hexdigest(),
        'source_coordinate_sha256_retained_after': hashlib.sha256(coordinate_bytes_after).hexdigest(),
        'retained_source_coordinates_bitwise_unchanged': True,
        'coordinate_displacement_m': 0.0,
        'source_surface_area_before_m2': area_before,
        'source_surface_area_after_m2': area_after,
        'surface_area_relative_change': (area_after - area_before) / area_before,
        'source_envelope_volume_before_m3': abs(_signed_volume(vertices, faces)),
        'source_envelope_volume_after_m3': abs(_signed_volume(compact_vertices, compact_faces)),
        'boundary_cap': cap_detail,
        'closed_topology_after': {
            'vertex_count': topology['vertex_count'], 'face_count': topology['face_count'],
            'edge_count': topology['edge_count'], 'euler_characteristic': topology['euler_characteristic'],
            'closed_oriented_manifold_candidate': topology['closed_oriented_manifold_candidate'],
        },
        'qualification': 'exact registered native Float32 self-embedding passed after local derived patch repair',
    }
    return compact_vertices, compact_faces, repair


def _body_local(world: np.ndarray) -> np.ndarray:
    rotation=_qrotation(TORSO_QUATERNION_XYZW)
    relative=world-TORSO_POSITION_WORLD_M
    return np.sum(relative[:,None,:]*rotation.T[None,:,:],axis=2)


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n=np.zeros_like(vertices,dtype=np.float64)
    p=vertices[faces]
    fn=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    for corner in range(3):np.add.at(n,faces[:,corner],fn)
    lengths=np.linalg.norm(n,axis=1)
    bad=lengths<=1e-15
    if bad.any():n[bad]=vertices[bad]-vertices.mean(axis=0);lengths=np.linalg.norm(n,axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths<=1e-15):raise ValueError('anatomy normals are degenerate')
    return n/lengths[:,None]


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.sum(points[:,None,:]*matrix[:3,:3][None,:,:],axis=2)+matrix[:3,3]


def _source_surfaces() -> tuple[list[dict],dict]:
    from . import physiology
    config=lung.configuration();d=lung.source_data(config)
    alignment=lung.registration(d,SOURCES,config)
    registration=json.loads(REGISTRATION_PATH.read_text())
    global_matrix=np.asarray(registration['coordinate_system']['global_source_mm_to_myosim_world_m'],dtype=np.float64)
    atlas_matrix=np.asarray(alignment['source_world_m_to_bodyparts_world_m'],dtype=np.float64)
    lung_matrix=global_matrix@np.diag([1000.0,1000.0,1000.0,1.0])@atlas_matrix
    surfaces=[];repairs={}
    for stable_id,name in LUNG_SPECS:
        mesh=next(x for x in d['objects'] if x['object_name']==name)
        v,f,repair=_close_lung(stable_id,name,[tuple(x) for x in mesh['vertices_world_m']],
                               [list(map(int,x)) for x in mesh['triangles']])
        world=_apply(lung_matrix,v);local=_body_local(world)
        if stable_id == 306:
            local,f,self_repair=_repair_registered_lung_306_self_intersection(
                local,np.asarray(f,dtype=np.int64),repair)
            repair['self_intersection_repair']=self_repair
            repair['self_intersections']='exact_native_float32_checked_zero'
        surfaces.append({'id':stable_id,'layer':7,'name':name,'source':'Z-Anatomy thorax atlas','vertices':local,
            'faces':f,'source_hash':config['export']['sha256'],'source_member':name,
            'source_license':'CC-BY-SA-4.0','repair':repair})
        repairs[str(stable_id)]=repair
    pleura=next(x for x in d['objects'] if x['object_name']=='Pleura')
    pv=np.asarray(pleura['vertices_world_m'],dtype=np.float64);pf=np.asarray(pleura['triangles'],dtype=np.int64)
    surfaces.append({'id':310,'layer':8,'name':'Pleura','source':'Z-Anatomy thorax atlas','vertices':_body_local(_apply(lung_matrix,pv)),
        'faces':pf,'source_hash':config['export']['sha256'],'source_member':'Pleura',
        'source_license':'CC-BY-SA-4.0','repair':None})
    for stable_id,(name,fma,label,fj,expected_sha) in zip(CAVITY_STABLE_IDS,CAVITIES,strict=True):
        archive,member,obj=human._bodyparts_obj_member(SOURCES,'part_of',fj)
        raw_sha=hashlib.sha256(obj).hexdigest()
        if raw_sha!=expected_sha:raise ValueError(f'{name} source OBJ SHA changed')
        parsed=parse_obj(obj,member);q=exact_coordinate_quotient(parsed)
        v=np.asarray(q['vertices_m'],dtype=np.float64);f=np.asarray(q['triangles'],dtype=np.int64)
        if not analyze_topology(v.tolist(),f.tolist())['closed_oriented_manifold_candidate']:
            raise ValueError(f'{name} source cavity is not closed/oriented')
        world=_apply(global_matrix,v*1000.0)
        local=_body_local(world)
        surfaces.append({'id':stable_id,'layer':9,'name':name,'source':'BodyParts3D v4.0',
            'vertices':local,'faces':f,'source_hash':raw_sha,'source_member':fj,
            'source_license':'CC-BY-4.0','repair':{'method':'exact_decimal_coordinate_quotient',
            'source_vertices':len(parsed['vertices_mm']),'quotient_vertices':len(v),'boundary_caps':[]}})
    for stable_id,layer,label,fj in MUSCLE_SPECS:
        archive,member,obj=human._bodyparts_obj_member(SOURCES,'is_a',fj)
        vmm,f=human._bodyparts_obj_triangles(obj,member)
        v=np.asarray(vmm,dtype=np.float64)*0.001
        world=_apply(global_matrix,v*1000.0);local=_body_local(world);faces=np.asarray(f,dtype=np.int64)
        surfaces.append({'id':stable_id,'layer':layer,'name':label,'source':'BodyParts3D v4.0',
            'vertices':local,'faces':faces,'source_hash':hashlib.sha256(obj).hexdigest(),
            'source_member':fj,'source_license':'CC-BY-4.0','repair':None})
    return surfaces,{'atlas_alignment':alignment,'lung_repairs':repairs,'registration':registration}


def _registered_passive_surfaces() -> tuple[list[dict],dict]:
    """Reuse the current BodyParts3D anatomy owner for trunk and head layers.

    The existing 304-entry trunk map already has exact owner-bound geometry and
    source transforms. Its NHANAT2 output is consumed here only as an input to
    the existing NHANAT5 scene payload. Brain and eye members are additional
    exact BodyParts3D part-of representations registered to the existing head
    link; they are passive display surfaces, not neural or ocular mechanics.
    """
    registration=json.loads(REGISTRATION_PATH.read_text())
    anatomy=human.parse_bodyparts3d(SOURCES,human.REPOSITORY_ROOT/'config/anatomy-classification.v1.json')
    from .physiology import load_anatomy
    bodyparts3d=load_anatomy(SOURCES)
    with tempfile.TemporaryDirectory(prefix='resting-anatomy-owner-') as temporary:
        owner_manifest=human.bodyparts_myosim_torso_anatomy_visual_payload(
            SOURCES,anatomy,REGISTRATION_PATH,RIGID_PATH.parent,Path(temporary))
        owner_path=Path(temporary)/owner_manifest['payload']['file']
        raw=owner_path.read_bytes()
        old_header=HEADER.unpack_from(raw)
        rigid_source_sha=struct.unpack_from('<8s10I32s',RIGID_PATH.read_bytes())[-1]
        registration_fingerprint=int(human.sha256(REGISTRATION_PATH)[:8],16)
        if (old_header[0]!=b'NHANAT1\0' or old_header[1] not in (1,2)
                or old_header[2]!=304 or old_header[5]!=registration_fingerprint
                or old_header[6]!=rigid_source_sha):
            raise ValueError('existing BodyParts3D anatomy owner has a different source/registration identity')
        count=old_header[2];vertex_count=old_header[3];index_count=old_header[4]
        record_offset=HEADER.size
        records=np.frombuffer(raw,dtype='<u4',count=count*8,offset=record_offset).reshape(-1,8)
        vertex_offset=record_offset+RECORD.size*count
        packed_vertices=np.frombuffer(raw,dtype='<f4',count=vertex_count*6,offset=vertex_offset).reshape(-1,6)
        index_offset=vertex_offset+vertex_count*24
        packed_indices=np.frombuffer(raw,dtype='<u4',count=index_count,offset=index_offset)
        if len(owner_manifest['source']['surfaces'])!=count:
            raise ValueError('existing anatomy owner receipt surface count differs from payload')
        owners=[]
        for record,declared in zip(records,owner_manifest['source']['surfaces'],strict=True):
            body,start_vertex,nvertices,start_index,nindices,stable_id,layer,_=map(int,record)
            if stable_id!=declared['stable_id'] or body!=declared['core_body_index']:
                raise ValueError('existing anatomy owner payload and source receipt disagree')
            faces=(packed_indices[start_index:start_index+nindices].reshape(-1,3)-start_vertex).astype(np.int64)
            owner_metadata={k:declared[k] for k in ('concept_id','label','member_id','member_sha256',
                'hierarchy','layer','myosim_body','core_body_index','organ_coverage',
                'source_structure_kind') if k in declared}
            owners.append({'id':stable_id,'layer':layer,'name':declared['label'],
                'source':'BodyParts3D v4.0 current registration owner',
                'vertices':packed_vertices[start_vertex:start_vertex+nvertices,:3].astype(np.float64),
                'normals':packed_vertices[start_vertex:start_vertex+nvertices,3:].astype(np.float64),
                'faces':faces,'body_index':body,'source_hash':declared['member_sha256'],
                'source_member':declared['member_id'],'source_license':'CC-BY-4.0',
                'repair':None,'source_owner_metadata':owner_metadata})

    # Resolve every atlas source frame against the exact NHRIGID-derived
    # reference manifest before adding head structures.
    _,bodies=human._bodyparts_runtime_bindings(registration,RIGID_PATH.parent)
    head=bodies.get('head')
    if head is None:
        raise ValueError('registered source reference has no head body for brain/eye surfaces')
    head_body_index,head_binding=head
    reference=human.read_json(RIGID_PATH.parent/'myosim-fullbody-reference.manifest.json')
    head_record=next((row for row in reference['core_tree']['source_body_records']
        if row['name']=='head'),None)
    if (head_record is None or head_record['core_body_index']!=head_body_index
            or head_record['source_body_id']!=head_binding['source_body_id']):
        raise ValueError('registered head body binding differs from exact rigid reference')
    coordinate=registration.get('coordinate_system',{})
    global_matrix=coordinate.get('global_source_mm_to_myosim_world_m')
    global_translation,global_quaternion,global_scale=human._bodyparts_visual_local_pose(
        global_matrix,'resting head anatomy global transform')
    global_rotation=human._myosim_matrix_from_quaternion_xyzw(global_quaternion)
    head_position=head_record['default_com_position_world_m']
    head_quaternion=head_record['default_inertial_quaternion_world_xyzw']
    head_inverse=human._matrix_transpose(human._myosim_matrix_from_quaternion_xyzw(head_quaternion))
    head_specs=(('FMA50801','brain'),('FMA12514','right eyeball'),('FMA12515','left eyeball'))
    head_members=[];seen=set()
    for concept,label in head_specs:
        members=bodyparts3d['tables']['part_of'].get((concept,label),set())
        if not members:
            raise ValueError(f'BodyParts3D source family is empty: {concept} {label}')
        for member_id in sorted(members):
            if member_id in seen:
                raise ValueError(f'head anatomy member appears in multiple families: {member_id}')
            seen.add(member_id)
            _,member,obj=human._bodyparts_obj_member(SOURCES,'part_of',member_id)
            vertices_mm,faces=human._bodyparts_obj_triangles(obj,member)
            source_normals=human._bodyparts_vertex_normals(vertices_mm,faces,member)
            world=human._bodyparts_source_mm_to_body_world(
                vertices_mm,[0.,0.,0.],[0.,0.,0.,1.],global_translation,global_quaternion,global_scale)
            local=human._bodyparts_world_to_body_stored_m(
                world,head_position,head_quaternion,[0.,0.,0.],[0.,0.,0.,1.],1.,
                f'resting head anatomy {member_id}')
            normals=[]
            for normal in source_normals:
                world_normal=human._bodyparts_unit_vector(
                    human._myosim_matrix_vector(global_rotation,list(normal)),
                    f'resting head anatomy {member_id} world normal')
                normals.append(human._bodyparts_unit_vector(
                    human._myosim_matrix_vector(head_inverse,world_normal),
                    f'resting head anatomy {member_id} local normal'))
            head_members.append({'id':322+len(head_members),'layer':1,'name':label,
                'source':'BodyParts3D v4.0 current head registration','vertices':np.asarray(local,dtype=np.float64),
                'normals':np.asarray(normals,dtype=np.float64),'faces':np.asarray(faces,dtype=np.int64),
                'body_index':head_body_index,'source_hash':hashlib.sha256(obj).hexdigest(),
                'source_member':member_id,'source_license':'CC-BY-4.0','repair':None,
                'head_family':{'concept_id':concept,'source_name':label,'hierarchy':'part_of'},
                'source_member_path':member})
    extra={'owner_payload':owner_manifest['payload'],
        'owner_source_map_sha256':human.sha256(human.REPOSITORY_ROOT/'config/bodyparts3d-myosim-torso-anatomy-map.v1.json'),
        'owner_surfaces':len(owners),'head_body_index':head_body_index,
        'head_source_families':[{'concept_id':concept,'source_name':label,'hierarchy':'part_of',
            'stable_ids':[row['id'] for row in head_members if row['head_family']['concept_id']==concept],
            'source_member_ids':sorted(bodyparts3d['tables']['part_of'][(concept,label)])}
            for concept,label in head_specs],
        'bodyparts3d_source':bodyparts3d['source']}
    return owners+head_members,extra


def compile_anatomy(output: Path) -> dict:
    surfaces,transforms=_source_surfaces()
    passive_surfaces,passive_source=_registered_passive_surfaces()
    surfaces=passive_surfaces+surfaces
    stable_ids=[s['id'] for s in surfaces]
    if len(set(stable_ids))!=len(stable_ids):
        raise ValueError('duplicate NHANATOMY stable surface ID')
    surfaces.sort(key=lambda s:s['id'])
    rigid_raw=RIGID_PATH.read_bytes()
    if len(rigid_raw)<80:raise ValueError('NHRIGID header is truncated')
    rigid_header=struct.unpack_from('<8s10I32s',rigid_raw)
    if rigid_header[0]!=b'NHRIGID2':raise ValueError('unexpected rigid header')
    source_sha=rigid_header[-1]
    bones_path=BONE_DIR/'bodyparts3d-myosim-major-bones.nhbones'
    bones_manifest=json.loads((BONE_DIR/'bodyparts3d-myosim-major-bones.manifest.json').read_text())
    bones_raw=bones_path.read_bytes();bh=HEADER.unpack_from(bones_raw)
    if bh[0]!=b'NHBONES1' or bh[5]!=int(bones_manifest['payload']['registration_fingerprint32'],16):
        raise ValueError('current NHBONES registration identity mismatch')
    bone_record=struct.Struct('<6I8fI');anchors=bones_manifest['source']['anchors']
    if len(anchors)!=bh[2] or bone_record.size!=60:raise ValueError('NHBONES anchor mapping does not match record count')
    bone_ids={}
    for member in (*RIB_MEMBERS,*STERNUM_MEMBERS):
        anchor_index=next(i for i,row in enumerate(anchors) if row['member_id']==member)
        record=bone_record.unpack_from(bones_raw,60+anchor_index*bone_record.size)
        if record[0]!=TORSO_BODY_INDEX:raise ValueError(f'{member} NHBONES owner is not torso')
        bone_ids[member]=record[5]
    raw_records=[];vertex_parts=[];index_parts=[];repair_summaries=[];nv=0;ni=0
    for s in surfaces:
        v=np.asarray(s['vertices'],dtype=np.float64);f=np.asarray(s['faces'],dtype=np.int64)
        if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=len(v):
            raise ValueError(f"malformed source geometry {s['name']}")
        normals=np.asarray(s['normals'],dtype=np.float64) if 'normals' in s else _normals(v,f)
        if normals.shape!=v.shape or not np.isfinite(normals).all():
            raise ValueError(f"invalid registered normal field {s['name']}")
        local=np.column_stack([v,normals]).astype('<f4')
        if s['layer'] in (7,9):
            post=analyze_topology(v.astype(float).tolist(),f.astype(int).tolist())
            if not post['closed_oriented_manifold_candidate']:
                raise ValueError(f"native float anatomy is open/nonmanifold: {s['name']}")
        raw_records.append(RECORD.pack(int(s.get('body_index',TORSO_BODY_INDEX)),nv,len(v),ni,f.size,s['id'],s['layer'],0))
        vertex_parts.append(local.tobytes());index_parts.append((f.ravel()+nv).astype('<u4').tobytes())
        if s.get('repair') is not None:repair_summaries.append({'stable_id':s['id'],'name':s['name'],'repair':s['repair']})
        nv+=len(v);ni+=f.size
    registration_fp=int(bones_manifest['payload']['registration_fingerprint32'],16)
    muscle_raw=NHTISS_PATH.read_bytes()
    muscle_manifest_raw=NHTISS_MANIFEST_PATH.read_bytes()
    muscle_manifest=json.loads(muscle_manifest_raw)
    muscle_header=HEADER.unpack_from(muscle_raw)
    if (hashlib.sha256(muscle_raw).hexdigest()!=NHTISS_SHA256
            or hashlib.sha256(muscle_manifest_raw).hexdigest()!=NHTISS_MANIFEST_SHA256
            or muscle_header[0]!=b'NHTISS4\0' or muscle_header[1]!=5
            or muscle_manifest.get('payload',{}).get('sha256')!=NHTISS_SHA256
            or muscle_manifest.get('payload',{}).get('registration_fingerprint32')!=f'{registration_fp:08x}'
            or muscle_manifest.get('source',{}).get('registration',{}).get('sha256')!=human.sha256(REGISTRATION_PATH)
            or muscle_manifest.get('payload',{}).get('surface_count')!=muscle_header[2]
            or muscle_manifest.get('payload',{}).get('binding_count')!=muscle_header[3]
            or muscle_manifest.get('payload',{}).get('vertex_count')!=muscle_header[4]
            or muscle_manifest.get('payload',{}).get('index_count')!=muscle_header[5]):
        raise ValueError('current NHTISS4 payload identity or registration mismatch')
    payload=HEADER.pack(b'NHANAT1\0',5,len(surfaces),nv,ni,registration_fp,source_sha)+b''.join(raw_records)+b''.join(vertex_parts)+b''.join(index_parts)
    output.mkdir(parents=True,exist_ok=True)
    for source_path in (NHTISS_PATH,NHTISS_MANIFEST_PATH):
        link=output/source_path.name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            if hashlib.sha256(link.read_bytes()).hexdigest()!=hashlib.sha256(source_path.read_bytes()).hexdigest():
                raise ValueError(f'refusing to replace nonmatching anatomy bundle member {link}')
            link.unlink()
        link.symlink_to(source_path)
    payload_path=output/'resting-thorax.nhanatomy';payload_path.write_bytes(payload)
    payload_sha=hashlib.sha256(payload).hexdigest()
    lung_volumes=[]
    for s in surfaces:
        if s['id'] in {x[0] for x in LUNG_SPECS}:
            lung_volumes.append(abs(_signed_volume(np.asarray(s['vertices'],dtype=np.float64),np.asarray(s['faces'],dtype=np.int64))))
    mass_geometry = _mass_and_skin_volume_audit()
    functional={
        'anatomy_payload_sha256':payload_sha,'torso_body_index':TORSO_BODY_INDEX,
        'lung_stable_ids':[x[0] for x in LUNG_SPECS],'pleura_stable_ids':[310],
        'cardiac_cavity_stable_ids':list(CAVITY_STABLE_IDS),
        'cardiac_chamber_compartment_bindings':CVSIM_CHAMBER_BINDINGS,
        'superior_axis_body':[0.0,1.0,0.0],'anterior_axis_body':[1.0,0.0,0.0],
        'diaphragm_stable_ids':[311],'intercostal_stable_ids':[312,313,314,315,316,317],
        'rib_bone_stable_ids':[bone_ids[x] for x in RIB_MEMBERS],
        'sternum_bone_stable_ids':[bone_ids[x] for x in STERNUM_MEMBERS],
        'head_body_index':passive_source['head_body_index'],
        'head_structure_stable_ids':[row['id'] for row in surfaces if 'head_family' in row],
        'bones_payload_sha256':bones_manifest['payload']['sha256'],
    }
    receipt={'schema':'numi.human.resting-anatomy-receipt.v1','functional_bindings':functional,
        'mass_geometry_accounting':mass_geometry,
        'payload':{'path':str(payload_path),'abi':5,'sha256':payload_sha,'surface_count':len(surfaces),'vertex_count':nv,'index_count':ni,
            'source_sha256':source_sha.hex(),'registration_fingerprint32':f'{registration_fp:08x}'},
        'thorax_source_volume_m3':{'five_lung_envelopes':lung_volumes,'sum':sum(lung_volumes),
            'interpretation':'registered geometric atlas envelope; includes tissue and gas space; not FRC or lung gas volume'},
        'provenance':{'rigid_payload':str(RIGID_PATH),'rigid_payload_sha256':hashlib.sha256(rigid_raw).hexdigest(),
            'bones_payload':str(bones_path),'bones_payload_sha256':bones_manifest['payload']['sha256'],
            'bodyparts3d_archives':[
                {'file':'partof_BP3D_4.0_obj_99.zip','hierarchy':'part_of','sha256':'9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97','license':'CC-BY-4.0'},
                {'file':'isa_BP3D_4.0_obj_99.zip','hierarchy':'is_a','sha256':'40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e','license':'CC-BY-4.0'}],
            'bodyparts3d_source':passive_source['bodyparts3d_source'],
            'cardiac_cavity_source_bindings':CVSIM_CHAMBER_BINDINGS,
            'cardiac_cavity_geometry_sha256':'849bd41e2fa7e172e8cda2c6889658eda203c4c5e240b3a2adb7c5649c9243c1',
            'cardiac_cavity_geometry_definition':'canonical extract_cavity_surfaces output: four source OBJs plus source-lock provenance and exact decimal-coordinate quotient topology',
            'cardiac_cavity_volume_ownership':'anatomical cavity surfaces are registered visual identities; reduced CVSim chamber volumes own blood accounting',
            'cardiac_cavity_intersection_audit':{
                'right_atrium_right_ventricle':'42 triangle-pair source intersections; exact-rational RA-first/RV-first ownership alternatives share 0.07275373956738762 ml; no valve-plane biological ownership selected',
                'tricuspid_valve_boundary':'source leaflets FJ2421/FJ2433/FJ2436 do not define an annular/orifice partition surface',
                'qualification':'overlap remains unresolved; no cavity surface was silently assigned blood mass'},
            'myosim_source_archive_sha256':bones_manifest['source']['myosim_source_archive_sha256'],
            'bodyparts3d_torso_owner_map_sha256':passive_source['owner_source_map_sha256'],
            'bodyparts3d_torso_owner_surface_count':passive_source['owner_surfaces'],
            'head_source_families':passive_source['head_source_families'],
            'native_muscle_surfaces':{'payload_path':str(output/NHTISS_PATH.name),
                'source_path':str(NHTISS_PATH),'sha256':NHTISS_SHA256,
                'manifest_path':str(output/NHTISS_MANIFEST_PATH.name),
                'source_manifest_path':str(NHTISS_MANIFEST_PATH),
                'manifest_sha256':NHTISS_MANIFEST_SHA256,
                'surface_count':muscle_header[2],'body_binding_count':muscle_header[3],
                'vertex_count':muscle_header[4],'index_count':muscle_header[5],
                'registration_fingerprint32':f'{registration_fp:08x}'},
            'head_body_index':passive_source['head_body_index'],
            'zanatomy_export_sha256':lung.configuration()['export']['sha256'],
            'zanatomy_config_sha256':human.sha256(lung.CONFIG),
            'bodyparts_registration_sha256':human.sha256(REGISTRATION_PATH),
            'bodyparts_license':'CC-BY-4.0','zanatomy_license':'CC-BY-SA-4.0',
            'torso_rest_position_world_m':TORSO_POSITION_WORLD_M.tolist(),
            'torso_rest_orientation_xyzw':TORSO_QUATERNION_XYZW.tolist(),
            'source_id_map':{str(s['id']):{'name':s['name'],'provider':s['source'],'source_member':s['source_member'],
                'source_sha256':s['source_hash'],'layer':s['layer'],'body_index':int(s.get('body_index',TORSO_BODY_INDEX)),
                'repair':s.get('repair'),'source_owner_metadata':s.get('source_owner_metadata'),
                'head_family':s.get('head_family')}
                for s in surfaces},'rib_source_member_to_stable_id':{m:bone_ids[m] for m in RIB_MEMBERS},
            'sternum_source_member_to_stable_id':{m:bone_ids[m] for m in STERNUM_MEMBERS},
            'transform_registration':transforms['atlas_alignment'],
            'native_lung_geometry_scope':'closed source envelope surfaces used as the shared thoracic geometry; no parenchymal microstructure or clinical registration'},
        'qualification':{'source_identity':'passed','registered_geometry':'source transforms bound to current rigid and bone registration',
            'lung_closure':'derived cap surfaces and one explicitly recorded local overlap-sheet removal; no source coordinates moved',
            'cardiac_cavity_ownership':'unresolved for overlapping RA/RV geometry; reduced CVSim is the sole blood-volume owner',
            'self_intersection':'not assessed','inter_lobe_intersection':'not assessed','organ interface audit':'not assessed',
            'functional_muscle_route':'reduced aggregate diaphragm/intercostal channels; source muscle meshes are visual identities only',
        'clinical_validation':False}}
    receipt_path=output/'resting-anatomy-receipt.json';receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    manifest={'schema':'numi.human.resting-anatomy-manifest.v1','payload':receipt['payload'],
        'receipt':{'path':str(receipt_path),'sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest()},
        'functional_bindings':functional,'qualification':receipt['qualification'],
        'native_muscle_surfaces':receipt['provenance']['native_muscle_surfaces'],
        'thorax_source_volume_m3':receipt['thorax_source_volume_m3'],'source_surfaces':receipt['provenance']['source_id_map'],
        'mass_geometry_accounting':mass_geometry}
    (output/'resting-anatomy-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest



def _copy_checked_sidecar(source: Path, destination: Path, expected_sha256: str) -> None:
    """Reuse an immutable owner sidecar beside a newly composed receipt."""
    source, destination = Path(source).resolve(), Path(destination)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"cardiac sidecar identity differs: {source}")
    if destination.exists():
        raise ValueError(f"cardiac sidecar destination already exists: {destination}")
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"relocated cardiac sidecar identity differs: {destination}")


def _compose_skin_candidate_receipt_document(
    base_receipt: dict,
    base_receipt_path: Path,
    candidate_manifest: dict,
    candidate_manifest_path: Path,
    candidate_payload_path: Path,
    mass_geometry_accounting: dict,
    output_receipt_path: Path,
) -> dict:
    """Replace only skin accounting/provenance in a pinned anatomy receipt."""
    inputs = candidate_manifest.get("inputs", {})
    source = inputs.get("source_payload", {})
    output = candidate_manifest.get("output_payload", {})
    runtime = inputs.get("runtime_reference", {})
    rigid = runtime.get("rigid", {})
    mass = base_receipt.get("mass_geometry_accounting", {})
    provenance = base_receipt.get("provenance", {})
    if base_receipt.get("schema") != "numi.human.resting-anatomy-receipt.v1":
        raise ValueError("base receipt is not a resting anatomy receipt")
    if (Path(source.get("path", "")).resolve() != Path(mass.get("skin_payload_path", "")).resolve()
            or source.get("sha256") != mass.get("skin_payload_sha256")):
        raise ValueError("skin candidate source differs from base receipt skin accounting")
    if (Path(output.get("path", "")).resolve() != candidate_payload_path.resolve()
            or output.get("sha256") != hashlib.sha256(candidate_payload_path.read_bytes()).hexdigest()):
        raise ValueError("skin candidate output identity differs from its manifest")
    if (rigid.get("sha256") != provenance.get("rigid_payload_sha256")
            or inputs.get("registration_sha256") != provenance.get("bodyparts_registration_sha256")):
        raise ValueError("skin candidate rigid/registration identity differs from base anatomy receipt")
    payload = base_receipt.get("payload", {})
    payload_path = Path(payload.get("path", ""))
    if not payload_path.is_file() or hashlib.sha256(payload_path.read_bytes()).hexdigest() != payload.get("sha256"):
        raise ValueError("base anatomy payload identity differs")
    if base_receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") != payload.get("sha256"):
        raise ValueError("base anatomy functional binding does not match its payload")
    if (mass_geometry_accounting.get("skin_payload_sha256") != output.get("sha256")
            or Path(mass_geometry_accounting.get("skin_payload_path", "")).resolve() != candidate_payload_path.resolve()):
        raise ValueError("recomputed skin accounting does not bind the candidate output")

    result = copy.deepcopy(base_receipt)
    result["mass_geometry_accounting"] = mass_geometry_accounting
    result["provenance"]["skin_visual_binding_candidate"] = {
        "schema": candidate_manifest.get("schema"),
        "status": candidate_manifest.get("status"),
        "manifest_path": str(candidate_manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(candidate_manifest_path.read_bytes()).hexdigest(),
        "source_payload_path": str(Path(source["path"]).resolve()),
        "source_payload_sha256": source["sha256"],
        "payload_path": str(candidate_payload_path.resolve()),
        "payload_sha256": output["sha256"],
        "registration_sha256": inputs.get("registration_sha256"),
        "base_anatomy_receipt": {
            "path": str(Path(base_receipt_path).resolve()),
            "sha256": hashlib.sha256(Path(base_receipt_path).read_bytes()).hexdigest(),
        },
        "nhtiss4_owner_alignment": candidate_manifest.get("nhtiss4_owner_alignment"),
        "nhtiss4_composed_base_alignment": candidate_manifest.get("nhtiss4_composed_base_alignment"),
        "preservation": candidate_manifest.get("preservation"),
        "changed_skin_regions": candidate_manifest.get("changed_skin_regions", {}).get("combined_geometry", {}).get("dominant_anatomical_region_breakdown"),
        "qualification_boundary": candidate_manifest.get("evidence_boundary"),
        "native_accepted_pose_geometry_audit": "pending",
    }
    result["provenance"]["skin_visual_binding_candidate"]["receipt_path"] = str(output_receipt_path.resolve())
    return result


def compose_skin_binding_candidate(
    base_receipt_path: Path,
    skin_candidate_payload: Path,
    skin_rebind_manifest_path: Path,
    output: Path,
) -> dict:
    """Compose a candidate NHSKIN into the existing anatomy receipt contract.

    NHA geometry and functional bindings remain byte/hash-identical. The
    skin's volume/closure audit is recalculated through this owner's existing
    accounting routine, and the candidate binding provenance is appended.
    """
    base_receipt_path = Path(base_receipt_path).resolve()
    skin_candidate_payload = Path(skin_candidate_payload).resolve()
    skin_rebind_manifest_path = Path(skin_rebind_manifest_path).resolve()
    output = Path(output).resolve()
    if not base_receipt_path.is_file() or not skin_candidate_payload.is_file() or not skin_rebind_manifest_path.is_file():
        raise ValueError("skin receipt composition requires existing base receipt and candidate files")
    if output == base_receipt_path.parent or base_receipt_path in output.parents:
        raise ValueError("skin receipt composition output must be a separate evidence directory")
    if output.exists():
        raise ValueError("skin receipt composition output directory already exists")
    receipt_path = output / "resting-anatomy-receipt.json"
    manifest_path = output / "resting-anatomy-manifest.json"
    if receipt_path.exists() or manifest_path.exists():
        raise ValueError("skin receipt composition refuses to overwrite an existing receipt")
    base_receipt = json.loads(base_receipt_path.read_text())
    base_manifest_path = base_receipt_path.with_name("resting-anatomy-manifest.json")
    base_receipt_sha = hashlib.sha256(base_receipt_path.read_bytes()).hexdigest()
    if base_manifest_path.is_file():
        base_manifest = json.loads(base_manifest_path.read_text())
        base_manifest_sha = hashlib.sha256(base_manifest_path.read_bytes()).hexdigest()
        if (base_manifest.get("receipt", {}).get("path") != str(base_receipt_path)
                or base_manifest.get("receipt", {}).get("sha256") != base_receipt_sha):
            raise ValueError("base anatomy manifest does not bind the selected receipt")
        manifest_lineage = {
            "source_receipt_path": str(base_receipt_path),
            "source_receipt_sha256": base_receipt_sha,
            "source_manifest_path": str(base_manifest_path),
            "source_manifest_sha256": base_manifest_sha,
        }
    else:
        # Some existing anatomy owners emit a receipt-only composition. Accept
        # it only when its embedded composition record binds the exact
        # predecessor receipt and payload; then reconstruct the same v1
        # manifest fields from the selected receipt itself.
        lineage = base_receipt.get("provenance", {}).get("airway_sibling_overlap_partition")
        if not isinstance(lineage, dict):
            raise ValueError("base anatomy manifest is missing and receipt has no verifiable composition lineage")
        predecessor_path = Path(lineage.get("base_receipt_path", "")).resolve()
        predecessor_sha = lineage.get("base_receipt_sha256")
        if (not predecessor_path.is_file() or not isinstance(predecessor_sha, str)
                or hashlib.sha256(predecessor_path.read_bytes()).hexdigest() != predecessor_sha):
            raise ValueError("base receipt composition lineage does not bind its predecessor receipt")
        predecessor = json.loads(predecessor_path.read_text())
        current_payload = base_receipt.get("payload", {})
        predecessor_payload = predecessor.get("payload", {})
        lineage_payload_path = Path(lineage.get("base_payload_path", "")).resolve()
        lineage_payload_sha = lineage.get("base_payload_sha256")
        output_payload_path = Path(lineage.get("output_payload_path", "")).resolve()
        output_payload_sha = lineage.get("output_payload_sha256")
        if (predecessor_payload.get("path") != str(lineage_payload_path)
                or predecessor_payload.get("sha256") != lineage_payload_sha
                or current_payload.get("path") != str(output_payload_path)
                or current_payload.get("sha256") != output_payload_sha
                or not lineage_payload_path.is_file()
                or hashlib.sha256(lineage_payload_path.read_bytes()).hexdigest() != lineage_payload_sha
                or not output_payload_path.is_file()
                or hashlib.sha256(output_payload_path.read_bytes()).hexdigest() != output_payload_sha):
            raise ValueError("base receipt composition lineage does not bind its anatomy payload")
        base_manifest = {
            "schema": "numi.human.resting-anatomy-manifest.v1",
            "payload": current_payload,
            "receipt": {"path": str(base_receipt_path), "sha256": base_receipt_sha},
            "functional_bindings": base_receipt.get("functional_bindings"),
            "qualification": base_receipt.get("qualification"),
            "native_muscle_surfaces": base_receipt.get("provenance", {}).get("native_muscle_surfaces"),
            "thorax_source_volume_m3": base_receipt.get("thorax_source_volume_m3"),
            "source_surfaces": base_receipt.get("provenance", {}).get("source_id_map"),
            "mass_geometry_accounting": base_receipt.get("mass_geometry_accounting"),
        }
        manifest_lineage = {
            "source_manifest_absent": str(base_manifest_path),
            "source_receipt_path": str(base_receipt_path),
            "source_receipt_sha256": base_receipt_sha,
            "predecessor_receipt_path": str(predecessor_path),
            "predecessor_receipt_sha256": predecessor_sha,
        }
    candidate_manifest = json.loads(skin_rebind_manifest_path.read_text())
    mass_geometry = _mass_and_skin_volume_audit(
        skin_path=skin_candidate_payload,
        skin_rebind_manifest_path=skin_rebind_manifest_path,
    )
    # Ensure a declared NHTISS4 owner is present and hash-current before
    # verifying its lower-limb rows against the candidate skin.
    muscle = base_receipt.get("provenance", {}).get("native_muscle_surfaces")
    if muscle is not None:
        if not isinstance(muscle, dict):
            raise ValueError("declared native muscle surface owner is malformed")
        required_muscle_fields = ("payload_path", "sha256", "manifest_path", "manifest_sha256")
        if any(not isinstance(muscle.get(key), str) or not muscle[key] for key in required_muscle_fields):
            raise ValueError("declared native muscle surface owner lacks payload/manifest identities")
        tissue_path = Path(muscle["payload_path"])
        tissue_manifest = Path(muscle["manifest_path"])
        if not tissue_path.is_absolute():
            tissue_path = base_receipt_path.parent / tissue_path
        if not tissue_manifest.is_absolute():
            tissue_manifest = base_receipt_path.parent / tissue_manifest
        if not tissue_path.is_file() or not tissue_manifest.is_file():
            raise ValueError("declared native muscle surface payload or manifest is missing")
        if (hashlib.sha256(tissue_path.read_bytes()).hexdigest() != muscle["sha256"]
                or hashlib.sha256(tissue_manifest.read_bytes()).hexdigest() != muscle["manifest_sha256"]):
            raise ValueError("declared native muscle surface payload or manifest hash is stale")
        from .skin_lower_limb_anchor_rebind import _verify_nhtiss_owner_alignment
        registration = json.loads(REGISTRATION_PATH.read_text())
        candidate_manifest["nhtiss4_composed_base_alignment"] = _verify_nhtiss_owner_alignment(
            tissue_payload_path=tissue_path,
            tissue_manifest_path=tissue_manifest,
            registration_sha256=hashlib.sha256(REGISTRATION_PATH.read_bytes()).hexdigest(),
            source_archive_sha256=candidate_manifest["payload_identity"]["source_archive_sha256"],
            registration_fingerprint32=int(candidate_manifest["payload_identity"]["registration_fingerprint32"], 16),
            candidate_skin=skin_candidate_payload.read_bytes(),
            registration=registration,
        )
        mass_geometry["nhtiss4_base_owner_alignment_sha256"] = hashlib.sha256(
            json.dumps(candidate_manifest["nhtiss4_composed_base_alignment"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    output_receipt = copy.deepcopy(base_receipt)
    common_sidecars = []
    common = output_receipt.get("provenance", {}).get("cardiac_geometry_binding", {}).get("common_field")
    if common is not None:
        if not isinstance(common, dict) or common.get("schema") != "numi.human.cardiac_common_field.v1":
            raise ValueError("unsupported common cardiac field receipt")
        for owner in ("map", "polynomials", "domain_boxes"):
            record = common.get(owner)
            if (not isinstance(record, dict) or not isinstance(record.get("path"), str)
                    or not isinstance(record.get("sha256"), str)):
                raise ValueError(f"missing common cardiac {owner} identity")
            source_path = Path(record["path"])
            if not source_path.is_absolute():
                source_path = base_receipt_path.parent / source_path
            source_path = source_path.resolve()
            digest = record["sha256"]
            if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"common cardiac {owner} source identity differs")
            destination = output / source_path.name
            if destination.exists():
                raise ValueError(f"common cardiac {owner} destination already exists")
            common_sidecars.append((source_path, destination, digest))
            record["path"] = source_path.name
    mass_geometry["skin_binding_rebind_receipt"] = str(skin_rebind_manifest_path)
    mass_geometry["skin_binding_rebind_receipt_sha256"] = hashlib.sha256(skin_rebind_manifest_path.read_bytes()).hexdigest()
    result = _compose_skin_candidate_receipt_document(
        output_receipt, base_receipt_path, candidate_manifest, skin_rebind_manifest_path,
        skin_candidate_payload, mass_geometry, receipt_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    for source_path, destination, digest in common_sidecars:
        _copy_checked_sidecar(source_path, destination, digest)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    composed_manifest = copy.deepcopy(base_manifest)
    composed_manifest["source_receipt_lineage"] = manifest_lineage
    composed_manifest["receipt"] = {
        "path": str(receipt_path),
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }
    composed_manifest["mass_geometry_accounting"] = mass_geometry
    manifest_path.write_text(json.dumps(composed_manifest, indent=2, sort_keys=True) + "\n")
    return {
        "status": "skin_receipt_composed_with_recomputed_open_surface_accounting",
        "base_receipt_path": str(base_receipt_path),
        "base_receipt_sha256": hashlib.sha256(base_receipt_path.read_bytes()).hexdigest(),
        "candidate_skin_path": str(skin_candidate_payload),
        "candidate_skin_sha256": hashlib.sha256(skin_candidate_payload.read_bytes()).hexdigest(),
        "candidate_manifest_path": str(skin_rebind_manifest_path),
        "candidate_manifest_sha256": hashlib.sha256(skin_rebind_manifest_path.read_bytes()).hexdigest(),
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "anatomy_payload_unchanged": result["payload"] == base_receipt["payload"],
        "functional_bindings_unchanged": result["functional_bindings"] == base_receipt["functional_bindings"],
        "mass_geometry_accounting": mass_geometry,
    }

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--patch-diaphragm-lung-interfaces',action='store_true',
        help='refine the pinned NHANAT5 resting anatomy payload with exact reciprocal diaphragm/lobe material interfaces')
    parser.add_argument('--base-payload',type=Path,
        help='existing NHANAT5 input for --patch-diaphragm-lung-interfaces')
    parser.add_argument('--base-receipt',type=Path,
        help='source receipt paired with --base-payload, or base receipt for skin composition')
    parser.add_argument('--compose-skin-binding-candidate', action='store_true',
        help='compose a registered NHSKIN candidate into an existing receipt and recalculate its skin audit')
    parser.add_argument('--skin-candidate-payload', type=Path,
        help='NHSKIN ABI 5 output from the lower-limb binding-rebind owner')
    parser.add_argument('--skin-rebind-manifest', type=Path,
        help='provenance manifest paired with --skin-candidate-payload')
    args=parser.parse_args()
    if args.compose_skin_binding_candidate:
        if args.base_receipt is None or args.skin_candidate_payload is None or args.skin_rebind_manifest is None:
            parser.error('--compose-skin-binding-candidate requires --base-receipt, --skin-candidate-payload, and --skin-rebind-manifest')
        if args.patch_diaphragm_lung_interfaces or args.base_payload is not None:
            parser.error('skin composition cannot be combined with diaphragm interface patch options')
        result=compose_skin_binding_candidate(args.base_receipt, args.skin_candidate_payload, args.skin_rebind_manifest, args.output)
    elif args.patch_diaphragm_lung_interfaces:
        from .resting_anatomy_interface_patch import BASE, build_candidate

        payload=args.base_payload or (BASE/'resting-thorax.nhanatomy')
        receipt=args.base_receipt or (BASE/'resting-anatomy-receipt.json')
        result=build_candidate(payload,receipt,args.output)
    else:
        if args.base_payload is not None or args.base_receipt is not None or args.skin_candidate_payload is not None or args.skin_rebind_manifest is not None:
            parser.error('base/candidate inputs require a corresponding anatomy composition option')
        result=compile_anatomy(args.output)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
