"""Compile a pinned joint diaphragm/lung/aggregate-liver NHANAT candidate.

Outputs use the existing NHANAT1 ABI5 payload and anatomy/respiration receipts.
This owner admits only the retained registered reference inputs below; it does
not define internal liver planes or segment volumes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import resting_anatomy_interface_patch as base
from . import resting_anatomy_conforming_refinement as refine
from . import resting_respiratory_conforming_field as conform
from . import resting_respiratory_mesh_quality as quality
from .resting_liver_surface_identity import liver_surface_identity_metadata, validate_liver_patch_coverage

HEADER = base.HEADER
RECORD = base.RECORD
PINNED_INPUT_SHA256 = {
    "composition_payload": "092076ae09ba45af4f2a4cdd3bc559cd010ab81c2bef165dec27ae18407cca77",
    "composition_receipt": "82f5b8efed37ecd2992263cd0476df49878783f8c3ddf04a088edf0f596b72d7",
    "base93_payload": "93a3a10f2ce3282b3a0b3903be584aa8a7b3ea3e3438db2ede69264b4dd7fedd",
    "resp_payload": "fcd21da7c0574e11e64b709362411b36c2ce756b77a3ea7e31262d2d70147b66",
    "resp_receipt": "e78ca6e133863f660d55f32fc88dd22c93626907ce9f053154765618134a6b51",
    "diaphragm_npz": "e245ccac924420c3037700881a148e5d150a730faf48aa27bfe00a682caa6860",
    "liver_npz": "97d273693bea34dfa30e5269756ae17e7a89bf16b1be591fbb1ac3088597f933",
    "guide_npz": "5d5513e8e353d7fc1d86a3101412b6914886908b6305f97790beea0f84dbd0b5",
    "guide_receipt": "b98cdfe8ae751938262777ada741b82e70f19a261d36a50feb4393dd4a172446",
    "diaphragm_candidate_manifest": "750a0585ab1b59942ef00a5ba12fd25e318585f0e5e335011c73377468057255",
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(v, tri): return tuple(sorted(tuple(float(x) for x in v[int(i),:3]) for i in tri))
def oriented(v, tri): return tuple(tuple(float(x) for x in v[int(i),:3]) for i in tri)
def identity(sid,row): return refine._surface_identity_sha256(sid,row)
def face_map(row, ids):
    out={}
    for fi in ids:
        k=ck(row['vertices6'],row['faces'][int(fi)])
        if k in out: raise ValueError(f'duplicate triangle key on surface {fi}')
        out[k]=(int(fi),oriented(row['vertices6'],row['faces'][int(fi)]))
    return out
def normal_rows(vertices, faces):
    xyz=np.asarray(vertices,dtype=np.float32); f=np.asarray(faces,dtype=np.int64)
    n=np.zeros_like(xyz,dtype=np.float64); tri=xyz[f].astype(np.float64)
    fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    for c in range(3): np.add.at(n,f[:,c],fn)
    lens=np.linalg.norm(n,axis=1)
    if not np.isfinite(lens).all() or np.any(lens<=1e-15): raise ValueError('cannot form source normals')
    return np.column_stack([xyz,(n/lens[:,None]).astype(np.float32)]).astype(np.float32)
def check_patch_union(surfaces):
    incidence={}
    for sid,row in surfaces.items():
        for tri in row['faces']:
            xyz=[tuple(float(x) for x in row['vertices6'][int(v),:3]) for v in tri]
            for a,b in ((xyz[0],xyz[1]),(xyz[1],xyz[2]),(xyz[2],xyz[0])):
                key=(min(a,b),max(a,b)); incidence.setdefault(key,[]).append(1 if a<b else -1)
    bad=[(k,v) for k,v in incidence.items() if len(v)!=2 or sum(v)!=0]
    if bad: raise ValueError(f'aggregate liver patch union has {len(bad)} invalid exact edge incidences')
    return len(incidence)
def verify_subset(a, a_ids, b, b_ids):
    am=face_map(a,a_ids); bm=face_map(b,b_ids)
    if set(am)!=set(bm): raise ValueError(f'reciprocal triangle sets differ: {len(set(am)-set(bm))}/{len(set(bm)-set(am))}')
    for key in am:
        aa=am[key][1]; bb=bm[key][1]
        rev=tuple(reversed(bb))
        if not any(aa==rev[k:]+rev[:k] for k in range(3)):
            raise ValueError('reciprocal triangles do not have opposite winding')
    return len(am)
def remap_ids(ids,mapping):
    return sorted({child for old in sorted(map(int,ids)) for child in mapping[int(old)]})
def subset_row(prepared):
    return {'vertices6':prepared[0],'faces':prepared[1]}

def record_reciprocal_child_pair(records, key, *, stable_id, diaphragm_face,
                                 liver_face, aggregate_parent_face,
                                 diaphragm_parent_faces, liver_parent_face):
    """Deduplicate shared flipped-child ancestry without hiding distinct faces."""
    owner=(int(stable_id),int(diaphragm_face),int(liver_face))
    if key not in records:
        records[key]={
            'owner':owner,
            'aggregate_parent_faces':{int(aggregate_parent_face)},
            'diaphragm_parent_faces':set(map(int,diaphragm_parent_faces)),
            'liver_parent_faces':{int(liver_parent_face)},
        }
        return
    row=records[key]
    if row['owner']!=owner:
        raise ValueError('one exact child triangle maps to multiple reciprocal surface owners')
    row['aggregate_parent_faces'].add(int(aggregate_parent_face))
    row['diaphragm_parent_faces'].update(map(int,diaphragm_parent_faces))
    row['liver_parent_faces'].add(int(liver_parent_face))


def build_candidate(*, composition_payload: Path, composition_receipt: Path,
                    immutable_base_payload: Path, respiration_payload: Path,
                    respiration_receipt: Path, diaphragm_candidate_manifest: Path,
                    diaphragm_npz: Path, liver_npz: Path, segment_guide_npz: Path,
                    segment_guide_receipt: Path, respiration_config: Path,
                    output_dir: Path) -> dict:
    """Build one source-locked reference geometry candidate without overwriting."""
    input_payload = Path(composition_payload)
    input_receipt = Path(composition_receipt)
    BASE93 = Path(immutable_base_payload)
    resp_payload = Path(respiration_payload)
    resp_receipt = Path(respiration_receipt)
    diaphragm_candidate_manifest = Path(diaphragm_candidate_manifest)
    diaphragm_npz = Path(diaphragm_npz)
    liver_npz = Path(liver_npz)
    GUIDE = Path(segment_guide_npz)
    GUIDE_REC = Path(segment_guide_receipt)
    CONFIG = Path(respiration_config)
    OUT = Path(output_dir)
    CAND = diaphragm_npz.parent
    EXPECTED = PINNED_INPUT_SHA256
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    required = (
        ("composition_payload", input_payload), ("composition_receipt", input_receipt),
        ("base93_payload", BASE93), ("resp_payload", resp_payload), ("resp_receipt", resp_receipt),
        ("diaphragm_candidate_manifest", diaphragm_candidate_manifest), ("diaphragm_npz", diaphragm_npz),
        ("liver_npz", liver_npz), ("guide_npz", GUIDE), ("guide_receipt", GUIDE_REC),
        ("respiration_config", CONFIG),
    )
    for name, path in required:
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    candidate_manifest = json.loads(diaphragm_candidate_manifest.read_text())
    if candidate_manifest.get("diaphragm-reciprocal_sha256") != sha(diaphragm_npz):
        raise ValueError("diaphragm candidate manifest does not bind the exact diaphragm array")
    if candidate_manifest.get("liver-main_sha256") != sha(liver_npz):
        raise ValueError("diaphragm candidate manifest does not bind the exact liver array")
    guide_meta = json.loads(GUIDE_REC.read_text())
    if guide_meta.get("output", {}).get("sha256") != sha(GUIDE):
        raise ValueError("segment guide receipt does not bind the exact guide array")
    if guide_meta.get("aggregate_surface", {}).get("sha256") != sha(liver_npz):
        raise ValueError("segment guide is not bound to the exact aggregate liver array")
    for key, path in (("composition_payload", input_payload), ("composition_receipt", input_receipt),
        ("base93_payload", BASE93), ("resp_payload", resp_payload), ("resp_receipt", resp_receipt),
        ("diaphragm_candidate_manifest", diaphragm_candidate_manifest), ("diaphragm_npz", diaphragm_npz),
        ("liver_npz", liver_npz), ("guide_npz", GUIDE), ("guide_receipt", GUIDE_REC)):
        if sha(path) != EXPECTED[key]:
            raise ValueError(f"input SHA mismatch for {key}: {sha(path)}")
    base_header, rows=base.parse_payload(input_payload); receipt=json.loads(input_receipt.read_text())
    resp_header, resp_rows=base.parse_payload(resp_payload); resp_meta=json.loads(resp_receipt.read_text())
    if base_header[5:]!=resp_header[5:]: raise ValueError('registration/source fingerprints differ')
    if receipt['payload']['sha256']!=sha(input_payload) or resp_meta['payload']['sha256']!=sha(resp_payload): raise ValueError('receipt does not bind exact payload')
    if not set(range(14,22)).issubset(rows) or not set((305,306,307,308,309,310,311)).issubset(rows): raise ValueError('composition input is missing required identities')
    # Ensure the input composition still preserves the immutable 93a non-liver arrays.
    _, rows93=base.parse_payload(BASE93)
    if sha(BASE93)!=EXPECTED['base93_payload']: raise ValueError('93a base payload SHA mismatch')
    for sid,row in rows.items():
        if sid in range(14,23): continue
        old=rows93[sid]
        if not (np.array_equal(row['vertices6'],old['vertices6']) and np.array_equal(row['faces'],old['faces'])):
            raise ValueError(f'composition input differs from 93a outside liver: {sid}')
    # Source lung lobes and diaphragm come from the exact source payload used by the reciprocal-interface owner.
    for sid in (305,306,307,308,309): rows[sid]=resp_rows[sid]
    old_interface=resp_meta['provenance']['diaphragm_lung_interface']
    old_lung_patch={}
    for entry in old_interface['interface_rows']:
        sid=int(entry['lung_stable_id'])
        old_lung_patch[sid]=refine._patch_face_ids(entry,'registered_lung_face_index_ranges')
    # Candidate D includes the prior exact lung patches and the 923-face liver-side clip result.
    z=np.load(diaphragm_npz); dv=np.asarray(z['vertices'],dtype=np.float32); df=np.asarray(z['triangles'],dtype=np.int64)
    if len(df)!=40924: raise ValueError('candidate D face count differs from retained receipt')
    d_by_key={}
    for fi,tri in enumerate(df): d_by_key.setdefault(ck(dv,tri),[]).append(int(fi))
    if any(len(v)!=1 for v in d_by_key.values()): raise ValueError('candidate D has duplicate exact faces')
    d_lung_ids={}
    for sid,oldids in old_lung_patch.items():
        lrow=resp_rows[sid]; selected=[]
        for key in {ck(lrow['vertices6'],lrow['faces'][fi]) for fi in oldids}:
            matches=d_by_key.get(key,[])
            if len(matches)!=1: raise ValueError(f'candidate D missing/duplicates reciprocal lung face for {sid}: {len(matches)}')
            selected.append(matches[0])
        d_lung_ids[sid]=sorted(selected)
        if len(selected)!=len(oldids): raise ValueError(f'lung reciprocal count changed for {sid}')
    # Candidate receipt identifies a contiguous 923-face diaphragm patch; 849 faces contact the retained liver shell and 74 restore D.
    d_liver_old_ids=list(range(len(df)-923,len(df)))
    if len(df)-len(d_liver_old_ids)!=40001: raise ValueError('candidate D source-face boundary changed')
    l=np.load(liver_npz); lv=np.asarray(l['vertices'],dtype=np.float32); lf=np.asarray(l['triangles'],dtype=np.int64)
    main_by_key={}
    for fi,tri in enumerate(lf):
        key=ck(lv,tri)
        if key in main_by_key: raise ValueError('duplicate liver main-shell triangle')
        main_by_key[key]=int(fi)
    d_liver_pairs=[]
    for fi in d_liver_old_ids:
        key=ck(dv,df[fi]); lfi=main_by_key.get(key)
        if lfi is None: continue
        do=oriented(dv,df[fi]); lo=oriented(lv,lf[lfi])
        if not any(do==lo[k:]+lo[:k] for k in range(3)) and not any(do==tuple(reversed(lo[k:]+lo[:k])) for k in range(3)):
            raise ValueError('triangle key did not reproduce source orientation')
        if not any(do==lo[k:]+lo[:k] for k in range(3)): d_liver_pairs.append((fi,lfi,key))
    if len(d_liver_pairs)!=849 or len(d_liver_old_ids)-len(d_liver_pairs)!=74: raise ValueError('D/liver contact or restored-fragment count differs from exact input receipt')
    # Verify the face guide is bound to this exact connected closed aggregate shell.
    g=np.load(GUIDE); gv=np.asarray(g['vertices'],dtype=np.float32); gf=np.asarray(g['triangles'],dtype=np.int64); gid=np.asarray(g['stable_id_per_triangle'],dtype=np.int64)
    if not (np.array_equal(gv,lv) and np.array_equal(gf,lf) and set(np.unique(gid))==set(range(14,22))): raise ValueError('segment guide does not bind the exact CSG liver shell')
    coverage=validate_liver_patch_coverage(vertices=lv,faces=lf,stable_id_per_face=gid,
        patch_rows={sid:rows[sid] for sid in range(14,22)},source_id_map=receipt['provenance']['source_id_map'])
    # Existing 8 NHA rows are source-segment display patches of this one aggregate shell.
    segment_source_parent={}; main_face_to_segment={}
    for sid in range(14,22):
        srcfaces=np.flatnonzero(gid==sid).astype(np.int64); old=rows[sid]
        oldkeys=face_map(old,np.arange(len(old['faces'])))
        bykey={ck(lv,lf[int(fi)]):int(fi) for fi in srcfaces}
        if len(srcfaces)!=len(old['faces']) or set(oldkeys)!=set(bykey): raise ValueError(f'published segment patch differs from closed liver source at {sid}')
        mapped=[]
        for localfi in range(len(old['faces'])):
            key=ck(old['vertices6'],old['faces'][localfi]); mainfi=bykey[key]
            mapped.append(mainfi); main_face_to_segment[mainfi]=(sid,localfi)
        segment_source_parent[sid]=mapped
    check_patch_union({sid:rows[sid] for sid in range(14,22)})
    # Sort source D faces into non-patch body, contiguous reciprocal lung-patch groups, and liver-side clipping patch.
    all_lung={fi for ids in d_lung_ids.values() for fi in ids}; all_liver=set(d_liver_old_ids)
    if all_lung & all_liver: raise ValueError('D lung and liver patches overlap in source face identity')
    order=[i for i in range(len(df)) if i not in all_lung and i not in all_liver]
    d_lung_new={}
    for sid in sorted(d_lung_ids):
        start=len(order); order.extend(d_lung_ids[sid]); d_lung_new[sid]=list(range(start,len(order)))
    d_liver_new=list(range(len(order),len(order)+len(d_liver_old_ids))); order.extend(d_liver_old_ids)
    if len(order)!=len(df) or len(set(order))!=len(df): raise ValueError('D face reorder lost or duplicated triangles')
    orig_to_new=np.empty(len(df),dtype=np.int64); orig_to_new[np.asarray(order,dtype=np.int64)]=np.arange(len(order))
    d_faces=df[np.asarray(order,dtype=np.int64)]
    d_liver_pair_old={int(fi) for fi,_,_ in d_liver_pairs}
    d_liver_pair_new=sorted(int(orig_to_new[fi]) for fi in d_liver_pair_old)
    d_used=np.unique(d_faces); d_remap=np.full(len(dv),-1,dtype=np.int64); d_remap[d_used]=np.arange(len(d_used)); d_active_faces=d_remap[d_faces]
    rows[311]=dict(rows[311]); rows[311]['vertices6']=normal_rows(dv[d_used],d_active_faces); rows[311]['faces']=d_active_faces
    for sid,ids in d_lung_new.items():
        if verify_subset(rows[sid],old_lung_patch[sid],rows[311],ids)!=len(ids): raise ValueError('D/lung source reciprocal-face check failed')
    # Jointly conform the 8 liver patches and 6 respiratory mechanical surfaces.
    modified=sorted(list(range(14,22))+[305,306,307,308,309,311])
    prepared={}
    for sid in modified:
        row=rows[sid]
        prepared[sid]=conform.conform_surface(row['vertices6'],row['faces'],
            progress=lambda fi,total,out,sid=sid: print('conform',sid,fi,total,out,flush=True),coordinate_resolution_m=0)
    short_report=conform.resolve_short_edges(prepared,1.25e-7,311)
    quality_report=quality.improve_sliver_faces(prepared)
    quality_report['implementation_sha256']=sha(Path(quality.__file__))
    mappings={sid:prepared[sid][2] for sid in modified}
    # Remap and update the exact native lung-patch ranges. Since D parent groups were reordered first,
    # remapped D child faces remain contiguous in the serialized D record.
    interface_rows=[]
    for old_entry in old_interface['interface_rows']:
        sid=int(old_entry['lung_stable_id'])
        l_old=old_lung_patch[sid]; d_old=d_lung_new[sid]
        l_new=remap_ids(l_old,mappings[sid]); d_new=remap_ids(d_old,mappings[311])
        if not l_new or len(l_new)!=len(d_new) or d_new!=list(range(d_new[0],d_new[-1]+1)):
            raise ValueError(f'joint conforming broke contiguous diaphragm patch {sid}')
        n=verify_subset(subset_row(prepared[sid]),l_new,subset_row(prepared[311]),d_new)
        entry=dict(old_entry)
        boundary=refine.face_set_boundary_edges(prepared[sid][1],l_new)
        entry.update({'registered_lung_face_index_ranges':base.index_ranges(l_new),'registered_lung_face_count':len(l_new),
            'diaphragm_patch_face_start':d_new[0],'diaphragm_patch_face_count':len(d_new),
            'patch_area_m2':refine._area(prepared[sid][0],prepared[sid][1],l_new),
            'shared_reciprocal_child_face_count':n,'shared_boundary_edge_count':len(boundary),
            'shared_boundary_vertex_count':len({v for e in boundary for v in e})})
        interface_rows.append(entry)
    # Transform the 849 source D/liver contacts through parent-face refinement and test every generated child pair.
    d_contact_parent=[int(orig_to_new[fi]) for fi in d_liver_pair_old]
    d_contact_children=remap_ids(d_contact_parent,mappings[311])
    d_child_by_key={}
    for fi in d_contact_children:
        key=ck(prepared[311][0],prepared[311][1][fi])
        if key in d_child_by_key: raise ValueError('duplicate child face in D/liver reciprocal region')
        d_child_by_key[key]=fi
    d_contact_parent_set=set(d_contact_parent)
    d_parent_by_child={}
    for parent,children in mappings[311].items():
        for child in children: d_parent_by_child.setdefault(int(child),set()).add(int(parent))
    liver_child_pairs_by_key={}
    liver_child_pair_lineage_occurrences=0
    for old_mainfi in range(len(lf)):
        sid,localfi=main_face_to_segment[old_mainfi]
        for childfi in remap_ids([localfi],mappings[sid]):
            key=ck(prepared[sid][0],prepared[sid][1][childfi])
            dfi=d_child_by_key.get(key)
            if dfi is None: continue
            do=oriented(prepared[311][0],prepared[311][1][dfi])
            lo=oriented(prepared[sid][0],prepared[sid][1][childfi])
            rev=tuple(reversed(lo))
            if not any(do==rev[k:]+rev[:k] for k in range(3)): raise ValueError('D/liver child contact has same or inconsistent winding')
            dparents=d_parent_by_child.get(int(dfi),set()) & d_contact_parent_set
            if not dparents: raise ValueError('reciprocal child has no diaphragm source-face ancestry')
            liver_child_pair_lineage_occurrences+=1
            record_reciprocal_child_pair(liver_child_pairs_by_key,key,stable_id=sid,
                diaphragm_face=dfi,liver_face=childfi,aggregate_parent_face=old_mainfi,
                diaphragm_parent_faces=dparents,liver_parent_face=localfi)
    expected=sum(len(mappings[main_face_to_segment[fi][0]][main_face_to_segment[fi][1]]) for _,fi,_ in d_liver_pairs)
    liver_child_pairs=list(liver_child_pairs_by_key.values())
    lineage_excess=expected-len(liver_child_pairs)
    recorded_lineage_excess=sum(len(x['aggregate_parent_faces'])-1 for x in liver_child_pairs)
    if liver_child_pair_lineage_occurrences!=expected:
        raise ValueError(f'D/liver child ancestry occurrence count {liver_child_pair_lineage_occurrences} != expected {expected}')
    if lineage_excess<0 or recorded_lineage_excess!=lineage_excess:
        raise ValueError(f'D/liver quality-flip ancestry mismatch: expected excess {lineage_excess}, recorded {recorded_lineage_excess}')
    duplicate_lineage=[]
    for key,row in liver_child_pairs_by_key.items():
        excess=len(row['aggregate_parent_faces'])-1
        if excess:
            duplicate_lineage.append({
                'canonical_triangle_sha256':hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest(),
                'stable_id':row['owner'][0],'diaphragm_child_face':row['owner'][1],
                'liver_child_face':row['owner'][2],'parent_lineage_occurrence_count':len(row['aggregate_parent_faces']),
                'aggregate_liver_parent_faces':sorted(row['aggregate_parent_faces']),
                'diaphragm_parent_faces':sorted(row['diaphragm_parent_faces']),
                'liver_surface_parent_faces':sorted(row['liver_parent_faces']),
            })
    if sum(x['parent_lineage_occurrence_count']-1 for x in duplicate_lineage)!=lineage_excess:
        raise ValueError('D/liver duplicate lineage audit does not reconcile')
    # Topology and source-volume checks on Float32 geometry.
    volumes={}
    for sid in (305,306,307,308,309):
        topo=base.topology_report(prepared[sid][1])
        if any(topo[k] for k in ('boundary_edge_count','nonmanifold_edge_count','orientation_error_edge_count','boundary_branch_vertex_count')):
            summary={k:v for k,v in topo.items() if k!='boundary_edges'}
            raise ValueError(f'lobe {sid} topology failed: {summary}')
        volumes[str(sid)]=base.signed_volume(prepared[sid][0][:,:3].astype(float),prepared[sid][1])
        if volumes[str(sid)]<=0: raise ValueError(f'lobe {sid} nonpositive volume')
    dtop=base.topology_report(prepared[311][1]); dtop_summary={k:v for k,v in dtop.items() if k!='boundary_edges'}
    if any(dtop[k] for k in ('boundary_edge_count','nonmanifold_edge_count','orientation_error_edge_count','boundary_branch_vertex_count')):
        raise ValueError(f'diaphragm topology failed after numerical seam quotient: {dtop_summary}')
    dvol=base.signed_volume(prepared[311][0][:,:3].astype(float),prepared[311][1])
    if dvol<=0: raise ValueError('diaphragm nonpositive signed volume')
    edge_count=check_patch_union({sid:subset_row(prepared[sid]) for sid in range(14,22)})
    liver_vol=sum(base.signed_volume(prepared[sid][0][:,:3].astype(float),prepared[sid][1]) for sid in range(14,22))
    if not (1.30e-3<liver_vol<1.45e-3): raise ValueError(f'liver aggregate volume outside registered range: {liver_vol}')
    # Derive the geometry owner area from post-quality closed lobe surfaces.
    area_rows=[]
    for sid in (305,306,307,308,309):
        v=prepared[sid][0][:,:3].astype(np.float64); f=prepared[sid][1]
        v0=base.signed_volume(v,f); value,_=conform.kuhn_basis(v); moved=v.copy(); moved[:,1]-=.01*value
        area=(base.signed_volume(moved,f)-v0)/.01
        area_rows.append({'lung_stable_id':sid,'effective_area_m2':area})
    area=sum(x['effective_area_m2'] for x in area_rows)
    if area<=0: raise ValueError('respiratory effective area nonpositive')
    for sid in modified:
        rows[sid]['vertices6']=prepared[sid][0]; rows[sid]['faces']=prepared[sid][1]
    # Verify non-target geometry remains byte-identical to the 93a source base.
    for sid,oldrow in rows93.items():
        if sid in modified or sid in range(14,23): continue
        if not (np.array_equal(rows[sid]['vertices6'],oldrow['vertices6']) and np.array_equal(rows[sid]['faces'],oldrow['faces'])):
            raise ValueError(f'unrelated 93a source record changed: {sid}')
    # Receipt derives from the current exact composition receipt plus respiratory IDs from refinement005.
    r=receipt
    r['payload']['path']=str(OUT/'resting-thorax.nhanatomy')
    r['functional_bindings']['liver_surface_identity']=liver_surface_identity_metadata()
    r['functional_bindings']['respiratory_geometry_binding']=dict(resp_meta['functional_bindings']['respiratory_geometry_binding'])
    rb=r['functional_bindings']['respiratory_geometry_binding']; prev_area=rb['diaphragm_effective_area_m2']
    rb.update({'basal_weight_interpolation':'conforming_kuhn_grid_v1','conforming_grid_spacing_m':conform.GRID_SPACING_M,
        'diaphragm_effective_area_m2':area,'per_lobe_effective_area_m2':area_rows,
        'source_refinement_area_update':{'previous_area_m2':prev_area,'refined_area_m2':area,'relative_change':(area-prev_area)/prev_area,
            'basis':'closed source-lobe volume derivative after joint diaphragm/liver reciprocal conforming, numerical seam quotient and shared quality conditioning'}})
    rb.pop('finite_displacement_sensitivity_m2',None)
    r['provenance']['diaphragm_lung_interface']=dict(old_interface)
    r['provenance']['diaphragm_lung_interface']['interface_rows']=interface_rows
    r['provenance']['diaphragm_lung_interface']['diaphragm_topology_after_joint_repair']=dtop_summary
    r['provenance']['diaphragm_lung_interface']['reciprocal_triangle_count_after_joint_repair']=sum(x['shared_reciprocal_child_face_count'] for x in interface_rows)
    r['provenance']['diaphragm_lung_interface']['qualification']='exact child-interface coordinate and winding checks pass after common-grid clipping, numerical-seam weld and shared sliver conditioning; native cycle audit pending'
    r['provenance']['source_id_map']={k:v for k,v in r['provenance']['source_id_map'].items() if int(k) not in {305,306,307,308,309,311}}
    for sid in (305,306,307,308,309,311): r['provenance']['source_id_map'][str(sid)]=resp_meta['provenance']['source_id_map'][str(sid)]
    r['provenance']['respiratory_liver_joint_geometry']={
        'immutable_93a_base_payload_sha256':EXPECTED['base93_payload'],'input_composed_liver_payload_sha256':EXPECTED['composition_payload'],
        'respiratory_refinement005_payload_sha256':EXPECTED['resp_payload'],'candidate_diaphragm_npz_sha256':EXPECTED['diaphragm_npz'],
        'candidate_closed_liver_npz_sha256':EXPECTED['liver_npz'],'segment_attribution_guide_sha256':EXPECTED['guide_npz'],
        'method':'Recover exact diaphragm/lung reciprocal face sets by canonical Float32 coordinate triangles; reorder D patch parents into contiguous runs; append the 923 diaphragm-side clipping faces; clip all five closed lung shells, diaphragm and eight aggregate liver exterior patches on the same exact rational Kuhn field; contract the qualified seam/short edges with the shared coordinate quotient; improve eligible sliver diagonals jointly across reciprocal owners.',
        'modified_stable_ids':modified,'diaphragm_lung_parent_reciprocal_faces':{str(sid):len(ids) for sid,ids in d_lung_ids.items()},
        'diaphragm_lung_recovered_by_exact_canonical_coordinates':True,'diaphragm_parent_liver_patch_face_count':len(d_liver_old_ids),
        'diaphragm_liver_parent_reciprocal_face_count':len(d_liver_pairs),'diaphragm_liver_restored_source_fragment_face_count':len(d_liver_old_ids)-len(d_liver_pairs),
        'diaphragm_liver_child_reciprocal_face_count':len(liver_child_pairs),
        'diaphragm_liver_child_lineage_occurrence_count':liver_child_pair_lineage_occurrences,
        'diaphragm_liver_quality_flip_shared_parent_lineage_excess_count':lineage_excess,
        'diaphragm_liver_quality_flip_shared_parent_lineage_excess':duplicate_lineage,
        'liver_aggregate_shell_volume_m3':liver_vol,
        'liver_display_segment_ids':list(range(14,22)),'liver_segment_volumes_owned_separately':False,
        'lung_lobe_signed_volumes_m3':volumes,'diaphragm_signed_volume_m3_after_numerical_seam_closure':dvol,
        'diaphragm_closed_topology_after_weld':dtop_summary,'liver_closed_union_unique_edge_count':edge_count,
        'short_edge_shared_quotient':short_report,'triangle_quality_conditioning':quality_report,'conforming_grid_spacing_m':conform.GRID_SPACING_M,
        'respiratory_effective_area_m2':area,'respiratory_per_lobe_area_m2':area_rows,
        'source_positions_preserved_before_declared_short_edge_weld':True,
        'qualification':'static exact interfaces and source topology checked; native complete-cycle and adjacent-organ checks remain required.'}
    r['provenance']['respiratory_liver_joint_geometry']['aggregate_face_coverage']=coverage
    r['provenance']['respiratory_liver_joint_geometry']['liver_display_patch_face_count_by_stable_id']={
        str(sid):len(rows[sid]['faces']) for sid in range(14,22)}
    for sid in range(14,22):
        r['provenance']['source_id_map'][str(sid)]['repair']['compiled_patch_face_count']=len(rows[sid]['faces'])
    r['provenance']['liver_diaphragm_geometry_candidate']={
        'candidate_source_path':str(CAND),'candidate_manifest_sha256':sha(diaphragm_candidate_manifest),
        'source_diaph_registration_payload_sha256':EXPECTED['resp_payload'],'main_liver_payload_sha256':EXPECTED['liver_npz'],
        'interface_ownership':'D and aggregate liver own exact reciprocal copies on the 849-face clipped material boundary; 74 source D fragments restore the surface intersected only by the excluded cranial remnant.',
        'status':'static reciprocal interface preserved through joint common-field conforming; native full-cycle/neighbor qualification pending'}
    r['thorax_source_volume_m3']['five_lung_envelopes']=[volumes[str(s)] for s in (305,306,307,308,309)]
    r['thorax_source_volume_m3']['sum']=sum(r['thorax_source_volume_m3']['five_lung_envelopes'])
    r['qualification']['diaphragm_lung_interface']='static exact reciprocal interface survives common conforming mesh, numerical seam closure and shared quality conditioning; native complete-cycle qualification pending'
    r['qualification']['liver_diaphragm_interface']='static exact reciprocal 849-face source interface extended through common conforming mesh; native cycle and neighboring-organ qualification pending'
    r['qualification']['liver_surface_identity']='registered Z-Anatomy aggregate exterior with inferred passive segment display patches; internal Couinaud planes and independent segment volumes are not represented'
    r['qualification']['self_intersection']='unrelated whole-body and dynamic interface checks remain separate; no whole-body intersection-free claim'
    # Save all interface parent-child face mappings for deterministic reconstruction.
    face_maps={}
    for sid in modified:
        mp=mappings[sid]
        if sid==311: selected=set().union(*[set(x) for x in d_lung_new.values()],set(d_liver_new))
        elif sid in old_lung_patch: selected=set(old_lung_patch[sid])
        elif sid in range(14,22): selected=set(range(len(segment_source_parent[sid]))) # source local face IDs before replacement
        else: selected=set()
        face_maps[str(sid)]={str(int(fi)):mp[int(fi)] for fi in sorted(selected)}
    OUT.mkdir(parents=True)
    raw,nv,ni=refine._pack(base_header,rows)
    payload=OUT/'resting-thorax.nhanatomy'; payload.write_bytes(raw); payload_sha=sha(payload)
    r['payload'].update({'path':str(payload),'sha256':payload_sha,'surface_count':len(rows),'vertex_count':nv,'index_count':ni})
    r['functional_bindings']['anatomy_payload_sha256']=payload_sha
    r['provenance']['cardiac_geometry_binding']['output_anatomy_payload_sha256']=payload_sha
    if 'ventricular_wall_binding' in r['provenance']['cardiac_geometry_binding']:
        r['provenance']['cardiac_geometry_binding']['ventricular_wall_binding']['output_anatomy_payload_sha256']=payload_sha
    r['provenance']['cardiac_geometry_binding']['downstream_respiratory_liver_joint_interface']={
        'input_payload_sha256':EXPECTED['composition_payload'],'output_payload_sha256':payload_sha,'modified_stable_ids':modified,'preserved_cardiac_subset_byte_identity':True}
    recpath=OUT/'resting-anatomy-receipt.json'; recpath.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    config=json.loads(CONFIG.read_text()); config['diaphragm_area_m2']=area
    config.setdefault('parameter_scope',{})['joint_respiratory_liver_conforming_mesh']='Inferred reference registered geometry with a common exact piecewise-affine respiratory field; area recomputed from the final closed lobe geometry. Not measured-subject data.'
    configpath=OUT/'resting-reference-respiration.json'; configpath.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
    (OUT/'interface-parent-child-face-map.json').write_text(json.dumps(face_maps,separators=(',',':'))+'\n')
    source_hashes={'conforming_field_sha256':sha(Path(conform.__file__)),'refinement_sha256':sha(Path(refine.__file__)),
        'interface_owner_sha256':sha(Path(base.__file__)),'mesh_quality_sha256':sha(Path(quality.__file__)),
        'builder_script_sha256':sha(Path(__file__)),'config_sha256':sha(CONFIG),'guide_receipt_sha256':sha(GUIDE_REC)}
    result={'payload_path':str(payload),'payload_sha256':payload_sha,'receipt_path':str(recpath),'receipt_sha256':sha(recpath),
        'config_path':str(configpath),'config_sha256':sha(configpath),'base_payload_sha256':EXPECTED['composition_payload'],
        'immutable_93a_base_payload_sha256':EXPECTED['base93_payload'],'surface_count':len(rows),'vertex_count':nv,'index_count':ni,
        'modified_stable_ids':modified,'preserved_other_surface_count':len(rows)-len(modified),'respiratory_effective_area_m2':area,
        'lung_signed_volumes_m3':volumes,'diaphragm_signed_volume_m3':dvol,'liver_aggregate_shell_volume_m3':liver_vol,
        'diaphragm_lung_source_reciprocal_counts':{str(sid):len(ids) for sid,ids in d_lung_ids.items()},
        'diaphragm_lung_reciprocal_child_count':r['provenance']['diaphragm_lung_interface']['reciprocal_triangle_count_after_joint_repair'],
        'diaphragm_liver_source_reciprocal_count':len(d_liver_pairs),'diaphragm_liver_reciprocal_child_count':len(liver_child_pairs),
        'diaphragm_liver_lineage_occurrence_count':liver_child_pair_lineage_occurrences,
        'diaphragm_liver_duplicate_lineage_excess_count':lineage_excess,
        'diaphragm_liver_duplicate_lineage_excess':duplicate_lineage,
        'diaphragm_topology':dtop_summary,'shared_short_edge_report':short_report,'quality_report':quality_report,'source_implementation_hashes':source_hashes}
    (OUT/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "composition-payload", "composition-receipt", "immutable-base-payload",
        "respiration-payload", "respiration-receipt", "diaphragm-candidate-manifest",
        "diaphragm-npz", "liver-npz", "segment-guide-npz", "segment-guide-receipt",
        "respiration-config", "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_candidate(**{name.replace("-", "_"): getattr(args, name.replace("-", "_")) for name in (
        "composition-payload", "composition-receipt", "immutable-base-payload",
        "respiration-payload", "respiration-receipt", "diaphragm-candidate-manifest",
        "diaphragm-npz", "liver-npz", "segment-guide-npz", "segment-guide-receipt",
        "respiration-config", "output-dir",
    )})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
