"""Compile a pinned joint diaphragm/lung/passive-viscera NHANAT candidate.

Outputs use the existing NHANAT1 ABI5 payload and anatomy/respiration receipts.
This owner admits only the retained registered reference inputs below; it does
not define internal liver planes or segment volumes. An optional source-bound
costal fit is applied as one passive coordinate field before joint conforming.
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
from . import resting_pleura_proxy as pleura
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
PINNED_IMPRESSION_SHA256 = {
    "liver_impression_npz": "14d70c30802a87e9c80084bd11c8bf260571c1d97e4fed583fcc8769f83df379",
    "liver_impression_report": "8284e97e910527246669ddcfe5e6268d74d3daaa1d3c91b3c9b955a27a6c72b8",
    "impression_guide_npz": "34add6ea6dc343b8ffce6dce5ee35f8997f158c959416f5d94114548c00039a6",
    "impression_guide_receipt": "102ea0cbbdd41a0f30b787e9180f3810a1d881f37d39c5a91eb41921a4fa60a3",
}
COSTAL_NEIGHBOR_IDS = (2, 4, 398, 454, 458, 460, 461)
COSTAL_COMMON_FIELD_IDS = tuple(range(305, 312))
PINNED_COSTAL_FIELD_SHA256 = {
    "controls": "6a1c196d98b7301f3424805644df293e70916bfdd84fb531705fdeef114d08d9",
    "receipt": "4722b03997f6da94a963d8f2aa91cfb008633eeb7643e8c46a3d1a2334427bf0",
    "field": "2ef6b18811dee0a35cba347c4623090b6e577e0c2ca22af1c921fdd293fddacc",
    "source_driver": "02947965a6aecca39e181d0613e169a3852b13463d5c91e06ac302256a342794",
    "control_values": "f4d31e0e59c0cfd4767b3c56049c5bdb570767229d36e38fe872239220bb52af",
    "source_payload": "93a3a10f2ce3282b3a0b3903be584aa8a7b3ea3e3438db2ede69264b4dd7fedd",
    "candidate_payload": "b9dbd123d45c7b87330cad28049d44390b426346959a26502da1b63d002629d9",
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


def validate_impression_report(report, candidate_sha256):
    """Fail closed unless the exact retained CSG derivative preserved all D contacts."""
    if report.get('candidate_sha256') != candidate_sha256:
        raise ValueError('visceral-impression report does not bind the exact liver candidate')
    if report.get('liver_source_sha256') != PINNED_INPUT_SHA256['liver_npz']:
        raise ValueError('visceral-impression report source liver identity differs from the pinned main mesh')
    if report.get('preserved_diaphragm_interface_faces') != 849:
        raise ValueError('visceral-impression candidate did not preserve all 849 diaphragm contacts')
    if report.get('lost_diaphragm_interface_faces') != 0:
        raise ValueError('visceral-impression candidate lost a diaphragm contact')


def load_costal_source_field(controls_path, receipt_path):
    """Load the exact receipt-bound source-space field from the costal fit owner."""
    controls_path=Path(controls_path); receipt_path=Path(receipt_path)
    if sha(controls_path)!=PINNED_COSTAL_FIELD_SHA256['controls']:
        raise ValueError('costal field controls SHA mismatch')
    if sha(receipt_path)!=PINNED_COSTAL_FIELD_SHA256['receipt']:
        raise ValueError('costal field controls receipt SHA mismatch')
    controls=json.loads(controls_path.read_text()); receipt=json.loads(receipt_path.read_text())
    expected={
        'schema':'numi.human.costal_source_field_controls.v1',
        'field_sha256':PINNED_COSTAL_FIELD_SHA256['field'],
        'controls_sha256':PINNED_COSTAL_FIELD_SHA256['control_values'],
        'input_payload_sha256':PINNED_COSTAL_FIELD_SHA256['source_payload'],
        'candidate_payload_sha256':PINNED_COSTAL_FIELD_SHA256['candidate_payload'],
        'control_count':1083,'anchor_point_count':9739,'surface_ids':list(COSTAL_COMMON_FIELD_IDS),
    }
    for key,value in expected.items():
        if controls.get(key)!=value: raise ValueError(f'costal field metadata mismatch for {key}')
    if receipt.get('schema')!='numi.human.costal_source_field_controls_receipt.v1' or receipt.get('sha256')!=sha(controls_path):
        raise ValueError('costal field receipt does not bind the exact controls file')
    if receipt.get('field_sha256')!=PINNED_COSTAL_FIELD_SHA256['field'] or receipt.get('driver_sha256')!=PINNED_COSTAL_FIELD_SHA256['source_driver']:
        raise ValueError('costal field receipt does not bind the retained field owner')
    algorithm=controls.get('algorithm',{})
    if (algorithm.get('support_radius_m')!=0.018 or algorithm.get('neighbor_count')!=24
            or algorithm.get('control_voxel_m')!=0.004
            or algorithm.get('anchor_fade')!={'clearance_m':0.002,'formula':'smoothstep(clamp((nearest_anchor_distance-0.002)/0.008,0,1))','transition_m':0.008}):
        raise ValueError('costal field algorithm differs from the source fit contract')
    points=np.asarray(controls.get('controls_source_local_m'),dtype=np.float64)
    targets=np.asarray(controls.get('target_displacements_m'),dtype=np.float64)
    anchors=np.asarray(controls.get('fixed_anchor_points_source_local_m'),dtype=np.float64)
    if points.shape!=(1083,3) or targets.shape!=points.shape or anchors.shape!=(9739,3):
        raise ValueError('costal field point arrays have invalid dimensions')
    if not np.isfinite(points).all() or not np.isfinite(targets).all() or not np.isfinite(anchors).all():
        raise ValueError('costal field contains nonfinite values')
    if float(np.linalg.norm(targets,axis=1).max(initial=0.0))>0.001000001:
        raise ValueError('costal source targets exceed the retained 1 mm bound')
    return {'controls':points,'targets':targets,'anchors':anchors,'radius_m':0.018,
        'controls_sha256':sha(controls_path),'receipt_sha256':sha(receipt_path),
        'field_sha256':controls['field_sha256'],'source_driver_sha256':receipt['driver_sha256'],
        'source_payload_sha256':controls['input_payload_sha256'],'candidate_payload_sha256':controls['candidate_payload_sha256']}


def evaluate_costal_source_field(points, field):
    """Evaluate SmoothCostalField with its exact compact radial/anchor weights."""
    from scipy.spatial import cKDTree
    p=np.asarray(points,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): raise ValueError('costal field points must be finite Nx3')
    controls=field['controls']; targets=field['targets']; radius=float(field['radius_m'])
    tree=cKDTree(controls); anchor_tree=cKDTree(field['anchors']); k=min(24,len(controls))
    out=np.zeros_like(p)
    for start in range(0,len(p),32768):
        batch=p[start:start+32768]
        d,ids=tree.query(batch,k=k,distance_upper_bound=radius,workers=1)
        if k==1: d,ids=d[:,None],ids[:,None]
        valid=np.isfinite(d)&(ids<len(controls)); t=np.zeros_like(d)
        t[valid]=np.clip(d[valid]/radius,0.0,1.0)
        w=np.zeros_like(d); w[valid]=(1.0-t[valid])**4*(4.0*t[valid]+1.0)
        vec=np.zeros((len(batch),3),dtype=np.float64); den=w.sum(axis=1)
        for j in range(k):
            ok=valid[:,j]
            vec[ok]+=w[ok,j,None]*targets[ids[ok,j]]
        vec=np.divide(vec,den[:,None],out=np.zeros_like(vec),where=den[:,None]>1e-12)
        nearest=np.min(np.where(valid,d,np.inf),axis=1)
        activation=np.where(np.isfinite(nearest),(1.0-np.clip(nearest/radius,0.0,1.0))**2,0.0)
        anchor_d,_=anchor_tree.query(batch,k=1,workers=1)
        u=np.clip((anchor_d-0.002)/0.008,0.0,1.0)
        activation*=u*u*(3.0-2.0*u)
        out[start:start+len(batch)]=vec*activation[:,None]
    return out


def apply_costal_source_positions(points, field):
    old=np.asarray(points,dtype=np.float32)
    moved=old.astype(np.float64)+evaluate_costal_source_field(old,field)
    new=moved.astype(np.float32)
    delta=moved-old.astype(np.float64)
    report={'vertex_count':len(old),'changed_position_count':int(np.count_nonzero(np.any(new!=old,axis=1))),
        'maximum_displacement_m':float(np.linalg.norm(delta,axis=1).max(initial=0.0)),
        'rms_displacement_m':float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))) if len(delta) else 0.0}
    return new,report


def apply_costal_source_field(row, field):
    old=np.asarray(row['vertices6'],dtype=np.float32)
    faces=np.asarray(row['faces'],dtype=np.int64)
    positions,report=apply_costal_source_positions(old[:,:3],field)
    new=normal_rows(positions,faces)
    output=dict(row); output['vertices6']=new; output['faces']=faces
    return output,report


def apply_costal_source_field_if_changed(row, field):
    """Keep exact source bytes when the registered Float32 surface is unmoved."""
    updated, report = apply_costal_source_field(row, field)
    return (updated if report['changed_position_count'] else row), report


def build_candidate(*, composition_payload: Path, composition_receipt: Path,
                    immutable_base_payload: Path, respiration_payload: Path,
                    respiration_receipt: Path, diaphragm_candidate_manifest: Path,
                    diaphragm_npz: Path, liver_npz: Path, segment_guide_npz: Path,
                    segment_guide_receipt: Path, respiration_config: Path,
                    output_dir: Path, liver_impression_npz: Path | None = None,
                    liver_impression_report: Path | None = None,
                    costal_field_controls: Path | None = None,
                    costal_field_receipt: Path | None = None) -> dict:
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
    if (liver_impression_npz is None) != (liver_impression_report is None):
        raise ValueError('liver impression candidate and report must be supplied together')
    impression_mode = liver_impression_npz is not None
    IMPRESSION = Path(liver_impression_npz) if liver_impression_npz is not None else None
    IMPRESSION_REPORT = Path(liver_impression_report) if liver_impression_report is not None else None
    SURFACE_LIVER = IMPRESSION if impression_mode else liver_npz
    EXPECTED = dict(PINNED_INPUT_SHA256)
    if impression_mode:
        EXPECTED.update(PINNED_IMPRESSION_SHA256)
        EXPECTED['guide_npz']=PINNED_IMPRESSION_SHA256['impression_guide_npz']
        EXPECTED['guide_receipt']=PINNED_IMPRESSION_SHA256['impression_guide_receipt']
    if (costal_field_controls is None) != (costal_field_receipt is None):
        raise ValueError('costal field controls and receipt must be supplied together')
    costal_mode=costal_field_controls is not None
    if costal_mode and not impression_mode:
        raise ValueError('the shared costal field branch requires the exact visceral-impression aggregate')
    COSTAL_PATH=Path(costal_field_controls) if costal_field_controls is not None else None
    COSTAL_REC=Path(costal_field_receipt) if costal_field_receipt is not None else None
    costal_field=load_costal_source_field(COSTAL_PATH,COSTAL_REC) if costal_mode else None
    costal_displacements={}
    costal_conformed_surface_ids=set()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    required = (
        ("composition_payload", input_payload), ("composition_receipt", input_receipt),
        ("base93_payload", BASE93), ("resp_payload", resp_payload), ("resp_receipt", resp_receipt),
        ("diaphragm_candidate_manifest", diaphragm_candidate_manifest), ("diaphragm_npz", diaphragm_npz),
        ("liver_npz", liver_npz), ("guide_npz", GUIDE), ("guide_receipt", GUIDE_REC),
        ("respiration_config", CONFIG),
    )
    required = list(required)
    if impression_mode:
        required.extend((("liver_impression_npz", IMPRESSION), ("liver_impression_report", IMPRESSION_REPORT)))
    if costal_mode:
        required.extend((("costal_field_controls", COSTAL_PATH), ("costal_field_receipt", COSTAL_REC)))
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
    if guide_meta.get("aggregate_surface", {}).get("sha256") != sha(SURFACE_LIVER):
        raise ValueError("segment guide is not bound to the exact aggregate liver array")
    if impression_mode:
        report_meta = json.loads(IMPRESSION_REPORT.read_text())
        validate_impression_report(report_meta, sha(IMPRESSION))
        if sha(IMPRESSION_REPORT) != EXPECTED['liver_impression_report']:
            raise ValueError('visceral-impression report SHA mismatch')
        guide_source = guide_meta.get('source_inputs', {})
        if guide_source.get('visceral_impression_candidate_report_sha256') != sha(IMPRESSION_REPORT):
            raise ValueError('segment guide is not bound to the exact visceral-impression report')
        if guide_source.get('original_aggregate_liver_source_sha256') != EXPECTED['liver_npz']:
            raise ValueError('segment guide source liver differs from the pinned original aggregate')
    for key, path in (("composition_payload", input_payload), ("composition_receipt", input_receipt),
        ("base93_payload", BASE93), ("resp_payload", resp_payload), ("resp_receipt", resp_receipt),
        ("diaphragm_candidate_manifest", diaphragm_candidate_manifest), ("diaphragm_npz", diaphragm_npz),
        ("liver_npz", liver_npz), ("guide_npz", GUIDE), ("guide_receipt", GUIDE_REC)):
        if sha(path) != EXPECTED[key]:
            raise ValueError(f"input SHA mismatch for {key}: {sha(path)}")
    if impression_mode:
        for key, path in (("liver_impression_npz", IMPRESSION), ("liver_impression_report", IMPRESSION_REPORT),
                          ("impression_guide_npz", GUIDE), ("impression_guide_receipt", GUIDE_REC)):
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
    if costal_mode:
        costal_displacements['respiratory_surfaces']={}
        for sid in (305,306,307,308,309):
            updated,report=apply_costal_source_field_if_changed(resp_rows[sid],costal_field)
            costal_displacements['respiratory_surfaces'][str(sid)]=report
            if report['changed_position_count']:
                resp_rows[sid]=updated
                costal_conformed_surface_ids.add(sid)
        costal_displacements['visceral_neighbors']={}
        for sid in COSTAL_NEIGHBOR_IDS:
            if rows[sid].get('body_index')!=20:
                raise ValueError(f'costal source field expected torso20 coordinates for neighbor {sid}')
            updated,report=apply_costal_source_field_if_changed(rows[sid],costal_field)
            costal_displacements['visceral_neighbors'][str(sid)]=report
            if report['changed_position_count']:
                rows[sid]=updated
                costal_conformed_surface_ids.add(sid)
    # Source lung lobes and diaphragm come from the exact source payload used by the reciprocal-interface owner.
    for sid in (305,306,307,308,309): rows[sid]=resp_rows[sid]
    old_interface=resp_meta['provenance']['diaphragm_lung_interface']
    old_lung_patch={}
    for entry in old_interface['interface_rows']:
        sid=int(entry['lung_stable_id'])
        old_lung_patch[sid]=refine._patch_face_ids(entry,'registered_lung_face_index_ranges')
    # Candidate D includes the prior exact lung patches and the 923-face liver-side clip result.
    z=np.load(diaphragm_npz); dv=np.asarray(z['vertices'],dtype=np.float32); df=np.asarray(z['triangles'],dtype=np.int64)
    if costal_mode: dv,costal_displacements['diaphragm']=apply_costal_source_positions(dv,costal_field)
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
    l=np.load(SURFACE_LIVER); lv=np.asarray(l['vertices'],dtype=np.float32)
    lface_key='faces' if 'faces' in l.files else 'triangles' if 'triangles' in l.files else None
    if lface_key is None: raise ValueError('aggregate liver candidate has no face array')
    lf=np.asarray(l[lface_key],dtype=np.int64)
    if costal_mode: lv,costal_displacements['aggregate_liver']=apply_costal_source_positions(lv,costal_field)
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
    if costal_mode: gv,_=apply_costal_source_positions(gv,costal_field)
    if not (np.array_equal(gv,lv) and np.array_equal(gf,lf) and set(np.unique(gid))==set(range(14,22))): raise ValueError('segment guide does not bind the exact CSG liver shell')
    if impression_mode:
        # Rebuild passive display patches over the inferred aggregate shell while
        # preserving each existing NHA stable identity and source metadata.
        for sid in range(14,22):
            selected=np.flatnonzero(gid==sid)
            if not len(selected): raise ValueError(f'impression guide has no faces for stable ID {sid}')
            source_faces=gf[selected]
            used=np.unique(source_faces)
            inverse=np.full(len(gv),-1,dtype=np.int64); inverse[used]=np.arange(len(used))
            row=dict(rows[sid]); row['vertices6']=normal_rows(gv[used],inverse[source_faces]); row['faces']=inverse[source_faces]
            rows[sid]=row
            repair=receipt['provenance']['source_id_map'][str(sid)]['repair']
            repair['prior_patch_face_count_before_visceral_impression']=repair.get('patch_face_count')
            repair['patch_face_count']=int(len(selected))
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
    conformed_ids=set(list(range(14,22))+[305,306,307,308,309,311])
    if costal_mode: conformed_ids.update(costal_conformed_surface_ids)
    conformed_ids=sorted(conformed_ids)
    modified=sorted(set(conformed_ids)|{310})
    prepared={}
    for sid in conformed_ids:
        row=rows[sid]
        prepared[sid]=conform.conform_surface(row['vertices6'],row['faces'],
            progress=lambda fi,total,out,sid=sid: print('conform',sid,fi,total,out,flush=True),coordinate_resolution_m=0)
    short_report=conform.resolve_short_edges(prepared,1.25e-7,311)
    quality_report=quality.improve_sliver_faces(prepared)
    quality_report['implementation_sha256']=sha(Path(quality.__file__))
    mappings={sid:prepared[sid][2] for sid in conformed_ids}
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
    for sid in conformed_ids:
        rows[sid]['vertices6']=prepared[sid][0]; rows[sid]['faces']=prepared[sid][1]
    # Stable ID 310 remains source-bound as generic Pleura, but its rendered
    # passive geometry is rederived from the final five lobe shells so its
    # coordinates and triangulation follow exactly the same accepted source map.
    pleura_vertices6,pleura_faces,pleura_derivation=pleura.derive_lung_union_exterior(
        {sid:(prepared[sid][0],prepared[sid][1]) for sid in (305,306,307,308,309)},
        coordinate_quantization_m=resp_meta.get('provenance',{}).get('conforming_respiratory_cells',{}).get('coordinate_resolution_m'))
    rows[310]=dict(rows[310]); rows[310]['vertices6']=pleura_vertices6; rows[310]['faces']=pleura_faces
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
    resp_bound_ids={305,306,307,308,309,311}
    r['provenance']['source_id_map']={k:v for k,v in r['provenance']['source_id_map'].items() if int(k) not in resp_bound_ids}
    for sid in sorted(resp_bound_ids): r['provenance']['source_id_map'][str(sid)]=resp_meta['provenance']['source_id_map'][str(sid)]
    r['provenance']['respiratory_liver_joint_geometry']={
        'immutable_93a_base_payload_sha256':EXPECTED['base93_payload'],'input_composed_liver_payload_sha256':EXPECTED['composition_payload'],
        'respiratory_refinement005_payload_sha256':EXPECTED['resp_payload'],'candidate_diaphragm_npz_sha256':EXPECTED['diaphragm_npz'],
        'candidate_closed_liver_npz_sha256':sha(SURFACE_LIVER),'source_csg_liver_npz_sha256':EXPECTED['liver_npz'],
        'segment_attribution_guide_sha256':sha(GUIDE),
        'method':'Recover exact diaphragm/lung reciprocal face sets by canonical Float32 coordinate triangles; reorder D patch parents into contiguous runs; append the 923 diaphragm-side clipping faces; clip all five closed lung shells, diaphragm and eight aggregate liver exterior patches on the same exact rational Kuhn field; contract the qualified seam/short edges with the shared coordinate quotient; improve eligible sliver diagonals jointly across reciprocal owners; derive stable ID 310 as the exact external union boundary of the final five lobe surfaces.',
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
        'aggregate_liver_surface_candidate_sha256':sha(SURFACE_LIVER),
        'interface_ownership':'D and aggregate liver own exact reciprocal copies on the 849-face clipped material boundary; 74 source D fragments restore the surface intersected only by the excluded cranial remnant.',
        'status':'static reciprocal interface preserved through joint common-field conforming; native full-cycle/neighbor qualification pending'}
    if impression_mode:
        owner_values, owner_counts = np.unique(np.asarray(np.load(IMPRESSION)['source_owner'], dtype=np.int64), return_counts=True)
        r['provenance']['liver_diaphragm_geometry_candidate']['visceral_impression']={
            'candidate_npz_sha256':sha(IMPRESSION),'candidate_report_sha256':sha(IMPRESSION_REPORT),
            'driver_sha256':report_meta.get('driver_sha256'),'source_owner_face_counts':{
                str(int(owner)):int(count) for owner,count in zip(owner_values,owner_counts)},
            'scope':report_meta.get('scope'),
            'interpretation':'inferred passive aggregate exterior with source-face organ lineage; nearest-source labels are display patches, not internal segment boundaries or segment volumes'}
    if costal_mode:
        control_doc=json.loads(COSTAL_PATH.read_text())
        r['provenance']['thorax_costal_source_registration']={
            'model':'smooth_compact_costal_relief_field_v1',
            'parameter_status':'inferred_reference_registration_not_measured_subject_geometry',
            'functional_role':'passive geometry registration only; no additional forces, mass or physiology',
            'source_payload_sha256':costal_field['source_payload_sha256'],
            'controls_file_sha256':costal_field['controls_sha256'],
            'controls_receipt_sha256':costal_field['receipt_sha256'],
            'field_sha256':costal_field['field_sha256'],
            'source_driver_sha256':costal_field['source_driver_sha256'],
            'candidate013_payload_sha256':costal_field['candidate_payload_sha256'],
            'source_lung_ids':list(range(305,310)),
            'common_field_source_ids':list(COSTAL_COMMON_FIELD_IDS),
            'jointly_transformed_surface_ids':modified,
            'directly_conformed_surface_ids':conformed_ids,
            'derived_after_lobe_conforming_surface_ids':[310],
            'fixed_airway_hilum_anchor_ids':control_doc['anchor_surface_ids'],
            'fixed_anchor_point_count':control_doc['anchor_point_count'],
            'accepted_source_fit_steps':control_doc['accepted_steps'],
            'source_coordinate_frame':'registered torso20 body-local metres',
            'field_parameters':control_doc['algorithm'],
            'per_surface_displacements_before_conforming':costal_displacements,
            'shared_coordinate_rule':'one field evaluation for equal source-local coordinates; transformed points are rounded at the existing Float32 NHANAT boundary, then reciprocal patches are jointly conformed; stable ID 310 is rebuilt from the final exact external lobe-union boundary',
            'qualification':'offline shared-source registration recomposed before common conforming; accepted native frame and complete-cycle interfaces require fresh validation'}
    r['thorax_source_volume_m3']['five_lung_envelopes']=[volumes[str(s)] for s in (305,306,307,308,309)]
    r['thorax_source_volume_m3']['sum']=sum(r['thorax_source_volume_m3']['five_lung_envelopes'])
    r['qualification']['diaphragm_lung_interface']='static exact reciprocal interface survives common conforming mesh, numerical seam closure and shared quality conditioning; native complete-cycle qualification pending'
    r['qualification']['liver_diaphragm_interface']='static exact reciprocal 849-face source interface extended through common conforming mesh; native cycle and neighboring-organ qualification pending'
    r['qualification']['liver_surface_identity']='registered Z-Anatomy aggregate exterior with inferred passive segment display patches; internal Couinaud planes and independent segment volumes are not represented'
    r['qualification']['self_intersection']='unrelated whole-body and dynamic interface checks remain separate; no whole-body intersection-free claim'
    if costal_mode:
        r['qualification']['costal_source_registration']='inferred passive common-coordinate reference fit; new dynamic and full-cycle native geometry validation pending'
    # Save all interface parent-child face mappings for deterministic reconstruction.
    face_maps={}
    for sid in conformed_ids:
        mp=mappings[sid]
        if sid==311: selected=set().union(*[set(x) for x in d_lung_new.values()],set(d_liver_new))
        elif sid in old_lung_patch: selected=set(old_lung_patch[sid])
        elif sid in range(14,22): selected=set(range(len(segment_source_parent[sid]))) # source local face IDs before replacement
        elif costal_mode and sid in COSTAL_NEIGHBOR_IDS: selected=set(mappings[sid])
        else: selected=set()
        face_maps[str(sid)]={str(int(fi)):mp[int(fi)] for fi in sorted(selected)}
    OUT.mkdir(parents=True)
    raw,nv,ni=refine._pack(base_header,rows)
    pre_pleura_payload=OUT/'pre-pleura-resting-thorax.nhanatomy'; pre_pleura_payload.write_bytes(raw)
    pre_pleura_sha=sha(pre_pleura_payload)
    r['payload'].update({'path':str(pre_pleura_payload),'sha256':pre_pleura_sha,'surface_count':len(rows),'vertex_count':nv,'index_count':ni})
    r['functional_bindings']['anatomy_payload_sha256']=pre_pleura_sha
    r['provenance']['cardiac_geometry_binding']['output_anatomy_payload_sha256']=pre_pleura_sha
    if 'ventricular_wall_binding' in r['provenance']['cardiac_geometry_binding']:
        r['provenance']['cardiac_geometry_binding']['ventricular_wall_binding']['output_anatomy_payload_sha256']=pre_pleura_sha
    r['provenance']['cardiac_geometry_binding']['downstream_respiratory_liver_joint_interface']={
        'input_payload_sha256':EXPECTED['composition_payload'],'output_payload_sha256':pre_pleura_sha,'modified_stable_ids':modified,'preserved_cardiac_subset_byte_identity':True}
    pre_pleura_receipt=OUT/'pre-pleura-resting-anatomy-receipt.json'
    pre_pleura_receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    pleura_result=pleura.build_candidate(pre_pleura_payload,pre_pleura_receipt,OUT)
    payload=Path(pleura_result['output_payload_path']); payload_sha=pleura_result['output_payload_sha256']
    recpath=Path(pleura_result['output_receipt_path']); r=json.loads(recpath.read_text())
    r['provenance']['cardiac_geometry_binding']['downstream_respiratory_liver_joint_interface']={
        'input_payload_sha256':EXPECTED['composition_payload'],'output_payload_sha256':payload_sha,
        'modified_stable_ids':modified,'preserved_cardiac_subset_byte_identity':True}
    recpath.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    final_header=base.HEADER.unpack_from(payload.read_bytes())
    nv,ni=int(final_header[3]),int(final_header[4])
    config=json.loads(CONFIG.read_text()); config['diaphragm_area_m2']=area
    config.setdefault('parameter_scope',{})['joint_respiratory_liver_conforming_mesh']='Inferred reference registered geometry with a common exact piecewise-affine respiratory field; area recomputed from the final closed lobe geometry. Not measured-subject data.'
    configpath=OUT/'resting-reference-respiration.json'; configpath.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
    (OUT/'interface-parent-child-face-map.json').write_text(json.dumps(face_maps,separators=(',',':'))+'\n')
    source_hashes={'conforming_field_sha256':sha(Path(conform.__file__)),'refinement_sha256':sha(Path(refine.__file__)),
        'interface_owner_sha256':sha(Path(base.__file__)),'mesh_quality_sha256':sha(Path(quality.__file__)),
        'pleura_proxy_owner_sha256':sha(Path(pleura.__file__)),
        'builder_script_sha256':sha(Path(__file__)),'config_sha256':sha(CONFIG),'guide_receipt_sha256':sha(GUIDE_REC)}
    if impression_mode:
        source_hashes['impression_candidate_sha256']=sha(IMPRESSION)
        source_hashes['impression_candidate_report_sha256']=sha(IMPRESSION_REPORT)
    if costal_mode:
        source_hashes['costal_field_controls_sha256']=costal_field['controls_sha256']
        source_hashes['costal_field_controls_receipt_sha256']=costal_field['receipt_sha256']
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
        'diaphragm_topology':dtop_summary,'shared_short_edge_report':short_report,'quality_report':quality_report,
        'pleura_derivation':pleura_derivation,'source_implementation_hashes':source_hashes}
    if costal_mode: result['costal_source_displacements_before_conforming']=costal_displacements
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
    parser.add_argument("--liver-impression-npz", type=Path)
    parser.add_argument("--liver-impression-report", type=Path)
    parser.add_argument("--costal-field-controls", type=Path)
    parser.add_argument("--costal-field-receipt", type=Path)
    args = parser.parse_args(argv)
    kwargs = {name.replace("-", "_"): getattr(args, name.replace("-", "_")) for name in (
        "composition-payload", "composition-receipt", "immutable-base-payload",
        "respiration-payload", "respiration-receipt", "diaphragm-candidate-manifest",
        "diaphragm-npz", "liver-npz", "segment-guide-npz", "segment-guide-receipt",
        "respiration-config", "output-dir",
    )}
    kwargs.update(liver_impression_npz=args.liver_impression_npz,
                  liver_impression_report=args.liver_impression_report,
                  costal_field_controls=args.costal_field_controls,
                  costal_field_receipt=args.costal_field_receipt)
    result = build_candidate(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
