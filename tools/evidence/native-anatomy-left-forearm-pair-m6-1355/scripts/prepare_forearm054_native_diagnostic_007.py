#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, pickle, shutil, struct, sys, time
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np

A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
ROOT=Path('/Users/n/numi-human-fds132-m6-1350')
PARENT=A/'temporalis164-native-m6-002'
SOURCE=A/'left-fds-native-m6-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
T=PARENT/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
MAN=ROOT/'attempt-050/arrangement-001/result-001/source-parent-refs-054/source-parent-refs-and-shared-interface.json'
MAP_REPORT=MAN.with_name('report.json')
MAP_V2=ROOT/'attempt-061-step0-capture-remesh-diagnostic/candidate060-adapted-source-parent-map-v2.pkl'
MAP_TEMPLATE=ROOT/'attempt-061-step0-capture-remesh-diagnostic/candidate060-passive-tissue-parent-bindings-template-v2.json'
STEP0_REPORT=ROOT/'attempt-061-step0-capture-remesh-diagnostic/attempt-060-paired-cut-sliver-contraction-001-step0-report-corrected.json'
EXACT_CHECK=ROOT/'attempt-060-paired-cut-sliver-contraction-001/exact-check-001/report.json'
CONSTRUCTION=ROOT/'attempt-060-paired-cut-sliver-contraction-001/candidate.pkl'
CONSTRUCTION_REPORT=ROOT/'attempt-060-paired-cut-sliver-contraction-001/report.json'
STEP0_CAPTURE=A/'forearm054-native-diagnostic-006/baseline/native-run/accepted-geometry/step-0.mrvpack'
STEP0_CAPTURE_RECEIPT=A/'forearm054-native-diagnostic-006/baseline/native-run/accepted-geometry/step-0.receipt.json'
GEOM={}
REST_AUDIT=ROOT/'attempt-052/rest-reproduction-001/registered-rest-audit-003.json'
H=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
P=Path('/Users/n/numi-human-complete-reference-inventory-1335')
W=Path('/Users/n/numi-lab-parent-bindings-1353')
B=Path('/Users/n/numi-lab-parent-bindings-build-1353-003')
BIN=Path('/Users/n/numi-lab-parent-bindings-build-1353-004/bin/numi-human-native')
BUILD_REPORT=Path('/Users/n/numi-lab-parent-bindings-build-1353-004/final-build-verification.json')
PARENT_PREP=A/'prepare_temporalis164_native_m6_002.py'
PARENT_VERIFY=A/'verify_temporalis164_native_m6_002.py'
PARENT_LAUNCH=PARENT/'launch_arm.py'
OUT=A/'forearm054-native-diagnostic-007-candidate060'
START=time.monotonic()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def req(x,msg):
    if not x: raise RuntimeError(msg)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
def rational(pair): return F(int(pair[0]),int(pair[1]))
def f32array(x): return np.asarray(x,dtype='<f4')
def vstats(parent,child):
    delta=parent.astype(np.float64)-child.astype(np.float64)
    d=np.linalg.norm(delta,axis=1)*1e6
    ix=int(np.argmax(d))
    return dict(count=int(len(d)),max_um=float(d[ix]),max_index=ix,rms_um=float(np.sqrt(np.mean(d*d))),
                median_um=float(np.median(d)),p95_um=float(np.quantile(d,.95)),
                over_2um=int(np.count_nonzero(d>2.0)),max_delta_um=delta[ix].tolist())

req(not OUT.exists(),f'preserve existing output; refusing to overwrite {OUT}')
OUT.mkdir()
(OUT/'baseline').mkdir()
(OUT/'assets').mkdir()
progress=OUT/'progress.jsonl'
def log(event,**kw):
    rec=dict(event=event,elapsed_seconds=time.monotonic()-START,**kw)
    with progress.open('a') as f:f.write(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps(rec,sort_keys=True),flush=True)

# Pin and verify the passing parent, its unchanged source rows, 054 map, and row geometry.
for path in (T,T.with_suffix('.manifest.json'),SOURCE,SOURCE.with_suffix('.manifest.json'),MAN,MAP_REPORT,MAP_V2,MAP_TEMPLATE,
             STEP0_REPORT,EXACT_CHECK,CONSTRUCTION,CONSTRUCTION_REPORT,STEP0_CAPTURE,STEP0_CAPTURE_RECEIPT,REST_AUDIT,H,BIN,BUILD_REPORT,PARENT_PREP,PARENT_VERIFY,PARENT_LAUNCH):
    req(Path(path).is_file(),f'missing pinned input {path}')
req(sha(T)=='9d512c808bc57e6f1481e520558a8a6ae83af755a0480f04dc85c54d49566f69','passing164002 parent hash changed')
req(sha(BIN)=='147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174','1353build004 native binary hash mismatch')
req(sha(B/'bin/numi-human-native')==sha(BIN),'tested1353 overlay binary differs from build004')
parent_map=json.loads(MAN.read_text())
req(parent_map.get('schema')=='numi.human.left-fds-fdp-source-parent-reference-input-054.v1','054 map schema mismatch')
map_template=pickle.loads(MAP_V2.read_bytes())
receipt_template=json.loads(MAP_TEMPLATE.read_text())
req(map_template.get('schema')=='numi.human.passive-tissue-parent-map-template.v1','060 parent-map template schema mismatch')
req(map_template.get('candidate_sha256')==sha(CONSTRUCTION),'060 candidate pin mismatch')
construction_report=json.loads(CONSTRUCTION_REPORT.read_text())
exact_check=json.loads(EXACT_CHECK.read_text())
step0_report=json.loads(STEP0_REPORT.read_text())
req(construction_report.get('complete') is True and construction_report.get('admitted') is False and construction_report.get('union_link_and_material_stratum_guard') is True,'060 construction status or union/material guard mismatch')
req(construction_report.get('candidate_sha256')==sha(CONSTRUCTION) and construction_report.get('max_actual_displacement_m',1)>0 and construction_report.get('max_actual_displacement_m')<=construction_report.get('cumulative_displacement_bound_m')==1e-5,'060 10um displacement bound mismatch')
req(exact_check.get('complete') is True and not exact_check.get('duplicate_orientation_failures'),'060 exact-check incomplete or orientation failures present')
for sid in ('132','134','union'):
    chk=exact_check.get('checks',{}).get(sid,{})
    req(chk.get('topology',{}).get('closed') is True and chk.get('self_count')==0,f'060 exact topology/self check failed {sid}')
req(exact_check['checks']['union']['topology'].get('face_components')==12,'060 exact union component count mismatch')
req(step0_report.get('status')=='step0_capture_remesh_self_clear' and step0_report.get('candidate_admitted') is False and step0_report.get('gpu_used') is False,'060 step0 evidence boundary mismatch')
actual_capture=step0_report['actual_native006_step0_capture']
req(actual_capture['pack_path']==str(STEP0_CAPTURE) and actual_capture['pack_sha256']==sha(STEP0_CAPTURE),'actual native006 step0 capture pin mismatch')
req(actual_capture['receipt_path']==str(STEP0_CAPTURE_RECEIPT) and actual_capture['receipt_sha256']==sha(STEP0_CAPTURE_RECEIPT),'actual native006 step0 receipt pin mismatch')
for sid in ('132','134'):
    audit=actual_capture['remeshed_surface_audits'][sid]
    req(audit.get('self_count')==0 and audit.get('degenerate_count')==0,f'060 actual step0 remesh audit failed row {sid}')
req(actual_capture.get('shared_seam_bitwise_equal') is True,'060 actual step0 seam is not bitwise equal')
req(map_template.get('source_parent_manifest_sha256')==sha(MAN),'060 source parent manifest mismatch')
req(map_template.get('candidate_exact_check_sha256')==sha(EXACT_CHECK),'060 exact-check pin mismatch')
req(map_template.get('step0_capture_remesh_report_sha256')==sha(STEP0_REPORT),'060 remesh report pin mismatch')
req(map_template.get('rest_registration_preflight_status')=='passed','060 parent-map rest preflight did not pass')
req(receipt_template==map_template.get('native_receipt_template'),'060 receipt template identity mismatch')
raw_audit=json.loads(REST_AUDIT.read_text())
req(raw_audit.get('gpu_used') is False and raw_audit.get('native_child_emitted') is False,'prior audit scope mismatch')
for sid in (132,134):
    baseline=raw_audit['child_rest_reproduction'][str(sid)]['baseline_054_vs_interpolated_child_LBS_at_defaultQ']
    req(baseline['max_um']<=2.0 and baseline['over_2um']==0, f'prior baseline 054 rest gate evidence failed row {sid}')

sys.path.insert(0,str(P/'src'))
from numilab_human import passive_attachment_composition as pc, resting_run as rr
h=module('forearm054_forward',H)
parent=pc._read_nhtiss4(T);source=pc._read_nhtiss4(SOURCE)
parent_tissue=h.load_tissue(T)
rows={sid:h.row_data(parent_tissue,sid) for sid in (132,134)}
source_rows={int(r[6]):pc._row_slices(source,r) for r in source['records'] if int(r[6]) in (132,134)}
parent_pc_rows={int(r[6]):pc._row_slices(parent,r) for r in parent['records'] if int(r[6]) in (132,134)}
for sid in (132,134):
    req(pc._row_local_equal(source_rows[sid],parent_pc_rows[sid]),f'raw source row {sid} changed in passing164002 parent')
    src_path=Path(parent_map['rows'][str(sid)]['candidate_parent_path'])
    req(sha(src_path)==parent_map['rows'][str(sid)]['candidate_parent_sha256'],f'054 source candidate parent hash mismatch row {sid}')
    with np.load(src_path,allow_pickle=False) as z:
        req(np.array_equal(z['vertices6'][:,:3],rows[sid]['positions']),f'raw parent candidate positions mismatch row {sid}')
        req(np.array_equal(z['faces'].astype(np.int64),rows[sid]['faces']),f'raw parent candidate faces mismatch row {sid}')
        req(np.array_equal(z['binding_indices'].astype(np.uint32),rows[sid]['local']),f'raw parent candidate binding indices mismatch row {sid}')
        req(np.array_equal(z['weights'].astype(np.float32),rows[sid]['weights']),f'raw parent candidate weights mismatch row {sid}')

# Decode actual NHRIGID defaultQ poses, same owner method used by registered-rest audit.
RIG=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
ARCH=Path('/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202/source-inputs/myosim/myo_sim-33c89c2b.tar.gz')
rb=RIG.read_bytes();head=struct.unpack_from('<8s10I32s',rb,0)
magic,pabi,eabi,source_count,body_count,joint_count,nq,nv,root,virtual,reserved,embedded=head
req(magic==b'NHRIGID2' and pabi==1 and nq==nv+1,'NHRIGID header mismatch')
req(embedded.hex()==sha(ARCH),'NHRIGID source archive identity mismatch')
body_off=224;joint_off=body_off+160*body_count;dof_off=joint_off+144*joint_count
q_off=dof_off+64*nv;q=np.frombuffer(rb,dtype='<f4',count=nq,offset=q_off).copy()
v_off=q_off+4*nq;map_off=v_off+4*nv
source_to_core=np.frombuffer(rb,dtype='<u4',count=source_count,offset=map_off).copy();pose_off=map_off+4*source_count
poses={}
for si,core in enumerate(source_to_core):
    x,y,z,qx,qy,qz,qw=struct.unpack_from('<7f',rb,pose_off+28*si)
    poses[int(core)]=(np.asarray([x,y,z],dtype=np.float32),np.asarray([qx,qy,qz,qw],dtype=np.float32))
req(pose_off+28*source_count==len(rb),'NHRIGID pose table size mismatch')

# Consume the exact retained 060 geometry and source-parent map. No 054 geometry is rebuilt here.
# Coordinates and rational source terms are checked against the unchanged raw source rows before serialization.
terms={}; exact_terms={}; points={}; faces={}; global_ids={}; gid_to_index={}
map_surfaces=map_template['surfaces']
for sid in (132,134):
    row=map_surfaces.get(str(sid),map_surfaces.get(sid))
    req(row is not None and int(row['stable_id'])==sid,f'060 map row missing or wrong stable ID {sid}')
    gids=list(map(int,row['global_vertex_ids']))
    req(len(gids)>0 and len(set(gids))==len(gids),f'060 retained global IDs invalid row {sid}')
    xyz_exact=[[x if isinstance(x,F) else F(x) for x in v] for v in row['rest_vertices_exact_m']]
    req(len(xyz_exact)==len(gids) and all(len(v)==3 for v in xyz_exact),f'060 exact coordinate inventory invalid row {sid}')
    p=np.asarray([[np.float32(float(x)) for x in v] for v in xyz_exact],dtype='<f4')
    f=np.asarray(row['faces'],dtype='<u4')
    req(f.ndim==2 and f.shape[1]==3 and int(f.min())>=0 and int(f.max())<len(p),f'060 face indices invalid row {sid}')
    req(len(row['source_parent_terms_exact'])==len(gids),f'060 parent terms do not cover every child vertex row {sid}')
    e_terms=[]; float_terms=[]
    for i,vertex_terms in enumerate(row['source_parent_terms_exact']):
        parsed=[]; canonical=[]; sumcoef=F(0); reconstructed=[F(0),F(0),F(0)]
        for term in vertex_terms:
            req(len(term)==5,f'060 exact source term arity invalid row {sid}/{i}')
            source_index,psid,vi,num,den=map(int,term); coeff=F(num,den)
            req(source_index==0 and psid in (132,134) and 0<=vi<len(rows[psid]['positions']),f'060 source reference out of range row {sid}/{i}')
            req(coeff>0,f'060 source coefficient must be positive row {sid}/{i}')
            parsed.append((source_index,psid,vi,coeff));canonical.append([source_index,psid,vi,float(coeff)])
            sumcoef+=coeff
            for k in range(3): reconstructed[k]+=coeff*F.from_float(float(np.float32(rows[psid]['positions'][vi,k])))
        req(parsed==sorted(parsed,key=lambda x:(x[0],x[1],x[2])) and len({x[:3] for x in parsed})==len(parsed),f'060 source terms not canonical row {sid}/{i}')
        req(sumcoef==1,f'060 source coefficients do not sum exactly to one row {sid}/{i}')
        req(all(reconstructed[k]==xyz_exact[i][k] for k in range(3)),f'060 exact point is not reproduced by raw source terms row {sid}/{i}')
        template_terms=receipt_template['surfaces'][0 if sid==132 else 1]['vertices'][i]
        req(canonical==template_terms,f'060 receipt-template terms disagree with exact map row {sid}/{i}')
        e_terms.append(parsed);float_terms.append(canonical)
    terms[sid]=float_terms;exact_terms[sid]=e_terms;points[sid]=p;faces[sid]=f;global_ids[sid]=gids;gid_to_index[sid]={gid:i for i,gid in enumerate(gids)}
    req(len(row['original_054_face_indices'])==len(f),f'060 original-face ancestry missing row {sid}')
# The exact union is a closed multi-component partition; preserve its shared candidate IDs and term identity.
shared_ids=set(global_ids[132]) & set(global_ids[134])
req(len(shared_ids)==1986,'060 shared global candidate vertex count mismatch')
for gid in shared_ids:
    i=gid_to_index[132][gid];j=gid_to_index[134][gid]
    req(points[132][i].tobytes()==points[134][j].tobytes(),f'060 shared vertex coordinates differ {gid}')
    req(terms[132][i]==terms[134][j],f'060 shared vertex canonical source terms differ {gid}')
# Preserve parent binding tables and extend each child only with binding identities referenced by its full map.
def binding_chunks(data,row):
    b=row['binding_bytes'];return [b[i:i+36] for i in range(0,len(b),36)]
parent_bindings={sid:binding_chunks(parent,parent_pc_rows[sid]) for sid in (132,134)}
child_bindings={}
for out_sid in (132,134):
    used=sorted({int(term[1]) for vertex in exact_terms[out_sid] for term in vertex})
    merged=list(parent_bindings[out_sid])
    for psid in used:
        for key in parent_bindings[psid]:
            if key not in merged:merged.append(key)
    child_bindings[out_sid]=merged
child_bind_index={sid:{b:i for i,b in enumerate(vals)} for sid,vals in child_bindings.items()}
source_binding_bytes=parent_bindings

def parent_mix(vertex_terms):
    accum=defaultdict(F)
    for source_index,psid,vi,coeff in vertex_terms:
        rawrow=rows[int(psid)]
        for lane in range(4):
            wt=float(rawrow['weights'][int(vi),lane])
            if wt<=0:continue
            local=int(rawrow['local'][int(vi),lane]);req(local<len(source_binding_bytes[int(psid)]),'source lane binding index out of range')
            key=source_binding_bytes[int(psid)][local]
            accum[key]+=coeff*F.from_float(float(np.float32(wt)))
    return {k:v for k,v in accum.items() if v>0}

def make_candidate(strategy):
    result={}
    for out_sid in (132,134):
        n=len(points[out_sid]);loc=np.zeros((n,4),dtype='<u4');wout=np.zeros((n,4),dtype='<f4')
        cmap=child_bind_index[out_sid]
        for i,vertex_terms in enumerate(exact_terms[out_sid]):
            if strategy=='dominant':
                source_index,psid,vi,coeff=max(vertex_terms,key=lambda x:(x[3],-x[1],-x[2]))
                src=rows[int(psid)]
                for lane in range(4):
                    wt=np.float32(src['weights'][int(vi),lane]);li=int(src['local'][int(vi),lane])
                    if wt>0:
                        key=source_binding_bytes[int(psid)][li];req(key in cmap,f'child binding union missing source transform for {out_sid}/{i}')
                        loc[i,lane]=cmap[key];wout[i,lane]=wt
            elif strategy=='top4':
                acc=parent_mix(vertex_terms);total=sum(acc.values(),F(0))
                ordered=sorted(acc,key=lambda b:(-acc[b],cmap[b]))
                selected_bindings=ordered[:4];req(len(selected_bindings)==4,'top4 has fewer than four positive lanes')
                selected_total=sum((acc[b] for b in selected_bindings),F(0));factor=total/selected_total
                selected_bindings.sort(key=lambda b:cmap[b])
                vals=[np.float32(float(acc[b]*factor)) for b in selected_bindings]
                for lane,(b,x) in enumerate(zip(selected_bindings,vals)):
                    loc[i,lane]=cmap[b];wout[i,lane]=x
            else:raise RuntimeError(strategy)
        req(np.isfinite(wout).all() and np.all(wout>=0),'candidate provisional weights invalid')
        req(np.max(np.abs(wout.sum(axis=1,dtype=np.float64)-1))<1e-5,f'{strategy} child weight partition invalid row {out_sid}')
        result[out_sid]=(loc,wout)
    return result

# Runtime-accurate rest registration estimate: source map is expanded exactly as the C++ parent loader;
# candidate child four-lane fields use the same NHRIGID defaultQ and Float32 accumulation as the gate.
tissue=h.load_tissue(T)
lane_world={}
for sid in (132,134):
    lane_world[sid]=[]
    for lane in range(4):
        one=np.zeros_like(rows[sid]['weights'],dtype='<f4');active=rows[sid]['weights'][:,lane]>0;one[active,lane]=1.0
        lane_world[sid].append(h.forward(rows[sid]['positions'],rows[sid]['local'],one,rows[sid],tissue,poses))

def parent_rest(out_sid):
    result=np.zeros((len(terms[out_sid]),3),dtype='<f4')
    for i,vertex_terms in enumerate(terms[out_sid]):
        for source_index,psid,vi,coeff in vertex_terms:
            psid=int(psid);vi=int(vi);c=float(coeff)
            for lane in range(4):
                raww=rows[psid]['weights'][vi,lane]
                if raww<=0:continue
                w32=np.float32(c*float(raww));world=lane_world[psid][lane][vi]
                result[i]=np.asarray([np.float32(result[i,k]+np.float32(w32*world[k])) for k in range(3)],dtype='<f4')
    return result
parent_world={sid:parent_rest(sid) for sid in (132,134)}

def candidate_rest(out_sid,fields):
    local,weights=fields[out_sid];binds=np.frombuffer(b''.join(child_bindings[out_sid]),dtype=tissue['bindings'].dtype).copy()
    custom_tissue=dict(tissue);custom_tissue['bindings']=binds
    row=dict(fb=0,bc=len(child_bindings[out_sid]))
    out=np.zeros_like(points[out_sid],dtype='<f4')
    for lane in range(4):
        one=np.zeros_like(weights,dtype='<f4');active=weights[:,lane]>0;one[active,lane]=1.0
        world=h.forward(points[out_sid],local,one,row,custom_tissue,poses)
        for i in range(len(out)):
            wt=weights[i,lane]
            if wt>0:out[i]=np.asarray([np.float32(out[i,k]+np.float32(wt*world[i,k])) for k in range(3)],dtype='<f4')
    return out

strategy_reports={}
for name in ('dominant','top4'):
    fields=make_candidate(name);stats={}
    for sid in (132,134):stats[str(sid)]=vstats(parent_world[sid],candidate_rest(sid,fields))
    strategy_reports[name]=dict(rows=stats,passes_all_rows=all(v['max_um']<=2.0 for v in stats.values()))
selected=next((name for name in ('dominant','top4') if strategy_reports[name]['passes_all_rows']),None)
rest_report=dict(schema='numi.human.left-fds-fdp-054-native-child-registration-preflight.v1',
    status='passed' if selected else 'failed',source_parent_map_sha256=sha(MAN),parent_payload_sha256=sha(T),
    exact_canonical_full_parent_terms_retained=True,shared_seam_vertex_count=len(shared_ids),
    provisional_child_weight_rules=strategy_reports,selected_provisional_child_weight_rule=selected,
    gate='Native parentRest-vs-childRest registration at actual NHRIGID defaultQ; every child vertex must be <=2.0 micrometres. This is preparation evidence, not native GPU admission.',
    method='Exact 054 barycentric references expanded through unchanged raw parent row positive Float32 weights and transforms; binary64 coefficients parsed/serialized as doubles. ChildRest is evaluated using candidate NHTISS row four-lane data and NHRIGID defaultQ body poses with Float32 forward/accumulation.',
    input_sha256={str(p):sha(p) for p in (T,T.with_suffix('.manifest.json'),SOURCE,SOURCE.with_suffix('.manifest.json'),MAN,MAP_REPORT,MAP_V2,MAP_TEMPLATE,STEP0_REPORT,EXACT_CHECK,CONSTRUCTION,CONSTRUCTION_REPORT,STEP0_CAPTURE,STEP0_CAPTURE_RECEIPT,REST_AUDIT,RIG,ARCH,H)})
(OUT/'rest-registration-preflight.json').write_text(json.dumps(rest_report,indent=2,sort_keys=True)+'\n')
log('rest-registration-preflight',selected_rule=selected,strategies=strategy_reports)
req(selected is not None,'no deterministic child four-lane provisional rule meets the fixed native <=2um registration gate')
fields=make_candidate(selected)

# Emit new NHTISS4 child with only 132/134 replaced; all other rows, bindings, raw vertices and faces replay byte-exactly.
raw=T.read_bytes();magic,abi,nr,nb,nv,ni,fp,src_digest=struct.unpack_from('<8s6I32s',raw)
req(magic==b'NHTISS4\0' and abi==5 and nr==parent['surface_count'],'NHTISS parent ABI mismatch')
rec=parent['records'].copy();out_records=rec.copy();all_bind=[];all_vertices=[];all_indices=[]
bind_cur=vert_cur=index_cur=0;old_row_slices={};changed=[]
for ri,record in enumerate(rec):
    sid=int(record[6]);old=pc._row_slices(parent,record);old_row_slices[sid]=old
    if sid in (132,134):
        v6=np.column_stack((points[sid],h.vertex_normals(points[sid],faces[sid]))).astype('<f4')
        local,weights=fields[sid]
        n=len(v6);vb=bytearray(n*56)
        np.ndarray((n,6),dtype='<f4',buffer=vb,offset=0,strides=(56,4))[:]=v6
        np.ndarray((n,4),dtype='<u4',buffer=vb,offset=24,strides=(56,4))[:]=local
        np.ndarray((n,4),dtype='<f4',buffer=vb,offset=40,strides=(56,4))[:]=weights
        vb=bytes(vb);f=faces[sid].astype('<u4');bind_bytes=b''.join(child_bindings[sid])
        changed.append(dict(stable_id=sid,vertex_count=len(v6),triangle_count=len(f),binding_count=len(child_bindings[sid]),
            parent_source_rows=sorted({int(x[1]) for rr in terms[sid] for x in rr}),
            candidate_binding_rule=selected,provisional_only_for_child_rest_registration=True))
    else:
        n=int(record[3]);vb=old['vertex_bytes'];f=np.asarray(old['local_faces'],dtype='<u4');
        start=64+nr*32+int(record[0])*36;count=int(record[1]);bind_bytes=raw[start:start+count*36]
    out_records[ri,0]=bind_cur;out_records[ri,1]=len(bind_bytes)//36
    out_records[ri,2]=vert_cur;out_records[ri,3]=n
    out_records[ri,4]=index_cur;out_records[ri,5]=f.size
    all_bind.append(bind_bytes);all_vertices.append(vb);all_indices.append((f+vert_cur).astype('<u4').tobytes())
    bind_cur+=len(bind_bytes)//36;vert_cur+=n;index_cur+=f.size
out=bytearray(raw[:64]);out[64:64+nr*32]=out_records.astype('<u4').tobytes();struct.pack_into('<III',out,16,bind_cur,vert_cur,index_cur)
out.extend(b''.join(all_bind));out.extend(b''.join(all_vertices));out.extend(b''.join(all_indices))
child_payload=OUT/'assets'/T.name;child_payload.write_bytes(out)
manifest=json.loads(T.with_suffix('.manifest.json').read_text())
manifest['payload'].update(file=child_payload.name,sha256=sha(child_payload),bytes=len(out),binding_count=bind_cur,vertex_count=vert_cur,index_count=index_cur)
for row in manifest['source']['surfaces']:
    if int(row['stable_id']) in (132,134):
        sid=int(row['stable_id']);row['vertex_count']=len(points[sid]);row['triangle_count']=len(faces[sid])
manifest['source']['experimental_left_fds_fdp_054_source_parent_map']={
    'schema':'numi.human.passive-tissue-parent-map-template.v1',
    'map_path':str(MAP_V2),'map_sha256':sha(MAP_V2),'map_report_path':str(STEP0_REPORT),'map_report_sha256':sha(STEP0_REPORT),
    'exact_check_path':str(EXACT_CHECK),'exact_check_sha256':sha(EXACT_CHECK),
    'candidate_path':str(CONSTRUCTION),'candidate_sha256':sha(CONSTRUCTION),
    'construction_report_path':str(CONSTRUCTION_REPORT),'construction_report_sha256':sha(CONSTRUCTION_REPORT),
    'original_054_manifest_path':str(MAN),'original_054_manifest_sha256':sha(MAN),
    'original_source_payload_path':str(SOURCE),'original_source_payload_sha256':sha(SOURCE),
    'direct_parent_payload_path':str(T),'direct_parent_payload_sha256':sha(T),
    'original_source_rows_132_134_equal_current_parent_rows':True,
    'changed_stable_ids':[132,134],'raw_parent_source_rows_132_134_exact':True,
    'full_parent_terms_retained_without_four_lane_reduction':True,
    '060_candidate_exact_union_certificate_path':str(EXACT_CHECK),
    '060_step0_actual_native_remesh_report_path':str(STEP0_REPORT),
    'canonical_terms':'sort by source index, raw stable ID, raw source vertex index; omit zero barycentric coefficients; coefficients serialized as binary64 JSON numbers',
    'shared_seam_terms_identical_for_both_child_rows':True,'shared_seam_vertex_count':len(shared_ids),
    'child_nhtiss_four_lane_fields':'deterministic provisional source/dominant or top4 fields used only by native childRest registration gate; full parent extension owns runtime motion',
    'provisional_child_weight_rule':selected,'rest_registration_preflight_path':str(OUT/'rest-registration-preflight.json'),
    'rest_registration_preflight_sha256':sha(OUT/'rest-registration-preflight.json'),
    'all_physical_routes_mass_tendons_support_physiology_and_non132_134_rows_unchanged':True,
    'qualification':'Experimental native diagnostic preparation only; no GPU run performed, no anatomical admission.'}
manifest['evidence_boundary']='Experimental forearm054 source-parent native diagnostic. The 054 exact parent map is retained intact. Child four-lane NHTISS fields are provisional solely for source registration; this preparation is not anatomical admission or native-cycle evidence.'
manifest['payload']['sha256']=sha(child_payload)
child_manifest=child_payload.with_suffix('.manifest.json');child_manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')

# Replay exact untouched rows and verify child maps before creating the receipt/run declaration.
child_data=pc._read_nhtiss4(child_payload)
for sid in (132,134):
    row=next(r for r in child_data['records'] if int(r[6])==sid);sl=pc._row_slices(child_data,row)
    req(np.array_equal(np.asarray(sl['local_faces'],dtype='<u4'),faces[sid]),f'child faces differ from 054 row {sid}')
    got=np.ndarray((int(row[3]),6),dtype='<f4',buffer=sl['vertex_bytes'],offset=0,strides=(56,4))
    req(np.array_equal(got[:,:3],points[sid]),f'child positions differ from 054 row {sid}')
    req(len(terms[sid])==int(row[3]),f'parent map does not cover every child vertex row {sid}')
for sid in (132,134):
    a=next(r for r in parent['records'] if int(r[6])==sid);b=next(r for r in child_data['records'] if int(r[6])==sid)
    # Original parent payload remains untouched and is the extension source; candidate source rows remain separately pinned.
    assert int(a[6])==int(b[6])
untouched=[]
for sid in [int(r[6]) for r in rec if int(r[6]) not in (132,134)]:
    a=old_row_slices[sid];b=pc._row_slices(child_data,next(r for r in child_data['records'] if int(r[6])==sid))
    req(pc._row_local_equal(a,b),f'non-target row {sid} changed during native diagnostic composition');untouched.append(sid)

# Copy the parent receipt and bind the experimental TISS; the immutable parent TISS remains the source of raw132/134.
parent_receipt=PARENT/'resting-anatomy-receipt.json';receipt=json.loads(parent_receipt.read_text())
common=receipt['provenance']['cardiac_geometry_binding']['common_field']
for owner in ('map','polynomials','domain_boxes'):
    record=common[owner];relative=Path(record['path'])
    req(not relative.is_absolute(),f'parent cardiac common path must stay receipt-relative: {owner}')
    original=PARENT/relative;dest=OUT/relative
    req(original.is_file() and sha(original)==record['sha256'],f'parent cardiac common asset mismatch {owner}')
    req(not dest.exists(),f'refusing to replace staged common asset {dest}')
    os.link(original,dest)
receipt['provenance']['native_muscle_surfaces'].update(payload_path=str(child_payload),sha256=sha(child_payload),
    manifest_path=str(child_manifest),manifest_sha256=sha(child_manifest))
receipt['passive_tissue_parent_bindings']={
    'child_payload_sha256':sha(child_payload),
    'source_payloads':[dict(path=str(T),sha256=sha(T))],
    'surfaces':[dict(stable_id=sid,vertices=terms[sid]) for sid in (132,134)]}
receipt['experimental_left_fds_fdp_054_source_parent_native_diagnostic']={
    'status':'prepared_not_admitted','map_path':str(MAP_V2),'map_sha256':sha(MAP_V2),
    'original_054_manifest_path':str(MAN),'original_054_manifest_sha256':sha(MAN),
    'original_map_source_payload_path':str(SOURCE),'original_map_source_payload_sha256':sha(SOURCE),
    'loaded_parent_payload_path':str(T),'loaded_parent_payload_sha256':sha(T),
    'raw_132_134_rows_equal_original_map_source':True,
    'candidate_path':str(CONSTRUCTION),'candidate_sha256':sha(CONSTRUCTION),
    'construction_report_path':str(CONSTRUCTION_REPORT),'construction_report_sha256':sha(CONSTRUCTION_REPORT),
    'max_actual_displacement_um':float(construction_report['max_actual_displacement_m'])*1e6,
    'declared_displacement_bound_um':float(construction_report['cumulative_displacement_bound_m'])*1e6,
    'step0_remesh_report_path':str(STEP0_REPORT),'step0_remesh_report_sha256':sha(STEP0_REPORT),
    'exact_union_check_path':str(EXACT_CHECK),'exact_union_check_sha256':sha(EXACT_CHECK),
    'changed_child_stable_ids':[132,134],'full_parent_terms_retained':True,
    'seam_terms_canonical_binary64_and_identical':True,'provisional_child_weight_rule':selected,
    'rest_registration_preflight_path':str(OUT/'rest-registration-preflight.json'),
    'rest_registration_preflight_sha256':sha(OUT/'rest-registration-preflight.json'),
    'gpu_run_not_performed':True,'anatomical_admission_not_claimed':True}
req(receipt['mass_geometry_accounting']==json.loads((PARENT/'resting-anatomy-receipt.json').read_text())['mass_geometry_accounting'],'mass accounting changed')
receipt_path=OUT/'resting-anatomy-receipt.json';receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')

# Reuse the passing164002 accepted declaration, replacing only output/receipt and selecting build004.
base_decl=json.loads((PARENT/'baseline/run-declaration.json').read_text());argv=list(base_decl['argv']);cmd_i=argv.index('numilab_human.resting_run');av=argv[cmd_i+1:]
for flag,val in [('--output',str(OUT/'baseline/native-run')),('--anatomy-receipt',str(receipt_path)),('--build',str(B)),('--lab',str(W))]:
    av[av.index(flag)+1]=val
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native_argv,assets=rr.command(args)
env=dict(x.split('=',1) for x in argv[2:cmd_i-2])
env.update(NUMI_BUILD_DIR=str(B),NUMI_LAB_ROOT=str(W),
           NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(OUT/'baseline/native-run/common-field-failure.json'),
           NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS='0,3807,4767,5023,5599,6207,7423,8000')
env['DYLD_LIBRARY_PATH']=':'.join(['/Users/n/numi-lab-complete-body-capture-build-1276-001/lib','/Users/n/numi-lab-complete-body-capture-build-1276-001/matter'])
env['DYLD_LIBRARY_PATH']=':'.join(str(Path(p).resolve()) for p in env['DYLD_LIBRARY_PATH'].split(':'))
runtime_rr=Path(env['PYTHONPATH'])/'numilab_human/resting_run.py'
req(sha(runtime_rr)==sha(Path(rr.__file__)),'runtime resting_run differs from prepared CLI owner')
argv2=['/usr/bin/env','-i']+[k+'='+v for k,v in sorted(env.items())]+['/Users/n/numi-human-prep-venv-20261005/bin/python3.13','-m','numilab_human.resting_run']+av
launch=OUT/'launch_arm.py';shutil.copyfile(PARENT_LAUNCH,launch)
proof=OUT/'preparation.json'
inputs=[Path(__file__),T,T.with_suffix('.manifest.json'),SOURCE,SOURCE.with_suffix('.manifest.json'),MAN,MAP_REPORT,MAP_V2,MAP_TEMPLATE,STEP0_REPORT,EXACT_CHECK,CONSTRUCTION,CONSTRUCTION_REPORT,STEP0_CAPTURE,STEP0_CAPTURE_RECEIPT,REST_AUDIT,RIG,ARCH,H,
        BIN,BUILD_REPORT,PARENT_PREP,PARENT_VERIFY,PARENT_LAUNCH,parent_receipt,PARENT/'baseline/run-declaration.json',
        P/'src/numilab_human/passive_attachment_composition.py',Path(rr.__file__),runtime_rr,
        W/'apps/NumiHumanRestingTissueParents.hpp',W/'apps/NumiHumanRestingVisual.hpp',W/'apps/numilab_human_myosim_visual_probe.mm',
        B/'build-report.json',BUILD_REPORT,child_payload,child_manifest,receipt_path,
        OUT/'rest-registration-preflight.json',launch]
pins={str(p):sha(p) for p in inputs}
req(pins[str(BIN)]=='147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174','wrong build004 binary in final pins')
req(any(Path(k).name=='numi-human-native' and v=='147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174' for k,v in assets.items()),'resting_run CLI did not select required native binary')
allpins={**assets,**pins}
decl=dict(schema='numi.human.native-smoke.run-declaration.v1',seconds=16.,accepted_steps=8000,
    capture_steps=[0,3807,4767,5023,5599,6207,7423,8000],immutable_assets=allpins,argv=argv2,
    launch_adapter=dict(path=str(launch),sha256=sha(launch)),owner_cli_preview=dict(native_argv=native_argv,
    asset_sha256=assets,recorded_invocation_environment=rr.invocation_environment(env)),execution_host='numi',
    hardware='Apple M6,12 GPU cores,16 GiB,macOS 27.0',qualification='Experimental candidate060 native diagnostic only. Exact 060 source-parent terms on stable132/134, current temporalis164002 parent, unchanged physical physiology; the capture grid includes known failure step3807 and omits prior diagnostic step6815. Native GPU run and all-pose admission are pending. This is not anatomical admission.')
dp=OUT/'baseline/run-declaration.json';dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+'\n')
(OUT/'launch-guard.json').write_text(json.dumps(dict(launcher_sha256=sha(launch),arms=dict(baseline=dict(run_declaration_sha256=sha(dp),native_output=str(OUT/'baseline/native-run')))),indent=2,sort_keys=True)+'\n')
(OUT/'launch-command.txt').write_text(f'/Users/n/numi-human-prep-venv-20261005/bin/python3.13 {launch} baseline\n')
prep=dict(schema='numi.human.left-fds-fdp-candidate060-native-diagnostic-preparation.v1',status='prepared_experimental_not_admitted',
    source_parent_payload_sha256=sha(T),original_054_source_payload_sha256=sha(SOURCE),
    loaded_parent_source_rows_132_134_exact_match_original=True,
    candidate060_path=str(CONSTRUCTION),candidate060_sha256=sha(CONSTRUCTION),construction_report_path=str(CONSTRUCTION_REPORT),construction_report_sha256=sha(CONSTRUCTION_REPORT),
    actual_max_displacement_um=float(construction_report['max_actual_displacement_m'])*1e6,declared_displacement_bound_um=float(construction_report['cumulative_displacement_bound_m'])*1e6,
    step0_capture_path=str(STEP0_CAPTURE),step0_capture_sha256=sha(STEP0_CAPTURE),step0_capture_receipt_sha256=sha(STEP0_CAPTURE_RECEIPT),map_template_path=str(MAP_V2),map_template_sha256=sha(MAP_V2),
    exact_union_check_path=str(EXACT_CHECK),exact_union_check_sha256=sha(EXACT_CHECK),step0_remesh_report_path=str(STEP0_REPORT),step0_remesh_report_sha256=sha(STEP0_REPORT),candidate_payload_path=str(child_payload),candidate_payload_sha256=sha(child_payload),
    candidate_manifest_path=str(child_manifest),candidate_manifest_sha256=sha(child_manifest),receipt_path=str(receipt_path),receipt_sha256=sha(receipt_path),
    map_path=str(MAP_V2),map_sha256=sha(MAP_V2),original_054_map_path=str(MAN),original_054_map_sha256=sha(MAN),map_report_path=str(STEP0_REPORT),map_report_sha256=sha(STEP0_REPORT),
    rest_registration_path=str(OUT/'rest-registration-preflight.json'),rest_registration_sha256=sha(OUT/'rest-registration-preflight.json'),
    provisional_child_weight_rule=selected,canonical_full_parent_terms=True,shared_seam_vertex_count=len(shared_ids),capture_steps=[0,3807,4767,5023,5599,6207,7423,8000],
    changed_rows=changed,unchanged_non_target_rows_exact=len(untouched),mass_geometry_accounting_parent_exact=True,
    native_binary_path=str(BIN),native_binary_sha256=sha(BIN),run_declaration_path=str(dp),run_declaration_sha256=sha(dp),
    launch_command=(OUT/'launch-command.txt').read_text().strip(),gpu_run_performed=False,all25_screen_required_for_this_experimental_diagnostic=False,
    input_sha256=pins,created_files=[str(child_payload),str(child_manifest),str(receipt_path),str(OUT/'rest-registration-preflight.json'),str(dp),str(OUT/'launch-guard.json'),str(launch)])
proof.write_text(json.dumps(prep,indent=2,sort_keys=True)+'\n')
log('preparation_complete',payload_sha256=sha(child_payload),manifest_sha256=sha(child_manifest),receipt_sha256=sha(receipt_path),declaration_sha256=sha(dp),launch_command=prep['launch_command'])
