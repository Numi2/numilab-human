#!/usr/bin/env python3
"""Canonicalize run055 I50/D264 outputs against Skin571/Fat573 parent rows."""
from pathlib import Path
from collections import Counter
import argparse, gc, hashlib, importlib.util, json, sys, time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
R=H/'paired-I-local-cavity-527-root'
P=R/'run055-current-I50-D264-source-I-baseline-context'
IROOT=R/'run051-current-canonical-I-measured-quotient-transfer/output-001'
SKIN_BASE=H/'skin-canonical-paired-I-current-571-root/skin-current-paired-I-canonical-volume-proposal.npz'
SKIN_BASE_RECEIPT=H/'skin-canonical-paired-I-current-571-root/receipt.json'
FAT_BASE=H/'fat-canonical-composed-DI-current-573-root/fat-current-composed-DI-canonical-volume-proposal.npz'
FAT_BASE_RECEIPT=H/'fat-canonical-composed-DI-current-573-root/receipt.json'
I_REPORT=IROOT/'report.json'
I_CANDIDATE=R/'run052-current-I-batch-typed-replay/typed-replay-006/candidate-I-typed-current.npz'
I_RECEIPT=R/'run052-current-I-batch-typed-replay/typed-replay-006/receipt.json'
D_REPORT=IROOT/'D411-indexed379-transfer-001/report.json'
D_CANDIDATE=IROOT/'D411-indexed379-transfer-001/fat-D411-on-I50-indexed379-volume-proposal.npz'
D_TYPED=IROOT/'D411-indexed379-transfer-001/typed-bind-003/candidate-D-typed-current.npz'
D_697=IROOT/'D411-indexed379-transfer-001/typed-bind-003/actual-697-001/actual-697-D411-I50-load-001.json'
CTX=P/'audit354/report.json'
SKIN_BOUND=P/'skin-target369-v2/skin-bound-current-I-target-volume-proposal.npz'
SKIN_BOUND_RECEIPT=P/'skin-target369-v2/receipt.json'
FAT_BOUND=P/'fat-target397-v2/fat-current-typed-target-rebound.npz'
FAT_BOUND_REPORT=P/'fat-target397-v2/report.json'
FAT_BOUND_RECEIPT=P/'fat-target397-v2/completion-receipt.json'
SKIN_I_SHA='b85f31bd0754d30614f4aafa94db12f5e4b6d67f8407c3c2e35b703762c37367'
FAT_I_SHA='2420a70b1fea86d6aa6ccf0dabd61d82c259f5e9ce382fc73b1f617debb8ae8e'
I_CAN_SHA='e7d76ae905c15ae3ad5c5dda7603cfdd18399934c12721b942fc4903571970a6'
I_RECEIPT_SHA='8893828313549c3ba63549e92c3d37b85a63367bd283a9a5c05af9c1c2917a78'
D_CAN_SHA='2b2d6aa8ea047dc3a87e5c4f44b9e4d6a4b38a201e707bd51af792861183ffd0'
D_697_SHA='14909a88197a4b8d30c7ba4ffd34fefe95c887aec75e0a65f8377bb91ff7d5d8'
CTX_SHA='d2662598b2347a4a3dfece39942afc6a1c154da1f3073090e4b85fa8388ae5a6'
SKIN_BOUND_SHA='624c31c98b25c371432d8a2aeaf74810ce9d0978ae03dccbf438e34063d2e4dc'
SKIN_BOUND_RECEIPT_SHA='1a1bfe91b73cc275e1ee04458c988ccdedff8d3d702499185ba5d1a5f2c77c15'
FAT_BOUND_SHA='ec49abcb75dcb3b6b584db80ade90d5ebb85bab24df84aa5638021265fc0ae65'
FAT_BOUND_REPORT_SHA='204ec614f6d39902b8e00be49b4ade37a38e13ebeff4d74b48e5a25d0b18b9ee'
FAT_BOUND_RECEIPT_SHA='1c0bbcf507ce70d51f792233d8f86c1a0ad4376e4eef402bf01d91833d7410bc'
D_VOLUME_SHA='773f7613291fc9c2b701ec76e3182bf16ee17f7b7aef0dbf06c93f99d5b47caa'
I_REPORT_SHA='cca39040cb6b4053b345a65786ab9cf468cbf7198fd46f66093a47c52c0fa709'
D_REPORT_SHA='59384762b0be315f5ea2dabe6a175f1372f11b5b7030700f164abb421954d416'
SKIN_BASE_SHA='4f8199854803eef6136ee80c940d75b9f294053c3ef70cb7a5ffb81839ee266b'
FAT_BASE_SHA='5ec64f7ca176d0acb8bbb2eb46842f45b8f4b02744bde0b8c77bfeef3666e86f'
SKIN_BASE_RECEIPT_SHA='e4200798c52c282d60b6af1960eb378e6315b5db7fd87d254e3d7b899d50a2c6'
FAT_BASE_RECEIPT_SHA='d173cb0338007c5e16217744e26dc5f19813d866a294f7e6547d07e6c362acc2'
OWNER571=H/'canonicalize_paired_I_skin_current_571_root.py'
OWNER573=H/'canonicalize_composed_DI_fat_current_573_root.py'
OWNER585=H/'fixed-boundary-composer-587-root'
OUT=P/'canonical-parent-unions-v3'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(path):return {'path':str(path),'sha256':sha(path),'size_bytes':Path(path).stat().st_size}
def read_json(path):return json.loads(Path(path).read_text())
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def load_mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def counter_faces(T,faces):return Counter(map(tuple,T.boundary_identity(np.asarray(faces,dtype=np.int64).reshape(-1,3))))
def f32bits(a):return np.ascontiguousarray(a,dtype=np.float32).view(np.uint32)
def incidence_csr(t,npoints):
    t=np.asarray(t,dtype=np.int64)
    cells=np.repeat(np.arange(len(t),dtype=np.int64),4)
    nodes=t.reshape(-1)
    order=np.lexsort((cells,nodes))
    counts=np.bincount(nodes,minlength=npoints)
    offsets=np.empty(npoints+1,dtype=np.int64);offsets[0]=0;np.cumsum(counts,out=offsets[1:])
    return offsets,cells[order]
def full_star(csr,nodes):
    offsets,cells=csr
    pieces=[cells[offsets[int(v)]:offsets[int(v)+1]] for v in nodes]
    pieces=[x for x in pieces if len(x)]
    if not pieces:return np.empty(0,dtype=np.int64)
    return np.unique(np.concatenate(pieces))
class DSU:
    def __init__(self,n):self.p=list(range(n))
    def find(self,x):
        x=int(x)
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)
def make_parent_unions(ncells,starsets,changed_origin):
    dsu=DSU(len(starsets));row_owner=np.full(ncells,-1,dtype=np.int64)
    total_star_rows=0
    for k,rows in enumerate(starsets):
        rows=np.asarray(rows,dtype=np.int64)
        if len(rows)==0 or np.any(rows<0) or np.any(rows>=ncells):raise ValueError('empty/out-of-domain endpoint star')
        total_star_rows+=len(rows)
        for row in rows:
            old=int(row_owner[int(row)])
            if old>=0:dsu.union(k,old)
            else:row_owner[int(row)]=k
    groups={}
    for k,rows in enumerate(starsets):groups.setdefault(dsu.find(k),set()).update(map(int,rows))
    group_arrays={k:np.asarray(sorted(v),dtype=np.int64) for k,v in groups.items()}
    offsets=[0];parents=[];changed_roots=set()
    for row in changed_origin:
        op=int(row_owner[int(row)])
        if op<0:raise ValueError(f'changed base row {int(row)} is outside all full endpoint stars')
        root=dsu.find(op);changed_roots.add(root);parents.extend(map(int,group_arrays[root]));offsets.append(len(parents))
    return row_owner,np.asarray(offsets,dtype=np.int64),np.asarray(parents,dtype=np.int64),len(groups),len(changed_roots),total_star_rows

def common_pins():
    expected={
      SKIN_BASE:SKIN_BASE_SHA,SKIN_BASE_RECEIPT:SKIN_BASE_RECEIPT_SHA,FAT_BASE:FAT_BASE_SHA,FAT_BASE_RECEIPT:FAT_BASE_RECEIPT_SHA,
      I_REPORT:I_REPORT_SHA,I_CANDIDATE:I_CAN_SHA,I_RECEIPT:I_RECEIPT_SHA,D_REPORT:D_REPORT_SHA,D_CANDIDATE:D_VOLUME_SHA,
      D_TYPED:D_CAN_SHA,D_697:D_697_SHA,CTX:CTX_SHA,SKIN_BOUND:SKIN_BOUND_SHA,SKIN_BOUND_RECEIPT:SKIN_BOUND_RECEIPT_SHA,
      FAT_BOUND:FAT_BOUND_SHA,FAT_BOUND_REPORT:FAT_BOUND_REPORT_SHA,FAT_BOUND_RECEIPT:FAT_BOUND_RECEIPT_SHA}
    for p,want in expected.items():
        got=sha(p)
        if got!=want:raise ValueError(f'input pin mismatch {p}: {got} != {want}')
    ir=read_json(I_REPORT);itr=read_json(I_RECEIPT);dr=read_json(D_REPORT);d697=read_json(D_697);ctx=read_json(CTX)
    sbr=read_json(SKIN_BOUND_RECEIPT);fbr=read_json(FAT_BOUND_REPORT);fcr=read_json(FAT_BOUND_RECEIPT)
    sbase=read_json(SKIN_BASE_RECEIPT);fbase=read_json(FAT_BASE_RECEIPT)
    if sbase.get('status')!='PASS_CANONICAL_CURRENT_PAIRED_I_SKIN_PARENT_UNIONS_AND_TARGET' or sbase.get('candidate_sha256')!=SKIN_BASE_SHA:raise ValueError('Skin571 base receipt mismatch')
    if fbase.get('status')!='PASS_CANONICAL_COMPOSED_D_I_FAT_PARENT_UNIONS_AND_TARGET' or fbase.get('candidate_sha256')!=FAT_BASE_SHA:raise ValueError('Fat573 base receipt mismatch')
    if ir.get('status')!='PASS_PAIRED_I_QUOTIENT_PHYSICAL_PREFLIGHT' or itr.get('status')!='PASS_PAIRED_I_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY':raise ValueError('I50 report/typed status mismatch')
    if dr.get('status')!='PASS_D411_INDEXED379_VOLUME_QUOTIENT_PROPOSAL' or dr.get('output_volume_candidate',{}).get('sha256')!=D_VOLUME_SHA:raise ValueError('D264 transfer report mismatch')
    if d697.get('physical_admission') is not False or d697.get('status') not in ('PASS_ACTUAL_697_217_TYPED_D_LOAD','PASS_ACTUAL_697_TYPED_D_LOADER_METADATA_VIEW'):raise ValueError('D264 697 receipt mismatch')
    if ctx.get('status')!='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185':raise ValueError('combined354 status mismatch')
    if ctx['changed_vs_complete_retained_OID_both_walls_all185']['count']!=0 or ctx['changed_OID_joint_self_pairs']['count']!=0:raise ValueError('combined354 collision count is nonzero')
    if sbr.get('status')!='PASS_EXACT_I_QUOTIENT_PHYSICAL_TARGET_BIND_ONLY' or sbr.get('candidate_sha256')!=SKIN_BOUND_SHA:raise ValueError('Skin369 receipt mismatch')
    if fbr.get('geometry_changed') is not False or not fbr.get('actual_vs_typed_target',{}).get('boundary_complete') or fbr['actual_vs_typed_target']['missing_oriented_records']!=0 or fbr['actual_vs_typed_target']['extra_oriented_records']!=0:raise ValueError('Fat397 exact target proof mismatch')
    if fbr.get('flux',{}).get('target_minus_actual_det6')!=0 or fcr.get('status')!='PASS_CURRENT_FAT_TYPED_TARGET_REBOUND_COMPLETE_SOURCE_ONLY':raise ValueError('Fat397 exact flux/completion receipt mismatch')
    if Path(sbr['exact_context_report']).resolve()!=CTX.resolve() or sbr['exact_context_report_sha256']!=sha(CTX):raise ValueError('Skin369 context pin mismatch')
    if fbr['inputs']['typed_I']['sha256']!=I_CAN_SHA or fbr['inputs']['typed_D']['sha256']!=D_CAN_SHA or fbr['inputs']['base_candidate']['sha256']!=D_VOLUME_SHA:raise ValueError('Fat397 typed/base inputs mismatch')
    cfgp=Path(ctx['config']['path']);cfg=json.loads(cfgp.read_text())
    if sha(cfgp)!=ctx['config']['sha256']:raise ValueError('context354 config pin mismatch')
    surfaces=cfg['candidate_surfaces']
    for role,e in surfaces.items():
        if sha(e['path'])!=e['sha256']:raise ValueError(f'context source {role} pin mismatch')
    if surfaces['I']['sha256']!=I_CAN_SHA or surfaces['D']['sha256']!=D_CAN_SHA:raise ValueError('context I/D candidate mismatch')
    return ir,itr,dr,d697,ctx,sbr,fbr,fcr,surfaces

def canonical_skin(ir,itr,ctx,surfaces,T,outdir):
    role='Skin';rr=next(x for x in ir['roles'] if x['role']==role);tr=next(x for x in itr['roles'] if x['role']==role)
    basep=Path(rr['input']);rawp=Path(rr['output']);boundp=SKIN_BOUND
    base=load_npz(basep);raw=load_npz(rawp);bound=load_npz(boundp)
    if sha(basep)!=SKIN_BASE_SHA or sha(rawp)!=SKIN_I_SHA or sha(boundp)!=SKIN_BOUND_SHA:raise ValueError('Skin geometry path hash changed')
    for k in ('input','input_sha256','output','output_sha256'):
        if tr[k]!=rr[k]:raise ValueError(f'Skin typed/report pin differs for {k}')
    if tr['bad_before']!=rr['global_native_bad_before'] or tr['bad_after']!=rr['global_native_bad_after']:raise ValueError('Skin typed/report quality count differs')
    p=np.asarray(base['points_world_m'],dtype=np.float32);ot=np.asarray(base['tetrahedra'],dtype=np.int64);om=np.asarray(base['material_indices'],dtype=np.int32)
    if not np.array_equal(f32bits(p),f32bits(base['points_world_m_f32'])):raise ValueError('Skin base point views differ')
    for k in ('points_world_m','points_world_m_f32','tetrahedra','material_indices'):
        if not np.array_equal(raw[k],bound[k]):raise ValueError(f'Skin bound geometry differs from I50 output: {k}')
    q=np.arange(len(p),dtype=np.int64);csr=incidence_csr(ot,len(p));ops=rr['operations'];stars=[]
    if len(ops)!=50 or tr['operations_replayed']!=50 or tr['forward_reverse_replay_exact'] is not True:raise ValueError('Skin I50 replay evidence mismatch')
    for op in ops:
        drop=int(op['retired_volume_node']);keep=int(op['kept_volume_node']);reported=np.asarray(op['source_cell_rows'],dtype=np.int64)
        if drop==keep or q[drop]!=drop:raise ValueError('dependent Skin I operation order')
        if not np.array_equal(np.sort(reported),full_star(csr,[drop])):raise ValueError('Skin I report source star is not exact retired-node star')
        full=full_star(csr,[drop,keep]);stars.append(full)
        q[q==drop]=q[keep]
    if not np.array_equal(q,np.asarray(raw['source_I_vertex_quotient'],dtype=np.int64)):raise ValueError('Skin I volume node quotient differs from raw output')
    mapped=q[ot];sorted4=np.sort(mapped,axis=1);deg=np.any(np.diff(sorted4,axis=1)==0,axis=1);origin=np.flatnonzero(~deg);t=mapped[origin];mat=om[origin]
    if not np.array_equal(t,raw['tetrahedra']) or not np.array_equal(mat,raw['material_indices']):raise ValueError('Skin I output order/material replay mismatch')
    if not np.array_equal(origin,raw['output_cell_to_input_cell']):raise ValueError('Skin I output→base row map mismatch')
    reverse=np.full(len(ot),-1,dtype=np.int64);reverse[origin]=np.arange(len(origin),dtype=np.int64)
    if not np.array_equal(reverse,raw['input_cell_to_output_cell']):raise ValueError('Skin I reverse cell map mismatch')
    face=q[np.asarray(base['desired_boundary_faces'],dtype=np.int64)];fs=np.sort(face,axis=1);face=face[~np.any(np.diff(fs,axis=1)==0,axis=1)]
    if counter_faces(T,face)!=counter_faces(T,bound['desired_boundary_faces']):raise ValueError('Skin target not exact quotient of Skin571 target')
    if not np.array_equal(p,bound['points_world_m']) or not np.array_equal(t,bound['tetrahedra']):raise ValueError('Skin canonical geometry not exact bound output')
    bd=T.boundary(t)
    if counter_faces(T,bd)!=counter_faces(T,bound['boundary_faces']) or counter_faces(T,bd)!=counter_faces(T,bound['desired_boundary_faces']):raise ValueError('Skin full current boundary/target mismatch')
    det,qual=T.m.measurements(p,t);bad=int(np.sum(~T.m.passes(det,qual)))
    if np.any(det<=0) or bad!=rr['global_native_bad_after']:raise ValueError('Skin global positivity/quality replay mismatch')
    changed=np.any(t!=ot[origin],axis=1);changed_out=np.flatnonzero(changed);changed_origin=origin[changed]
    rowowner,po,pr,ngroups,nchangedgroups,totalstar=make_parent_unions(len(ot),stars,changed_origin)
    out_to_in=origin.copy();out_to_in[changed]=-1
    in_to_out=np.full(len(ot),-1,dtype=np.int64);in_to_out[origin[~changed]]=np.flatnonzero(~changed)
    if np.any(in_to_out[origin[changed]]!=-1):raise ValueError('changed Skin parent cell unexpectedly has retained output identity')
    payload=dict(base)
    target_keys=('boundary_faces','desired_boundary_faces')
    for k in target_keys:payload[k]=bound[k]
    payload['points_world_m']=p;payload['points_world_m_f32']=p.copy();payload['tetrahedra']=t;payload['material_indices']=mat
    payload.update(current_geometry_parent_path=np.asarray([str(basep)]),current_geometry_parent_sha256=np.asarray([sha(basep)]),
      output_cell_to_input_cell=out_to_in,input_cell_to_output_cell=in_to_out,retained_input_cell_rows=origin[~changed],
      replacement_parent_offsets=po,replacement_parent_cell_rows=pr,replacement_tet_offsets=np.asarray([0,len(changed_out)],np.int64),
      replacement_tetrahedra=t[changed],input_node_to_output_node=q,output_node_to_input_node=np.arange(len(p),dtype=np.int64),
      output_point_to_input_point_row=np.arange(len(p),dtype=np.int64),output_point_parent_offsets=np.zeros(len(p)+1,np.int64),
      output_point_parent_input_cell_rows=np.empty(0,np.int64),source_I_vertex_quotient=q,
      canonical_parent_union_schema=np.asarray('numi.run055.current-immediate-parent-unions.v1'),
      canonical_parent_union_operation_order=np.asarray('I50'))
    for source_role,e in surfaces.items():
        payload['typed_'+source_role+'_candidate_path']=np.asarray([e['path']]);payload['typed_'+source_role+'_candidate_sha256']=np.asarray([e['sha256']])
    # Do not carry the bound producer's intermediate-only source-cell operation tag.
    payload.pop('source_cell_operation_rows',None)
    outdir.mkdir(parents=True,exist_ok=False);cp=outdir/'skin-current-I50-canonical-parent-unions.npz';np.savez_compressed(cp,**payload)
    written=load_npz(cp)
    if set(written)!=set(payload) or any(not np.array_equal(written[k],v) for k,v in payload.items()):raise ValueError('Skin canonical output serialization readback mismatch')
    for k in ('points_world_m','points_world_m_f32','tetrahedra','material_indices','boundary_faces','desired_boundary_faces'):
        if not np.array_equal(written[k],bound[k]):raise ValueError(f'Skin canonical output changed bound array {k}')
    return {'candidate':pin(cp),'base':pin(basep),'I50_raw_output':pin(rawp),'bound_target':pin(boundp),'cells':len(t),'input_cells':len(ot),'removed_cells':int(len(ot)-len(t)),'native_quality_failures':bad,'nonpositive':0,'boundary_faces':len(bd),'target_faces':len(bound['desired_boundary_faces']),'missing_target_faces':0,'extra_actual_faces':0,'changed_output_cells':int(changed.sum()),'full_endpoint_star_rows_sum':int(totalstar),'endpoint_star_union_rows':int(len(np.unique(np.concatenate(stars)))),'parent_union_components':int(ngroups),'components_used_by_changed_cells':int(nchangedgroups),'replacement_parent_row_entries':int(len(pr)),'operations':len(ops),'all_point_coordinate_bits_fixed':True,'output_cell_order_exact':True,'unrelated_retained_cell_identity_exact':True,'target_quotient_exact':True,'source330_mass':'IMMUTABLE_CURRENT_MAPPING_REQUIRED'}

def canonical_fat(ir,dr,ctx,surfaces,T,outdir):
 role='Fat';irr=next(x for x in ir['roles'] if x['role']==role)
 basep=Path(irr['input']);ip=Path(irr['output']);dp=D_CANDIDATE;boundp=FAT_BOUND
 base=load_npz(basep);iout=load_npz(ip);dout=load_npz(dp);bound=load_npz(boundp)
 if sha(basep)!=FAT_BASE_SHA or sha(ip)!=FAT_I_SHA or sha(dp)!=D_VOLUME_SHA or sha(boundp)!=FAT_BOUND_SHA:raise ValueError('Fat geometry path hash changed')
 if dr['input_Fat573']['path']!=str(basep) or dr['input_Fat573']['sha256']!=sha(basep):raise ValueError('D report base is not canonical Fat573')
 if dr['input_I50_Fat']['path']!=str(ip) or dr['input_I50_Fat']['sha256']!=sha(ip):raise ValueError('D report is not based on I50-first Fat')
 if dr['output_volume_candidate']['path']!=str(dp) or dr['output_volume_candidate']['sha256']!=sha(dp):raise ValueError('D report final candidate pin mismatch')
 if not np.array_equal(iout['points_world_m'],base['points_world_m']) or not np.array_equal(dout['points_world_m'],base['points_world_m']) or not np.array_equal(bound['points_world_m'],base['points_world_m']):raise ValueError('Fat quotient changed point rows/coordinates')
 for k in ('points_world_m_f32','tetrahedra','material_indices'):
  if not np.array_equal(dout[k],bound[k]):raise ValueError(f'Fat397 geometry differs from D264 candidate: {k}')
 p=np.asarray(base['points_world_m'],dtype=np.float32);ot=np.asarray(base['tetrahedra'],dtype=np.int64);om=np.asarray(base['material_indices'],dtype=np.int32)
 iops=irr['operations'];witnesses=dr['operation_local_witnesses'];
 if len(iops)!=50 or len(witnesses)!=264 or dr['transferred_operation_count']!=264:raise ValueError('Fat I50/D264 operation counts mismatch')
 csr_base=incidence_csr(ot,len(p));qI=np.arange(len(p),dtype=np.int64);i_stars=[]
 # Skin and Fat share the same 50 surface-edge operations, but their volume node row IDs
 # belong to different layer domains. Replay Fat's own authenticated endpoint rows here.
 for idx,op in enumerate(iops):
  drop=int(op['retired_volume_node']);keep=int(op['kept_volume_node'])
  if qI[drop]!=drop:raise ValueError('dependent I50 operation in Fat order')
  reported=np.asarray(op['source_cell_rows'],dtype=np.int64)
  if not np.array_equal(np.sort(reported),full_star(csr_base,[drop])):raise ValueError(f'I50 Fat operation {idx} source star mismatch')
  i_stars.append(full_star(csr_base,[drop,keep]))
  qI[qI==drop]=qI[keep]
 if not np.array_equal(qI,iout['source_I_vertex_quotient']):raise ValueError('I50 map differs from the stored source-I quotient')
 i_map=np.asarray(iout['output_cell_to_input_cell'],dtype=np.int64)
 it=qI[ot];isort=np.sort(it,axis=1);ideg=np.any(np.diff(isort,axis=1)==0,axis=1);i_origin=np.flatnonzero(~ideg);it=it[i_origin];imat=om[i_origin]
 if not np.array_equal(it,iout['tetrahedra']) or not np.array_equal(imat,iout['material_indices']) or not np.array_equal(i_origin,i_map):raise ValueError('I50 exact base→output cell order replay failed')
 ireverse=np.full(len(ot),-1,dtype=np.int64);ireverse[i_origin]=np.arange(len(i_origin),dtype=np.int64)
 if not np.array_equal(ireverse,iout['input_cell_to_output_cell']):raise ValueError('I50 reverse cell map mismatch')
 qD=np.asarray(dout['input_node_to_output_node'],dtype=np.int64)
 if qD.shape!=(len(p),) or np.any(qD<0) or np.any(qD>=len(p)):raise ValueError('D264 node quotient shape/domain mismatch')
 dcsr=incidence_csr(it,len(p));d_stars=[];d_base_stars=[];cross_star_cell_pairs=0
 for idx,w in enumerate(witnesses):
  drop=int(w['drop_volume_row']);keep=int(w['keep_volume_row']);rows=np.asarray(w['actual_current_input_cell_rows'],dtype=np.int64)
  if qI[drop]!=drop or qI[keep]!=keep:raise ValueError(f'D264 operation {idx} uses an I50-retired endpoint')
  if qD[drop]!=qD[keep]:raise ValueError(f'D264 operation {idx} drop/keep not identified by D quotient')
  drop_star=full_star(dcsr,[drop]);star_i=full_star(dcsr,[drop,keep])
  # The D report's current-cell witness is the retired endpoint star. Rebuild
  # the complete endpoint union (including keep-only cells) from authenticated I50 tets.
  if not np.array_equal(np.sort(rows),drop_star):raise ValueError(f'D264 operation {idx} reported drop-star mismatch')
  deleted=np.asarray(w['deleted_current_input_cell_rows'],dtype=np.int64)
  mapped_star=qD[it[star_i]];srt=np.sort(mapped_star,axis=1);deg=np.any(np.diff(srt,axis=1)==0,axis=1)
  if not np.array_equal(np.sort(star_i[deg]),np.sort(deleted)):raise ValueError(f'D264 operation {idx} deleted-cell witness mismatch')
  d_stars.append(star_i);d_base_stars.append(i_origin[star_i])
 d_origin_i=np.flatnonzero(~np.any(np.diff(np.sort(qD[it],axis=1),axis=1)==0,axis=1))
 dt=qD[it[d_origin_i]];dmat=imat[d_origin_i]
 if not np.array_equal(dt,dout['tetrahedra']) or not np.array_equal(dmat,dout['material_indices']):raise ValueError('D264 output order/material replay from I50 failed')
 if not np.array_equal(d_origin_i,dout['output_cell_to_input_cell']):raise ValueError('D264 output→I50 map mismatch')
 dreverse=np.full(len(it),-1,dtype=np.int64);dreverse[d_origin_i]=np.arange(len(d_origin_i),dtype=np.int64)
 if not np.array_equal(dreverse,dout['input_cell_to_output_cell']):raise ValueError('D264 reverse cell map mismatch')
 qtotal=qD[qI];mapped=qtotal[ot];ms=np.sort(mapped,axis=1);deg=np.any(np.diff(ms,axis=1)==0,axis=1);origin=np.flatnonzero(~deg);t=mapped[origin];mat=om[origin]
 expected_origin=i_origin[d_origin_i]
 if not np.array_equal(origin,expected_origin):raise ValueError('I50-then-D264 composed parent row order mismatch')
 if not np.array_equal(t,dout['tetrahedra']) or not np.array_equal(mat,dout['material_indices']):raise ValueError('composed I50-then-D264 base→final exact cell/material replay failed')
 if not np.array_equal(t,bound['tetrahedra']) or not np.array_equal(mat,bound['material_indices']):raise ValueError('Fat397 bound output does not match composed cell order')
 final_faces=qtotal[np.asarray(base['desired_boundary_faces'],dtype=np.int64)];fs=np.sort(final_faces,axis=1);final_faces=final_faces[~np.any(np.diff(fs,axis=1)==0,axis=1)]
 if counter_faces(T,final_faces)!=counter_faces(T,bound['desired_boundary_faces']):raise ValueError('Fat target is not exact I50-then-D264 quotient of Fat573 target')
 bd=T.boundary(t)
 if counter_faces(T,bd)!=counter_faces(T,bound['boundary_faces']) or counter_faces(T,bd)!=counter_faces(T,bound['desired_boundary_faces']):raise ValueError('Fat full boundary/target mismatch after canonical replay')
 det,qual=T.m.measurements(p,t);bad=int(np.sum(~T.m.passes(det,qual)))
 if np.any(det<=0) or bad!=dr['global_native_bad_cells_after']:raise ValueError('Fat global positivity/quality replay mismatch')
 # D operation endpoint stars are rebased from current I50 rows onto canonical Fat573 rows.
 d_base_starsets=[np.unique(i_origin[x]) for x in d_stars]
 allstars=i_stars+d_base_starsets
 # Count cross-operation star intersections without changing the parent domain.
 i_union=set().union(*(set(map(int,x)) for x in i_stars)) if i_stars else set()
 d_union=set().union(*(set(map(int,x)) for x in d_base_starsets)) if d_base_starsets else set()
 cross_star_cell_pairs=len(i_union & d_union)
 changed=np.any(t!=ot[origin],axis=1);changed_out=np.flatnonzero(changed);changed_origin=origin[changed]
 rowowner,po,pr,ngroups,nchangedgroups,totalstar=make_parent_unions(len(ot),allstars,changed_origin)
 out_to_in=origin.copy();out_to_in[changed]=-1
 in_to_out=np.full(len(ot),-1,dtype=np.int64);in_to_out[origin[~changed]]=np.flatnonzero(~changed)
 if np.any(in_to_out[origin[changed]]!=-1):raise ValueError('changed Fat parent cell unexpectedly has retained output identity')
 payload=dict(base)
 # Import only current target metadata from the target397 binder; raw D/I lineage sidecars are omitted.
 target_exact={'boundary_faces','desired_boundary_faces','desired_boundary_source_rows','desired_boundary_degenerate_source_rows','desired_boundary_source_group','desired_boundary_source_face_rows','previous_desired_boundary_faces','previous_desired_boundary_source_rows','desired_boundary_provenance_status'}
 target_prefixes=('fat_current_target_rebind_','typed_I_candidate_','typed_D_candidate_','typed_O_candidate_')
 for k,v in bound.items():
  if k in target_exact or k.startswith(target_prefixes):payload[k]=v
 payload['points_world_m']=p;payload['points_world_m_f32']=p.copy();payload['tetrahedra']=t;payload['material_indices']=mat
 payload.update(current_geometry_parent_path=np.asarray([str(basep)]),current_geometry_parent_sha256=np.asarray([sha(basep)]),
   output_cell_to_input_cell=out_to_in,input_cell_to_output_cell=in_to_out,retained_input_cell_rows=origin[~changed],
   replacement_parent_offsets=po,replacement_parent_cell_rows=pr,replacement_tet_offsets=np.asarray([0,len(changed_out)],np.int64),
   replacement_tetrahedra=t[changed],input_node_to_output_node=qtotal,output_node_to_input_node=np.arange(len(p),dtype=np.int64),
   output_point_to_input_point_row=np.arange(len(p),dtype=np.int64),output_point_parent_offsets=np.zeros(len(p)+1,np.int64),
   output_point_parent_input_cell_rows=np.empty(0,np.int64),source_I_vertex_quotient=qI,
   canonical_parent_union_schema=np.asarray('numi.run055.current-immediate-parent-unions.v1'),
   canonical_parent_union_operation_order=np.asarray('I50_then_D264'))
 for source_role,e in surfaces.items():
  payload['typed_'+source_role+'_candidate_path']=np.asarray([e['path']]);payload['typed_'+source_role+'_candidate_sha256']=np.asarray([e['sha256']])
 # Explicitly exclude stale intermediate mappings and CSR from the D raw bound file.
 for k in list(payload):
  if k.startswith(('canonical573_','D411_source_operation_','D411_rebased_operation_','proposed_current_desired_','run051_current_','run051_output_')) or k in ('D411_operation_report_path','D411_transfer_input_pins_json','source_cell_operation_rows'):
   payload.pop(k)
 outdir.mkdir(parents=True,exist_ok=False);cp=outdir/'fat-current-I50-D264-canonical-parent-unions.npz';np.savez_compressed(cp,**payload)
 written=load_npz(cp)
 if set(written)!=set(payload) or any(not np.array_equal(written[k],v) for k,v in payload.items()):raise ValueError('Fat canonical output serialization readback mismatch')
 for k in ('points_world_m','points_world_m_f32','tetrahedra','material_indices','boundary_faces','desired_boundary_faces'):
  if not np.array_equal(written[k],bound[k]):raise ValueError(f'Fat canonical output changed bound array {k}')
 return {'candidate':pin(cp),'base':pin(basep),'I50_raw_output':pin(ip),'D264_output':pin(dp),'bound_target':pin(boundp),'input_cells':len(ot),'after_I50_cells':len(it),'cells':len(t),'I50_removed_cells':int(len(ot)-len(it)),'D264_removed_cells':int(len(it)-len(t)),'native_quality_failures':bad,'nonpositive':0,'boundary_faces':len(bd),'target_faces':len(bound['desired_boundary_faces']),'missing_target_faces':0,'extra_actual_faces':0,'changed_output_cells':int(changed.sum()),'I50_operations':len(iops),'D264_operations':len(witnesses),'full_endpoint_star_rows_sum':int(totalstar),'I_full_endpoint_star_rows_sum':int(sum(map(len,i_stars))),'D_full_endpoint_star_rows_sum':int(sum(map(len,d_base_starsets))),'I_endpoint_star_union_rows':int(len(i_union)),'D_endpoint_star_union_rows':int(len(d_union)),'I_D_cross_star_row_intersection':int(cross_star_cell_pairs),'parent_union_components':int(ngroups),'components_used_by_changed_cells':int(nchangedgroups),'replacement_parent_row_entries':int(len(pr)),'I_then_D_cell_order_exact':True,'all_point_coordinate_bits_fixed':True,'unrelated_retained_cell_identity_exact':True,'target_quotient_exact':True,'source330_mass':'IMMUTABLE_CURRENT_MAPPING_REQUIRED'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);args=ap.parse_args();start=time.monotonic()
 if args.out.exists():raise FileExistsError(args.out)
 # Pin the two normalization owners whose format is being reused, without editing either owner.
 for owner in (OWNER571,OWNER573):
  if not owner.is_file():raise FileNotFoundError(owner)
 ir,itr,dr,d697,ctx,sbr,fbr,fcr,surfaces=common_pins()
 spec=load_mod('run055_transfer_surface_contractions_indexed_379',H/'transfer_surface_contractions_indexed_379_root.py');T=spec
 args.out.mkdir(parents=True)
 skin=canonical_skin(ir,itr,ctx,surfaces,T,args.out/'skin')
 gc.collect()
 fat=canonical_fat(ir,dr,ctx,surfaces,T,args.out/'fat')
 # Report every immutable source that defines both normalizations.
 inputs={
  'Skin571':pin(SKIN_BASE),'Skin571_receipt':pin(SKIN_BASE_RECEIPT),'Fat573':pin(FAT_BASE),'Fat573_receipt':pin(FAT_BASE_RECEIPT),
  'I50_report':pin(I_REPORT),'I50_typed_candidate':pin(I_CANDIDATE),'I50_typed_receipt':pin(I_RECEIPT),
  'D264_report':pin(D_REPORT),'D264_volume_candidate':pin(D_CANDIDATE),'D264_typed_candidate':pin(D_TYPED),'D264_actual697':pin(D_697),
  'context354':pin(CTX),'Skin369_candidate':pin(SKIN_BOUND),'Skin369_receipt':pin(SKIN_BOUND_RECEIPT),
  'Fat397_candidate':pin(FAT_BOUND),'Fat397_report':pin(FAT_BOUND_REPORT),'Fat397_completion_receipt':pin(FAT_BOUND_RECEIPT),
  'normalization_owner571':pin(OWNER571),'normalization_owner573':pin(OWNER573)}
 receipt={'schema':'numi.run055.current-immediate-parent-unions.v1','status':'PASS_CANONICAL_RUN055_I50_D264_IMMEDIATE_PARENT_UNIONS','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','operation_order':{'Skin':'I50','Fat':'I50_then_D264'},'skins':skin,'fat':fat,'input_pins':inputs,'stale_bound_lineage_policy':'The raw bound candidates intermediate-only cell maps, source_cell_operation_rows, canonical573 CSR, D411 source-row CSR and row-domain sidecars were not carried into the canonical candidates. Replacement parent unions were rebuilt from complete I/D endpoint stars in Skin571/Fat573 row domains.','elapsed_s':time.monotonic()-start,'producer_path':str(Path(__file__)),'producer_sha256':sha(Path(__file__))}
 rp=args.out/'receipt.json';rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':receipt['status'],'receipt':str(rp),'receipt_sha256':sha(rp),'skin':skin,'fat':fat,'elapsed_s':receipt['elapsed_s']},indent=2),flush=True)
if __name__=='__main__':main()
