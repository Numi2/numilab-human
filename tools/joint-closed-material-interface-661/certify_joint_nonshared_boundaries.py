#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
import hashlib, importlib.util, json, sys, time
import numpy as np

P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
R=Path(__file__).resolve().parent; OUT=R/'output'
OWNER_LOADER=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py'
OWNER=P.parent.parent/'publication-206-stage-001/tools/cardiac_cavity_intersections.py'
SKIN=P/'paired-fine-headwall-rim-635/run-013/output/fine-skin-closed-boundary.npz'
FAT=P/'paired-fine-headwall-rim-635/run-010/output/fine-fat-closed-boundary.npz'
SKIN_REPORT=P/'paired-fine-headwall-rim-635/run-013/output/report.json'
FAT_REPORT=P/'paired-fine-headwall-rim-635/run-010/output/report.json'
SKIN_MAP=P/'paired-fine-headwall-rim-635/run-013/output/closed-skin-source-map.npz'
CROSS=P/'complete-cross-surface-pair-audit-654/run-001/report.json'
DI=P/'whole-D-fine-I-interface-627/run-001/report.json'
SHARED_I=P/'joint-closed-material-interface-652/output/exact-stage.json'
RUN007=R.parent/'run-007'
RUN007_REPORT=RUN007/'output/report.json'
RUN007_MAP=RUN007/'output/fat-component-source-row-map.npz'
BOOLEAN=P/'joint-closed-material-interface-652/component-diagnostic-653/run-006/output/boolean-stage.json'
BOOLEAN_REPORT=P/'joint-closed-material-interface-652/component-diagnostic-653/run-006/output/report.json'
BOOLEAN_MESH=P/'joint-closed-material-interface-652/component-diagnostic-653/run-006/output/boolean-intersection-diagnostic.npz'
LATTICE_DEN=1<<149
MAX_FACE_RETRIES=64

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def loadnpz(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def progress(stage,**kw):
 row={'stage':stage,'elapsed_s':time.monotonic()-START,**kw};write_json(OUT/'progress.json',row);print(json.dumps(row,sort_keys=True),flush=True)
def face_components(allfaces,selected_rows):
 faces=np.asarray(allfaces[selected_rows],dtype=np.int64); rows=np.asarray(selected_rows,dtype=np.int64)
 n=len(faces); edge_rows=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0)
 und=np.sort(edge_rows,axis=1);_,inv,cnt=np.unique(und,axis=0,return_inverse=True,return_counts=True)
 if np.any(cnt>2):raise ValueError('nonshared boundary has nonmanifold edge incidence')
 parent=list(range(n));rank=bytearray(n)
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a=find(a);b=find(b)
  if a==b:return
  if rank[a]<rank[b]:a,b=b,a
  parent[b]=a
  if rank[a]==rank[b]:rank[a]+=1
 edge_face=np.tile(np.arange(n,dtype=np.int64),3)
 order=np.argsort(inv,kind='stable');groups=np.split(edge_face[order],np.cumsum(cnt)[:-1])
 for g in groups:
  if len(g)==2:union(int(g[0]),int(g[1]))
 roots=np.fromiter((find(i) for i in range(n)),dtype=np.int64,count=n)
 _,labels=np.unique(roots,return_inverse=True)
 return [rows[labels==i] for i in range(int(labels.max())+1)],cnt
def prepare_index(owner,points,faces):
 records,_=owner.float32_surface_records(np.asarray(points,dtype=np.float32),faces,source_seam_groups=())
 limit=owner.MAX_PREPARED_POINT_LOCATION_FACES
 chunks=tuple(owner.prepare_point_location(records[i:i+limit]) for i in range(0,len(records),limit))
 if not chunks or len(chunks)>owner.MAX_POINT_LOCATION_CHUNKS:raise ValueError('exact point-location index exceeds bounded chunk limit')
 return chunks,[x.record_identity_sha256 for x in chunks]
def query_face_centroid(owner,points,faces,face_rows,index,role):
 attempts=[]
 for face0 in np.asarray(face_rows,dtype=np.int64)[:MAX_FACE_RETRIES]:
  face=int(face0);keys=[owner.float32_point_lattice_key(v) for v in points[faces[face]]]
  sums=[sum(int(k[a]) for k in keys) for a in range(3)]
  q=tuple(Fraction(x,3) for x in sums)
  loc=owner.point_location_prepared_indexed_chunks(q,index)
  attempts.append({'face_row':face,'location':loc.get('location'),'crossings':loc.get('crossings'),'ray':loc.get('ray'),'attempts':loc.get('attempts')})
  if loc.get('location')=='boundary':continue
  if loc.get('location') not in ('inside','outside'):raise ValueError(f'indeterminate exact location for {role} face {face}: {loc}')
  return {'face_row':face,'lattice_centroid_sum_numerators':[str(x) for x in sums],'lattice_centroid_denominator':3,'point_world_m':[float(Fraction(x,3*LATTICE_DEN)) for x in sums],'exact_location':loc,'attempts':attempts}
 raise ValueError(f'{role} has no nonboundary exact centroid within {MAX_FACE_RETRIES} deterministic face rows')

def main():
 global START
 OUT.mkdir(parents=True,exist_ok=False);START=time.monotonic()
 expected={
  SKIN:'352dc5e1ebfeca7f8c7dc13075dbcf0abc8d15a842f52057ba7744f998d5fa23',
  FAT:'b02a064f088e625dfb11c98fd2859e13bd271224bd13d050c7b848c217d856b2',
  SKIN_REPORT:'24ed76b7725d1b097e92d1859662b9725532e651abea5052f42dc0f14fb87b8c',
  FAT_REPORT:'65ca93243fbec108e48ee8151f80994721ee65f3e689d7f8fdaf141388eb9e75',
  SKIN_MAP:'2f8643089a3723928a5c13492373db657c9e5f0b4ec3fc6999da8bcbde5942d9',
  CROSS:'ffbb037969d575f0eec1c4fa7ab2f23d1120561834ce3ee0b4eead5799e604ae',
  DI:'ea5ef3a3879f4b9e78ae49f0c4dfe8c699e718825cba20b573f03bf03bba22f3',
  SHARED_I:'85eb6f386ce5ebf74c40893593e59d93d5bae57b21768825ee0d43e48711daf1',
  RUN007_REPORT:'eb6cf1c8b427d209b075d16368516fa4eddffb1c6adaab566854f58778e5f479',
  RUN007_MAP:'16c21822b274c1f535916c6c3bc48b51c0728ffe3c1cf99aabcd1466598ba99d',
  BOOLEAN:'6cfceb7b825d78e96dc091b3a9a23410d58a308050b6cc0f268c36af95b6689f',
  BOOLEAN_REPORT:'8bb06e552ba5b23c8c6e000b0caa0e324c84d9b3fbc8e421e9cc44491827bd03',
  BOOLEAN_MESH:'b54388634c30bcc4e5964dcdfab6b70ebe2e2f0b3bb1906f7eefb9cde39190e6',
  OWNER_LOADER:'f1d697b42fc9d97472fbd70cec2d8adfe178bb591d3513680609396ec4c204ad',
  OWNER:'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
 }
 paths={'SKIN':SKIN,'FAT':FAT,'SKIN_REPORT':SKIN_REPORT,'FAT_REPORT':FAT_REPORT,'SKIN_MAP':SKIN_MAP,'CROSS':CROSS,'DI':DI,'SHARED_I':SHARED_I,'RUN007_REPORT':RUN007_REPORT,'RUN007_MAP':RUN007_MAP,'BOOLEAN':BOOLEAN,'BOOLEAN_REPORT':BOOLEAN_REPORT,'BOOLEAN_MESH':BOOLEAN_MESH,'OWNER_LOADER':OWNER_LOADER,'OWNER':OWNER}
 pins={k:sha(p) for k,p in paths.items()}
 for p,h in expected.items():
  if sha(p)!=h:raise ValueError('input hash changed '+str(p))
 sr=json.loads(SKIN_REPORT.read_text());fr=json.loads(FAT_REPORT.read_text());cross=json.loads(CROSS.read_text());di=json.loads(DI.read_text());shared=json.loads(SHARED_I.read_text());r7=json.loads(RUN007_REPORT.read_text())
 if sr.get('status')!='PASS_FINE_PAIRED_SKIN_RIM_GEOMETRY_DIAGNOSTIC' or fr.get('status')!='PASS_FINE_PAIRED_RIM_GEOMETRY_DIAGNOSTIC':raise ValueError('closed surface source reports not PASS')
 if not cross.get('complete') or cross.get('status')!='PASS_ALL_REQUESTED_CROSS_LAYER_PAIRS' or any(x.get('exact_pair_count')!=0 or not x.get('decision_complete') for x in cross['pairs']):raise ValueError('whole cross-surface pair evidence not complete/zero')
 if not di.get('complete') or di.get('status')!='PASS_SURFACE_PAIR_GATE' or di.get('exact_triangle_pair_count')!=0:raise ValueError('D-I exact pair gate mismatch')
 if shared.get('status')!='PASS_EXACT_CROSS_RIM_INTERFACE' or not shared.get('I_coordinates_bit_identical') or not shared.get('I_faces_exactly_opposite'):raise ValueError('shared I opposite-interface proof mismatch')
 if r7.get('status')!='PASS_ALL_91_DETACHED_COMPONENTS_OUTSIDE_SKIN':raise ValueError('run007 detached-component source proof not complete')
 if len(r7['detached_components_exactly_located_in_closed_skin']['components'])!=91 or any(x['exact_skin_sample']['location']['location']!='outside' for x in r7['detached_components_exactly_located_in_closed_skin']['components']):raise ValueError('run007 detached component classifications mismatch')
 progress('all_prior_exact_surface_proofs_authenticated',inputs=len(pins),detached_components=91)
 S=loadnpz(SKIN);F=loadnpz(FAT);smap=loadnpz(SKIN_MAP);m7=loadnpz(RUN007_MAP)
 sp=np.asarray(S['points_world_m_f64'],dtype=np.float64);sf=np.asarray(S['faces'],dtype=np.int64)
 fp=np.asarray(F['points_world_m_f64'],dtype=np.float64);ff=np.asarray(F['faces'],dtype=np.int64)
 if not np.array_equal(sp.astype(np.float32).astype(np.float64),sp) or not np.array_equal(fp.astype(np.float32).astype(np.float64),fp):raise ValueError('closed surface coords are not canonical Float32')
 nO=len(S['fine_O_faces']);nI=len(S['reversed_fine_I_faces']);nSr=len(S['oriented_rim_faces'])
 nIf=len(F['fine_I_faces']);nD=len(F['reversed_fine_D_faces']);nFr=len(F['oriented_rim_faces'])
 if not np.array_equal(sf,np.vstack((S['fine_O_faces'],S['reversed_fine_I_faces'],S['oriented_rim_faces']))):raise ValueError('skin face layout/winding differs from pinned O/I/rim source arrays')
 Ipath=Path(fr['fine_surfaces']['I']['path']);Dpath=Path(fr['fine_surfaces']['D']['path']);Opath=Path(sr['fine_surfaces']['O']['path'])
 fine_source_pins={str(Opath):sha(Opath),str(Ipath):sha(Ipath),str(Dpath):sha(Dpath)}
 if any(r7['fine_source_pins'].get(k)!=v for k,v in fine_source_pins.items()):raise ValueError('fine source candidate hashes disagree with pinned run007')
 Osrc,Isrc,Dsrc=(loadnpz(q) for q in (Opath,Ipath,Dpath))
 nOp=len(Osrc['points_world_m']);nIp=len(Isrc['points_world_m']);nDp=len(Dsrc['points_world_m'])
 source_fat_points=np.concatenate((Isrc['points_world_m'].astype(np.float32),Dsrc['points_world_m'].astype(np.float32)),axis=0)
 source_fat_faces=np.vstack((F['fine_I_faces'],F['reversed_fine_D_faces'],F['oriented_rim_faces'])).astype(np.int64)
 source_fat_used=np.unique(source_fat_faces)
 if len(source_fat_used)!=len(fp) or not np.array_equal(source_fat_points[source_fat_used].astype(np.float64),fp):raise ValueError('closed fat point compaction differs from exact source I/D rows')
 source_fat_inverse=np.full(len(source_fat_points),-1,dtype=np.int64);source_fat_inverse[source_fat_used]=np.arange(len(source_fat_used),dtype=np.int64)
 if not np.array_equal(source_fat_inverse[source_fat_faces],ff):raise ValueError('closed fat faces/winding differ from source I/D/rim faces')
 global_points=np.concatenate((Osrc['points_world_m'].astype(np.float32),Isrc['points_world_m'].astype(np.float32),Dsrc['points_world_m'].astype(np.float32)),axis=0)
 fat_global_rows=np.where(source_fat_used<nIp,nOp+source_fat_used,nOp+nIp+(source_fat_used-nIp)).astype(np.int64)
 if not np.array_equal(global_points[fat_global_rows].astype(np.float64),fp):raise ValueError('closed fat compaction does not reconstruct from source roles')
 skin_role=np.asarray(smap['output_vertex_source_layer'],dtype=np.int64);skin_source_row=np.asarray(smap['output_vertex_source_row'],dtype=np.int64)
 if len(skin_role)!=len(sp) or len(skin_source_row)!=len(sp) or not np.isin(skin_role,[1,2]).all():raise ValueError('closed skin source-row map malformed')
 skin_global_rows=np.where(skin_role==2,nOp+skin_source_row,skin_source_row).astype(np.int64)
 if not np.array_equal(global_points[skin_global_rows].astype(np.float64),sp):raise ValueError('closed skin points do not reconstruct from source O/I rows')
 if (nO,nI,nSr,nIf,nD,nFr)!=(112063,205239,90,205239,257827,90):raise ValueError('nonshared face-domain counts changed')
 owner=loadmod('joint_material_nonshared_exact_owner_008',OWNER_LOADER).load_exact_owner()
 # Skin nonshared surface is O plus its authenticated 90-face rim; the I interface is excluded.
 skin_rows=np.concatenate((np.arange(nO,dtype=np.int64),np.arange(nO+nI,nO+nI+nSr,dtype=np.int64)))
 skin_components,skin_edge_counts=face_components(sf,skin_rows)
 if len(skin_components)!=1:raise ValueError(f'expected one connected skin O+rim component, got {len(skin_components)}')
 skin_comp=skin_components[0]
 skin_O_rows=skin_comp[skin_comp<nO]
 if len(skin_O_rows)!=nO or np.count_nonzero(skin_comp>=nO)!=nSr:raise ValueError('skin nonshared component does not cover exact O plus all rim faces')
 # Fat nonshared surface is reverse-D plus its authenticated 90-face rim; the I interface is excluded.
 fat_rows=np.arange(nIf,len(ff),dtype=np.int64)
 fat_components,fat_edge_counts=face_components(ff,fat_rows)
 if len(fat_components)!=92:raise ValueError(f'expected main D/rim plus 91 detached components, got {len(fat_components)}')
 compmeta=[];fat_comp_run007={}
 for cid,rows in enumerate(fat_components):
  dclosed=rows[(rows>=nIf)&(rows<nIf+nD)]
  dsource=(dclosed-nIf).astype(np.int64)
  rimrows=rows[rows>=nIf+nD]
  if not len(dsource):raise ValueError(f'nonshared fat component {cid} has no D face')
  fullids=np.unique(m7['face_component_id'][nIf+dsource])
  if len(fullids)!=1:raise ValueError(f'nonshared fat component {cid} maps to multiple full fat components {fullids}')
  fullid=int(fullids[0])
  old_off=m7['component_source_D_face_offsets']
  oldrows=np.asarray(m7['component_source_D_face_rows'][old_off[fullid]:old_off[fullid+1]],dtype=np.int64)
  if not np.array_equal(np.sort(dsource),np.sort(oldrows)):raise ValueError(f'nonshared fat component {cid} D rows do not match run007 component {fullid}')
  if len(rimrows) and not np.all(m7['face_component_id'][rimrows]==fullid):raise ValueError(f'nonshared fat component {cid} rim rows map to another full-fat component')
  if fullid in fat_comp_run007:raise ValueError('duplicate mapped full fat component')
  fat_comp_run007[fullid]=cid
  compmeta.append({'nonshared_component_id':cid,'run007_full_fat_component_id':fullid,'closed_fat_face_count':int(len(rows)),'source_D_face_count':int(len(dsource)),'fat_rim_face_count':int(len(rimrows)),'source_D_face_rows_sha256':hashlib.sha256(np.asarray(np.sort(dsource),dtype='<i8').tobytes()).hexdigest(),'source_D_face_rows_first_last':[int(dsource.min()),int(dsource.max())],'rim_closed_fat_face_rows':rimrows.tolist() if len(rimrows)<=90 else None})
 if set(fat_comp_run007)!=set(range(92)):raise ValueError('nonshared fat component partition does not map bijectively to all run007 components')
 main_matches=[cid for cid,meta in enumerate(compmeta) if meta['run007_full_fat_component_id']==20]
 if len(main_matches)!=1:raise ValueError('main D/rim component mapping is not unique')
 main_fat_nonshared_id=main_matches[0]
 detached_map={}
 for c in r7['detached_components_exactly_located_in_closed_skin']['components']:
  detached_map[int(c['component_id'])]=c
 if set(detached_map)!=set(range(92))-{20}:raise ValueError('run007 component report missing or extra detached components')
 progress('nonshared_edge_components_and_run007_crosswalk_verified',skin_components=1,fat_nonshared_components=92,main_fat_component=main_fat_nonshared_id,detached_reused=91)
 # Exact integer-lattice indexed point-location; preparation is bounded to owner chunks.
 skin_index,skin_index_hashes=prepare_index(owner,sp,sf)
 progress('closed_skin_integer_lattice_index_ready',faces=len(sf),chunks=len(skin_index),record_hashes=skin_index_hashes)
 fat_index,fat_index_hashes=prepare_index(owner,fp,ff)
 progress('closed_fat_integer_lattice_index_ready',faces=len(ff),chunks=len(fat_index),record_hashes=fat_index_hashes)
 # Representative of the single Skin(O+rim) component, located against the complete closed fat solid.
 skin_sample=query_face_centroid(owner,sp,sf,np.sort(skin_O_rows),fat_index,'skin_O_nonshared_component')
 # Representative of the main Fat(D+rim) component, located against the complete closed skin solid.
 mainrows=fat_components[main_fat_nonshared_id]
 main_dclosed=mainrows[(mainrows>=nIf)&(mainrows<nIf+nD)]
 main_sample=query_face_centroid(owner,fp,ff,main_dclosed,skin_index,'main_fat_D_nonshared_component')
 main_sample['source_D_face_row']=int(main_sample['face_row']-nIf)
 # Detached D-only components reuse run007 exact centroid locations after the full-face/component/D-row bijection above.
 reused=[]
 for fullid in sorted(detached_map):
  c=detached_map[fullid]
  reused.append({'nonshared_component_id':fat_comp_run007[fullid],'run007_full_fat_component_id':fullid,'source_D_face_rows_offset':int(c['source_D_face_rows_offset']),'source_D_face_rows_count':int(c['source_D_face_rows_count']),'source_D_face_rows_sha256':c['source_D_face_rows_sha256'],'signed_volume_m3':float(c['signed_volume_m3']),'orientation_sign':int(c['orientation_sign']),'negative_cavity_shell':bool(c['negative_cavity_shell']),'representative_source_D_face_row':int(c['exact_skin_sample']['source_D_face_row']),'representative_location_in_closed_skin':c['exact_skin_sample']['location'],'exact_receipt':'run007 exact integer-lattice location; source-D CSR set revalidated against nonshared D component'})
 skin_out=skin_sample['exact_location']['location']=='outside'
 fat_main_out=main_sample['exact_location']['location']=='outside'
 detached_out=all(x['representative_location_in_closed_skin']['location']=='outside' for x in reused)
 statuses={'skin_O_rim_component_outside_closed_fat':skin_out,'main_fat_D_rim_component_outside_closed_skin':fat_main_out,'all_91_detached_fat_components_outside_closed_skin':detached_out}
 # Keep source face rows for every nonshared component in a compact CSR map.
 skin_offsets=np.concatenate(([0],np.cumsum([len(x) for x in skin_components]))).astype(np.int64)
 fat_offsets=np.concatenate(([0],np.cumsum([len(x) for x in fat_components]))).astype(np.int64)
 skin_faces_csr=np.concatenate(skin_components).astype(np.int64)
 fat_faces_csr=np.concatenate(fat_components).astype(np.int64)
 map_path=OUT/'nonshared-boundary-component-map.npz'
 np.savez_compressed(map_path,skin_nonshared_face_offsets=skin_offsets,skin_nonshared_closed_face_rows=skin_faces_csr,fat_nonshared_face_offsets=fat_offsets,fat_nonshared_closed_face_rows=fat_faces_csr,fat_nonshared_component_run007_full_component_id=np.asarray([x['run007_full_fat_component_id'] for x in compmeta],dtype=np.int16),skin_sample_closed_face_row=np.asarray(skin_sample['face_row'],dtype=np.int64),fat_main_sample_closed_face_row=np.asarray(main_sample['face_row'],dtype=np.int64),fat_component_reuse_ids=np.asarray([x['run007_full_fat_component_id'] for x in reused],dtype=np.int16),fat_component_reuse_sample_D_rows=np.asarray([x['representative_source_D_face_row'] for x in reused],dtype=np.int64),fat_component_reuse_location_codes=np.asarray([1 if x['representative_location_in_closed_skin']['location']=='inside' else 0 for x in reused],dtype=np.int8))
 boolean=json.loads(BOOLEAN.read_text())
 result={'schema':'numi.joint-exact-nonshared-boundary-component-interior.v1','status':'PASS_EXACT_NO_COMMON_MATERIAL_INTERIOR_BOUNDARY_CERTIFICATE' if all(statuses.values()) else 'FAIL_EXACT_NONSHARED_BOUNDARY_CONTAINMENT','inputs':pins,'closed_surface_topology':{'skin_faces':len(sf),'fat_faces':len(ff),'skin_nonshared_O_plus_rim_faces':len(skin_rows),'fat_nonshared_reversed_D_plus_rim_faces':len(fat_rows),'skin_nonshared_component_count':len(skin_components),'fat_nonshared_component_count':len(fat_components),'skin_edge_incidence_max':int(skin_edge_counts.max()),'fat_edge_incidence_max':int(fat_edge_counts.max()),'shared_I_face_count':nI,'shared_I_coordinates_bit_identical':bool(json.loads(SHARED_I.read_text())['I_coordinates_bit_identical']),'shared_I_faces_exactly_opposite':bool(json.loads(SHARED_I.read_text())['I_faces_exactly_opposite'])},'exact_nonshared_component_classification':{'skin_O_plus_rim_component':{'component_count':1,'faces':len(skin_comp),'source_O_face_count':len(skin_O_rows),'skin_rim_face_count':nSr,'representative':skin_sample,'location_in_complete_closed_fat':'outside' if skin_out else skin_sample['exact_location']['location']},'fat_reversed_D_plus_rim_components':{'component_count':len(fat_components),'main_component_id':main_fat_nonshared_id,'main_component_run007_full_fat_id':20,'main_component':{'faces':len(mainrows),'source_D_face_count':len(main_dclosed),'rim_face_count':int(np.count_nonzero(mainrows>=nIf+nD)),'representative':main_sample,'location_in_complete_closed_skin':'outside' if fat_main_out else main_sample['exact_location']['location']},'detached_components_reused_from_run007':reused,'all_91_detached_outside_skin':detached_out},'all_nonshared_component_sides_outside':statuses},'exact_cross_boundary_evidence':{'open_O_I_O_D_skin_rim_D_fat_rim_O':{'report_sha256':pins['CROSS'],'all_requested_exact_pair_counts_zero':True},'open_D_I':{'report_sha256':pins['DI'],'exact_pairs':0},'skin_rim_vs_complete_O_I':{'report_sha256':pins['SKIN_REPORT'],'exact_pairs':sr['exact_rim_surface_audits']['rim_vs_complete_fine_O_and_reversed_I']['pairs']},'fat_rim_vs_complete_I_D':{'report_sha256':pins['FAT_REPORT'],'exact_pairs':fr['exact_rim_surface_audits']['rim_vs_complete_fine_I_and_reversed_D']['pairs']},'cross_rim_and_shared_I':{'report_sha256':pins['SHARED_I'],'cross_rim_pairs':shared['cross_rim']['count'],'I_opposite_orientation':shared['I_faces_exactly_opposite']}},'formal_no_common_interior_basis':{'conditions_verified':all(statuses.values()),'statement':'For these closed, oriented source-boundary shells, every connected component of each nonshared boundary lies outside the opposite closed solid, and exhaustive exact pair audits show no crossing between nonshared surfaces; the only shared interface is the bit-identical, oppositely oriented I boundary. Under these verified conditions the two material interiors have no common open 3D region.','method_scope':'Exact point-location of deterministic boundary-face centroids per connected component, with component constancy supported by exhaustive zero-crossing audits; detached fat locations reused only after source-D row/component bijection replay.'},'positive_manifold_boolean_retained_as_contradictory_diagnostic':{'status':'UNRESOLVED_NUMERIC_BOOLEAN_RESULT_RETAINED','stage_sha256':pins['BOOLEAN'],'report_sha256':pins['BOOLEAN_REPORT'],'intersection_mesh_sha256':pins['BOOLEAN_MESH'],'reported_intersection_volume_m3':float(boolean['skin_fat_boolean_volume_m3']),'dominant_component_index':457,'dominant_component_volume_m3':1.0353422257115748e-14,'no_repair_or_tolerance_override':True},'source_map':{'path':str(map_path),'sha256':sha(map_path),'bytes':map_path.stat().st_size},'scope':{'geometry_unchanged':True,'components_removed':False,'physical_admission':False,'runtime_created':0,'accepted_physical_steps':0,'full_pair_rescan':False,'new_boolean_operation':False,'tolerance_or_interface_waiver':False}}
 write_json(OUT/'report.json',result)
 progress('nonshared_exact_joint_certificate_complete',status=result['status'],map_sha256=sha(map_path),report=str(OUT/'report.json'))
 print(json.dumps({'status':result['status'],'classifications':statuses,'skin_component_count':len(skin_components),'fat_nonshared_component_count':len(fat_components),'boolean_volume_retained':float(boolean['skin_fat_boolean_volume_m3']),'report':str(OUT/'report.json'),'map':str(map_path),'elapsed_s':time.monotonic()-START},sort_keys=True),flush=True)
if __name__=='__main__':main()
