#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
import hashlib, importlib.util, json, math, sys, time
import numpy as np

P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
R=Path(__file__).resolve().parent
OUT=R/'output'
OWNER_LOADER=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py'
OWNER=P.parent.parent/'publication-206-stage-001/tools/cardiac_cavity_intersections.py'
SKIN=P/'paired-fine-headwall-rim-635/run-013/output/fine-skin-closed-boundary.npz'
FAT=P/'paired-fine-headwall-rim-635/run-010/output/fine-fat-closed-boundary.npz'
SKIN_REPORT=P/'paired-fine-headwall-rim-635/run-013/output/report.json'
FAT_REPORT=P/'paired-fine-headwall-rim-635/run-010/output/report.json'
CROSS=P/'complete-cross-surface-pair-audit-654/run-001/report.json'
DI=P/'whole-D-fine-I-interface-627/run-001/report.json'
SHARED_I=P/'joint-closed-material-interface-652/output/exact-stage.json'
RIM=P/'paired-rim-006-632/run-001/report.json'
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
 r={'stage':stage,'elapsed_s':time.monotonic()-START,**kw};write_json(OUT/'progress.json',r);print(json.dumps(r,sort_keys=True),flush=True)
def boundary_edges(faces):
 edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0)
 und=np.sort(edges,axis=1);_,inv,cnt=np.unique(und,axis=0,return_inverse=True,return_counts=True)
 dirs=np.where(edges[:,0]<edges[:,1],1,-1)
 bal=np.bincount(inv,weights=dirs,minlength=len(cnt)).astype(np.int64)
 return cnt,bal
def connected_faces(faces,counts):
 # Same deterministic union-find/labeling owner as the pinned run-010 report.
 n=len(faces);parent=list(range(n));rank=bytearray(n)
 def find(x):
  while parent[x]!=x:
   parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a=find(a);b=find(b)
  if a==b:return
  if rank[a]<rank[b]:a,b=b,a
  parent[b]=a
  if rank[a]==rank[b]:rank[a]+=1
 edge_face=np.tile(np.arange(n,dtype=np.int64),3)
 edge_rows=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0)
 _,inv,_=np.unique(np.sort(edge_rows,axis=1),axis=0,return_inverse=True,return_counts=True)
 order=np.argsort(inv,kind='stable')
 groups=np.split(edge_face[order],np.cumsum(counts)[:-1])
 for g in groups:
  if len(g)==2:union(int(g[0]),int(g[1]))
 roots=np.fromiter((find(i) for i in range(n)),dtype=np.int64,count=n)
 _,labels=np.unique(roots,return_inverse=True)
 return labels

def main():
 global START
 OUT.mkdir(parents=True,exist_ok=False);START=time.monotonic()
 expected={
  SKIN:'352dc5e1ebfeca7f8c7dc13075dbcf0abc8d15a842f52057ba7744f998d5fa23',
  FAT:'b02a064f088e625dfb11c98fd2859e13bd271224bd13d050c7b848c217d856b2',
  SKIN_REPORT:'24ed76b7725d1b097e92d1859662b9725532e651abea5052f42dc0f14fb87b8c',
  FAT_REPORT:'65ca93243fbec108e48ee8151f80994721ee65f3e689d7f8fdaf141388eb9e75',
  CROSS:'ffbb037969d575f0eec1c4fa7ab2f23d1120561834ce3ee0b4eead5799e604ae',
  DI:'ea5ef3a3879f4b9e78ae49f0c4dfe8c699e718825cba20b573f03bf03bba22f3',
  SHARED_I:'85eb6f386ce5ebf74c40893593e59d93d5bae57b21768825ee0d43e48711daf1',
  RIM:'495d5544c674c29aa93ac3c175306d03f561e92c8511a9e8f3dd5be472796666',
  BOOLEAN:'6cfceb7b825d78e96dc091b3a9a23410d58a308050b6cc0f268c36af95b6689f',
  BOOLEAN_REPORT:'8bb06e552ba5b23c8c6e000b0caa0e324c84d9b3fbc8e421e9cc44491827bd03',
  BOOLEAN_MESH:'b54388634c30bcc4e5964dcdfab6b70ebe2e2f0b3bb1906f7eefb9cde39190e6',
  OWNER_LOADER:'f1d697b42fc9d97472fbd70cec2d8adfe178bb591d3513680609396ec4c204ad',
  OWNER:'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
 }
 pins={str(p):sha(p) for p in expected}
 for p,h in expected.items():
  if pins[str(p)]!=h:raise ValueError('pinned input changed: '+str(p))
 sr=json.loads(SKIN_REPORT.read_text());fr=json.loads(FAT_REPORT.read_text())
 cross=json.loads(CROSS.read_text());di=json.loads(DI.read_text());shared=json.loads(SHARED_I.read_text());rim=json.loads(RIM.read_text())
 if sr.get('status')!='PASS_FINE_PAIRED_SKIN_RIM_GEOMETRY_DIAGNOSTIC' or fr.get('status')!='PASS_FINE_PAIRED_RIM_GEOMETRY_DIAGNOSTIC':raise ValueError('closed boundary report status mismatch')
 if sr.get('runtime_created')!=0 or fr.get('runtime_created')!=0 or sr.get('accepted_physical_steps')!=0 or fr.get('accepted_physical_steps')!=0:raise ValueError('unexpected runtime or physical step in source evidence')
 if not cross.get('complete') or cross.get('status')!='PASS_ALL_REQUESTED_CROSS_LAYER_PAIRS' or any(x.get('exact_pair_count')!=0 or not x.get('decision_complete') for x in cross['pairs']):raise ValueError('cross-pair proof incomplete/nonzero')
 if not di.get('complete') or di.get('status')!='PASS_SURFACE_PAIR_GATE' or di.get('exact_triangle_pair_count')!=0:raise ValueError('D-I interface proof mismatch')
 if shared.get('status')!='PASS_EXACT_CROSS_RIM_INTERFACE' or not shared.get('I_coordinates_bit_identical') or not shared.get('I_faces_exactly_opposite'):raise ValueError('shared-I exact proof mismatch')
 if rim.get('status')!='PASS_TOPOLOGY_AND_TET_BOUNDARY_ONLY':raise ValueError('coarse paired-rim proof mismatch')
 progress('prior_exact_boundary_and_pair_receipts_pinned',pinned_files=len(pins),cross_pairs=len(cross['pairs']),all_cross_pairs_zero=True)
 S=loadnpz(SKIN);F=loadnpz(FAT)
 sp=np.asarray(S['points_world_m_f64'],dtype=np.float64);sf=np.asarray(S['faces'],dtype=np.int64)
 fp=np.asarray(F['points_world_m_f64'],dtype=np.float64);ff=np.asarray(F['faces'],dtype=np.int64)
 if not np.array_equal(sp.astype(np.float32).astype(np.float64),sp) or not np.array_equal(fp.astype(np.float32).astype(np.float64),fp):raise ValueError('closed boundary coordinates are not canonical Float32 emissions')
 nI=int(len(F['fine_I_faces']));nD=int(len(F['reversed_fine_D_faces']));nR=int(len(F['oriented_rim_faces']))
 if (nI,nD,nR,len(ff))!=(205239,257827,90,463156):raise ValueError('closed fat face row domains changed')
 Ipath=Path(fr['fine_surfaces']['I']['path']);Dpath=Path(fr['fine_surfaces']['D']['path']);Opath=Path(sr['fine_surfaces']['O']['path'])
 source_pins={str(p):sha(p) for p in (Opath,Ipath,Dpath)}
 if source_pins[str(Opath)]!='2958a704db4b8b68422640a5e991c6b51af25f19d525c49adc8a742f0c207d7a' or source_pins[str(Ipath)]!='5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410' or source_pins[str(Dpath)]!='c17794a1012f76924dcd9e4e754bb026c774deb5a3d37e2f1c79635ccfe9dc83':raise ValueError('fine O/I/D source candidate changed')
 O,I,D=(loadnpz(p) for p in (Opath,Ipath,Dpath))
 nO=len(O['points_world_m']);nIfine=len(I['points_world_m']);nDfine=len(D['points_world_m'])
 source_points=np.concatenate((I['points_world_m'].astype(np.float32),D['points_world_m'].astype(np.float32)),axis=0)
 source_faces=np.vstack((F['fine_I_faces'],F['reversed_fine_D_faces'],F['oriented_rim_faces'])).astype(np.int64)
 used=np.unique(source_faces)
 if len(used)!=len(fp) or not np.array_equal(source_points[used].astype(np.float64),fp):raise ValueError('closed fat point compaction is not exact source I/D row mapping')
 inverse=np.full(len(source_points),-1,dtype=np.int64);inverse[used]=np.arange(len(used),dtype=np.int64)
 if not np.array_equal(inverse[source_faces],ff):raise ValueError('closed fat face row mapping/winding changed')
 global_points=np.concatenate((O['points_world_m'].astype(np.float32),I['points_world_m'].astype(np.float32),D['points_world_m'].astype(np.float32)),axis=0)
 fat_global_source_rows=np.where(used<nIfine,nO+used,nO+nIfine+(used-nIfine)).astype(np.int64)
 if not np.array_equal(global_points[fat_global_source_rows].astype(np.float64),fp):raise ValueError('fat compacted points fail exact source-role reconstruction')
 smap_path=P/'paired-fine-headwall-rim-635/run-013/output/closed-skin-source-map.npz'
 smap=loadnpz(smap_path)
 srole=np.asarray(smap['output_vertex_source_layer'],dtype=np.int64);srow=np.asarray(smap['output_vertex_source_row'],dtype=np.int64)
 if len(srole)!=len(sp) or len(srow)!=len(sp):raise ValueError('skin source map domain mismatch')
 skin_global_rows=np.where(srole==1,srow,nO+srow).astype(np.int64)
 if not np.array_equal(global_points[skin_global_rows].astype(np.float64),sp):raise ValueError('skin compacted points fail exact source-role reconstruction')
 # Face component and signed-volume replay exactly follows run-010.
 edge_count,edge_balance=boundary_edges(ff)
 if not np.all(edge_count==2) or not np.all(edge_balance==0):raise ValueError('fat closed oriented edge incidence changed')
 labels=connected_faces(ff,edge_count);ncomp=int(labels.max())+1
 ref=fr['closed_fat_boundary']['face_connected_components']
 if ncomp!=92 or len(ref)!=92:raise ValueError(f'expected 92 edge-connected fat components, got {ncomp}')
 used_vertices=np.unique(ff);origin=fp[used_vertices].astype(np.float64).mean(axis=0)
 tri=fp[ff].astype(np.float64)-origin
 terms=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6.0
 volumes=[math.fsum(map(float,terms[labels==i])) for i in range(ncomp)]
 component_faces=[np.flatnonzero(labels==i).astype(np.int64) for i in range(ncomp)]
 main_ids=[];component_meta=[];D_csr=[];D_offsets=[0];face_csr=[];face_offsets=[0]
 for cid,rows in enumerate(component_faces):
  Irows=rows[rows<nI]
  Dclosed=rows[(rows>=nI)&(rows<nI+nD)]
  Dsource=(Dclosed-nI).astype(np.int64)
  Rrows=rows[rows>=nI+nD]
  expectation=ref[cid]
  sign=1 if volumes[cid]>0 else -1
  if expectation.get('component')!=cid or expectation.get('faces')!=len(rows):raise ValueError(f'component {cid} face grouping differs from run-010 report')
  if volumes[cid]!=float(expectation['signed_volume_m3']):raise ValueError(f'component {cid} signed volume differs from run-010 report')
  if sign!=int(expectation['actual_orientation_sign']):raise ValueError(f'component {cid} orientation sign differs from run-010')
  if len(Irows):main_ids.append(cid)
  if cid!=20 and (len(Irows) or len(Rrows) or not len(Dsource)):raise ValueError(f'detached component {cid} is not D-only')
  if cid==20 and (len(Irows)!=nI or len(Rrows)!=nR):raise ValueError('main component does not own all I/rim faces')
  face_csr.extend(rows.tolist());face_offsets.append(len(face_csr))
  D_csr.extend(Dsource.tolist());D_offsets.append(len(D_csr))
  component_meta.append({'component_id':cid,'face_count':len(rows),'I_source_face_count':len(Irows),'D_source_face_count':len(Dsource),'rim_source_face_count':len(Rrows),'signed_volume_m3':volumes[cid],'orientation_sign':sign,'run010_nesting_depth':int(expectation['nesting_depth']),'run010_expected_orientation_sign':int(expectation['expected_orientation_sign']),'run010_orientation_matches_nesting':bool(expectation['orientation_matches_nesting']),'source_D_face_rows_offset':D_offsets[-2],'source_D_face_rows_count':len(Dsource),'source_D_face_rows_sha256':hashlib.sha256(np.asarray(Dsource,dtype='<i8').tobytes()).hexdigest(),'closed_fat_face_rows_sha256':hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest()})
 if main_ids!=[20] or sum(x['D_source_face_count'] for x in component_meta)!=nD:raise ValueError(f'main I-bearing component mismatch or D-row coverage mismatch: main={main_ids}')
 all_D=np.asarray(D_csr,dtype=np.int64)
 if len(np.unique(all_D))!=nD or not np.array_equal(np.sort(all_D),np.arange(nD,dtype=np.int64)):raise ValueError('component D-row CSR is not exact complete partition')
 detached=[i for i in range(ncomp) if i!=20]
 if len(detached)!=91:raise ValueError('expected 91 detached components')
 progress('all_92_components_replayed_against_run010',main_component=20,detached=91,positive_detached=sum(volumes[i]>0 for i in detached),negative_detached=sum(volumes[i]<0 for i in detached),D_rows=nD)
 owner=loadmod('joint653_exact_lattice_owner_007',OWNER_LOADER).load_exact_owner()
 records,_=owner.float32_surface_records(sp.astype(np.float32),sf,source_seam_groups=())
 limit=owner.MAX_PREPARED_POINT_LOCATION_FACES
 prepared=tuple(owner.prepare_point_location(records[k:k+limit]) for k in range(0,len(records),limit))
 if len(prepared)>owner.MAX_POINT_LOCATION_CHUNKS:raise ValueError('skin exact index exceeds owner chunk limit')
 progress('closed_skin_exact_integer_lattice_index_ready',faces=len(records),chunks=len(prepared),record_hashes=[x.record_identity_sha256 for x in prepared])
 classifications=[];sample_closed=[];sample_source=[];sample_codes=[];sample_crossings=[];stop_positive=None
 for ordinal,cid in enumerate(detached,1):
  rows=component_faces[cid];dclosed=rows[(rows>=nI)&(rows<nI+nD)]
  attempts=[];chosen=None;boundary_n=0
  for face_row0 in dclosed[:MAX_FACE_RETRIES]:
   face_row=int(face_row0);drow=face_row-nI
   tri_points=fp[ff[face_row]]
   keys=[owner.float32_point_lattice_key(v) for v in tri_points]
   sums=[sum(int(k[a]) for k in keys) for a in range(3)]
   query=tuple(Fraction(x,3) for x in sums)
   loc=owner.point_location_prepared_indexed_chunks(query,prepared)
   attempts.append({'closed_fat_face_row':face_row,'source_D_face_row':drow,'location':loc.get('location'),'crossings':loc.get('crossings'),'ray':loc.get('ray'),'attempts':loc.get('attempts')})
   if loc.get('location')=='boundary':
    boundary_n+=1;continue
   if loc.get('location') not in ('inside','outside'):raise ValueError(f'indeterminate exact skin location for component{cid}, source D face{drow}: {loc}')
   chosen={'closed_fat_face_row':face_row,'source_D_face_row':drow,'lattice_centroid_sum_numerators':[str(x) for x in sums],'lattice_centroid_denominator':3,'point_world_m':[float(Fraction(x,3*LATTICE_DEN)) for x in sums],'location':loc}
   break
  if chosen is None:raise ValueError(f'component{cid} had no non-boundary exact centroid in first{MAX_FACE_RETRIES} deterministic D rows')
  sign=1 if volumes[cid]>0 else -1
  entry={'component_id':cid,'source_D_face_rows_offset':component_meta[cid]['source_D_face_rows_offset'],'source_D_face_rows_count':component_meta[cid]['D_source_face_count'],'source_D_face_rows_sha256':component_meta[cid]['source_D_face_rows_sha256'],'signed_volume_m3':volumes[cid],'orientation_sign':sign,'negative_cavity_shell':sign<0,'run010_nesting_depth':component_meta[cid]['run010_nesting_depth'],'run010_expected_orientation_sign':component_meta[cid]['run010_expected_orientation_sign'],'run010_orientation_matches_nesting':component_meta[cid]['run010_orientation_matches_nesting'],'exact_skin_sample':chosen,'boundary_sample_retries':boundary_n,'deterministic_attempts':attempts}
  classifications.append(entry);sample_closed.append(chosen['closed_fat_face_row']);sample_source.append(chosen['source_D_face_row']);sample_codes.append(1 if chosen['location']['location']=='inside' else 0);sample_crossings.append(int(chosen['location']['crossings']))
  progress('detached_component_exact_skin_location',ordinal=ordinal,total=91,component=cid,sign=sign,location=chosen['location']['location'],source_D_face_row=chosen['source_D_face_row'],boundary_retries=boundary_n)
  if sign>0 and chosen['location']['location']=='inside':
   stop_positive={'component_id':cid,'source_D_face_row':chosen['source_D_face_row'],'closed_fat_face_row':chosen['closed_fat_face_row'],'signed_volume_m3':volumes[cid],'exact_location':chosen['location']}
   break
 map_path=OUT/'fat-component-source-row-map.npz'
 np.savez_compressed(map_path,face_component_id=np.asarray(labels,dtype=np.int16),component_id=np.arange(ncomp,dtype=np.int16),component_face_offsets=np.asarray(face_offsets,dtype=np.int64),component_closed_fat_face_rows=np.asarray(face_csr,dtype=np.int64),component_source_D_face_offsets=np.asarray(D_offsets,dtype=np.int64),component_source_D_face_rows=all_D,classified_detached_component_ids=np.asarray([x['component_id'] for x in classifications],dtype=np.int16),sample_closed_fat_face_rows=np.asarray(sample_closed,dtype=np.int64),sample_source_D_face_rows=np.asarray(sample_source,dtype=np.int64),sample_location_code=np.asarray(sample_codes,dtype=np.int8),sample_crossing_count=np.asarray(sample_crossings,dtype=np.int32))
 boolean=json.loads(BOOLEAN.read_text())
 result={'schema':'numi.joint-exact-detached-fat-component-location.v1','status':'STOP_POSITIVE_COMPONENT_INSIDE_SKIN' if stop_positive else 'PASS_ALL_91_DETACHED_COMPONENTS_OUTSIDE_SKIN','input_pins':pins,'fine_source_pins':source_pins,'closed_surface_rows':{'skin_points':len(sp),'skin_faces':len(sf),'fat_points':len(fp),'fat_faces':len(ff),'fat_source_I_faces':nI,'fat_source_reversed_D_faces':nD,'fat_rim_faces':nR},'prior_cross_boundary_evidence':{'complete_source_pair_audit_sha256':pins[str(CROSS)],'all_four_cross_audits_zero':True,'D_vs_I_exact_pairs':0,'D_vs_I_report_sha256':pins[str(DI)],'shared_I_coordinates_bit_identical':shared['I_coordinates_bit_identical'],'shared_I_faces_exactly_opposite':shared['I_faces_exactly_opposite'],'shared_I_face_count':shared['shared_I_faces']},'component_partition':{'count':ncomp,'main_fat_component_id':20,'main_fat_signed_volume_m3':volumes[20],'main_fat_orientation_sign':1,'detached_count':len(detached),'positive_detached_count':sum(volumes[i]>0 for i in detached),'negative_detached_count':sum(volumes[i]<0 for i in detached),'run010_exact_nesting_orientation_report_sha256':pins[str(FAT_REPORT)],'all_component_signed_volumes_match_run010':True,'components':component_meta,'source_D_row_partition_npz':str(map_path),'source_D_row_partition_sha256':sha(map_path),'source_D_row_partition_bytes':map_path.stat().st_size},'detached_components_exactly_located_in_closed_skin':{'required':91,'classified':len(classifications),'components':classifications,'positive_inside_stop_witness':stop_positive},'manifold_boolean_overlap_retained_separately':{'status':'UNRESOLVED_POSITIVE_BOOLEAN_VOLUME_RETAINED','boolean_stage_sha256':pins[str(BOOLEAN)],'diagnostic_report_sha256':pins[str(BOOLEAN_REPORT)],'intersection_mesh_sha256':pins[str(BOOLEAN_MESH)],'volume_m3':float(boolean['skin_fat_boolean_volume_m3']),'dominant_intersection_component_index':457,'dominant_component_volume_m3':1.0353422257115748e-14,'dominant_component_faces':4,'dominant_component_vertices':4,'interpretation':'No exact detached-component classification overwrites or waives the positive Manifold Boolean result.'},'scope':{'source_geometry_modified':False,'components_removed':False,'physical_admission':False,'runtime_created':0,'accepted_physical_steps':0,'tolerance_or_contact_waiver':False,'exact_location_method':'face centroid from pinned Float32 source-D triangle; exact integer-lattice keys summed and divided by 3; queried through pinned exact indexed point-location owner','boundary_fallback_face_limit':MAX_FACE_RETRIES}}
 write_json(OUT/'report.json',result)
 progress('detached_component_report_complete',status=result['status'],classified=len(classifications),source_D_partition_sha256=sha(map_path),report= str(OUT/'report.json'))
 print(json.dumps({'status':result['status'],'classified':len(classifications),'required':91,'positive_inside':stop_positive,'report':str(OUT/'report.json'),'map':str(map_path),'elapsed_s':time.monotonic()-START},sort_keys=True),flush=True)
if __name__=='__main__':main()
