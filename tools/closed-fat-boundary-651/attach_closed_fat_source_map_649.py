from pathlib import Path
import numpy as np,json,hashlib,csv
p=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209");out=p/'closed-fat-source-map-649';out.mkdir(exist_ok=False)
rdir=p/'paired-fine-headwall-rim-635/run-010/output'
report=json.loads((rdir/'report.json').read_text())
I=Path(report['fine_surfaces']['I']['path']);D=Path(report['fine_surfaces']['D']['path'])
with np.load(I) as z:ip=z['points_world_m'];iff=z['faces']
with np.load(D) as z:dp=z['points_world_m'];dff=z['faces']
with np.load(rdir/'fine-fat-closed-boundary.npz') as z:points=z['points_world_m_f64'];faces=z['faces'];rim=z['oriented_rim_faces']
sourcepoints=np.concatenate((ip,dp))
sourcefaces=np.vstack((iff,dff[:,[0,2,1]]+len(ip),rim))
used=np.unique(sourcefaces);inverse=np.full(len(sourcepoints),-1,np.int64);inverse[used]=np.arange(len(used))
assert np.array_equal(sourcepoints[used].astype(float),points)
assert np.array_equal(inverse[sourcefaces],faces)
assert len(used)==len(points) and not np.any(inverse[sourcefaces]<0)
facetpath=p/'conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets/volume/facet-ancestry.csv'
with facetpath.open() as f:facets=list(csv.DictReader(f))
rimrows=np.asarray([i for i,v in enumerate(facets) if v['source_partition']=='fat' and v['face_source_role']=='donor_fat_ID_headwall'],np.int64)
assert len(rimrows)==90
with np.load(rdir/'fine-rim-source-map.npz') as z:source_rim_rows=z['fat_headwall_source_face_rows']
assert np.array_equal(source_rim_rows,[int(facets[i]['face_source_row']) for i in rimrows])
vrole=np.where(used<len(ip),2,3).astype(np.uint8)
vrow=np.where(used<len(ip),used,used-len(ip)).astype(np.int64)
frole=np.concatenate((np.full(len(iff),2,np.uint8),np.full(len(dff),3,np.uint8),np.full(90,4,np.uint8)))
frow=np.concatenate((np.arange(len(iff)),np.arange(len(dff)),rimrows)).astype(np.int64)
flipped=np.concatenate((np.zeros(len(iff),bool),np.ones(len(dff),bool),np.zeros(90,bool)))
np.savez_compressed(out/'closed-fat-source-map.npz',output_vertex_input_rows=used,output_vertex_source_layer=vrole,output_vertex_source_row=vrow,face_source_layer=frole,face_source_face_row=frow,face_source_winding_reversed=flipped,headwall_original_source_face_rows=source_rim_rows)
# Reconstruct geometry solely through per-vertex typed source identities.
rebuild=np.empty_like(points)
sel=vrole==2;rebuild[sel]=ip[vrow[sel]];rebuild[~sel]=dp[vrow[~sel]]
assert np.array_equal(rebuild,points)
registry={'2':{'kind':'derived_fine_I_surface','path':str(I),'sha256':hashlib.sha256(I.read_bytes()).hexdigest(),'point_count':len(ip),'face_count':len(iff),'orientation':'retained'},'3':{'kind':'derived_fine_D_surface','path':str(D),'sha256':hashlib.sha256(D.read_bytes()).hexdigest(),'point_count':len(dp),'face_count':len(dff),'orientation':'reversed_for_fat_solid'},'4':{'kind':'registered_545_fat_headwall','path':str(facetpath),'sha256':hashlib.sha256(facetpath.read_bytes()).hexdigest(),'face_count':len(facets),'face_rows_domain':'CSV zero-based data-record rows','orientation':report['source330_pairing']['rim_orientation']}}
(out/'source-registry.json').write_text(json.dumps(registry,indent=2)+'\n')
receipt={'status':'PASS_EXACT_SOURCE_AND_COMPACTION_MAP','closed_boundary_sha256':hashlib.sha256((rdir/'fine-fat-closed-boundary.npz').read_bytes()).hexdigest(),'source_map_sha256':hashlib.sha256((out/'closed-fat-source-map.npz').read_bytes()).hexdigest(),'source_registry_sha256':hashlib.sha256((out/'source-registry.json').read_bytes()).hexdigest(),'points':len(points),'faces':len(faces),'unused_D_rows_excluded':len(sourcepoints)-len(used),'point_coordinates_unchanged':True,'face_order_and_winding_unchanged':True,'all_vertices_reconstructed_from_typed_parent_rows':True,'all_faces_reconstructed_from_typed_parent_rows_or_authenticated_rim':True,'physical_admission':False,'runtime':0,'accepted_steps':0,'note':'Original optional fine_I_faces/reversed_fine_D_faces in the geometry NPZ use the uncompressed concatenated input point domain. The geometry faces array uses compacted points. This source map names both domains explicitly without changing geometry.'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
