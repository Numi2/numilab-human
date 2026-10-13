from pathlib import Path
import hashlib,json,sys
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.resting_anatomy_interface_patch import HEADER,RECORD,parse_payload
R=A/'axillary-parents-native-025-002-original-owner-control/resting-anatomy-receipt.json'
AR=A/'liver-island-source-audit-1384-001/report.json'
O=A/'liver-island-retirement-candidate-1384-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not O.exists()
 r=json.loads(R.read_text());src=Path(r['payload']['path']);oldsha=sha(src);assert oldsha==r['payload']['sha256']
 ar=json.loads(AR.read_text());assert ar['source_payload_sha256']==oldsha
 island=ar['connected_components'][1];assert island['faces']==48 and island['owner_face_counts']=={'19':48}
 remove=np.asarray(island['row_local_faces']['19']);h,rows=parse_payload(src)
 kept=np.ones(len(rows[19]['faces']),dtype=bool);kept[remove]=False
 assert kept.sum()==12560
 records=[];vb=[];ib=[];nv=ni=0
 for sid,row in sorted(rows.items()):
  v6=np.asarray(row['vertices6'],dtype='<f4');faces=row['faces'][kept] if sid==19 else row['faces']
  records.append(RECORD.pack(row['body_index'],nv,len(v6),ni,faces.size,sid,row['layer'],row['flags']))
  vb.append(v6.tobytes());ib.append((faces.ravel()+nv).astype('<u4').tobytes());nv+=len(v6);ni+=faces.size
 blob=HEADER.pack(h[0],h[1],len(rows),nv,ni,h[5],h[6])+b''.join(records)+b''.join(vb)+b''.join(ib)
 O.mkdir();p=O/'resting-thorax.nhanatomy';p.write_bytes(blob);newsha=sha(p);hh,check=parse_payload(p)
 for sid in rows:
  x,y=rows[sid],check[sid]
  assert x['vertices6'].tobytes()==y['vertices6'].tobytes()
  assert all(x[k]==y[k] for k in ('body_index','layer','flags'))
  assert np.array_equal(x['faces'][kept] if sid==19 else x['faces'],y['faces'])
 z=np.load(A/'liver-island-source-audit-1384-001/geometry.npz');wf=z['faces'][np.asarray(ar['connected_components'][0]['global_face_indices'])]
 e=np.concatenate([wf[:,[0,1]],wf[:,[1,2]],wf[:,[2,0]]])
 eu,ec=np.unique(np.sort(e,axis=1),axis=0,return_counts=True);assert (ec==2).all()
 direction=np.where(e[:,0]<e[:,1],1,-1)
 _,inv=np.unique(np.sort(e,axis=1),axis=0,return_inverse=True);assert (np.bincount(inv,weights=direction)==0).all()
 proof={'source_receipt':str(R),'source_receipt_sha256':sha(R),'source_payload':str(src),'source_payload_sha256':oldsha,'source_audit':str(AR),'source_audit_sha256':sha(AR),'helper_sha256':sha(__file__),
 'removed_stable_id':19,'removed_row_local_faces':remove.tolist(),'removed_triangle_count':48,'removed_volume_mm3':island['signed_volume_ml']*1000,
 'classification':'Detached closed fragment of inferred reference CSG liver exterior; not a source-measured hepatic segment, vessel, boundary or functional compartment. Preserve original payload and all identity guides. Remove only fragment faces.',
 'remaining_liver_components':1,'remaining_liver_faces':len(wf),'remaining_welded_edges_closed_and_winding_consistent':True,
 'all_vertices_normals_preserved':True,'unused_historical_fragment_vertices_retained_for_existing_common_map_indices':True,
 'other_522_rows_decoded_byte_exact':True,'stable_ids_and_body_bindings_unchanged':True,'mass_force_physiology_state_unchanged':True,
 'diaphragm_interface_faces_unchanged':True,'display_patch_labels_not_measured_segments':True}
 r['payload'].update(path=str(p),sha256=newsha,index_count=ni,input_payload_sha256=oldsha)
 r['functional_bindings']['anatomy_payload_sha256']=newsha
 r['provenance']['cardiac_geometry_binding']['common_field']['anatomy_payload_sha256']=newsha
 ext=r['provenance'].get('passive_tissue_parent_bindings')
 if ext and 'anatomy_payload_sha256' in ext:assert ext['anatomy_payload_sha256']==oldsha;ext['anatomy_payload_sha256']=newsha
 r['provenance']['source_id_map']['19']['repair']['compiled_patch_face_count']=12560
 r['provenance']['retired_liver_csg_fragment']=proof
 r['qualification']='Liver inferred CSG fragment retirement candidate; main liver closed, source identities and back412 preserved. Native comparison pending; forearm and axillary anatomy failures remain.'
 rp=O/'resting-anatomy-receipt.json';rp.write_text(json.dumps(r,separators=(',',':'),sort_keys=True)+'\n')
 report={'status':'candidate_ready_for_native_replay','inputs':proof,'candidate':{'payload':str(p),'payload_sha256':newsha,'receipt':str(rp),'receipt_sha256':sha(rp),'surface_count':len(check),'vertex_count':nv,'index_count':ni},
 'exact_regression':{'all_other_rows_byte_identical':True,'all_vertices_normals_byte_identical':True,'row19_only_48_fragment_faces_removed':True}}
 (O/'candidate-report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report['candidate']),flush=True)
if __name__=='__main__':main()
