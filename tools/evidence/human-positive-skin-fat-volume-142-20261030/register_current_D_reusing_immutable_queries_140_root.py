"""Reuse exact immutable-parent queries for the current typed D; use 230 registration."""
from pathlib import Path
import numpy as np,json,csv,hashlib,sys,importlib.util,time
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');ROOT=H.parent
OUT=H/'registered-current-refined-D-140-root'
D=H/'fat-inferred-boundary-D-refinement-128-v2/candidate-D-inferred-boundary-refined-typed.npz'
PROOF=H/'D-refinement-exact-geometry-audit-137-root/report.json'
DERIVATION=H/'fat-inferred-boundary-D-refinement-128-v2/receipt.json'
OLD=H/'downstream-003-current-typed-228/registered-sidecars-228'
BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
POSE=BASE.parent/'registered-skin-poses-100.json'
OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def key(p):return tuple(np.asarray(p,dtype='<f4').view('<u4').tolist())
def main():
 assert not OUT.exists();OUT.mkdir();started=time.monotonic();proof=json.loads(PROOF.read_text());dr=json.loads(DERIVATION.read_text())
 assert proof['status']=='PASS_EXACT_CURRENT_D_BOUNDARY_REFINEMENT_DELTA'and proof['sources']['D']['sha256']==sha(D)
 assert sha(D)=='ff73291bce77dda81eff985fecdbba1b198f24e95a7f8f9af1d6058172d4da45'
 assert sha(DERIVATION)=='a2800f08914ff3c7a4b968009887130216ff0371380c323ca1f91559a1f1b150'
 runner=imp('actual697D140',H/'current-D-proof-015/handoff-source004/run_native_domain_217_typedD_697.py');m=runner.load_typed_owner();typed=m.load_surface(D,'D')
 with np.load(D)as z:
  raw={k:z[k]for k in z.files}
 assert np.array_equal(typed['points'],raw['points_world_m'])and np.array_equal(typed['faces'],raw['faces'])
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 reg=imp('reg230D140',OWNER)
 parent,index,material_indices,pm=reg.load_parent_current(BASE);frames,pmeta,bodies=reg.load_frames(POSE);weights,wm=reg.load_parent_full86_weights(Path(str(BASE)+'.weights.csv'),parent,pm)
 old=json.loads((OLD/'receipt.json').read_text())
 for k in ['nodeSHA256','massSHA256','sourceSHA256','tetsSHA256','labelsSHA256']:assert pm[k]==old['parentCurrent'][k]
 assert pmeta['sha256']==old['poseSnapshot']['sha256']and wm['sha256']==old['parentFull86Weights']['sha256']
 assert old['limits']['maximumParentRegistrationDistanceM']==.02 and old['limits']['tieToleranceM']==1e-8
 oldcsv=OLD/'registered-D.csv';assert sha(oldcsv)==old['outputSHA256'][oldcsv.name]
 wanted={key(p)for p in typed['points']};cache={};bad=set();cache_rows=0
 with oldcsv.open(newline='')as f:
  for row in csv.DictReader(f):
   p=np.array([float(row[k])for k in ['wx','wy','wz']],float)
   if not np.array_equal(p,p.astype(np.float32).astype(float)):continue
   k=key(p)
   if k not in wanted:continue
   tet=int(row['parent_tet_row']);b=np.array([float(row['parent_bary'+str(i)])for i in range(4)])
   residual=np.array([float(row['parent_residual_'+a+'_m'])for a in 'xyz']);dist=float(row['parent_registration_distance_m'])
   r={'tet':tet,'bary':b,'residual':residual,'projected':p-residual,'distance':dist,'tieCount':int(row['parent_registration_tie_count']),'ties':json.loads(row['parent_tied_tet_rows_json']),'material':int(row['parent_material_index']),'donor':int(row['parent_geometric_donor_body']),'region':int(row['parent_anatomical_region_id']),'stableFace':int(row['parent_stable_source_face_id'])}
   assert r['material']==16 and r['material']==int(parent['materials'][tet])and r['donor']==int(parent['donors'][tet])and r['region']==int(parent['regions'][tet])and r['stableFace']==int(parent['faces'][tet])
   assert abs(float(b.sum())-1)<1e-8 and b.min()>=-1e-8 and np.linalg.norm((b@parent['points'][parent['tets'][tet]])+residual-p)<1e-11 and abs(np.linalg.norm(residual)-dist)<1e-11
   if k in cache and (cache[k]['tet']!=tet or cache[k]['ties']!=r['ties']or not np.array_equal(cache[k]['bary'],b)):bad.add(k)
   else:cache[k]=r
   cache_rows+=1
 for k in bad:cache.pop(k,None)
 actual_closest=reg.closest_same_material;counts={'reused':0,'fresh':0}
 def cached(parent0,index0,mats0,material,p,maxdist,tietol):
  k=key(p)
  if material==16 and maxdist==.02 and tietol==1e-8 and k in cache:
   counts['reused']+=1;return cache[k]
  counts['fresh']+=1;return actual_closest(parent0,index0,mats0,material,p,maxdist,tietol)
 reg.closest_same_material=cached
 arrays={k:raw[k]for k in ['face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1']}
 schema=json.loads(str(raw['face_source_identity_schema_by_original_id_json'][0]))
 ancestry={str(s):{'outputFaces':int(np.sum(raw['face_source_original_id']==s)),'sourceIdentitySchema':schema[str(s)],'typedConsumerProofSHA256':sha(DERIVATION),'geometryProofSHA256':sha(PROOF),'scope':'Typed union ancestry, not a claim of one exact measured source face.'}for s in np.unique(raw['face_source_original_id'])}
 surface={'path':str(D),'sha256':sha(D),'layer':'D','role':'deepSurface','roleIndex':3,'domain':reg.DOMAIN,'vertexCount':len(typed['points']),'faceCount':len(typed['faces']),'dtype':'float32','points':typed['points'].astype(float),'faces':typed['faces'],'sourceSchema':schema,'faceAncestry':ancestry,'faceArrays':arrays,'sourceOriginalFaceIDs':sorted(map(int,np.unique(raw['face_source_original_id']))),'sourceRowsAllPointRowsDirect':True,'sourceParentLineagePreservedPerFace':True}
 (OUT/'progress.json').write_text(json.dumps({'status':'REGISTERING_CURRENT_D','cache_rows':cache_rows,'unique_reusable_queries':len(cache),'conflicting_cache_queries_recomputed':len(bad),'expected_nodes':len(typed['points']),'physical_admission':False},indent=2)+'\n')
 result=reg.register_layer(surface,parent,index,material_indices,frames,weights,bodies,.02,1e-8)
 for name,header,rows in [('registered-D.csv',reg.NODE_HEADER,result['nodes']),('registered-D-faces.csv',reg.FACE_HEADER,result['faces']),('parent-tet-full-weights-D.csv',reg.WEIGHT_HEADER,result['weightRows']),('parent-tet-full-weight-owners-D.csv',reg.WEIGHT_OWNER_HEADER,result['weightOwnerRows']),('node-parent-registration.csv',reg.PARENT_HEADER,result['registrations']),('donor-frames.csv',reg.FRAME_HEADER,reg.frame_csv_rows(frames,bodies))]:reg.write_csv(OUT/name,header,rows)
 (OUT/'source-domain-registry.json').write_text(json.dumps([result['sourceDomain']],indent=2)+'\n')
 if result.get('exactNearTieFallbacks'):(OUT/'exact-parent-near-tie-fallbacks-D.json').write_text(json.dumps(result['exactNearTieFallbacks'],indent=2)+'\n')
 receipt={'schema':'registered-authoring-surface-sidecars-receipt.v1','status':'reference_registration_ready_geometry_not_admitted','physical_admission':False,'geometry_admission':False,'native_steps':0,'selectedRoles':['D'],'candidateSurfaces':{'D':{'path':str(D),'sha256':sha(D),'points':len(typed['points']),'faces':len(typed['faces']),'sourceFaceAncestry':ancestry}},'sourceDomainRegistry':[result['sourceDomain']],'parentCurrent':pm,'poseSnapshot':pmeta,'parentFull86Weights':wm,'bindingBodyByLane':bodies,'results':{'D':result['metrics']},'limits':old['limits'],'typedSurfaceLoader':'actual697/004 typed-union consumer plus exact pinned child derivation and current whole-coverage geometry proof; legacy230 schema-only gate was not asserted','sourceDerivationReceipt':{'path':str(DERIVATION),'sha256':sha(DERIVATION)},'sourceGeometryReceipt':{'path':str(PROOF),'sha256':sha(PROOF)},'registrationReuse':{'method':'Memoize the existing230 closest-same-material query only for exact identical world coordinate/material, immutable parent geometry/labels/mass, limits and frames. Recompute all normals, full86 interpolation, direct source-row identity and every changed/new query.','sourceReceipt':str(OLD/'receipt.json'),'sourceReceiptSHA256':sha(OLD/'receipt.json'),'sourceCSV':str(oldcsv),'sourceCSVSHA256':sha(oldcsv),'counts':counts,'conflictingQueriesRecomputed':len(bad)},'sourceOwner':{'path':str(OWNER),'sha256':sha(OWNER)},'elapsed_s':time.monotonic()-started}
 receipt['outputSHA256']={p.name:sha(p)for p in OUT.iterdir()if p.is_file()and p.name!='progress.json'}
 (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':receipt['status'],'counts':counts,'nodes':len(typed['points']),'elapsed_s':receipt['elapsed_s'],'receipt_sha256':sha(OUT/'receipt.json')}),flush=True)
if __name__=='__main__':main()
