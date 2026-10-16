from pathlib import Path
import hashlib,json,sys,time,traceback
import numpy as np
H=Path('/Users/n/numi-human-anatomy-increment-publish-20261016');sys.path.insert(0,str(H/'src'))
from numilab_human import resting_anatomy_verify as v,surface_precision_retriangulation as sr,cardiac_cavity_intersections as ci,gate_preparation_boundary as gp,resting_shared_partition_verify as sp,passive_attachment_composition as c
D=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378')
P=D/'fdp761-fcr117-source768-native-emission-770/native-run/accepted-geometry/step-0.mrvpack'
R=D/'fdp761-fcr117-owner-rest-conformance-768/resting-anatomy-receipt.json'
O=D/'fdp133-fcr117-actual770-paired-local-retessellation-777'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not O.exists()
pins=[Path(__file__),P,P.with_suffix('.receipt.json'),R,D/'fdp761-fcr117-native770-owner-and-interface-proof-773/report.json',*[Path(m.__file__) for m in (v,sr,ci,gp,sp,c)]]
pre=gp.freeze_preparation(O.with_suffix('.preparation.json'),dict(status='PASS_PINNED_PAIRED_LOCAL_RETESSELLATION',predicate_phase_started=False,scope='773actualGPUFDP1336self/FCR117zero;1517commonpointsbitexact,873unrelatedgeometryandacceptedstatebitexact754. Bound4positionpreservinglocalflips, pairedreciprocalflipifbothincidentfacescommon; rejectone-sidedinterfaceboundaryflipsorunprovednewcommonfacets. Authenticatefullparentmaterialtokens andactualorderedsourcefacetmapping, notcoordinateidentityalone. Allsourcepositions/fullfields/allocations/attachmentroutesunchanged. Dualfaceancestry, strictself0/closedFDP4FCR1 required. Noanatomy/nativequalification.'),runner_argv=[sys.executable,*sys.argv],input_pins={str(p):sha(p) for p in pins})
gp.verify_frozen_preparation(pre);O.mkdir();start=time.monotonic()
try:
 cap,_=v.load_capture(P,P.with_suffix('.receipt.json'));receipt=json.loads(R.read_text());T=Path(receipt['provenance']['native_muscle_surfaces']['payload_path']);data=c._read_nhtiss4(T)
 meshes={};tokens={};ancestry={};pairs={}
 for sid in (117,133):
  p,f,n,g=cap[(51005,sid)];meshes[sid]=[p,f.copy(),g,n];ancestry[sid]=[dict(native770_source_face=i) for i in range(len(f))]
  row=next(z for z in data['records'] if int(z[6])==sid);arrays=c._nhtiss_row_arrays(data,row)
  _,proof=v.retain_bound_source_faces((51005,sid),cap[(51005,sid)],arrays['faces'],arrays['first_vertex'],np.arange(len(f),dtype=np.int64))
  source_tokens,_=sp._parent_carrier_material((51005,sid),sha(T),dict(path=str(R),sha256=sha(R)),arrays['vertex_count'])
  slots=g.astype(np.int64)-proof['source_capture_vertex_namespace_offset']-arrays['first_vertex']
  tokens[sid]=[(source_tokens[int(slot)],p[i].tobytes()) for i,slot in enumerate(slots)]
  audit=v.prepare_surface((51005,sid),p,f,normals=n,source_vertex_ids=g).self_audit();write(O/f'input-{sid}-self.json',audit);pairs[sid]=set(tuple(row) for row in audit['triangle_pairs'])
 assert len(pairs[117])==0 and len(pairs[133])==6,'773actualselfwitnesscountdrift'
 def face_map(sid,faces):
  result={}
  for fi,face in enumerate(faces):
   key=tuple(sorted(tokens[sid][int(i)] for i in face))
   if key in result:raise ValueError('ambiguous full-parent facet token')
   result[key]=fi
  return result
 def common_maps(faces):
  a=face_map(117,faces[117]);b=face_map(133,faces[133]);keys=set(a)&set(b)
  return keys,{b[k]:a[k] for k in keys}
 history=[]
 for iteration in range(4):
  if not any(pairs.values()):break
  sid=max(pairs,key=lambda key:len(pairs[key]));other=117 if sid==133 else 133;p,f,g,n=meshes[sid]
  adjacency=sr.build_surface_adjacency(f);original_common,common=common_maps({key:row[1] for key,row in meshes.items()})
  if sid==117:common={v:k for k,v in common.items()}
  edges=sorted({tuple(sorted((int(a),int(b)))) for pair in pairs[sid] for fi in pair for a,b in zip(f[fi],np.roll(f[fi],-1))})
  choices=[];chosen=None
  for edge in edges:
   incident=adjacency.edge_faces.get(edge,())
   if len(incident)!=2:continue
   affected={key:{pair for pair in pairs[key] if set(pair)&set(incident)} if key==sid else set() for key in meshes}
   if not affected[sid]:continue
   count=sum(fi in common for fi in incident)
   if count==1:continue
   try:
    trial,proof=sr.flip_interior_edge(p,f,edge=edge,face_indices=incident,minimum_altitude_m=1e-8,adjacency=adjacency,repair_folded_source=True)
    trials={key:row[1] for key,row in meshes.items()};trials[sid]=trial;changed={sid:tuple(incident)};proofs={sid:proof}
    if count==2:
     counterpart=tuple(common[fi] for fi in incident);op,of,og,on=meshes[other];lookup={token:i for i,token in enumerate(tokens[other])};otheredge=tuple(lookup[tokens[sid][i]] for i in edge);oadj=sr.build_surface_adjacency(of)
     assert set(oadj.edge_faces.get(tuple(sorted(otheredge)),()))==set(counterpart),'pairededgefanmismatch'
     ot,oprf=sr.flip_interior_edge(op,of,edge=otheredge,face_indices=counterpart,minimum_altitude_m=1e-8,adjacency=oadj,repair_folded_source=True)
     trials[other]=ot;changed[other]=counterpart;proofs[other]=oprf
     affected[other]={pair for pair in pairs[other] if set(pair)&set(counterpart)}
    after_common,_=common_maps(trials)
    if count==0:assert after_common==original_common,'independentflipchangedcommonmaterialfacetset'
    else:
     oldkeys={tuple(sorted(tokens[sid][int(i)] for i in f[fi])) for fi in incident}
     newkeys={tuple(sorted(tokens[sid][int(i)] for i in trial[fi])) for fi in incident}
     assert after_common==(original_common-oldkeys)|newkeys,'pairedflipchangedunrelatedcommonfacets'
    residual={};locals={}
    for owner,indices in changed.items():
     prepared=v.prepare_surface((51005,owner),meshes[owner][0],trials[owner]);local=ci._audit_self_pairs_touching_faces(prepared.records,list(indices));locals[owner]=local;residual[owner]=(pairs[owner]-affected[owner])|{tuple(x) for x in local['triangle_pairs']}
    score=sum(len(residual.get(owner,pairs[owner])) for owner in meshes)
    choices.append(dict(edge=list(edge),changed_faces={str(k):list(q) for k,q in changed.items()},paired=count==2,remaining_self=score))
    if score<sum(len(q) for q in pairs.values()):chosen=(trials,changed,proofs,residual,score);break
   except (ValueError,AssertionError) as error:choices.append(dict(edge=list(edge),rejected=str(error)));continue
  write(O/f'census-{iteration}.json',choices)
  if chosen is None:raise ValueError('No improving interface-preservingflip; switchlocalpatchremesh')
  trials,changed,proofs,residual,score=chosen
  for owner,indices in changed.items():
   prior=[ancestry[owner][fi] for fi in indices]
   for fi in indices:ancestry[owner][fi]=dict(inferred_float32_local_retriangulation=True,dual_source_ancestry=prior,prior_source_face_ids=list(indices),paired_shared_material=(len(changed)==2))
   meshes[owner][1]=trials[owner];pairs[owner]=residual[owner]
  history.append(dict(iteration=iteration,changed_faces={str(k):list(q) for k,q in changed.items()},construction={str(k):q for k,q in proofs.items()},remaining_self={str(k):len(q) for k,q in pairs.items()}));write(O/'history.json',history)
 audits=[]
 for sid,(p,f,g,n) in meshes.items():
  np.savez_compressed(O/f'row-{sid}-candidate.npz',capture_positions_m=p,capture_faces=f,capture_global_vertex_ids=g)
  write(O/f'row-{sid}-face-ancestry.json',ancestry[sid])
  surface=v.prepare_surface((51005,sid),p,f);audit=surface.self_audit();write(O/f'row-{sid}-full-self.json',audit);audits.append(dict(stable_id=sid,self_count=audit['count'],closed=surface.closed,components=surface.quotient_topology['face_component_count']))
 passed=all(row['self_count']==0 and row['closed'] and row['components']==(4 if row['stable_id']==133 else 1) for row in audits)
 write(O/'report.json',dict(status='PASS_PAIRED_LOCAL_RETESSELLATION_ONLY' if passed else 'FAIL_PAIRED_LOCAL_RETESSELLATION',audits=audits,history=history,positions_material_and_ids_unchanged=True,source_pins={str(p):sha(p) for p in pins},elapsed_seconds=time.monotonic()-start,anatomical_admission=False,native_admission=False));assert passed
except BaseException as error:
 write(O/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start));raise
