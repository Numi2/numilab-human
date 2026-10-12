#!/usr/bin/env python3
import json,hashlib,importlib.util,sys,os,math,time,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
R=Path('/Users/n/numi-human-back-m6-1349');OUT=R/'attempt-407-synchronized-step2655-row156-interface-sliver-repair-001';OUT.mkdir(parents=True,exist_ok=True)
D366=R/'attempt-366-cap-multi-ulp-conditioned-edgeflips-001/candidate-before-audit.json';D402=R/'attempt-402-step31-raw-parent-subresolution-edge-contraction-001/candidate.json';P399=R/'attempt-399-sixrow-raw-parent-only-pose-audit-001'
CAP31=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/back-native-014-raw-parent-003/baseline/native-run/rejected-surface-audit/step-31.mrvpack')
CAP2655=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/back-native-014-raw-parent-005/baseline/native-run/rejected-surface-audit/step-2655.mrvpack')
CENSUS=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/back-sliver-altitude-census-001/report.json')
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci

def imp(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
M=imp('back364',R/'attempt-364-paired-indexed-source-edge-contraction-001.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
def parse_q(q):return tuple(F(int(a),int(b)) for a,b in q)
def parse366term(t):
 sid,vi,num,den=t;sid=int(sid);return (0 if sid in (153,155) else 1,sid,int(vi),F(int(num),int(den)))
def parse402term(t):return (int(t[0]),int(t[1]),int(t[2]),F(int(t[3]),int(t[4])))
def qjson(q):return [[str(x.numerator),str(x.denominator)] for x in q]
def term5(t):return [int(t[0]),int(t[1]),int(t[2]),str(t[3].numerator),str(t[3].denominator)]
def f32zeros(xyz,fs):
 lat=[ci.float32_point_lattice_key(v) for v in np.asarray(xyz,dtype='<f4')];out=[]
 for i,f in enumerate(fs):
  if f is None:continue
  a,b,c=(lat[int(v)] for v in f);ab=tuple(y-x for x,y in zip(a,b));ac=tuple(y-x for x,y in zip(a,c));n=(ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0])
  if not any(n):out.append(i)
 return out
def opposite(pa,pb):return any(pa==[pb[(k-r)%3] for r in range(3)] for k in range(3))
def vol(top):return sum((F(c['signed_volume_m3']) for c in top['components']),F(0))
def tri_area(pts,f):
 a,b,c=[tuple(float(x) for x in pts[v]) for v in f];u=tuple(b[k]-a[k] for k in range(3));v=tuple(c[k]-a[k] for k in range(3));n=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);return .5*math.sqrt(sum(x*x for x in n))

t0=time.monotonic();d=json.load(open(D366));desc_doc=json.load(open(D402));dg=desc_doc['geometry']
pts={};maps={};faces={};attrs={}
for s in (153,154,155,156):
 k=str(s);pts[s]=[parse_q(q) for q in d['points_exact'][k]];maps[s]=[tuple(sorted(parse366term(t) for t in ts)) for ts in d['parent_maps'][k]];faces[s]=[None if f is None else tuple(map(int,f)) for f in d['faces'][k]];attrs[s]=d['face_ancestry'][k]
for s in (157,158):
 k=str(s);pts[s]=[parse_q(q) for q in dg['points_exact'][k]];maps[s]=[tuple(parse402term(t) for t in ts) for ts in dg['parent_maps'][k]];faces[s]=[None if f is None else tuple(map(int,f)) for f in dg['faces'][k]];attrs[s]=dg['face_ancestry'][k]
source_pts={s:list(pts[s]) for s in pts};source_maps={s:list(maps[s]) for s in maps};source_faces={s:list(faces[s]) for s in faces};source_attrs={s:list(attrs[s]) for s in attrs}
PAIRS=((153,155),(154,156),(155,157),(156,158),(157,158));BASE_COUNTS={(153,155):816,(154,156):815,(155,157):457,(156,158):451,(157,158):254}
base_top={s:M.row_topology(faces[s],pts[s]) for s in range(153,159)};assert all(t['ok'] for t in base_top.values())
base_builder=M.Builder(pts,maps,faces);base_pairs={ab:base_builder.face_pairs(ab)[0] for ab in PAIRS}
for ab in PAIRS:assert len(base_pairs[ab])==BASE_COUNTS[ab] and not base_builder.face_pairs(ab)[1],(ab,len(base_pairs[ab]))
base_unions={ab:base_builder.union(ab) for ab in PAIRS}
# The step2655 failing face4100 is in source row156 face 3901. Its paired interface facet4104/8113 requires a synchronized edit.
interface_pair=next((i,j) for i,j in base_pairs[(156,158)] if i==4104);assert interface_pair==(4104,8113),interface_pair
fac156=faces[156][4104];fac158=faces[158][8113];vertex_match={}
for v in fac156:
 cand=[w for w in fac158 if pts[156][v]==pts[158][w] and maps[156][v]==maps[158][w]]
 assert len(cand)==1,(v,cand);vertex_match[v]=cand[0]
assert vertex_match[2360]==343 and vertex_match[2362]==341,vertex_match
operations=[];removed_by_row={};changed_by_row={};remaps={};affected_by_row={};volumes={};area_ledger={}
def contract(s,keep,drop,expected_parent_faces=None):
 oldp=list(pts[s]);oldm=list(maps[s]);oldf=list(faces[s]);olda=list(attrs[s]);inc=[i for i,f in enumerate(oldf) if f is not None and drop in f]
 assert M.edge_link_condition(oldf,keep,drop)['ok'],(s,keep,drop,M.edge_link_condition(oldf,keep,drop))
 if expected_parent_faces is not None:assert all(olda[i]['parent_face_index'] in expected_parent_faces for i in inc),(s,[(i,olda[i]) for i in inc])
 changed=[];removed=[];trial=[]
 for i,f in enumerate(oldf):
  if f is None:trial.append(None);continue
  nf=tuple(keep if x==drop else x for x in f)
  if len(set(nf))<3:trial.append(None);removed.append({'face_index':i,**olda[i],'collapsed_by_edge':[keep,drop]});continue
  if drop in f:
   assert M.dot(M.normal(tuple(oldp[x] for x in f)),M.normal(tuple(oldp[x] for x in nf)))>0,(s,i,'normal inversion')
   changed.append(i)
  trial.append(nf)
 mp={i:(i if i<drop else i-1) for i in range(len(oldp)) if i!=drop}
 faces[s]=[None if f is None else tuple(mp[x] for x in f) for f in trial];pts[s]=[q for i,q in enumerate(oldp) if i!=drop];maps[s]=[q for i,q in enumerate(oldm) if i!=drop]
 remaps[s]=mp;removed_by_row[s]=removed;changed_by_row[s]=changed;affected_by_row[s]=inc
 shift=math.dist(tuple(map(float,oldp[drop])),tuple(map(float,oldp[keep])))
 area0=sum(tri_area(oldp,oldf[i]) for i in inc);area1=sum(tri_area(pts[s],faces[s][i]) for i in inc if faces[s][i] is not None)
 operations.append({'row':s,'keep_old_vertex':keep,'drop_old_vertex':drop,'old_edge_length_m':shift,'max_vertex_reassignment_m':shift,'incident_faces':inc,'removed_faces':removed,'changed_surviving_faces':changed,'incident_parent_stable_ids':sorted({olda[i]['parent_stable_id'] for i in inc}),'incident_parent_face_indices':sorted({olda[i]['parent_face_index'] for i in inc}),'parent_face_ancestry_retained_for_every_survivor':True,'incident_patch_area_before_m2':area0,'incident_patch_area_after_m2':area1,'incident_patch_area_delta_m2':area1-area0,'old_vertex_count':len(oldp),'new_vertex_count':len(pts[s])})
 return shift
shift156=contract(156,2360,2362,{3901});shift158=contract(158,343,341,{3901,6279,6280})
assert abs(shift156-shift158)<1e-15 and shift156<1e-6,(shift156,shift158)
assert [x['face_index'] for x in removed_by_row[156]]==[4100,4104]
assert [x['face_index'] for x in removed_by_row[158]]==[266,8113]
# The synchronized interface contraction must only remove the paired 4104/8113 facet; all other source facets retain identity.
post_builder=M.Builder(pts,maps,faces);post_pairs={ab:post_builder.face_pairs(ab)[0] for ab in PAIRS}
POST_COUNTS={(153,155):816,(154,156):815,(155,157):457,(156,158):450,(157,158):254}
interfaces={}
for ab in PAIRS:
 pairs,bad=post_builder.face_pairs(ab);assert len(pairs)==POST_COUNTS[ab] and not bad,(ab,len(pairs),bad[:2])
 interfaces[f'{ab[0]}+{ab[1]}']={'before':len(base_pairs[ab]),'after':len(pairs),'removed_pairs':[[i,j] for i,j in base_pairs[ab] if (i,j) not in set(pairs)],'identity_orientation_errors':len(bad)}
ok,why=M.assert_interface_preserved(base_pairs,post_builder);assert ok,why
assert interfaces['156+158']['removed_pairs']==[[4104,8113]],interfaces['156+158']
# Save build script, schemas, and exact candidate checkpoints before pose and topology audits.
upper=json.loads(json.dumps(d));upper['points_exact']['156']=[q for i,q in enumerate(d['points_exact']['156']) if i!=2362];upper['parent_maps']['156']=[d['parent_maps']['156'][i] for i in range(len(source_maps[156])) if i!=2362]
upper['faces']['156']=[None if f is None else list(f) for f in faces[156]];upper['removed_face_ancestry']={'156':removed_by_row[156]};upper['operation']={'synchronized_pair':[4104,8113],'keep_old_vertex':2360,'drop_old_vertex':2362,'source_parent_stable_id':156,'source_parent_face_index':3901,'removed_face_indices':[4100,4104],'candidate_admission':False,'native_admission':False};upper['candidate_admission']=False;upper['native_admission']=False;upper['schema']='numi.human.trapezius.attempt407-upper-candidate-v1'
write(OUT/'upper_candidate.json',upper)
desc_out=json.loads(json.dumps(desc_doc));gout=desc_out['geometry'];gout['points_exact']['158']=[qjson(q) for q in pts[158]];gout['parent_maps']['158']=[dg['parent_maps']['158'][i] for i in range(len(source_maps[158])) if i!=341];gout['faces']['158']=[None if f is None else list(f) for f in faces[158]];gout.setdefault('old_to_new_vertex_index',{})['158']={str(k):v for k,v in remaps[158].items()};gout.setdefault('removed_face_ancestry',{})['158']=removed_by_row[158];desc_out['candidate_admission']=False;desc_out['native_admission']=False;desc_out['paired_repair']={'keep_old_vertex':343,'drop_old_vertex':341,'paired_interface_facet_removed':[4104,8113]};write(OUT/'descending_overlay.json',desc_out)
full={'schema':'numi.human.trapezius.attempt407-full-six-row-source-candidate.v1','candidate_admission':False,'native_admission':False,'points_exact':{str(s):[qjson(q) for q in pts[s]] for s in range(153,159)},'parent_maps':{str(s):[[term5(t) for t in ts] for ts in maps[s]] for s in range(153,159)},'faces':{str(s):[None if f is None else list(f) for f in faces[s]] for s in range(153,159)},'face_ancestry':{str(s):attrs[s] for s in range(153,159)},'removed_face_ancestry':{str(s):removed_by_row[s] for s in (156,158)},'source_files':{'attempt366':{'path':str(D366),'sha256':sha(D366)},'attempt402':{'path':str(D402),'sha256':sha(D402)}},'map_term_schema':'[source_payload_index,source_stable_id,source_vertex_index,coefficient_numerator,coefficient_denominator]; all positive exact rational terms sum to 1; flattened raw-parent support preserved'}
full['operations']=operations;write(OUT/'full_candidate.json',full);shutil.copyfile(__file__,OUT/'build_attempt407.py');os.chmod(OUT/'build_attempt407.py',0o755)
# Emit descendant NPZ overlay for source loaders that require an NPZ alongside JSON.
keepfaces=[(i,f) for i,f in enumerate(faces[158]) if f is not None];np.savez_compressed(OUT/'candidate-row-158.npz',faces=np.asarray([f for i,f in keepfaces],dtype=np.int64),face_parent_stable_id=np.asarray([attrs[158][i]['parent_stable_id'] for i,f in keepfaces],dtype=np.uint32),face_parent_index=np.asarray([attrs[158][i]['parent_face_index'] for i,f in keepfaces],dtype=np.uint32),face_partition_role=np.asarray([attrs[158][i]['partition_role'] for i,f in keepfaces],dtype=np.uint8),face_origin_kind=np.asarray([attrs[158][i]['origin_kind'] for i,f in keepfaces],dtype=np.uint8),face_input_index=np.asarray([attrs[158][i]['input_index'] for i,f in keepfaces],dtype=np.int64))
# Exact source map integrity and child topology after checkpoint.
map_integrity={};topology={}
for s in range(153,159):
 bad=[i for i,ts in enumerate(maps[s]) if not ts or any(t[3]<=0 for t in ts) or sum((t[3] for t in ts),F(0))!=1];map_integrity[str(s)]={'vertex_count':len(maps[s]),'invalid_vertex_count':len(bad),'sample':bad[:5]};assert not bad,(s,bad[:4])
 tp=M.row_topology(faces[s],pts[s]);topology[str(s)]={'before_ok':base_top[s]['ok'],'after_ok':tp['ok'],'components_before':len(base_top[s]['components']),'components_after':len(tp['components']),'component_signs_before':[c['positive'] for c in base_top[s]['components']],'component_signs_after':[c['positive'] for c in tp.get('components',[])],'signed_volume_before_m3':str(vol(base_top[s])),'signed_volume_after_m3':str(vol(tp)),'exact_signed_volume_delta_m3':str(vol(tp)-vol(base_top[s])),'relative_abs_signed_volume_delta':float(abs(vol(tp)-vol(base_top[s]))/abs(vol(base_top[s])))};assert tp['ok'] and len(tp['components'])==len(base_top[s]['components']),(s,tp)
# Exact cancelled-interface union topology: compare with each untouched baseline; do not waive prior failures.
union_audit={}
for ab in PAIRS:
 bu=base_unions[ab];pu=post_builder.union(ab);bt=bu['topology'];pt=pu['topology'];same=(bt.get('reason')==pt.get('reason') and bt.get('bad_vertex_count')==pt.get('bad_vertex_count') and len(bt.get('components',[]))==len(pt.get('components',[])))
 union_audit[f'{ab[0]}+{ab[1]}']={'before_pair_count':bu['pair_count'],'after_pair_count':pu['pair_count'],'before_union_errors':len(bu['unpaired_signature_errors']),'after_union_errors':len(pu['unpaired_signature_errors']),'before_same_row_equivalence_count':bu['same_row_equivalence_count'],'after_same_row_equivalence_count':pu['same_row_equivalence_count'],'before_topology_reason':bt.get('reason'),'after_topology_reason':pt.get('reason'),'before_bad_vertex_count':bt.get('bad_vertex_count'),'after_bad_vertex_count':pt.get('bad_vertex_count'),'before_component_count':len(bt.get('components',[])),'after_component_count':len(pt.get('components',[])),'no_regression':same and bu['pair_count']-pu['pair_count']==len(interfaces[f'{ab[0]}+{ab[1]}']['removed_pairs'])};assert union_audit[f'{ab[0]}+{ab[1]}']['no_regression'],(ab,union_audit[f'{ab[0]}+{ab[1]}'])
# Four pinned poses: compiled step0/8000, and actual rejected native packs at steps31/2655.
pose_data={};capture_checks={};steps=(0,8000,31,2655)
for step in (0,8000):
 pose_data[step]={}
 for s in range(153,159):
  src=(R/'attempt-402-step31-raw-parent-subresolution-edge-contraction-001'/f'step-{step}-row-157-compiled-position-check.npz') if s==157 else (P399/f'step-{step}-compiled-row-{s}.npz')
  arr=np.load(src,allow_pickle=False)['positions'].astype('<f4');assert arr.shape==(len(source_pts[s]),3),(step,s,arr.shape,len(source_pts[s]))
  pose_data[step][s]=np.asarray([arr[i] for i in range(len(arr)) if i not in remaps[s].keys() if False],dtype='<f4') if False else arr
  if s in remaps[s] if False else False: pass
  if s in remaps and s in (156,158):
   drop=2362 if s==156 else 341;pose_data[step][s]=np.asarray([arr[i] for i in range(len(arr)) if i!=drop],dtype='<f4')
for step,cap in ((31,CAP31),(2655,CAP2655)):
 pos,surfs,meta=ca._pack_surfaces(cap,{(51005,s) for s in range(153,159)});pose_data[step]={};capture_checks[str(step)]={}
 old391=json.load(open(R/'attempt-391-paired-descending-sliver-contractions-001/candidate.json'))['geometry'] if step==31 else None
 remap157={int(k):int(v) for k,v in dg['old_to_new_vertex_index']['157'].items()} if step==31 else None
 inv157={v:k for k,v in remap157.items()} if step==31 else None
 for s in range(153,159):
  fg=np.asarray(surfs[(51005,s)]['faces'],dtype=np.int64);off=int(fg.min());local=fg-off
  if step==31 and s==157:
   oldf=[tuple(map(int,f)) for f in old391['faces']['157']]
   assert np.array_equal(local,np.asarray(oldf,dtype=np.int64)),(step,s,'rejected row157 pack differs from pinned attempt391 face list')
   expected_live=np.asarray([f for f in source_faces[s] if f is not None],dtype=np.int64)
   mapped=[]
   for f in oldf:
    ff=tuple(3929 if x==3926 else x for x in f)
    if len(set(ff))<3:continue
    mapped.append(tuple(remap157[x] for x in ff))
   assert np.array_equal(np.asarray(mapped,dtype=np.int64),expected_live),(step,s,'attempt391 to attempt402 face remap mismatch')
   oldxyz=pos[off:off+len(old391['points_exact']['157'])].astype('<f4')
   arr=np.asarray([oldxyz[inv157[i]] for i in range(len(pts[s]))],dtype='<f4')
   pose_data[step][s]=arr
   capture_checks[str(step)][str(s)]={'capture_face_map':'attempt391 exact; explicitly remapped through402 index map','capture_live_face_count':len(local),'candidate_live_face_count':len(expected_live),'vertex_offset':off,'source_vertex_count':len(source_pts[s])}
   continue
  live_source=np.asarray([f for f in source_faces[s] if f is not None],dtype=np.int64)
  assert np.array_equal(local,live_source),(step,s,'captured face map differs from pinned source candidate')
  arr=pos[off:off+len(source_pts[s])].astype('<f4');pose_data[step][s]=arr
  drop=(2362 if s==156 else 341 if s==158 else None)
  if drop is not None:pose_data[step][s]=np.asarray([arr[i] for i in range(len(arr)) if i!=drop],dtype='<f4')
  capture_checks[str(step)][str(s)]={'capture_live_face_count':len(local),'source_candidate_live_face_count':len(live_source),'vertex_offset':off,'source_vertex_count':len(source_pts[s])}
# Exact area and shared-interface Float32 identity checks over all six rows.
pose_degeneracy={};pose_interfaces={}
for step in steps:
 rows={}
 for s in range(153,159):
  z=f32zeros(pose_data[step][s],faces[s]);rows[str(s)]={'zero_area_count':len(z),'zero_area_faces':z[:20]};assert not z,(step,s,z[:5])
 pose_degeneracy[str(step)]=rows;iface={}
 for ab in PAIRS:
  pairs=post_pairs[ab];matches=0;bad=[]
  for i,j in pairs:
   pa=[tuple(pose_data[step][ab[0]][v].view(np.uint32).tolist()) for v in faces[ab[0]][i]]
   pb=[tuple(pose_data[step][ab[1]][v].view(np.uint32).tolist()) for v in faces[ab[1]][j]]
   if opposite(pa,pb):matches+=1
   else:bad.append([i,j])
  iface[f'{ab[0]}+{ab[1]}']={'expected':len(pairs),'bit_identical_opposite':matches,'mismatch_count':len(bad),'sample':bad[:3]};assert not bad,(step,ab,bad[:3])
 pose_interfaces[str(step)]=iface
# Instrument regression: capture 2655 source row156 face4100 must be exactly the raw zero-area witness.
pos,surfs,meta=ca._pack_surfaces(CAP2655,{(51005,156)});fg=np.asarray(surfs[(51005,156)]['faces'],dtype=np.int64);off=int(fg.min());raw=pos[off:off+len(source_pts[156])].astype('<f4');assert f32zeros(raw,source_faces[156])==[4100]
# Census pins every reported low-altitude count for follow-up batch conditioning; no thresholds waived here.
census=json.load(open(CENSUS));census_summary={step:{s:vals['below_altitude_um'].get('0.1',0) for s,vals in item['rows'].items()} for step,item in census['captures'].items()}
report={'schema':'numi.human.trapezius.attempt407-synchronized-row156-f32-sliver-repair.v1','candidate_admission':False,'native_admission':False,'candidate_files':{name:{'path':str(OUT/name),'sha256':sha(OUT/name)} for name in ('upper_candidate.json','descending_overlay.json','full_candidate.json','candidate-row-158.npz')},'builder_script':{'path':str(OUT/'build_attempt407.py'),'sha256':sha(OUT/'build_attempt407.py')},'inputs':{'attempt366_sha256':sha(D366),'attempt402_sha256':sha(D402),'step31_rejected_pack_sha256':sha(CAP31),'step2655_rejected_pack_sha256':sha(CAP2655),'sliver_census_path':str(CENSUS),'sliver_census_sha256':sha(CENSUS)},'operations':operations,'interface_pair_counts':interfaces,'all_surviving_source_interface_facets_exact_and_orientation_preserved':True,'row_exact_topology_and_volume':topology,'cancelled_interface_union_comparison':union_audit,'positive_parent_map_integrity':map_integrity,'native_capture_face_index_identity':capture_checks,'f32_exact_zero_area_by_pose':pose_degeneracy,'f32_shared_interface_bit_identity_by_pose':pose_interfaces,'confirmed_original_step2655_row156_zero_area_faces':[4100],'sliver_census_below_0_1um_by_pose':census_summary,'interpretation':'Two synchronized canonical endpoint contractions remove the one exact Float32 zero-area row156 face at accepted step2655. The 0.904um reassignment is applied to both sides of the exact 156/158 seam; both copies retain identical positive raw-parent maps. The 156/158 interface loses exactly matched face pair 4104/8113. All five interface guards remain consistent, and affected face ancestry is retained. This is a bounded inferred remesh, not measured anatomy or a native admission. Existing unrelated union failures in 155+157 and 157+158 are preserved unchanged; low-altitude faces in the retained census require subsequent native/full-pose screening and are not declared cleared.'}
write(OUT/'report.json',report)
print(json.dumps({'phase':'complete','candidate_files':report['candidate_files'],'report_sha256':sha(OUT/'report.json'),'builder_sha256':report['builder_script']['sha256'],'operations':operations,'interfaces':interfaces,'pose_degeneracy':pose_degeneracy,'pose_interfaces':pose_interfaces,'union_audit':union_audit},sort_keys=True),flush=True)
