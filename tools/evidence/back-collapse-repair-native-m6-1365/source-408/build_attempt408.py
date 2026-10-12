#!/usr/bin/env python3
import json,hashlib,importlib.util,sys,os,math,copy,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
R=Path('/Users/n/numi-human-back-m6-1349')
BASE=R/'attempt-407-synchronized-step2655-row156-interface-sliver-repair-001'
OUT=R/'attempt-408-row157-native7967-subresolution-edge-repair-001';OUT.mkdir(parents=True,exist_ok=False)
sp=importlib.util.spec_from_file_location('M',R/'attempt-364-paired-indexed-source-edge-contraction-001.py');M=importlib.util.module_from_spec(sp);sys.modules['M']=M;sp.loader.exec_module(M)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
def qparse(x):return tuple(F(int(a),int(b)) for a,b in x)
def tparse(x):return (int(x[0]),int(x[1]),int(x[2]),F(int(x[3]),int(x[4])))
def qjson(q):return [[str(x.numerator),str(x.denominator)] for x in q]
def tjson(t):return [int(t[0]),int(t[1]),int(t[2]),str(t[3].numerator),str(t[3].denominator)]
def vol(top):return sum((F(c['signed_volume_m3']) for c in top['components']),F(0))
def tri_area(pts,f):
 a,b,c=[tuple(float(x) for x in pts[v]) for v in f];u=tuple(b[k]-a[k] for k in range(3));v=tuple(c[k]-a[k] for k in range(3));n=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);return .5*math.sqrt(sum(x*x for x in n))
full=copy.deepcopy(json.load(open(BASE/'full_candidate.json')))
pts={s:[qparse(x) for x in full['points_exact'][str(s)]] for s in range(153,159)}
maps={s:[tuple(tparse(t) for t in x) for x in full['parent_maps'][str(s)]] for s in range(153,159)}
faces={s:[None if x is None else tuple(map(int,x)) for x in full['faces'][str(s)]] for s in range(153,159)}
attrs={s:copy.deepcopy(full['face_ancestry'][str(s)]) for s in range(153,159)}
PAIRS=((153,155),(154,156),(155,157),(156,158),(157,158));COUNTS={(153,155):816,(154,156):815,(155,157):457,(156,158):450,(157,158):254}
base_top={s:M.row_topology(faces[s],pts[s]) for s in range(153,159)};assert all(x['ok'] for x in base_top.values())
B0=M.Builder(pts,maps,faces);oldpairs={ab:B0.face_pairs(ab)[0] for ab in PAIRS};base_unions={ab:B0.union(ab) for ab in PAIRS}
for ab in PAIRS: assert len(oldpairs[ab])==COUNTS[ab] and not B0.face_pairs(ab)[1],(ab,len(oldpairs[ab]),B0.face_pairs(ab)[1][:2])
# Exact invalid step7967 witness: candidate row157 face 7997 uses vertices 4527,4528,4529; 4528-4529 collapse in native Float32.
row=157;bad_face=7997;keep=4528;drop=4529;edge=[keep,drop]
assert faces[row][bad_face]==(4527,4528,4529),faces[row][bad_face]
inc=[i for i,f in enumerate(faces[row]) if f is not None and keep in f and drop in f]
assert inc==[7997,9370],inc
link=M.edge_link_condition(faces[row],keep,drop);assert link['ok'] and link['edge_faces']==inc,link
oldp=list(pts[row]);oldm=list(maps[row]);oldf=list(faces[row]);olda=copy.deepcopy(attrs[row])
shift=math.dist(tuple(map(float,oldp[keep])),tuple(map(float,oldp[drop])))
assert shift<0.1e-6,shift
trial=[];removed=[];changed=[]
for i,f in enumerate(oldf):
 if f is None:trial.append(None);continue
 nf=tuple(keep if v==drop else v for v in f)
 if len(set(nf))<3:
  trial.append(None);removed.append({'face_index':i,**olda[i],'collapsed_by_edge_old_indices':edge});continue
 if drop in f:
  oldn=M.normal(tuple(oldp[v] for v in f));newn=M.normal(tuple(oldp[v] for v in nf));assert M.dot(oldn,newn)>0,(i,'normal inversion')
  changed.append(i)
 trial.append(nf)
remap={i:(i if i<drop else i-1) for i in range(len(oldp)) if i!=drop}
faces[row]=[None if f is None else tuple(remap[v] for v in f) for f in trial]
pts[row]=[p for i,p in enumerate(oldp) if i!=drop];maps[row]=[m for i,m in enumerate(oldm) if i!=drop]
assert [x['face_index'] for x in removed]==[7997,9370],removed
assert len(pts[row])==len(maps[row]) and all(f is None or max(f)<len(pts[row]) for f in faces[row])
assert all(sum((t[3] for t in ts),F(0))==1 and all(t[3]>0 for t in ts) for ts in maps[row])
# Same vertex maps, positions and faces outside this local fan; preserve ancestry for all survivors.
assert all(pts[row][remap[i]]==oldp[i] and maps[row][remap[i]]==oldm[i] for i in remap)
assert all((faces[row][i] is None)==(oldf[i] is None or i in [7997,9370]) for i in range(len(oldf)))
post_top=M.row_topology(faces[row],pts[row]);assert post_top['ok'] and len(post_top['components'])==len(base_top[row]['components'])
assert all(c['positive'] for c in post_top['components'])
B1=M.Builder(pts,maps,faces);interface={};
for ab in PAIRS:
 pairs,bad=B1.face_pairs(ab);assert len(pairs)==COUNTS[ab] and not bad,(ab,len(pairs),bad[:2]);interface[f'{ab[0]}+{ab[1]}']={'before':len(oldpairs[ab]),'after':len(pairs),'removed':sorted(set(oldpairs[ab])-set(pairs)),'added':sorted(set(pairs)-set(oldpairs[ab]))}
assert all(not v['removed'] and not v['added'] for v in interface.values()),interface
ok,reason=M.assert_interface_preserved(oldpairs,B1);assert ok,reason
union_audit={}
for ab in PAIRS:
 u0=base_unions[ab];u1=B1.union(ab);t0=u0['topology'];t1=u1['topology']
 union_audit[f'{ab[0]}+{ab[1]}']={'before_pairs':u0['pair_count'],'after_pairs':u1['pair_count'],'before_reason':t0.get('reason'),'after_reason':t1.get('reason'),'before_bad_vertices':t0.get('bad_vertex_count'),'after_bad_vertices':t1.get('bad_vertex_count'),'before_components':len(t0.get('components',[])),'after_components':len(t1.get('components',[])),'unpaired_before':len(u0['unpaired_signature_errors']),'unpaired_after':len(u1['unpaired_signature_errors'])}
 assert union_audit[f'{ab[0]}+{ab[1]}']['before_reason']==union_audit[f'{ab[0]}+{ab[1]}']['after_reason'] and union_audit[f'{ab[0]}+{ab[1]}']['before_bad_vertices']==union_audit[f'{ab[0]}+{ab[1]}']['after_bad_vertices'] and union_audit[f'{ab[0]}+{ab[1]}']['before_components']==union_audit[f'{ab[0]}+{ab[1]}']['after_components'],union_audit[ f'{ab[0]}+{ab[1]}']
area0=sum(tri_area(oldp,oldf[i]) for i in inc);area1=sum(tri_area(pts[row],faces[row][i]) for i in inc if faces[row][i] is not None)
volume_delta=vol(post_top)-vol(base_top[row])
full['points_exact'][str(row)]=[qjson(p) for p in pts[row]]
full['parent_maps'][str(row)]=[[tjson(t) for t in ts] for ts in maps[row]]
full['faces'][str(row)]=[None if f is None else list(f) for f in faces[row]]
full['removed_face_ancestry'].setdefault(str(row),[]).extend(removed)
full.setdefault('operations',[]).append({'operation':'endpoint_contraction_for_native_step7967_exact_f32_zero_area','row':row,'target_old_face':bad_face,'old_edge_vertices':[keep,drop],'edge_length_m':shift,'max_vertex_reassignment_m':shift,'incident_faces':inc,'removed_faces':removed,'changed_surviving_faces':changed,'face_ancestry_for_survivors_preserved':True,'parent_maps_for_survivors_unchanged':True,'interface_signatures_unchanged':True})
full['candidate_admission']=False;full['native_admission']=False;write(OUT/'full_candidate.json',full)
# Descending-only compatibility wrapper for root's composition loader.
base_overlay=json.load(open(BASE/'descending_overlay.json'));g=base_overlay['geometry'];g['points_exact'][str(row)]=[qjson(p) for p in pts[row]];g['parent_maps'][str(row)]=[[tjson(t) for t in ts] for ts in maps[row]];g['faces'][str(row)]=[None if f is None else list(f) for f in faces[row]];g.setdefault('old_to_new_vertex_index',{})[str(row)]={str(k):v for k,v in remap.items()};g.setdefault('removed_face_ancestry',{}).setdefault(str(row),[]).extend(removed);base_overlay['candidate_admission']=False;base_overlay['native_admission']=False;base_overlay['paired_repair_7967']={'row':row,'face':bad_face,'keep_old_vertex':keep,'drop_old_vertex':drop,'removed_faces':[7997,9370]};write(OUT/'descending_overlay.json',base_overlay)
shutil.copyfile(BASE/'upper_candidate.json',OUT/'upper_candidate.json')
# Materialized surface + ancestry convenience for the native composer.
live=[(i,f) for i,f in enumerate(faces[row]) if f is not None]
np.savez_compressed(OUT/'candidate-row-157.npz',faces=np.asarray([f for i,f in live],dtype=np.int64),face_parent_stable_id=np.asarray([attrs[row][i]['parent_stable_id'] for i,f in live],dtype=np.uint32),face_parent_index=np.asarray([attrs[row][i]['parent_face_index'] for i,f in live],dtype=np.uint32),face_partition_role=np.asarray([attrs[row][i]['partition_role'] for i,f in live],dtype=np.uint8),face_origin_kind=np.asarray([attrs[row][i]['origin_kind'] for i,f in live],dtype=np.uint8),face_input_index=np.asarray([attrs[row][i]['input_index'] for i,f in live],dtype=np.int64))
report={'schema':'numi.human.trapezius.attempt408-row157-native7967-edge-repair.v1','candidate_admission':False,'native_admission':False,'input':{'attempt407_full_candidate':str(BASE/'full_candidate.json'),'sha256':sha(BASE/'full_candidate.json')},'witness':{'native_run':'back-native-014-raw-parent-006','accepted_step':7967,'surface_stable_id':157,'local_face':7997,'candidate_face_vertices':[4527,4528,4529],'rendered_positions_f32_m':[[-0.02512214332818985,-0.2795204222202301,0.091245792806148529],[-0.02510879747569561,-0.2795238494873047,0.091241948306560516],[-0.02510879747569561,-0.2795238494873047,0.091241948306560516]]},'repair':{'row':row,'keep_old_vertex':keep,'drop_old_vertex':drop,'edge_source_length_m':shift,'max_vertex_reassignment_m':shift,'incident_faces':inc,'removed_face_ancestry':removed,'changed_surviving_faces':changed,'incident_patch_area_before_m2':area0,'incident_patch_area_after_m2':area1,'incident_patch_area_delta_m2':area1-area0,'exact_signed_volume_delta_m3':str(volume_delta),'all_surviving_vertex_maps_unchanged':True,'no_new_interface_vertex_weld':True},'row_topology':{'before':base_top[row],'after':post_top},'all_interface_pair_counts_and_signatures_unchanged':interface,'cancelled_union_topology_comparison':union_audit,'candidate_files':{p.name:sha(p) for p in (OUT/'full_candidate.json',OUT/'upper_candidate.json',OUT/'descending_overlay.json',OUT/'candidate-row-157.npz')},'limitations':'Only the native step7967 exact-zero-area face and its incident 13.7nm edge were repaired. The full default native cycle remains required. Existing low-altitude faces from the retained census remain unconditioned; no tolerance or general interface waiver was applied.'}
write(OUT/'report.json',report);shutil.copyfile(__file__,OUT/'build_attempt408.py');os.chmod(OUT/'build_attempt408.py',0o755)
report['builder_script_sha256']=sha(OUT/'build_attempt408.py');write(OUT/'report.json',report)
print(json.dumps({'output':str(OUT),'operations':full['operations'][-1],'interface':interface,'post_topology':{'ok':post_top['ok'],'components':len(post_top['components']),'faces':post_top['face_count']},'union':union_audit,'candidate_shas':report['candidate_files'],'builder_sha256':sha(OUT/'build_attempt408.py'),'report_sha256':sha(OUT/'report.json')},sort_keys=True),flush=True)
