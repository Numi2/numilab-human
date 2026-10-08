import json, math, hashlib, sys
from pathlib import Path
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005'); D=E/'native-lung-seam-structural-audit-927'; R=E/'native-lung-seam-cycle-925/accepted-geometry'; OUT=E/'native-lung-seam-signed-range-review-931'
report=json.load(open(D/'targeted-native-report-v2.json')); poses=report['native_pose_unique_unallowed_face_pairs']; sourcecmp=json.load(open(D/'source-native-hit-pair-comparison.json')); source_pairs={(tuple(x['owners']),tuple(x['face_ids'])):x for x in sourcecmp['face_pairs']}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(x):
 x=np.asarray(x,float); n=np.linalg.norm(x)
 if not np.isfinite(n) or n==0: raise ValueError('bad direction')
 return x/n
def rotate(q,v):
 q=np.asarray(q,float); v=np.asarray(v,float); u=q[:3]; w=q[3]
 return v+2*np.cross(u,np.cross(u,v)+w*v)
def ptri(p,t):
 a,b,c=t; ab=b-a; ac=c-a; ap=p-a; d1=ab@ap; d2=ac@ap
 if d1<=0 and d2<=0:return a
 bp=p-b; d3=ab@bp; d4=ac@bp
 if d3>=0 and d4<=d3:return b
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:return a+d1/(d1-d3)*ab
 cp=p-c; d5=ab@cp; d6=ac@cp
 if d6>=0 and d5<=d6:return c
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:return a+d2/(d2-d6)*ac
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:return b+(d4-d3)/((d4-d3)+(d5-d6))*(c-b)
 den=1/(va+vb+vc); return a+ab*(vb*den)+ac*(vc*den)
def seg(p,q,r,s):
 u=q-p; v=s-r; w=p-r; aa=u@u; bb=u@v; cc=v@v; dd=u@w; ee=v@w; den=aa*cc-bb*bb
 sc=0 if abs(den)<1e-30 else np.clip((bb*ee-cc*dd)/den,0,1); tc=np.clip((bb*sc+ee)/cc,0,1) if cc else 0
 sc=np.clip((bb*tc-dd)/aa,0,1) if aa else 0
 return p+sc*u,r+tc*v
def closest(A,B):
 out=[]
 for p in A: q=ptri(p,B); out.append((np.linalg.norm(q-p),p,q))
 for q in B: p=ptri(q,A); out.append((np.linalg.norm(p-q),p,q))
 for i in range(3):
  for j in range(3):
   p,q=seg(A[i],A[(i+1)%3],B[j],B[(j+1)%3]);out.append((np.linalg.norm(q-p),p,q))
 return min(out,key=lambda x:x[0])
def basis(n,A):
 edges=[A[1]-A[0],A[2]-A[1],A[0]-A[2]]
 e=max(edges,key=lambda x:np.linalg.norm(x-np.dot(x,n)*n))
 u=unit(e-np.dot(e,n)*n);return u,np.cross(n,u)
def cr2(a,b): return float(a[0]*b[1]-a[1]*b[0])
def clip2(poly,clip):
 def ar(p):return sum(cr2(p[i],p[(i+1)%len(p)]) for i in range(len(p))) if len(p)>2 else 0
 clip=[np.array(x,float) for x in clip];poly=[np.array(x,float) for x in poly]
 if ar(clip)<0:clip=clip[::-1]
 for i in range(len(clip)):
  a=clip[i];e=clip[(i+1)%len(clip)]-a; inp=poly;poly=[]
  if not inp:break
  def inside(p):return cr2(e,p-a)>=-1e-23
  def intersect(p,q):
   d=q-p; den=cr2(e,d)
   return (p+q)*.5 if abs(den)<1e-30 else p+cr2(e,a-p)/den*d
  s=inp[-1];si=inside(s)
  for t in inp:
   ti=inside(t)
   if ti:
    if not si:poly.append(intersect(s,t))
    poly.append(t)
   elif si:poly.append(intersect(s,t))
   s=t;si=ti
 return poly
def gap(A,B,n):
 u,v=basis(n,A); A2=[np.array([p@u,p@v]) for p in A];B2=[np.array([p@u,p@v]) for p in B]; P=clip2(A2,B2)
 if not P:return None
 def zs(T2,T3):
  p0,p1,p2=T2;den=cr2(p1-p0,p2-p0)
  if abs(den)<1e-25:return None
  zz=T3@n; out=[]
  for p in P:
   l1=cr2(p-p0,p2-p0)/den; l2=cr2(p1-p0,p-p0)/den; out.append(zz[0]*(1-l1-l2)+zz[1]*l1+zz[2]*l2)
  return out
 zA=zs(A2,A);zB=zs(B2,B)
 if zA is None or zB is None:return None
 g=np.asarray(zB)-zA
 return {'min_m':float(g.min()),'max_m':float(g.max()),'projected_span_m':max((float(np.linalg.norm(x-y)) for i,x in enumerate(P) for y in P[i+1:]),default=0),'vertices':len(P)}
allrows=[]; pose_summary=[]; mapstats={}
for pp in poses:
 step=pp['step'];rec=json.load(open(R/f'step-{step}.receipt.json'));bp=next(x for x in rec['accepted_registered_body_poses'] if x['body_index']==20);q=np.array(bp['quaternion_xyzw'],float);t=np.array(bp['position_m'],float); rows=[]
 for ev in pp['face_pairs']:
  A=np.array(ev['source_vertex_xyz_m'][0],float);B=np.array(ev['source_vertex_xyz_m'][1],float);AW=np.array(ev['captured_world_vertex_xyz_m'][0],float);BW=np.array(ev['captured_world_vertex_xyz_m'][1],float)
  if ev['shared_source_coordinate_count']:
   ns=unit(np.cross(A[1]-A[0],A[2]-A[0]));method='owner-A source winding'
  else:
   d,pa,pb=closest(A,B);ns=unit(pb-pa);method='source closest vector A->B'
  nw=unit(rotate(q,ns)); sr=gap(A,B,ns); nr=gap(AW,BW,nw); sm=source_pairs.get((tuple(ev['owners']),tuple(ev['face_ids'])))
  if sm is None: raise ValueError('source comparison missing canonical face pair')
  if sr is None:
   ds,pa,pb=closest(A,B); sg=float(np.dot(pb-pa,ns)); sr={'min_m':sg,'max_m':sg,'projected_span_m':0.0,'vertices':1,'fallback':'source closest-pair signed gap (projected footprints touch at zero-area boundary)'}
  PA=np.array([rotate(q,x)+t for x in A]);PB=np.array([rotate(q,x)+t for x in B]);rv=np.r_[AW-PA,BW-PB];res=np.linalg.norm(rv,axis=1); normalres=np.abs(rv@nw)
  def hu(T):
   z=np.asarray(T,dtype=np.float32);sp=np.abs(np.spacing(z).astype(float));return max(.5*np.dot(np.abs(nw),x) for x in sp)
  ub=hu(AW)+hu(BW)
  row={'step':step,'owners':ev['owners'],'face_ids':ev['face_ids'],'source_face_vertex_ids':ev['face_vertex_ids'],'shared_source_coordinate_count':ev['shared_source_coordinate_count'],'source_orientation_method':method,'candidate_source_intersection':sm['candidate_source_intersection'],'candidate_source_intersection_allowed_vertex_edge':sm['candidate_source_intersection_allowed_vertex_edge'],'candidate_source_shared_vertex_count':sm['candidate_source_shared_vertex_count'],'source_signed_gap_m':sr,'native_signed_gap_m':nr,'source_closest_distance_m':float(closest(A,B)[0]),'native_intersection_span_m':float(ev['diameter_m']),'half_ulp_pair_normal_bound_m':float(ub),'max_rigid_pose_vertex_residual_m':float(res.max()),'rms_rigid_pose_vertex_residual_m':float(np.sqrt(np.mean(res**2))),'max_normal_component_rigid_pose_residual_m':float(normalres.max()),'point_count':ev['point_count'],'captured_common_vertex_edge_count':ev['captured_common_vertex_edge_count']}
  if nr and sr:
   row['native_minus_source_range_delta_m']=[nr['min_m']-sr['min_m'],nr['max_m']-sr['max_m']]
   row['inward_excursion_beyond_source_min_m']=max(0.,sr['min_m']-nr['min_m']);row['inward_excursion_over_half_ulp_bound']=row['inward_excursion_beyond_source_min_m']/ub if ub else None;row['positive_gap_extension_beyond_source_max_m']=max(0.,nr['max_m']-sr['max_m']);row['positive_gap_extension_over_half_ulp_bound']=row['positive_gap_extension_beyond_source_max_m']/ub if ub else None;row['max_endpoint_range_change_m']=max(row['inward_excursion_beyond_source_min_m'],row['positive_gap_extension_beyond_source_max_m'])
  rows.append(row);allrows.append(row)
 mapstats[str(step)]={'body_index':20,'position_m':t.tolist(),'quaternion_xyzw':q.tolist(),'max_event_vertex_residual_m':max(x['max_rigid_pose_vertex_residual_m'] for x in rows),'rms_event_vertex_residual_m':float(np.sqrt(np.mean([x['rms_rigid_pose_vertex_residual_m']**2 for x in rows]))),'max_event_normal_residual_m':max(x['max_normal_component_rigid_pose_residual_m'] for x in rows)}
 pose_summary.append({'step':step,'count':len(rows),'owner_counts':{f'{op[0]}-{op[1]}':sum(tuple(r['owners'])==op for r in rows) for op in [(305,308),(306,307),(307,309)]}})
agg={}
for op in [(305,308),(306,307),(307,309)]:
 key=f'{op[0]}-{op[1]}';agg[key]=[]
 for st in [p['step'] for p in poses]:
  rs=[r for r in allrows if r['step']==st and tuple(r['owners'])==op]
  if not rs:continue
  agg[key].append({'step':st,'events':len(rs),'max_tangential_span_um':max(r['native_intersection_span_m'] for r in rs)*1e6,'source_signed_range_nm':[min(r['source_signed_gap_m']['min_m'] for r in rs)*1e9,max(r['source_signed_gap_m']['max_m'] for r in rs)*1e9],'native_signed_range_nm':[min(r['native_signed_gap_m']['min_m'] for r in rs)*1e9,max(r['native_signed_gap_m']['max_m'] for r in rs)*1e9],'max_inward_excursion_beyond_source_min_nm':max(r['inward_excursion_beyond_source_min_m'] for r in rs)*1e9,'max_positive_gap_extension_beyond_source_max_nm':max(r['positive_gap_extension_beyond_source_max_m'] for r in rs)*1e9,'max_half_ulp_bound_nm':max(r['half_ulp_pair_normal_bound_m'] for r in rs)*1e9,'max_inward_excursion_over_half_ulp':max(r['inward_excursion_over_half_ulp_bound'] for r in rs),'max_positive_gap_extension_over_half_ulp':max(r['positive_gap_extension_over_half_ulp_bound'] for r in rs),'max_rigid_pose_vertex_residual_nm':max(r['max_rigid_pose_vertex_residual_m'] for r in rs)*1e9,'max_normal_component_rigid_pose_residual_nm':max(r['max_normal_component_rigid_pose_residual_m'] for r in rs)*1e9})
reportout={'schema':'numi.human.native925.lung-interface-signed-range-review.v1','inputs':{'source_nha':{'path':str(E/'lung-choroid-composition-924/choroid-v2/resting-thorax.nhanatomy'),'sha256':'3c444be7736c066a992988cc32b687917e1c4c5c3968a16b4d5f0106d5b5024e'},'targeted_native_report':{'path':str(D/'targeted-native-report-v2.json'),'sha256':sha(D/'targeted-native-report-v2.json')},'source_native_comparison':{'path':str(D/'source-native-hit-pair-comparison.json'),'sha256':sha(D/'source-native-hit-pair-comparison.json')},'source_reciprocal_patches':{'path':str(D/'source-reciprocal-patches.json'),'sha256':sha(D/'source-reciprocal-patches.json')},'native_captures':str(R)+'/step-{0,4991,5375,5759,6111,6495,7743,9999}.mrvpack and matching receipts'},'method':{'event_unit':'canonical pose + owner pair + face-ID pair from 927 face_pairs; no intersections removed','orientation_before_native':'exact source-shared pairs use owner-A source face winding; disjoint pairs use source closest-point vector A to B','native_direction':'rotate source-derived unit normal using accepted body index 20 quaternion from corresponding 925 receipt','signed_gap':'owner B minus owner A axial range over triangle-footprint overlap projected on source-derived normal; report inward excursion below the source minimum separately from positive-side gap extension above the source maximum','ulp_bound':'sum for both captured triangles of max_vertex(0.5*sum_axis(abs(n_axis)*abs(float32_spacing(coord_axis)))); coordinate quantization only','interpretation':'All exact native hits remain reported. Signed range and bound diagnose normal extent and do not relax exact intersection checks.'},'canonical_event_totals':{'per_pose':pose_summary,'by_owner_pair':{k:sum(x['events'] for x in v) for k,v in agg.items()},'total':len(allrows),'unique_face_pairs':len(source_pairs),'unique_source_pair_class_counts':{'source_exact_vertex_or_edge_intersection':sum(bool(x['candidate_source_intersection']) for x in source_pairs.values()),'source_disjoint':sum(not x['candidate_source_intersection'] for x in source_pairs.values()),'all_exact_source_intersections_allowed_vertex_edge':all((not x['candidate_source_intersection']) or x['candidate_source_intersection_allowed_vertex_edge'] for x in source_pairs.values())},'reconciliation':'Canonical 927 total is 93=81+6+6. The prior 79+6+6=91 result is superseded and does not match canonical 927 counts. No canonical event is dropped in this report. A scan of exact point-set signatures finds one duplicate pair of face-pair records at step0: (305,308) (24552,26583) and (24554,26582). Merging only that exact point set would yield 92, not 91; the former 79 result has no retained event-level list to identify its additional omitted record. Keep all 93 canonical pose/face-pair rows.'},'body20_rigid_map_witness_residuals':mapstats,'aggregate_by_owner_pair_pose':agg,'all_events':allrows,'limitations':{'primary':'922/924 source plus 925 captures; targeted to pairs (305,308),(306,307),(307,309)','failed_alternatives':'928 v4 and 930 v5 remain failed/source-map alternatives, not native replacements. v4 separately contains source-disjoint pair 305/308 face 40219 x 46134 with nearest source vertices about 164.234 um; outside the primary 927 list.','not_claimed':'No zero-intersection or globally clear anatomical interface claim. Half-ULP bound is only a coordinate-quantization diagnostic, not a bound for respiratory deformation, shader arithmetic, or registration uncertainty. Whole-vector source-to-capture residuals grow to mm at inspiration and are retained separately; these demonstrate non-rigid respiratory deformation, not rounding.'}}
srcnha=E/'lung-choroid-composition-924/choroid-v2/resting-thorax.nhanatomy'
reportout['inputs']['source_nha']['sha256']=sha(srcnha)
reportout['inputs']['native_capture_hashes']={}
for step in [x['step'] for x in poses]:
 pack=R/f'step-{step}.mrvpack'; receipt=R/f'step-{step}.receipt.json'
 reportout['inputs']['native_capture_hashes'][str(step)]={'pack_sha256':sha(pack),'receipt_sha256':sha(receipt)}
reportout['code']={'path':str(OUT/'analyze.py'),'sha256':sha(OUT/'analyze.py'),'command':'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-resting-evidence-20261005/native-lung-seam-signed-range-review-931/analyze.py','python':sys.version.split()[0],'numpy':np.__version__}
(OUT/'report.json').write_text(json.dumps(reportout,indent=2,sort_keys=True)+'\n')
print('events',len(allrows),'pair counts',{k:sum(x['events'] for x in v) for k,v in agg.items()})
print('report sha',sha(OUT/'report.json'))
print('pose counts',json.dumps(pose_summary))
print('map residual nm',json.dumps({k:{'max':v['max_event_vertex_residual_m']*1e9,'rms':v['rms_event_vertex_residual_m']*1e9} for k,v in mapstats.items()}))
for k,v in agg.items():print(k,json.dumps(v))
