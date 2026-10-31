from pathlib import Path
import hashlib,json,time,collections
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'paired-D307-I364-volume-composition-007/independent-replay-001';OUT.mkdir(exist_ok=False)
BASE=H/'fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz'
D=H/'d-shortedge-quotient-351-run002/volume-run003/fat-D-shortedge-quotient-volume-proposal.npz'
I352=H/'paired-inner-interface-quotient-352-root/run001/fat-coupled-I-quotient-volume-proposal.npz'
I364=H/'paired-I-interface-364-root/run001/fat-coupled-I-quotient-volume-proposal.npz'
DREPORT=H/'d-shortedge-quotient-351-run002/volume-run003/report.json'
I352R=H/'paired-inner-interface-quotient-352-root/run001/report.json'
I364R=H/'paired-I-interface-364-root/run001/report.json'
C=H/'paired-D307-I364-volume-composition-007/fat-i364-d307-composed.npz'
R=H/'paired-D307-I364-volume-composition-007/composition-report.json'
SIDE=H/'d-shortedge-quotient-351-run002/volume-run003/affected-fat-stars-001.npz'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(v,copy=True) for k,v in z.items()}
def report(role,rp):
 j=json.loads(rp.read_text());return next(x for x in j['roles'] if x['role']==role)['operations']
def union_find(n,ops):
 p=np.arange(n,dtype=np.int64)
 def f(x):
  x=int(x)
  while p[x]!=x:p[x]=p[p[x]];x=int(p[x])
  return x
 for op in ops:
  a=f(op['retired_volume_node']);b=f(op['kept_volume_node'])
  if a!=b:p[a]=b
 return np.fromiter((f(i) for i in range(n)),np.int64,count=n)
def degenerate(t):
 t=np.asarray(t,np.int64)
 return np.any((t[:,:,None]==t[:,None,:]) & (~np.eye(t.shape[1],dtype=bool)[None,:,:]),axis=(1,2))
def boundary(t):
 x=np.asarray(t,np.int64)
 f=np.concatenate([x[:,[1,2,3]],x[:,[0,3,2]],x[:,[0,1,3]],x[:,[0,2,1]]],axis=0)
 k=np.sort(f,axis=1);_,ix,c=np.unique(k,axis=0,return_index=True,return_counts=True)
 if np.any(c>2):raise ValueError('tet face incidence above two')
 return f[ix[c==1]]
def ckey(f):
 a,b,c=map(int,f);return min((a,b,c),(b,c,a),(c,a,b))
def oriented_map(faces,label):
 d={}
 for f in np.asarray(faces,np.int64):
  u=tuple(sorted(map(int,f)));q=ckey(f)
  if u in d:raise ValueError(f'{label} duplicate unoriented face {u}')
  d[u]=q
 return d
def map_faces(faces,node_map):
 q=node_map[np.asarray(faces,np.int64)]
 keep=~degenerate(q)
 return q[keep]
t0=time.time()
base=load(BASE);d=load(D);i352=load(I352);i364=load(I364);c=load(C)
N=len(base['points_world_m']); NC=len(base['tetrahedra']); basep=np.asarray(base['points_world_m'],np.float32); baset=np.asarray(base['tetrahedra'],np.int64);basem=np.asarray(base['material_indices'])
opsI=report('Fat',I352R)+report('Fat',I364R);im=union_find(N,opsI)
dmap=np.asarray(d['input_node_to_output_node'],np.int64); d_out_to_base=np.asarray(d['output_node_to_input_node'],np.int64)
if dmap.shape!=(N,):raise ValueError('D point map not in full base domain')
# Independently derive each surviving source tet row and composed compact output connectivity.
fullmap=dmap[im]
source_tets=baset
mapped_dout=fullmap[source_tets]
keep=~degenerate(mapped_dout)
if np.any(mapped_dout[keep]<0):raise ValueError('a surviving composed source tet maps to a missing D output node')
expected_src=np.flatnonzero(keep).astype(np.int64)
# Candidate cell source map is declared in d4906 input rows; compare exact sets and connectivity.
actual_src=np.asarray(c['output_cell_to_input_cell'],np.int64)
if not np.array_equal(actual_src,expected_src):
 if not np.array_equal(np.sort(actual_src),np.sort(expected_src)):raise ValueError('composed source-cell row set differs from independent node quotient')
 # align by source row, not output ordering
actual_t=np.asarray(c['tetrahedra'],np.int64); out_to_base=np.asarray(c['output_node_to_input_node'],np.int64)
if len(np.unique(out_to_base))!=len(out_to_base) or np.any(out_to_base<0):raise ValueError('candidate output node representatives are not unique')
source_to_compact=np.full(N,-1,np.int64)
for outrow,srcrow in enumerate(out_to_base):source_to_compact[int(srcrow)]=outrow
expected_compact=source_to_compact[d_out_to_base[mapped_dout[expected_src]]]
if np.any(expected_compact<0):raise ValueError('direct composed tetra references no candidate output node')
# Source rows are sorted in final candidate; independently reorder by source ID.
order=np.argsort(actual_src);actual_sorted=actual_src[order]
if not np.array_equal(actual_sorted,expected_src):raise ValueError('candidate source-cell rows are not exact composed source rows')
if not np.array_equal(actual_t[order],expected_compact):raise ValueError('candidate tet connectivity differs from direct quotient replay')
if not np.array_equal(np.asarray(c['material_indices'])[order],basem[expected_src]):raise ValueError('candidate material indices differ from exact source-cell rows')
# Exact f32 point bytes against d4906 representative source rows.
cp=np.asarray(c['points_world_m'],np.float32)
if not np.array_equal(cp.view(np.uint32),basep[out_to_base].view(np.uint32)):raise ValueError('candidate point bytes differ from d4906 output representatives')
# Boundary recomputation and independently transformed intermediate oriented boundaries.
act=boundary(actual_t)
actmap=oriented_map(act,'candidate actual boundary')
if oriented_map(np.asarray(c['boundary_faces'],np.int64),'candidate declared boundary')!=actmap:raise ValueError('candidate declared boundary differs from direct tet boundary')
# D output boundary -> d4906 representative -> I quotient -> composed compact.
d_boundary=np.asarray(d['boundary_faces'],np.int64)
d_base_faces=d_out_to_base[d_boundary]
if np.any(d_base_faces<0):raise ValueError('D boundary output has no source representative')
d_comp_map=source_to_compact[im[d_base_faces]]
d_exp=map_faces(d_boundary,np.asarray(c['output_node_to_input_node'],np.int64)) # to test same output rows via exact source reps
# Independently map I364 boundary through D map, then compact via candidate source reps.
i_boundary=np.asarray(i364['boundary_faces'],np.int64)
i_drows=dmap[i_boundary]
i_keep=np.all(i_drows>=0,axis=1) & ~degenerate(i_drows)
i_comp=source_to_compact[d_out_to_base[i_drows[i_keep]]]
if np.any(i_comp<0):raise ValueError('transformed I boundary references absent combo representative')
# D transformed boundary could include faces absent after I quotient; remove degenerate and canonicalize.
d_comp=source_to_compact[im[d_base_faces]]
d_keep=np.all(d_comp>=0,axis=1) & ~degenerate(d_comp)
d_comp=d_comp[d_keep]
if np.any(d_comp<0):raise ValueError('transformed D boundary references absent combo representative')
if oriented_map(d_comp,'D boundary transformed through I')!=actmap:raise ValueError('D boundary oriented chain does not match composed tetra boundary')
if oriented_map(i_comp,'I boundary transformed through D')!=actmap:raise ValueError('I boundary oriented chain does not match composed tetra boundary')
# Base source boundary transformed through both maps.
# Independently verify full-star disjointness of the authorized operation sets.
i_endpoint=np.unique(np.asarray([int(o[k]) for o in opsI for k in ('kept_volume_node','retired_volume_node')],np.int64))
i_stars=np.flatnonzero(np.any(np.isin(baset,i_endpoint),axis=1))
side=load(SIDE); d_affected=np.asarray(side['affected_volume_cell_rows'],np.int64)
overlap=np.intersect1d(i_stars,d_affected)
if len(overlap):raise ValueError('D307/I352-I364 full old-cell stars overlap')
# Verify declared source candidate report and result content pins.
cr=json.loads(R.read_text())
if cr['status']!='PASS_EXACT_D307_ON_I364_OLD_TETRAHEDRON_COMPOSITION' or cr['composition']['output_sha256']!=sha(C):raise ValueError('composition producer receipt does not bind candidate')
report={'schema':'h706.d307-i364-composition-independent-replay.v1','status':'PASS_INDEPENDENT_EXACT_CELL_POINT_MATERIAL_AND_ORIENTED_BOUNDARY_REPLAY','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','elapsed_s':time.time()-t0,
 'input_pins':{str(p):sha(p) for p in [BASE,D,I352,I364,DREPORT,I352R,I364R,C,R,SIDE]},
 'independent_replay':{'base_d4906_points':N,'base_d4906_cells':NC,'expected_retained_source_cells':len(expected_src),'candidate_source_cells':len(actual_src),'tet_connectivity_mismatches':0,'material_mismatches':0,'point_f32_byte_mismatches':0,'candidate_output_points':len(cp),'candidate_output_cells':len(actual_t),'candidate_oriented_boundary_faces':len(act),'candidate_declared_boundary_exact':True,
  'D_transformed_oriented_boundary_matches_composition':True,'I_transformed_oriented_boundary_matches_composition':True,'boundary_orientation_comparison':'canonical oriented triangle key modulo cyclic rotation; reversed winding distinct','boundary_missing_or_extra_or_reversed':0},
 'operation_disjointness':{'D307_affected_base_cell_rows':len(d_affected),'I352_I364_endpoint_rows':len(i_endpoint),'I352_I364_full_star_cells':len(i_stars),'full_star_cell_overlap':len(overlap),'status':'PASS_DISJOINT'},
 'limitations':['The current full target boundary remains incomplete; this proves a composed partial-volume branch only.','Native quality and mass mapping are not claimed by this replay.']}
p=OUT/'report.json';p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(p),'sha256':sha(p),'status':report['status'],'source_cells':len(expected_src),'boundary_faces':len(act),'D_I_star_overlap':len(overlap)},indent=2))
