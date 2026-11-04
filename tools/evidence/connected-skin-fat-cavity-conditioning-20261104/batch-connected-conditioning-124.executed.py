exec(open('/tmp/root20-connected-cavity-118.py').read().split('rr={seed}')[0])
from itertools import combinations
import hashlib,sys
spec=importlib.util.spec_from_file_location('solver',V/'complete-surface-cavity-220-root/solve_oriented_cavity_milp_220.py');solver=importlib.util.module_from_spec(spec);spec.loader.exec_module(solver)
base_mask=solver.native_quality_mask
goal=1e-3
def strict_mask(pp,tt):
 ok,d,q=base_mask(pp,tt)
 return ok&(q>=goal),d,q
solver.native_quality_mask=strict_mask
c=np.load(D/'conditioning-native-mass-screened-003/fixed-boundary-volume-proposal.npz')
po=c['replacement_tet_offsets'];pa=c['replacement_parent_offsets'];pr=c['replacement_parent_cell_rows'];repl=c['replacement_tetrahedra']
base=[]
for k in range(len(po)-1):
 a,b=map(int,po[k:k+2]);rows=pr[pa[a]:pa[a+1]]
 base.append({'old_rows':rows.tolist(),'new_cells':repl[a:b].tolist(),'legacy_index':k})
row_owner=np.full(len(t),-1,np.int32)
for k,o in enumerate(base):row_owner[o['old_rows']]=k
kept=c['retained_input_cell_rows'];d,q=m.measurements(p,t)
seeds=np.r_[seed,kept[np.argsort(q[kept],kind='stable')]]
seeds=seeds[np.r_[True,np.logical_and(seeds[1:]!=seed,q[seeds[1:]]<1e-4)]]
output=D/'connected-conditioning-124';output.mkdir(exist_ok=False)
ops=[];used=set();reject=collections.Counter();trials=[];tic=time.monotonic()
def closure(rows):
 extra=set()
 for r in rows:
  k=row_owner[r]
  if k>=0:extra.update(base[int(k)]['old_rows'])
 return rows|extra
for ss in seeds:
 if time.monotonic()-tic>120 or len(ops)>=500:break
 ss=int(ss)
 if ss in used:continue
 rr={ss};taken=None
 for depth in range(1,4):
  for r in list(rr):
   for f in t[r][m.PAT]:
    rr.update(int(x) for x in facecells(f) if np.array_equal(labs[x],labs[ss]))
  rr=closure(rr)
  if rr&used:reject['disjoint_source_cells']+=1;break
  if len(rr)>32:reject['cell_budget']+=1;break
  rows=np.array(sorted(rr));old=t[rows];verts=np.unique(old)
  if len(verts)>16:reject['vertex_budget']+=1;break
  if not (labs[rows]==labs[ss]).all():reject['material_donor_region']+=1;break
  b=boundary(old);ix={int(v):k for k,v in enumerate(verts)};lp=p[verts];lb=np.array([[ix[int(v)] for v in f] for f in b],np.int64)
  ce=m.orient(lp,np.array(list(combinations(range(len(lp)),4)),np.int64));ok,_,_=strict_mask(lp,ce)
  cf={tuple(sorted(f)) for f in ce[ok][:,m.PAT].reshape(-1,3)}
  if any(tuple(sorted(f)) not in cf for f in lb):reject['no_quality_boundary_cover']+=1;continue
  try:report,nt=solver.solve(lp,lb,seconds=.15,max_points=16,material=int(labs[ss,0]))
  except ValueError as e:
   if str(e)!='closed manifold boundary required':raise
   reject['nonmanifold_cavity_boundary']+=1;continue
  if nt is None:reject[report['status']]+=1;continue
  nn=verts[nt]
  valid,why=m.certify(p,old,nn)
  if not valid:raise ValueError(why)
  if not np.array_equal(np.unique(nn),verts):reject['unused_vertex']+=1;continue
  taken={'seed':ss,'depth':depth,'old_rows':rows.tolist(),'new_cells':nn.tolist(),'old_min_q':float(q[rows].min()),'new_min_q':float(m.measurements(p,nn)[1].min()),'labels':labs[ss].tolist(),'construction_min_q':goal}
  used.update(rr);ops.append(taken);break
 if len(trials)<300:trials.append({'seed':ss,'accepted':taken is not None})
 if len(trials)%25==0:print(json.dumps({'elapsed_s':time.monotonic()-tic,'attempted':len(trials),'accepted':len(ops),'rejections':dict(reject)}),flush=True)
assert any(o['seed']==seed for o in ops),'measured failing cell not repaired'
dropped=[o for o in base if set(o['old_rows'])&used]
assert all(set(o['old_rows'])<=used for o in dropped)
allops=[o for o in base if not set(o['old_rows'])&used]+ops
retired=np.array([r for o in allops for r in o['old_rows']],np.int64);assert len(np.unique(retired))==len(retired)
retained=np.flatnonzero(~np.isin(np.arange(len(t)),retired))
new=np.array([cell for o in allops for cell in o['new_cells']],np.int64)
tetoff=np.r_[0,np.cumsum([len(o['new_cells']) for o in allops])]
parentoff=np.r_[0,np.cumsum([len(o['old_rows']) for o in allops for _ in o['new_cells']])]
parents=np.array([r for o in allops for _ in o['new_cells'] for r in o['old_rows']],np.int64)
source=V/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
sha=lambda pp:hashlib.file_digest(Path(pp).open('rb'),'sha256').hexdigest()
np.savez_compressed(output/'fixed-boundary-volume-proposal.npz',
 retained_input_cell_rows=retained,replacement_tetrahedra=new,replacement_tet_offsets=tetoff,
 replacement_parent_offsets=parentoff,replacement_parent_cell_rows=parents,source_volume_sha256=np.array(sha(source)))
before=np.r_[q[kept],m.measurements(p,repl)[1]];nd,nq=m.measurements(p,new);after=np.r_[q[retained],nq]
assert m.passes(nd,nq).all()
report={'status':'EXACT_CONNECTED_CONDITIONING_PROPOSAL_NOT_NATIVE_QUALIFIED','source_sha256':sha(source),'proposal_sha256':sha(output/'fixed-boundary-volume-proposal.npz'),'script_sha256':sha(__file__),
 'points_bit_identical':True,'exact_oriented_boundary_and_volume':True,'material_donor_region_preserved':True,'construction_min_q':goal,
 'base_cavities':len(base),'replaced_base_cavities':len(dropped),'connected_cavities':len(ops),'total_cavities':len(allops),
 'cells_before':len(before),'cells_after':len(after),'q_lt_1e_4_before':int((before<1e-4).sum()),'q_lt_1e_4_after':int((after<1e-4).sum()),
 'q_lt_1e_3_before':int((before<1e-3).sum()),'q_lt_1e_3_after':int((after<1e-3).sum()),
 'elapsed_s':time.monotonic()-tic,'rejections':dict(reject),'connected_patches':ops,'trials':trials,
 'remaining_gates':['independent native exact geometry','conservative nodal/source mass and frozen transport radius','full accepted physical transaction']}
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('connected_patches','trials')}),flush=True)

