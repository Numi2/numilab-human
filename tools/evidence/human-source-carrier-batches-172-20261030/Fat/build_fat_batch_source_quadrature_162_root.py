"""Offline, positive degree1 source-cell quadrature for centroid-proxy outliers.
No source or child geometry changes; no additional physical state."""
import csv,hashlib,json,sys,struct,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
D=Path(__file__).resolve().parent
P=D.parent.parent;H=P/'fine-physical-volume-706'
sys.path.insert(0,str(H));from batched_reference_tet_projection_149_root import project_pairs
S=P/'native-physical-parent-mass-330/native-source-identity-control-003'
T=S/'compiled-parent-cell-mass.csv'
out=D/'source-quadrature-162-root';out.mkdir(exist_ok=False)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
start=time.monotonic()
manifest=json.loads((D/'preparation-manifest.json').read_text())
assert all(sha(Path(p))==v['sha256'] for p,v in manifest['inputs'].items())
nodes=read(T.with_name(T.name+'.nodes.csv'));tetrows=read(T.with_name(T.name+'.tets.csv'));labs=read(T.with_name(T.name+'.labels.csv'))
px=np.asarray([[float(r['rest_'+a+'_m']) for a in 'xyz'] for r in nodes])
pt=np.asarray([[int(r['parent_node'+str(k)]) for k in range(4)] for r in tetrows])
global_rows=np.asarray([i for i,r in enumerate(tetrows) if int(r['material_index'])==16],dtype=np.uint32)
inverses=np.linalg.inv(np.transpose(px[pt[:,1:]]-px[pt[:,0,None]],(0,2,1)))
raw=np.loadtxt(H/'combined-child-handoff-142/fat.node',skiprows=1,usecols=(0,1,2,3))
ids=raw[:,0].astype(np.int64);cx=raw[:,1:].astype(np.float32).astype(np.float64)
ele=np.loadtxt(H/'combined-child-handoff-142/fat.ele',skiprows=1)
ct=np.searchsorted(ids,ele[:,1:5].astype(np.int64));assert np.array_equal(ids[ct],ele[:,1:5].astype(np.int64))
cc=cx[ct].mean(axis=1)
with np.load(H/'connected-region-mass-support-143-root/fat-connected-region-support.npz',allow_pickle=False) as z:
 ao=z['cell_admissible_region_offsets'];ai=z['cell_admissible_region_ids']
census=json.loads((D/'all-parent-reachability-census.json').read_text())
assert census['parents_without_any_admissible_child_within_bound']==4
R=.019353


from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
with np.load(D/'child-reachability-157-root/uncovered-child-p1-source-pools.npz',allow_pickle=False) as z:h={k:z[k] for k in z.files}
holes=h['child_rows'];npf=len(global_rows);global_to_local={int(g):i for i,g in enumerate(global_rows)}
source_mass=np.asarray([float(tetrows[g]['parent_mass_kg']) for g in global_rows])
preg=np.asarray([int(labs[g]['anatomical_region_id']) for g in global_rows])
pc=px[pt[global_rows]].mean(axis=1)
# Actual emitted-F32 volumes; these are used only to plan minimum receiver mass.
dv=cx[ct[holes]][:,1:]-cx[ct[holes]][:,:1]
vol=np.abs(np.einsum('ij,ij->i',dv[:,0],np.cross(dv[:,1],dv[:,2]))/6)
demand=850*vol
region_parents={int(reg):np.flatnonzero(preg==reg) for reg in np.unique(preg)}
region_trees={reg:cKDTree(pc[pp]) for reg,pp in region_parents.items()}
candidate_child=[];candidate_parent=[]
for ih,c in enumerate(holes):
 pool=set(map(int,h['parent_pool_global_rows'][h['parent_pool_offsets'][ih]:h['parent_pool_offsets'][ih+1]]))
 for reg in ai[ao[c]:ao[c+1]]:
  pp=region_parents[int(reg)];_,jj=region_trees[int(reg)].query(cc[c],k=min(8,len(pp)))
  pool.update(map(int,global_rows[pp[np.atleast_1d(jj)]]))
 for g in sorted(pool):
  if int(labs[g]['anatomical_region_id']) in ai[ao[c]:ao[c+1]]:
   candidate_child.append(ih);candidate_parent.append(g)
ci=np.asarray(candidate_child,dtype=np.int64);gp=np.asarray(candidate_parent,dtype=np.int64)
projection=project_pairs(cc[holes[ci]],gp,px,pt,inverses)
bs=projection['bary']*(1-1e-6)+.25e-6;points=np.einsum('ij,ijk->ik',bs,px[pt[gp]])
dist=np.linalg.norm(points-cc[holes[ci]],axis=1);valid=dist<R
ci=ci[valid];gp=gp[valid];bs=bs[valid];dist=dist[valid]
lp_parent=np.asarray([global_to_local[int(g)] for g in gp],dtype=np.int64)
active,actidx=np.unique(lp_parent,return_inverse=True)
columns=np.arange(len(ci))
# Variables are fractions of a receiver's minimum mass. Child equalities sum1.
eq=coo_matrix((np.ones(len(ci)),(ci,columns)),shape=(len(holes),len(ci))).tocsr()
ineq_rows=(actidx[:,None]*4+np.arange(4)[None,:]).ravel()
ineq_cols=np.repeat(columns,4)
ineq_values=(bs*demand[ci,None]/source_mass[lp_parent,None]).ravel()
ineq=coo_matrix((ineq_values,(ineq_rows,ineq_cols)),shape=(len(active)*4,len(ci))).tocsr()
lp=linprog((dist/R)**2+1e-6*np.linalg.norm(pc[lp_parent]-cc[holes[ci]],axis=1)/R,
 A_ub=ineq,b_ub=np.full(len(active)*4,.20),A_eq=eq,b_eq=np.ones(len(holes)),
 bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
if not lp.success:
 (out/'failure.json').write_text(json.dumps({'status':'INCONCLUSIVE_REFERENCE_CARRIER_MASS_PLANNING_LP','message':lp.message,'candidate_pairs':len(ci),'child_holes':len(holes)},indent=2)+'\n')
 raise ValueError(lp.message)
if np.max(np.abs(eq@lp.x-1))>1e-8 or np.max(ineq@lp.x-.20)>1e-8:raise ValueError('LP primal certificate')
required={}
for j in np.flatnonzero(lp.x>1e-12):
 p=int(lp_parent[j]);c=int(holes[ci[j]]);w=float(lp.x[j]*demand[ci[j]]/source_mass[p])
 required.setdefault(p,[]).append((w,bs[j],c,float(dist[j])))
# The four original source-side holes also need a source carrier construction.
for witness in census['witnesses']:required.setdefault(int(witness['fat_parent_row']),[])
child_for_slot=np.repeat(np.arange(len(cc),dtype=np.int32),np.diff(ao))
order=np.argsort(ai,kind='stable');ordered=ai[order];sorted_child=child_for_slot[order]
ur,st,nn=np.unique(ordered,return_index=True,return_counts=True)
allowed_by_region={int(reg):np.unique(sorted_child[start:start+count]) for reg,start,count in zip(ur,st,nn)}
carrier_rows={};parent_reports=[];failures=[]
with np.load(D/'source-quadrature-154-root/source-quadrature.npz',allow_pickle=False) as z:prior={k:z[k] for k in z.files}

goal_bary=list(np.eye(4))
for j in range(4):
 for k in range(j+1,4):
  b=np.zeros(4);b[j]=b[k]=.5;goal_bary.append(b)
goal_bary.append(np.full(4,.25))
for number,(p,mandatory) in enumerate(sorted(required.items())):
 g=int(global_rows[p]);reg=int(preg[p]);v=px[pt[g]]
 mandatory_weight=np.asarray([x[0] for x in mandatory])
 mandatory_bary=np.asarray([x[1] for x in mandatory]).reshape(-1,4)
 rhs=np.full(4,.25)-(mandatory_weight@mandatory_bary if len(mandatory) else 0)
 if np.min(rhs)<-1e-10:raise ValueError('source nodal mass budget')
 allowed=allowed_by_region[reg]
 selected=allowed[np.all((cc[allowed]>=v.min(axis=0)-R)&(cc[allowed]<=v.max(axis=0)+R),axis=1)]
 projection=project_pairs(cc[selected],np.full(len(selected),g,dtype=np.int64),px,pt,inverses)
 take=projection['distance']<R;base_bary=projection['bary'][take];targets=selected[take]
 base=base_bary@v;allb=[];allt=[];alld=[]
 for goal in goal_bary:
  direction=goal@v-base;relative=base-cc[targets]
  aa=np.einsum('ij,ij->i',direction,direction);bb=2*np.einsum('ij,ij->i',direction,relative)
  dd=np.einsum('ij,ij->i',relative,relative)-R*R
  tmax=np.divide(-bb+np.sqrt(np.maximum(0,bb*bb-4*aa*dd)),2*aa,out=np.ones_like(aa),where=aa>0)
  tt=.95*np.minimum(1,np.maximum(0,tmax))
  bary=(base_bary*(1-tt[:,None])+goal[None,:]*tt[:,None])*(1-1e-6)+.25e-6
  ds=np.linalg.norm(bary@v-cc[targets],axis=1);keep=ds<R
  allb.extend(bary[keep]);allt.extend(targets[keep]);alld.extend(ds[keep])
 # Retain positive support at every previous source site. Replacing a source
 # quadrature must not uncover previously reachable receivers elsewhere.
 nearest,ind=cKDTree(cc[allowed]).query(v.mean(axis=0))
 reserve=.05
 old_begin,old_end=int(prior['parent_offsets'][p]),int(prior['parent_offsets'][p+1])
 if old_begin!=old_end:
  for j in range(old_begin,old_end):
   mandatory.append((float(prior['carrier_weights'][j]*reserve),prior['carrier_barycentric'][j],int(prior['carrier_target_child_rows'][j]),float(prior['carrier_target_distance_m'][j])))
 else:
  if nearest>=R:raise ValueError('previous source site has no authenticated witness')
  mandatory.append((reserve,np.full(4,.25),int(allowed[ind]),float(nearest)))
 rhs-=reserve*.25
 if np.any(rhs<0):raise ValueError('support reserve exhausted source moment')
 if nearest<R:allb.append(np.full(4,.25));allt.append(allowed[ind]);alld.append(nearest)
 b=np.asarray(allb);tg=np.asarray(allt,dtype=np.int64);ds=np.asarray(alld)
 fill=linprog(ds**2,A_eq=b.T,b_eq=rhs,bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
 if not fill.success:
  failures.append({'parent_row':p,'global_parent':g,'mandatory_sites':len(mandatory),'message':fill.message,'remaining_bary_mass':rhs.tolist()});continue
 keep=np.flatnonzero(fill.x>0)
 proposals=[('raw_positive_LP',fill.x[keep]),('refined_LP_basis',np.linalg.lstsq(b[keep].T,rhs,rcond=None)[0])]
 passed=[];proposal_witnesses=[]
 for method,fw in proposals:
  if np.min(fw)<-5e-13:
   proposal_witnesses.append({'method':method,'minimum_filler_weight':float(np.min(fw)),'status':'negative'});continue
  fw=np.maximum(0,fw);positive=fw>0
  rows0=mandatory+[(float(w),bb,int(c),float(d)) for w,bb,c,d in zip(fw[positive],b[keep][positive],tg[keep][positive],ds[keep][positive])]
  ww=np.asarray([x[0] for x in rows0]);bb=np.asarray([x[1] for x in rows0])
  sumerr=abs(float(ww.sum())-1);berr=float(np.max(np.abs(ww@bb-.25)))
  worlderr=float(np.linalg.norm(ww@(bb@v)-v.mean(axis=0)))
  witness={'method':method,'sum_error':sumerr,'barycentric_moment_error':berr,'world_moment_error_m':worlderr,'minimum_weight':float(ww.min())}
  proposal_witnesses.append(witness)
  if np.all(ww>0) and sumerr<=2e-12 and berr<=2e-10 and worlderr<=1e-11:
   passed.append((max(sumerr,berr,worlderr),rows0,ww,bb,method))
 if not passed:
  failures.append({'parent_row':p,'global_parent':g,'mandatory_sites':len(mandatory),'remaining_bary_mass':rhs.tolist(),'LP_basis_rows':keep.tolist(),'proposal_checks':proposal_witnesses});continue
 _,rows,weights,bary,selected_refinement=min(passed,key=lambda x:x[0])
 carrier_rows[p]=rows
 parent_reports.append({'parent_row':p,'global_parent':g,'mandatory_sites':len(mandatory),'filler_sites':len(rows)-len(mandatory),'refinement':selected_refinement,'proposal_checks':proposal_witnesses,'minimum_weight':float(weights.min()),'first_moment_error':float(np.max(np.abs(weights@bary-.25)))})
 if number%25==0:(out/'progress.json').write_text(json.dumps({'planned_parents':len(required),'completed_parents':number+1,'failures':len(failures),'elapsed_s':time.monotonic()-start})+'\n')
if failures:
 (out/'failure.json').write_text(json.dumps({'status':'INCONCLUSIVE_SOURCE_REMAINDER_QUADRATURE','failures':failures,'passed_parents':parent_reports},indent=2)+'\n');raise ValueError('source remainder quadrature failures '+str(len(failures)))
offsets=[0];weights=[];bary=[];targets=[];distances=[]
for i in range(npf):
 for we,bb,ch,dd in carrier_rows.get(i,[]):
  weights.append(we);bary.append(bb);targets.append(ch);distances.append(dd)
 offsets.append(len(weights))
arrays={'source_parent_tet_rows':np.arange(npf,dtype=np.uint32),'global_source330_tet_rows':global_rows,'parent_offsets':np.asarray(offsets,dtype=np.uint32),'carrier_weights':np.asarray(weights,dtype=np.float64),'carrier_barycentric':np.asarray(bary,dtype=np.float64),'carrier_target_child_rows':np.asarray(targets,dtype=np.uint32),'carrier_target_distance_m':np.asarray(distances,dtype=np.float64)}
np.savez_compressed(out/'source-quadrature.npz',**arrays)
with open(out/'source-quadrature.bin','wb') as f:
 f.write(b'FSSCARR01');f.write(struct.pack('<QQ',npf,len(weights)))
 for k in ('source_parent_tet_rows','global_source330_tet_rows','parent_offsets','carrier_weights','carrier_barycentric','carrier_target_child_rows','carrier_target_distance_m'):
  a=arrays[k];f.write(a.astype('<f8' if a.dtype.kind=='f' else '<u4').tobytes())
pins={p:sha(Path(p)) for p in manifest['inputs']}
for p in (T.with_name(T.name+'.nodes.csv'),T.with_name(T.name+'.tets.csv'),D/'semantic-support.bin',D/'child-reachability-157-root/report.json',D/'child-reachability-157-root/uncovered-child-p1-source-pools.npz',H/'connected-region-mass-support-143-root/independent-verification.json',D/'source-quadrature-154-root/source-quadrature.npz',D/'source-quadrature-154-root/independent-verification.json',Path(__file__),H/'batched_reference_tet_projection_149_root.py'):pins[str(p)]=sha(p)
receipt={'schema':'numi.human.source-tetra-centroid-carrier-quadrature.v1','status':'PASS_SOURCE_ONLY_BATCH_POSITIVE_FIRST_MOMENT_QUADRATURE','role':'Fat','physical_admission':False,'native_steps':0,'counts':{'source_parents':npf,'carrierized_parents':len(carrier_rows),'carriers':len(weights),'uncovered_children_addressed':len(holes),'minimum_demand_planning_pairs':len(ci)},'maximum_transport_distance_m':R,'minimum_receiver_mass_kg':float(demand.sum()),'method':'Positive degree1 source-tet quadrature. Required witness site mass is planned against actual850kg/m3 receiver minimum and at most80percentofeachoriginalsourcevertexlumpedbudget. Fivepercentofeachmodifiedparentspreviousvalidquadratureisretainedtopreserveeverypreviousgeometricadjacency. Remaining mass/firstmoment completed with authenticated same-material ball-bounded source sites. Exact allocation and re-census pending; not intersection-volume remapping. Source geometry, mass, donor/region and child density bounds unchanged.','inputs':pins,'parents':parent_reports,'outputs':{str(p):{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in (out/'source-quadrature.npz',out/'source-quadrature.bin')},'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','counts','minimum_receiver_mass_kg','elapsed_s')},indent=2))
