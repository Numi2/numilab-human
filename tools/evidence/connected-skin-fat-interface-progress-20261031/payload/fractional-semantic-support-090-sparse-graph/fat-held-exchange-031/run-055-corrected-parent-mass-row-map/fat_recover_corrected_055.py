#!/Users/numi/numi-human-prep-venv-20261005/bin/python
"""Corrected Fat fixed-radius finite-site recovery; parent masses keyed by global role rows."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csr_matrix
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'; BASE=S/'fat-held-exchange-031'; SRC=BASE/'run-050-radius-separation-continuation-049-cuts'; PREV=BASE/'run-053-primal-recovery-farkas-site-cut'; OUT=BASE/'run-055-corrected-parent-mass-row-map'/'attempt-001'
FULL=SRC/'full-tet-input.npz'; ITER=SRC/'iteration-19.npz'; SRC_REPORT=SRC/'report.json'; SRC_FREEZE=SRC/'source-freeze.json'; SEED=SRC/'continuation-seed.json'; LPIN=PREV/'recovery-lp-input.npz'; PREV_REPORT=PREV/'report.json'; PARENTCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'
R=.019353; LO=850.; HI=950.; TIME_LIMIT=55.
def sha(p): return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def readcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def matrices(parent_local,recipient_local,bary,parent_mass_per_site,full_low,full_high,fixed,P,NR):
 n=len(parent_local); B=np.asarray(bary,dtype=np.float64); pm=np.asarray(parent_mass_per_site,dtype=np.float64)
 erows=(parent_local[:,None]*4+np.arange(4,dtype=np.int32)[None,:]).ravel(); ecols=np.repeat(np.arange(n,dtype=np.int32),4)
 Aeq=coo_matrix((B.ravel(),(erows,ecols)),shape=(4*P,n)).tocsr(); beq=np.full(4*P,.25,dtype=np.float64)
 coef=pm/full_high[recipient_local]; ur=2*recipient_local; lr=ur+1
 A=coo_matrix((np.r_[coef,-coef],(np.r_[ur,lr],np.r_[np.arange(n),np.arange(n)])),shape=(2*NR,n)).tocsr()
 b=np.empty(2*NR,dtype=np.float64); b[::2]=(full_high-fixed)/full_high; b[1::2]=(fixed-full_low)/full_high
 return Aeq,beq,A,b

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 paths=(FULL,ITER,SRC_REPORT,SRC_FREEZE,SEED,LPIN,PREV_REPORT,PARENTCSV)
 pins={str(p):sha(p) for p in paths}
 prev=json.loads(PREV_REPORT.read_text()); src=json.loads(SRC_REPORT.read_text())
 if prev.get('status')!='LP_INFEASIBLE_FINITE_RECOVERY_SITE_SET_DIAGNOSTIC':raise RuntimeError('wrong frozen source candidate status')
 if src.get('status')!='LP_OPTIMAL_CONSERVATIVE_RADIUS_TARGET_NOT_CONVERGED':raise RuntimeError('wrong continuous source status')
 for p,h in src['input_hashes'].items():
  if sha(Path(p))!=h:raise RuntimeError('source run050 pin mismatch '+p)
 with np.load(FULL,allow_pickle=False) as z:
  parents=z['selected_parent_role_rows'].astype(np.int32,copy=True); globalRows=z['selected_parent_global_source330_rows'].astype(np.int32,copy=True)
  pairParent=z['pair_parent_role_rows'].astype(np.int32,copy=True); pairChild=z['pair_child_rows'].astype(np.int32,copy=True); nearest=z['pair_nearest_barycentric'].astype(np.float64,copy=True)
  V0=z['pair_parent_source_vertices_m'].astype(np.float64,copy=True); E=z['pair_parent_source_edges_m'].astype(np.float64,copy=True); C=z['pair_child_centroids_m'].astype(np.float64,copy=True)
  recipients=z['recipient_child_rows'].astype(np.int32,copy=True); fixed=z['fixed_mass_kg'].astype(np.float64,copy=True); childV=z['child_volume_m3'].astype(np.float64,copy=True); low=z['full_low_kg'].astype(np.float64,copy=True); high=z['full_high_kg'].astype(np.float64,copy=True)
 with np.load(ITER,allow_pickle=False) as z:
  if not np.array_equal(z['pair_parent_role_rows'],pairParent) or not np.array_equal(z['pair_child_rows'],pairChild):raise RuntimeError('continuous pair row maps differ')
 par=readcsv(PARENTCSV); parentRows=np.asarray([int(r['parent_row']) for r in par],dtype=np.int32); pMass=np.asarray([float(r['mass_kg']) for r in par],dtype=np.float64)
 if not np.array_equal(parentRows,np.arange(len(par),dtype=np.int32)):raise RuntimeError('parent CSV row identity not contiguous')
 P=len(parents); NR=len(recipients); Ecount=len(pairParent)
 if P!=268 or Ecount!=27016 or NR!=2397:raise RuntimeError('unexpected pinned row domains')
 pairParentLocal=np.searchsorted(parents,pairParent); pairRecLocal=np.searchsorted(recipients,pairChild)
 if not np.array_equal(parents[pairParentLocal],pairParent) or not np.array_equal(recipients[pairRecLocal],pairChild):raise RuntimeError('pair to selected row map mismatch')
 with np.load(LPIN,allow_pickle=False) as z:
  sitePair=z['site_pair_rows'].astype(np.int32,copy=True); siteParent=z['site_parent_role_rows'].astype(np.int32,copy=True); siteChild=z['site_child_rows'].astype(np.int32,copy=True); siteB=z['site_barycentric'].astype(np.float64,copy=True); siteD=z['site_distance_m'].astype(np.float64,copy=True); siteKind=z['site_kind'].astype('U20',copy=True)
  Aeq_old=csr_matrix((z['Aeq_data'].copy(),z['Aeq_indices'].copy(),z['Aeq_indptr'].copy()),shape=tuple(z['Aeq_shape']))
  Aub_old=csr_matrix((z['Aub_data'].copy(),z['Aub_indices'].copy(),z['Aub_indptr'].copy()),shape=tuple(z['Aub_shape']))
 if len(sitePair)!=186472 or not np.array_equal(siteParent,pairParent[sitePair]) or not np.array_equal(siteChild,pairChild[sitePair]):raise RuntimeError('frozen site-to-pair map mismatch')
 siteParentLocal=np.searchsorted(parents,siteParent); siteRecLocal=np.searchsorted(recipients,siteChild)
 if not np.array_equal(parents[siteParentLocal],siteParent) or not np.array_equal(recipients[siteRecLocal],siteChild):raise RuntimeError('site selected role row crosswalk mismatch')
 if np.any(siteB<0.) or np.max(np.abs(siteB.sum(axis=1)-1.))>2e-13 or np.any(siteD>R):raise RuntimeError('candidate sites fail frozen simplex/radius constraints')
 # Noncontiguous row-ID control catches local/global source mass confusion in the actual matrix helper.
 toyIDs=np.array([4274,6006],dtype=np.int32); toyMass=pMass
 toyLocal=np.array([0,1],dtype=np.int32); toyRec=np.array([0,0],dtype=np.int32); toyB=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
 tAeq,tbeq,tAub,tbub=matrices(toyLocal,toyRec,toyB,toyMass[toyIDs],np.array([0.]),np.array([.02]),np.array([0.]),2,1)
 expected=np.array([toyMass[4274]/.02,toyMass[6006]/.02])
 got=tAub[0].toarray().ravel()
 assert np.array_equal(got,expected) and not np.array_equal(got,np.array([toyMass[0]/.02,toyMass[1]/.02]))
 controls={'status':'PASS_NONCONTIGUOUS_GLOBAL_PARENT_MASS_ROW_CONTROL','selected_global_role_rows':toyIDs.tolist(),'local_parent_rows':toyLocal.tolist(),'coefficient_expected_global_mass_over_receiver_high':expected.tolist(),'coefficient_actual':got.tolist(),'incorrect_local_index_values':(toyMass[toyLocal]/.02).tolist()}
 # Rebuild every LP coefficient from immutable global row identities. The run053 A_ub is only retained as a comparison witness.
 Aeq,beq,Aub,bub=matrices(siteParentLocal,siteRecLocal,siteB,pMass[siteParent],low,high,fixed,P,NR)
 if Aeq.shape!=Aeq_old.shape or (Aeq-Aeq_old).nnz and np.max(np.abs((Aeq-Aeq_old).data),initial=0.)>0.:raise RuntimeError('source-corner equality matrix differs from frozen site matrix')
 diff=(Aub-Aub_old).tocsr(); maxDiff=float(np.abs(diff.data).max(initial=0.));
 # The first persisted row demonstrates the old local-index error quantitatively.
 first=0; firstPair=int(sitePair[first]); firstGlobal=int(siteParent[first]); firstLocal=int(siteParentLocal[first]); firstRec=int(siteRecLocal[first]); stored=float(Aub_old[2*firstRec,first]); actual=float(Aub[2*firstRec,first])
 if not np.isclose(actual,pMass[firstGlobal]/high[firstRec],rtol=0,atol=1e-12):raise RuntimeError('corrected mass coefficient does not use global role row')
 cost=siteD/R+np.arange(len(sitePair),dtype=np.float64)*(1e-14/max(1,len(sitePair)))
 t0=time.time(); res=linprog(cost,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=[(0.,1.)]*len(sitePair),method='highs',options={'time_limit':TIME_LIMIT,'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10}); solveSec=time.time()-t0
 report={'schema':'numi.human.fat-radius-corrected-global-parent-row-recovery.v1','status':'SOLVER_'+str(res.status),'method':'Run053 radius-valid candidate sites, re-solved with receiver mass coefficients keyed by global source role parent row; mass M/4 corner constraints and receiver density bounds unchanged.','geometry_scope':'baseline142_only_not_final_mesh','physical_admission':False,'mass_transport':'NOT_ADMITTED','native_steps':0,'hard_radius_m':R,'density_bounds_kg_m3':[LO,HI],'parent_variable_unit':'x_site is dimensionless fraction of the actual global role-row source-parent mass; each parent has four sum(barycentric*x_site)=0.25 equations.','receiver_coefficient_unit':'actual source-parent mass kg divided by recipient full-high kg; receiver rows are dimensionless fractions of full-high.','selected_parent_count':P,'recipient_count':NR,'candidate_site_count':len(sitePair),'pair_count':Ecount,'source_parent_mass_selected_kg':float(pMass[parents].sum()),'candidate_site_max_distance_m':float(siteD.max(initial=0.)),'candidate_barycentric_min':float(siteB.min(initial=0.)),'solver_success':bool(res.success),'solver_status':int(res.status),'solver_message':str(res.message),'solver_seconds':solveSec,'time_limit_seconds':TIME_LIMIT,'input_pins':pins,'legacy_run053_A_ub_status':'superseded due parent mass indexing by selected-parent local row','legacy_run053_A_ub_max_abs_difference':maxDiff,'first_site_mass_index_witness':{'site_row':first,'pair_row':firstPair,'global_parent_role_row':firstGlobal,'selected_parent_local_row':firstLocal,'recipient_local_row':firstRec,'actual_source_mass_kg':float(pMass[firstGlobal]),'incorrect_local_source_mass_kg':float(pMass[firstLocal]),'actual_receiver_coefficient':actual,'persisted_legacy_coefficient':stored,'receiver_full_high_kg':float(high[firstRec])},'noncontiguous_row_control':controls,'input_site_census_sha256':sha(PREV/'recovery-site-census.json'),'input_recovery_matrix_sha256':sha(LPIN),'source_report_sha256':sha(SRC_REPORT),'run053_infeasibility_is_superseded':True}
 if res.x is not None:
  x=np.asarray(res.x,dtype=np.float64); pos=x>1e-14; mass=pMass[siteParent[pos]]*x[pos]
  cm=np.zeros((P,4),dtype=np.float64)
  for k in range(4):np.add.at(cm[:,k],siteParentLocal[pos],mass*siteB[pos,k])
  srcTotal=cm.sum(axis=1); target=np.repeat((pMass[parents]*.25)[:,None],4,axis=1)
  delivered=np.bincount(siteRecLocal[pos],weights=mass,minlength=NR); final=fixed+delivered; density=final/childV
  distances=siteD[pos]; bary=siteB[pos]; parentRel=float(np.max(np.abs(srcTotal-pMass[parents])/np.maximum(pMass[parents],1e-300)))
  cornerRel=float(np.max(np.abs(cm-target)/np.maximum(target,1e-300)))
  recTol=1e-10*np.maximum(high,1e-300); recPass=bool(np.all(final>=low-recTol)&np.all(final<=high+recTol))
  hardPass=bool(np.all(distances<=R)); simplexPass=bool(np.all(bary>=0.) and np.all(np.abs(bary.sum(axis=1)-1.)<=2e-13)); boxPass=bool(np.all(x>=-1e-12)&np.all(x<=1.+1e-12))
  report.update({'positive_site_count':int(pos.sum()),'recovered_source_mass_kg':float(srcTotal.sum()),'selected_source_mass_kg':float(pMass[parents].sum()),'recovered_flow_mass_kg':float(mass.sum()),'max_source_parent_total_relative_error':parentRel,'max_source_parent_corner_M_over_4_relative_error':cornerRel,'max_receiver_low_deficit_kg':float(np.maximum(low-final,0.).max(initial=0.)),'max_receiver_overcap_kg':float(np.maximum(final-high,0.).max(initial=0.)),'receiver_bounds_pass':recPass,'source_corner_moments_pass':bool(cornerRel<=1e-9),'source_parent_mass_pass':bool(parentRel<=1e-9),'hard_radius_pass':hardPass,'simplex_pass':simplexPass,'site_variable_bounds_pass':boxPass,'density_min_kg_m3':float(density.min(initial=np.inf)),'density_max_kg_m3':float(density.max(initial=0.)),'max_assembled_eq_abs_residual':float(np.max(np.abs(Aeq@x-beq),initial=0.)),'max_assembled_ub_violation':float(np.maximum(Aub@x-bub,0.).max(initial=0.))})
  if res.success and hardPass and simplexPass and boxPass and recPass and cornerRel<=1e-9 and parentRel<=1e-9:
   proposal=OUT/'proposal.npz'; sourcexyz=V0[sitePair[pos]]+np.einsum('ni,nij->nj',bary[:,1:],E[sitePair[pos]])
   np.savez_compressed(proposal,site_rows=np.flatnonzero(pos).astype(np.int32),pair_rows=sitePair[pos],parent_role_rows=siteParent[pos],global_source330_tet_rows=globalRows[siteParent[pos]],child_rows=siteChild[pos],mass_kg=mass,barycentric=bary,source_point_world_m=sourcexyz,child_centroid_world_m=C[sitePair[pos]],distance_m=distances,site_kind=siteKind[pos],parent_corner_mass_kg=mass[:,None]*bary,recipient_child_rows=recipients,recipient_final_mass_kg=final,recipient_density_kg_m3=density,parent_rows=parents,parent_corner_mass_total_kg=cm)
   report.update({'status':'PASS_BASELINE142_ONLY_FINITE_SITE_SOURCE_ONLY_CANDIDATE','proposal_path':str(proposal),'proposal_sha256':sha(proposal),'complete_fat_allocation':False,'current_geometry_final':False})
 if res.status==2:report['status']='LP_INFEASIBLE_CORRECTED_GLOBAL_PARENT_MASS_SITE_SET'
 reportPath=OUT/'report.json'; reportPath.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 freeze={'schema':'numi.human.fat-radius-corrected-global-parent-row-freeze.v1','driver_path':str(Path(__file__)),'driver_sha256':sha(Path(__file__)),'input_pins':pins,'output_pins':{'report.json':sha(reportPath)},'physical_admission':False,'native_steps':0}
 (OUT/'source-freeze.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:report.get(k) for k in ['status','solver_status','solver_message','solver_seconds','candidate_site_count','legacy_run053_A_ub_max_abs_difference','first_site_mass_index_witness','positive_site_count','recovered_flow_mass_kg','max_source_parent_corner_M_over_4_relative_error','receiver_bounds_pass','hard_radius_pass','density_min_kg_m3','density_max_kg_m3','proposal_sha256']},indent=2),flush=True)
if __name__=='__main__':main()
