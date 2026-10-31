#!/Users/numi/numi-human-prep-venv-20261005/bin/python
"""Replace selected baseline Fat held rows with the independently replayed run055 primal."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'; BASE=S/'fat-held-exchange-031'; HELD=BASE/'run-009-global-held-merge-corrected/merged-held-state.npz'; HELD_RECEIPT=BASE/'run-009-global-held-merge-corrected/receipt.json'; QUAD=S/'fat-853406-run-003-root/source-quadrature-162-root/source-quadrature.npz'; SRC=BASE/'run-050-radius-separation-continuation-049-cuts'; FULL=SRC/'full-tet-input.npz'; PARENTCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'; DONORCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent-donor.csv'; RUN=BASE/'run-055-corrected-parent-mass-row-map/attempt-002'; PROPOSAL=RUN/'proposal.npz'; REPORT=RUN/'report.json'; INDEP=RUN/'independent-verification/report.json'; OUT=BASE/'run-056-global-held-merge-corrected055'/'attempt-002'; RLO=850.; RHI=950.; R=.019353

def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def rows(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def aggregate_donor(parent,child,mass,dOffsets,dIds,dMass,parentMass):
 counts=np.diff(dOffsets)[parent].astype(np.int64); total=int(counts.sum())
 flowIdx=np.repeat(np.arange(len(parent),dtype=np.int64),counts)
 if total==0:return np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float64),np.empty(0,np.int32),np.empty(0,np.float64)
 starts=np.repeat(dOffsets[parent],counts).astype(np.int64)
 repStarts=np.repeat(np.r_[0,np.cumsum(counts[:-1])],counts)
 within=np.arange(total,dtype=np.int64)-repStarts
 donorRow=starts+within
 donorId=dIds[donorRow]; val=mass[flowIdx]*(dMass[donorRow]/parentMass[parent[flowIdx]])
 order=np.lexsort((donorId,child[flowIdx])); ch=child[flowIdx][order]; di=donorId[order]; va=val[order]
 new=np.r_[True,(ch[1:]!=ch[:-1])|(di[1:]!=di[:-1])]
 starts2=np.flatnonzero(new); keyChild=ch[starts2]; keyDonor=di[starts2]; keyMass=np.add.reduceat(va,starts2)
 return keyChild.astype(np.int32),keyDonor.astype(np.int32),keyMass.astype(np.float64),donorId.astype(np.int32),val.astype(np.float64)
def equal_key_rows(c1,d1,m1,c2,d2,m2,rtol=2e-12):
 if not np.array_equal(c1,c2) or not np.array_equal(d1,d2) or m1.shape!=m2.shape:return False,float('inf')
 e=np.abs(m1-m2); rel=e/np.maximum(np.maximum(np.abs(m1),np.abs(m2)),1e-300)
 return bool(np.all(rel<=rtol)),float(rel.max(initial=0.))

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 paths=(HELD,HELD_RECEIPT,QUAD,FULL,PARENTCSV,DONORCSV,PROPOSAL,REPORT,INDEP)
 pins={str(p):sha(p) for p in paths}
 prod=json.loads(REPORT.read_text()); iv=json.loads(INDEP.read_text()); oldrec=json.loads(HELD_RECEIPT.read_text())
 if prod.get('status')!='PASS_BASELINE142_ONLY_FINITE_SITE_SOURCE_ONLY_CANDIDATE' or iv.get('status')!='PASS_SOURCE_ONLY_CORRECTED_FINITE_SITE_REPLAY':raise RuntimeError('run055 producer or independent replay not admitted to merge')
 if sha(PROPOSAL)!=prod['proposal_sha256'] or iv['input_pins'].get(str(PROPOSAL))!=sha(PROPOSAL):raise RuntimeError('run055 proposal is not pinned by both receipts')
 if oldrec.get('physical_admission') is not False:raise RuntimeError('old held state has unexpected admission status')
 parent=rows(PARENTCSV); donors=rows(DONORCSV)
 pMass=np.asarray([float(x['mass_kg']) for x in parent]); pVol=np.asarray([float(x['reference_volume_m3']) for x in parent]); pRows=np.asarray([int(x['parent_row']) for x in parent],dtype=np.int32)
 if not np.array_equal(pRows,np.arange(len(parent),dtype=np.int32)):raise RuntimeError('source parent row domain invalid')
 dp=np.asarray([int(x['parent_row']) for x in donors],dtype=np.int32); did=np.asarray([int(x['source_donor_body']) for x in donors],dtype=np.int32); dm=np.asarray([float(x['mass_kg']) for x in donors],dtype=np.float64)
 dOrder=np.lexsort((did,dp)); dp=dp[dOrder];did=did[dOrder];dm=dm[dOrder]; dCounts=np.bincount(dp,minlength=len(parent)); dOffsets=np.r_[0,np.cumsum(dCounts)].astype(np.int64)
 dByParent=np.bincount(dp,weights=dm,minlength=len(parent)); donorParentRel=float(np.max(np.abs(dByParent-pMass)/np.maximum(pMass,1e-300),initial=0.))
 if donorParentRel>1e-12:raise RuntimeError('source parent donor shares fail parent mass closure')
 with np.load(HELD,allow_pickle=False) as z:
  oldChild=z['child_row'].astype(np.int32,copy=True);oldParent=z['parent_row'].astype(np.int32,copy=True);oldExpanded=z['expanded_parent_row'].astype(np.uint32,copy=True);oldCarrier=z['carrier_row'].astype(np.uint32,copy=True);oldMass=z['mass_kg'].astype(np.float64,copy=True);oldMeasure=z['source_measure_m3'].astype(np.float64,copy=True);oldDist=z['distance_m'].astype(np.float64,copy=True)
  storedDonorChild=z['child_donor_child_row'].astype(np.int32,copy=True);storedDonorId=z['child_donor_id'].astype(np.int32,copy=True);storedDonorMass=z['child_donor_mass_kg'].astype(np.float64,copy=True)
  childVol=z['child_reference_volume_m3'].astype(np.float64,copy=True); oldChildMass=z['child_mass_kg'].astype(np.float64,copy=True);oldHeld=z['parent_mass_held_kg'].astype(np.float64,copy=True);oldRemain=z['parent_mass_remaining_kg'].astype(np.float64,copy=True);oldCorner=z['parent_corner_mass_held_kg'].astype(np.float64,copy=True);oldRemainCorner=z['parent_corner_mass_remaining_kg'].astype(np.float64,copy=True);oldRemainVol=z['parent_reference_volume_remaining_m3'].astype(np.float64,copy=True)
  sourceDonorIDs=z['source_donor_ids'].astype(np.int32,copy=True);oldDonorExpected=z['source_donor_mass_expected_kg'].astype(np.float64,copy=True);oldDonorHeld=z['source_donor_mass_held_kg'].astype(np.float64,copy=True);oldDonorRemain=z['source_donor_mass_remaining_kg'].astype(np.float64,copy=True)
 with np.load(QUAD,allow_pickle=False) as z:
  qParent=z['source_parent_tet_rows'].astype(np.uint32,copy=True);qGlobal=z['global_source330_tet_rows'].astype(np.uint32,copy=True);qOff=z['parent_offsets'].astype(np.int64,copy=True);qB=z['carrier_barycentric'].astype(np.float64,copy=True)
 with np.load(FULL,allow_pickle=False) as z:
  selected=z['selected_parent_role_rows'].astype(np.int32,copy=True);fullGlobal=z['selected_parent_global_source330_rows'].astype(np.int32,copy=True)
  recipientRows=z['recipient_child_rows'].astype(np.int32,copy=True);localLow=z['full_low_kg'].astype(np.float64,copy=True);localHigh=z['full_high_kg'].astype(np.float64,copy=True);localVol=z['child_volume_m3'].astype(np.float64,copy=True)
 with np.load(PROPOSAL,allow_pickle=False) as z:
  newChild=z['child_rows'].astype(np.int32,copy=True);newParent=z['parent_role_rows'].astype(np.int32,copy=True);newGlobal=z['global_source330_tet_rows'].astype(np.int32,copy=True);newMass=z['mass_kg'].astype(np.float64,copy=True);newB=z['barycentric'].astype(np.float64,copy=True);newDist=z['distance_m'].astype(np.float64,copy=True);newSite=z['site_rows'].astype(np.int32,copy=True);proposalParentRows=z['parent_rows'].astype(np.int32,copy=True);proposalRecipientRows=z['recipient_child_rows'].astype(np.int32,copy=True)
 if not np.array_equal(selected,proposalParentRows) or not np.isin(newParent,selected).all():raise RuntimeError('proposal source parent domain differs from selected closure')
 newParentLocal=np.searchsorted(selected,newParent)
 if not np.array_equal(selected[newParentLocal],newParent) or not np.array_equal(newGlobal,fullGlobal[newParentLocal]):raise RuntimeError('proposal role-parent to source330 global row map mismatch')
 if not np.array_equal(localVol,childVol[recipientRows]):raise RuntimeError('baseline142 local recipient volume differs from the held ledger')
 if len(oldParent)!=len(oldChild) or len(oldParent)!=len(oldMass):raise RuntimeError('old held flow fields shape mismatch')
 if int(oldChild.max(initial=0))>=len(childVol) or int(newChild.max(initial=0))>=len(childVol):raise RuntimeError('child row outside current Fat domain')
 if not np.array_equal(qParent,np.arange(len(parent),dtype=np.uint32)) or len(qOff)!=len(parent)+1 or int(qOff[-1])!=len(qB) or not np.array_equal(qGlobal,np.asarray([int(x['global_source330_row']) for x in parent],dtype=np.uint32)):raise RuntimeError('source quadrature parent domain/global identity mismatch')
 # Reconstruct actual old source-corner moments from old centroid rows and pinned source quadrature sites.
 sentinel=np.iinfo(np.uint32).max; oldB=np.full((len(oldMass),4),.25,dtype=np.float64); isCarrier=oldCarrier!=sentinel
 if np.any(isCarrier):
  cr=oldCarrier[isCarrier].astype(np.int64); pr=oldParent[isCarrier]
  if np.any(cr<qOff[pr]) or np.any(cr>=qOff[pr+1]):raise RuntimeError('old carrier row is not in its source-parent CSR')
  oldB[isCarrier]=qB[cr]
 oldCornerReplay=np.zeros((len(parent),4),dtype=np.float64)
 for k in range(4):np.add.at(oldCornerReplay[:,k],oldParent,oldMass*oldB[:,k])
 cornerOldRel=float(np.max(np.abs(oldCornerReplay-oldCorner)/np.maximum(pMass[:,None]*.25,1e-300),initial=0.))
 if cornerOldRel>2e-10:raise RuntimeError('old held source-corner ledger does not replay from held rows')
 oldMassReplay=np.bincount(oldParent,weights=oldMass,minlength=len(parent)); parentOldRel=float(np.max(np.abs(oldMassReplay-oldHeld)/np.maximum(pMass,1e-300),initial=0.))
 if parentOldRel>2e-10 or np.max(np.abs(oldRemain-(pMass-oldHeld))/np.maximum(pMass,1e-300),initial=0.)>2e-10:raise RuntimeError('old parent held/remaining ledger does not replay')
 oldChildReplay=np.bincount(oldChild,weights=oldMass,minlength=len(childVol)); childOldErr=float(np.max(np.abs(oldChildReplay-oldChildMass),initial=0.))
 if childOldErr>2e-12*max(1.,float(oldMass.sum())):raise RuntimeError('old child held mass does not replay')
 # Rebuild old donor ledger independently from each original parent donor composition.
 oc,od,om,_,_=aggregate_donor(oldParent,oldChild,oldMass,dOffsets,did,dm,pMass)
 order=np.lexsort((storedDonorId,storedDonorChild)); sc=storedDonorChild[order];sd=storedDonorId[order];sm=storedDonorMass[order]
 donorOldPass,donorOldRel=equal_key_rows(oc,od,om,sc,sd,sm,rtol=2e-10)
 if not donorOldPass:raise RuntimeError(f'old child donor ledger fails source-parent split replay (rel={donorOldRel})')
 # Replace every prior held flow from the selected parent closure once; keep all other old rows bitwise in row order.
 retire=np.isin(oldParent,selected); keep=~retire; retiredRows=np.flatnonzero(retire).astype(np.int32)
 oldSelectedMass=float(oldMass[retire].sum()); expectedOldSelected=float(oldHeld[selected].sum()); selectedHeldRows=int(retire.sum())
 if abs(oldSelectedMass-expectedOldSelected)>2e-12*max(1.,expectedOldSelected):raise RuntimeError('selected held-row retirement mass differs from parent ledger')
 # New local proposal allocates the entire actual source mass of each selected parent; source measure uses its pinned source volume/mass ratio.
 if np.any(newMass<=0.) or np.any(newDist>R) or np.any(newB<0.) or np.max(np.abs(newB.sum(axis=1)-1.),initial=0.)>2e-13:raise RuntimeError('new proposal no longer passes independent local geometry/mass checks')
 if not np.array_equal(np.unique(newParent),selected):raise RuntimeError('new proposal does not cover every selected source parent')
 newMeasure=newMass*pVol[newParent]/pMass[newParent]
 flowChild=np.r_[oldChild[keep],newChild];flowParent=np.r_[oldParent[keep],newParent];flowMass=np.r_[oldMass[keep],newMass];flowMeasure=np.r_[oldMeasure[keep],newMeasure];flowDistance=np.r_[oldDist[keep],newDist]
 origin=np.r_[np.zeros(int(keep.sum()),dtype=np.uint8),np.ones(len(newMass),dtype=np.uint8)]
 siteRows=np.r_[np.full(int(keep.sum()),-1,dtype=np.int32),newSite]
 priorExpanded=np.r_[oldExpanded[keep].astype(np.int64),np.full(len(newMass),-1,dtype=np.int64)]
 priorCarrier=np.r_[oldCarrier[keep].astype(np.int64),np.full(len(newMass),-1,dtype=np.int64)]
 # Recompute whole parent/corner, child, donor and residual ledgers from merged flows.
 heldByParent=np.bincount(flowParent,weights=flowMass,minlength=len(parent)); remByParent=pMass-heldByParent
 if np.any(remByParent < -1e-10*np.maximum(pMass,1e-300)):raise RuntimeError('merged held flow exceeds a source parent mass')
 cornerByParent=oldCornerReplay.copy()
 localCorner=np.zeros((len(parent),4),dtype=np.float64)
 for k in range(4):np.add.at(localCorner[:,k],newParent,newMass*newB[:,k])
 cornerByParent=oldCornerReplay.copy();
 for k in range(4):np.add.at(cornerByParent[:,k],oldParent[retire],-oldMass[retire]*oldB[retire,k])
 cornerByParent+=localCorner
 targetCorner=np.repeat((pMass*.25)[:,None],4,axis=1); remCorner=targetCorner-cornerByParent
 cornerRel=float(np.max(np.abs(cornerByParent[selected]-targetCorner[selected])/np.maximum(targetCorner[selected],1e-300),initial=0.))
 if cornerRel>1e-9 or np.min(remCorner)<-1e-10*np.max(pMass,initial=1.):raise RuntimeError('merged corner M/4 or remaining-corner ledger fails')
 childMass=np.bincount(flowChild,weights=flowMass,minlength=len(childVol)); density=childMass/childVol
 low=RLO*childVol;high=RHI*childVol;deficit=np.maximum(low-childMass,0.);over=np.maximum(childMass-high,0.)
 oldLowDef=np.maximum(low-oldChildMass,0.);oldOver=np.maximum(oldChildMass-high,0.)
 donorChild,donorID,donorMass,flowDonorIDs,flowDonorVals=aggregate_donor(flowParent,flowChild,flowMass,dOffsets,did,dm,pMass)
 allDonorIds=np.unique(did);expectedDonor=np.bincount(np.searchsorted(allDonorIds,did),weights=dm,minlength=len(allDonorIds));heldDonor=np.bincount(np.searchsorted(allDonorIds,flowDonorIDs),weights=flowDonorVals,minlength=len(allDonorIds));remainDonor=expectedDonor-heldDonor
 if np.any(remainDonor < -1e-10*np.maximum(expectedDonor,1e-300)):raise RuntimeError('merged donor held mass exceeds immutable donor source mass')
 childDonorTotal=np.bincount(donorChild,weights=donorMass,minlength=len(childVol)); donorChildErr=float(np.max(np.abs(childDonorTotal-childMass),initial=0.))
 if donorChildErr>2e-10*max(1.,float(childMass.sum())):raise RuntimeError('merged child donor contributions do not sum to held child mass')
 # Exact before/after global mass/count/deficit ledger.
 oldHeldTotal=float(oldMass.sum()); newHeldTotal=float(flowMass.sum()); expectedDelta=float(newMass.sum()-oldSelectedMass)
 assert abs((newHeldTotal-oldHeldTotal)-expectedDelta)<=2e-12*max(1.,oldHeldTotal)
 out=OUT/'merged-held-state-055.npz'
 np.savez_compressed(out,flow_child_row=flowChild,flow_parent_role_row=flowParent,flow_mass_kg=flowMass,flow_source_measure_m3=flowMeasure,flow_distance_m=flowDistance,flow_origin_kind=origin,flow_proposal_site_row=siteRows,flow_prior_expanded_parent_row=priorExpanded,flow_prior_carrier_row=priorCarrier,retired_original_held_flow_rows=retiredRows,selected_replaced_parent_rows=selected,proposal_site_parent_rows=newParent,proposal_site_child_rows=newChild,proposal_site_barycentric=newB,proposal_site_rows=newSite,child_mass_kg=childMass,child_reference_volume_m3=childVol,child_density_kg_m3=density,child_lower_deficit_kg=deficit,child_upper_excess_kg=over,child_donor_child_row=donorChild,child_donor_id=donorID,child_donor_mass_kg=donorMass,parent_mass_held_kg=heldByParent,parent_mass_remaining_kg=remByParent,parent_corner_mass_held_kg=cornerByParent,parent_corner_mass_remaining_kg=remCorner,parent_reference_volume_remaining_m3=pVol-np.bincount(flowParent,weights=flowMeasure,minlength=len(parent)),source_donor_ids=allDonorIds.astype(np.int32),source_donor_mass_expected_kg=expectedDonor,source_donor_mass_held_kg=heldDonor,source_donor_mass_remaining_kg=remainDonor)
 # Read the compressed output back and confirm retained rows and candidate append survived serialization.
 with np.load(out,allow_pickle=False) as z:
  if not np.array_equal(z['flow_parent_role_row'],flowParent) or not np.array_equal(z['flow_child_row'],flowChild) or not np.array_equal(z['flow_origin_kind'],origin):raise RuntimeError('merged ledger row maps changed after serialization')
  if not np.array_equal(z['retired_original_held_flow_rows'],retiredRows):raise RuntimeError('retired flow domain changed after serialization')
 report={'schema':'numi.human.fat-held-ledger-plus-run055-local-replacement.v1','status':'PASS_SOURCE_ONLY_LOCAL_REPLACEMENT_MERGED_GLOBAL_LEDGER_INCOMPLETE','geometry_scope':'baseline142 only; not final Fat geometry','physical_admission':False,'mass_transport':'PARTIAL_SOURCE_ONLY_NOT_ADMITTED','native_steps':0,'inputs':pins,'producer_report_sha256':sha(REPORT),'independent_local_replay_sha256':sha(INDEP),'output_path':str(out),'output_sha256':sha(out),'selected_parent_count':len(selected),'selected_parent_global_rows_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),'retired_old_held_flow_rows':selectedHeldRows,'retired_old_held_mass_kg':oldSelectedMass,'replacement_proposal_rows':len(newMass),'replacement_proposal_mass_kg':float(newMass.sum()),'net_held_mass_change_kg':float(newHeldTotal-oldHeldTotal),'held_mass_before_kg':oldHeldTotal,'held_mass_after_kg':newHeldTotal,'selected_source_mass_kg':float(pMass[selected].sum()),'selected_parent_remaining_mass_before_kg':float(oldRemain[selected].sum()),'selected_parent_remaining_mass_after_kg':float(remByParent[selected].sum()),'global_source_mass_kg':float(pMass.sum()),'global_held_parent_mass_before_kg':float(oldHeld.sum()),'global_held_parent_mass_after_kg':float(heldByParent.sum()),'global_remaining_parent_mass_before_kg':float(oldRemain.sum()),'global_remaining_parent_mass_after_kg':float(remByParent.sum()),'global_corner_remaining_kg_before':float(oldRemainCorner.sum()),'global_corner_remaining_kg_after':float(remCorner.sum()),'selected_parent_M_over_4_max_relative_error':cornerRel,'old_corner_reconstruction_max_relative_error':cornerOldRel,'old_parent_mass_reconstruction_max_relative_error':parentOldRel,'old_child_mass_reconstruction_max_abs_error_kg':childOldErr,'old_child_donor_reconstruction_max_relative_error':donorOldRel,'new_child_donor_total_abs_error_kg':donorChildErr,'global_child_count':int(len(childVol)),'unfunded_children_before_count':int(np.count_nonzero(oldLowDef>1e-15)),'unfunded_children_after_count':int(np.count_nonzero(deficit>1e-15)),'unfunded_children_before_total_deficit_kg':float(oldLowDef.sum()),'unfunded_children_after_total_deficit_kg':float(deficit.sum()),'unfunded_children_before_max_deficit_kg':float(oldLowDef.max(initial=0.)),'unfunded_children_after_max_deficit_kg':float(deficit.max(initial=0.)),'upper_excess_children_after_count':int(np.count_nonzero(over>1e-15)),'source_donor_count':int(len(allDonorIds)),'source_donor_expected_mass_kg':float(expectedDonor.sum()),'source_donor_held_mass_kg':float(heldDonor.sum()),'source_donor_remaining_mass_kg':float(remainDonor.sum()),'proposal_radius_max_m':float(newDist.max(initial=0.)),'local_recipients':len(recipientRows),'local_receiver_max_low_deficit_kg':float(np.maximum(localLow-childMass[recipientRows],0.).max(initial=0.)),'local_receiver_max_overcap_kg':float(np.maximum(childMass[recipientRows]-localHigh,0.).max(initial=0.)),'local_receiver_bounds_pass':bool(np.all(childMass[recipientRows]>=localLow-1e-10*np.maximum(localHigh,1e-300)) and np.all(childMass[recipientRows]<=localHigh+1e-10*np.maximum(localHigh,1e-300))),'global_allocation_complete':False,'global_receiver_bounds_pass':bool(not np.any(deficit>1e-15) and not np.any(over>1e-15)),'row_semantics':'kept prior flow rows unchanged except every row from selected source parents; appended run055 barycentric proposal rows with source measure = proposal mass * pinned source parent volume / pinned source parent mass. New rows have explicit proposal-site IDs; prior carrier IDs are -1 there and must be converted to a production carrier packet before export.'}
 reportPath=OUT/'report.json'; reportPath.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 freeze={'schema':'numi.human.fat-held-ledger-run055-merge-freeze.v1','driver_path':str(Path(__file__)),'driver_sha256':sha(Path(__file__)),'input_pins':pins,'output_pins':{'merged-held-state-055.npz':report['output_sha256'],'report.json':sha(reportPath)},'status':report['status'],'physical_admission':False,'native_steps':0}
 (OUT/'source-freeze.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:report[k] for k in ['status','retired_old_held_flow_rows','retired_old_held_mass_kg','replacement_proposal_rows','replacement_proposal_mass_kg','held_mass_before_kg','held_mass_after_kg','global_remaining_parent_mass_before_kg','global_remaining_parent_mass_after_kg','global_corner_remaining_kg_before','global_corner_remaining_kg_after','unfunded_children_before_count','unfunded_children_after_count','unfunded_children_before_total_deficit_kg','unfunded_children_after_total_deficit_kg','old_corner_reconstruction_max_relative_error','old_child_donor_reconstruction_max_relative_error','selected_parent_M_over_4_max_relative_error','output_sha256']},indent=2),flush=True)
if __name__=='__main__':main()
