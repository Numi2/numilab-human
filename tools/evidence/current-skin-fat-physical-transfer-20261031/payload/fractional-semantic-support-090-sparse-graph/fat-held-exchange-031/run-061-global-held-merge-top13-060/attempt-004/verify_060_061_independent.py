#!/Users/numi/numi-human-prep-venv-20261005/bin/python
"""Independent source-only replay of frozen run060 and run061."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
T0=time.time()
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209'); H=ROOT/'fine-physical-volume-706'; S=ROOT/'fractional-semantic-support-090-sparse-graph'; B=S/'fat-held-exchange-031'
OLD=B/'run-056-global-held-merge-corrected055/attempt-002/merged-held-state-055.npz'; PROP=B/'run-060-top13-exact-tet-closure/attempt-002/proposal.npz'; PR=PROP.parent/'report.json'; PF=PROP.parent/'source-freeze.json'; MERGE=B/'run-061-global-held-merge-top13-060/attempt-004/merged-held-state-060.npz'; MR=MERGE.parent/'report.json'; MF=MERGE.parent/'source-freeze.json'; OUT=MERGE.parent/'independent-verification-003.json'
PCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'; DCsv=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent-donor.csv'; Q=S/'fat-853406-run-003-root/source-quadrature-162-root/source-quadrature.npz'; NODE=H/'combined-child-handoff-142/fat.node'; ELE=H/'combined-child-handoff-142/fat.ele'; SUP=H/'connected-region-mass-support-143-root/fat-connected-region-support.npz'; SNODE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.nodes.csv'; TET=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.tets.csv'
R=.019353; LO=850.; HI=950.; U32=np.iinfo(np.uint32).max
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
def rows(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def eq(a,b,n):
 if not np.array_equal(a,b):raise RuntimeError('mismatch: '+n)
def rel(a,b):return float(np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-300),initial=0.))
def aggregate_donor(par,ch,mass,off,ids,dmass,pmass):
 counts=np.diff(off)[par].astype(np.int64);total=int(counts.sum());fi=np.repeat(np.arange(len(par),dtype=np.int64),counts);starts=np.repeat(off[par],counts);base=np.repeat(np.r_[0,np.cumsum(counts[:-1])],counts);di=starts+np.arange(total)-base;did=ids[di];val=mass[fi]*(dmass[di]/pmass[par[fi]]);o=np.lexsort((did,ch[fi]));c=ch[fi][o];d=did[o];v=val[o];start=np.r_[True,(c[1:]!=c[:-1])|(d[1:]!=d[:-1])];ix=np.flatnonzero(start);return c[ix].astype(np.int32),d[ix].astype(np.int32),np.add.reduceat(v,ix)
def main():
 mr=json.loads(MR.read_text());pr=json.loads(PR.read_text());pf=json.loads(PF.read_text());mf=json.loads(MF.read_text())
 for p,h in mr['input_pins'].items():
  if sha(Path(p))!=h:raise RuntimeError('061 input pin mismatch '+p)
 if sha(MERGE)!=mr['output_sha256'] or sha(PROP)!=pr['proposal_sha256']:raise RuntimeError('proposal or merge output pin mismatch')
 if pf['output_pins'].get('proposal.npz')!=sha(PROP) or pf['output_pins'].get('report.json')!=sha(PR):raise RuntimeError('060 source freeze mismatch')
 if mf['output_pins'].get('merged-held-state-060.npz')!=sha(MERGE) or mf['output_pins'].get('report.json')!=sha(MR):raise RuntimeError('061 source freeze mismatch')
 parents=rows(PCSV); donor=rows(DCsv); npar=len(parents); prow=np.array([int(x['parent_row']) for x in parents],np.int32);grow=np.array([int(x['global_source330_row']) for x in parents],np.int32);pm=np.array([float(x['mass_kg']) for x in parents]);pv=np.array([float(x['reference_volume_m3']) for x in parents]);pmt=np.array([int(x['material']) for x in parents]);preg=np.array([int(x['region']) for x in parents]);eq(prow,np.arange(npar,dtype=np.int32),'source parent rows')
 dp=np.array([int(x['parent_row']) for x in donor]);did=np.array([int(x['source_donor_body']) for x in donor]);dm=np.array([float(x['mass_kg']) for x in donor]);o=np.lexsort((did,dp));dp=dp[o];did=did[o];dm=dm[o];dcount=np.bincount(dp,minlength=npar);doff=np.r_[0,np.cumsum(dcount)].astype(np.int64)
 if rel(np.bincount(dp,weights=dm,minlength=npar),pm)>1e-12:raise RuntimeError('source donor to parent mass')
 with np.load(OLD,allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
 with np.load(PROP,allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
 with np.load(Q,allow_pickle=False) as z:qoff=z['parent_offsets'].astype(np.int64);qb=z['carrier_barycentric'].copy();qglobal=z['global_source330_tet_rows'].astype(np.int64)
 eq(qglobal,grow,'source quadrature global map');sel=p['parent_rows'].astype(np.int32);pp=p['parent_role_rows'].astype(np.int32);cc=p['child_rows'].astype(np.int32);m=p['mass_kg'];b=p['barycentric'];d=p['distance_m'];eq(sel,np.sort(np.unique(sel)),'selected parents');eq(np.unique(pp),sel,'proposal parent roster');eq(p['global_source330_rows'],grow[pp],'proposal parent-global rows');eq(sel,np.asarray(pr['selected_parent_rows'],np.int32),'producer parent roster')
 if np.any(m<=0) or np.any(d>R) or np.any(b<0) or np.max(np.abs(b.sum(1)-1),initial=0)>2e-13:raise RuntimeError('proposal mass/radius/simplex')
 sn=rows(SNODE);sx=np.asarray([[float(x['rest_'+a+'_m']) for a in 'xyz'] for x in sn]);tr=rows(TET);tets=np.asarray([[int(x['parent_node'+str(k)]) for k in range(4)] for x in tr],np.int32);xyz=np.einsum('ni,nij->nj',b,sx[tets[grow[pp]]])
 if np.max(np.abs(xyz-p['source_point_world_m']),initial=0)>5e-15:raise RuntimeError('proposal source coordinates')
 nd=np.loadtxt(NODE,skiprows=1,usecols=(0,1,2,3),ndmin=2);nids=nd[:,0].astype(np.int64);nxyz=nd[:,1:4].astype(np.float32).astype(np.float64);el=np.loadtxt(ELE,skiprows=1,ndmin=2);ix=np.searchsorted(nids,el[:,1:5].astype(np.int64))
 if np.any(ix>=len(nids)) or not np.array_equal(nids[ix],el[:,1:5].astype(np.int64)):raise RuntimeError('child node map')
 cent=nxyz[ix].mean(1);cmat=el[:,5].astype(np.int32)
 if not np.array_equal(cent[cc],p['child_centroid_world_m']):raise RuntimeError('proposal child centroids')
 dcalc=np.linalg.norm(xyz-cent[cc],axis=1)
 if np.max(np.abs(dcalc-d),initial=0)>5e-15 or np.any(dcalc>R):raise RuntimeError('proposal distance')
 with np.load(SUP,allow_pickle=False) as z:ro=z['cell_admissible_region_offsets'];ri=z['cell_admissible_region_ids']
 if np.any(pmt[pp]!=cmat[cc]):raise RuntimeError('proposal material')
 for x,y in zip(pp,cc):
  if not np.any(ri[ro[y]:ro[y+1]]==preg[x]):raise RuntimeError('proposal one-hop region')
 corner=np.zeros((len(sel),4));loc=np.searchsorted(sel,pp)
 for k in range(4):np.add.at(corner[:,k],loc,m*b[:,k])
 target=np.repeat((pm[sel]*.25)[:,None],4,axis=1)
 if rel(corner,target)>1e-9 or np.max(np.abs(corner-p['parent_corner_mass_total_kg']),initial=0)>5e-12:raise RuntimeError('proposal M/4')
 with np.load(MERGE,allow_pickle=False) as z:out={k:z[k].copy() for k in z.files}
 oc=old['flow_child_row'];op=old['flow_parent_role_row'];om=old['flow_mass_kg'];ret=np.isin(op,sel);keep=~ret;nr=p['recipient_child_rows'];fixed=old['child_mass_kg'][nr]-np.bincount(np.searchsorted(nr,oc[ret]),weights=om[ret],minlength=len(nr));final=fixed+np.bincount(np.searchsorted(nr,cc),weights=m,minlength=len(nr));cv=old['child_reference_volume_m3']
 if np.max(np.abs(fixed-p['recipient_fixed_mass_kg']),initial=0)>5e-12 or np.max(np.abs(final-p['recipient_final_mass_kg']),initial=0)>5e-12:raise RuntimeError('local fixed/final recipient reconstruction')
 if np.any(final<LO*cv[nr]-1e-10*np.maximum(HI*cv[nr],1e-300)) or np.any(final>HI*cv[nr]+1e-10*np.maximum(HI*cv[nr],1e-300)):raise RuntimeError('local density bounds')
 fc=out['flow_child_row'];fp=out['flow_parent_role_row'];fm=out['flow_mass_kg'];eq(fc,np.r_[oc[keep],cc],'merged child row order');eq(fp,np.r_[op[keep],pp],'merged parent row order');eq(fm,np.r_[om[keep],m],'merged flow mass/order');eq(out['flow_distance_m'],np.r_[old['flow_distance_m'][keep],d],'merged distance/order');measure=m*pv[pp]/pm[pp];eq(out['flow_source_measure_m3'],np.r_[old['flow_source_measure_m3'][keep],measure],'merged source measure/order');eq(out['retired_original_held_flow_rows'],np.flatnonzero(ret).astype(np.int32),'retired rows');eq(out['selected_replaced_parent_rows'],sel,'replaced parents')
 kind=out['flow_origin_kind'];car=out['flow_prior_carrier_row'];site=out['flow_proposal_site_row'];bb=np.full((len(fp),4),.25);isc=(kind==0)&(car!=U32);isp=kind==1
 if np.any(isc):
  qr=car[isc].astype(np.int64);px=fp[isc]
  if np.any(qr<qoff[px])|np.any(qr>=qoff[px+1]):raise RuntimeError('old carrier parent CSR')
  bb[isc]=qb[qr]
 sr=out['proposal_site_rows'];sp=out['proposal_site_parent_rows'];sc=out['proposal_site_child_rows'];sb=out['proposal_site_barycentric'];si=np.searchsorted(sr,site[isp])
 if np.any(si>=len(sr)) or not np.array_equal(sr[si],site[isp]) or not np.array_equal(sp[si],fp[isp]) or not np.array_equal(sc[si],fc[isp]):raise RuntimeError('proposal site ancestry')
 bb[isp]=sb[si]
 if np.min(bb)<-2e-12 or np.max(np.abs(bb.sum(1)-1),initial=0)>2e-12:raise RuntimeError('merged barycentric simplex')
 sxflow=np.einsum('ni,nij->nj',bb,sx[tets[grow[fp]]]);dr=np.linalg.norm(sxflow-cent[fc],axis=1)
 if np.any(dr>R+1e-12) or np.max(np.abs(dr-out['flow_distance_m']),initial=0)>5e-12:raise RuntimeError('whole merged flow geometry')
 held=np.bincount(fp,weights=fm,minlength=npar);cor=np.zeros((npar,4))
 for k in range(4):np.add.at(cor[:,k],fp,fm*bb[:,k])
 remain=pm-held;qremain=pm[:,None]*.25-cor
 if np.min(remain)<-1e-10 or np.min(qremain)<-1e-10:raise RuntimeError('source mass overdraw')
 for a,z,scale,n in [(held,out['parent_mass_held_kg'],pm,'held'),(remain,out['parent_mass_remaining_kg'],pm,'remaining'),(cor,out['parent_corner_mass_held_kg'],pm[:,None]*.25,'corner held'),(qremain,out['parent_corner_mass_remaining_kg'],pm[:,None]*.25,'corner remaining')]:
  if np.max(np.abs(a-z)/np.maximum(scale,1e-300),initial=0)>2e-12:raise RuntimeError('source ledger '+n)
 chmass=np.bincount(fc,weights=fm,minlength=len(cv))
 if np.max(np.abs(chmass-out['child_mass_kg']),initial=0)>2e-12:raise RuntimeError('child mass')
 dc,di,dv=aggregate_donor(fp,fc,fm,doff,did,dm,pm);order=np.lexsort((out['child_donor_id'],out['child_donor_child_row']))
 if not np.array_equal(out['child_donor_child_row'],out['child_donor_child_row'][order]) or not np.array_equal(out['child_donor_id'],out['child_donor_id'][order]) or not np.array_equal(dc,out['child_donor_child_row']) or not np.array_equal(di,out['child_donor_id']) or not np.allclose(dv,out['child_donor_mass_kg'],rtol=2e-10,atol=0):raise RuntimeError('source-donor transfer replay')
 donorchild=np.bincount(out['child_donor_child_row'],weights=out['child_donor_mass_kg'],minlength=len(cv))
 if not np.allclose(donorchild,chmass,rtol=1e-10,atol=0):raise RuntimeError('child donor closure')
 deficit=np.maximum(LO*cv-chmass,0);excess=np.maximum(chmass-HI*cv,0)
 if np.any(excess>1e-14) or not np.allclose(out['child_lower_deficit_kg'],deficit,rtol=1e-10,atol=1e-20):raise RuntimeError('global receiver density ledger')
 result={'schema':'numi.human.fat-held-060-061-independent-replay.v1','status':'PASS_SOURCE_ONLY_060_PROPOSAL_AND_061_MERGE_REPLAY','geometry_scope':'baseline142 Fat only; incomplete, not final geometry','proposal_rows':len(m),'selected_parent_count':len(sel),'retired_old_rows':int(ret.sum()),'preserved_old_rows':int(keep.sum()),'merged_rows':len(fm),'proposal_max_radius_m':float(d.max(initial=0)),'proposal_M4_max_relative_error':rel(corner,target),'whole_flow_max_radius_m':float(dr.max(initial=0)),'parent_held_ledger_max_relative_error':rel(held,out['parent_mass_held_kg']),'corner_ledger_max_relative_error':rel(cor,out['parent_corner_mass_held_kg']),'child_mass_max_abs_error_kg':float(np.max(np.abs(chmass-out['child_mass_kg']),initial=0)),'source_donor_child_max_relative_error':rel(donorchild,chmass),'global_source_mass_kg':float(pm.sum()),'global_held_mass_kg':float(fm.sum()),'unfunded_children_gt1e-15':int(np.count_nonzero(deficit>1e-15)),'remaining_lower_deficit_kg':float(deficit.sum()),'upper_excess_children_gt1e-15':int(np.count_nonzero(excess>1e-15)),'upper_excess_kg':float(excess.sum()),'complete_allocation':False,'physical_admission':False,'native_steps':0,'mass_transport':'NOT_ADMITTED','elapsed_seconds':time.time()-T0}
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
