#!/Users/numi/numi-human-prep-venv-20261005/bin/python
"""Independent direct-array replay for corrected run055 source-only proposal."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'; BASE=S/'fat-held-exchange-031'; SRC=BASE/'run-050-radius-separation-continuation-049-cuts'; RUN=BASE/'run-055-corrected-parent-mass-row-map/attempt-002'; OUT=RUN/'independent-verification'
FULL=SRC/'full-tet-input.npz'; PARENTCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'; PROPOSAL=RUN/'proposal.npz'; PRODUCER=RUN/'report.json'; DRIVER=BASE/'run-055-corrected-parent-mass-row-map/fat_recover_corrected_055_attempt002.py'; R=.019353

def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 pins={str(p):sha(p) for p in (FULL,PARENTCSV,PROPOSAL,PRODUCER,DRIVER)}
 producer=json.loads(PRODUCER.read_text())
 if producer.get('status')!='PASS_BASELINE142_ONLY_FINITE_SITE_SOURCE_ONLY_CANDIDATE' or producer.get('physical_admission') is not False:raise RuntimeError('producer status is not a source-only finite-site pass')
 if sha(PROPOSAL)!=producer.get('proposal_sha256'):raise RuntimeError('proposal hash does not match producer')
 rows=list(csv.DictReader(open(PARENTCSV,newline=''))); pMass=np.asarray([float(x['mass_kg']) for x in rows],dtype=np.float64)
 if [int(x['parent_row']) for x in rows]!=list(range(len(rows))):raise RuntimeError('source role parent CSV row order is not identity')
 with np.load(FULL,allow_pickle=False) as z:
  parents=z['selected_parent_role_rows'].astype(np.int32,copy=True); globalRows=z['selected_parent_global_source330_rows'].astype(np.int32,copy=True)
  pairParent=z['pair_parent_role_rows'].astype(np.int32,copy=True); pairChild=z['pair_child_rows'].astype(np.int32,copy=True)
  V0=z['pair_parent_source_vertices_m'].astype(np.float64,copy=True); E=z['pair_parent_source_edges_m'].astype(np.float64,copy=True); C=z['pair_child_centroids_m'].astype(np.float64,copy=True)
  recipients=z['recipient_child_rows'].astype(np.int32,copy=True); fixed=z['fixed_mass_kg'].astype(np.float64,copy=True); childV=z['child_volume_m3'].astype(np.float64,copy=True); low=z['full_low_kg'].astype(np.float64,copy=True); high=z['full_high_kg'].astype(np.float64,copy=True)
 with np.load(PROPOSAL,allow_pickle=False) as z:
  siteRows=z['site_rows'].astype(np.int32,copy=True); pair=z['pair_rows'].astype(np.int32,copy=True); parent=z['parent_role_rows'].astype(np.int32,copy=True); globalSource=z['global_source330_tet_rows'].astype(np.int32,copy=True); child=z['child_rows'].astype(np.int32,copy=True)
  mass=z['mass_kg'].astype(np.float64,copy=True); b=z['barycentric'].astype(np.float64,copy=True); xyz=z['source_point_world_m'].astype(np.float64,copy=True); cent=z['child_centroid_world_m'].astype(np.float64,copy=True); dist=z['distance_m'].astype(np.float64,copy=True)
  recRows=z['recipient_child_rows'].astype(np.int32,copy=True); storedFinal=z['recipient_final_mass_kg'].astype(np.float64,copy=True); storedDensity=z['recipient_density_kg_m3'].astype(np.float64,copy=True); pRows=z['parent_rows'].astype(np.int32,copy=True); storedCorner=z['parent_corner_mass_total_kg'].astype(np.float64,copy=True)
 n=len(mass); assert n==len(siteRows)==len(pair)==len(parent)==len(child)==len(b)==len(xyz)==len(dist)
 if not np.array_equal(recRows,recipients) or not np.array_equal(pRows,parents):raise RuntimeError('proposal full parent/recipient row domains differ')
 if np.any(siteRows<0) or np.any(pair<0) or np.any(pair>=len(pairParent)):raise RuntimeError('proposal site/pair row outside map')
 if not np.array_equal(parent,pairParent[pair]) or not np.array_equal(child,pairChild[pair]):raise RuntimeError('proposal parent/child rows do not match pair map')
 parentLocal=np.searchsorted(parents,parent); recLocal=np.searchsorted(recipients,child)
 if not np.array_equal(parents[parentLocal],parent) or not np.array_equal(recipients[recLocal],child):raise RuntimeError('proposal parent/recipient local map mismatch')
 if not np.array_equal(globalSource,globalRows[parentLocal]):raise RuntimeError('proposal source330 global rows mismatch selected role map')
 if np.min(mass,initial=0.)<0. or not np.all(np.isfinite(mass)):raise RuntimeError('invalid flow mass')
 if np.min(b,initial=0.)<0. or np.max(np.abs(b.sum(axis=1)-1.),initial=0.)>2e-13:raise RuntimeError('barycentric simplex failure')
 calcXYZ=V0[pair]+np.einsum('ni,nij->nj',b[:,1:],E[pair]); calcCent=C[pair]; calcDist=np.linalg.norm(calcXYZ-calcCent,axis=1)
 coordScale=np.maximum(np.abs(calcXYZ).sum(axis=1),1e-300); xyzTol=8*np.finfo(np.float64).eps*coordScale
 xyzErr=np.max(np.abs(calcXYZ-xyz),axis=1)
 dTol=8*np.finfo(np.float64).eps*np.maximum(np.maximum(calcDist,dist),R)
 if np.any(xyzErr>xyzTol) or not np.array_equal(calcCent,cent) or np.any(np.abs(calcDist-dist)>dTol):raise RuntimeError('point/centroid/distance replay mismatch')
 if np.any(calcDist>R):raise RuntimeError('hard radius violation')
 # Recompute actual mass and exact first moments by global role parent row.
 corner=np.zeros((len(parents),4),dtype=np.float64)
 for k in range(4):np.add.at(corner[:,k],parentLocal,mass*b[:,k])
 target=np.repeat((pMass[parents]*.25)[:,None],4,axis=1)
 parentRel=float(np.max(np.abs(corner-target)/np.maximum(target,1e-300),initial=0.))
 sourceTotal=corner.sum(axis=1); sourceRel=float(np.max(np.abs(sourceTotal-pMass[parents])/np.maximum(pMass[parents],1e-300),initial=0.))
 if parentRel>1e-9 or sourceRel>1e-9:raise RuntimeError('source parent mass/four-corner moment replay failed')
 delivered=np.bincount(recLocal,weights=mass,minlength=len(recipients)); final=fixed+delivered; density=final/childV
 tol=1e-10*np.maximum(high,1e-300)
 lowDef=np.maximum(low-final,0.); over=np.maximum(final-high,0.)
 if np.any(final<low-tol) or np.any(final>high+tol):raise RuntimeError('receiver density/mass bounds fail')
 if not np.allclose(final,storedFinal,rtol=1e-13,atol=0.) or not np.allclose(density,storedDensity,rtol=1e-13,atol=0.):raise RuntimeError('producer receiver aggregation differs from independent replay')
 if not np.allclose(corner,storedCorner,rtol=1e-13,atol=0.):raise RuntimeError('producer corner aggregation differs from independent replay')
 report={'schema':'numi.human.fat-corrected-finite-site-recovery-independent-replay.v1','status':'PASS_SOURCE_ONLY_CORRECTED_FINITE_SITE_REPLAY','scope':'268 selected Fat source parents / 2,397 selected receiver children on frozen baseline142 geometry; not full Fat allocation or current final geometry','physical_admission':False,'mass_transport':'NOT_ADMITTED','native_steps':0,'input_pins':pins,'producer_run055_report_sha256':sha(PRODUCER),'positive_site_count':n,'selected_parent_count':len(parents),'recipient_count':len(recipients),'source_mass_kg':float(pMass[parents].sum()),'flow_mass_kg':float(mass.sum()),'max_parent_total_relative_error':sourceRel,'max_parent_corner_M_over_4_relative_error':parentRel,'max_point_coordinate_abs_error_m':float(xyzErr.max(initial=0.)),'max_point_coordinate_tolerance_m':float(xyzTol.max(initial=0.)),'max_distance_abs_error_m':float(np.abs(calcDist-dist).max(initial=0.)),'max_hard_radius_m':float(calcDist.max(initial=0.)),'hard_radius_m':R,'receiver_max_low_deficit_kg':float(lowDef.max(initial=0.)),'receiver_max_overcap_kg':float(over.max(initial=0.)),'receiver_max_low_relative_deficit':float(np.max(lowDef/np.maximum(high,1e-300),initial=0.)),'receiver_max_overcap_relative':float(np.max(over/np.maximum(high,1e-300),initial=0.)),'density_min_kg_m3':float(density.min(initial=np.inf)),'density_max_kg_m3':float(density.max(initial=0.)),'producer_stale_local_mass_matrix_superseded':True}
 reportPath=OUT/'report.json';reportPath.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
