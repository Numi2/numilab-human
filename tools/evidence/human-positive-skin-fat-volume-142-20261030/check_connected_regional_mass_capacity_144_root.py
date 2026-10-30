"""Necessary regional capacity bounds for exact current support; no flow claims."""
from pathlib import Path
import numpy as np,csv,json,hashlib,time
H=Path(__file__).resolve().parent;ROOT=H.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();base=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
 with base.open()as f:mass=np.array([float(r['mass_kg'])for r in csv.DictReader(f)])
 with Path(str(base)+'.labels.csv').open()as f:
  labels=[r for r in csv.DictReader(f)]
 mats=np.array([int(r['material_index'])for r in labels]);regions=np.array([int(r['anatomical_region_id'])for r in labels])
 datasets=[
 ('Skin',15,900.,1100.,H/'skin010-restored-cavity-batch-072-expand1/skin-restored-boundary-proposal.npz',H/'connected-region-mass-support-143-root/skin-connected-region-support.npz'),
 ('Fat',16,850.,950.,H/'fat-positive-PLC-prefix-compaction-135-root/fat-positive-compact-PLC-prefix.npz',H/'connected-region-mass-support-143-root/fat-connected-region-support.npz')]
 result=[]
 for role,mat,lo,hi,vp,sp in datasets:
  with np.load(vp)as z:p=z['points_world_m'].astype(float);t=z['tetrahedra']
  x=p[t];vol=np.einsum('ij,ij->i',x[:,1]-x[:,0],np.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]))/6
  assert np.all(vol>0)
  with np.load(sp)as z:g=z['group_tuples'];ids=z['cell_admissible_region_ids'];off=z['cell_admissible_region_offsets']
  assert len(off)==len(t)+1
  child=np.repeat(np.arange(len(t),dtype=np.int64),np.diff(off));region=ids.astype(np.int64);factor=int(max(regions.max(),region.max()))+1
  pairs=np.unique(child*factor+region);children=pairs//factor;regs=pairs%factor
  counts=np.bincount(children,minlength=len(t));assert np.all(counts>0)
  witnesses=[];rows=[]
  for r in sorted(set(regions[mats==mat].tolist())|set(regs.tolist())):
   source=float(mass[(mats==mat)&(regions==r)].sum());eligible=children[regs==r];exclusive=eligible[counts[eligible]==1]
   minimum=lo*float(vol[exclusive].sum());maximum=hi*float(vol[eligible].sum())
   tol=1e-11*max(1.,source)
   status='PASS_NECESSARY_BOUND'if minimum<=source+tol and maximum+tol>=source else 'FAIL_NECESSARY_BOUND'
   row={'region':int(r),'source_mass_kg':source,'exclusive_child_count':len(exclusive),'eligible_child_count':len(eligible),'minimum_required_mass_kg':minimum,'maximum_permitted_mass_kg':maximum,'status':status}
   if status.startswith('FAIL'):witnesses.append(row)
   rows.append(row)
  result.append({'role':role,'material':mat,'volume_path':str(vp),'volume_sha256':sha(vp),'support_path':str(sp),'support_sha256':sha(sp),'source_mass_kg':float(mass[mats==mat].sum()),'density_bounds_kg_m3':[lo,hi],'rows':rows,'failed_regions':witnesses})
 out=H/'regional-mass-capacity-necessary-bounds-144-root';out.mkdir(exist_ok=False)
 report={'status':'PASS_NECESSARY_REGIONAL_BOUNDS'if all(not r['failed_regions']for r in result)else'FAIL_NECESSARY_REGIONAL_BOUNDS','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','scope':'Necessary region-volume bounds for authenticated one-hop same-tissue source region adjacency; ignoring spatial distance. Success is not transport feasibility. Original source mass/region ledgers and current P1 fields unchanged.','results':result,'elapsed_s':time.monotonic()-start}
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'results':[{'role':r['role'],'failed_regions':r['failed_regions']}for r in result],'elapsed_s':report['elapsed_s']}),flush=True)
if __name__=='__main__':main()
