#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np
H=Path(sys.argv[1]); out=H/'fat-material-attribution-135/run-001';

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def need(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def rows(off,v,r):return v[int(off[r]):int(off[r+1])]
report=json.load(open(out/'report.json')); side=loadz(out/'fat-material-attribution.npz')
p135=H/'fat-positive-PLC-prefix-compaction-135-root/fat-positive-compact-PLC-prefix.npz'; p96=H/'fat-local-milp-composite-096/fat-current-milp-restored-source-composite.npz'; s96p=H/'fat-material-attribution-096/run-001/fat-material-attribution.npz'
a135=loadz(p135); src=loadz(s96p)
need(sha(p135)==report['candidate_sha256']==report['input_pins']['candidate135']['sha256'],'135 candidate pin')
need(sha(out/'fat-material-attribution.npz')==report['sidecar_sha256'],'sidecar SHA')
need(report['physical_admission'] is False and report['runtime']==0 and report['mass_mapping']=='NOT_RUN','source-only status')
need(len(side['output_cell_row'])==len(a135['tetrahedra'])==853406,'cell count')
need(np.array_equal(side['output_cell_row'],np.arange(len(a135['tetrahedra']))),'output row domain')
need(np.all(side['material_index']==16),'material row values')
need(np.array_equal(side['current_candidate_region_attribute'],a135['material_indices']),'raw candidate attributes')
source96=side['source_096_cell_row']; patchids=side['attribution_patch_id']; direct=source96>=0; derived=~direct
need(int(direct.sum())==report['source096_cells_mapped_directly'] and int(derived.sum())==report['derived_output_cells_with_patch_union'],'lineage row counts')
need(np.all(patchids[direct]==src['attribution_patch_id'][source96[direct]]),'inherited patch IDs')
need(np.all(side['source_066_cell_row'][direct]==src['source_066_cell_row'][source96[direct]]),'inherited scalar 066')
need(np.all(side['source_064_candidate_cell_row'][direct]==src['source_064_candidate_cell_row'][source96[direct]]),'inherited scalar 064')
need(np.all(side['source_tetgen_element_id'][direct]==src['source_tetgen_element_id'][source96[direct]]),'inherited tetgen IDs')
need(np.all(side['source_region_attribute_059'][direct]==src['source_region_attribute_059'][source96[direct]]),'inherited 059 attrs')
need(np.all(side['corrected_region_attribute_062'][direct]==src['corrected_region_attribute_062'][source96[direct]]),'inherited 062 attrs')
need(np.all(side['attribution_method'][direct]==src['attribution_method'][source96[direct]]),'inherited attribution method')
need(np.all(side['lineage_granularity'][direct]==src['lineage_granularity'][source96[direct]]),'inherited lineage granularity')
need(np.all(side['source_066_cell_row'][derived]==-1) and np.all(side['source_064_candidate_cell_row'][derived]==-1) and np.all(side['source_tetgen_element_id'][derived]==-1),'derived scalar sentinels')
need(np.all(side['source_region_attribute_059'][derived]==-1) and np.all(side['corrected_region_attribute_062'][derived]==16),'derived source region semantics')
need(np.all(side['attribution_method'][derived]==11) and np.all(side['lineage_granularity'][derived]==3),'derived patch-level semantics')
need(np.all(np.isin(side['source_parent_material_index'],[16])),'parent source material attributes')
need(np.all(np.isin(side['source_parent_region_attribute_062'],[16,17,18,19,20,21,22])),'parent corrected region scope')
# Exact row-by-row copied CSR for every directly inherited candidate-096 cell.
csr={
 'source_parent_tetgen_element':('source_parent_tetgen_element_offsets','source_parent_tetgen_element_ids'),
 'source_066_parent_candidate_row':('source_066_parent_candidate_row_offsets','source_066_parent_candidate_rows'),
 'source_parent_084_candidate_row':('source_parent_084_candidate_row_offsets','source_parent_084_candidate_rows'),
 'source_parent_080_candidate_row':('source_parent_080_candidate_row_offsets','source_parent_080_candidate_rows')}
for stem,(offk,valk) in csr.items():
 oo=side[offk]; bo=src[offk]; direct_rows=np.flatnonzero(direct); direct_sources=source96[direct_rows]
 need(np.array_equal(np.diff(oo)[direct_rows],np.diff(bo)[direct_sources]),f'{stem} direct lengths')
 for r,s in zip(direct_rows,direct_sources):need(np.array_equal(rows(oo,side[valk],int(r)),rows(bo,src[valk],int(s))),f'{stem} direct row {r}')
# Recompute every derived patch union from frozen096 sidecar and compare every derived output row.
patch_records={int(x['patch_id']):x for x in report['derived_patch_records']}
for pid,record in patch_records.items():
 outrows=np.flatnonzero(patchids==pid); sr=np.asarray(record['source096_cell_union_rows'],np.int64)
 need(len(outrows)==record['output_cell_count'],'patch output count')
 expected={}
 for stem,(offk,valk) in csr.items():expected[stem]=np.unique(np.concatenate([rows(src[offk],src[valk],int(r)) for r in sr]))
 tet_off=src['source_parent_tetgen_element_offsets']; ids=[]; raw=[]; fix=[]; mat=[]
 for r in sr:
  a,b=int(tet_off[r]),int(tet_off[r+1]); ids.append(src['source_parent_tetgen_element_ids'][a:b]);raw.append(src['source_parent_region_attribute_059'][a:b]);fix.append(src['source_parent_region_attribute_062'][a:b]);mat.append(src['source_parent_material_index'][a:b])
 ids=np.concatenate(ids); raw=np.concatenate(raw); fix=np.concatenate(fix); mat=np.concatenate(mat); order=np.argsort(ids,kind='stable');ids,raw,fix,mat=ids[order],raw[order],fix[order],mat[order]
 starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1]
 need(np.array_equal(np.minimum.reduceat(raw,starts),np.maximum.reduceat(raw,starts)),'patch raw parent conflicts')
 need(np.array_equal(np.minimum.reduceat(fix,starts),np.maximum.reduceat(fix,starts)),'patch corrected parent conflicts')
 need(np.array_equal(np.minimum.reduceat(mat,starts),np.maximum.reduceat(mat,starts)),'patch material parent conflicts')
 expected['source_parent_tetgen_element']=ids[starts]
 expected['source_parent_region_attribute_059']=raw[starts];expected['source_parent_region_attribute_062']=fix[starts];expected['source_parent_material_index']=mat[starts]
 for r in outrows:
  for stem,(offk,valk) in csr.items():need(np.array_equal(rows(side[offk],side[valk],int(r)),expected[stem]),f'patch {pid} {stem} row {r}')
  a,b=int(side['source_parent_tetgen_element_offsets'][r]),int(side['source_parent_tetgen_element_offsets'][r+1])
  need(np.array_equal(side['source_parent_tetgen_element_ids'][a:b],expected['source_parent_tetgen_element']),f'patch {pid} parent tet IDs row {r}')
  need(np.array_equal(side['source_parent_region_attribute_059'][a:b],expected['source_parent_region_attribute_059']),f'patch {pid} raw lineage row {r}')
  need(np.array_equal(side['source_parent_region_attribute_062'][a:b],expected['source_parent_region_attribute_062']),f'patch {pid} corrected lineage row {r}')
  need(np.array_equal(side['source_parent_material_index'][a:b],expected['source_parent_material_index']),f'patch {pid} material lineage row {r}')
  need(np.array_equal(rows(side['source_096_cell_parent_offsets'],side['source_096_cell_parent_rows'],int(r)),sr),f'patch {pid} source096 parent map row {r}')
# Every output cell has valid, nonempty, correctly aligned CSR.
for stem,(offk,valk) in csr.items():
 need(len(side[offk])==len(side['output_cell_row'])+1 and np.all(np.diff(side[offk])>0),f'{stem} CSR shape/nonempty')
need(len(side['source_parent_tetgen_element_ids'])==len(side['source_parent_region_attribute_059'])==len(side['source_parent_region_attribute_062'])==len(side['source_parent_material_index']),'aligned parent attrs')
need(len(side['source_096_cell_parent_offsets'])==len(side['output_cell_row'])+1 and np.all(np.diff(side['source_096_cell_parent_offsets'])>0),'source096 CSR shape')
receipt={'schema':'numi.human.fat-material-attribution-135-independent-verification.v1','status':'PASS_SIDECAR_LINEAGE_AND_MATERIAL_ROWS','candidate135_sha256':sha(p135),'sidecar_sha256':sha(out/'fat-material-attribution.npz'),'producer_report_sha256':sha(out/'report.json'),'checks':{'direct096_scalar_rows_exact':True,'direct096_csr_rows_exact':True,'all_patch_parent_unions_recomputed_from096':True,'all_derived_csr_and_aligned_parent_attributes_exact':True,'current_candidate_raw_region_attributes_unchanged':True,'all_material_index_rows_16':True,'source_only_no_admission':True},'direct_rows':int(direct.sum()),'derived_rows':int(derived.sum()),'patch_ids':sorted(patch_records)}
path=out/'independent-verification.json';path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,indent=2,sort_keys=True))
