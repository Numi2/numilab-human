from pathlib import Path
import json,numpy as np,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125');V=D.parents[2]
report=json.loads((D/'connected-conditioning-124/report.json').read_text());fail=json.loads(Path('/tmp/connected-cavity-geometry-127.log').read_text().split('=',1)[1])
first=report['base_cavities']-report['replaced_base_cavities'];bad={f['cavity']-first for f in fail['failures']}
assert min(bad)>=0 and 0 not in bad
ops=[o for k,o in enumerate(report['connected_patches']) if k not in bad]
used={r for o in ops for r in o['old_rows']}
z=np.load(D/'conditioning-native-mass-screened-003/fixed-boundary-volume-proposal.npz');po=z['replacement_tet_offsets'];pa=z['replacement_parent_offsets'];pr=z['replacement_parent_cell_rows'];repl=z['replacement_tetrahedra']
base=[]
for k in range(len(po)-1):
 a,b=map(int,po[k:k+2]);rows=pr[pa[a]:pa[a+1]]
 if set(map(int,rows))&used:
  assert set(map(int,rows))<=used
  continue
 base.append({'old_rows':rows.tolist(),'new_cells':repl[a:b].tolist(),'legacy_index':k})
allops=base+ops
source=V/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
with np.load(source) as src:
 nt=len(src['volume_tetrahedra'])
retired=np.array([r for o in allops for r in o['old_rows']],np.int64);assert len(np.unique(retired))==len(retired)
retained=np.flatnonzero(~np.isin(np.arange(nt),retired));new=np.array([cell for o in allops for cell in o['new_cells']],np.int64)
tetoff=np.r_[0,np.cumsum([len(o['new_cells']) for o in allops])];parentoff=np.r_[0,np.cumsum([len(o['old_rows']) for o in allops for _ in o['new_cells']])];parents=np.array([r for o in allops for _ in o['new_cells'] for r in o['old_rows']],np.int64)
output=D/'connected-conditioning-128';output.mkdir(exist_ok=False)
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
np.savez_compressed(output/'fixed-boundary-volume-proposal.npz',retained_input_cell_rows=retained,
 replacement_tetrahedra=new,replacement_tet_offsets=tetoff,replacement_parent_offsets=parentoff,replacement_parent_cell_rows=parents,source_volume_sha256=np.array(sha(source)))
result={'status':'CONSTRUCTION_PROPOSAL_NATIVE_GEOMETRY_CHECK_PENDING','source_sha256':sha(source),'proposal_sha256':sha(output/'fixed-boundary-volume-proposal.npz'),
 'points_bit_identical':True,'exact_oriented_boundary_and_volume':True,'material_donor_region_preserved':True,
 'base_cavities':len(po)-1,'replaced_base_cavities':len(po)-1-len(base),'connected_cavities':len(ops),'total_cavities':len(allops),
 'cells_before':len(z['retained_input_cell_rows'])+len(repl),'cells_after':len(retained)+len(new),
 'excluded_candidates':[{**report['connected_patches'][k],'reason':'native cavity is not a closed sphere'} for k in bad],
 'parent_report_sha256':sha(D/'connected-conditioning-124/report.json'),'connected_patches':ops,
 'remaining_gates':['independent native exact geometry','conservative source and nodal mass / frozen transport radius','full accepted physical transaction']}
(output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('connected_patches','excluded_candidates')},indent=2))

