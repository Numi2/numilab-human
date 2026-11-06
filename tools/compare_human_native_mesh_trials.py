"""Extract authoritative acceptance and active-work counters; no physical promotion."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def summarize(directory):
 d=Path(directory).resolve();p=d/'install-motion.plan.json';ex=d/'install-motion.execution.json'
 plan=json.loads(p.read_text());execution=json.loads(ex.read_text())
 trace=d/'install-motion/resting-coupled.csv'
 rows=list(csv.DictReader(trace.open())) if trace.exists() else []
 events={}
 for line in (d/'install-motion.stderr.log').read_text().splitlines():
  key,sep,value=line.partition('=')
  if sep and key in ['matter_fgmres_active_work','matter_solver_domain_norms','matter_contact_work']:
   events.setdefault(key,[]).append(json.loads(value))
 active=events.get('matter_fgmres_active_work',[])
 args=plan['stages'][0]['argv'];dt=float(args[args.index('--muscle-step-seconds')+1])
 count=int(args[args.index('--muscle-step-count')+1]);accepted=len(rows)
 body=None
 with gzip.open(d/'install-motion.stdout.log.gz','rt') as f:
  for line in f:
   if 'resting_integrated_body=completed ' in line:
    body=dict(v.split('=',1) for v in line.split('resting_integrated_body=completed ',1)[1].split() if '=' in v)
 failure=d/'install-motion/diagnostics/common-field-failure.json'
 if failure.exists():
  f=json.loads(failure.read_text());assert f['segment_start']+f['stand_completed_steps']==accepted
  assert f['respiratory_accepted_status'][0]==accepted
 assert all(v['root']<count for v in active)
 r=dict(directory=str(d),plan=ref(p),execution=execution,accepted_roots=accepted,requested_roots=count,dt_s=dt,accepted_time_s=accepted*dt,launcher_accepted_time_factor=accepted*dt/execution['elapsed_s'],full_requested_horizon_pass=execution['exit_code']==0 and accepted==count and execution['resource_stop_reason'] is None,active_work_all_attempts={key:sum(v[key] for v in active) for key in ['used_columns_sum','cycles_with_work','budget_exhausted_solves']},active_work_accepted_roots={key:sum(v[key] for v in active if v['root']<accepted) for key in ['used_columns_sum','cycles_with_work','budget_exhausted_solves']},per_root=events,body=body,coupled_trace=ref(trace) if trace.exists() else None,failure=ref(failure) if failure.exists() else None,scope='Short initialization transient, no supported-rest, sustained throughput or visual motion admission.')
 if body:
  r['body_real_time_factor']=accepted*dt/float(body['wall_s'])
 return r
def control_differences(candidate_plan,control_plan):
 # Geometry and output paths may differ. Numerical controls and loaded code may not.
 a=candidate_plan['stages'][0];b=control_plan['stages'][0]
 def canonical(stage):
  root=stage['argv'][4]
  geometry=lambda k:k.startswith('NUMI_H706_FRACTIONAL_') or k.startswith('NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_') or k in {'NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ','NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256'}
  def value(v):return '<OUTPUT>'+v[len(root):] if v.startswith(root) else v
  env={k:value(v) for k,v in stage['environment'].items() if not geometry(k)}
  argv=[value(v) for v in stage['argv']]
  return dict(argv=argv,environment=env)
 aa,bb=canonical(a),canonical(b);different=[]
 if aa['argv']!=bb['argv']:different.append('argv')
 for k in sorted(aa['environment'].keys()|bb['environment'].keys()):
  if aa['environment'].get(k)!=bb['environment'].get(k):different.append('environment:'+k)
 if candidate_plan['native_binary']['sha256']!=control_plan['native_binary']['sha256']:different.append('native_binary')
 def deps(p):return {k:v['sha256'] for k,v in p['runtime_dependencies'].items()}
 if deps(candidate_plan)!=deps(control_plan):different.append('runtime_dependencies')
 return different

def align_source_indexed_mesh_pair(a,b,raw_a,raw_b,object_name,*,report_label_differences=False):
 """Align compiler permutations through exact unique emitted source coordinates.

 This is deliberately limited to a source mesh with unchanged connectivity.
 No proximity, nearest-node, sorted triangle count or guessed identity is used.
 """
 import numpy as np
 if not np.array_equal(raw_a['volume_tetrahedra'],raw_b['volume_tetrahedra']):raise ValueError('raw source topology changed')
 if a['objects'].keys()!=b['objects'].keys():raise ValueError('native object identities changed')
 def keys(values):
  values=np.ascontiguousarray(values)
  return values.view(np.dtype((np.void,values.dtype.itemsize*values.shape[1]))).reshape(-1)
 def source_rows(z,raw):
  nodes=np.flatnonzero(z['obj']==z['objects'][object_name]);active=np.unique(raw['volume_tetrahedra'])
  p=np.asarray(raw['points_reference_world_m_f64'],np.float32)[active]
  if len(nodes)!=len(active) or not np.isfinite(p).all():raise ValueError('native/source active domain mismatch')
  sourcekeys=keys(p);order=np.argsort(sourcekeys);ordered=sourcekeys[order]
  if np.any(ordered[1:]==ordered[:-1]):raise ValueError('ambiguous source coordinates require typed mapping')
  current=np.asarray(z['rest'][nodes],np.float32)
  if not np.array_equal(current.astype(float),z['rest'][nodes]):raise ValueError('native rest is not exact emitted Float32')
  lookup=np.searchsorted(ordered,keys(current))
  if np.any(lookup>=len(ordered)) or not np.array_equal(ordered[lookup],keys(current)):raise ValueError('native rest does not replay source coordinates')
  rows=active[order[lookup]]
  if len(np.unique(rows))!=len(rows):raise ValueError('native source node map is not bijective')
  return nodes,rows
 an,ar=source_rows(a,raw_a);bn,br=source_rows(b,raw_b)
 raw_to_a=np.full(len(raw_a['points_reference_world_m_f64']),-1,np.int64);raw_to_a[ar]=an
 b_to_a=np.full(len(b['rest']),-1,np.int64);b_to_a[bn]=raw_to_a[br]
 for name in a['objects']:
  if name==object_name:continue
  aa=np.flatnonzero(a['obj']==a['objects'][name]);bb=np.flatnonzero(b['obj']==b['objects'][name])
  if len(aa)!=len(bb) or not np.array_equal(a['rest'][aa],b['rest'][bb]):raise ValueError('unrelated object rest/domain changed: '+name)
  b_to_a[bb]=aa
 if np.any(b_to_a<0) or not np.array_equal(np.sort(b_to_a),np.arange(len(a['rest']))):raise ValueError('native node mapping is not bijective')
 a_to_b=np.argsort(b_to_a);mapped=b_to_a[b['tets']]
 ak=keys(np.sort(np.asarray(a['tets'],dtype=np.int64),axis=1));bk=keys(np.sort(np.asarray(mapped,dtype=np.int64),axis=1));order=np.argsort(bk);sorted_b=bk[order]
 if np.any(sorted_b[1:]==sorted_b[:-1]):raise ValueError('duplicate native tetrahedra')
 lookup=np.searchsorted(sorted_b,ak)
 if np.any(lookup>=len(sorted_b)) or not np.array_equal(sorted_b[lookup],ak):raise ValueError('canonical source/native cell sets disagree')
 rows=order[lookup]
 mismatched=np.flatnonzero(a['treg']!=b['treg'][rows])
 if len(mismatched) and not report_label_differences:raise ValueError('canonical cell anatomical labels changed')
 aligned=dict(rest=b['rest'][a_to_b],u=b['u'][a_to_b],tets=a['tets'],treg=b['treg'][rows],native_node_rows=a_to_b,native_cell_rows=rows)
 receipt=dict(status='PASS_EXACT_SOURCE_NODE_AND_CELL_BIJECTION',native_nodes=len(a_to_b),native_cells=len(rows),permuted_nodes=int(np.count_nonzero(a_to_b!=np.arange(len(a_to_b)))),permuted_cells=int(np.count_nonzero(rows!=np.arange(len(rows)))),source_node_identity='Unique exact emitted Float32 coordinates mapped separately to each authenticated raw source node domain',source_cell_identity='Exact four canonical source node IDs; all anatomical cell labels checked',nearest_node_matching=False,physical_admission=False)
 receipt['anatomical_label_differences']=len(mismatched)
 receipt['anatomical_label_witnesses']=[dict(reference_native_cell=int(i),candidate_native_cell=int(rows[i]),reference_region=int(a['treg'][i]),candidate_region=int(b['treg'][rows[i]])) for i in mismatched[:16]]
 if len(mismatched):receipt['status']='EXACT_SOURCE_BIJECTION_WITH_ANATOMICAL_LABEL_DIFFERENCES_NOT_ADMITTED'
 return aligned,receipt


def strain_pair_witnesses(a,b,limit=12):
 """Localize endpoint differences on matched topology; never relax a gate."""
 import numpy as np
 if not np.array_equal(a['tets'],b['tets']):raise ValueError('strain witnesses require matched topology')
 if a['rest'].shape!=b['rest'].shape or a['u'].shape!=b['u'].shape:raise ValueError('strain witness point domain mismatch')
 if limit<1:raise ValueError('witness limit must be positive')
 for z in [a,b]:
  if not np.isfinite(z['rest']).all() or not np.isfinite(z['u']).all():raise ValueError('nonfinite endpoint field')
 changed=np.any(a['rest']!=b['rest'],axis=1);best=[];max_unchanged=0.;peak_a=0.;peak_b=0.
 for start in range(0,len(a['tets']),8192):
  cells=a['tets'][start:start+8192];norms=[]
  for z in [a,b]:
   rest=z['rest'][cells];u=z['u'][cells];dm=np.swapaxes(rest[:,1:]-rest[:,:1],1,2);du=np.swapaxes(u[:,1:]-u[:,:1],1,2)
   if np.any(np.linalg.det(dm)==0):raise ValueError('degenerate rest cell')
   f=np.eye(3)+du@np.linalg.inv(dm)
   if np.any(np.linalg.det(f)<=0):raise ValueError('nonpositive deformed cell')
   norms.append(np.linalg.norm(.5*(np.swapaxes(f,1,2)@f-np.eye(3)),axis=(1,2)))
  delta=np.abs(norms[1]-norms[0]);unaltered=~changed[cells].any(axis=1)
  if unaltered.any():max_unchanged=max(max_unchanged,float(delta[unaltered].max()))
  peak_a=max(peak_a,float(norms[0].max()));peak_b=max(peak_b,float(norms[1].max()))
  rows=np.lexsort((np.arange(len(cells)),-delta))[:limit]
  best.extend((float(delta[i]),start+int(i),float(norms[0][i]),float(norms[1][i])) for i in rows)
  best=sorted(best,key=lambda row:(-row[0],row[1]))[:limit]
 witnesses=[]
 for difference,row,ea,eb in best:
  cell=a['tets'][row];details=[]
  for z in [a,b]:
   x=z['rest'][cell];dm=(x[1:]-x[:1]).T
   details.append(dict(volume_m3=abs(float(np.linalg.det(dm)))/6,rest_matrix_condition=float(np.linalg.cond(dm)),maximum_displacement_m=float(np.linalg.norm(z['u'][cell],axis=1).max()),rest_m=x.tolist(),displacement_m=z['u'][cell].tolist()))
  witnesses.append(dict(tetrahedron=row,candidate_native_tetrahedron=int(b['native_cell_rows'][row]) if 'native_cell_rows' in b else row,nodes=cell.tolist(),candidate_native_nodes=b['native_node_rows'][cell].tolist() if 'native_node_rows' in b else cell.tolist(),region=int(a['treg'][row]),candidate_region=int(b['treg'][row]),rest_coordinates_unchanged=not bool(changed[cell].any()),reference_green_strain_norm=ea,candidate_green_strain_norm=eb,absolute_green_strain_norm_difference=difference,reference=details[0],candidate=details[1]))
 return dict(scope='Matched-topology endpoint localization only; not a force, material-energy, motion or clinical certificate',changed_rest_nodes=int(changed.sum()),maximum_accepted_displacement_difference_m=float(np.linalg.norm(b['u']-a['u'],axis=1).max()),maximum_absolute_strain_difference_on_unchanged_cells=max_unchanged,reference_peak_green_strain_norm=peak_a,candidate_peak_green_strain_norm=peak_b,witnesses=witnesses,physical_admission=False)


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--candidate-dir',required=True,type=Path)
 ap.add_argument('--control-dir',required=True,type=Path)
 ap.add_argument('--output',required=True,type=Path)
 args=ap.parse_args()
 assert not args.output.exists(),'retain previous analysis; choose an unused output'
 candidate=summarize(args.candidate_dir);control=summarize(args.control_dir)
 cp=json.loads(Path(candidate['plan']['path']).read_text());bp=json.loads(Path(control['plan']['path']).read_text())
 differences=control_differences(cp,bp)
 if differences:raise ValueError('Unmatched native controls: '+', '.join(differences))
 r=dict(candidate=candidate,matched_solver_control=control,
        accepted_roots_change=candidate['accepted_roots']-control['accepted_roots'],
        candidate_exceeds_control_accepted_horizon=candidate['accepted_roots']>control['accepted_roots'],
        force_strain_energy_fidelity_admitted=False,physical_admission=False,
        analysis_owner=ref(__file__),control_compatibility=dict(matched=True,geometry_and_output_paths_may_differ=True))
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in candidate.items() if k not in ['per_root','execution']},indent=2))
if __name__=='__main__':main()
