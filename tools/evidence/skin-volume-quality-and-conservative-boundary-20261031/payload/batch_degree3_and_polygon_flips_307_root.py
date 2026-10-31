"""Bounded sequential fixed-boundary flips; keep compact exact-replay deltas."""
from pathlib import Path
import argparse,types,importlib.util,numpy as np,json,hashlib,time
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('volume_polygon_flips_202',H/'fixed_boundary_degree3_edge_removal_306_root.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--input-npz',type=Path);ap.add_argument('--node',type=Path);ap.add_argument('--ele',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--rounds',type=int,default=8);ap.add_argument('--seconds-per-round',type=float,default=90);ap.add_argument('--maximum-degree',type=int,default=8);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();current=a.input_npz;history=[];previous_owned=None
for i in range(a.rounds):
 d=a.output/('pass-%03d'%i)
 assert current is not None
 args=types.SimpleNamespace(input_npz=current,output=d,seconds=a.seconds_per_round,maximum_degree=a.maximum_degree,maximum_operations=100000)
 m.run(args)
 r=json.loads((d/'report.json').read_text());full=d/'fixed-boundary-volume-proposal.npz'
 with np.load(full) as z:
  x=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices'];keep=z['retained_input_cell_rows'];parents=z['replacement_parent_cell_rows'];new=z['replacement_tetrahedra'];parent_offsets=z['replacement_parent_offsets'];new_offsets=z['replacement_tet_offsets']
 if current:
  with np.load(current) as z:oldp=z['points_world_m'];oldt=z['tetrahedra'];oldm=z['material_indices']
 else:
  nd=np.loadtxt(a.node,skiprows=1);ids=nd[:,0].astype(np.int64);oldp=nd[:,1:4].astype(np.float32)
  ee=np.loadtxt(a.ele,skiprows=1);oldt=np.searchsorted(ids,ee[:,1:5].astype(np.int64));oldm=ee[:,5].astype(np.int32)
 mask=np.ones(len(oldt),bool);mask[parents]=False
 assert np.array_equal(keep,np.flatnonzero(mask)) and np.array_equal(x,oldp)
 reconstructed=np.vstack((oldt[mask],new))
 newmat=np.concatenate([np.full(new_offsets[k+1]-new_offsets[k],oldm[parents[parent_offsets[k]]],np.int32) for k in range(len(parent_offsets)-1)]) if len(parents) else np.empty(0,np.int32)
 assert np.array_equal(reconstructed,t) and np.array_equal(np.r_[oldm[mask],newmat],mat)
 delta=d/'exact-connectivity-delta.npz'
 np.savez_compressed(delta,replacement_parent_cell_rows=parents,replacement_parent_offsets=parent_offsets,replacement_tetrahedra=new,replacement_tet_offsets=new_offsets,replacement_material_indices=newmat)
 # Only retire a previous full intermediate after proving its entire connectivity
 # is represented by the authenticated source plus retained ordered deltas.
 h={'pass':i,'report_sha256':sha(d/'report.json'),'delta_sha256':sha(delta),'proposal_sha256':sha(full),'passing_operations':r['passing_operations'],'bad_cells_before':r['bad_cells_before'],'bad_cells_after':r['bad_cells_after'],'exact_delta_reconstruction':'PASS','native_steps':0,'physical_admission':False}
 if previous_owned is not None:
  assert previous_owned.parent.parent==a.output
  previous_owned.unlink()
  h['previous_owned_full_intermediate_retired_after_exact_replay_proof']=str(previous_owned)
 history.append(h);current=full;previous_owned=full
 if not r['passing_operations']:break
last=json.loads((current.parent/'report.json').read_text())
report={'status':'PASS_BOUNDED_FIXED_BOUNDARY_CONNECTIVITY_BATCH','physical_admission':False,'native_steps':0,'input_node':str(a.node) if a.node else None,'input_element':str(a.ele) if a.ele else None,'input_npz':str(a.input_npz) if a.input_npz else None,'script_sha256':sha(Path(__file__)),'operator_sha256':sha(H/'fixed_boundary_degree3_edge_removal_306_root.py'),'rounds':len(history),'passing_operations':sum(x['passing_operations'] for x in history),'bad_cells_before':history[0]['bad_cells_before'],'bad_cells_after':last['bad_cells_after'],'final_proposal':str(current),'final_proposal_sha256':sha(current),'history':history,'elapsed_s':time.monotonic()-start}
(a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='history'}))
