from pathlib import Path
import numpy as np,csv,json,hashlib,importlib.util,time
A=Path(__file__).resolve().parent.parent;P=A.parent.parent
OUT=A/'full-OOO-native-bank32-006'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 base=P/'reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz'
 nr=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'
 ma=P/'conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets/volume/combined-node-map.csv'
 mask=A/'run-007-inner-mask-coupled-buffer/mask-inputs.npz'
 owner=A/'authoring-source-chain/build_minimal_coupled_bone32_buffer_003.py'
 z=np.load(base,allow_pickle=False);n=np.load(nr,allow_pickle=False);m=np.load(mask,allow_pickle=False)
 role={int(r['joint_node_id'])-1:r['role'] for r in csv.DictReader(ma.open())}
 ff=m['source_skin23_faces_global_node_rows'];orows=np.array([i for i,t in enumerate(ff) if all(role[int(v)]=='O' for v in t)])
 p=z['points_current_world_m_f32'];F=ff[orows]
 i=np.flatnonzero((n['bank_body']==23)&(n['bank_bone']==32));assert len(i)==1;i=int(i[0])
 off=n['bank_face_offsets'];BF=n['faces'][off[i]:off[i+1]];bp=n['points_world_f32']
 spec=importlib.util.spec_from_file_location('owner049',owner);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);ex=mod.exact_setup()
 ar,_=ex.float32_surface_records(p,F);br,_=ex.float32_surface_records(bp,BF)
 assert len(ar)==len(F) and len(br)==len(BF)
 start=time.time();v=ex._audit_pair(ar,br,same_surface=False);pairs=np.asarray(v.get('triangle_pairs',[]),np.int64).reshape(-1,2)
 source=orows[pairs[:,0]] if len(pairs) else np.array([],int)
 OUT.mkdir()
 np.savez_compressed(OUT/'witnesses.npz',pair_local_rows=pairs,outer_face_region_rows=orows,region_source_faces=ff,native_bank_face_global_rows=np.arange(off[i],off[i+1]),native_bank_faces=BF)
 r={'status':'EXACT_F32_WHOLE_REGIONAL_OUTER_VS_NATIVE_BANK_WITNESSES','outer_triangles':len(F),'native_bank_triangles':len(BF),'dropped_or_degenerate_triangles':0,'triangle_pairs':len(pairs),'offending_skin_region_source_faces':sorted(set(map(int,source))),'offending_global_O_nodes':sorted(set(map(int,ff[source].ravel()))) if len(source) else [],'seconds':time.time()-start,'runtime_created':0,'accepted_steps':0,'physical_admission':False,'inputs':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [base,nr,ma,mask,owner,Path(__file__)]}}
 (OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
