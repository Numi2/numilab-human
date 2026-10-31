#!/usr/bin/env python3
from pathlib import Path
import importlib.util,json,sys,numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
A=mod('adapter280fixture',H/'bind_joint_boundary_typed_274_root.py');J=mod('joint272fixture',H/'audit_joint_native_apex_272_root.py');HELP=mod('helper280fixture',A.B.HELPER);DISK=mod('disk280fixture',A.B.DISK);DP=mod('dp004fixture',H/'joint_native_exact_polygon_dp_005.py')
PTH=H/'joint-native-OI-ears-279-root/O-batch002-run001/typed-280-002/candidate-O-typed-current.npz'
ROWS=np.asarray([44380,44384,44385,44387,44390,44391,44393,44394,44395,44396,44397,44398,44399,44400,44401,44404,44405,44406,44407,44408,44409,44410,44411,44412,44415,44416,44418,151213,152893,152894,152895,152896,152897,152898,153133,153134,153142],np.int64)
class Owner:
 def _records(self,lattice,faces):return [(0,0,0,i,tuple(map(int,t))) for i,t in enumerate(faces)]
 def _audit_record_pairs(self,pairs,*,same_surface):return {'count':0}
class Filtered:
 def __init__(self):self._owner=Owner();self.excluded_face_ids=set()
 def _audit_pair_prepared_first(self,index,recs,*,same_surface):return {'count':0}
 def _audit_record_pairs(self,pairs,*,same_surface):return {'count':0}
def main():
 src=A.A.load(PTH);P=np.asarray(src['points_world_m_f32'],np.float32);F=np.asarray(src['faces'],np.int64)
 loop=HELP.patchloop(F,ROWS);assert loop is not None and len(loop)==15
 edgeidx=HELP.edgeindex(F,len(P));external=HELP.external(loop,edgeidx,ROWS,set())
 filtered=Filtered();filtered.excluded_face_ids=set(map(int,ROWS));proxy=HELP.Proxy(filtered,200000,len(P))
 owner=A.B.module('owner042fixture',A.B.OWNER).load_exact_owner()
 lattice=tuple(owner.float32_point_lattice_key(v) for v in P)
 nf,recs,detail=DP.triangulate(loop,P,external,object(),proxy,filtered,lattice,[],7000000,DISK,lambda fs:J.score(P,fs),max_seconds=10,max_plans=50000,minimum_altitude_m=20e-9,quality_floor=1e-5)
 assert nf.shape==(len(loop)-2,3) and int(nf.max())<len(P)
 assert all(all(200000<=int(v)<200000+len(P) for v in r[4]) for r in recs)
 topo=DISK.validate_disk(nf,loop,external);assert topo['faces']==len(loop)-2
 keep=np.asarray([i for i in range(len(F)) if i not in set(map(int,ROWS))],np.int64);G=np.concatenate((F[keep],nf),axis=0)
 b0,t0=A.B.topology(F);b1,t1=A.B.topology(G);assert np.array_equal(b0,b1) and t1['nonmanifold_edges']==0 and t1['same_direction_edges']==0
 union=A.B.union_tuples(src,ROWS);ancestry=A.B.ancestry_union(src,ROWS);assert union and ancestry
 uoff,ud=A.A.csr([union for _ in nf],7);aoff,ad=A.A.csr([ancestry for _ in nf],3)
 assert len(uoff)==len(nf)+1 and len(aoff)==len(nf)+1
 for i in range(len(nf)):
  lo,hi=map(int,uoff[i:i+2]);assert [tuple(map(int,x)) for x in ud[lo:hi]]==union
  lo,hi=map(int,aoff[i:i+2]);assert [list(map(int,x)) for x in ad[lo:hi]]==ancestry
 print(json.dumps({'status':'PASS_ACTUAL_37_FACE_PATCH_DP_LOCAL_OUTPUT_TYPED_UNION_FIXTURE','parent_candidate_sha256':A.sha(PTH),'patch_face_rows':len(ROWS),'boundary_loop_vertices':len(loop),'output_faces':len(nf),'output_local_vertex_max':int(nf.max()),'proxy_vertex_offset':200000,'typed_union_tuples':len(union),'source_face_ancestry_tuples':len(ancestry),'full_fixture_boundary_edges_unchanged':True,'nonmanifold_edges':t1['nonmanifold_edges'],'same_direction_edges':t1['same_direction_edges'],'dp_statistics':detail['statistics']},indent=2))
if __name__=='__main__':main()
