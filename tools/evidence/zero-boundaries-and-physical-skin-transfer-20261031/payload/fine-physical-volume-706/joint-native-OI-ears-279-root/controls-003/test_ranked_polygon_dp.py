#!/usr/bin/env python3
import importlib.util,json,sys
from pathlib import Path
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
A=module('adapterdpcontrol',H/'bind_joint_boundary_typed_274_root.py');J=module('jointdpcontrol',H/'audit_joint_native_apex_272_root.py');D=module('diskdpcontrol',A.B.DISK);HELP=module('helpersdpcontrol',A.B.HELPER);DP=module('exactpoly003control',H/'joint_native_exact_polygon_dp_003.py')
P=np.asarray([[0,0,0],[1e-3,0,0],[1e-3,1e-3,0],[0,1e-3,0]],np.float32);F=np.asarray([[0,1,2],[0,2,3]],np.int64);loop=[0,1,2,3];rows=np.asarray([0,1],np.int64);ext=HELP.external(loop,HELP.edgeindex(F,len(P)),rows,set())
class Pred:
 def __init__(self,collide=False):self.collide=collide
 def _records(self,lattice,tris):return [(0,0,0,i,tuple(map(int,t))) for i,t in enumerate(tris)]
 def _audit_record_pairs(self,pairs,*,same_surface):return {'count':int(self.collide)}
class Filtered:
 def __init__(self,collide=False):self.collide=collide
 def _audit_pair_prepared_first(self,index,recs,*,same_surface):return {'count':int(self.collide)}
score=lambda f:J.score(P,f)
def main():
 ok=DP.triangulate(loop,P,ext,object(),Pred(),Filtered(),[0]*4,[],100,D,score,max_seconds=2,minimum_altitude_m=1e-9,quality_floor=1e-5)
 assert len(ok[0])==2 and ok[2]['topology']['faces']==2
 try:DP.triangulate(loop,P,ext,object(),Pred(),Filtered(True),[0]*4,[],200,D,score,max_seconds=2,minimum_altitude_m=1e-9,quality_floor=1e-5)
 except ValueError as e:external=str(e)
 else:raise AssertionError('external collision control did not reject')
 try:DP.triangulate(loop,P,ext,object(),Pred(True),Filtered(),[0]*4,[],300,D,score,max_seconds=2,minimum_altitude_m=1e-9,quality_floor=1e-5)
 except ValueError as e:internal=str(e)
 else:raise AssertionError('pair collision control did not reject')
 assert 'no external-clean triangle' in external and 'no exact-clean interval-DP plan' in internal
 print(json.dumps({'status':'PASS_INTERVAL_DP_SUCCESS_EXTERNAL_COLLISION_AND_SELF_COLLISION_CONTROLS','fixture_points':len(P),'boundary_loop':loop,'success_faces':len(ok[0]),'success_triangles':ok[2]['statistics']['geometric_candidates'],'external_reject_closed':True,'self_reject_closed':True},indent=2))
if __name__=='__main__':main()
