from __future__ import annotations
import csv,json,importlib.util,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"register_current_combined_volume_nodes_217_1034.py"
spec=importlib.util.spec_from_file_location("registrar1034",SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def pose_row(body,phase,origin,quat,matrix):
 return {"body_index":body,"position_world_m":origin,"orientation_world_xyzw":quat,
         "local_to_world_row_major_f32":matrix}
def frame_row(binding,donor,poses):
 row={"binding":str(binding),"donor":str(donor)}
 for phase in ("current","rest"):
  p=poses[phase]
  origin=np.asarray(p["position_world_m"],dtype=np.float32).astype(np.float64)
  matrix=np.asarray(p["local_to_world_row_major_f32"],dtype=np.float32).astype(np.float64).reshape(3,3)
  for axis,key in enumerate("xyz"): row[f"{phase}_{key}"]=m.fs(origin[axis])
  for i in range(3):
   for j in range(3): row[f"{phase}_R{i}{j}"]=m.fs(matrix[i,j])
 return row
def fails(fn,phrase):
 try: fn()
 except ValueError as e:
  assert phrase in str(e),str(e)
 else: raise AssertionError("expected ValueError containing "+phrase)
def main():
 q=np.array([0.1,0.2,-0.3,0.9],dtype=np.float64);q/=np.linalg.norm(q)
 rows=[]
 for body in range(157):
  phase_data={}
  for phase,scale in (("current",1.0),("rest",0.75)):
   quat=q.copy() if body==1 else np.array([0.,0.,0.,1.])
   if phase=="rest" and body==1: quat=np.array([-0.2,0.1,0.2,0.95]);quat/=np.linalg.norm(quat)
   matrix=Rotation.from_quat(quat).as_matrix().astype(np.float32).astype(np.float64).reshape(-1)
   origin=[0.,0.,0.] if body!=1 else [0.123456789*scale,-0.234567891,1.00000003]
   phase_data[phase]={"position_world_m":origin,"orientation_world_xyzw":quat.tolist(),
                      "local_to_world_row_major_f32":matrix.tolist()}
  rows.append({"body_index":body,**phase_data})
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);pose_path=root/"poses.json";frame_path=root/"frames.csv"
  pose_path.write_text(json.dumps({"all_bodies":rows}))
  with frame_path.open("w",newline="",encoding="utf-8") as f:
   w=csv.DictWriter(f,fieldnames=m.FRAME_HEADER);w.writeheader();w.writerow(frame_row(0,1,rows[1]))
  poses,bindings=m.read_poses(pose_path,frame_path)
  assert bindings=={1:0}
  for phase in ("current","rest"):
   assert np.array_equal(poses[1][phase+"_origin"],np.asarray(rows[1][phase]["position_world_m"],dtype=np.float32).astype(np.float64))
   assert np.array_equal(poses[1][phase+"_R"],np.asarray(rows[1][phase]["local_to_world_row_major_f32"],dtype=np.float64).reshape(3,3))
  missing_rows=json.loads(pose_path.read_text())
  missing_rows["all_bodies"][1]["current"].pop("local_to_world_row_major_f32")
  missing_path=root/"missing-native-matrix.json"
  missing_path.write_text(json.dumps(missing_rows))
  fails(lambda:m.read_poses(missing_path,frame_path),"lacks authoritative native Float32 local_to_world matrix")
  with frame_path.open("r",newline="",encoding="utf-8") as f: d=list(csv.DictReader(f))
  d[0]["current_R00"]=m.fs(float(d[0]["current_R00"])+1e-4)
  bad=root/"wrong.csv"
  with bad.open("w",newline="",encoding="utf-8") as f:
   w=csv.DictWriter(f,fieldnames=m.FRAME_HEADER);w.writeheader();w.writerows(d)
  fails(lambda:m.read_poses(pose_path,bad),"differs from exact all_bodies current pose")
 print("PASS 1034 native F32 frame exactness, missing-matrix rejection, and wrong-matrix rejection")
if __name__=="__main__": main()
