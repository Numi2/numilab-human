from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"register_current_combined_volume_nodes_217_1036.py"
spec=importlib.util.spec_from_file_location("registrar1036_parent",SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DummyOwner: pass

def record(key,tet,distance):
 return {"tuple":key,"tet":tet,"beta":[0.25,0.25,0.25,0.25],"registered":[0.,0.,0.],
         "residual":[0.,0.,0.],"distance":distance,"ties":0}

def run():
 original=m.nearest_for_tuple
 try:
  data={}
  def nearest(point,key,lookup,owner,maxdist,tol): return data[key]
  m.nearest_for_tuple=nearest
  # I preserves original material when both incident materials are bounded, even if the other is nearer.
  data={(15,7,3):record((15,7,3),11,0.010),(16,8,3):record((16,8,3),12,0.001)}
  got,why=m.select_incident_parent_registration([0,0,0],"I",(15,9,3),set(data),{},DummyOwner(),0.02,1e-10,4,9)
  assert why is None and got["tuple"]==(15,7,3) and got["tet"]==11
  # If original-material options exceed the bound, I may use another actually incident material.
  data={(15,7,3):record((15,7,3),11,0.02001),(16,8,3):record((16,8,3),12,0.019)}
  got,why=m.select_incident_parent_registration([0,0,0],"I",(15,9,3),set(data),{},DummyOwner(),0.02,1e-10,4,9)
  assert why is None and got["tuple"]==(16,8,3)
  # O/D cannot select a parent in the opposite role material.
  data={(16,7,3):record((16,7,3),11,0.001),(15,8,3):record((15,8,3),12,0.002)}
  got,why=m.select_incident_parent_registration([0,0,0],"O",(15,1,1),set(data),{},DummyOwner(),0.02,1e-10,4,9)
  assert why is None and got["tuple"]==(15,8,3)
  got,why=m.select_incident_parent_registration([0,0,0],"D",(16,1,1),set(data),{},DummyOwner(),0.02,1e-10,4,9)
  assert why is None and got["tuple"]==(16,7,3)
  # Equal distance selection is stable by tuple then parent tet row.
  data={(15,6,3):record((15,6,3),17,0.003),(15,5,3):record((15,5,3),18,0.003)}
  got,why=m.select_incident_parent_registration([0,0,0],"O",(15,1,1),set(data),{},DummyOwner(),0.02,1e-10,4,9)
  assert why is None and got["tuple"]==(15,5,3)
  # The full failure is returned as a witness so build_rows can collect it with later nodes.
  data={(15,6,3):record((15,6,3),17,0.030)}
  got,why=m.select_incident_parent_registration([0,0,0],"O",(15,1,1),set(data),{},DummyOwner(),0.02,1e-10,4,9)
  assert got is None and why["failure"].startswith("no incident semantic parent")
  e=m.ParentSemanticResolutionBatchError([why,{"role":"interior"}])
  assert e.counts=={"O":1,"interior":1} and len(e.witnesses)==2
 finally:
  m.nearest_for_tuple=original
 print("PASS 217 incident-parent rebinding controls: role materials, I preference/fallback, deterministic ties, bounded-failure witness aggregation")

if __name__=="__main__": run()
