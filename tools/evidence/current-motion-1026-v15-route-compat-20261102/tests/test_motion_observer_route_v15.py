import copy, importlib.util, unittest
from pathlib import Path
p=Path(__file__).with_name("verify_current_motion_geometry_1026_v15.py")
spec=importlib.util.spec_from_file_location("motion_v15",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Contract(unittest.TestCase):
 def direct(self):
  d={"direct_O_rows":12,"direct_O_volume_nodes":100}
  obs={"status":"PASS_BOUNDED_DIRECT_O_TRACES_AND_MRV","schema":"numi.human.current-native-direct-o-measurement-observer-receipt.v1","native_runtime_executed":True,"gpu_executed":True,"direct_O":d}
  r={"status":"PASS_DIRECT_O_INSTALL_AND_ACCEPTED_STATE_MOTION","schema":"numi.human.current-native-direct-o-motion-receipt.v1","requested_accepted_steps":2,"observed_accepted_steps":2,"registered_volume_install_pass":True,"legacy_detail_bridge_enabled":False,"native_runtime_executed":True,"gpu_executed":True,"native_exit_code":0,"direct_O":copy.deepcopy(d)}
  return obs,{"status":r["status"]},r
 def check(self,o,e,r):
  m.validate_motion_observer(o);m.validate_motion_route(o,e,r,2,{"O_rows":12,"volume_nodes":100})
 def test_direct_pass(self): self.check(*self.direct())
 def test_legacy_pass(self):
  o={"status":"PASS_BOUNDED_ACCEPTED_STATE_TRACE_AND_MRV_CHECKS","native_runtime_executed":True,"gpu_executed":True};r={"status":"PASS_INSTALL_AND_ACCEPTED_STATE_MOTION","requested_accepted_steps":2,"observed_accepted_steps":2,"registered_volume_install_pass":True};self.check(o,{"status":r["status"]},r)
 def test_reject_bad_contracts(self):
  changes=[(0,"status","PASS"),(0,"gpu_executed",False),(1,"status","PASS_INSTALL_AND_ACCEPTED_STATE_MOTION"),(2,"status","PASS_INSTALL_AND_ACCEPTED_STATE_MOTION"),(2,"schema","wrong"),(0,"schema","wrong"),(2,"legacy_detail_bridge_enabled",True),(2,"observed_accepted_steps",1),(2,"registered_volume_install_pass",False),(2,"native_exit_code",1),(2,"native_runtime_executed",False),(2,"direct_O",{"direct_O_rows":12,"direct_O_volume_nodes":13})]
  for idx,key,value in changes:
   with self.subTest(idx=idx,key=key,value=value):
    x=list(self.direct());x[idx][key]=value
    with self.assertRaises(ValueError):self.check(*x)
 def test_reject_bad_direct_counts(self):
  for count in [0,-1,True,1.5]:
   o,e,r=self.direct();o["direct_O"]={"direct_O_rows":count,"direct_O_volume_nodes":count};r["direct_O"]=copy.deepcopy(o["direct_O"])
   with self.subTest(count=count),self.assertRaises(ValueError):self.check(o,e,r)
if __name__=="__main__": unittest.main()
