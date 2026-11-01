from __future__ import annotations
import csv
import importlib.util
import tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"register_current_combined_volume_nodes_217_1036.py"
spec=importlib.util.spec_from_file_location("registrar1036",SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

NODE_FIELDS=[
 "vertex","donor","wx","wy","wz","lx","ly","lz",
 "m00","m01","m02","m10","m11","m12","m20","m21","m22",
 "source_domain","frame_owner_method","source_mapping_kind","source_vertex_row","source_face_row",
 "source_bary0","source_bary1","source_bary2","source_residual_x_m","source_residual_y_m","source_residual_z_m"
]
IDENTITY={"m00":"1","m01":"0","m02":"0","m10":"0","m11":"1","m12":"0","m20":"0","m21":"0","m22":"1"}
def make_node(role,i,active):
 r={k:"" for k in NODE_FIELDS}
 r.update({"vertex":str(i),"wx":str(float(i)),"wy":"0","wz":"0",
           "source_domain":m.CURRENT_TYPED_SOURCE_DOMAIN,"source_mapping_kind":"vertex",
           "source_vertex_row":str(i),"source_face_row":"-1",
           "source_bary0":"0","source_bary1":"0","source_bary2":"0",
           "source_residual_x_m":"0","source_residual_y_m":"0","source_residual_z_m":"0"})
 if active:
  r.update({"donor":"0","lx":str(float(i)),"ly":"0","lz":"0",
            "frame_owner_method":"exact_current_role_vertex_identity"})
  r.update(IDENTITY)
 return r
def write_nodes(directory, inactive_donor=False):
 for role in ("O","I","D"):
  path=directory/f"registered-{role}.csv"
  with path.open("w",newline="",encoding="utf-8") as f:
   w=csv.DictWriter(f,fieldnames=NODE_FIELDS);w.writeheader()
   w.writerow(make_node(role,0,True))
   bad=make_node(role,1,False)
   if inactive_donor:
    bad["donor"]="0"
   w.writerow(bad)
   w.writerow(make_node(role,2,True))
def write_regs(directory, rows):
 with (directory/"node-parent-registration.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=m.REG_HEADER);w.writeheader();w.writerows(rows)
def reg(role,row,method="exact_current_role_vertex_identity"):
 return {"layer":role,"surface_vertex_row":str(row),
   "parent_registration_kind":"source_parent_tet_reference_registration","parent_tet_row":"0",
   "parent_bary0":"1","parent_bary1":"0","parent_bary2":"0","parent_bary3":"0",
   "parent_residual_x_m":"0","parent_residual_y_m":"0","parent_residual_z_m":"0",
   "parent_registration_distance_m":"0","parent_registration_tie_count":"0",
   "frame_owner_method":method,"source_domain":m.CURRENT_TYPED_SOURCE_DOMAIN,
   "source_vertex_row":str(row),"source_face_row":"-1","source_bary0":"0","source_bary1":"0",
   "source_bary2":"0","source_residual_x_m":"0","source_residual_y_m":"0","source_residual_z_m":"0"}
def fails(fn, phrase):
 try: fn()
 except ValueError as e:
  assert phrase in str(e),str(e)
 else: raise AssertionError("expected ValueError containing "+phrase)

def main():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);active={r:{0,2} for r in ("O","I","D")}
  write_nodes(p)
  surfaces=m.read_surfaces(p,current_typed=True,active_rows_by_role=active)
  assert len(surfaces["O"])==3 and surfaces["O"][2]["source_vertex"]==2
  assert surfaces["O"][2]["physical_active"] is True
  assert surfaces["O"][1]["physical_active"] is False
  assert surfaces["O"][1]["donor"] is None and surfaces["O"][1]["matrix"] is None
  regs=[reg(r,i) for r in ("O","I","D") for i in (0,2)]
  write_regs(p,regs)
  parsed=m.read_parent_registrations(p,surfaces,{"tets":np.zeros((1,4),dtype=np.int64)},0.02,
                                     current_typed=True,active_rows_by_role=active)
  assert set(parsed)=={(r,i) for r in ("O","I","D") for i in (0,2)}
  mapped={r:{0,2} for r in ("O","I","D")}
  assert m.validate_active_surface_coverage(mapped,surfaces,active)=={"O":2,"I":2,"D":2}
  fails(lambda:m.validate_active_surface_coverage({"O":{0},"I":{0,2},"D":{0,2}},surfaces,active),
        "O map does not exactly cover")
  write_regs(p,[reg(r,i) for r in ("O","I","D") for i in (0,)])
  fails(lambda:m.read_parent_registrations(p,surfaces,{"tets":np.zeros((1,4),dtype=np.int64)},0.02,
       current_typed=True,active_rows_by_role=active),"exactly cover active O rows")
  write_regs(p,regs+[reg("O",1)])
  fails(lambda:m.read_parent_registrations(p,surfaces,{"tets":np.zeros((1,4),dtype=np.int64)},0.02,
       current_typed=True,active_rows_by_role=active),"physical parent registration for inactive O source row 1")
  write_nodes(p,inactive_donor=True)
  fails(lambda:m.read_surfaces(p,current_typed=True,active_rows_by_role=active),
        "inactive provenance row 1 contains physical donor/frame data")
 print("PASS 217 full-source/face-active controls: preserve original IDs; active coverage exact; inactive parent or donor rejected")
if __name__=="__main__": main()
