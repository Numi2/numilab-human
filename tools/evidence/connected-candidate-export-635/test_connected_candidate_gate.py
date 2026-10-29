from pathlib import Path
import importlib.util,json,hashlib,numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209");O=P/"connected-candidate-export-634"
sp=importlib.util.spec_from_file_location("connected_gate",O/"verify_and_export_candidate.py")
g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
with np.load(g.BASE,allow_pickle=False) as z: base={k:z[k] for k in z.files}
checks=[]
def run(name,edit,expected):
 c={k:v.copy() for k,v in base.items()};edit(c)
 try:g.check_candidate(base,c);actual="PASS";reason=None
 except ValueError as e:actual="REJECTED";reason=str(e)
 assert actual==expected,(name,actual,reason)
 checks.append({"name":name,"actual":actual,"reason":reason})
def position(c,i,p):
 c["points_current_world_m_f32"][i]=p
 c["points_reference_world_m_f64"][i]=c["points_current_world_m_f32"][i].astype(np.float64)
run("unchanged_real_006",lambda c:None,"PASS")
pin=int(base["pinned_volume_node_rows"][0])
pins=set(map(int,base["pinned_volume_node_rows"]))
movable=next(i for i in range(len(base["points_current_world_m_f32"])) if i not in pins)
def harmless(c):
 x=c["points_current_world_m_f32"][movable].copy()
 x[0]=np.nextafter(x[0],np.float32(np.inf))
 position(c,movable,x)
run("unprotected_one_ULP_change",harmless,"PASS")
run("protected_row_move",lambda c:position(c,pin,c["points_current_world_m_f32"][pin]+np.float32(.000001)),"REJECTED")
def stale(c):c["points_current_world_m_f32"][movable,0]+=np.float32(.00001)
run("stale_reference_alias",stale,"REJECTED")
run("nonfinite_coordinate",lambda c:position(c,movable,[np.nan,0,0]),"REJECTED")
def bad_weight(c):c["full86_weights"][movable,0]+=1e-9
run("altered_binding",bad_weight,"REJECTED")
def bad_source(c):c["stable_source_face_id"][0]+=1
run("altered_source_identity",bad_source,"REJECTED")
def bad_fraction(c):c["refinement_child_source_measure_fractions"][0]+=.001
run("altered_mass_ancestry",bad_fraction,"REJECTED")
cell=next(t for t in base["volume_tetrahedra"] if int(t[0]) not in pins)
run("collapsed_emitted_cell",lambda c:position(c,int(cell[0]),c["points_current_world_m_f32"][int(cell[1])]),"REJECTED")
(O/"regression.json").write_text(json.dumps({"status":"PASS","cases":checks,"source_sha256":g.digest(O/"verify_and_export_candidate.py"),"physical_admission":False},indent=2)+"\n")
print(json.dumps({"status":"PASS","cases":len(checks),"expected_rejections":7}))
