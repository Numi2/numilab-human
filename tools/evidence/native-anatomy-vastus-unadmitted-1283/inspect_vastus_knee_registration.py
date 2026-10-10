from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,importlib.util,numpy as np
from PIL import Image,ImageDraw
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");N=A/"twenty-two-surface-native-composition-001/baseline/native-run";O=A/"vastus-knee-registration-inspection-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-positive-winding-1279/src")
from numilab_human import cardiac_cavity_intersections as ci, common_atlas_skin_clearance as ca,myosim_bone_proximity as bp
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pack=N/"accepted-geometry/step-8000.mrvpack";receipt=pack.with_suffix(".receipt.json")
rc=json.loads(receipt.read_text());assert sha(pack)==rc["pack_file_sha256"]
keys={(51005,x) for x in [61,62,65,66]}|{(51004,x) for x in [2,3,4,5,24,25]}
positions,surfaces,_=ca._pack_surfaces(pack,keys)
pins={str(p):sha(p) for p in [Path(__file__),pack,receipt,Path(ci.__file__),Path(ca.__file__),Path(bp.__file__)]}
report={"scope":"Independent localization of actual native distal VI/VM interface witnesses against registered knee bones. Orthographic projections of actual native triangles, no inferred footprint or physical-force qualification.","pins":pins,"rows":[]}
img=Image.new("RGB",(1800,1120),"white");draw=ImageDraw.Draw(img)
colors={(51004,2):(188,180,158),(51004,3):(188,180,158),(51004,4):(203,197,179),(51004,5):(203,197,179),(51004,24):(220,194,91),(51004,25):(220,194,91),(51005,61):(128,172,215),(51005,62):(128,172,215),(51005,65):(185,113,160),(51005,66):(185,113,160)}
for rowindex,(sid,vm,femur,tibia,patella) in enumerate([(61,65,2,4,24),(62,66,3,5,25)]):
 source_tri=positions[np.asarray(surfaces[(51005,sid)]["faces"])[3134]];target_tri=positions[np.asarray(surfaces[(51005,vm)]["faces"])[1128]]
 exact=lambda a:tuple(tuple(Fraction.from_float(float(x)) for x in v) for v in a)
 ip=ci.triangle_intersection_points(exact(source_tri),exact(target_tri));assert ip
 witness=np.asarray([[float(x) for x in p] for p in ip]);center=witness.mean(0)
 row={"stable_id":sid,"target_id":vm,"candidate_face":3134,"target_face":1128,"native_witness_m":witness.tolist(),"bone_distances_m":{}}
 centers={}
 for bid in [femur,tibia,patella]:
  triangles=positions[np.asarray(surfaces[(51004,bid)]["faces"])].astype(float);centers[bid]=triangles.reshape(-1,3).mean(0)
  ds=bp._point_triangle_distances_squared(center,triangles,np);idx=int(np.argmin(ds));point,bary=bp._closest_point_on_triangle(center,triangles[idx],np)
  row["bone_distances_m"][str(bid)]={"distance_m":float(np.sqrt(ds[idx])),"nearest_triangle":idx,"nearest_point_m":point.tolist(),"barycentric":bary}
 proximal=centers[femur]-centers[tibia];proximal/=np.linalg.norm(proximal)
 anterior=centers[patella]-(centers[femur]+centers[tibia])/2;anterior-=proximal*(proximal@anterior);anterior/=np.linalg.norm(anterior)
 lateral=np.cross(anterior,proximal);basis=np.column_stack((lateral,anterior,proximal))
 row["inspection_basis_world_columns_lateral_anterior_proximal"]=basis.tolist();report["rows"].append(row)
 for panel,(ax0,ax1,title) in enumerate([(0,2,"Lateral / proximal"),(1,2,"Anterior / proximal"),(0,1,"Lateral / anterior")]):
  left=panel*600;top=rowindex*560
  draw.text((left+12,top+12),f"Native row {sid} VI (blue), {vm} VM (pink), knee bones. {title}",fill="black")
  draw.text((left+12,top+30),"50 mm square centred on exact interface witness (red). Patella gold.",fill="black")
  polys=[]
  for key in [(51004,femur),(51004,tibia),(51004,patella),(51005,sid),(51005,vm)]:
   tris=positions[np.asarray(surfaces[key]["faces"])].astype(float)
   local=(tris-center)@basis;sel=np.flatnonzero(np.all(local.max(1)>=-.025,axis=1)&np.all(local.min(1)<=.025,axis=1))
   for i in sel:
    x=local[i][:,[ax0,ax1]]
    points=[(left+300+float(v[0])*9000,top+300-float(v[1])*9000) for v in x]
    if all(left+55<=x<left+545 and top+55<=y<top+545 for x,y in points):polys.append((float(local[i][:,3-ax0-ax1].mean()),points,colors[key]))
  for _,points,color in sorted(polys,key=lambda x:x[0]):draw.polygon(points,fill=color,outline=tuple(max(0,c-35) for c in color))
  draw.ellipse((left+296,top+296,left+304,top+304),fill="red")
  draw.rectangle((left+75,top+75,left+525,top+525),outline="black",width=1)
img.save(O/"registered-knee-interface.png")
report["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());assert report["inputs_unchanged"]
(O/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report["rows"],indent=2))
