#!/Users/numi/numi-human-prep-venv-20261005/bin/python
from __future__ import annotations
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CONFIG_PATH=HERE/"adapter-config.json"
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=P/"fine-physical-volume-706"
EXPECTED_SCHEMA="numi.human.tetgen-source004-role-adapter-config.v1"
EXPECTED_CONFIG_SHA256="1e20cb90b301dfc648cab1be9772542e1cff3a8a7578bbe2f1105e84fefe35ef"
REGION_ENV="NUMI_TETGEN_REGION_SEED_COMPLETION"

def sha256(path:Path)->str:
 d=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):d.update(block)
 return d.hexdigest()

def verify_pins(config:dict, path_overrides:dict[str,Path]|None=None)->dict[str,str]:
 if config.get("schema")!=EXPECTED_SCHEMA:raise ValueError("adapter config schema mismatch")
 paths=path_overrides or {k:Path(v["path"]) for k,v in config.get("pins",{}).items()}
 actual={}
 for key,pin in config.get("pins",{}).items():
  path=Path(paths[key])
  if not path.is_file():raise FileNotFoundError(f"pinned dependency missing: {key}")
  got=sha256(path);actual[key]=got
  if got!=pin.get("sha256"):raise ValueError(f"pinned dependency SHA mismatch: {key}")
 if config.get("improvement_mode")!="explicit-zero-with-nonmutating-get_tetqual-scan":
  raise ValueError("adapter must retain zero-improvement validation mode")
 if config.get("zero_improvement_suffix")!="O0/0/0":raise ValueError("unexpected TetGen option suffix")
 policy=config.get("region_seed_completion",{})
 if policy.get("maximum_reruns_per_material")!=1 or policy.get("enabled_by_code") is not True:
  raise ValueError("region-seed retry policy is not bounded and explicit")
 for key in ("requires_exact_poly_shell_point_face_marker_match","requires_positive_signed_source_shell_volume",
             "requires_existing_and_added_region_material_equal_role","preserves_existing_region_rows_literally",
             "requires_signed_triangle_volume_matches_manifold_volume"):
  if policy.get(key) is not True:raise ValueError(f"required source004 region-seed gate disabled: {key}")
 receipt=json.loads(Path(paths["tetgen054_compile_receipt"]).read_text())
 if receipt.get("binary_sha256")!=actual["tetgen054_binary"] or receipt.get("source_sha256")!=actual["tetgen054_source"]:
  raise ValueError("054 binary/source do not match its compile receipt")
 return actual

def load_config(path:Path=CONFIG_PATH)->dict:
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=EXPECTED_CONFIG_SHA256:raise ValueError("adapter config SHA mismatch")
 config=json.loads(raw)
 if config.get("schema")!=EXPECTED_SCHEMA:raise ValueError("adapter config schema mismatch")
 return config

def region_retry_enabled(environ:dict[str,str])->bool:
 value=environ.get(REGION_ENV,"")
 if value in ("","0"):return False
 if value=="1":return True
 raise ValueError(f"{REGION_ENV} must be unset, 0, or 1")

def compose_tetgen_command(binary:Path,args:list[str],config:dict)->list[str]:
 if len(args)!=2:raise ValueError("source004 must pass exactly one option string and one PLC path")
 expected=config.get("required_parent_argv_option")
 if args[0]!=expected:raise ValueError("unexpected source004 TetGen option string")
 return [str(binary),args[0]+config["zero_improvement_suffix"],args[1]]

def load_module(name:str,path:Path,extra_paths:list[Path]):
 sys.path[:0]=[str(p) for p in extra_paths]
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return module

def parse_mesh(node_path:Path,ele_path:Path):
 def data_lines(path):
  return [line.strip() for line in Path(path).read_text().splitlines()
          if line.strip() and not line.lstrip().startswith("#")]
 nl=data_lines(node_path);el=data_lines(ele_path)
 nh=list(map(int,nl[0].split()))
 if len(nh)<4 or nh[1]!=3 or nh[2]!=0 or nh[3] not in (0,1):raise ValueError("unexpected TetGen node table header")
 if len(nl)!=nh[0]+1:raise ValueError("TetGen node row count mismatch")
 node_ids=[];points=[]
 for line in nl[1:]:
  v=line.split()
  if len(v)<4:raise ValueError("short TetGen node row")
  node_ids.append(int(v[0]));points.append([float(x) for x in v[1:4]])
 if len(set(node_ids))!=len(node_ids):raise ValueError("duplicate TetGen node IDs")
 index={node_id:i for i,node_id in enumerate(node_ids)}
 eh=list(map(int,el[0].split()))
 if len(eh)!=3 or eh[1]!=4 or eh[2]!=1:raise ValueError("TetGen element table must carry exactly one region attribute")
 if len(el)!=eh[0]+1:raise ValueError("TetGen element row count mismatch")
 tets=[];attrs=[]
 for line in el[1:]:
  v=line.split()
  if len(v)<1+4+eh[2]:raise ValueError("short TetGen element row")
  try: tet=[index[int(x)] for x in v[1:5]]
  except KeyError as exc:raise ValueError("element references absent node ID") from exc
  attr=float(v[5])
  if not np.isfinite(attr) or attr!=round(attr):raise ValueError("noninteger TetGen region attribute")
  tets.append(tet);attrs.append(int(attr))
 p=np.asarray(points,dtype=np.float64);t=np.asarray(tets,dtype=np.int64);a=np.asarray(attrs,dtype=np.int64)
 if not np.isfinite(p).all():raise ValueError("nonfinite TetGen node coordinates")
 return p,t,a

def absolute_volume_sum(points:np.ndarray,tets:np.ndarray)->float:
 q=points[tets]
 v=np.einsum("ij,ij->i",q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6.0
 if not np.isfinite(v).all():raise ValueError("nonfinite tetrahedron volume")
 return float(np.abs(v).sum())

def poly_sections(path:Path):
 lines=[x for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
 if not lines:raise ValueError("empty TetGen poly")
 nh=list(map(int,lines[0].split()))
 if len(nh)!=4 or nh[1:]!=[3,0,1] or nh[0]<4:raise ValueError("poly points must retain expected 3D, no attributes, marker layout")
 point_count=nh[0];pos=1;points=[];point_markers=[]
 for row in range(point_count):
  v=lines[pos].split();pos+=1
  if len(v)!=5 or int(v[0])!=row+1:raise ValueError("poly point rows are not canonical 1-based rows")
  xyz=[float(x) for x in v[1:4]];marker=int(v[4])
  if not np.isfinite(xyz).all():raise ValueError("nonfinite poly point")
  points.append(xyz);point_markers.append(marker)
 fh=list(map(int,lines[pos].split()));pos+=1
 if len(fh)!=2 or fh[1]!=1 or fh[0]<1:raise ValueError("poly facets must be one-polygon marker facets")
 face_count=fh[0];faces=[];face_markers=[]
 for _ in range(face_count):
  ph=list(map(int,lines[pos].split()));pos+=1
  tri=list(map(int,lines[pos].split()));pos+=1
  if len(ph)!=3 or ph[0]!=1 or ph[1]!=0:raise ValueError("unexpected poly facet header")
  if len(tri)!=4 or tri[0]!=3:raise ValueError("poly facet must be triangular")
  if min(tri[1:])<1 or max(tri[1:])>point_count:raise ValueError("poly facet references invalid point")
  faces.append([x-1 for x in tri[1:]]);face_markers.append(ph[2])
 if pos>=len(lines):raise ValueError("truncated poly hole section")
 hole_count=int(lines[pos]);pos+=1
 if hole_count<0 or pos+hole_count>=len(lines):raise ValueError("truncated poly hole rows")
 holes=[]
 for row in range(hole_count):
  fields=lines[pos].split();pos+=1
  if len(fields)!=4 or int(fields[0])!=row+1:raise ValueError("poly hole rows are not canonical")
  xyz=[float(x) for x in fields[1:4]]
  if not np.isfinite(xyz).all():raise ValueError("nonfinite poly hole point")
  holes.append(lines[pos-1])
 if pos>=len(lines):raise ValueError("truncated poly region section")
 region_index=pos;region_count=int(lines[pos]);pos+=1
 if region_count<0 or pos+region_count!=len(lines):raise ValueError("poly region row count mismatch")
 regions=[]
 for row in range(region_count):
  fields=lines[pos].split();pos+=1
  if len(fields)!=6 or int(fields[0])!=row+1:raise ValueError("poly region rows are not canonical six-field rows")
  xyz=[float(x) for x in fields[1:4]];attribute=int(fields[4]);max_volume=float(fields[5])
  if not np.isfinite(xyz).all() or not np.isfinite(max_volume):raise ValueError("nonfinite poly region row")
  regions.append({"line":lines[pos-1],"point":xyz,"material":attribute,"max_volume":max_volume})
 return {"lines":lines,"region_index":region_index,"region_count":region_count,
         "point_count":point_count,"face_count":face_count,"hole_count":hole_count,
         "points":np.asarray(points,dtype=np.float64),"point_markers":np.asarray(point_markers,dtype=np.int64),
         "faces":np.asarray(faces,dtype=np.int64),"face_markers":np.asarray(face_markers,dtype=np.int64),
         "holes":holes,"regions":regions}

def assert_poly_matches_shell(poly:Path,shell:dict,expected_material:int)->dict:
 parsed=poly_sections(poly)
 source_points=np.asarray(shell["points_world_m"],dtype=np.float64)
 source_faces=np.asarray(shell["faces"],dtype=np.int64)
 source_markers=np.asarray(shell["face_marker"],dtype=np.int64)
 if source_points.ndim!=2 or source_points.shape[1]!=3 or not np.isfinite(source_points).all():raise ValueError("invalid source shell points")
 if source_faces.shape!=(len(source_markers),3):raise ValueError("invalid source shell face/marker rows")
 if parsed["point_count"]!=len(source_points) or not np.array_equal(parsed["points"],source_points):
  raise ValueError("poly points differ from the loaded source shell NPZ")
 if np.any(parsed["point_markers"]!=0):raise ValueError("poly point markers differ from source004 point marker domain")
 if parsed["face_count"]!=len(source_faces) or not np.array_equal(parsed["faces"],source_faces):
  raise ValueError("poly facets differ from the loaded source shell NPZ")
 if not np.array_equal(parsed["face_markers"],source_markers):raise ValueError("poly facet markers differ from the loaded source shell NPZ")
 if parsed["region_count"]<1 or any(r["material"]!=expected_material for r in parsed["regions"]):
  raise ValueError("source poly region material differs from requested role")
 return {"point_count":parsed["point_count"],"face_count":parsed["face_count"],
         "hole_count":parsed["hole_count"],"region_count":parsed["region_count"],
         "region_materials":sorted({r["material"] for r in parsed["regions"]}),
         "points_faces_markers_match_shell_npz":True}

def assert_same_shell_and_existing_regions(source:Path,completed:Path,seeds:list[dict],expected_material:int)->str:
 old=poly_sections(source);new=poly_sections(completed)
 if (old["region_index"],old["point_count"],old["face_count"],old["hole_count"]) != \
    (new["region_index"],new["point_count"],new["face_count"],new["hole_count"]):
  raise ValueError("completed poly changed shell section counts")
 if old["lines"][:old["region_index"]]!=new["lines"][:new["region_index"]]:
  raise ValueError("completed poly changed source points/facets/holes")
 old_region_lines=[r["line"] for r in old["regions"]]
 new_region_lines=[r["line"] for r in new["regions"]]
 if new_region_lines[:old["region_count"]]!=old_region_lines:
  raise ValueError("completed poly changed existing material region seeds")
 if new["region_count"]!=old["region_count"]+len(seeds):
  raise ValueError("completed poly region count does not match added seeds")
 if any(r["material"]!=expected_material for r in old["regions"]+new["regions"]):
  raise ValueError("source or completed region material differs from requested role")
 for actual,seed in zip(new["regions"][old["region_count"]:],seeds):
  if actual["material"]!=int(seed["material"]) or not np.array_equal(
      np.asarray(actual["point"],dtype=np.float64),np.asarray(seed["point_world_m"],dtype=np.float64)):
   raise ValueError("completed poly changed a verified region seed point or material")
 return hashlib.sha256("\n".join(old["lines"][:old["region_index"]]).encode()).hexdigest()

def signed_surface_volume(points:np.ndarray,faces:np.ndarray)->float:
 p=np.asarray(points,dtype=np.float64);f=np.asarray(faces,dtype=np.int64)
 if p.ndim!=2 or p.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not np.isfinite(p).all():
  raise ValueError("invalid closed shell arrays for signed volume")
 if len(f)==0 or int(f.min())<0 or int(f.max())>=len(p):raise ValueError("closed shell faces reference invalid points")
 q=p[f]-p.mean(axis=0)
 six=np.einsum("ij,ij->i",q[:,0],np.cross(q[:,1],q[:,2]))
 value=float(six.sum(dtype=np.float64)/6.0)
 if not np.isfinite(value):raise ValueError("nonfinite signed source shell volume")
 return value

def volume_closure(mesh_abs_volume:float,shell_signed_volume:float,policy:dict)->dict:
 if not np.isfinite(mesh_abs_volume) or mesh_abs_volume<=0:raise ValueError("first TetGen mesh has invalid total absolute volume")
 if not np.isfinite(shell_signed_volume) or shell_signed_volume<=0:raise ValueError("source closed shell has nonpositive signed volume")
 tol=max(float(policy["source_shell_volume_abs_tolerance_m3"]),
         float(policy["source_shell_volume_relative_tolerance"])*shell_signed_volume)
 error=abs(mesh_abs_volume-shell_signed_volume)
 if error>tol:raise ValueError("first TetGen total absolute volume differs from source closed signed shell volume")
 return {"tet_abs_volume_sum_m3":mesh_abs_volume,"shell_signed_volume_m3":shell_signed_volume,
         "absolute_error_m3":error,"tolerance_m3":tol,"pass":True}

def copy_expected_tables(source_prefix:Path,target_prefix:Path)->dict[str,str]:
 result={}
 for ext in ("node","ele"):
  src=Path(str(source_prefix)+".1."+ext);dst=Path(str(target_prefix)+".1."+ext)
  if not src.is_file():raise FileNotFoundError(f"rerun output is missing: {src}")
  shutil.copy2(src,dst)
  if sha256(src)!=sha256(dst):raise IOError("final expected TetGen table copy changed bytes")
  result[ext]=sha256(dst)
 return result

def run_wrapper(args:list[str],config:dict,dependencies:dict[str,str])->int:
 if len(args)!=2:raise ValueError("expected source004 option and poly path")
 poly=Path(args[1]).resolve()
 if poly.suffix!=".poly" or poly.stem not in ("skin","fat"):raise ValueError("only source004 skin.poly or fat.poly is accepted")
 role=poly.stem;material=15 if role=="skin" else 16;outdir=poly.parent
 side_dir=outdir/f"tetgen-region-seed-completion-{role}"
 side_dir.mkdir(exist_ok=False)
 binary=H/"tetgen-explicit-no-improvement-054"/"tetgen"
 first_cmd=compose_tetgen_command(binary,[args[0],str(poly)],config)
 optin=region_retry_enabled(dict(os.environ));timeout_cap=int(config["adapter_total_timeout_s"])
 if timeout_cap<1 or timeout_cap>=int(config["outer_owner_timeout_s"]):raise ValueError("adapter timeout must leave margin under source004 timeout")
 start=time.monotonic();deadline=start+timeout_cap; shell_path=outdir/f"{role}-shell.npz"
 receipt={"schema":"numi.human.tetgen-source004-role-adapter-run.v1","role":role,"material":material,
  "source_poly_sha256":sha256(poly),"source_shell_npz_sha256":sha256(shell_path) if shell_path.is_file() else None,
  "wrapper_argv":args,"first_argv":first_cmd,"dependencies_sha256":dependencies,
  "region_seed_retry_opt_in":optin,"region_seed_retry_status":"NOT_CHECKED_OPT_IN_DISABLED" if not optin else "NOT_CHECKED",
  "maximum_region_seed_reruns":1,"physical_admission":False,"runtime":0,"accepted_steps":0,
  "source004_final_cell_gate":"DELEGATED_TO_UNCHANGED_OWNER_POSITIVE_F32_ALL_MATERIALS_AND_CAPACITY_CHECKS"}
 receipt_path=outdir/f"tetgen-adapter-{role}.json"
 def save_receipt():receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
 try:
  if not shell_path.is_file():raise FileNotFoundError("source004 shell NPZ is missing beside PLC")
  with np.load(shell_path,allow_pickle=False) as z:shell={k:z[k] for k in z.files}
  poly_match=assert_poly_matches_shell(poly,shell,material)
  owner_path=Path(config["pins"]["native217_source004"]["path"])
  owner=load_module("tetgen_owner_for_region_seed_adapter",owner_path,[owner_path.parent,P.parent/"authoring-deps"])
  whole=owner._manifold(shell["points_world_m"],shell["faces"],shell["face_marker"])
  shell_signed_volume=signed_surface_volume(shell["points_world_m"],shell["faces"])
  manifold_volume=float(whole.volume())
  if not np.isfinite(shell_signed_volume) or shell_signed_volume<=0:raise ValueError("source closed shell has nonpositive signed volume")
  volume_tol=max(float(config["region_seed_completion"]["source_shell_volume_abs_tolerance_m3"]),
                 float(config["region_seed_completion"]["source_shell_volume_relative_tolerance"])*shell_signed_volume)
  if not np.isfinite(manifold_volume) or abs(manifold_volume-shell_signed_volume)>volume_tol:
   raise ValueError("source shell signed volume differs from closed Manifold volume")
  receipt["source_poly_shell_match"]=poly_match
  receipt["source_shell_signed_volume_m3"]=shell_signed_volume
  receipt["closed_manifold_volume_m3"]=manifold_volume
 except Exception as error:
  receipt["status"]="SOURCE_PLC_PREFLIGHT_REJECTED_NO_TETGEN_RUN";receipt["rejection_reason"]=str(error);save_receipt();raise
 first_remaining=deadline-time.monotonic()
 if first_remaining<=1:
  receipt["status"]="SOURCE_PLC_PREFLIGHT_EXHAUSTED_TIMEOUT_MARGIN_NO_TETGEN_RUN"
  receipt["rejection_reason"]="preflight left less than one second under the bounded adapter timeout"
  save_receipt();raise TimeoutError(receipt["rejection_reason"])
 first=subprocess.run(first_cmd,cwd=outdir,capture_output=True,text=True,timeout=first_remaining)
 (side_dir/"first.stdout.log").write_text(first.stdout);(side_dir/"first.stderr.log").write_text(first.stderr)
 first_prefix=outdir/role;first_node=Path(str(first_prefix)+".1.node");first_ele=Path(str(first_prefix)+".1.ele")
 receipt.update({"first_exit_code":first.returncode,"first_wall_s":time.monotonic()-start})
 if first.returncode!=0 or not first_node.is_file() or not first_ele.is_file():
  receipt["status"]="FIRST_TETGEN_FAILED_NO_PHYSICAL_ADMISSION";save_receipt()
  sys.stdout.write(first.stdout);sys.stderr.write(first.stderr)
  return first.returncode if first.returncode else 2
 if not optin:
  receipt["status"]="FIRST_TETGEN_COMPLETE_REGION_RETRY_DISABLED_NO_PHYSICAL_ADMISSION"
  receipt["first_node_sha256"]=sha256(first_node);receipt["first_element_sha256"]=sha256(first_ele);save_receipt()
  sys.stdout.write(first.stdout);sys.stderr.write(first.stderr)
  return 0
 # Byte-copy both first-pass tables before a retry; do not hardlink or overwrite them.
 saved_node=side_dir/f"first-{role}.1.node";saved_ele=side_dir/f"first-{role}.1.ele"
 shutil.copy2(first_node,saved_node);shutil.copy2(first_ele,saved_ele)
 if sha256(saved_node)!=sha256(first_node) or sha256(saved_ele)!=sha256(first_ele):
  raise IOError("failed to preserve the first TetGen output bytes")
 try:
  helper_path=Path(config["pins"]["region_seed_helper_056"]["path"])
  helper=load_module("region_seed_helper_056",helper_path,[])
  points,tets,attrs=parse_mesh(first_node,first_ele)
  mesh_volume=absolute_volume_sum(points,tets)
  closure=volume_closure(mesh_volume,shell_signed_volume,config["region_seed_completion"])
  seeds=helper.seeds_from_unowned(points,tets,attrs,material,lambda p:owner._point_inside(whole,p))
  receipt["first_node_sha256"]=sha256(first_node);receipt["first_element_sha256"]=sha256(first_ele)
  receipt["first_output_snapshots"]={saved_node.name:sha256(saved_node),saved_ele.name:sha256(saved_ele)}
  receipt["volume_closure"]=closure;receipt["unowned_component_count"]=len(seeds)
  if not seeds:
   receipt["status"]="FIRST_TETGEN_COMPLETE_NO_UNOWNED_COMPONENTS_NO_PHYSICAL_ADMISSION"
   receipt["region_seed_retry_status"]="NOT_NEEDED";save_receipt()
   sys.stdout.write(first.stdout);sys.stderr.write(first.stderr)
   return 0
  completed_poly=outdir/f"{role}-seed-completed.poly"
  poly_summary=helper.complete_poly(poly,completed_poly,seeds,owner)
  geom_sha=assert_same_shell_and_existing_regions(poly,completed_poly,seeds,material)
  receipt["region_seed_retry_status"]="PREPARED_AND_RUNNING_ONE_RERUN"
  receipt["region_seed_records"]=seeds;receipt["poly_completion_summary"]=poly_summary
  receipt["completed_poly_sha256"]=sha256(completed_poly);receipt["source_shell_poly_section_sha256"]=geom_sha
 except Exception as error:
  receipt["status"]="REGION_SEED_RETRY_REJECTED_FIRST_OUTPUTS_RETAINED_NO_PHYSICAL_ADMISSION"
  receipt["region_seed_retry_status"]="REJECTED_BEFORE_RERUN"
  receipt["rejection_reason"]=str(error)
  receipt["first_output_snapshots"]={saved_node.name:sha256(saved_node),saved_ele.name:sha256(saved_ele)}
  save_receipt();raise
 second_cmd=compose_tetgen_command(binary,[args[0],str(completed_poly)],config)
 remaining=deadline-time.monotonic()
 if remaining<=1:
  receipt["status"]="REGION_SEED_RETRY_TIMEOUT_BUDGET_REJECTED_FIRST_OUTPUTS_RETAINED_NO_PHYSICAL_ADMISSION"
  receipt["region_seed_retry_status"]="NOT_RUN_TIMEOUT_MARGIN";save_receipt()
  raise TimeoutError("source004 60s outer timeout margin would be exceeded before seed rerun")
 second=subprocess.run(second_cmd,cwd=outdir,capture_output=True,text=True,timeout=remaining)
 (side_dir/"second.stdout.log").write_text(second.stdout);(side_dir/"second.stderr.log").write_text(second.stderr)
 second_prefix=outdir/f"{role}-seed-completed"
 second_node=Path(str(second_prefix)+".1.node");second_ele=Path(str(second_prefix)+".1.ele")
 receipt["second_argv"]=second_cmd;receipt["second_exit_code"]=second.returncode
 receipt["second_wall_s"]=time.monotonic()-start-float(receipt["first_wall_s"])
 receipt["second_node_sha256"]=sha256(second_node) if second_node.is_file() else None
 receipt["second_element_sha256"]=sha256(second_ele) if second_ele.is_file() else None
 if second.returncode!=0 or not second_node.is_file() or not second_ele.is_file():
  receipt["status"]="SEED_RERUN_FAILED_FIRST_OUTPUTS_RETAINED_NO_PHYSICAL_ADMISSION";save_receipt()
  sys.stdout.write(first.stdout+"\n"+second.stdout);sys.stderr.write(first.stderr+"\n"+second.stderr)
  return second.returncode if second.returncode else 2
 final_hashes=copy_expected_tables(second_prefix,first_prefix)
 receipt["final_expected_table_sha256"]=final_hashes
 receipt["final_expected_tables_match_second_pass"]={"node":sha256(first_node)==sha256(second_node),"ele":sha256(first_ele)==sha256(second_ele)}
 if not all(receipt["final_expected_tables_match_second_pass"].values()):
  receipt["status"]="FINAL_COPY_MISMATCH_NO_PHYSICAL_ADMISSION";save_receipt()
  raise IOError("final expected TetGen outputs do not match the seed-completed pass")
 receipt["status"]="ONE_REGION_SEED_RERUN_COMPLETE_NO_PHYSICAL_ADMISSION"
 receipt["region_seed_retry_status"]="COMPLETED_ONCE"
 receipt["cell_attributes_mutated_by_adapter"]=False
 receipt["cell_rows_may_be_reordered_by_tetgen"]=True
 save_receipt()
 sys.stdout.write(first.stdout+"\n--- one verified region-seed retry ---\n"+second.stdout)
 sys.stderr.write(first.stderr+"\n"+second.stderr)
 return 0

def main():
 config=load_config()
 deps=verify_pins(config)
 try:return run_wrapper(sys.argv[1:],config,deps)
 except subprocess.TimeoutExpired as e:
  print(f"TetGen adapter bounded timeout: {e}",file=sys.stderr);return 124
 except Exception as e:
  print(f"TetGen adapter rejected input: {e}",file=sys.stderr);return 2

if __name__=="__main__":raise SystemExit(main())
