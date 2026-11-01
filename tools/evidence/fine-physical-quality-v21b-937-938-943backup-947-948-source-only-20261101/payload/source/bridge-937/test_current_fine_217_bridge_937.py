#!/usr/bin/env python3
import csv,gzip,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
ADAPTER=HERE/"current_fine_volume_to_217_937.py"
P1ADAPTER=HERE/"current_domain_p1_export_937.py"
def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
bridge=loadmod("bridge937_test",ADAPTER)
p1bridge=loadmod("p1bridge937_test",P1ADAPTER)
owner=bridge.import_owner()

def hbytes(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def _records(face_markers,roles,mat):
    out=[]
    for i,(mk,role) in enumerate(zip(face_markers,roles)):
        out.append({"marker":int(mk),"role":role,"source_row":i,"exact_source_row":i,
          "source_original_id":100+i,"source_layer":0,"source_body":1,"source_bone":1,
          "run_original_id":1,"source_face_id":i,"ancestry":[i],
          "ancestry_original_ids":[100+i],"ancestry_parent_rows":[-1],"ancestry_kind":1,
          "parent_tet_row":-1,"parent_material_index":mat,"parent_source_nodes":[-1,-1,-1]})
    return owner._merge_face_records([out])

def _volume(role):
    I=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    unused=np.asarray([[7,7,7]],np.float32)
    if role=="Skin":
        O=I.copy();O[:,2]=1
        p=np.vstack((I,unused,O)).astype(np.float32)
        t=np.asarray([[0,1,2,6],[0,1,5,6],[0,4,5,6]],np.int64)
        mat=15
    else:
        D=I.copy();D[:,2]=-1
        p=np.vstack((I,unused,D)).astype(np.float32)
        t=np.asarray([[4,5,6,2],[4,5,1,2],[4,0,1,2]],np.int64)
        mat=16
    pp,tt,_=owner.positive_f32_tetrahedra(p,t,mat)
    return pp,tt,np.full(len(tt),mat,np.int32),I

def _shell(role,p,t,I):
    tissue_role=role
    boundary=owner._tetrahedron_boundary_faces({"points":p,"tetrahedra":t,
      "element_ids":np.arange(1,len(t)+1),"material_indices":np.full(len(t),15 if role=="Skin" else 16)})
    if role=="Skin":
        surface_points=np.vstack((p[[4,5,6]],I)).astype(np.float32)
        mesh_to_shell={0:3,1:4,2:5,4:0,5:1,6:2}
        o_rows=[0,1,2];i_rows=[3,4,5];d_rows=[]
        o_marker, i_marker, wall_marker=1,2,3
        o_role,i_role,wall_role="outer_O","inner_I","donor_skin_OI_headwall"
    else:
        surface_points=np.vstack((I,p[[4,5,6]])).astype(np.float32)
        mesh_to_shell={0:0,1:1,2:2,4:3,5:4,6:5}
        o_rows=[];i_rows=[0,1,2];d_rows=[3,4,5]
        o_marker,i_marker,wall_marker=4,5,6
        o_role,i_role,wall_role="outer_I","inner_D","donor_fat_ID_headwall"
    all_faces=[];markers=[];roles=[];i_face_row=-1;o_face_row=-1;d_face_row=-1
    for key,entries in boundary.items():
        if len(entries)!=1:continue
        face=entries[0]["oriented_nodes"]
        mapped=[mesh_to_shell[int(v)] for v in face]
        if all(v in o_rows for v in mapped):
            marker,role=o_marker,o_role;o_face_row=len(all_faces)
        elif all(v in i_rows for v in mapped):
            if tissue_role=="Skin":
                marker,role=i_marker,i_role
            else:
                marker,role=o_marker,o_role
            i_face_row=len(all_faces)
        elif d_rows and all(v in d_rows for v in mapped):
            marker,role=i_marker,i_role;d_face_row=len(all_faces)
        else:marker,role=wall_marker,wall_role
        all_faces.append(mapped);markers.append(marker);roles.append(role)
    face_array=np.asarray(all_faces,np.int64)
    ancestry=_records(markers,roles,15 if tissue_role=="Skin" else 16)
    shell={"points_world_m":surface_points,"faces":face_array,
      "face_marker":np.asarray(markers,np.int32),"face_ancestry":ancestry}
    if tissue_role=="Skin":
        shell.update({"registered_O_source_vertex_rows":np.arange(3,dtype=np.int64),
          "registered_O_shell_node_rows":np.asarray([0,1,2]),"registered_O_source_faces":np.asarray([[0,1,2]]),
          "registered_O_shell_face_rows":np.asarray([o_face_row]),
          "registered_I_source_vertex_rows":np.arange(3,dtype=np.int64),
          "registered_I_shell_node_rows":np.asarray([3,4,5]),"registered_I_source_faces":np.asarray([[0,1,2]]),
          "registered_I_shell_faces":face_array[[i_face_row]],"registered_I_shell_face_rows":np.asarray([i_face_row])})
    else:
        shell.update({"registered_I_source_vertex_rows":np.arange(3,dtype=np.int64),
          "registered_I_shell_node_rows":np.asarray([0,1,2]),"registered_I_source_faces":np.asarray([[0,1,2]]),
          "registered_I_shell_faces":face_array[[i_face_row]],"registered_I_shell_face_rows":np.asarray([i_face_row]),
          "registered_D_source_vertex_rows":np.arange(3,dtype=np.int64),
          "registered_D_shell_node_rows":np.asarray([3,4,5]),"registered_D_source_faces":np.asarray([[0,1,2]]),
          "registered_D_shell_face_rows":np.asarray([d_face_row])})
    shell.update({"registered_I_source_face_rows":np.asarray([0]),"registered_I_source_exact_face_rows":np.asarray([0]),
      "registered_I_source_face_ids":np.asarray([77]),"registered_I_source_ancestry_offsets":np.asarray([0,1]),
      "registered_I_source_ancestry_rows":np.asarray([0])})
    return shell

def _write_shell(path,shell):
    d={"points_world_m":shell["points_world_m"],"faces":shell["faces"],"face_marker":shell["face_marker"],
       **{k:v for k,v in shell["face_ancestry"].items()},
       **{k:v for k,v in shell.items() if k.startswith("registered_")}}
    np.savez_compressed(path,**d)

def _write_volume(path,role,p,t,mat,current_state_path,current_state_sha,base_state_path,base_state_sha,basevol_path,basevol_sha,currentvol_path,currentvol_sha):
    n=len(t);offsets=np.arange(n+1,dtype=np.int64);rows=np.arange(n,dtype=np.int64)+(1000 if role=="Fat" else 0)
    payload={"points_world_m_f32":p,"tetrahedra":t,"material_indices":mat,
      "canonical_parent_union_offsets":offsets,"canonical_parent_union_parent_rows":rows,
      "canonical_parent_union_base_state_path":np.asarray(str(base_state_path)),
      "canonical_parent_union_base_state_sha256":np.asarray(base_state_sha),
      "canonical_parent_union_base_volume_path":np.asarray(str(basevol_path)),
      "canonical_parent_union_base_volume_sha256":np.asarray(basevol_sha),
      "canonical_parent_union_current_state_path":np.asarray(str(current_state_path)),
      "canonical_parent_union_current_state_sha256":np.asarray(current_state_sha),
      "canonical_parent_union_current_volume_path":np.asarray(str(currentvol_path)),
      "canonical_parent_union_current_volume_sha256":np.asarray(currentvol_sha),
      "canonical_parent_union_parent_row_domain":np.asarray("canonical_base_volume_cell_rows"),
      "canonical_parent_union_physical_status":np.asarray("SOURCE_ONLY_PHYSICAL_ADMISSION_FALSE")}
    np.savez_compressed(path,**payload)
    return offsets,rows

def _pin(p):
    p=Path(p).resolve()
    return {"path":str(p),"sha256":bridge.sha(p),"size_bytes":p.stat().st_size}

def end_to_end_fixture(tmp):
    tmp=Path(tmp)
    skinp,skint,skinm,I=_volume("Skin");fatp,fatt,fatm,_=_volume("Fat")
    owner.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
    # The separate state inputs are the upstream 818b volumes. The 834 NPZs
    # below carry the canonical parent unions and exact current-state pins.
    for role,p,t,m in (("Skin",skinp,skint,skinm),("Fat",fatp,fatt,fatm)):
        np.savez_compressed(tmp/f"{role.lower()}-pre834.npz",points_world_m_f32=p,tetrahedra=t,material_indices=m)
    source={}
    for role in ("O","I","D"):
        q=tmp/f"{role}.npz"
        if role=="I":np.savez_compressed(q,points_world_m=I,faces=np.asarray([[0,1,2]]),
             face_cache_row=np.asarray([0]),face_exact_source_row=np.asarray([0]),face_id=np.asarray([77]))
        else:np.savez_compressed(q,points_world_m=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32),
             faces=np.asarray([[0,1,2]]),face_source_row=np.asarray([0]),face_exact_source_row=np.asarray([0]),face_id=np.asarray([role=="D"],np.int64))
        source[role]=_pin(q)
    state={"Skin_current":_pin(tmp/"skin-pre834.npz"),"Fat_current":_pin(tmp/"fat-pre834.npz"),
      "source_surfaces":source,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
    state_path=tmp/"verified-state.json";state_path.write_text(json.dumps(state))
    base_state=tmp/"base-state.json";base_state.write_text(json.dumps({"physical_admission":False,"native_steps":0}))
    basevol=tmp/"base-role.npz";np.savez_compressed(basevol,tetrahedra=np.asarray([[0,1,2,3]],np.int64))
    base_state_pin=_pin(base_state);basevol_pin=_pin(basevol)
    canonical={}
    roleinfo={}
    for role,p,t,m in (("Skin",skinp,skint,skinm),("Fat",fatp,fatt,fatm)):
        original=tmp/f"{role.lower()}-pre834.npz"
        out=tmp/f"{role.lower()}-canonical.npz"
        off,rows=_write_volume(out,role,p,t,m,state_path,bridge.sha(state_path),base_state_pin["path"],base_state_pin["sha256"],
          basevol_pin["path"],basevol_pin["sha256"],original,bridge.sha(original))
        canonical[role]=out
        roleinfo[role]={"base_volume":basevol_pin,"current_volume":_pin(original),"output":_pin(out),
          "canonical_parent_offsets_entries":len(off),"canonical_parent_references":len(rows),
          "canonical_parent_offsets_sha256":hbytes(off.astype("<i8")),"canonical_parent_rows_sha256":hbytes(rows.astype("<i8"))}
    manifest=tmp/"849-manifest.json";manifest.write_text(json.dumps({"schema":bridge.MANIFEST_SCHEMA,
      "base_state":base_state_pin,"current_state":_pin(state_path),"roles":{}}))
    manpin=_pin(manifest)
    report834={"schema":bridge.CANONICAL_REPORT_SCHEMA,"status":"PASS_SOURCE_ONLY_CANONICAL_CURRENT_VOLUME_PARENT_UNIONS",
      "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","manifest":manpin,
      "current_state":_pin(state_path),"base_state":base_state_pin,"roles":roleinfo}
    report834_path=tmp/"834-report.json";report834_path.write_text(json.dumps(report834))
    report849={"schema":bridge.MANIFEST_REPORT_SCHEMA,"status":"PASS_SOURCE_ONLY_AUTODISCOVERED_CELL_ANCESTRY_MANIFEST",
      "physical_admission":False,"native_steps":0,"current_state":_pin(state_path),
      "manifest_path":str(manifest),"manifest_sha256":bridge.sha(manifest)}
    report849_path=tmp/"849-report.json";report849_path.write_text(json.dumps(report849))
    report818={"schema":bridge.PREFIX_SCHEMA,"status":"PASS_CURRENT_SOURCE_VOLUME_METADATA_PREFIX",
      "geometry_changed":False,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
      "output_state":_pin(state_path),"producer":{"path":str(bridge.OWNER818),"sha256":bridge.OWNER818_SHA}}
    report818_path=tmp/"818b-report.json";report818_path.write_text(json.dumps(report818))
    skin_shell=_shell("Skin",skinp,skint,I);fat_shell=_shell("Fat",fatp,fatt,I)
    sp=tmp/"skin-shell.npz";fp=tmp/"fat-shell.npz";_write_shell(sp,skin_shell);_write_shell(fp,fat_shell)
    shellreceipt=tmp/"shell-adapter-receipt.json"
    shellreceipt.write_text(json.dumps({"physical_admission":False,"outputs":{"skin-shell.npz":bridge.sha(sp),"fat-shell.npz":bridge.sha(fp)},
      "inputs":{v["path"]:v["sha256"] for v in source.values()}}))
    return {"skin_volume":canonical["Skin"],"fat_volume":canonical["Fat"],"skin_shell":sp,"fat_shell":fp,
      "inner":tmp/"I.npz","state":state_path,"report834":report834_path,"report849":report849_path,
      "report818":report818_path,"shell_receipt":shellreceipt,"skin_tets":skint,"fat_tets":fatt,
      "skin_points":skinp,"fat_points":fatp,"skin_union":roleinfo["Skin"],"fat_union":roleinfo["Fat"]}

class BridgeTests(unittest.TestCase):
    def test_full_source_only_roundtrip_and_active_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            q=end_to_end_fixture(td)
            result=bridge.execute(q["skin_volume"],q["fat_volume"],q["skin_shell"],q["fat_shell"],q["inner"],
               q["state"],q["report834"],q["shell_receipt"],Path(td)/"bundle",
               manifest_849_report_path=q["report849"],prefix_818b_report_path=q["report818"])
            self.assertEqual(result["status"],bridge.HANDOFF_STATUS)
            self.assertFalse(result["physical_admission"]);self.assertEqual(result["native_steps"],0)
            self.assertEqual(result["combined_skin_fat_child_mesh"]["combined_tetrahedra"],6)
            self.assertEqual(result["combined_skin_fat_child_mesh"]["registered_I_source_faces"],1)
            self.assertTrue(result["combined_skin_fat_child_mesh"]["only_registered_I_nodes_merged"])
            combined_mesh=owner.read_tetgen_mesh(Path(td)/"bundle"/"skin-fat-combined.node",
                                                 Path(td)/"bundle"/"skin-fat-combined.ele",None)
            self.assertEqual(combined_mesh["material_indices"].tolist(),[15,15,15,16,16,16])
            self.assertEqual(len(combined_mesh["points"]),result["combined_skin_fat_child_mesh"]["nodes"])
            self.assertEqual(result["role_meshes"]["Skin"]["removed_unreferenced_point_rows"],1)
            self.assertEqual(result["role_meshes"]["Fat"]["removed_unreferenced_point_rows"],1)
            # Keep original source row IDs after dropping unused row 3.
            with gzip.open(Path(td)/"bundle"/"skin-fat-combined-node-map.csv.gz","rt",newline="") as f:
                rows=list(csv.DictReader(f))
            skin_ids={int(r["source_tetgen_node_id"]) for r in rows if r["source_partition"]=="skin"}
            fat_ids={int(r["source_tetgen_node_id"]) for r in rows if r["source_partition"]=="fat"}
            self.assertEqual(skin_ids,{1,2,3,5,6,7});self.assertEqual(fat_ids,{1,2,3,5,6,7})
            with gzip.open(Path(td)/"bundle"/"skin-fat-combined-element-map.csv.gz","rt",newline="") as f:
                emap=list(csv.DictReader(f))
            self.assertEqual([int(r["child_tet_row"]) for r in emap],list(range(6)))
            self.assertEqual([r["source_partition"] for r in emap],["skin"]*3+["fat"]*3)
            self.assertEqual(result["canonical_parent_unions"]["Skin"]["rows_sha256"],q["skin_union"]["canonical_parent_rows_sha256"])
            self.assertEqual(result["canonical_parent_unions"]["Fat"]["rows_sha256"],q["fat_union"]["canonical_parent_rows_sha256"])
            self.assertFalse(result["canonical_parent_unions"]["Fat"]["source330_ancestry_claim"])

    def test_role_material_identity_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            q=end_to_end_fixture(td)
            owner=bridge.import_owner()
            with np.load(q["fat_volume"],allow_pickle=False) as z: payload={k:z[k] for k in z.files}
            payload["material_indices"]=np.full(len(payload["tetrahedra"]),15,np.int32)
            bad=Path(td)/"bad-fat-material.npz";np.savez_compressed(bad,**payload)
            with self.assertRaisesRegex(ValueError,"material IDs"):
                bridge.read_union_volume(bad,"Fat",owner)

    def test_shared_i_identity_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            q=end_to_end_fixture(td)
            with np.load(q["skin_shell"],allow_pickle=False) as z:sk={k:z[k] for k in z.files}
            with np.load(q["fat_shell"],allow_pickle=False) as z:ft={k:z[k] for k in z.files}
            ft["registered_I_source_vertex_rows"]=np.asarray([1,0,2],np.int64)
            with tempfile.TemporaryDirectory() as other:
                s=Path(other)/"s.npz";f=Path(other)/"f.npz";np.savez_compressed(s,**sk);np.savez_compressed(f,**ft)
                with self.assertRaisesRegex(ValueError,"identity"):
                    bridge.validate_shared_i(owner,bridge.load_shell_npz(s),bridge.load_shell_npz(f),
                      owner.load_surface(q["inner"],"I"))

    def test_capacity_failure_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            q=end_to_end_fixture(td);owner=bridge.import_owner()
            with self.assertRaisesRegex(ValueError,"capacity"):
                bridge.read_union_volume(q["skin_volume"],"Skin",owner,max_cells=2)

    def test_current_p1_config_pins_state_volume_support_and_handoff(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            raw={}
            for role,mat in (("Skin",15),("Fat",16)):
                raw[role]=td/f"{role.lower()}-raw.npz"
                np.savez_compressed(raw[role],points_world_m_f32=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32),
                    tetrahedra=np.asarray([[0,1,2,3]],np.int64),material_indices=np.asarray([mat],np.int32))
            state=td/"state.json"
            state_doc={"Skin_current":_pin(raw["Skin"]),"Fat_current":_pin(raw["Fat"]),
                       "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
            state.write_text(json.dumps(state_doc,sort_keys=True))
            current={}
            supports={}
            for role,mat in (("Skin",15),("Fat",16)):
                r=role.lower()
                current[role]=td/f"{r}-current.npz"
                np.savez_compressed(current[role],
                    points_world_m_f32=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32),
                    tetrahedra=np.asarray([[0,1,2,3]],np.int64),material_indices=np.asarray([mat],np.int32),
                    canonical_parent_union_offsets=np.asarray([0,1],np.int64),
                    canonical_parent_union_parent_rows=np.asarray([0],np.int64),
                    canonical_parent_union_current_state_path=np.asarray(str(state.resolve())),
                    canonical_parent_union_current_state_sha256=np.asarray(p1bridge.sha(state)),
                    canonical_parent_union_current_volume_path=np.asarray(str(raw[role].resolve())),
                    canonical_parent_union_current_volume_sha256=np.asarray(p1bridge.sha(raw[role])))
                supports[role]={}
                supports[role]["npz"]=td/f"{r}-support.npz"
                np.savez_compressed(supports[role]["npz"],cell_group_offsets=np.asarray([0,1],np.int64))
                receipt=td/f"{r}-support-receipt.json"
                receipt.write_text(json.dumps({"status":f"PASS_SOURCE_ONLY_{role.upper()}",
                    "physical_admission":False,"native_steps":0}))
                supports[role]["receipt"]=receipt
                supports[role]["status"]=f"PASS_SOURCE_ONLY_{role.upper()}"
                for kind in ("node","ele"):
                    f=td/f"{r}.{kind}";f.write_text(f"{role}-{kind}\n")
                    supports[role][kind]=f
            combined_files={}
            for name in ("element_map","combined_node_map","combined_node","combined_ele"):
                f=td/name;f.write_text(name+"\n");combined_files[name]=f
            handoff=td/"handoff.json"
            handoff.write_text(json.dumps({"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
                "inputs":{"current_state":_pin(state),
                  "skin_volume":_pin(current["Skin"]),"fat_volume":_pin(current["Fat"])}}))
            combined_files["handoff_receipt"]=handoff
            # Pinned source330 resources are immutable existing owners.
            full86=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.weights.csv")
            projection=p1owner_path()
            parent_mass=td/"parent-mass.csv";parent_mass.write_text("row,mass\n")
            pins=lambda p:{"path":str(Path(p).resolve()),"sha256":p1bridge.sha(p)}
            wrapped=td/"wrapped-state.json"
            wrapped.write_text(json.dumps({"status":"SOURCE_ONLY","current_verified_state_20261031":state_doc},sort_keys=True))
            self.assertIs(p1bridge.read_current_state(wrapped)["physical_admission"],False)
            config={"schema":p1bridge.CONFIG_SCHEMA,"current_state":pins(state),"counts":{"skin_cells":1,"fat_cells":1,"combined_cells":2},
                "owners":{"p1_owner":{"path":str(p1bridge.P1_SOURCE),"sha256":p1bridge.P1_SOURCE_SHA},
                          "label_exporter":{"path":str(p1bridge.LABEL_SOURCE),"sha256":p1bridge.LABEL_SOURCE_SHA}},
                "roles":{},"combined":{},"source":{}}
            for role,mat in (("Skin",15),("Fat",16)):
                r=role.lower()
                config["roles"][r]={"support_npz":pins(supports[role]["npz"]),
                    "support_receipt":pins(supports[role]["receipt"]),"current_volume":pins(current[role]),
                    "node":pins(supports[role]["node"]),"ele":pins(supports[role]["ele"]),
                    "cell_count":1,"material_index":mat,"status":supports[role]["status"]}
            for name,path in combined_files.items():config["combined"][name]=pins(path)
            config["source"]={"parent_mass_csv":pins(parent_mass),"full86_weights":pins(full86),
                "projection_helper":pins(projection)}
            config_path=td/"current-pins.json";config_path.write_text(json.dumps(config,sort_keys=True))
            checked,validated=p1bridge._validate_config(config_path,p1bridge.sha(config_path))
            self.assertEqual(validated["counts"]["combined_cells"],2)
            self.assertEqual(validated["current_state"]["sha256"],p1bridge.sha(state))
            argv=["--p1-supported-reference-anchor","--capacity-profile","full-reference-500k-nodes-2m-cells"]
            flags={"skin":"--p1-skin","fat":"--p1-fat"}
            for role in ("skin","fat"):
                f=flags[role]
                argv += [f+"-support-npz",config["roles"][role]["support_npz"]["path"],
                         f+"-support-receipt-json",config["roles"][role]["support_receipt"]["path"]]
            for flag,key in (("--combined-element-map-csv-gz","element_map"),
                             ("--combined-node-map-csv-gz","combined_node_map"),
                             ("--combined-handoff-receipt-json","handoff_receipt"),
                             ("--child-node","combined_node"),("--child-ele","combined_ele")):
                argv += [flag,config["combined"][key]["path"]]
            argv += ["--parent-mass-csv",config["source"]["parent_mass_csv"]["path"],
                     "--parent-full86-weights-csv",config["source"]["full86_weights"]["path"],
                     "--projection-helper-path",config["source"]["projection_helper"]["path"],
                     "--projection-helper-sha256",config["source"]["projection_helper"]["sha256"]]
            p1bridge._assert_cli_matches(argv,checked,validated)
            wrong=list(argv)
            wrong[wrong.index("--combined-node-map-csv-gz")+1]=str(parent_mass)
            with self.assertRaisesRegex(ValueError,"differs from explicit current-domain pins"):
                p1bridge._assert_cli_matches(wrong,checked,validated)
            # A coherent file-pin set cannot override the current-state volume ancestry.
            bad=dict(config);bad["current_state"]={"path":str(state.resolve()),"sha256":"0"*64}
            badpath=td/"bad-current-pins.json";badpath.write_text(json.dumps(bad,sort_keys=True))
            with self.assertRaisesRegex(ValueError,"current_state file pin mismatch"):
                p1bridge._validate_config(badpath,p1bridge.sha(badpath))

    def test_current_p1_dynamic_pins_are_explicit_and_isolated(self):
        p1owner=p1bridge._load_p1_owner({"roles":{
          "skin":{"support_npz":{"sha256":"a"*64},"support_receipt":{"sha256":"b"*64},"cell_count":2,"material_index":15,"status":"PASS_SOURCE_ONLY_SKIN"},
          "fat":{"support_npz":{"sha256":"c"*64},"support_receipt":{"sha256":"d"*64},"cell_count":3,"material_index":16,"status":"PASS_SOURCE_ONLY_FAT",
                 "current_volume":{"sha256":"e"*64}}},
          "counts":{"skin_cells":2,"fat_cells":3,"combined_cells":5},
          "combined":{k:{"sha256":chr(97+i)*64} for i,k in enumerate(("element_map","handoff_receipt","combined_node","combined_ele","combined_node_map","skin_node","fat_node","skin_ele","fat_ele"))},
          "source":{"full86_weights":{"sha256":p1bridge.FULL86_SHA},"projection_helper":{"path":str(p1owner_path()),"sha256":"02b04cff325b4adea24d055fdfd2121ab4d596a3c075ab6fe8183094eb46e1d6"}}})
        self.assertEqual((p1owner.SKIN_CELLS,p1owner.FAT_CELLS,p1owner.COMBINED_CELLS),(2,3,5))
        self.assertEqual(p1owner.FAT_SUPPORT_VOLUME_SHA256,"e"*64)
        self.assertEqual(p1owner.P1_PINS["skin"]["material"],15)
        self.assertEqual(p1owner.P1_PINS["fat"]["material"],16)
        # The immutable historical source007 module retains its old row domain.
        original=loadmod("p1_reference_anchor_937_original_check",p1bridge.P1_SOURCE)
        self.assertEqual((original.SKIN_CELLS,original.FAT_CELLS),(743497,853406))

def p1owner_path():
    return H/"batched_reference_tet_projection_149_root.py"

if __name__=="__main__":unittest.main(verbosity=2)
