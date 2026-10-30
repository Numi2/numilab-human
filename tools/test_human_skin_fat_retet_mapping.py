#!/usr/bin/env python3
import unittest
import numpy as np
import csv, json, subprocess, sys, tempfile
from pathlib import Path
from human_skin_fat_retet_mapping import (
    assemble_selected_retet, signed_volume, validate_source330_cell_map_row,
    validate_historical_node_registration, validate_operational_node_registration,
    validate_combined_node_map, NODE_MAP_HEADER,
)

def fixture():
    p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],dtype=np.float32)
    t=np.array([[0,1,2,3],[0,2,1,4]],dtype=np.int64)
    m=np.array([15,15]);d=np.array([23,23]);r=np.array([23,23])
    base={"points_current_world_m_f32":p,
      "points_reference_world_m_f64":p.astype(np.float64),
      "volume_tetrahedra":t,"material_index":m,"geometric_donor_body":d,
      "anatomical_region_id":r,"stable_source_face_id":np.array([100,100]),
      "pinned_volume_node_rows":np.array([0],dtype=np.int64)}
    return base

def retet_identity(base):
    return {"points_current_world_m_f32":base["points_current_world_m_f32"][:4].copy(),
      "volume_tetrahedra":base["volume_tetrahedra"][[0],:4].copy(),
      "material_index":np.array([15]),"geometric_donor_body":np.array([23]),
      "anatomical_region_id":np.array([23]),
      "retet_node_base_rows":np.array([0,1,2,3])}

def retet_four_child(base):
    p=base["points_current_world_m_f32"][:4].copy()
    center=p.astype(np.float64).mean(axis=0).astype(np.float32)
    p=np.vstack((p,center))
    old=base["volume_tetrahedra"][0]
    faces=[]
    for o in range(4):
        q=[int(old[j]) for j in range(4) if j!=o]
        if o%2:q[1],q[2]=q[2],q[1]
        child=np.array([q[0],q[1],q[2],4],dtype=np.int64)
        if signed_volume(p,np.array([child]))[0]<0:child[[0,1]]=child[[1,0]]
        faces.append(child)
    return {"points_current_world_m_f32":p,"volume_tetrahedra":np.asarray(faces),
      "material_index":np.full(4,15),"geometric_donor_body":np.full(4,23),
      "anatomical_region_id":np.full(4,23),
      "retet_node_base_rows":np.array([0,1,2,3,-1])}


def retet_retire_apex(base):
    p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.5]],dtype=np.float32)
    return {"points_current_world_m_f32":p,
      "volume_tetrahedra":np.array([[0,1,2,3]],dtype=np.int64),
      "material_index":np.array([15]),"geometric_donor_body":np.array([23]),
      "anatomical_region_id":np.array([23]),"retet_node_base_rows":np.array([0,1,2,-1]),
      "protected_node_local_rows":np.array([3]),
      "coarse_boundary_faces_O":np.array([[1,2,3]],dtype=np.int64),
      "coarse_to_fine_face_O":np.array([0],dtype=np.int64),
      "coarse_to_fine_bary_O":np.array([[.2,.3,.5]],dtype=np.float64)}


def combined_map_fixture(points,tets,materials,donors,regions):
    p=np.asarray(points,dtype=np.float32);t=np.asarray(tets,dtype=np.int64)
    source_points=p.astype(np.float64)+np.array([0.5,0,0],dtype=np.float64)
    source_tets=t.copy()
    source_labels=[{"material_index":int(materials[i]),"geometric_donor_body":int(donors[i]),
        "anatomical_region_id":int(regions[i]),"stable_source_face_id":100} for i in range(len(t))]
    incident=[[] for _ in range(len(p))]
    for ci,tet in enumerate(t):
        for n in tet:incident[int(n)].append(ci)
    rows=[]
    for node,x in enumerate(p):
        support=incident[node][0];tet=t[support];corner=int(np.flatnonzero(tet==node)[0])
        source_row=next(i for i,st in enumerate(source_tets) if node in st)
        source_corner=int(np.flatnonzero(source_tets[source_row]==node)[0])
        beta=[0.0]*4;beta[corner]=1.0
        sbeta=[0.0]*4;sbeta[source_corner]=1.0
        row={k:"" for k in NODE_MAP_HEADER}
        row.update({"joint_node_id":str(node+1),"source_partition":"skin","source_tetgen_node_id":str(node+1),
          "registered_I_source_vertex_row":"-1","merge_peer_partition":"","merge_peer_tetgen_node_id":"",
          "registered_surface_layer":"","registered_surface_vertex_row":"-1","role":"interior",
          "current_x_m":str(float(x[0])),"current_y_m":str(float(x[1])),"current_z_m":str(float(x[2])),
          "donor_body":"23","donor_binding":"-1","donor_local_x_m":str(float(x[0]),),
          "donor_local_y_m":str(float(x[1])),"donor_local_z_m":str(float(x[2])),
          "source_domain":"derived_volume_interior_no_surface_ancestry","source_mapping_kind":"none","source_vertex_row":"-1","source_face_row":"-1",
          "source_bary0":"0","source_bary1":"0","source_bary2":"0",
          "source_residual_x_m":"0","source_residual_y_m":"0","source_residual_z_m":"0",
          "frame_owner_method":"derived_full_all_bodies_geometric_donor_pose","parent_registration_kind":"source_parent_tet_reference_registration",
          "parent_tet_row":str(source_row),"parent_current_residual_x_m":"-0.5","parent_current_residual_y_m":"0","parent_current_residual_z_m":"0",
          "parent_registration_distance_m":"0.5","parent_registration_tie_count":"0","incident_material_mask":str(1<<int(materials[support])),
          "geometric_donor_body":str(int(donors[support])),"anatomical_region_id":str(int(regions[support])),
          "source_parent_node_row":str(node),"operational_reference_tet_row":str(support),
          "operational_reference_residual_x_m":"0","operational_reference_residual_y_m":"0","operational_reference_residual_z_m":"0",
          "operational_reference_distance_m":"0","source_parent_geometric_donor_body":"23","source_parent_anatomical_region_id":"23"})
        for k in range(3):row[f"parent_bary{k}"]=str(sbeta[k])
        row["parent_bary3"]=str(sbeta[3])
        for k in range(4):row[f"operational_reference_bary{k}"]=str(beta[k])
        rows.append(row)
    return rows,source_points,source_tets,source_labels

class SpliceTests(unittest.TestCase):
    def setUp(self): self.base=fixture()
    def test_identity_retet_preserves_outside_and_cut(self):
        out,report=assemble_selected_retet(self.base,[0],retet_identity(self.base))
        self.assertEqual(report["outside_source_cells"],1)
        self.assertTrue(np.array_equal(out["volume_tetrahedra"][0],self.base["volume_tetrahedra"][1]))
        self.assertTrue(np.array_equal(out["material_index"][:1],self.base["material_index"][1:]))
        self.assertTrue(np.all(signed_volume(out["points_current_world_m_f32"],out["volume_tetrahedra"])>0))
        self.assertEqual(report["cut_faces_rechecked"],1)
    def test_four_child_topology_change_keeps_outside_exact(self):
        out,report=assemble_selected_retet(self.base,[0],retet_four_child(self.base))
        self.assertEqual(report["retet_cells"],4)
        self.assertTrue(np.array_equal(out["volume_tetrahedra"][0],self.base["volume_tetrahedra"][1]))
        self.assertTrue(np.array_equal(out["points_current_world_m_f32"][:5],self.base["points_current_world_m_f32"]))
        self.assertTrue(np.all(out["material_index"][1:]==15))
        self.assertTrue(np.all(signed_volume(out["points_current_world_m_f32"],out["volume_tetrahedra"])>0))
        self.assertEqual(report["cut_faces_rechecked"],1)
    def test_selected_domain_covers_each_semantic_group(self):
        b=self.base
        b["anatomical_region_id"]=np.array([23,54],dtype=np.int64)
        retet={"points_current_world_m_f32":b["points_current_world_m_f32"].copy(),
            "volume_tetrahedra":b["volume_tetrahedra"].copy(),"material_index":b["material_index"].copy(),
            "geometric_donor_body":b["geometric_donor_body"].copy(),"anatomical_region_id":b["anatomical_region_id"].copy(),
            "retet_node_base_rows":np.arange(5,dtype=np.int64)}
        out,report=assemble_selected_retet(b,[0,1],retet)
        self.assertEqual(report["retet_cells"],2)
        self.assertEqual(set(report["semantic_groups_retet"]),{"15/23/23","15/23/54"})
        bad=dict(retet);bad["anatomical_region_id"]=np.array([23,23],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,"semantic groups differ"):
            assemble_selected_retet(b,[0,1],bad)

    def test_semantic_tuple_cannot_be_relabelled(self):
        bad=retet_identity(self.base);bad["geometric_donor_body"]=np.array([24])
        with self.assertRaisesRegex(ValueError,"semantic groups differ"):
            assemble_selected_retet(self.base,[0],bad)
    def test_selected_only_node_retirement_and_protected_retarget(self):
        b=fixture()
        b["pinned_volume_node_rows"]=np.array([3],dtype=np.int64)
        b["coarse_boundary_faces_O"]=np.array([[1,2,3]],dtype=np.int64)
        b["coarse_to_fine_face_O"]=np.array([0],dtype=np.int64)
        b["coarse_to_fine_bary_O"]=np.array([[1/3,1/3,1/3]],dtype=np.float64)
        retet=retet_retire_apex(b)
        out,report=assemble_selected_retet(b,[0],retet)
        self.assertEqual(report["retired_base_node_rows"].tolist(),[3])
        self.assertEqual(out["base_node_output_rows"].tolist(),[0,1,2,-1,3])
        self.assertEqual(out["protected_base_node_rows"].tolist(),[3])
        self.assertEqual(out["protected_node_output_rows"].tolist(),[4])
        self.assertEqual(out["protected_node_mapping_kind"].tolist(),["explicit_retet_descendant"])
        self.assertEqual(out["pinned_volume_node_rows"].tolist(),[4])
        self.assertTrue(np.array_equal(out["volume_tetrahedra"][0],np.array([0,2,1,3])))
        self.assertTrue(np.array_equal(out["coarse_boundary_faces_O"],np.array([[1,2,4]])))
        self.assertTrue(np.all(signed_volume(out["points_current_world_m_f32"],out["volume_tetrahedra"])>0))

    def test_executable_splicer_compacts_retired_node_with_sidecars(self):
        root=Path(__file__).resolve().parent
        script=root/"splice_selected_tissue_retet.py"
        b=fixture();b["pinned_volume_node_rows"]=np.array([3],dtype=np.int64)
        b["coarse_boundary_faces_O"]=np.array([[1,2,3]],dtype=np.int64)
        b["coarse_to_fine_face_O"]=np.array([0],dtype=np.int64)
        b["coarse_to_fine_bary_O"]=np.array([[1/3,1/3,1/3]],dtype=np.float64)
        n=len(b["points_current_world_m_f32"]);p=b["points_current_world_m_f32"]
        b["points_donor_local_m_f64"]=p.astype(np.float64)
        b["parent_volume_node_row"]=np.arange(n,dtype=np.int64)
        b["full86_weights"]=np.zeros((n,86),dtype=np.float64);b["full86_weights"][:,0]=1.0
        b["target_current_world_m"]=p.astype(np.float64)
        b["refinement_node_source_offsets"]=np.arange(n+1,dtype=np.int64)
        b["refinement_node_source_rows"]=np.arange(n,dtype=np.int64)
        b["refinement_node_source_weights"]=np.ones(n,dtype=np.float64)
        packet=retet_retire_apex(b)
        q=packet["points_current_world_m_f32"]
        packet["points_donor_local_m_f64"]=q.astype(np.float64)
        packet["parent_volume_node_row"]=np.array([0,1,2,-1],dtype=np.int64)
        packet["full86_weights"]=np.zeros((4,86),dtype=np.float64);packet["full86_weights"][:3,0]=1.0;packet["full86_weights"][3,:2]=.5
        packet["target_current_world_m"]=q.astype(np.float64)
        packet["refinement_node_source_offsets"]=np.array([0,1,2,3,5],dtype=np.int64)
        packet["refinement_node_source_rows"]=np.array([0,1,2,2,4],dtype=np.int64)
        packet["refinement_node_source_weights"]=np.array([1,1,1,.5,.5],dtype=np.float64)
        with tempfile.TemporaryDirectory(prefix="retet671-retire-") as td:
            temp=Path(td);basepath=temp/"base.npz";selpath=temp/"selection.csv"
            np.savez_compressed(basepath,**b)
            with selpath.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["source_tet_row","material_index","geometric_donor_body","anatomical_region_id"])
                w.writerow([0,15,23,23])
            rp=temp/"retet.npz";out=temp/"out";receipt=out/"receipt.json"
            np.savez_compressed(rp,**packet)
            proc=subprocess.run([sys.executable,str(script),"--base-npz",str(basepath),
                "--selected-cells-csv",str(selpath),"--retet-npz",str(rp),
                "--output-dir",str(out),"--receipt-json",str(receipt)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
            report=json.loads(receipt.read_text())
            self.assertEqual(report["counts"]["retired_base_node_count"],1)
            self.assertEqual(report["counts"]["protected_rows_retargeted"],1)
            with np.load(report["outputs"]["geometry_npz"]["path"],allow_pickle=False) as g:
                self.assertEqual(g["base_node_output_rows"].tolist(),[0,1,2,-1,3])
                self.assertEqual(g["pinned_volume_node_rows"].tolist(),[4])
            with np.load(report["outputs"]["node_sidecars_npz"]["path"],allow_pickle=False) as sc:
                self.assertEqual(sc["base_node_output_rows"].tolist(),[0,1,2,-1,3])
                self.assertEqual(sc["protected_node_output_rows"].tolist(),[4])
                self.assertEqual(len(sc["full86_weights"]),5)


    def test_source_registration_keeps_historical_residual_separate(self):
        pts=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
        tets=np.array([[0,1,2,3]],dtype=np.int64)
        center=np.array([0.75,0.75,0.75])
        beta=np.full(4,0.25)
        reg=beta@pts[tets[0]]
        residual=center-reg
        row={"selected_parent_tet_row":"0","material_index":"15","geometric_donor_body":"23",
          "anatomical_region_id":"23","stable_source_face_id":"100",
          "centroid_barycentric0":"0.25","centroid_barycentric1":"0.25",
          "centroid_barycentric2":"0.25","centroid_barycentric3":"0.25",
          "centroid_registered_reference_x_m":str(reg[0]),"centroid_registered_reference_y_m":str(reg[1]),
          "centroid_registered_reference_z_m":str(reg[2]),
          "centroid_reference_residual_x_m":str(residual[0]),"centroid_reference_residual_y_m":str(residual[1]),
          "centroid_reference_residual_z_m":str(residual[2]),"centroid_reference_residual_norm_m":str(np.linalg.norm(residual)),
          "label_match_kind":"bounded_closest_parent_tet_reference_registration",
          "reference_registration_distance_m":str(np.linalg.norm(residual)),
          "reference_registration_max_distance_m":"1.0"}
        labels=[{"material_index":15,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":100}]
        result=validate_source330_cell_map_row(row,center,(15,23,23,100),labels,pts,tets)
        self.assertGreater(result["residual_norm_m"],0.4)
        stale=dict(row);stale["selected_parent_tet_row"]="9"
        with self.assertRaisesRegex(ValueError,"out of domain"):
            validate_source330_cell_map_row(stale,center,(15,23,23,100),labels,pts,tets)
        wrong=dict(row);wrong["stable_source_face_id"]="101"
        with self.assertRaisesRegex(ValueError,"semantic/stable-face"):
            validate_source330_cell_map_row(wrong,center,(15,23,23,100),labels,pts,tets)
    def test_combined_map_validates_source_history_and_incident_operational_support(self):
        b=self.base
        rows,source_points,source_tets,source_labels=combined_map_fixture(
            b["points_current_world_m_f32"],b["volume_tetrahedra"],b["material_index"],
            b["geometric_donor_body"],b["anatomical_region_id"])
        result=validate_combined_node_map(rows,b["points_current_world_m_f32"],
            b["points_reference_world_m_f64"],b["volume_tetrahedra"],b["material_index"],
            b["geometric_donor_body"],b["anatomical_region_id"],source_points,source_tets,source_labels)
        skin_i=dict(rows[0]);skin_i.update({"role":"I","registered_surface_layer":"I",
            "registered_surface_vertex_row":"0","registered_I_source_vertex_row":"0",
            "merge_peer_partition":"fat","merge_peer_tetgen_node_id":"101",
            "source_domain":"registered_I","source_mapping_kind":"vertex","source_vertex_row":"0",
            "frame_owner_method":"test-source-frame"})
        fat_i=dict(skin_i);fat_i.update({"source_partition":"fat","source_tetgen_node_id":"101",
            "merge_peer_partition":"skin","merge_peer_tetgen_node_id":"1"})
        paired=[skin_i,fat_i]+[dict(x) for x in rows[1:]]
        paired_report=validate_combined_node_map(paired,b["points_current_world_m_f32"],
            b["points_reference_world_m_f64"],b["volume_tetrahedra"],b["material_index"],
            b["geometric_donor_body"],b["anatomical_region_id"],source_points,source_tets,source_labels)
        self.assertEqual(paired_report["shared_interface_nodes"],1)
        with self.assertRaisesRegex(ValueError,"lacks exactly reciprocal"):
            validate_combined_node_map([skin_i]+[dict(x) for x in rows[1:]],b["points_current_world_m_f32"],
                b["points_reference_world_m_f64"],b["volume_tetrahedra"],b["material_index"],
                b["geometric_donor_body"],b["anatomical_region_id"],source_points,source_tets,source_labels)
        self.assertEqual(result["volume_nodes"],5)
        self.assertAlmostEqual(result["max_historical_residual_m"],.5)
        self.assertEqual(result["max_operational_residual_m"],0.0)
        stale=[dict(x) for x in rows];stale[0]["parent_tet_row"]="99"
        with self.assertRaisesRegex(ValueError,"parent tet row invalid"):
            validate_combined_node_map(stale,b["points_current_world_m_f32"],b["points_reference_world_m_f64"],
                b["volume_tetrahedra"],b["material_index"],b["geometric_donor_body"],b["anatomical_region_id"],
                source_points,source_tets,source_labels)
        wrong=[dict(x) for x in rows];wrong[0]["source_parent_anatomical_region_id"]="54"
        with self.assertRaisesRegex(ValueError,"parent donor/region mismatch"):
            validate_combined_node_map(wrong,b["points_current_world_m_f32"],b["points_reference_world_m_f64"],
                b["volume_tetrahedra"],b["material_index"],b["geometric_donor_body"],b["anatomical_region_id"],
                source_points,source_tets,source_labels)
        outside=[dict(x) for x in rows];outside[3]["operational_reference_tet_row"]="1"
        with self.assertRaisesRegex(ValueError,"not incident"):
            validate_combined_node_map(outside,b["points_current_world_m_f32"],b["points_reference_world_m_f64"],
                b["volume_tetrahedra"],b["material_index"],b["geometric_donor_body"],b["anatomical_region_id"],
                source_points,source_tets,source_labels)
        far=[dict(x) for x in rows]
        far[0]["operational_reference_bary0"]="0.9995";far[0]["operational_reference_bary1"]="0.0005"
        far[0]["operational_reference_residual_x_m"]="-0.0005";far[0]["operational_reference_distance_m"]="0.0005"
        with self.assertRaisesRegex(ValueError,"fixed bound"):
            validate_combined_node_map(far,b["points_current_world_m_f32"],b["points_reference_world_m_f64"],
                b["volume_tetrahedra"],b["material_index"],b["geometric_donor_body"],b["anatomical_region_id"],
                source_points,source_tets,source_labels)

    def test_executable_splicer_identity_and_changed_topology(self):
        root=Path(__file__).resolve().parent
        script=root/"splice_selected_tissue_retet.py"
        base=self.base
        base["coarse_boundary_faces_O"]=np.array([[1,2,3]],dtype=np.int64)
        n=len(base["points_current_world_m_f32"]);p=base["points_current_world_m_f32"]
        base["points_donor_local_m_f64"]=p.astype(np.float64)
        base["parent_volume_node_row"]=np.arange(n,dtype=np.int64)
        base["full86_weights"]=np.zeros((n,86),dtype=np.float64);base["full86_weights"][:,0]=1.0
        base["target_current_world_m"]=p.astype(np.float64)
        base["refinement_node_source_offsets"]=np.arange(n+1,dtype=np.int64)
        base["refinement_node_source_rows"]=np.arange(n,dtype=np.int64)
        base["refinement_node_source_weights"]=np.ones(n,dtype=np.float64)
        identity=retet_identity(base)
        identity["coarse_boundary_faces_O"]=np.array([[1,2,3]],dtype=np.int64)
        changed=retet_four_child(base)
        changed["coarse_boundary_faces_O"]=np.array([[1,2,3]],dtype=np.int64)
        changed_points=changed["points_current_world_m_f32"]
        changed["points_donor_local_m_f64"]=changed_points.astype(np.float64)
        changed["parent_volume_node_row"]=np.array([0,1,2,3,-1],dtype=np.int64)
        changed["full86_weights"]=np.zeros((5,86),dtype=np.float64);changed["full86_weights"][:,0]=1.0
        changed["target_current_world_m"]=changed_points.astype(np.float64)
        changed["refinement_node_source_offsets"]=np.array([0,1,2,3,4,6],dtype=np.int64)
        changed["refinement_node_source_rows"]=np.array([0,1,2,3,0,1],dtype=np.int64)
        changed["refinement_node_source_weights"]=np.array([1,1,1,1,.5,.5],dtype=np.float64)
        with tempfile.TemporaryDirectory(prefix="retet671-") as td:
            temp=Path(td);basepath=temp/"base.npz";selpath=temp/"selection.csv"
            np.savez_compressed(basepath,**base)
            with selpath.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["source_tet_row","material_index","geometric_donor_body","anatomical_region_id"])
                w.writerow([0,15,23,23])
            for name,packet in (("identity",identity),("changed",changed)):
                rp=temp/f"{name}-retet.npz";out=temp/f"{name}-out";receipt=out/"receipt.json"
                np.savez_compressed(rp,**packet)
                proc=subprocess.run([sys.executable,str(script),"--base-npz",str(basepath),
                    "--selected-cells-csv",str(selpath),"--retet-npz",str(rp),
                    "--output-dir",str(out),"--receipt-json",str(receipt)],capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
                self.assertTrue(receipt.is_file())
                report=json.loads(receipt.read_text())
                self.assertFalse(report["physical_admission"])
                with np.load(report["outputs"]["geometry_npz"]["path"],allow_pickle=False) as g:
                    self.assertEqual(len(g["volume_tetrahedra"]),2 if name=="identity" else 5)
                    self.assertTrue(np.array_equal(g["volume_tetrahedra"][0],base["volume_tetrahedra"][1]))
                    self.assertTrue(np.array_equal(g["coarse_boundary_faces_O"],base["coarse_boundary_faces_O"]))
                with np.load(report["outputs"]["node_sidecars_npz"]["path"],allow_pickle=False) as sc:
                    self.assertEqual(len(sc["full86_weights"]),n if name=="identity" else n+1)
                    self.assertEqual(int(sc["base_node_output_rows"][-1]),n-1)

    def test_source330_packer_end_to_end_changed_topology(self):
        root=Path(__file__).resolve().parent
        splice_script=root/"splice_selected_tissue_retet.py"
        packer=root/"prepare_spliced_child_inputs_source330.py"
        b=fixture();b["child_parent_tet_row"]=np.array([0,1],dtype=np.int64)
        b["stable_source_face_id"]=np.array([100,100],dtype=np.int64)
        p=b["points_current_world_m_f32"];n=len(p)
        b["points_donor_local_m_f64"]=p.astype(np.float64)
        b["parent_volume_node_row"]=np.arange(n,dtype=np.int64)
        b["full86_weights"]=np.zeros((n,86),dtype=np.float64);b["full86_weights"][:,0]=1.0
        b["target_current_world_m"]=p.astype(np.float64)
        b["refinement_node_source_offsets"]=np.arange(n+1,dtype=np.int64)
        b["refinement_node_source_rows"]=np.arange(n,dtype=np.int64)
        b["refinement_node_source_weights"]=np.ones(n,dtype=np.float64)
        packet=retet_four_child(b)
        q=packet["points_current_world_m_f32"]
        packet["points_donor_local_m_f64"]=q.astype(np.float64)
        packet["parent_volume_node_row"]=np.array([0,1,2,3,-1],dtype=np.int64)
        packet["full86_weights"]=np.zeros((5,86),dtype=np.float64);packet["full86_weights"][:,0]=1.0
        packet["target_current_world_m"]=q.astype(np.float64)
        packet["refinement_node_source_offsets"]=np.array([0,1,2,3,4,6],dtype=np.int64)
        packet["refinement_node_source_rows"]=np.array([0,1,2,3,0,1],dtype=np.int64)
        packet["refinement_node_source_weights"]=np.array([1,1,1,1,.5,.5],dtype=np.float64)
        with tempfile.TemporaryDirectory(prefix="retet671-source330-") as td:
            temp=Path(td);basepath=temp/"base.npz";selpath=temp/"selection.csv"
            retetpath=temp/"retet.npz";spliced=temp/"spliced";splicereceipt=spliced/"receipt.json"
            np.savez_compressed(basepath,**b);np.savez_compressed(retetpath,**packet)
            with selpath.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["source_tet_row","material_index","geometric_donor_body","anatomical_region_id"])
                w.writerow([0,15,23,23])
            proc=subprocess.run([sys.executable,str(splice_script),"--base-npz",str(basepath),
                "--selected-cells-csv",str(selpath),"--retet-npz",str(retetpath),
                "--output-dir",str(spliced),"--receipt-json",str(splicereceipt)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
            with np.load(splicereceipt.parent/"geometry.npz",allow_pickle=False) as z:
                g={k:np.asarray(z[k]) for k in z.files}
            pts=g["points_current_world_m_f32"].astype(np.float64)
            t=g["volume_tetrahedra"];mat=g["material_index"];donor=g["geometric_donor_body"];region=g["anatomical_region_id"]
            self.assertTrue(np.all(signed_volume(pts,t)>0))
            # Use an explicit source330 reference mesh with the same typed cells.
            srcpts=pts+np.array([.5,0,0]);srct=t.copy()
            nodelines=["parent_node_index,current_x_m,current_y_m,current_z_m"]
            nodelines += [f"{i},{x[0]:.17g},{x[1]:.17g},{x[2]:.17g}" for i,x in enumerate(srcpts)]
            parentnodes=temp/"parent.nodes.csv";parentnodes.write_text("\n".join(nodelines)+"\n")
            tetheader=["parent_tet_row","parent_node0","parent_node1","parent_node2","parent_node3","material_index","donor_body"]
            tetrows=[",".join(tetheader)]
            labrows=["parent_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id"]
            for i,tt in enumerate(srct):
                tetrows.append(",".join(map(str,[i,*map(int,tt),int(mat[i]),int(donor[i])])))
                labrows.append(",".join(map(str,[i,int(mat[i]),int(donor[i]),int(region[i]),100])))
            parenttets=temp/"parent.tets.csv";parenttets.write_text("\n".join(tetrows)+"\n")
            parentlabels=temp/"parent.labels.csv";parentlabels.write_text("\n".join(labrows)+"\n")
            fields=["child_tet_row","selected_parent_tet_row","label_match_kind"]
            fields += [f"centroid_barycentric{k}" for k in range(4)]
            fields += [f"centroid_registered_reference_{a}_m" for a in "xyz"]
            fields += [f"centroid_reference_residual_{a}_m" for a in "xyz"]
            fields += ["centroid_reference_residual_norm_m","reference_registration_distance_m",
                "reference_registration_max_distance_m","reference_registration_ambiguity_tolerance_m",
                "matching_parent_tet_count","semantic_tie_count","material_index","geometric_donor_body",
                "anatomical_region_id","stable_source_face_id"]
            cellmap=temp/"cell-map.csv"
            with cellmap.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for child in range(1,len(t)):
                    centroid=pts[t[child]].mean(axis=0);beta=np.full(4,.25)
                    registered=beta@srcpts[srct[child]];residual=centroid-registered
                    row={"child_tet_row":child,"selected_parent_tet_row":child,
                        "label_match_kind":"bounded_closest_parent_tet_reference_registration",
                        "centroid_reference_residual_norm_m":np.linalg.norm(residual),
                        "reference_registration_distance_m":np.linalg.norm(residual),
                        "reference_registration_max_distance_m":1.0,
                        "reference_registration_ambiguity_tolerance_m":1e-8,
                        "matching_parent_tet_count":1,"semantic_tie_count":0,
                        "material_index":int(mat[child]),"geometric_donor_body":int(donor[child]),
                        "anatomical_region_id":int(region[child]),"stable_source_face_id":100}
                    for k in range(4):row[f"centroid_barycentric{k}"]=beta[k]
                    for j,a in enumerate("xyz"):
                        row[f"centroid_registered_reference_{a}_m"]=registered[j]
                        row[f"centroid_reference_residual_{a}_m"]=residual[j]
                    w.writerow(row)
            # The end-to-end source map is independent of the operational node map.
            node_rows,combined_source_pts,combined_source_tets,combined_labels=combined_map_fixture(
                g["points_current_world_m_f32"],t,mat,donor,region)
            node_rows[-1]["source_parent_node_row"]="-1"
            combined=temp/"combined-node-map.csv"
            with combined.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=NODE_MAP_HEADER);w.writeheader();w.writerows(node_rows)
            out=temp/"packed";receipt=out/"receipt.json"
            proc=subprocess.run([sys.executable,str(packer),"--base-npz",str(basepath),
                "--geometry-npz",str(spliced/"geometry.npz"),"--cell-ancestry-csv",str(cellmap),
                "--parent-labels-csv",str(parentlabels),"--parent-nodes-csv",str(parentnodes),
                "--parent-tets-csv",str(parenttets),"--node-sidecars-npz",str(spliced/"node-sidecars.npz"),
                "--output-dir",str(out),"--combined-node-map-csv",str(combined),
                "--receipt-json",str(receipt)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
            report=json.loads(receipt.read_text())
            self.assertFalse(report["physical_admission"])
            self.assertFalse(report["mass_fractions_emitted"])
            self.assertEqual(report["counts"]["direct_source330_centroid_registration_checks"],4)
            self.assertEqual(report["counts"]["cut_collar_faces_checked"],1)
            self.assertEqual(report["counts"]["retired_base_node_rows"],0)
            nodefile=report["outputs"]["mesh.node"]["path"]
            elemfile=report["outputs"]["mesh.ele"]["path"]
            self.assertEqual(int(Path(nodefile).read_text().splitlines()[0].split()[0]),len(pts))
            self.assertEqual(int(Path(elemfile).read_text().splitlines()[0].split()[0]),len(t))


    def test_source330_packer_retargeted_pin_keeps_history_separate_from_operational_support(self):
        root=Path(__file__).resolve().parent
        splice=root/"splice_selected_tissue_retet.py";packer=root/"prepare_spliced_child_inputs_source330.py"
        b=fixture();b["pinned_volume_node_rows"]=np.array([3],dtype=np.int64)
        b["child_parent_tet_row"]=np.array([0,1],dtype=np.int64)
        b["stable_source_face_id"]=np.array([100,100],dtype=np.int64)
        p0=b["points_current_world_m_f32"];n=len(p0)
        b["points_donor_local_m_f64"]=p0.astype(np.float64)
        b["parent_volume_node_row"]=np.arange(n,dtype=np.int64)
        b["full86_weights"]=np.zeros((n,86));b["full86_weights"][:,0]=1
        b["target_current_world_m"]=p0.astype(np.float64)
        b["refinement_node_source_offsets"]=np.arange(n+1,dtype=np.int64)
        b["refinement_node_source_rows"]=np.arange(n,dtype=np.int64)
        b["refinement_node_source_weights"]=np.ones(n)
        retet=retet_retire_apex(b)
        for key in ("coarse_boundary_faces_O","coarse_to_fine_face_O","coarse_to_fine_bary_O"):retet.pop(key,None)
        q=retet["points_current_world_m_f32"];retet["points_reference_world_m_f64"]=q.astype(np.float64)
        retet["points_donor_local_m_f64"]=q.astype(np.float64)
        retet["parent_volume_node_row"]=np.array([0,1,2,-1])
        retet["full86_weights"]=np.zeros((4,86));retet["full86_weights"][:,0]=1
        retet["target_current_world_m"]=q.astype(np.float64)
        retet["refinement_node_source_offsets"]=np.array([0,1,2,3,5])
        retet["refinement_node_source_rows"]=np.array([0,1,2,0,3])
        retet["refinement_node_source_weights"]=np.array([1,1,1,.5,.5])
        with tempfile.TemporaryDirectory(prefix="retet671-retarget-e2e-") as td:
            d=Path(td);base=d/"base.npz";rp=d/"retet.npz";sel=d/"selected.csv";sp=d/"splice"
            np.savez_compressed(base,**b);np.savez_compressed(rp,**retet)
            with sel.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["source_tet_row","material_index","geometric_donor_body","anatomical_region_id"]);w.writerow([0,15,23,23])
            proc=subprocess.run([sys.executable,str(splice),"--base-npz",str(base),"--selected-cells-csv",str(sel),
                "--retet-npz",str(rp),"--output-dir",str(sp),"--receipt-json",str(sp/"receipt.json")],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
            with np.load(sp/"geometry.npz",allow_pickle=False) as z:g={k:np.asarray(z[k]) for k in z.files}
            with np.load(sp/"node-sidecars.npz",allow_pickle=False) as z:side={k:np.asarray(z[k]) for k in z.files}
            self.assertEqual(g["retired_base_node_rows"].tolist(),[3])
            self.assertEqual(g["protected_node_mapping_kind"].tolist(),["explicit_retet_descendant"])
            self.assertEqual(g["protected_node_output_rows"].tolist(),[4])
            self.assertEqual(int(side["parent_volume_node_row"][4]),-1)
            points,tets=g["points_current_world_m_f32"].astype(np.float64),g["volume_tetrahedra"]
            rows,source_points,source_tets,source_labels=combined_map_fixture(p0,b["volume_tetrahedra"],
                b["material_index"],b["geometric_donor_body"],b["anatomical_region_id"])
            out_to_old=[0,1,2,4,3];node_rows=[dict(rows[i]) for i in out_to_old]
            for node,(row,x) in enumerate(zip(node_rows,points)):
                row["joint_node_id"]=str(node+1);row["source_tetgen_node_id"]=str(node+1)
                for j,a in enumerate("xyz"):row[f"current_{a}_m"]=str(float(x[j]))
                support=next(ci for ci,tt in enumerate(tets) if node in tt);corner=int(np.flatnonzero(tets[support]==node)[0])
                row["operational_reference_tet_row"]=str(support)
                for k in range(4):row[f"operational_reference_bary{k}"]="1" if k==corner else "0"
                for a in "xyz":row[f"operational_reference_residual_{a}_m"]="0"
                row["operational_reference_distance_m"]="0"
            # Output row 4 is a regenerated descendant of old protected node 3.
            target=node_rows[4];target["source_parent_node_row"]="-1";target["parent_tet_row"]="0"
            pb=np.array([.5,0,0,.5]);source_xyz=pb@source_points[source_tets[0]]
            resid=points[4]-source_xyz
            for k in range(4):target[f"parent_bary{k}"]=str(pb[k])
            for j,a in enumerate("xyz"):target[f"parent_current_residual_{a}_m"]=str(resid[j])
            target["parent_registration_distance_m"]=str(np.linalg.norm(resid))
            # Build immutable source330 inputs from the original two-tet packet.
            nlines=["parent_node_index,current_x_m,current_y_m,current_z_m"]
            nlines += [f"{i},{x[0]:.17g},{x[1]:.17g},{x[2]:.17g}" for i,x in enumerate(source_points)]
            pn=d/"parent.nodes.csv";pn.write_text("\n".join(nlines)+"\n")
            th=["parent_tet_row","parent_node0","parent_node1","parent_node2","parent_node3","material_index","donor_body"]
            tr=[",".join(th)];lr=["parent_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id"]
            for i,tt in enumerate(source_tets):
                tr.append(",".join(map(str,[i,*map(int,tt),int(b["material_index"][i]),int(b["geometric_donor_body"][i])])))
                lr.append(",".join(map(str,[i,int(b["material_index"][i]),int(b["geometric_donor_body"][i]),int(b["anatomical_region_id"][i]),100])))
            pt=d/"parent.tets.csv";pt.write_text("\n".join(tr)+"\n")
            pl=d/"parent.labels.csv";pl.write_text("\n".join(lr)+"\n")
            fields=["child_tet_row","selected_parent_tet_row","label_match_kind"]+[f"centroid_barycentric{k}" for k in range(4)]
            fields += [f"centroid_registered_reference_{a}_m" for a in "xyz"]+[f"centroid_reference_residual_{a}_m" for a in "xyz"]
            fields += ["centroid_reference_residual_norm_m","reference_registration_distance_m","reference_registration_max_distance_m",
                "reference_registration_ambiguity_tolerance_m","matching_parent_tet_count","semantic_tie_count",
                "material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"]
            cm=d/"cell-map.csv";beta=np.array([.375,.25,.25,.125]);cent=points[tets[1]].mean(axis=0)
            registered=beta@source_points[source_tets[0]];cr=cent-registered
            row={"child_tet_row":1,"selected_parent_tet_row":0,"label_match_kind":"bounded_closest_parent_tet_reference_registration",
                "centroid_reference_residual_norm_m":np.linalg.norm(cr),"reference_registration_distance_m":np.linalg.norm(cr),
                "reference_registration_max_distance_m":.6,"reference_registration_ambiguity_tolerance_m":1e-8,
                "matching_parent_tet_count":1,"semantic_tie_count":0,"material_index":15,"geometric_donor_body":23,
                "anatomical_region_id":23,"stable_source_face_id":100}
            for k in range(4):row[f"centroid_barycentric{k}"]=beta[k]
            for j,a in enumerate("xyz"):
                row[f"centroid_registered_reference_{a}_m"]=registered[j];row[f"centroid_reference_residual_{a}_m"]=cr[j]
            with cm.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
            nm=d/"combined-node-map.csv"
            with nm.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=NODE_MAP_HEADER).writeheader();csv.DictWriter(f,fieldnames=NODE_MAP_HEADER).writerows(node_rows)
            out=d/"packed"
            proc=subprocess.run([sys.executable,str(packer),"--base-npz",str(base),"--geometry-npz",str(sp/"geometry.npz"),
                "--cell-ancestry-csv",str(cm),"--parent-labels-csv",str(pl),"--parent-nodes-csv",str(pn),
                "--parent-tets-csv",str(pt),"--node-sidecars-npz",str(sp/"node-sidecars.npz"),
                "--output-dir",str(out),"--combined-node-map-csv",str(nm),"--receipt-json",str(out/"receipt.json")],
                capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
            report=json.loads((out/"receipt.json").read_text())
            self.assertFalse(report["physical_admission"]);self.assertFalse(report["mass_fractions_emitted"])
            self.assertEqual(report["counts"]["retired_base_node_rows"],1)
            self.assertEqual(report["counts"]["protected_rows_retargeted"],1)
            self.assertEqual(report["counts"]["protected149_rows"],1)
            self.assertEqual(report["counts"]["protected_targets"][0]["mapping_kind"],"explicit_retet_descendant")
            with nm.open("r",newline="") as f:checked={int(x["joint_node_id"])-1:x for x in csv.DictReader(f)}
            self.assertEqual(checked[4]["source_parent_node_row"],"-1")
            self.assertAlmostEqual(float(checked[4]["parent_registration_distance_m"]),.5)
            self.assertEqual(float(checked[4]["operational_reference_distance_m"]),0.0)

    def test_historical_parent_and_operational_maps_are_separate(self):
        p=self.base["points_current_world_m_f32"].astype(np.float64)
        t=self.base["volume_tetrahedra"];m=self.base["material_index"];d=self.base["geometric_donor_body"];r=self.base["anatomical_region_id"]
        # Historical residual is 0.5 m but reconstructs exactly and is allowed in split mode.
        srcp=np.array([[x+0.5,y,z] for x,y,z in p],dtype=np.float64)
        srct=self.base["volume_tetrahedra"]
        src_labels=[{"geometric_donor_body":23,"anatomical_region_id":23} for _ in range(2)]
        hist={"parent_tet_row":"0","parent_bary0":"1","parent_bary1":"0","parent_bary2":"0","parent_bary3":"0",
          "parent_current_residual_x_m":"-0.5","parent_current_residual_y_m":"0","parent_current_residual_z_m":"0",
          "parent_registration_distance_m":"0.5","source_parent_geometric_donor_body":"23",
          "source_parent_anatomical_region_id":"23","source_parent_node_row":"0"}
        self.assertGreater(validate_historical_node_registration(hist,srcp,srct,src_labels,p[0])["historical_distance_m"],0.4)
        op={"operational_reference_tet_row":"0","geometric_donor_body":"23","anatomical_region_id":"23",
          "operational_reference_bary0":"1","operational_reference_bary1":"0","operational_reference_bary2":"0","operational_reference_bary3":"0",
          "operational_reference_residual_x_m":"0","operational_reference_residual_y_m":"0","operational_reference_residual_z_m":"0",
          "operational_reference_distance_m":"0"}
        validate_operational_node_registration(op,0,p,p,t,m,d,r)
        bad=dict(op);bad["operational_reference_tet_row"]="1"
        # Node 3 is absent from tet 1, so the false support must fail.
        with self.assertRaisesRegex(ValueError,"not incident"):
            validate_operational_node_registration(bad,3,p,p,t,m,d,r)
        far=dict(op);far["operational_reference_distance_m"]="0.0005"
        with self.assertRaisesRegex(ValueError,"fixed bound"):
            validate_operational_node_registration(far,0,p,p,t,m,d,r)

if __name__=="__main__": unittest.main()
