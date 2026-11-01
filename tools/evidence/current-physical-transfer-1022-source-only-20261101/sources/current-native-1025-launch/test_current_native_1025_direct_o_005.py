#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import hashlib

MODULE_PATH=Path(__file__).with_name("prepare_current_native_1025_direct_o_005.py")
spec=importlib.util.spec_from_file_location("direct_o_004_tested",MODULE_PATH)
mod=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class DirectOTests(unittest.TestCase):
    def make_route(self,root:Path):
        files={}
        for name in ("parent.csv","parent.csv.labels.csv","parent.csv.nodes.csv","parent.csv.source.csv","parent.csv.tets.csv","registered-skin-poses-100.json","o-nodes.csv","o-source.nhskin"):
            p=root/name; p.write_text(name+"\n"); files[str(p)]=sha(p)
        parent=root/"parent.csv"
        pose=root/"registered-skin-poses-100.json"
        route=SimpleNamespace(
            REQUIRED_ENV={
                "PARENT_PACKET_CSV","PARENT_PACKET_SHA256","PARENT_LABELS_CSV","PARENT_LABELS_SHA256",
                "PARENT_NODES_SHA256","PARENT_SOURCE_SHA256","PARENT_TETS_SHA256",
                "MAXIMUM_MASS_TRANSPORT_DISTANCE_M","MAX_PARENT_REGISTRATION_DISTANCE_M","COORDINATE_FRAME_SHA256",
                "SURFACE_O_NODES_CSV","SURFACE_O_NODES_SHA256","SURFACE_O_STABLE_IDENTIFIER",
                "ROLE_O_SOURCE_DOMAIN","ROLE_O_SOURCE_ASSET","ROLE_O_SOURCE_ASSET_SHA256",
                "DETAIL_TO_BOUNDARY_CSV","DETAIL_TO_BOUNDARY_SHA256"
            },
            PARENT=parent,SOURCE330=root,
            PINNED_PARENT={
                str(parent):files[str(parent)],
                str(parent)+".labels.csv":files[str(parent)+".labels.csv"],
                str(parent)+".nodes.csv":files[str(parent)+".nodes.csv"],
                str(parent)+".source.csv":files[str(parent)+".source.csv"],
                str(parent)+".tets.csv":files[str(parent)+".tets.csv"],
                str(pose):files[str(pose)]
            },
            MAX_MASS_TRANSPORT_M="0.019353",
            MAX_NATIVE_PARENT_REGISTRATION_M="0.0004",
            PREFIX="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_",
            REQUIRED_GEOMETRY_CLAIMS=set(), REQUIRED_MASS_CLAIMS=set()
        )
        env={
            "PARENT_PACKET_CSV":str(parent),"PARENT_PACKET_SHA256":files[str(parent)],
            "PARENT_LABELS_CSV":str(parent)+".labels.csv","PARENT_LABELS_SHA256":files[str(parent)+".labels.csv"],
            "PARENT_NODES_SHA256":files[str(parent)+".nodes.csv"],
            "PARENT_SOURCE_SHA256":files[str(parent)+".source.csv"],
            "PARENT_TETS_SHA256":files[str(parent)+".tets.csv"],
            "MAXIMUM_MASS_TRANSPORT_DISTANCE_M":"0.019353",
            "MAX_PARENT_REGISTRATION_DISTANCE_M":"0.0004",
            "COORDINATE_FRAME_SHA256":files[str(pose)],
            "SURFACE_O_NODES_CSV":str(root/"o-nodes.csv"),
            "SURFACE_O_NODES_SHA256":files[str(root/"o-nodes.csv")],
            "SURFACE_O_STABLE_IDENTIFIER":"51005",
            "ROLE_O_SOURCE_DOMAIN":"current433:O",
            "ROLE_O_SOURCE_ASSET":str(root/"o-source.nhskin"),
            "ROLE_O_SOURCE_ASSET_SHA256":files[str(root/"o-source.nhskin")]
        }
        pins=dict(files)
        return route,env,pins,files

    def test_direct_o_accepts_without_any_detail_map(self):
        with tempfile.TemporaryDirectory() as temp:
            route,env,pins,_=self.make_route(Path(temp))
            result=mod.direct_env(route,{"native_preflight_env":env},pins)
            self.assertEqual(result,env)
            self.assertFalse(any(k.startswith("DETAIL_") for k in result))

    def test_direct_o_rejects_legacy_detail_map_even_when_pinned(self):
        with tempfile.TemporaryDirectory() as temp:
            route,env,pins,files=self.make_route(Path(temp))
            env["DETAIL_TO_BOUNDARY_CSV"]=str(Path(temp)/"parent.csv")
            env["DETAIL_TO_BOUNDARY_SHA256"]=files[env["DETAIL_TO_BOUNDARY_CSV"]]
            with self.assertRaisesRegex(ValueError,"forbids legacy detail"):
                mod.direct_env(route,{"native_preflight_env":env},pins)

    def test_direct_o_rejects_stale_o_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            route,env,pins,_=self.make_route(Path(temp))
            env["ROLE_O_SOURCE_ASSET_SHA256"]="0"*64
            with self.assertRaisesRegex(ValueError,"path/hash"):
                mod.direct_env(route,{"native_preflight_env":env},pins)

    def test_cpp_direct_o_pass_marker_requires_positive_readback(self):
        stdout=("registered_volume_native_preflight=PASS parent_nodes=2 O_direct_render_rows=3 "
                "O_direct_render_nodes=10 O_direct_render=typed_one_hot_prepared physical_admission=0 "
                "scene_installation=0 simulation_steps=0\n")
        self.assertEqual(mod.parse_o_pass(stdout),{"direct_O_rows":3,"direct_O_volume_nodes":10})
        with self.assertRaisesRegex(ValueError,"positive"):
            mod.parse_o_pass(stdout.replace("O_direct_render_rows=3","O_direct_render_rows=0"))

    def test_cpp_pass_marker_does_not_accept_legacy_bridge_export(self):
        stdout=("registered_volume_native_preflight=PASS O_direct_render_rows=3 O_direct_render_nodes=10 "
                "O_direct_render=typed_one_hot_prepared physical_admission=0 scene_installation=0 simulation_steps=0\n"
                "registered_volume_typed_O_canonical_face_export_source_only=PASS\n")
        with self.assertRaisesRegex(ValueError,"unexpectedly executed"):
            mod.parse_o_pass(stdout)

    def test_generated_stage_forces_bridge_off_and_keeps_baseline_settings(self):
        class Route:
            PREFIX="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
            def build_stage_env(self,scene,suffix,**kwargs):
                result=dict(scene)
                result.update({self.PREFIX+k:v for k,v in suffix.items()})
                result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE"]="1"
                result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"]="1"
                result["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"]="1"
                return result
        env=mod.apply_direct_env(Route(),{"BASELINE_PHYSICS":"preserved","NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE":"1"},
                                 {"SURFACE_O_NODES_CSV":"/tmp/o.csv","DETAIL_TO_BOUNDARY_CSV":"/tmp/old.csv"},True,True)
        self.assertEqual(env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE"],"0")
        self.assertEqual(env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"],"0")
        self.assertEqual(env["NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE"],"0")
        self.assertEqual(env["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"],"1")
        self.assertIn("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_NODES_CSV",env)
        self.assertNotIn("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_DETAIL_TO_BOUNDARY_CSV",env)

if __name__=="__main__": unittest.main()

