import copy
import struct
import unittest
from numilab_human.model import ImportError
from numilab_human.support_primitives import compile_support_primitives

class SupportPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.exported = {"support_contact": {
            "ground": {"point_world_m": [0,0,0], "normal_world": [0,0,1], "friction_tangential": 1},
            "geometries": [{"name": "foot", "id": 42, "body": 7, "friction_tangential": 0.8,
                "primitive": {"kind": "capsule", "radius_m": 0.02,
                              "endpoints_local_com_m": [[-0.1,0,0],[0.1,0,0]],
                              "endpoint_plane_gaps_m": [-0.02,-0.02]}}]}}
        self.manifest = {"source": {"archive_sha256": "12"*32}, "core_tree": {"engine_body_count": 3},
            "support_contact": {"support_geometries": [
                {"name": "foot", "source_geom_id": 42, "source_body_id": 7, "core_body_index": 2}]}}
        self.exported["source"]=copy.deepcopy(self.manifest["source"])

    def test_foreign_or_missing_source_provenance_fails_closed(self):
        for source in (None,{"archive_sha256":"34"*32}):
            self.exported["source"]=source
            with self.assertRaises(ImportError):compile_support_primitives(self.exported,self.manifest)

    def test_two_endpoints_keep_native_body_identity_and_source_binding(self):
        metadata, payload = compile_support_primitives(self.exported, self.manifest)
        self.assertEqual(len(payload), 180)
        self.assertEqual(payload[:8], b"NHCNT2\0\0")
        self.assertEqual(payload[24:56], bytes.fromhex("12"*32))
        self.assertEqual(struct.unpack_from("<4I", payload, 84), (2,42,2,0))
        self.assertEqual(metadata["expanded_contact_count"], 2)
        self.assertEqual([r["endpoint"] for r in metadata["rows"]], [0,1])
        self.assertEqual(compile_support_primitives(self.exported, self.manifest), (metadata,payload))
    def test_invalid_and_nonfinite_radius(self):
        for radius in (-1,0,float("nan"),float("inf"),True,1e-99,1e99):
            e=copy.deepcopy(self.exported);e["support_contact"]["geometries"][0]["primitive"]["radius_m"]=radius
            with self.assertRaises(ImportError):compile_support_primitives(e,self.manifest)
    def test_foreign_identity_and_missing_or_duplicate_primitive(self):
        for mutation in ("foreign", "missing", "duplicate", "unsupported"):
            e=copy.deepcopy(self.exported);g=e["support_contact"]["geometries"]
            if mutation=="foreign":g[0]["body"]=99
            if mutation=="missing":g.clear()
            if mutation=="duplicate":g.append(copy.deepcopy(g[0]))
            if mutation=="unsupported":g[0]["primitive"]["kind"]="box"
            with self.assertRaises(ImportError):compile_support_primitives(e,self.manifest)
    def test_ellipsoid_keeps_shape_frame_and_positive_axes(self):
        p=self.exported["support_contact"]["geometries"][0]["primitive"]
        p.update(kind="ellipsoid", radius_m=0, radii_m=[0.1,0.2,0.3], orientation_xyzw=[0,0,0,1],
                 endpoints_local_com_m=[[0,0,0],[0,0,0]])
        metadata,payload=compile_support_primitives(self.exported,self.manifest)
        self.assertEqual(metadata["expanded_contact_count"],1)
        self.assertEqual(struct.unpack_from("<I",payload,92)[0],3)
        self.assertEqual(struct.unpack_from("<4f",payload,148),(0,0,0,1))
        p["radii_m"][1]=0
        with self.assertRaises(ImportError):compile_support_primitives(self.exported,self.manifest)

    def test_sphere_requires_coincident_endpoints(self):
        g=self.exported["support_contact"]["geometries"][0]["primitive"];g["kind"]="sphere"
        with self.assertRaises(ImportError):compile_support_primitives(self.exported,self.manifest)
        g["endpoints_local_com_m"][1]=g["endpoints_local_com_m"][0].copy()
        metadata,_=compile_support_primitives(self.exported,self.manifest)
        self.assertEqual(metadata["expanded_contact_count"],1)
