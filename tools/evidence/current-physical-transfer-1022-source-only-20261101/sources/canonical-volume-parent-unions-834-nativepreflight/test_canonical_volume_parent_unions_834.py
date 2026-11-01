#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("canonical_volume_parent_unions_834", HERE / "canonical_volume_parent_unions_834.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def step(parent_t, parent_m, retained, replacements, rep_parents, direct_tail=None):
    retained = np.asarray(retained, np.int64)
    replacements = np.asarray(replacements, np.int64).reshape((-1, 4))
    out_t = np.vstack((parent_t[retained], replacements))
    out_m = np.r_[parent_m[retained], np.full(len(replacements), 7, np.int64)]
    rep_offsets = np.zeros(len(rep_parents) + 1, np.int64)
    for i, x in enumerate(rep_parents):
        rep_offsets[i + 1] = rep_offsets[i] + len(x)
    rep_rows = np.concatenate([np.asarray(x, np.int64) for x in rep_parents]) if rep_offsets[-1] else np.empty(0, np.int64)
    tail = [int(x[0]) for x in rep_parents] if direct_tail is None else list(direct_tail)
    outmap = np.r_[retained, np.asarray(tail, np.int64)]
    inv = np.full(len(parent_t), -1, np.int64)
    inv[retained] = np.arange(len(retained))
    return {
        "tetrahedra": out_t, "materials": out_m,
        "output_cell_to_input_cell_rows": outmap,
        "input_cell_to_output_cell_rows": inv,
        "retained_input_cell_rows": retained,
        "replacement_tetrahedra": replacements,
        "replacement_parent_offsets": rep_offsets,
        "replacement_parent_cell_rows": rep_rows,
    }


class CanonicalUnionTests(unittest.TestCase):
    def test_two_step_retained_and_full_replacement_composition(self):
        base_t = np.array([[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8]], np.int64)
        base_m = np.full(6, 7, np.int64)
        s1 = step(base_t, base_m, [0,2,3,5], [[1,2,3,4],[4,5,6,7]], [[1,4],[1,2,4]])
        p1 = {"tetrahedra":s1["tetrahedra"],"materials":s1["materials"],"points":9,"path":"p1","sha256":"x"}
        p1["union_offsets"], p1["union_rows"], _ = M._mapped_step_unions(
            {"tetrahedra":base_t,"materials":base_m,"points":9,"path":"base","sha256":"b","union_offsets":np.arange(7,dtype=np.int64),"union_rows":np.arange(6,dtype=np.int64)},
            p1, {"path":"p1"}, "s1", step_arrays=s1)
        s2 = step(p1["tetrahedra"],p1["materials"],[0,1,2,4],[[0,2,3,6],[4,5,7,8]],[[0,2],[3,4]])
        p2={"tetrahedra":s2["tetrahedra"],"materials":s2["materials"],"points":9,"path":"p2","sha256":"y"}
        off, rows, _=M._mapped_step_unions(p1,p2,{"path":"p2"},"s2",step_arrays=s2)
        got=[rows[off[i]:off[i+1]].tolist() for i in range(len(off)-1)]
        self.assertEqual(got,[[0],[2],[3],[1,4],[0,3],[1,4,5]])
        self.assertEqual(len(off),7)

    def test_identity_step_requires_exact_ordered_cell_domain(self):
        t=np.array([[0,1,2,3],[1,2,3,4]],np.int64);m=np.array([1,1])
        p={"tetrahedra":t,"materials":m,"points":5}
        c={"tetrahedra":t.copy(),"materials":m.copy(),"points":5}
        off,rows,_=M._identity_unions(p,c,"identity")
        self.assertTrue(np.array_equal(off,[0,1,2]))
        self.assertTrue(np.array_equal(rows,[0,1]))
        c["tetrahedra"][0]=[0,1,2,4]
        with self.assertRaises(ValueError):M._identity_unions(p,c,"bad identity")

    def test_representative_must_be_in_complete_parent_union(self):
        t=np.array([[0,1,2,3],[1,2,3,4]],np.int64);m=np.array([1,1])
        child={
          "tetrahedra":np.array([[0,1,2,3],[0,1,3,4]],np.int64),"materials":np.array([1,1]),
          "points":5,"path":"child","sha256":"x",
          "output_cell_to_input_cell_rows":np.array([0,1]),
          "input_cell_to_output_cell_rows":np.array([0,-1]),
          "retained_input_cell_rows":np.array([0]),
          "replacement_tetrahedra":np.array([[0,1,3,4]]),
          "replacement_parent_offsets":np.array([0,1]),
          "replacement_parent_cell_rows":np.array([0])}
        parent={"tetrahedra":t,"materials":m,"points":5,"path":"parent","sha256":"p",
                "union_offsets":np.array([0,1,2]),"union_rows":np.array([0,1])}
        # Direct representative 1 is absent from CSR [0].
        with self.assertRaises(ValueError):M._mapped_step_unions(parent,child,{"path":"child"},"bad map",step_arrays=child)

    def test_current_domain_selects_singular_when_plural_is_stale(self):
        parent_t=np.array([[0,1,2,3],[1,2,3,4]],np.int64);parent_m=np.array([1,1])
        data=step(parent_t,parent_m,[0],[[0,2,3,4]],[[1]])
        # The old plural lane has the correct output length but its reverse lane is
        # from an older parent domain. The current singular pair is coherent.
        data["output_cell_to_input_cell"]=data["output_cell_to_input_cell_rows"].copy()
        data["input_cell_to_output_cell"]=data["input_cell_to_output_cell_rows"].copy()
        data["input_cell_to_output_cell_rows"]=np.array([0,-1,-1],np.int64)
        p={"tetrahedra":parent_t,"materials":parent_m,"points":5,"path":"p","sha256":"p",
           "union_offsets":np.array([0,1,2]),"union_rows":np.array([0,1])}
        c={"tetrahedra":data["tetrahedra"],"materials":data["materials"],"points":5,"path":"c","sha256":"c"}
        off,rows,detail=M._mapped_step_unions(p,c,{"path":"c"},"stale plural",step_arrays=data)
        self.assertEqual(detail["selected_map_lane"],"output_cell_to_input_cell/input_cell_to_output_cell")
        self.assertEqual([rows[off[i]:off[i+1]].tolist() for i in range(2)],[[0],[1]])

    def test_bad_point_map_domain_rejected(self):
        parent_t=np.array([[0,1,2,3]],np.int64);parent_m=np.array([1])
        data=step(parent_t,parent_m,[0],[],[])
        data["input_node_to_output_node_rows"]=np.array([0,1,2,4],np.int64)
        p={"tetrahedra":parent_t,"materials":parent_m,"points":4,"path":"p","sha256":"p",
           "union_offsets":np.array([0,1]),"union_rows":np.array([0])}
        c={"tetrahedra":data["tetrahedra"],"materials":data["materials"],"points":4,"path":"c","sha256":"c"}
        # Point lanes are read from an NPZ, so a malformed temporary bundle is used.
        with tempfile.TemporaryDirectory() as d:
            q=Path(d)/"bad.npz"
            np.savez(q,**data)
            with self.assertRaises(ValueError):
                M._mapped_step_unions(p,c,{"path":str(q)},"bad point map")

    def test_serialization_preserves_old_array_payloads(self):
        t=np.array([[0,1,2,3]],np.int64)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/"in.npz";dst=root/"out.npz"
            old=np.arange(12,dtype=np.float32).reshape(4,3)
            np.savez_compressed(src,points_world_m=old,tetrahedra=t,material_indices=np.array([1]),history=np.arange(17,dtype=np.int64),
                                canonical_parent_union_operation_order=np.asarray("old-order"))
            offsets=np.array([0,1],np.int64);rows=np.array([0],np.int64)
            M.write_npz_with_unions(src,dst,offsets,rows,{
                "operation_order":"base824->current",
                "base_state_path":"base-state","base_state_sha256":"b",
                "base_volume_path":"base.npz","base_volume_sha256":"bv",
                "current_state_path":"current-state","current_state_sha256":"c",
                "current_volume_path":"current.npz","current_volume_sha256":"cv"})
            with np.load(src,allow_pickle=False) as a,np.load(dst,allow_pickle=False) as b:
                self.assertTrue(np.array_equal(a["history"],b["history"]))
                self.assertTrue(np.array_equal(a["points_world_m"],b["points_world_m"]))
                self.assertEqual(str(b["canonical_parent_union_operation_order"].item()),"base824->current")
                self.assertTrue(np.array_equal(b["canonical_parent_union_parent_rows"],rows))


if __name__ == "__main__":
    unittest.main()
