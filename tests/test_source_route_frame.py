import importlib.util, math, sys, unittest
from pathlib import Path
s=importlib.util.spec_from_file_location("source_frame_model",Path(__file__).resolve().parents[1]/"src/numilab_human/model.py")
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
G=[[.001,0.,0.,.1],[0.,.001,0.,.2],[0.,0.,.001,.3],[0.,0.,0.,1.]]
def A(t=(0,0,0),scale=.001):
    return [[float(v) for v in r] for r in [[scale,0,0,t[0]],[0,scale,0,t[1]],[0,0,scale,t[2]],[0,0,0,1]]]
def body(pos=(0,0,0),q=(0,0,0,1)):
    return dict(default_com_position_world_m=list(map(float,pos)),default_inertial_quaternion_world_xyzw=list(map(float,q)))
def apply(M,p):return [sum(M[i][j]*p[j] for j in range(3))+M[i][3] for i in range(3)]
class Tests(unittest.TestCase):
    def test_pullback_roundtrip_distinct_bone_frames(self):
        # Source point [2,3,4]mm -> local [1.004,2.006,3.008]m.
        # Body quarter turn about z and translation -> world [3-.006,6+.004,9+.008].
        p=[2.994,6.004,9.008]
        nodes=[dict(body="finger4",world_m=p,site="kept")]
        out=m._bodyparts_myosim_route_points_in_source_registration(nodes,{"finger4":A((1,2,3),.002)},{"finger4":body((5,5,6),(0,0,math.sqrt(.5),math.sqrt(.5)))},G,"test")
        for actual,expected in zip(out[0]["source_registration_query_world_m"],[.102,.203,.304]):self.assertAlmostEqual(actual,expected,places=12)
        self.assertEqual(out[0]["site"],"kept")
        self.assertEqual(out[0]["world_m"],p)
        self.assertEqual(nodes[0]["world_m"],p)
    def test_recovers_digit_branch_from_inconsistent_global_query(self):
        nodes=[dict(body="finger2",world_m=[0.,0.,0.]),dict(body="finger4",world_m=[.11,0.,0.])]
        glob=A()
        # Source branch at [0,0,0]; finger4 anchor moves it 110mm in target.
        transforms={"finger2":A((-.02,0,0)),"finger4":A((.11,0,0))}
        bodies={n:body() for n in transforms}
        old=m._bodyparts_myosim_route_weights_for_world_point([0,0,0],list(transforms),nodes,"old")
        self.assertEqual(old[0],[1.,0.])
        pulled=m._bodyparts_myosim_route_points_in_source_registration(nodes,transforms,bodies,glob,"test")
        new=m._bodyparts_myosim_route_weights_for_world_point([0,0,0],list(transforms),pulled,"new")
        self.assertEqual(new[0],[0.,1.])
    def test_equivalent_registration_keeps_weights(self):
        # If every anchor emits exactly G at this pose, nothing changes.
        nodes=[dict(body="a",world_m=[.111,.212,.313]),dict(body="b",world_m=[.12,.23,.34])]
        ts={n:G for n in ("a","b")};bs={n:body() for n in ts}
        pulled=m._bodyparts_myosim_route_points_in_source_registration(nodes,ts,bs,G,"same")
        self.assertEqual([p["source_registration_query_world_m"] for p in pulled],[p["world_m"] for p in nodes])
        query=[.112,.214,.316]
        self.assertEqual(m._bodyparts_myosim_route_weights_for_world_point(query,list(ts),nodes,"a"),m._bodyparts_myosim_route_weights_for_world_point(query,list(ts),pulled,"b"))
    def test_missing_anchor_fails_closed(self):
        with self.assertRaises(m.ImportError):
            m._bodyparts_myosim_route_points_in_source_registration([dict(body="a",world_m=[0.,0.,0.])],{},{"a":body()},G,"test")
    def test_bad_registration_rejected(self):
        for bad in (A(scale=0),A(scale=-.001),[[1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]]):
            with self.subTest(bad=bad),self.assertRaises(m.ImportError):
                m._bodyparts_myosim_route_points_in_source_registration([dict(body="a",world_m=[0.,0.,0.])],{"a":bad},{"a":body()},G,"bad")
    def test_nonfinite_node_and_empty_routes_rejected(self):
        for nodes in ([],[dict(body="a",world_m=[float("nan"),0,0])],[dict(body="a",world_m=[0,0])]):
            with self.subTest(nodes=nodes),self.assertRaises(m.ImportError):
                m._bodyparts_myosim_route_points_in_source_registration(nodes,{"a":A()},{"a":body()},G,"bad")
    def test_invalid_source_query_rejected(self):
        with self.assertRaises(m.ImportError):
            m._bodyparts_myosim_route_weights_for_world_point(
                [0.,0.,0.],["a"],[dict(body="a",world_m=[0.,0.,0.],
                source_registration_query_world_m=[float("nan"),0.,0.])],"bad")
    def test_refinement_preserves_force_nodes_and_queries_source_metric(self):
        vs=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,-1.,0.]]
        ns=[[0.,0.,1.] for _ in vs]
        routes=[dict(body="a",world_m=[100.,0.,0.],source_registration_query_world_m=[0.,0.,0.]),
                dict(body="b",world_m=[110.,0.,0.],source_registration_query_world_m=[1.,0.,0.])]
        out=m._bodyparts_refine_conforming_route_edge(
            1,(0,1),vs,ns,[v[:] for v in vs],[[1.,0.],[0.,1.],[1.,0.],[0.,1.]],
            [[0,1,2],[1,0,3]],A(),["a","b"],routes)
        self.assertEqual(out[3][-1],[.5,.5])
        self.assertEqual(out[5]["authored_route_points"],routes)
if __name__=="__main__":unittest.main()
