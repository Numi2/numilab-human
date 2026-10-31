from pathlib import Path
import importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("quality_lp_629_test",HERE/"quality_lp_629.py")
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q);M=q._M
def controls():
 p=np.asarray([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[1e-8,1e-8,1e-8],[.03,0.,0.],[0.,.03,0.],[0.,0.,.03]],np.float32)
 t=M.orient(p,np.asarray([[0,1,2,3],[3,4,5,6]],np.int64));mat=np.asarray([15,16],np.int32)
 free=np.zeros(len(p),bool);free[3]=True;p0=p.copy();t0=t.copy()
 d0,q0=M.measurements(p,t);good0=M.passes(d0,q0);assert good0.tolist()==[False,True],(d0,q0,good0)
 g=q.determinant_gradients(p,t[:1])[0];eps=1.e-6
 for c in range(4):
  for ax in range(3):
   x=p.copy();x[t[0,c],ax]+=np.float32(eps);dp=M.measurements(x,t[:1])[0][0]
   x[t[0,c],ax]-=np.float32(2*eps);dm=M.measurements(x,t[:1])[0][0];fd=(dp-dm)/(2*eps)
   assert abs(fd-g[c,ax])<=max(1.e-14,abs(g[c,ax])*2.e-3),(c,ax,fd,g[c,ax])
 out,r=q.propose_quality_repair(p,t,mat,free,2.e-5,time_limit_s=2.)
 assert r["status"]=="PASS_F32_COUPLED_QUALITY_PROPOSAL",r
 d1,q1=M.measurements(out,t);good1=M.passes(d1,q1);assert good1.tolist()==[True,True],(d1,q1,good1)
 assert np.array_equal(t,t0) and np.array_equal(out[~free].view(np.uint32),p0[~free].view(np.uint32))
 assert r["good_cell_regressions"]==0 and r["material_cell_counts"]=={"15":1,"16":1}
 assert r["point_rows_changed"]==[3] and r["exact_F32_positive_cells"]==2
 no,e=q.propose_quality_repair(p,t,mat,np.zeros_like(free),2.e-5,time_limit_s=2.)
 assert e["status"]=="NO_AFFECTED_CELLS" and np.array_equal(no.view(np.uint32),p.view(np.uint32))
 return {"schema":"numi.human.coupled-quality-lp-629-controls.v1","status":"PASS_SMALL_COUPLED_FIXED_TOPOLOGY_CONTROLS",
         "physical_admission":False,"native_steps":0,
         "controls":["analytic_det6_gradient_finite_difference","thin_bad_cell_repaired_Float32",
                     "good_cell_inequality_prevents_regression","shared_material_point_uses_one_global_variable",
                     "protected_rows_bitwise_unchanged","topology_and_material_rows_unchanged",
                     "exact_Float32_positive_determinants","empty_free_mask_fails_closed"],
         "proposal":r,"determinants_before":d0.tolist(),"normalized_determinants_before":q0.tolist(),
         "determinants_after":d1.tolist(),"normalized_determinants_after":q1.tolist(),"materials":mat.tolist()}
if __name__=="__main__":
 r=controls();(HERE/"controls.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":r["status"],"bad_before":r["proposal"]["bad_before"],
                   "bad_after":r["proposal"]["bad_after"],"changed_rows":r["proposal"]["point_rows_changed"]},sort_keys=True))

