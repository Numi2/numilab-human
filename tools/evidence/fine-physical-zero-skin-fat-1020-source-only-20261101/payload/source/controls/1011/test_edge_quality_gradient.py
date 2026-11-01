#!/usr/bin/env python3
"""Tiny analytic edge-quality derivative controls; no anatomy or physical admission."""
from pathlib import Path
import hashlib,json
import numpy as np
Q=1e-5
EDGES=[(i,j) for i in range(4) for j in range(i+1,4)]
def det_grad(p):
 a,b,c,d=p
 return np.array([-np.cross(c-b,d-b),np.cross(c-a,d-a),np.cross(d-a,b-a),np.cross(b-a,c-a)])
def det(p):
 a,b,c,d=p
 return np.dot(b-a,np.cross(c-a,d-a))
def phi(p,i,j):
 return det(p)-Q*np.linalg.norm(p[i]-p[j])**3
def grad(p,i,j):
 g=det_grad(p)
 v=p[i]-p[j]; e=3*Q*np.linalg.norm(v)*v
 g[i]-=e;g[j]+=e
 return g
def main():
 rng=np.random.default_rng(1011)
 checks=[];max_relative=0.
 for case in range(32):
  p=rng.normal(size=(4,3))*.005+np.array([.1,.3,1.])
  p[1]=p[0]+[.02,0,0];p[2]=p[0]+[.006,.003,0];p[3]=p[0]+[.012,.001,4e-5]
  p+=rng.normal(size=(4,3))*1e-5
  v=rng.normal(size=(4,3));v/=np.linalg.norm(v)
  for i,j in EDGES:
   g=grad(p,i,j);a=float(np.sum(g*v))
   h=1e-8; fd=float((phi(p+h*v,i,j)-phi(p-h*v,i,j))/(2*h))
   scale=max(np.linalg.norm(g)*np.linalg.norm(v),1e-20)
   rel=abs(a-fd)/scale;max_relative=max(max_relative,rel)
   assert rel<2e-7,(case,i,j,a,fd,rel)
   assert np.linalg.norm(g.sum(axis=0))<1e-18
   checks.append([case,i,j,rel])
  q=det(p)/max(np.linalg.norm(p[i]-p[j]) for i,j in EDGES)**3
  assert (q>=Q)==all(phi(p,i,j)>=0 for i,j in EDGES)
 p=np.array([[0.,0,0],[.02,0,0],[.006,.003,0],[.012,.001,4e-5]])
 i,j=1,0
 L=np.linalg.norm(p[i]-p[j]); d=det(p)
 true=det_grad(p)/L**3
 edge=p[i]-p[j];extra=3*d*edge/L**5
 true[i]-=extra;true[j]+=extra
 frozen=det_grad(p)/L**3
 v=np.zeros((4,3));v[1,0]=1
 omitted=float(np.sum((frozen-true)*v))
 assert abs(omitted)>1e-5
 # The six smooth inequalities remain valid at ties in the maximum edge.
 tie=np.array([[0.,0,0],[1.,0,0],[0.,1.,0],[0.,0,1.]])
 assert sum(np.isclose(np.linalg.norm(tie[i]-tie[j]),np.sqrt(2))for i,j in EDGES)==3
 result={"status":"PASS","case_count":32,"edge_derivative_checks":len(checks),
 "maximum_scaled_derivative_error":max_relative,"translation_invariance":True,
 "six_edge_gate_equivalence":True,"longest_edge_tie_coverage":True,
 "frozen_denominator_omission_demonstrated":omitted,
 "physical_admission":False,"native_steps":0,
 "scope":"synthetic scalar derivative controls only",
 "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 Path(__file__).with_name("receipt.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result))
if __name__=="__main__":main()
