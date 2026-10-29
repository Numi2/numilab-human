#!/usr/bin/env python3
"""Check the exact shared I interface and paired rims of closed fine skin/fat."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
sys.path.insert(0,str(P.parent/'authoring-deps'))
import manifold3d as mf
OUT=Path(__file__).resolve().parent/'output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False)
 fat=P/'paired-fine-headwall-rim-635/run-010/output/fine-fat-closed-boundary.npz'
 skin=P/'paired-fine-headwall-rim-635/run-013/output/fine-skin-closed-boundary.npz'
 sr=P/'paired-fine-headwall-rim-635/run-013/output/report.json'
 fr=P/'paired-fine-headwall-rim-635/run-010/output/report.json'
 pins={fat:'b02a064f088e625dfb11c98fd2859e13bd271224bd13d050c7b848c217d856b2',skin:'352dc5e1ebfeca7f8c7dc13075dbcf0abc8d15a842f52057ba7744f998d5fa23',sr:'24ed76b7725d1b097e92d1859662b9725532e651abea5052f42dc0f14fb87b8c',fr:'65ca93243fbec108e48ee8151f80994721ee65f3e689d7f8fdaf141388eb9e75'}
 for p,d in pins.items():require(sha(p)==d,'pinned source changed: '+str(p))
 fs,ss=load(fat),load(skin);rd=json.loads(sr.read_text())
 O=Path(rd['fine_surfaces']['O']['path']);I=Path(rd['fine_surfaces']['I']['path'])
 fd=json.loads(fr.read_text());D=Path(fd['fine_surfaces']['D']['path'])
 for p,expected in ((O,rd['fine_surfaces']['O']['sha256']),(I,rd['fine_surfaces']['I']['sha256']),(D,fd['fine_surfaces']['D']['sha256'])):require(sha(p)==expected,'source changed')
 o,i,d=load(O),load(I),load(D)
 op,ip,dp=o['points_world_m'],i['points_world_m'],d['points_world_m']
 iff=i['faces'];no=len(op);ni=len(ip);nif=len(iff);nof=len(o['faces'])
 require(np.array_equal(fs['points_world_m_f64'][fs['faces'][:nif]],ip[iff]),'fat I interface differs from pinned I')
 require(np.array_equal(ss['points_world_m_f64'][ss['faces'][nof:nof+nif]],ip[iff[:,[0,2,1]]]),'skin I interface differs from reversed pinned I')
 whole=np.concatenate((op,ip,dp)).astype(np.float32)
 skinrim=ss['oriented_rim_faces'].astype(np.int64)
 fatrim=fs['oriented_rim_faces'].astype(np.int64)+no
 loader=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py'
 pred=module('joint652loader',loader).load_exact_owner()
 require(sha(pred.__file__)=='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d','exact owner changed')
 a,_=pred.float32_surface_records(whole,skinrim,source_seam_groups=())
 b,_=pred.float32_surface_records(whole,fatrim,source_seam_groups=())
 # Direct bipartite enumeration intentionally has no self-pair face-ID ordering
 # filter; same_surface enables only true common-index edge/vertex interfaces.
 index=pred._prepare_surface_aabb(b)
 exact=pred._audit_record_pairs(pred._query_surface_aabb(a,index),same_surface=True)
 stage={'status':'PASS_EXACT_CROSS_RIM_INTERFACE' if exact['count']==0 else 'FAIL_EXACT_CROSS_RIM_INTERFACE','skin_fat_shared_I_points':ni,'shared_I_faces':nif,'I_coordinates_bit_identical':True,'I_faces_exactly_opposite':True,'cross_rim':exact,'physical_admission':False,'runtime_created':0,'accepted_steps':0}
 (OUT/'exact-stage.json').write_text(json.dumps(stage,indent=2)+'\n')
 require(exact['count']==0,'skin/fat headwall rim intersection')
 material=[]
 for name,z in (('skin',ss),('fat',fs)):
  m=mf.Manifold(mf.Mesh64(vert_properties=np.ascontiguousarray(z['points_world_m_f64']),tri_verts=np.ascontiguousarray(z['faces'],dtype=np.uint64)))
  require(m.status()==mf.Error.NoError,'invalid '+name+' solid')
  material.append(m)
 inter=material[0]^material[1]
 require(inter.status()==mf.Error.NoError,'skin/fat intersection invalid')
 intersection=float(inter.volume())
 require(intersection==0.0,'skin/fat material overlap: '+str(intersection))
 union=material[0]+material[1]
 require(union.status()==mf.Error.NoError,'skin/fat union invalid')
 result={**stage,'status':'PASS_SHARED_INTERFACE_AND_NO_MATERIAL_OVERLAP','input_pins':{str(p):h for p,h in pins.items()},'exact_owner_sha256':sha(pred.__file__),'skin_volume_m3':float(material[0].volume()),'fat_volume_m3':float(material[1].volume()),'intersection_volume_m3':intersection,'union_volume_m3':float(union.volume()),'union_minus_sum_m3':float(union.volume()-sum(m.volume() for m in material)),'elapsed_s':time.monotonic()-start,'scope':'Source-only joint material interface; shared I is the intended skin/fat interface. This does not qualify execution cells, transfer, thickness, mass mapping, contacts or motion.'}
 (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
