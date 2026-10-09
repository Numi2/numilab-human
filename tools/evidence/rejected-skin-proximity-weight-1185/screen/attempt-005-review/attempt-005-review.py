#!/usr/bin/env python3
"""Read-only anatomical localization of exact candidate skin self-pairs."""
import importlib.util,json,sys,hashlib,math
from pathlib import Path
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005')
HERE=E/'native-common-skin-bone-proximity-saved-pose-screen-1185'
RUN=HERE/'attempt-005'
OUT=HERE/'attempt-005-review'
CAND=E/'native-common-skin-bone-proximity-weight-candidate-1185/candidate-build-008/compact-2.5x-full.nhskin'
PUB=Path('/Users/n/numi-human-free-apex-publication-1159/src')
ROOT=Path('/Users/n/numi-human-resting-final-integration-001/src')
sys.path.insert(0,str(ROOT))
import numilab_human.common_atlas_skin_clearance as clearance

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
scr=load(HERE/'attempt-005.py','screen005_review')
source=scr.parse_skin(scr.SOURCE_SKIN); cand=scr.parse_skin(CAND)
step=47519; pack=scr.SCENE/'accepted-geometry'/f'step-{step}.mrvpack'
keys=scr.load_targets(); positions,surfaces,_=clearance._pack_surfaces(pack,set(keys))
skin_faces=surfaces[(51007,1)]['faces']; base=int(skin_faces.min())
arr=np.load(RUN/'compact-2.5x-full'/f'step-{step}-changed-xyz.npz')
newpos=positions.copy(); ids=arr['source_vertex_ids'].astype(np.int64); newpos[base+ids]=arr['candidate_xyz_f32']
records=[json.loads(x) for x in (RUN/'compact-2.5x-full'/f'step-{step}-self-pairs.jsonl').open()]

def area(tri):return .5*float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0])))
def weights_summary(rows, vertexids):
    sums={}
    for vid in vertexids:
        w=rows[vid]
        for j,bind in enumerate(source['bindings']):
            if w[j]: sums[int(bind[0])]=sums.get(int(bind[0]),0.)+float(w[j])/3
    return {str(k):round(v,6) for k,v in sorted(sums.items(),key=lambda kv:-kv[1])[:6]}
def pt_m(x):return float(x['numerator'])/float(x['denominator'])/(1<<149)
items=[]
for rec in records:
    ra,rb=int(rec['skin_face_row_a']),int(rec['skin_face_row_b'])
    va=source['faces'][ra].astype(int);vb=source['faces'][rb].astype(int)
    xa=positions[base+va].astype(np.float64);xb=positions[base+vb].astype(np.float64)
    ya=newpos[base+va].astype(np.float64);yb=newpos[base+vb].astype(np.float64)
    ca,cb=xa.mean(axis=0),xb.mean(axis=0);da,db=ya.mean(axis=0),yb.mean(axis=0)
    sa,sb=source['xyz'][va].astype(np.float64),source['xyz'][vb].astype(np.float64)
    wa=weights_summary(source['weights'],va);wb=weights_summary(source['weights'],vb)
    ca2,cb2=weights_summary(cand['weights'],va),weights_summary(cand['weights'],vb)
    pts=np.array([[pt_m(c) for c in p] for p in rec['points_lattice_rational']],dtype=float)
    diam=float(np.max(np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2))) if len(pts)>1 else 0.0
    na=np.cross(xa[1]-xa[0],xa[2]-xa[0]);nb=np.cross(xb[1]-xb[0],xb[2]-xb[0])
    nca=np.cross(ya[1]-ya[0],ya[2]-ya[0]);ncb=np.cross(yb[1]-yb[0],yb[2]-yb[0])
    def cosine(x,y):
        q=float(np.linalg.norm(x)*np.linalg.norm(y));return float(np.dot(x,y)/q) if q else None
    items.append({'face_rows':[ra,rb],'source_vertex_ids':[va.tolist(),vb.tolist()],
      'shared_vertex_count':int(len(set(va)&set(vb))), 'rest_centroids_m':[sa.mean(axis=0).tolist(),sb.mean(axis=0).tolist()],
      'captured_centroid_separation_m':float(np.linalg.norm(ca-cb)), 'candidate_centroid_separation_m':float(np.linalg.norm(da-db)),
      'source_face_area_m2':[area(sa),area(sb)], 'captured_face_area_m2':[area(xa),area(xb)],'candidate_face_area_m2':[area(ya),area(yb)],
      'candidate_to_captured_area_ratio':[area(ya)/area(xa) if area(xa) else None,area(yb)/area(xb) if area(xb) else None],
      'normal_cosine_captured_to_candidate':[cosine(na,nca),cosine(nb,ncb)],
      'max_vertex_displacement_m':[float(np.linalg.norm(ya-xa,axis=1).max()),float(np.linalg.norm(yb-xb,axis=1).max())],
      'original_average_top_body_weights':[wa,wb],'candidate_average_top_body_weights':[ca2,cb2],
      'intersection_point_count':rec['intersection_point_count'],'intersection_diameter_m':diam,'intersection_points_m':pts.tolist()})
OUT.mkdir(parents=True,exist_ok=False)
report={'schema':'numi.human.skin-weight-candidate-self-pair-anatomical-review.v1','status':'read_only_diagnostic','candidate':{'path':str(CAND),'sha256':scr.sha(CAND)},'source_skin':{'path':str(scr.SOURCE_SKIN),'sha256':scr.sha(scr.SOURCE_SKIN)},'step':step,'self_intersection_pair_count':len(items),'faces':items,
'interpretation_limits':['Body-weight labels and rest-coordinate centroids localize but do not establish clinical anatomy.','Candidate pose points are CPU weight-delta predictions added to the actual accepted Float32 capture, not native Metal candidate capture.','Exact self-pair count is from the unchanged exact Float32-lattice predicate.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'out':str(OUT/'report.json'),'sha256':scr.sha(OUT/'report.json'),'self_pair_count':len(items),'groups':sorted({tuple(sorted(set(x['original_average_top_body_weights'][0])|set(x['original_average_top_body_weights'][1]))) for x in items})},sort_keys=True))
