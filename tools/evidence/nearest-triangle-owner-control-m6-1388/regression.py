#!/usr/bin/env python3
# Check conservative broadphase against the existing full-distance owner.
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--owner-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(), 'retain prior reports'
    sys.path.insert(0,str(args.owner_source))
    from numilab_human import myosim_bone_proximity as bp
    helper=Path(__file__).with_name('bounded_triangle_nearest_052.py')
    spec=importlib.util.spec_from_file_location('bounded_nearest_control',helper)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    distance=lambda p,t:bp._point_triangle_distances_squared(p,t,np)
    pins={str(p):sha(p) for p in [helper,Path(__file__),Path(bp.__file__)]}
    rng=np.random.default_rng(1388)
    random_triangles=rng.normal(0,0.1,(2304,3,3))
    far=rng.uniform(2,4,(2304,1,3))+rng.normal(0,0.001,(2304,3,3))
    # The closest triangle has a distant centroid. A nearest-centroid-only
    # shortlist cannot find it; the enclosing-radius broadphase must.
    far[1000]=np.array([[0.,0.,0.],[100.,0.,0.],[0.,100.,0.]])
    duplicate=np.tile(np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]),(2304,1,1))
    cases=[
        ('long_triangle_distant_centroid',far,np.array([[.001,.001,.0001],[0.,0.,0.],[20.,20.,.0002]])),
        ('coincident_triangle_global_tie',duplicate,np.array([[.2,.2,.01],[0.,0.,0.],[.5,0.,0.]])),
        ('random_static_target',random_triangles,rng.normal(0,.2,(96,3))),
        ('small_full_scan_fallback',random_triangles[:440],rng.normal(0,.2,(24,3))),
    ]
    rows=[]
    for name,triangles,points in cases:
        idx=module.TriangleNearestIndex(triangles,distance)
        if name=='small_full_scan_fallback':
            assert idx.tree is None
        for point in points:
            values=distance(point,triangles)
            owner_i=int(np.argmin(values))
            assert idx.query(point)==(owner_i,float(values[owner_i])), (name,point.tolist())
        if name=='long_triangle_distant_centroid':
            initial=np.atleast_1d(idx.tree.query(points[0],k=16)[1])
            assert 1000 not in initial
            assert idx.query(points[0])[0]==1000
        if name=='coincident_triangle_global_tie':
            assert all(idx.query(point)[0]==0 for point in points)
        rows.append({'case':name,'target_faces':len(triangles),'queries':len(points),'face_and_distance_mismatches':0})
    assert all(sha(p)==h for p,h in pins.items())
    report={'status':'conservative_nearest_regression_pass','rows':rows,'input_sha256':pins,'gpu_used':False,'scope':'Offline exact-owner nearest-query selection, not physical simulation or anatomy admission.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'queries':sum(r['queries'] for r in rows),'report_sha256':sha(args.output)}))
if __name__=='__main__':
    main()
