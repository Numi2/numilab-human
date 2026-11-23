"""Emit pinned reference/starting nodes and the conservative mass update through the current native format."""
import sys,json,os,subprocess
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from certify_transported_material import measure
def run(job,h,lib,field,credits,out):
    from layer_contacts import run as layers
    base=h/'closed-native/native';s=np.load(field/'candidate.npz');v=s['nodes'];y=s['positions'].astype(float);w=np.load(credits/'repaired-credits.npz')
    env=json.loads((job/'registered-donor-cook/scene/launch.json').read_bytes())['environment']
    c=json.loads((credits/'result.json').read_bytes())
    if c['geometry']!=digest(field/'candidate.npz') or c['updated']!=digest(credits/'repaired-credits.npz'):raise ValueError('Stale mass source')
    out.mkdir();pose=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']);raw=base/'mesh.node'
    e=np.loadtxt(base/'mesh.ele',skiprows=1,dtype='i4');t=e[:,1:5]-1;roles=e[:,5];z=np.load(job/'registered-boundary/registered-boundary.npz');groups=[z[k+'_node_joint'][z[k+'_face_nodes']] for k in 'OID']
    def load(path):return np.loadtxt(path,skiprows=1,usecols=(1,2,3),dtype='f4').astype(float)
    def emit(inp,dest):
        rows=dict(zip(v,y))
        with inp.open() as f,dest.open('x') as g:
            g.write(next(f))
            for i,line in enumerate(f):
                if i in rows:
                    fields=line.split();assert int(fields[0])==i+1 and len(fields)==4
                    line=str(i+1)+' '+ ' '.join(format(float(a),'.17g') for a in rows[i])+'\n'
                g.write(line)
    x=load(raw);p=load(pose);affected=np.unique(t[np.isin(t,v).any(1)])
    if not np.array_equal(p[affected],x[affected]):raise ValueError('Material and loaded pose differ in connected repair star')
    for key,src in [('mesh.node',raw),('current.node',pose)]:emit(src,out/key)
    for k in ['mesh.ele','material-source-indices.npz','current-child.labels.csv']:os.link(base/k,out/k)
    x=load(out/'mesh.node');p=load(out/'current.node')
    def jac(Y):
        d,l=measure(Y,t);return d,l
    dx,lx=jac(x);dp,lp=jac(p);J=dp/dx
    mass=np.fromfile(base/'cell-mass.f64','<f8');mass[np.flatnonzero(roles==16)[w['role_cells']]]=w['cell_mass'];mass.astype('<f8').tofile(out/'cell-mass.f64')
    nm=sum(np.bincount(t[:,k],weights=mass*.25,minlength=len(x)) for k in range(4));nm.astype('<f8').tofile(out/'node-mass.f64')
    if min(nm)<=0 or np.any(dx<=6e-18) or np.any(dp<=6e-18) or np.any(dx<1e-5*lx) or np.any(dp<1e-5*lp) or min(J)<.4 or max(J)>2.5:raise ValueError('Volume geometry rejected')
    args=json.loads((base/'result.json').read_bytes())['command'];args[1]=str(out)
    with (out/'native-import.json').open('w') as log,(out/'native-import.stderr').open('w') as err:
        native=subprocess.run(args,stdout=log,stderr=err)
    if native.returncode:raise ValueError('Native importer rejected repaired material')
    scans={}
    for name,Y in [('reference',x),('starting',p)]:
        dest=out/(name+'-layers.json');layers(Y,groups,dest,lib);scans[name]=json.loads(dest.read_bytes())['pair_counts']
    report=dict(native=json.loads((out/'native-import.json').read_bytes()),native_command=args,material=digest(out/'mesh.node'),initial=digest(out/'current.node'),cells=len(t),unchanged_topology=digest(out/'mesh.ele'),minimum_normalized_shape=[float(min(dx/lx)),float(min(dp/lp))],loaded_jacobian_range=[float(min(J)),float(max(J))],layer_intersections=scans,source_mass=credits.name,cell_mass=digest(out/'cell-mass.f64'),node_mass=digest(out/'node-mass.f64'))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if any(np.array(v).any() for v in scans.values()):raise ValueError('Crossing found')
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
