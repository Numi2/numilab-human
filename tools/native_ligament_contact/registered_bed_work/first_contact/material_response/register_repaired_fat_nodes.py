"""Reexpress the coupled repaired nodes in their actual source triangle, donor and volume owner frames."""
import sys,json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from bind_material_interior import HistoricalTissue
from register_material_surfaces import area_normals

def run(job,native,field,out):
    env=json.loads((job/'registered-donor-cook/scene/launch.json').read_bytes())['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def source(k):
        p=Path(env[prefix+k+'_CSV'])
        if digest(p)!=env[prefix+k+'_SHA256']:raise ValueError(k+' changed')
        return p
    stats={};bindings={};hist=HistoricalTissue(env);s=np.load(field/'candidate.npz');moved=s['nodes'];x=np.loadtxt(native/'mesh.node',skiprows=1,usecols=(1,2,3))
    fset=np.load(job/'registered-boundary/registered-boundary.npz');nodes=fset['D_node_joint'];faces=fset['D_face_nodes'];idx=np.flatnonzero(np.isin(nodes,moved));active=np.unique(faces[np.isin(faces,idx).any(1)]);mov={int(nodes[q]):int(q) for q in idx};assert len(mov)==len(moved) and min(idx)>=len(fset['D_retained_source_node'])
    out.mkdir();newnorm=area_normals(x[nodes],faces);newnorm/=np.linalg.norm(newnorm,axis=1)[:,None];oldrows=np.loadtxt(source('SURFACE_D_NODES'),skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7));sign=np.sign(np.sum(newnorm[active]*oldrows[active,3:],axis=1))
    if np.any(sign==0):raise ValueError('Undefined registration normal')
    newnorm[active]*=sign[:,None];old=oldrows[:,:3];N=set(active);change=set(idx);sr={};P={};fmt=lambda a:[format(float(v),'.17g') for v in a]
    dom=Path(env[prefix+'ROLE_D_SOURCE_ASSET']);assert digest(dom)==env[prefix+'ROLE_D_SOURCE_ASSET_SHA256'];d=json.loads(dom.read_bytes());vv=np.array([r['current_world_m'] for r in d['vertices']]);ff=np.array([r['surface_vertex_rows'] for r in d['faces']])
    fk='SURFACE_D_NODES';dest=out/'D-nodes.csv'
    with source(fk).open() as fi,dest.open('x') as fo:
        fo.write(next(fi))
        for j,line in enumerate(fi):
            if j not in N:fo.write(line);continue
            a=line.rstrip().split(',');a[5:8]=fmt(newnorm[j])
            if j in change:
                n=nodes[j];delta=x[n]-old[j];b=np.array(a[23:26],float);base=b@vv[ff[int(a[22])]];res=x[n]-base;matrix=np.array(a[11:20],float).reshape(3,3)
                if np.linalg.norm(base+np.array(a[26:29],float)-old[j])>1e-9:raise ValueError('Inconsistent actual material source')
                loc=np.array(a[8:11],float)+matrix@delta;a[2:5]=fmt(x[n]);a[8:11]=fmt(loc);a[26:29]=fmt(res);sr[int(n)]=(a[8:11],a[26:29])
            fo.write(','.join(a)+'\n')
    def record(k,p):bindings.update({prefix+k+'_CSV':str(p),prefix+k+'_SHA256':digest(p)})
    record(fk,dest)
    k='COMBINED_NODE_MAP';dest=out/'combined-node-map.csv'
    with source(k).open() as fi,dest.open('x') as fo:
        fo.write(next(fi))
        for line in fi:
            n=int(line[:line.index(',')])-1
            if n not in mov:fo.write(line);continue
            a=line.rstrip().split(',');j=mov[n];assert a[8]=='D' and int(a[7])==j and np.array_equal(np.array(a[9:12],float),old[j]) and a[28]=='source_parent_tet_reference_registration'
            q=int(a[29]);key=tuple(hist.labels[q]);hist.groups[key]=[q];which,bc,ties=hist.project(x[n],key);assert which==q
            res=x[n]-bc@hist.points[hist.tet[q]];a[9:12]=fmt(x[n]);a[14:17],a[24:27]=sr[n];a[30:34]=fmt(bc);a[34:37]=fmt(res);a[37:39]=[*fmt([np.linalg.norm(res)]),str(ties)]
            if np.count_nonzero(np.array(a[44:48],float))!=1 or any(float(v) for v in a[48:52]):raise ValueError('Operational material corner changed')
            P[j]=(a[30:39],a[24:27]);fo.write(','.join(a)+'\n')
    assert len(P)==len(moved);record(k,dest)
    k='NODE_REGISTRATION';dest=out/'node-registration.csv'
    with source(k).open() as fi,dest.open('x') as fo:
        fo.write(next(fi));seen=set()
        for line in fi:
            if line[0]!='D' or int(line.split(',',2)[1]) not in P:fo.write(line);continue
            a=line.rstrip().split(',');j=int(a[1]);a[4:13]=P[j][0];a[20:23]=P[j][1];fo.write(','.join(a)+'\n');seen.add(j)
    assert seen==set(P);record(k,dest)
    (out/'result.json').write_text(json.dumps(dict(environment=bindings,moved=len(P),normals=len(N),material=digest(native/'mesh.node')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
