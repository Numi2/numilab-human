"""Compile immutable cortical ownership and reserve empty GPU-discovered facet slots."""
from pathlib import Path
import json,sys
import numpy as np
from material_context import sha

def prepare(base,out):
    doc=json.loads((base/'manifest.json').read_text())
    if 'constitutive_boundary' not in doc or 'swept_finite_coverage' in doc:raise ValueError('Expected original certified pressure map')
    def read(name,typ,shape):return np.fromfile(base/(name+'.bin'),typ).reshape((-1,*shape))
    faces=read('closed-material-triangles','u4',(3,4))[:,0]
    triangles=read('cortical-certificate-faces','u4',(4,))
    original=read('shared-finite-faces','u4',(4,))
    pools=read('pool-material-points','f4',(7,4));spec=doc['closed_reverse']
    p=pools[spec['first']:spec['first']+spec['unique_vertices']]
    meta=np.ascontiguousarray(p[:,3]).view('u4');bank=np.ascontiguousarray(p[:,4]).view('u4')
    raw=read('cortical-triangles','f4',(10,4))
    ids=np.empty((len(triangles),4),'u4')
    for r in doc['cortical_certificate']['banks']:
        a=r['first'];b=a+r['count'];c=triangles[a:b];base_id=bank[c[0,0],0]
        ids[a:b,0]=np.arange(b-a)+base_id;ids[a:b,1]=r['cortical_body'];ids[a:b,2]=c[:,3];ids[a:b,3]=meta[c[:,0],1]
        if not np.array_equal(raw[ids[a:b,0],:3,:3],p[c[:,:3],2,:3]) or not np.all(meta[c[:,0],2]==r['cortical_body']):raise ValueError('Cortical ancestry mismatch')
    rows={(int(f[3]),tuple(sorted(map(int,f[:3])))):i for i,f in enumerate(faces)}
    if len(rows)!=len(faces):raise ValueError('Ambiguous regional surface face')
    known=np.asarray(sorted((int(u[3]),rows[int(triangles[u[3],3]),tuple(sorted(map(int,u[:3])))]) for u in original),'u4')
    if np.any(np.all(known[1:]==known[:-1],axis=1)):raise ValueError('Duplicate registered force pair')
    reserve=256; old=len(original); total=2*reserve
    modified={'pool-material-points','pool-material-regions','pelvic-finite-pairs','shared-finite-faces'}
    out.mkdir()
    for src in base.iterdir():
        if src.name!='manifest.json' and src.stem not in modified:(out/src.name).hardlink_to(src)
    def add(name,block):
        path=out/(name+'.bin')
        with path.open('wb') as f:
            src=base/path.name
            if src.exists():
                import shutil
                with src.open('rb') as inp:shutil.copyfileobj(inp,f)
            block.tofile(f)
        doc['files'][path.name]=sha(path)
    pairs=np.zeros((total,17,4),'f4');pairs[:,0,2].view('u4')[:]=np.arange(total)//reserve
    sites=np.zeros((total,16,7,4),'f4')
    regions=np.zeros((total,4),'u4');regions[:,0]=len(pools)+16*np.arange(total);regions[:,1]=16;regions[:,2]=256*(np.arange(total)//reserve)
    add('pelvic-finite-pairs',pairs);add('shared-finite-faces',np.zeros((total,4),'u4'))
    add('pool-material-points',sites);add('pool-material-regions',regions)
    add('motion-cortical-owners',ids);add('motion-registered-faces',known)
    doc['finite_facets']['number']+=total;doc['contact_pool']['count']+=16*total
    doc['motion_contact_registry']={'source_manifest':sha(base/'manifest.json'),'original_pairs':old,'slots_per_side':reserve,'observed_hit_capacity_per_side':256,'faces':len(triangles),'sorted_pairs':len(known)}
    (out/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
if __name__ == "__main__":
    import argparse
    from tempfile import TemporaryDirectory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--material-partition",type=Path)
    parser.add_argument("--registration-receipt",type=Path)
    args = parser.parse_args()
    if (args.material_partition is None) != (args.registration_receipt is None):
        parser.error("--material-partition and --registration-receipt must be specified together")
    if args.material_partition is None:
        prepare(args.base,args.out)
    else:
        from complete_surface_force_map import build
        with TemporaryDirectory(dir=args.out.parent) as scratch:
            registry = Path(scratch)/"registry"
            prepare(args.base,registry)
            build(args.material_partition,registry,args.registration_receipt,args.out)
