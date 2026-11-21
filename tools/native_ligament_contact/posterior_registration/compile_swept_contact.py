"""Add spatially discovered finite pressure pairs using the existing P1 force owner."""
from pathlib import Path
import json,sys
import numpy as np
from material_context import sha

def build(base,candidates,out):
    doc=json.loads((base/'manifest.json').read_text())
    if 'constitutive_boundary' not in doc:raise ValueError('Paired dual-precision boundary required')
    if doc['finite_facets'].get('rows_per_pair')!=16:raise ValueError('Unexpected feature layout')
    modified={'pelvic-finite-pairs.bin','pool-material-points.bin','pool-material-regions.bin','shared-finite-faces.bin'}
    out.mkdir(exist_ok=False)
    for p in base.iterdir():
        if p.name!='manifest.json' and p.name not in modified:(out/p.name).hardlink_to(p)
    def read(name,dtype,width):return np.fromfile(base/(name+'.bin'),dtype).reshape(-1,*width)
    pair=read('pelvic-finite-pairs','f4',(17,4));pool=read('pool-material-points','f4',(7,4))
    groups=read('pool-material-regions','u4',(4,));shared=read('shared-finite-faces','u4',(4,))
    surface=read('closed-material-triangles','u4',(3,4));shapes=read('closed-material-shapes','f4',(3,4))
    cortex=read('cortical-certificate-faces','u4',(4,));raw=read('cortical-triangles','f4',(10,4))
    r=np.load(candidates)
    if len(np.unique(r,axis=0))!=len(r) or not len(r):raise ValueError('Invalid finite pressure-pair queue')
    r=r[np.lexsort((r[:,1],r[:,0]))];b,m=r.T;n=len(b)
    first=doc['closed_reverse']['first'];points=pool[first:first+doc['closed_reverse']['unique_vertices']]
    meta=points[:,3].copy().view('u4');bank=points[:,4].copy().view('u4')
    side=cortex[b,3];donor=meta[cortex[b,0],1];body=meta[cortex[b,0],2]
    if not np.array_equal(side,surface[m,0,3]) or not np.array_equal(donor,np.where(side==0,131,145)):
        raise ValueError('Pressure pair crosses a material identity')
    firstFace={(x['muscle'],x['cortical_body']):x['first'] for x in doc['cortical_certificate']['banks']}
    rawface=np.array([int(k)-firstFace[29+int(s),int(owner)]+int(bank[cortex[k,0],0]) for k,s,owner in zip(b,side,body)],dtype='u4')
    if not np.array_equal(raw[rawface,:3,:3],points[cortex[b,:3],2,:3]):
        raise ValueError('Registered cortical face does not match the source')
    if np.max(abs(shapes[m].sum(axis=2)-1))>2e-6:raise ValueError('P1 shape rows do not sum to unity')
    block=np.zeros((n,17,4),'f4');block[:,0].view('u4')[:]=np.column_stack((rawface,body,side,m))
    block[:,1:4]=shapes[m];block[:,16].view('u4')[:]=surface[m,2]
    bindings=np.column_stack((surface[m,0,:3],b)).astype('u4')
    old={(int(row[3]),tuple(sorted(row[:3]))) for row in shared}
    if any((int(row[3]),tuple(sorted(row[:3]))) in old for row in bindings):raise ValueError('Duplicate registered pressure pair')
    sample=np.zeros((n,7,4),'f4');sample[:,3].view('u4')[:]=np.column_stack((np.full(n,128),donor,body,np.ones(n)))
    sample[:,4,0].view('u4')[:]=rawface;sample[:,5].view('u4')[:]=surface[m,2];sample[:,6]=shapes[m].mean(axis=1)
    starts=np.r_[0,np.flatnonzero(np.diff(b)!=0)+1];length=np.diff(np.r_[starts,n])
    regions=np.column_stack((len(pool)+16*starts,16*length,256*side[starts],np.zeros(len(starts)))).astype('u4')
    for name,previous,extra in [('pelvic-finite-pairs',pair,block),('pool-material-regions',groups,regions),
            ('shared-finite-faces',shared,bindings),('pool-material-points',pool,None)]:
        file=out/(name+'.bin')
        with file.open('wb') as f:
            previous.tofile(f)
            if extra is not None:extra.tofile(f)
            else:
                for i in range(0,n,256):np.repeat(sample[i:i+256],16,axis=0).tofile(f)
        doc['files'][file.name]=sha(file)
    doc['finite_facets']['number']+=n;doc['contact_pool']['count']+=16*n
    doc['swept_finite_coverage']={'source_manifest':sha(base/'manifest.json'),'candidate_rows':sha(candidates),
        'previous_pairs':len(pair),'added_pairs':n,'added_regions':len(regions),
        'ancestry':'native cortical row, material face, and existing four P1 material parents',
        'search':json.loads(candidates.with_suffix('.json').read_text())}
    (out/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
    return doc['swept_finite_coverage']
if __name__=='__main__':print(build(*map(Path,sys.argv[1:])))
