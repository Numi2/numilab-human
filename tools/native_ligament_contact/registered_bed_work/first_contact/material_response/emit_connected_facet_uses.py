"""Regenerate registered face-use ancestry for both shared material sides and historical head walls."""
import csv,json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from classify_connected_boundary import pack

def run(h,job,registration,out):
    out.mkdir();env=json.loads((job/'posed-app/scene/launch.json').read_bytes())['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    source=Path(env[pre+'FACET_ANCESTRY_CSV'])
    if digest(source)!=env[pre+'FACET_ANCESTRY_SHA256']:raise ValueError('Unpinned source ancestry')
    with np.load(job/'registered-boundary/registered-boundary.npz') as arrays:
        src={k:arrays[k] for k in arrays.files if k.endswith(('_global_facet','_retained_source_face','_new_face_sample_source'))}
    parent=np.load(h/'scene-registration/node-map.npz')['source_to_emitted']
    b=np.load(job/'boundaries/faces.npz');facet=b['face'];cell=b['cell'];kind=b['inherited_kind'];sides=b['side'];nodes=facet
    t=np.loadtxt(h/'closed-native/native/mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');fats=np.flatnonzero(t[:,4]==16);t=t[:,:4]-1
    ids=src['I_global_facet'];k=pack(nodes[ids]);order=np.argsort(k);keys=k[order];pair=np.full(len(ids),-1,'i4');pair_side=np.full(len(ids),-1,'i4')
    corners=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    for s,ijk in enumerate(corners):
        kk=pack(t[fats[:,None],ijk]);i=np.searchsorted(keys,kk);valid=(i<len(keys))&(keys[np.minimum(i,len(keys)-1)]==kk)
        dest=order[i[valid]]
        if len(np.unique(dest))!=len(dest) or np.any(pair[dest]>=0):raise ValueError('Multiple fat neighbors')
        pair[dest]=fats[valid];pair_side[dest]=s
    if np.any(pair<0):raise ValueError('Unpaired skin/fat interface')
    original={};remaining={};need={};head=kind==3;headkey=pack(nodes[head]);head_global=np.flatnonzero(head)
    for k in 'OID':
        retained=src[k+'_retained_source_face'];rd=np.full(retained.max()+1,-1,'i4');rd[retained]=np.arange(len(retained));remaining[k]=rd
        required=set(src[k+'_new_face_sample_source'].ravel().tolist()) if k!='O' else set()
        for part in (('skin','fat') if k=='I' else ('skin',) if k=='O' else ('fat',)):
            original[part,k]={};need[part,k]=required
    counts={'skin':0,'fat':0};heads=[];rows=0;new_rows=0;path=out/'facet-ancestry-current.csv'
    with source.open() as inp,path.open('x') as dest:
        reader=csv.reader(inp);writer=csv.writer(dest,lineterminator='\n');writer.writerow(next(reader))
        def emit(r,ids,child):
            nonlocal rows
            if not set(ids).issubset(t[child]):raise ValueError('Invalid oriented child use')
            part=r[0];r[1]=counts[part];counts[part]+=1;r[17]=child+1;r[18]=child;r[19:22]=[int(v)+1 for v in ids]
            writer.writerow(r);rows+=1
        for rr in reader:
            pr,rrname=rr[0],rr[3]
            if 'headwall' in rrname:
                n=parent[np.array(rr[19:22],int)-1];kk=pack(n.reshape(1,3))[0];idx=np.searchsorted(headkey,kk)
                if idx>=len(headkey) or headkey[idx]!=kk:raise ValueError('Lost authored head wall')
                current=int(cell[head_global[idx]])
                if rr[16]=='[-1,-1,-1]':rr[23]=json.dumps([current])
                heads.append((rr,n,current));continue
            key=rr[7];old=int(rr[4]);rid=remaining[key];i=int(rid[old]) if old<len(rid) else -1
            if old in need[pr,key]:original[pr,key][old]=rr.copy()
            if i<0:continue
            j=src[key+'_global_facet'][i];a=parent[np.array(rr[19:22],int)-1]
            if not np.array_equal(np.sort(a),np.sort(nodes[j])):raise ValueError('Immutable retained face nodes changed')
            rr[4]=i
            emit(rr,a,int(pair[i] if pr=='fat' and key=='I' else cell[j]))
        for key in 'ID':
            ids=src[key+'_global_facet'];cut=len(src[key+'_retained_source_face']);sample=src[key+'_new_face_sample_source']
            for part in (('skin','fat') if key=='I' else ('fat',)):
                for a,parents in enumerate(sample):
                    i=cut+a;parent=int(parents[3]);records=original[part,key];r=records[parent].copy();j=ids[i]
                    f=int(pair[i] if part=='fat' and key=='I' else cell[j])
                    ij=corners[pair_side[i] if part=='fat' and key=='I' else sides[j]]
                    vv=t[f,ij];used=sorted(set(map(int,parents)))
                    if set(vv)!=set(nodes[j]):raise ValueError('New child face mismatch')
                    # Preserve the source-cell and stable face of the centroid and retain
                    # all distinct source faces supporting the corner and interior samples.
                    r[4]=i;r[5]=-1;r[13]=json.dumps(used,separators=(',',':'));r[22]=json.dumps([int(records[s][6]) for s in used],separators=(',',':'));r[23]=r[13]
                    emit(r,vv,f);new_rows+=1
        for r,ns,c in heads:emit(r,ns,c)
    expected=len(src['O_global_facet'])+2*len(src['I_global_facet'])+len(src['D_global_facet'])
    if rows!=expected+180:raise ValueError('Incomplete material/sidewall face use')
    config=json.loads((registration/'result.json').read_bytes());config['files']['FACET_ANCESTRY']=dict(path=str(path),sha256=digest(path));config['face_uses']=dict(registered=expected,authored_head=len(heads),retriangulated=new_rows,shared_neighbors=len(pair),source=digest(source),cortical_faces_separately_registered=int((kind==4).sum())+len(np.load(job/'wrist/cortical-facets.npz')['nodes']))
    (out/'result.json').write_text(json.dumps(config,indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
