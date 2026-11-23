"""Carry exact material identities, resting bed witnesses and deep anchors into the native scene binding."""
from pathlib import Path
import csv,json,re,sys
import numpy as np
from compile_connected_joint import digest


def run(launch,source,wrist,joined,header,out):
    cfg=json.loads(launch.read_text());env=cfg['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    old=json.loads((source/'native-import.json').read_text());new=json.loads((joined/'native-import.json').read_text())
    if not new['native_imported'] or any(new[k] for k in ('below_shape','below_volume','inverted','nonfinite','density_outliers','zero_mass_nodes')):raise ValueError('Unqualified physical material')
    for k,field in [('TETGEN_NODE','node_sha256'),('TETGEN_ELEMENTS','element_sha256'),('CHILD_LABELS','label_sha256')]:
        if env[pre+k+'_SHA256']!=old[field]:raise ValueError('Native scene does not match the remap source')
    w=np.load(wrist/'material-source-indices.npz')['original_to_emitted']
    s=np.load(joined/'material-source-indices.npz')['original_to_emitted']
    parent=np.full(len(w),-1,int);parent[w>=0]=s[w[w>=0]]
    read=lambda p:np.loadtxt(p/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4')
    x=read(source);y=read(joined);keep=parent>=0
    if len(x)!=len(w) or len(np.unique(parent[keep]))!=int(keep.sum()) or not np.array_equal(x[keep],y[parent[keep]]):raise ValueError('Surviving node identity is not exact')
    hm=header.read_text();a=re.search(r'constexpr Anchor anchors\[\]=\{(.*?)\};',hm,re.S)
    ids=[]
    def update(m):
        stable,j,b=map(int,m.groups());k=parent[j]
        if k<0:raise ValueError(f'Physically anchored node removed: {stable}')
        ids.append([stable,j,int(k),b]);return '{%du,%du,%du,{'%(stable,k,b)
    region=re.sub(r'\{(\d+)u,(\d+)u,(\d+)u,\{',update,a[1])
    if len(ids)!=a[1].count('},') and len(ids)!=len(re.findall(r'^\{',a[1],re.M)):raise ValueError('Partial native anchor capture')
    if not ids:raise ValueError('Missing native deep anchors')
    hm=hm[:a.start(1)]+region+hm[a.end(1):]
    q=re.search(r'registeredBedContactNodes\[\]=\{([^}]+)\}',hm);bed=[int(v) for v in re.findall(r'(\d+)u',q[1])]
    if not bed or min(parent[bed])<0:raise ValueError('First bed contact was lost')
    hm=hm[:q.start(1)]+','.join(str(parent[j])+'u' for j in bed)+hm[q.end(1):]
    for name,field in [('referenceSHA','node_sha256'),('nodeSHA','node_sha256'),('elementsSHA','element_sha256')]:
        pattern=r'(constexpr char '+name+r'\[\]=\x22)([0-9a-f]{64})(\x22;)'
        match=re.search(pattern,hm)
        if not match or match[2]!=old[field]:raise ValueError(f'Unmatched {name}')
        hm=hm[:match.start(2)]+new[field]+hm[match.end(2):]
    for name in ('nodes','cells'):
        if hm.count(f'unsigned {name}={old[name]};')!=1:raise ValueError('Wrong header domain')
        hm=hm.replace(f'unsigned {name}={old[name]};',f'unsigned {name}={new[name]};')
    roles={};layer={};registered={}
    with open(env[pre+'COMBINED_NODE_MAP_CSV']) as f:
        for row in csv.DictReader(f):
            j=int(row['joint_node_id'])-1;k=row['role'];lo=roles.setdefault(k,[0,0]);lo[0]+=1;lo[1]+=int(parent[j]>=0)
            if k in ('O','I','D'):
                v=int(row['registered_surface_vertex_row'])
                key=(k,v)
                previous=registered.get(key)
                if previous is not None:
                    if k!='I' or previous[0]!=j or previous[1]==row['source_partition']:raise ValueError('Conflicting native skin/fat interface identity')
                    continue
                registered[key]=(j,row['source_partition'])
                layer.setdefault(k,{})
                layer[k][v]=int(parent[j])
    if roles['O'][0]!=roles['O'][1] or roles['O'][0]!=int(re.search(r'outerNodes=(\d+)',hm)[1]):raise ValueError('Lost outer skin')
    out.mkdir()
    (out/header.name).write_text(hm)
    np.savez_compressed(out/'node-map.npz',source_to_emitted=parent,**{k:np.array([d[i] for i in range(len(d))],dtype='i4') for k,d in layer.items()})
    result=dict(launch=digest(launch),parent=digest(source/'material-source-indices.npz'),wrist=digest(wrist/'material-source-indices.npz'),joined=digest(joined/'result.json'),header=digest(out/header.name),map=digest(out/'node-map.npz'),source_nodes=len(x),current_nodes=len(y),surviving=int(keep.sum()),new_nodes=int(len(y)-keep.sum()),roles=roles,anchors=ids,first_contact_old=bed,first_contact_new=parent[bed].tolist(),max_survivor_displacement_m=0.)
    (out/'registration.json').write_text(json.dumps(result,indent=2)+'\n');print(roles,len(ids),parent[bed])
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
