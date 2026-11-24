"""Register one derived interior material node and preserve every external surface mapping."""
from pathlib import Path
import csv,json
import numpy as np
from compile_connected_joint import digest


def run(work):
    stage=work/'witness-material';v=stage/'volume';out=stage/'binding/tables';out.mkdir(exist_ok=True,parents=True)
    read=lambda p:json.loads(p.read_bytes())
    prior=read(work/'binding/tables/result.json')['environment']
    env=read(work/'native-integrated-01/launch.json')['environment'];pins=dict(prior)
    pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def source(key):
        f=Path(env[pre+key+'_CSV'])
        if digest(f)!=env[pre+key+'_SHA256']:raise ValueError('Changed '+key)
        return f
    def save(key,f):pins.update({pre+key+'_CSV':str(f),pre+key+'_SHA256':digest(f)})
    fmt=lambda x:format(float(x),'.17g')
    overlay=np.load(v/'source-overlay.npz');oldrows=set(map(int,overlay['old_cells']));newrows=overlay['new_cells']
    rows=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');newcells=rows[newrows,:4]-1
    point=np.load(work/'witness-cavity/material-star.npz')['apex_reference'];ni=int(newcells.max())
    label=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')[newrows[0],1:]
    nodes=np.loadtxt(env['NUMI_H706_FRACTIONAL_SOURCE330_NODES'],skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7))
    tet=np.loadtxt(env['NUMI_H706_FRACTIONAL_SOURCE330_TETS'],skiprows=1,delimiter=',',usecols=(3,4,5,6),dtype='i4')
    parent=np.loadtxt(source('PARENT_LABELS'),skiprows=1,delimiter=',',dtype='i8')
    s=np.flatnonzero((parent[:,1:]==label).all(1))
    p=nodes[tet[s],3:]
    b=np.linalg.solve((p[:,1:]-p[:,:1]).transpose(0,2,1),(point-p[:,0])[:,:,None])[:,:,0]
    b=np.column_stack((1-b.sum(1),b));inside=np.flatnonzero((b>=0).all(1))
    if len(inside)!=1:raise ValueError('New interior source parent is ambiguous')
    parent_row=int(s[inside[0]]);weights=b[inside[0]];residual=point-weights@nodes[tet[parent_row],:3]
    with source('DONOR_FRAMES').open()as f:
        donor,=[r for r in csv.DictReader(f) if int(r['donor'])==label[1]]
    R=np.array([float(donor[f'current_R{i}{j}'])for i in range(3)for j in range(3)]).reshape(3,3)
    origin=np.array([float(donor['current_'+k])for k in 'xyz'])
    local=np.linalg.solve(R,point-origin)
    if np.linalg.norm(origin+R@local-point)>1e-10:raise ValueError('Donor registration error')
    def owners(n):
        idx,slot=np.where(newcells==n)
        if not len(idx):raise ValueError('Lost original joint node')
        k=np.argmin(newrows[idx]);return int(newrows[idx[k]]),int(slot[k])
    f=out/'combined-node-map.csv';maximum=0;count=0;prototype=None
    with source('COMBINED_NODE_MAP').open()as inp,f.open('w')as dst:
        dst.write(inp.readline())
        for line in inp:
            r=line.rstrip('\r\n').split(',');n=int(r[0])-1
            if r[1]=='fat':maximum=max(maximum,int(r[2]))
            if prototype is None and r[8]=='interior' and r[18]=='none' and int(r[12])==label[1] and int(r[29])==parent_row:
                prototype=r.copy()
            if int(r[43]) not in oldrows:dst.write(line);continue
            if np.any(np.array(r[48:52],float)) or r[1]!='fat':raise ValueError('Unexpected cavity anchor')
            k,slot=owners(n)
            r[40:42]=list(map(str,label[1:3]));r[43:48]=[str(k),*[str(int(i==slot)) for i in range(4)]]
            dst.write(','.join(r)+'\n');count+=1
        if prototype is None:raise ValueError('Unrecognized interior registration schema')
        r=prototype.copy();r[:9]=[str(ni+1),'fat',str(maximum+1),'-1','','','','-1','interior']
        r[9:17]=[*map(fmt,point),str(label[1]),donor['binding'],*map(fmt,local)]
        r[17:29]=['derived_volume_interior_no_surface_ancestry','none','-1','-1','0','0','0','0','0','0','derived_full_all_bodies_geometric_donor_pose','source_parent_tet_reference_registration']
        r[29:39]=[str(parent_row),*map(fmt,weights),*map(fmt,residual),fmt(np.linalg.norm(residual)),'0']
        k,slot=owners(ni);r[39:]=[str(1<<int(label[0])),str(label[1]),str(label[2]),'-1',str(k),*[str(int(i==slot))for i in range(4)],'0','0','0','0',str(label[1]),str(label[2])]
        dst.write(','.join(r)+'\n')
    save('COMBINED_NODE_MAP',f)
    f=out/'facet-ancestry.csv';bound=0
    with source('FACET_ANCESTRY').open()as inp,f.open('w')as dst:
        rd=csv.reader(inp);wr=csv.writer(dst,lineterminator='\n');wr.writerow(next(rd))
        for r in rd:
            c=int(r[18])
            if c in oldrows:
                corners=[int(a)-1 for a in r[19:22]];ids=np.flatnonzero(np.isin(newcells,corners).sum(1)==3)
                if len(ids)!=1 or r[0]!='fat':raise ValueError('Nonunique remeshed boundary')
                j=int(newrows[ids[0]]);r[17:19]=[j+1,j];bound+=1
                if 'headwall' in r[3] and r[16]=='[-1,-1,-1]':
                    if json.loads(r[23])!=[c]:raise ValueError('Ambiguous headwall history')
                    r[23]=json.dumps([j])
            wr.writerow(r)
    save('FACET_ANCESTRY',f)
    (out/'result.json').write_text(json.dumps(dict(environment=pins,retained_surface_tables=5,new_interior_node=ni,parent_tet=parent_row,barycentric=weights.tolist(),geometric_parent_residual_m=float(np.linalg.norm(residual)),rebound_existing_node_rows=count,rebound_facet_rows=bound),indent=2)+'\n')
if __name__=='__main__':run(Path(__file__).resolve().parent)
