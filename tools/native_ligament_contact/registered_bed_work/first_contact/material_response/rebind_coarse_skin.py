"""Rebuild physical coarse donor registers from the fine native face embeddings in their original material frame."""
from pathlib import Path
import json,re,struct,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest

def block(s,name):
    text=s.split(' '+name+'=',1)[1].split(';',1)[0]
    return text.strip().strip('{} \n')
def floats(s):
    return np.array([float(k) for k in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',s)])
def donors(text,key):
    group=re.search(r'constexpr Body '+key+r'\[\]=\{\n(.*?)\n\};',text,re.S)
    rows={}
    for line in group[1].splitlines():
        idx=int(re.match(r'\{(\d+)',line)[1]);v=np.asarray([float.fromhex(v) for v in re.findall(r'(-?0x[0-9a-f.p+\-]+)f\b',line)],'f4')
        rows[idx]=(v[:3].astype(float),Rotation.from_quat(v[3:]).as_matrix())
    return rows
def ints(s):return np.fromstring(s.replace('\n','').replace('u','').strip().strip(','),dtype=int,sep=',')
def enum(s):
    return s.replace('std::numeric_limits<std::uint32_t>::max()','-1').replace('kWholeBodySkinTransferUnassigned','-1')
def run(author,build,job,native,tissues,out):
    key='NumiHumanSelectedSkinReference.hpp';ori=(author/key).read_text()
    oldr=donors((author/'NumiHumanLoadedSkinPoseData.inc').read_text(),'reference');newr=donors((build/'NumiHumanLoadedSkinPoseData.inc').read_text(),'reference')
    frames=floats(block(ori,'frames')).reshape(-1,19);pts=floats(block(ori,'points')).reshape(-1,3);res=floats(block(ori,'residual')).reshape(-1,3)
    donor=frames[:,0].astype(int);l=frames[:,1:4];m=frames[:,4:13].reshape(-1,3,3);normal=frames[:,16:19]
    if not np.array_equal(pts.astype('f4'),frames[:,13:16].astype('f4')):raise ValueError('Inconsistent retained material positions')
    anc=(author/'NumiHumanWholeBodySkinTransferAncestry.hpp').read_text();face=ints(enum(block(anc,'kWholeBodySkinSourceEmbeddingFaces')))
    bary=floats(block(anc,'kWholeBodySkinSourceEmbeddingBarycentric')).reshape(-1,3)
    vertices=ints(enum(block((author/'NumiHumanWholeBodySkinReduction.hpp').read_text(),'kWholeBodySkinSourceVertices')))
    src=Path(json.loads((job/'launch.json').read_bytes())['argv'][json.loads((job/'launch.json').read_bytes())['argv'].index('--skin-payload')+1]);candidate=tissues/src.name
    if digest(src) not in ori:raise ValueError('Previous fine skin identity differs')
    def fine(asset,trial):
        raw=asset.read_bytes();n,nv,ni=struct.unpack_from('<3I',raw,12);faces=np.frombuffer(raw,'<u4',ni,60+36*n+56*nv).reshape(-1,3)
        data=np.loadtxt(trial/'registered-skin-native.csv',delimiter=',',skiprows=1);ii=face>=0
        if len(data)!=nv or not np.array_equal(data[:,0],np.arange(nv)) or (vertices[~ii]<0).any():raise ValueError('Native detailed identity differs')
        pos=data[vertices,2:5].copy();norm=data[vertices,5:8].copy()
        for j,col in ((pos,slice(2,5)),(norm,slice(5,8))):j[ii]=np.einsum('ri,rij->rj',bary[ii],data[faces[face[ii]],col])
        norm/=np.linalg.norm(norm,axis=1)[:,None]
        return pos,norm,faces
    pos,oldn,f=fine(src,job);xyz,newn,f2=fine(candidate,native)
    if not np.array_equal(f,f2):raise ValueError('Fine-skin connectivity changed')
    # Recover the original source-to-supported placement independently from the first material row.
    G=oldr[donor[0]][1]@m[0];G=Rotation.from_matrix(G).as_matrix()
    shift=pos[0]-G@(pts[0]-(res[0] if vertices[0]<0 else 0.))
    world=pts@G.T+shift
    def audit(estim,limit,label):
        e=float(np.max(np.abs(estim)))
        if e>limit:raise ValueError(f'{label}: {e} exceeds {limit}')
        return e
    proj=vertices<0;ctrl=pos.copy();ctrl[proj]+=res[proj]@G.T
    stats=dict(control_point_m=audit(ctrl-world,2.e-5,'control source placement'),control_normal=audit(oldn-normal@G.T,2.e-4,'control material normal'))
    def source_frames(rows):
        R=np.array([rows[b][1].T for b in donor]);T=np.array([rows[b][0] for b in donor])
        return np.einsum('rij,rj->ri',R,world-T),R@G
    lo,mo=source_frames(oldr);ln,mn=source_frames(newr)
    stats['control_local_m']=audit(lo-l,2e-5,'old donor locals');stats['control_frame']=audit(mo-m,2e-5,'old rotation')
    rr=res.copy();rr[proj]=pts[proj]-(xyz[proj]-shift)@G
    stats['max_embedding_residual_change_m']=audit(rr-res,2e-6,'new source residual');stats['direct_material_position_m']=audit(xyz[~proj]-world[~proj],2e-5,'direct material geometry')
    stats['max_material_normal_change']=float(np.linalg.norm(newn-oldn,axis=1).max())
    def numbers(x):return '{'+','.join(format(float(v),'.17g') for v in np.ravel(x))+'}'
    def array(name,kind,lines):return 'inline constexpr std::array<'+kind+','+str(len(pts))+'> '+name+'={{\n  '+',\n  '.join(lines)+'\n}};\n'
    new=['#pragma once\nnamespace selected_skin_reference {\nusing namespace ::metalrobo::human::coarse_tissue;\n','inline constexpr std::string_view payload="'+digest(candidate)+'";\n']
    new.append(array('frames','WholeBodySkinDonorFrameV1',('{'+str(b)+'u,'+','.join(numbers(np.asarray(x,dtype='f4')) for x in (ll,mm,pp,nn))+'}' for b,ll,mm,pp,nn in zip(donor,ln,mn,pts,newn@G))))
    new.append(array('points','std::array<double,3>',map(numbers,pts)));new.append(array('residual','std::array<double,3>',map(numbers,rr)));new.append('}\n')
    out.mkdir();(out/key).write_text(''.join(new));(out/'result.json').write_text(json.dumps(dict(source=digest(src),candidate=digest(candidate),control_native=digest(job/'registered-skin-native.csv'),registered_native=digest(native/'registered-skin-native.csv'),original=digest(author/key),header=digest(out/key),coarse_rows=len(pts),projected=int(proj.sum()),errors=stats),indent=2)+'\n')
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
