"""Bind the audited shoulder paths to the existing native passive ligament rows."""
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from audit_fiber_load_bounds import read_pose
from rebind_ligament_material import banks

def exact(s,old,new,n=1):
    if s.count(old)!=n:raise ValueError(f'Expected {n} source occurrences of {old[:95]}, found {s.count(old)}')
    return s.replace(old,new)
def fl(x):return float(np.float32(x)).hex()+'f'
def v4(v):return 'float4('+','.join(map(fl,v))+')'
def arr(v):return '{'+','.join(float(x).hex() for x in v)+'}'
def write(path,content):
    if path.is_symlink():path.unlink()
    path.write_text(content)
def make(h,out):
    ref=h.parent/'native-ligament-material/stderr.log'
    src=next((h.parent/'supported-bones').glob('*.nhbones'))
    pp=read_pose(ref);pose=np.array(pp['bodyPoses']);bank=banks(src);bonehash=digest(src)
    if pp['bonePayloadSha256']!=bonehash or pp['sourceBodyMap']:raise ValueError('Unexpected input frame')
    data=[];features=[];hdr=[];sign=[];identity=[];sites=[];rows=[];guards=[]
    ranges=np.fromfile(h/'body-pose-wide/ranges.bin','<u4').reshape(-1,4)
    edges=np.fromfile(h/'body-pose-wide/edges.bin','<f4').reshape(-1,4,4)
    for edge in edges:features.append('{'+','.join(v4(x) for x in edge)+'}')
    trials=json.loads((h/'slack-upper-bound.json').read_text())
    for i,path in enumerate(sorted((h/'oriented-routes').glob('*.json'))):
        r=json.loads(path.read_text());section=h/'section-routes'/path.name;s=json.loads(section.read_text())
        if r['section']!=digest(section) or r['registered_bones']!=bonehash or s['pose']!=digest(ref) or r['relaxed_interiors']:raise ValueError('Uncertified reference route')
        name,side=r['fiber'].split('/');side=int(side)
        canonical=84 if side==0 else 34
        points=np.array(r['world_m']);edges_ref=r['edges'];ends=s['sites']
        if ends[0]['body']!=canonical:points=points[::-1];edges_ref=edges_ref[::-1];ends=ends[::-1]
        bodies=[x['body'] for x in ends];owners=np.array([bodies[0]]+[e['body'] for e in edges_ref]+[bodies[1]])
        split,=np.flatnonzero(np.diff(owners));P,Q=points[split:split+2]
        rot=[Rotation.from_quat(pose[b,3:]) for b in bodies]
        offs=[rot[0].inv().apply(P-points[0]),rot[1].inv().apply(Q-points[-1])]
        length=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
        if abs(length-r['relaxed_length_m'])>1e-10:raise ValueError('Path length mismatch')
        for a,p in zip(ends,points[[0,-1]]):
            b=a['body'];f,v,_=bank[b,a['bone']];xyz=Rotation.from_quat(pose[b,3:]).apply(np.asarray(a['barycentric'])@v[f[a['triangle']]])+pose[b,:3]
            if np.linalg.norm(xyz-p)>2e-8:raise ValueError('Material site drift')
        tag=int(digest(path)[:8],16)
        arc=length-np.linalg.norm(Q-P)
        cc=name=='coracoclavicular_complex'
        if cc and (split!=len(edges_ref) or i!=side):raise ValueError('Expected one loaded free span')
        if not cc and (length>=s['reference_m'] or trials['trials'][r['fiber']]['passed']!=13):raise ValueError('No slack certificate')
        if not cc:
            # Each departure is a source cortical edge. Require the moving free
            # span to leave the exterior of its oriented dihedral at both ends.
            for b,k,sg in [(0,split-1,1),(1,split,-1)]:
                e=edges_ref[k];body=bodies[b];f,v,_=bank[body,e['stable_id']];tri=v[f[e['incident_faces']]]
                normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
                tri_all=v[f];vol=np.einsum('ij,ij->',np.cross(tri_all[:,1]-tri_all[:,0],tri_all[:,2]-tri_all[:,0]),tri_all[:,0]-v.mean(0)).sum()
                normals=normal*np.sign(vol)
                edge=v[e['source_edge']];op=[next(x for x in t if min(np.linalg.norm(edge-x,axis=1))>1e-10) for t in tri]
                convex=np.dot(normals[0],op[1]-edge[0])<0
                guards.append((i,b,normals,convex))
        rr=ranges[side] if cc else [0]*4
        rows.append('{uint4('+','.join(str(int(k))+'u' for k in rr)+'),'+v4([*offs[0],arc])+','+v4([*offs[1],0])+'}')
        area={'coracoclavicular_complex':10e-6,'coracohumeral_capsuloligament':53.7e-6,'superior_glenohumeral_capsuloligament':11.3e-6}[name]
        stiffness={'coracoclavicular_complex':103000.,'coracohumeral_capsuloligament':36700.,'superior_glenohumeral_capsuloligament':17400.}[name]
        d=s['reference_m'];c3=stiffness*d/(area*.001)
        sign.append(f'case {tag}u: return {i}u;')
        identity.append(f'case {i}u: return f.firstBodyIndex=={bodies[0]}u && f.secondBodyIndex=={bodies[1]}u && f.reference.x=={fl(d)} && f.reference.y=={fl(area)} && f.reference.z==1.0f && f.material.x=={fl(c3)} && f.material.y=={fl(.001)} && f.material.z=={fl(c3*.001*np.exp(.001*.05))} && f.material.w=={fl(1.05)};')
        sites.append('{"'+name+'",'+str(side)+','+float(d).hex()+','+','.join('{'+','.join(str(a[k]) for k in ('body','bone','triangle'))+','+arr(a['barycentric'])+'}' for a in ends)+'},')
        hdr.append('{"'+name+'",'+str(side)+','+str(tag)+'u,'+float(length).hex()+','+arr(points[0])+','+arr(points[-1])+'},')
        data.append(dict(fiber=r['fiber'],tag=tag,route=digest(path),section=digest(section),length=length,reference=d,owners=bodies,loaded=cc))
    if len(data)!=6 or len(set(x['tag'] for x in data))!=6:raise ValueError('Incomplete route catalog')
    cases='inline unsigned nhCorticalIndex(unsigned tag){switch(tag){\n'+'\n'.join(sign)+'\ndefault:return 6u;}}\n'
    cases+='inline bool nhKnownCorticalRow(NMNumiHumanPassiveLigamentGPU f){switch(nhCorticalIndex(f.reserved0)){\n'+'\n'.join(identity)+'\ndefault:return false;}}\n'
    write(out/'CorticalIdentity.inc',cases)
    table='constant NHCorticalLigamentEdge nhRegisteredEdges[]={\n'+',\n'.join(features)+'\n};\n'
    table+='struct NHCorticalRoute {uint4 range;float4 firstOffset,secondOffset;};\nconstant NHCorticalRoute nhRouteTable[]={\n'+',\n'.join(rows)+'\n};\n'
    guard='inline bool nhSlackRouteClear(uint index,float4 qa,float4 qb,float3,float3 u){switch(index){\n'
    for i in sorted({g[0] for g in guards}):
        expr=[]
        for _,b,ns,convex in (g for g in guards if g[0]==i):
            q=('qa','qb')[b];d=('u','-u')[b]
            terms=[f'dot(humanQuaternionRotate({q},{v4([*n,0])}.xyz),{d})' for n in ns]
            expr.append(('max' if convex else 'min')+'('+','.join(terms)+') >= -2e-6f')
        guard+=f'case {i}u:return '+' && '.join(expr)+';\n'
    guard+='default:return false;}}\n'
    write(out/'CorticalTables.metalinc',cases+table+guard)
    source=h/'native-scene-build'
    mat=(source/'NumiHumanRegisteredLigamentMaterial.inc').read_text()
    for entry in sites:
        prefix=entry[:entry.index(',',entry.index(',')+1)+1]
        matches=[x for x in mat.splitlines() if x.startswith(prefix)]
        if len(matches)!=1:raise ValueError('Ambiguous native material row')
        mat=exact(mat,matches[0],entry)
    write(out/'NumiHumanRegisteredLigamentMaterial.inc',mat)
    validation='''struct RegisteredCorticalFiber {std::string_view name;unsigned side,tag;double length;V3 start,end;};
inline constexpr RegisteredCorticalFiber registeredCorticalFibers[]={
'''+ '\n'.join(hdr)+'\n};\ninline constexpr float registeredCorticalPose[][7]={\n'+',\n'.join('{'+','.join(map(fl,p))+'}' for p in pose)+'\n};\n'
    write(out/'NumiHumanCorticalRegistration.inc',validation)
    write(out/'registration.json',json.dumps(dict(bones=bonehash,pose=digest(ref),records=data,source_meshes=len(bank)),indent=2)+'\n')
if __name__=='__main__':make(*(Path(x).resolve() for x in sys.argv[1:]))
