"""Transport each of the 34 native ligament fibers by its actual cortical triangles and elastic rest length."""
from pathlib import Path
import json,re,sys
import numpy as np
from scipy.spatial.transform import Rotation
from audit_registered_skeleton import HEADER,RECORD,digest

def banks(path):
    raw=path.read_bytes();h=HEADER.unpack_from(raw);n,nv=h[2:4];vo=HEADER.size+RECORD.size*n;io=vo+24*nv;result={}
    for i in range(n):
        rec=RECORD.unpack_from(raw,HEADER.size+i*RECORD.size);b,start,size,first,count,sid=rec[:6]
        v=np.frombuffer(raw,'<f4',6*size,vo+24*start).reshape(-1,6)
        faces=np.frombuffer(raw,'<u4',count,io+4*first).reshape(-1,3).astype('i8')-start
        xyz=Rotation.from_quat(rec[9:13]).apply(v[:,:3].astype(float)*rec[13])+rec[6:9]
        result[b,sid]=(faces,xyz,v.copy())
    return result

def run(author,control,parsed,capture,compiled,output):
    saved=(capture/'native-attachments.txt').read_text().splitlines()
    src=json.loads((control/'launch.json').read_bytes());original=Path(src['argv'][3]);candidate=next(compiled.glob('*.nhbones'));manifest=json.loads(candidate.with_suffix('.manifest.json').read_bytes())
    if digest(original)!=src['pins'][str(original)] or manifest['source']['connected_shoulder_registration']['parent_bone_sha256']!=digest(original):raise ValueError('Wrong cortical material source')
    key='NumiHumanRegisteredLigaments.hpp';header=(author/key).read_text();old=bytes.fromhex(digest(original));new=bytes.fromhex(digest(candidate))
    match=re.search(r'kReferenceBonesPayloadSha256\{\{(.*?)\}\};',header,re.S)
    if bytes(int(v,16) for v in re.findall(r'0x([a-f0-9]{2})',match[1]))!=old:raise ValueError('Native ligament source and previous cage differ')
    srcbanks=banks(original);dst=banks(candidate)
    if set(srcbanks)!=set(dst) or len(dst)!=185:raise ValueError('Cortical bank ownership differs')
    for sid,(f,x,v) in srcbanks.items():
        if not np.array_equal(f,dst[sid][0]) or not np.array_equal(v,dst[sid][2]):raise ValueError('Cortex source triangles or coordinates were edited')
    table={};spec={tuple(row.split()[:2]):row.split() for row in parsed.read_text().splitlines()};report={}
    if len(spec)!=34 or len(saved)!=34:raise ValueError('Incomplete native catalog capture')
    for line in saved:
        t=line.split();name,side=t[:2];s=spec[name,side];a,b,c,d,e,f=map(int,t[2:8]);L=float(t[8]);points=np.asarray(t[9:15],float).reshape(2,3)
        if list(map(int,s[2:6]))!=[a,b,d,e] or L<=0 or float(s[14])!=float(t[15]) or float(s[15])!=float(t[16]):raise ValueError('Anatomical or constitutive identity changed')
        pair=[];stats=[]
        for (body,sid,tri),p in zip(((a,b,c),(d,e,f)),points):
            faces,x,_=srcbanks[body,sid];v=x[faces[tri]];uv=np.linalg.lstsq((v[1:]-v[0]).T,p-v[0],rcond=None)[0];w=np.r_[1-uv.sum(),uv]
            # Float32 GPU body coordinates and retained body-local points can round to the other side of an edge.
            w=np.maximum(0.,w);w/=w.sum();err=np.linalg.norm(w@v-p)
            if err>2e-6:raise ValueError(f'{name} {side} bone {sid} triangle {tri}, material projection error {err}')
            if not np.isfinite(w).all():raise ValueError('Nonfinite barycentrics')
            pair.append('{'+','.join(map(str,[body,sid,tri]))+',{'+','.join(float(v).hex() for v in w)+'}}')
            stats.append(dict(body=body,bone=sid,triangle=tri,barycentric=w.tolist(),source_error_m=float(err)))
        index=(name,int(side))
        if index in table:raise ValueError('Duplicate original fiber')
        table[index]='{"'+name+'",'+side+','+float(L).hex()+','+','.join(pair)+'}'
        report[name+'/'+side]=dict(endpoints=stats,reference_length_m=L,stiffness_n_per_m=float(t[15]),area_m2=float(t[16]))
    output.mkdir();inc=output/'NumiHumanRegisteredLigamentMaterial.inc'
    inc.write_text('struct MaterialEndpoint { unsigned body,stable,triangle; V3 w; };\nstruct RegisteredMaterialFiber { std::string_view name; unsigned side; double length; MaterialEndpoint a,b; };\ninline constexpr RegisteredMaterialFiber materialFibers[]={\n'+',\n'.join(table.values())+'\n};\n')
    header=header[:match.start(1)]+','.join('0x%02x'%v for v in new)+header[match.end(1):]
    header=header.replace('struct Spec {','#include "NumiHumanRegisteredLigamentMaterial.inc"\nstruct Spec {',1)
    anchor=' auto p0=inferred_acl::detail::project(x,u),p1=inferred_acl::detail::project(y,v);'
    if header.count(anchor)!=1:raise ValueError('Unknown native projection')
    impl='''
 const RegisteredMaterialFiber* material=nullptr;
 if(s.requiresPinnedSourceIdentity){
  for(const auto& row:materialFibers)if(row.name==s.name&&row.side==s.side) material=&row;
  if(!material)throw std::runtime_error("missing registered material fiber");
  auto endpoint=[](const MaterialEndpoint& e,const inferred_acl::BoneSurface& bank) {
    if(e.stable!=bank.stableId||3ull*e.triangle+2>=bank.triangleIndices.size())throw std::runtime_error("mismatched ligament material face");
    V3 p{};
    for(unsigned k=0;k<3;++k)for(unsigned j=0;j<3;++j)p[j]+=e.w[k]*bank.worldVertices[bank.triangleIndices[3*e.triangle+k]][j];
    return p;
  };
  if(material->a.body!=s.body0||material->b.body!=s.body1)throw std::runtime_error("mismatched source ligament bodies");
  x=endpoint(material->a,u);y=endpoint(material->b,v);
 }
'''
    header=header.replace(anchor,impl+anchor).replace(' if(!std::isfinite(d0)||',' if(material&&(d0>2e-6||d1>2e-6))throw std::runtime_error("ligament material projection lost its registered attachment");\n if(!std::isfinite(d0)||',1)
    header=header.replace(' double c3=s.stiffnessNPerM*L/',' const double referenceLength=material?material->length:L;\n double c3=s.stiffnessNPerM*referenceLength/').replace('r.reference={float(L),','r.reference={float(referenceLength),').replace('q.referenceLengthM=L;','q.referenceLengthM=referenceLength;q.inSituStretch=L/referenceLength;')
    (output/key).write_text(header)
    (output/'registration.json').write_text(json.dumps(dict(control=digest(original),candidate=digest(candidate),native_capture=digest(capture/'native-attachments.txt'),source_header=digest(author/key),header=digest(output/key),material=digest(inc),fibers=report),indent=2)+'\n')
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
