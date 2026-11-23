"""Register the material compiler's supported reference and validation frames to the physical joint edits."""
from pathlib import Path
import json,re,shutil,struct,subprocess,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest,matrix


def run(control,job,rigid,tissues,output):
    header=control/'NumiHumanLoadedSkinPoseData.inc'
    provenance=json.loads((rigid/'source-bind.json').read_bytes())
    if provenance['asset_sha256']!=digest(rigid/'source-bound.nhrigid') or json.loads((tissues/'result.json').read_bytes())['rigid']!=provenance['asset_sha256']:
        raise ValueError('Independently compiled joint or surface source differs')
    raw=(rigid/'source-bound.nhrigid').read_bytes();frames=json.loads((rigid/'frames.json').read_bytes())
    poses=json.loads((job/'native-initial-bodies.json').read_bytes())['all_bodies']
    native=[matrix(b['current']['position_world_m']+b['current']['orientation_world_xyzw']) for b in poses]
    parent={i+1:struct.unpack_from('<I',raw,224+160*157+144*i)[0] for i in range(156)}
    def region(body):
        while body and parent[body] not in (34,84):body=parent[body]
        return parent[body] if body else None
    registered={b:native[b]@matrix(f['parent_translation_base_m']+f['parent_rotation_xyzw'])@np.linalg.inv(native[b]) for b,f in zip((84,34),frames)}
    text=header.read_text();changed={};original={};now={};position_error=0.;rotation_error=0.
    for key in ('reference','target'):
        group=re.search(r'constexpr Body '+key+r'\[\]=\{\n(.*?)\n\};',text,re.S);lines=[];changed[key]=0;original[key]={};now[key]={}
        for line in group[1].splitlines():
            idx=int(re.match(r'\{(\d+)',line)[1]);values=np.array([np.float32(float.fromhex(v)) for v in re.findall(r'(-?0x[0-9a-f.p+\-]+)f\b',line)],float)
            if len(values)!=7 or idx in original[key]:raise ValueError('Invalid donor')
            old=matrix(values);original[key][idx]=old
            if np.linalg.norm(old[:3,3]-native[idx][:3,3])>2e-6 or np.linalg.norm(old[:3,:3]-native[idx][:3,:3])>6e-6:raise ValueError('Control registration and native donor disagree')
            root=region(idx)
            if root is None:
                now[key][idx]=old;lines.append(line);continue
            new=registered[root]@old;q=Rotation.from_matrix(new[:3,:3]).as_quat()
            if q@values[3:]<0:q=-q
            coords=np.asarray([*new[:3,3],*q],dtype='f4');emitted=matrix(coords);now[key][idx]=emitted
            position_error=max(position_error,np.linalg.norm(emitted[:3,3]-new[:3,3]))
            rotation_error=max(rotation_error,np.linalg.norm(emitted[:3,:3]-new[:3,:3]))
            vals=[float(v).hex()+'f' for v in coords]
            lines.append('{'+str(idx)+',{'+','.join(vals[:3])+'},{'+','.join(vals[3:])+'}},');changed[key]+=1
        text=text[:group.start(1)]+'\n'.join(lines)+text[group.end(1):]
    anchors=re.search(r'constexpr Anchor anchors\[\]=\{\n(.*?)\n\};',text,re.S)[1]
    aids=[]
    for line in anchors.splitlines():
        stable,node,donor=map(int,re.match(r'\{(\d+)u,(\d+)u,(\d+)u',line).groups());aids.append((stable,node,donor))
        if region(donor) is not None or not np.array_equal(original['reference'][donor],now['reference'][donor]):raise ValueError('Deep attachment needs material-space re-registration')
    if len(aids)!=14 or len(set(aids))!=14:raise ValueError('Unexpected deep attachment identities')
    # Verify source-skin bind transforms and both material donor banks agree.
    orig=json.loads((job/'launch.json').read_bytes())['argv'];sk=Path(orig[orig.index('--skin-payload')+1])
    before=sk.read_bytes();after=(tissues/sk.name).read_bytes();n=struct.unpack_from('<I',before,12)[0];binderr=0.
    for i in range(n):
        a=struct.unpack_from('<I8f',before,60+36*i);b=struct.unpack_from('<I8f',after,60+36*i)
        if a[0]!=b[0] or a[8]!=b[8]:raise ValueError('Skin donor or material scale changed')
        for key in ('reference','target'):
            if a[0] not in original[key]:continue
            old=original[key][a[0]]@matrix(a[1:8]);new=now[key][b[0]]@matrix(b[1:8]);binderr=max(binderr,float(np.max(abs(new-old))))
    if binderr>2e-6:raise ValueError('Rebound bulk skin disagrees with exterior bind map')
    output.mkdir();(output/header.name).write_text(text)
    # Sibling includes inside the translation unit must resolve to this isolated build.
    original_commands=json.loads((control/'build.json').read_bytes())['commands']
    for item in control.iterdir():
        if item.suffix in ('.hpp','.inc','.mm','.h') and item!=header:
            # Use hard links: includes relative to symlinks can resolve to the old directory.
            import os
            os.link(item,output/item.name)
    cmds=[]
    for cmd in original_commands:
        cmd=[str(output/Path(x).name) if Path(x).parent==control and Path(x).name in ('numilab_human_myosim_visual_probe.mm','numi-human-native','numi-human-native.o') else x for x in cmd]
        cmds.append(cmd)
        with (output/'compiler.log').open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    report=dict(physical_source=provenance['asset_sha256'],native_reference=digest(job/'native-initial-bodies.json'),original_header=digest(header),registered_header=digest(output/header.name),changed=changed,deep_attachments=aids,max_position_emission_error_m=position_error,max_rotation_matrix_error=rotation_error,max_bind_map_error=binderr,commands=cmds,executable=digest(output/'numi-human-native'))
    (output/'registration.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
