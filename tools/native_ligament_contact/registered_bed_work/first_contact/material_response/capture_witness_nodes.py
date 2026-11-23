"""Limit the native accepted-endpoint force dump to the skin/fat nodes from a verified failure receipt."""
import json
import os
from pathlib import Path
import subprocess
import sys
from register_material_surfaces import sha
from run_cavity_motion import execute_native

def replace(code,old,new):
    if code.count(old)!=1: raise ValueError('Ambiguous endpoint capture insertion')
    return code.replace(old,new)

def build(parent,dest):
    j=json.loads((parent/'build.json').read_bytes())
    app='numilab_human_myosim_visual_probe.mm'
    if sha(parent/app)!=j['application_sha256'] or sha(parent/'numi-human-native')!=j['executable_sha256']:
        raise ValueError('Native parent differs')
    dest.mkdir()
    for p in parent.iterdir():
        if p.name!=app and p.suffix in ('.mm','.metal','.metalinc','.inc','.hpp','.h'):os.link(p,dest/p.name)
    s=(parent/app).read_text()
    k='            const auto& ranges=continuumTransaction->restingFEMAuditRanges;'
    s=replace(s,k,k+'''
            const char* selection=std::getenv("NUMI_HUMAN_FEM_ENDPOINT_SKIN_FAT_NODES");
            std::vector<uint32_t> selected;
            if(selection) {
                std::ifstream in(selection);
                require(in.is_open(),"Missing skin/fat endpoint selection");
                uint64_t index;
                while(in>>index) {
                    require(index<UINT32_MAX && selected.size()<endpointNodeCount &&
                            (selected.empty() || index>selected.back()),"Invalid or unsorted endpoint node");
                    selected.push_back(static_cast<uint32_t>(index));
                }
                require(in.eof() && !selected.empty() && std::count_if(ranges.begin(),ranges.end(),[&](const auto& r){
                    return r.objectName==continuumTransaction->restingFEMRegionLabelObjectName &&
                           r.firstNode==continuumTransaction->restingFEMRegionLabelFirstNode &&
                           selected.back()<r.nodeCount;
                })==1,"Selection does not resolve to the registered skin/fat object");
            }''')
    n='                    std::cout<<"resting_FEM_node,object="'
    s=replace(s,n,'''                    if(selection && (range.objectName!=continuumTransaction->restingFEMRegionLabelObjectName ||
                                     !std::binary_search(selected.begin(),selected.end(),local))) continue;
'''+n)
    n='                    std::cout<<"resting_FEM_tet,object="'
    s=replace(s,n,'                    if(selection) continue;\n'+n)
    (dest/app).write_text(s)
    commands=[[v.replace(str(parent)+'/',str(dest)+'/') if v.startswith(str(parent)+'/') else v for v in cmd] for cmd in j['commands']]
    for c in commands:subprocess.run(c,check=True)
    (dest/'build.json').write_text(json.dumps(dict(parent_build=sha(parent/'build.json'),parent_executable=sha(parent/'numi-human-native'),
        application_sha256=sha(dest/app),executable_sha256=sha(dest/'numi-human-native'),commands=commands),indent=2)+'\n')

def run(source,parent,root):
    build_dir=root/'force-build';build(parent,build_dir)
    w=json.loads((source/'cell-failures.json').read_bytes())
    j=json.loads((source/'launch.json').read_bytes())
    if w['accepted_pose']!=sha(source/'physical-accepted.node') or sha(Path(j['argv'][0]))!=sha(parent/'numi-human-native'):
        raise ValueError('Unmatched failure source')
    nodes=sorted({n for row in w['failures'] for n in row['nodes']})
    out=root/'force-release';out.mkdir();(out/'native-traces').mkdir()
    indices=out/'witness-nodes.txt';indices.write_text(''.join(str(n)+'\n' for n in nodes))
    argv=[v.replace(str(source)+'/',str(out)+'/') for v in j['argv']]
    argv[0]=str(build_dir/'numi-human-native')
    argv[argv.index('--muscle-step-count')+1]='1'
    env={k:v.replace(str(source)+'/',str(out)+'/') for k,v in j['environment'].items()}
    env.pop('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS',None)
    env.update(NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE='1',NUMI_HUMAN_FEM_ENDPOINT_SKIN_FAT_NODES=str(indices),
               NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP='1')
    (out/'launch.json').write_text(json.dumps(dict(j,argv=argv,environment=env,source_launch=sha(source/'launch.json'),
        node_selection=sha(indices),selection_source=sha(source/'cell-failures.json'),build=sha(build_dir/'build.json')),indent=2)+'\n')
    r=execute_native(argv,j['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,
                     minimum_free_bytes=4*2**30,timeout_s=600)
    print(r,flush=True)
    return r['exit']
if __name__=='__main__':raise SystemExit(run(*(Path(k).resolve() for k in sys.argv[1:])))
