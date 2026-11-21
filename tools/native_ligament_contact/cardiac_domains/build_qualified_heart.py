"""Stage the qualified nested-domain gate into the same native respiratory owner."""
from pathlib import Path
import hashlib,json,shutil,sys,subprocess
from certify_heart import sha

def metal_tube(d,idx):
    padded=[list(row)+[float(d['pivot']) if j==0 else float(idx) if j==1 else 0.] for j,row in enumerate(d['tube'])]
    def row(r):return '{'+', '.join('float4('+', '.join(float(x).hex()+'f' for x in r[n:n+4])+')' for n in (0,4))+'}'
    return '{'+', '.join(row(r) for r in padded)+'}'

def emit(contracts,refinements):
    parents=contracts['domains'];children=[];spans=[]
    assert refinements['full_atlas_sha256']==contracts['extended_table_input_sha256']
    for ordinal,(p,r) in enumerate(zip(parents,refinements['parents'],strict=True)):
        assert r['parent']==ordinal
        first=len(children);children.extend(r['children'])
        if r['unproven']:raise ValueError('Physiological forecast still contains unresolved geometry')
        allowed=r['parent_accepted_without_children']
        if allowed and (r['witness_rows'] or r['children']):raise ValueError('Parent eligibility mismatch')
        if not allowed and sorted(p['ids'])!=sorted(i for c in r['children'] for i in c['ids']):raise ValueError('Incomplete subdomain coverage')
        spans.append((first,len(children)-first,int(allowed)))
    def table(name,rows):
        return 'constant CardiacCertifiedTube '+name+'[] = {\n'+',\n'.join(metal_tube(r,i) for i,r in enumerate(rows))+'\n};\n'
    assert children
    return (table('heartParentDomains',parents)+table('heartChildDomains',children)+
            'constant uint3 heartChildSpans[] = {'+', '.join(f'uint3({f},{n},{a})' for f,n,a in spans)+'};\n'+
            '''static inline bool cardiacNestedDomain(thread const float* q, thread uint& witness) {
    for(uint p=0;p<sizeof(heartParentDomains)/sizeof(heartParentDomains[0]);++p) {
        if(!cardiacInsideCertifiedTube(heartParentDomains[p],q)) continue;
        const uint3 span=heartChildSpans[p];
        if(span.z) { witness=p; return true; }
        for(uint c=span.x;c<span.x+span.y;++c) {
            if(cardiacInsideCertifiedTube(heartChildDomains[c],q)) {
                witness=uint(sizeof(heartParentDomains)/sizeof(heartParentDomains[0]))+c;
                return true;
            }
        }
    }
    return false;
}
''')
def main(shader,host,proof,app,out,source):
    from check_heart_proof import check
    check(proof,source)
    out.mkdir(exist_ok=True);(out/'metalrobo').mkdir(exist_ok=True)
    ctr=json.loads((shader/'tube-contract.json').read_text())
    assert ctr['source_anatomy_sha256']=='3b0acb6146358f3302f2b85e2214bafa9f2fae7b0e51af70fd0838d6609da4a4' and ctr['source_field_sha256']=='0d88db155f7dfe9a93030badf6e570f49050164a7d0aa0ae011065aeada7d3da'
    for p in ('cardiac_certified_tube.metalinc','cardiac_proved_table.metalinc'):
        shutil.copyfile(shader/p,out/p)
    src=(shader/'human_respiration.metal').read_text()
    loc='#include "cardiac_proved_table.metalinc"';assert src.count(loc)==1
    src=src.replace(loc,loc+'\n#include "native_cardiac_subdomains.metalinc"')
    needle='''    output[0].first=status?float4(NAN):float4(coordinates[0],coordinates[1],coordinates[2],coordinates[3]);'''
    a=src.index('if(status==5u)');b=src.index(needle,a)
    src=src[:b]+'''    if(status==5u) {
        uint witness=0;
        if(cardiacNestedDomain(coordinates,witness)) {
            // Original table identifiers may be sparse.
            matched=0x40000000u+witness;
            status=0u;
        }
    }
'''+src[b:]
    (out/'human_respiration.metal').write_text(src)
    (out/'native_cardiac_subdomains.metalinc').write_text(emit(json.loads((proof/'four-domains.json').read_text()),json.loads((proof/'refinement.json').read_text())))
    orig=json.loads((shader/'skin-build-commands.json').read_text())
    def shader_path(x):
        if x in [str(shader/'human_respiration.metal'),str(shader/'HumanRestingSkin.air'),str(shader/'metalrobo/HumanRestingSkin.metallib')]:
            return str(out/Path(x).relative_to(shader))
        return x
    cmds=[[shader_path(x) for x in cmd] for cmd in orig]
    for p in (*host.glob('*.hpp'),*host.glob('*.mm')):shutil.copyfile(p,out/p.name)
    shutil.copyfile(host/'metalrobo/MetalRobo.metallib',out/'metalrobo/MetalRobo.metallib')
    source_cmds=json.loads((host/'build-commands.json').read_text())
    def host_path(s):
        if s.startswith('-DNUMI_HUMAN_RESPIRATION_METALLIB='):return '-DNUMI_HUMAN_RESPIRATION_METALLIB="'+str(out/'metalrobo/HumanRestingSkin.metallib')+'"'
        return str(out/Path(s).name) if Path(s).parent==host else s
    cmds.extend([[host_path(x) for x in cmd] for cmd in source_cmds])
    (out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    for cmd in cmds:subprocess.run(cmd,cwd=app/'build-native',check=True)
    names=['human_respiration.metal','native_cardiac_subdomains.metalinc','cardiac_certified_tube.metalinc','cardiac_proved_table.metalinc','metalrobo/HumanRestingSkin.metallib','metalrobo/MetalRobo.metallib','numi-human-native']
    (out/'identities.json').write_text(json.dumps({n:sha(out/n) for n in names},indent=2)+'\n')
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
