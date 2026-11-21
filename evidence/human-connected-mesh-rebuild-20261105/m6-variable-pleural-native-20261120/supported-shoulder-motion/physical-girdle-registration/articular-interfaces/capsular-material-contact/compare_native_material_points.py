"""Cross-check every emitted material point using the existing native bone baker."""
from pathlib import Path
import json,subprocess,sys
import numpy as np

def run(catalog,project,source,bone,sites,out):
    code=(catalog/'cook_checked.cpp').read_text()
    header=code[:code.index('    const auto rigidMagic')]
    header=header.replace('if (argc != 3)', 'if (argc != 2)').replace(', rigid(argv[2], std::ios::binary)','')
    check=header+"""
    unsigned sid, face; double u,v,w;
    MRBodyStateGPU pose{}; pose.orientation.w=1.0f;
    while (std::cin >> sid >> face >> u >> v >> w) {
        const auto it=std::find_if(records.begin(),records.end(),[&](const auto& r){return r.stableId==sid;});
        if (it==records.end()) return 5;
        auto s=inferred_acl::bakeRegisteredBoneSurface(*it,it->sourceRecord,std::span<const BoneVertex>(vertices),
            std::span<const uint32_t>(indices),pose,lig::kReferenceBonesPayloadSha256);
        using namespace inferred_acl::detail;
        auto p=add(add(mul(s.vertices.at(s.triangles.at(3*face)),u),
                       mul(s.vertices.at(s.triangles.at(3*face+1)),v)),
                       mul(s.vertices.at(s.triangles.at(3*face+2)),w));
        std::cout << std::setprecision(17) << p[0] << ' ' << p[1] << ' ' << p[2] << '\\n';
    }
    return 0;
}
"""
    p=out.parent/'compare_emitted_sites.cpp';p.write_text(check)
    exe=p.with_suffix('')
    cmd=['/usr/bin/clang++','-std=c++23','-O2','-I'+str(catalog),'-I'+str(project/'floor-scene-build'),'-I'+str(source/'include'),
        '-I'+str(source/'matter/include'),'-I'+str(source/'matter/tools'),str(p),'-o',str(exe)]
    subprocess.run(cmd,check=True)
    rows=[e for r in json.loads(sites.read_text())['rows'] for e in r['attachments']]
    request=''.join(' '.join(map(str,[e['stable'],e['triangle'],*e['barycentric']]))+'\n' for e in rows)
    result=subprocess.run([exe,bone],input=request,text=True,check=True,capture_output=True)
    points=np.asarray([list(map(float,s.split())) for s in result.stdout.splitlines()])
    errors=np.linalg.norm(points-np.asarray([e['local_m'] for e in rows]),axis=1)
    if points.shape!=(4,3) or not np.all(errors<1.e-12):raise ValueError('Native material evaluation differs')
    out.write_text(json.dumps(dict(maximum_local_error_m=errors.max(),errors_m=errors.tolist(),
       material_faces=[[e['stable'],e['triangle']] for e in rows]),indent=2)+'\n')
if __name__=='__main__':run(*[Path(p).resolve() for p in sys.argv[1:]])
