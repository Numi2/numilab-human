"""Reuse the existing interval prover on the parent's unresolved face pairs."""
from pathlib import Path
import subprocess,sys
src,out=map(Path,sys.argv[1:])
s=src.read_text()
start=s.index('    const std::string_view requested(argv[3]);')
entry=s.rindex('    if(argc',0,start)
s=s[:entry]+'''    if(argc!=5) throw std::runtime_error("usage: probe subset.bin witness.csv split-tube pairs.bin");
    std::ifstream listed(argv[4],std::ios::binary);
    uint32_t pairCount{};take(listed,pairCount);
    const auto unresolved=readMany<std::array<uint32_t,2>>(listed,pairCount);
    if(listed.peek()!=EOF) throw std::runtime_error("unexpected pair list trailer");
'''+s[start:]
a=s.index('    const unsigned first=argc');b=s.index('    if(first>=end',a)
s=s[:a]+'    const unsigned first=0,end=windows;\n'+s[b:]
a=s.index('        s.order.resize(nf);');b=s.index('        witnesses.flush();',a)
s=s[:a]+'''        uint64_t candidatePairs=0,refined=0,visits=0;
        for(const auto& pair:unresolved) {
            const uint32_t i=pair[0], k=pair[1];
            if(i>=k||k>=nf) throw std::runtime_error("invalid inherited pair");
            const auto& a=s.triangles[i];const auto& other=s.triangles[k];
            if(!a.bounds.overlaps(other.bounds)) continue;
            ++candidatePairs;
            const auto status=s.certify(a,other);
            if(status&&status!=4) {
                PairRefinement candidate(s,reserve,a.ids,other.ids);
                const bool ok=candidate.prove(lower,upper,0);
                visits+=255u-candidate.budget;
                refined+=ok;
                if(ok) continue;
            }
            if(status) {
                if(fails[status]<witnessBudget) witnesses<<w<<','<<status<<','<<i<<','<<k<<'\\n';
                ++fails[status];
            }
        }
'''+s[b:]
p=out.with_suffix('.cpp');p.write_text(s)
subprocess.run(['/usr/bin/clang++','-std=c++20','-O3','-fno-fast-math','-ffp-contract=off',str(p),'-o',str(out)],check=True)
