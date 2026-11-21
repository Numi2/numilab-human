"""Keep all undecided parent pairs for inherited finer certification."""
import hashlib,sys,subprocess
from pathlib import Path
src,out=map(Path,sys.argv[1:])
s=src.read_text()
s='#include <atomic>\n#include <future>\n#include <thread>\n'+s
begin=s.index('        uint64_t candidatePairs=0,refined=0,visits=0;')
end=s.index('        witnesses.flush();',begin)
base=s[begin:end]
a=base.index('                    if(status&&status!=4)');b=base.index('                    if(status) {',a)
base=base[:a]+base[b:]
walk=base[base.index('            const auto& a=s.triangles[i];'):base.rindex('        }')]
a=walk.index('                        if(fails[status]<witnessBudget)')
b=walk.index('                    }',a)
walk=walk[:a]+'''                        batch.records.push_back({status,i,k});
'''+walk[b:]
walk=walk.replace('++candidatePairs;', '++batch.pairs;')
new='''        // A complete negative list is cheaper than recursive proofs over a wide
        // region. Only these pairs need another proof on its subdomains.
        struct Batch {
            uint64_t pairs = 0;
            std::vector<std::array<uint32_t,3>> records;
        };
        constexpr uint32_t stride=1024;
        std::vector<Batch> batches((nf+stride-1)/stride);
        std::atomic<uint32_t> cursor=0;
        auto worker=[&] {
            for (;;) {
                const auto b=cursor.fetch_add(1,std::memory_order_relaxed);
                if (b>=batches.size()) return;
                auto& batch=batches[b];
                for(uint32_t i=b*stride;i<std::min(nf,(b+1)*stride);++i) {
'''+walk+'''
                }
            }
        };
        std::vector<std::future<void>> workers;
        for(uint32_t i=0;i<std::min<uint32_t>(6,batches.size());++i)
            workers.push_back(std::async(std::launch::async,worker));
        for(auto& job:workers) job.get();
        uint64_t candidatePairs=0,refined=0,visits=0;
        for(const auto& batch:batches) {
            candidatePairs+=batch.pairs;
            for(const auto& v:batch.records) {
                witnesses<<w<<','<<v[0]<<','<<v[1]<<','<<v[2]<<'\\n';
                ++fails.at(v[0]);
            }
        }
'''
s=s[:begin]+new+s[end:]
s=s.replace('if(fails[0]<witnessBudget) witnesses<<','witnesses<<')
p=out.with_suffix('.cpp');p.write_text(s)
subprocess.run(['/usr/bin/clang++','-std=c++20','-O3','-fno-fast-math','-ffp-contract=off',str(p),'-o',str(out)],check=True)
