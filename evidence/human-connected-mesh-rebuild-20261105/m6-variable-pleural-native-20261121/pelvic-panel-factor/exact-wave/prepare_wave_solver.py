from pathlib import Path
import json,sys
p=Path(sys.argv[1]);a=Path('motion-registry-owner/pelvic_contact.metalinc').read_text()
s=a[a.index('kernel void nm_pelvic_bulk_factor('):a.index('kernel void nm_pelvic_require_bulk')]
s=s.replace('void nm_pelvic_bulk_factor','void nm_pelvic_wave_factor').replace('i+=128','i+=32').replace('for(uint k=sg;k<s.y;k+=4)','for(uint k=0;k<s.y;++k)').replace('threadgroup_barrier(mem_flags::mem_threadgroup)','simdgroup_barrier(mem_flags::mem_threadgroup)')
s=s.replace(',uint sg [[simdgroup_index_in_threadgroup]]','')
(p/'pelvic_wave_factor.metalinc').write_text(s)
v=json.loads(Path('repartition/rounded-work/motion-p1-boundary/manifest.json').read_text())['regional_factor']
head=(p/'bench.metal').read_text().split('// Each block')[0]
head+='constant uint2 pelvicFactorUpper[2]={'+','.join('uint2(%d,%d)'%tuple(d['upper']) for d in v)+'};\n'
(p/'wave.metal').write_text(head+s)
q=Path('compare_pelvic_factors.mm').read_text().replace('@"nm_pelvic_panel_factor"','@"nm_pelvic_wave_factor"')
start=q.index('        for(unsigned i=0;i<4;++i)b[i]=blob');end=q.index('        auto queue=',start)
q=q[:start]+q[end:];q=q.replace('assets=argv[3],parts=argv[4],inputs','assets=argv[3],inputs')
q=q.replace('threadsPerThreadgroup:MTLSizeMake(128,1,1)','threadsPerThreadgroup:MTLSizeMake(usePanel?32:128,1,1)')
(p/'compare-wave.mm').write_text(q)
