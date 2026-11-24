"""Execute the production scheduler's coloring block on the current Metal device."""
from pathlib import Path
import json,subprocess,sys
from compile_connected_joint import digest
def run(build,out,execute=False):
    if execute:
        process=subprocess.run([str(out/'check'),str(out/'check.metallib')],capture_output=True,text=True,check=True)
        (out/'result.json').write_text(json.dumps(dict(native=process.stdout.strip(),shader=digest(out/'check.metal'),production=digest(build/'metal/src/metal/fem.metalinc')),indent=2)+'\n')
        return
    out.mkdir()
    p=build/'metal/src/metal/fem.metalinc';source=p.read_text();size=64 if 'kernel void nm_fem_registered_volume_active_incident64(' in source else 32
    s=source.split('kernel void nm_fem_registered_volume_active_incident'+('64' if size==64 else '')+'(',1)[1]
    start=s.index('    if(live==0u || group!=0u)return;');end=s.index('    // No node is read concurrently',start)
    coloring=s[start:end]
    shader='''#include <metal_stdlib>
using namespace metal;
struct Cell {uint4 nodes;};
kernel void checkColors(device const uint* rows [[buffer(0)]],device const Cell* tets [[buffer(1)]],device uint* result [[buffer(2)]],uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]){
    constexpr uint count=96;
    const uint live=rows[count];
    if(live>CAPACITY)return;
'''+coloring+'''
    if(lane<live)result[sorted[lane]]=color[lane];
}'''
    (out/'check.metal').write_text(shader.replace('CAPACITY',str(size)+'u').replace('if(lane<live)result[sorted[lane]]=color[lane];','for(uint i=lane;i<live;i+=32u)result[sorted[i]]=color[i];'))
    for cmd in [['xcrun','metal','-c',str(out/'check.metal'),'-o',str(out/'check.air')],['xcrun','metallib',str(out/'check.air'),'-o',str(out/'check.metallib')],['clang++','-DRAW_CAPACITY='+str(size),'-fobjc-arc','-std=c++17','-framework','Foundation','-framework','Metal',str(Path(__file__).parent/'test_incident_schedule.mm'),'-o',str(out/'check')]]:
        subprocess.run(cmd,check=True)
if __name__=='__main__':run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),len(sys.argv)>3)
