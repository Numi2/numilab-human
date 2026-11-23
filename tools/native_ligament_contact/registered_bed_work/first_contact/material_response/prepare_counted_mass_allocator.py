"""Add read-only augmentation telemetry to the existing conservative fixed-point preparation tool."""
import sys,subprocess
from pathlib import Path
from compile_connected_joint import digest
def run(original,out):
    s=original.read_text();old='while(bfs()){std::fill(cursor.begin(),cursor.end(),0);while(send(send,s,Integer(1)<<125)){}}'
    if s.count(old)!=1:raise ValueError('Unsupported allocator source')
    new='unsigned long long augmentations=0;while(bfs()){std::fill(cursor.begin(),cursor.end(),0);while(send(send,s,Integer(1)<<125)){++augmentations;}}std::fprintf(stderr,"augmentations=%llu\\n",augmentations);'
    file=out.with_suffix('.cpp');file.write_text(s.replace(old,new));subprocess.run(['xcrun','clang++','-O2','-std=c++17',str(file),'-o',str(out)],check=True)
    print(digest(original),digest(file),digest(out))
if __name__=='__main__':run(*(Path(x) for x in sys.argv[1:]))
