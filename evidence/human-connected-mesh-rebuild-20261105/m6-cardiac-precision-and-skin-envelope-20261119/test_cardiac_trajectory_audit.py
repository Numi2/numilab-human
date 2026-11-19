"""Check moving-axis and adaptive subdivision treatment of canonical contact."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

def simulate(binary, vertices, faces, offsets=None):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        packed=path/"case.bin"
        with packed.open("wb") as f:
            f.write(struct.pack("<III",len(vertices),len(faces),1))
            for i,v in enumerate(vertices):
                translation=(.01,.01,.01) if offsets is None else offsets[i]
                f.write(struct.pack("<24f",*v,*translation,*([0.0]*18)))
            for face in faces: f.write(struct.pack("<3I",*face))
            f.write(struct.pack("<28fI",-.125,*([0.]*6),.125,*([0.]*20),0))
        passed=subprocess.check_output([str(binary),str(packed),str(path/"result.csv"),"split-tube"],text=True)
        i,pairs,area,separate,shared_vertex,shared_edge,duplicate,split,visits=map(int,passed.split())
        assert i==0 and pairs>=max(separate,shared_vertex,shared_edge,duplicate)
        return area,separate,shared_vertex,shared_edge,duplicate

def main():
    binary=Path(sys.argv[1]).resolve()
    triangle=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)]
    trials=[
        (triangle+[(0.,-1.,0.)],[(0,1,2),(1,0,3)],(0,0,0,0,0),None),
        (triangle+[(.2,.25,0.)],[(0,1,2),(1,0,3)],(0,0,0,1,0),None),
        (triangle+[(.4,.4,-1.),(.4,.4,1.),(.2,.3,0.)],[(0,1,2),(3,4,5)],(0,1,0,0,0),None),
        (triangle,[(0,1,2),(1,0,2)],(0,0,0,0,1),None),
        (triangle+[(1.,1.,0.),(1.,.5,0.),(.5,1.,0.)],[(0,1,2),(3,4,5)],(0,0,0,0,0),None),
        (triangle+[(.1,.25,.1),(.5,.25,.1),(.15,.65,.1)],[(0,1,2),(3,4,5)],(0,1,0,0,0),
            [(0.,0.,0.)]*3+[(0.,0.,1.6)]*3)]
    for vertices,faces,expected,offsets in trials:
        assert simulate(binary,vertices,faces,offsets)==expected,(vertices,faces,expected)
    print("6 geometry fixtures passed, including a crossing strictly inside the pressure interval.")

if __name__=="__main__":
    main()
