"""Check the separation auditor rejects overlaps and preserves indexed seams."""
from pathlib import Path
import struct
import subprocess
import tempfile

BINARY=Path(__file__).resolve().parent/"right-heart-interval-correlated"

def run(vertices, faces, interval=.0):
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/"geometry"
        table=Path(directory)/"witnesses.csv"
        with source.open("wb") as out:
            out.write(struct.pack("<III",len(vertices),len(faces),1))
            for v in vertices:
                # A common displacement exercises cancellation across the interval.
                basis=(.01,.01,.01)+(0.0,)*18
                out.write(struct.pack("<24f",*v,*basis))
            for face in faces:
                out.write(struct.pack("<III",*face))
            lower=(-interval,)+(0.0,)*6
            upper=(interval,)+(0.0,)*6
            out.write(struct.pack("<14f",*lower,*upper))
        report=subprocess.check_output([str(BINARY),str(source),str(table),"fused"],text=True)
        idx,pairs,area,cross,one_vertex,one_edge,duplicate=map(int,report.split())
        assert idx==0
        return pairs,(area,cross,one_vertex,one_edge,duplicate)

def main():
    base=[(0.0,0.0,0.0),(1.0,0.0,0.0),(0.0,1.0,0.0)]
    good=base+[(0.0,-1.0,0.0)]
    assert run(good,[(0,1,2),(1,0,3)],.125)==(1,(0,0,0,0,0))
    fold=base+[(.25,.25,.0)]
    assert run(fold,[(0,1,2),(1,0,3)])==(1,(0,0,0,1,0))
    vertex=[(.5,.5,-.5),(.5,.5,.5),(.25,.25,0.0)]
    assert run(base+vertex,[(0,1,2),(3,4,5)])==(1,(0,1,0,0,0))
    assert run(base,[(0,1,2),(1,0,2)])==(1,(0,0,0,0,1))
    assert run(base+[(2.,2.,2.),(1.,2.,2.),(2.,1.,2.)],[(0,1,2),(3,4,5)])==(0,(0,0,0,0,0))
    print("five outward-rounded separation and failure fixtures passed")

if __name__=="__main__":
    main()
