"""Run one bounded parallel authoring audit and retain every rejected interval."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import json
from pathlib import Path
import subprocess

def main():
    parse=argparse.ArgumentParser(description=__doc__)
    parse.add_argument("audit_binary",type=Path)
    parse.add_argument("atlas",type=Path)
    parse.add_argument("output",type=Path)
    parse.add_argument("--workers",type=int,choices=range(1,4),default=3)
    args=parse.parse_args()
    source=json.loads(args.atlas.with_suffix(".json").read_text())
    count=len(source["windows"])
    args.output.mkdir()
    width=(count+args.workers-1)//args.workers
    ranges=[(s,min(s+width,count)) for s in range(0,count,width)]
    def work(interval):
        start,end=interval
        stem=args.output / f"windows-{start:03d}-{end:03d}"
        log=stem.with_suffix(".log")
        witnesses=stem.with_suffix(".csv")
        with log.open("w") as stdout,stem.with_suffix(".stderr").open("w") as stderr:
            subprocess.run([str(args.audit_binary.resolve()),str(args.atlas),str(witnesses),"split-tube",
                            str(start),str(end)],stdout=stdout,stderr=stderr,check=True)
        return [tuple(map(int,l.split())) for l in log.read_text().splitlines()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        parts=list(pool.map(work,ranges))
    records=[r for part in parts for r in part]
    assert [r[0] for r in records]==list(range(count))
    traces=[dict(index=r[0],source_window=source["windows"][r[0]],candidate_pairs=r[1],
                 degenerate_uncertainties=r[2],nonadjacent_uncertainties=r[3],
                 shared_vertex_uncertainties=r[4],shared_edge_uncertainties=r[5],
                 duplicate_uncertainties=r[6],successful_pair_refinements=r[7],refinement_calls=r[8])
            for r in records]
    passed=[t["index"] for t,r in zip(traces,records) if not any(r[2:7])]
    result=dict(intervals=traces,passed_windows=passed,total_windows=count,
                all_bounds_proved=len(passed)==count,native_admission_changed=False)
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"{len(passed)} of {count} intervals provided complete geometry bounds.")

if __name__=="__main__":
    main()
