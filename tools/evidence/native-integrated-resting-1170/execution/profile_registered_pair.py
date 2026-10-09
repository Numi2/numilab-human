"""Read closed registered runs; summarize existing native timing instruments."""
import argparse,hashlib,json,math,pathlib,re,statistics
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--study",type=pathlib.Path,required=True)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise ValueError("refusing existing summary")
    out={"scope":"Measured native instrumentation from completed registered arms; not an exclusive whole-host benchmark.","arms":{}}
    prefixes=("resting_native_profile","resting_render_profile","resting_present_profile")
    for trial in ("resting-baseline","resting-drive-half"):
        t=args.study/"trials"/trial
        receipt=t/"receipt.json"
        d=json.loads(receipt.read_text())["payload"]
        if d.get("failure") is not None or d.get("returncode")!=0:raise ValueError("failed registered arm "+trial)
        p=t/"output/scene/native.log"
        groups={k:{} for k in prefixes}
        summaries={}
        with p.open() as f:
            for line in f:
                prefix=line.split(" ",1)[0]
                if prefix in groups:
                    for key,value in re.findall(r"(\w+)=([^ ]+)",line):
                        if not key.endswith("_ms"):continue
                        v=float(value)
                        if not math.isfinite(v) or v<0:raise ValueError("invalid timing")
                        groups[prefix].setdefault(key,[]).append(v)
                elif line.startswith(("resting_integrated_body=completed ","resting_integrated_throughput ","resting_integrated_observer_profile ")):
                    key=line.split(" ",1)[0]
                    if key in summaries:raise ValueError("duplicate final instrument "+key)
                    summaries[key]=line.strip()
        if "resting_integrated_body=completed" not in summaries:raise ValueError("missing native completion")
        metrics={}
        for group,items in groups.items():
            metrics[group]={}
            for key,vs in items.items():
                ordered=sorted(vs)
                metrics[group][key]={"n":len(vs),"mean":statistics.fmean(vs),"median":statistics.median(vs),
                  "p95_observed":ordered[math.ceil(.95*len(vs))-1],"max":max(vs)}
        out["arms"][trial]={"native_log":str(p),"native_log_sha256":sha(p),
          "registered_receipt":str(receipt),"registered_receipt_sha256":sha(receipt),
          "metrics_ms":metrics,"authoritative_final_instruments":summaries,
          "sampling_note":"Native profile rows are existing observer samples, not an attribution of every component or an independent replacement for the final throughput instrument."}
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"sha256":sha(args.output)}))
if __name__=="__main__":main()
