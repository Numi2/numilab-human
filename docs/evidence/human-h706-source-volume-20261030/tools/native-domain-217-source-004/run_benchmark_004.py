from pathlib import Path
import subprocess,sys,json,time,hashlib
O=Path(__file__).resolve().parent
start=time.monotonic();result={}
for mode in ("prior","compact"):
    p=subprocess.run([sys.executable,str(O/"benchmark_volume_index_004.py"),mode],cwd=O,text=True,capture_output=True)
    (O/(mode+"-benchmark-process.json")).write_text(json.dumps(dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+"\n")
    if p.returncode:raise RuntimeError(mode+": "+p.stderr)
    result[mode]=json.loads(p.stdout)
if result["prior"]["output_sha256"]!=result["compact"]["output_sha256"]:raise RuntimeError("face incidence output parity failed")
result["output_parity"]=True
result["stage_wall_seconds"]=time.monotonic()-start
result["peak_rss_ratio"]=result["prior"]["peak_rss_bytes"]/result["compact"]["peak_rss_bytes"]
result["preparation_time_ratio"]=result["prior"]["total_index_and_boundary_query_seconds"]/result["compact"]["total_index_and_boundary_query_seconds"]
(O/"benchmark-receipt.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
