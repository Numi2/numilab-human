import datetime,hashlib,json,os,pathlib,subprocess,sys,time
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
args=sys.argv[2:]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
with (out/'output.log').open('wb') as f:
 p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
record={'schema':'numi-command-execution-v1','argv':args,'cwd':os.getcwd(),'startedUTC':start,'finishedUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsedSeconds':time.monotonic()-t,'exitCode':p.returncode,'outputSHA256':hashlib.sha256((out/'output.log').read_bytes()).hexdigest()}
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));print((out/'output.log').read_text(errors='replace')[-18000:]);sys.exit(p.returncode)
