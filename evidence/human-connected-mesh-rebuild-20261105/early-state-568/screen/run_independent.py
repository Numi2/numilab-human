from pathlib import Path
import json,subprocess,sys,time
O=Path(__file__).resolve().parent
results=[]
for i in range(4):
 start=time.time()
 with (O/('independent-field-'+str(i)+'.log')).open('w') as f:r=subprocess.run([sys.executable,str(O/'independent_response.py'),str(i)],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(i,r.returncode)
 report=json.loads((O/('independent-field-'+str(i)+'.json')).read_text());assert report['calibration_pass'] and report['force_failures']==report['energy_failures']==0,i
 results.append(dict(field_index=i,wall_s=time.time()-start,calibration_relative=report['calibration_relative'],force_failures=report['force_failures'],energy_failures=report['energy_failures']))
 (O/'independent-response-execution.json').write_text(json.dumps(results,indent=2)+'\n')
 print('PASS',i,flush=True)
