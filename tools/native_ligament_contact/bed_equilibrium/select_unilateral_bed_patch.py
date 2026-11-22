"""Generate a provisional contact active set, with all proposed pressure rows free to lift off."""
import json,sys
from pathlib import Path
a,b=map(Path,sys.argv[1:]);j=json.loads(a.read_text());j['select_unilateral']=True;b.write_text(json.dumps(j,indent=2)+'\n')
