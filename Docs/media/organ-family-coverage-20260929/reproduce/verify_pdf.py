from pathlib import Path
import subprocess,hashlib,json
from pypdf import PdfReader
base=Path('/Users/home/NumiSuiteReports');old=base/'board-20260929-lung-envelopes/Numi-Suite-Board-Visuals-lung-envelope-update.pdf';new=base/'board-20260929-organ-families/Numi-Suite-Board-Visuals-organ-family-update.pdf'
tool='/Users/home/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm'
assert len(PdfReader(old).pages)==26 and len(PdfReader(new).pages)==28
rows=[]
for page in range(1,27):
 data=[]
 for p in [old,new]:
  cmd=[tool,'-f',str(page),'-l',str(page),'-scale-to','800','-singlefile','-png',str(p)]
  r=subprocess.run(cmd,capture_output=True,check=True);data.append(r.stdout)
 assert data[0]==data[1],page
 rows.append({'page':page,'pixel_render_byte_identical':True,'png_sha256':hashlib.sha256(data[0]).hexdigest()})
report={'old_pages':26,'new_pages':28,'all_original_pages_pixel_identical':True,'scale_to':800,'rows':rows}
(base/'board-20260929-organ-families/pdf-page-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print('26 original PDF pages pixel-identical; 28 pages total')
