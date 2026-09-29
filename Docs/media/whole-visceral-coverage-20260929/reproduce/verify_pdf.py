from pathlib import Path
import subprocess,hashlib,json
from pypdf import PdfReader
b=Path('/Users/home/NumiSuiteReports');old=b/'board-20260929-organ-families/Numi-Suite-Board-Visuals-organ-family-update.pdf';out=b/'board-20260929-whole-visceral';new=out/'Numi-Suite-Board-Visuals-whole-visceral-update.pdf'
tool='/Users/home/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm'
assert len(PdfReader(old).pages)==28 and len(PdfReader(new).pages)==30
rows=[]
for page in range(1,29):
 data=[]
 for p in [old,new]:
  cmd=[tool,'-f',str(page),'-l',str(page),'-scale-to','800','-singlefile','-png',str(p)]
  r=subprocess.run(cmd,capture_output=True,check=True);data.append(r.stdout)
 assert data[0]==data[1],page
 rows.append({'page':page,'pixel_render_byte_identical':True,'png_sha256':hashlib.sha256(data[0]).hexdigest()})
report={'old_pages':28,'new_pages':30,'all_original_pages_pixel_identical':True,'scale_to':800,'rows':rows}
(out/'pdf-page-parity.json').write_text(json.dumps(report,indent=2)+'\n')
for page in [29,30]:
 cmd=[tool,'-f',str(page),'-l',str(page),'-scale-to','1600','-singlefile','-png',str(new)]
 (out/f'review-page-{page}.png').write_bytes(subprocess.run(cmd,capture_output=True,check=True).stdout)
print('28 original PDF pages pixel-identical; 30 pages total')
