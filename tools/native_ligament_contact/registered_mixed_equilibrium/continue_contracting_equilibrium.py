"""Continue the convergent mixed solve under the existing GPU acceptance latch."""
from pathlib import Path
import sys,shutil
src,dst=map(Path,sys.argv[1:]);dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
def replace(p,a,b):
    s=p.read_text()
    if s.count(a)!=1:raise ValueError(a)
    p.write_text(s.replace(a,b))
p=dst/'pelvic_contact.metalinc';replace(p,'uint base=info.x*18+side;','uint base=info.x*50+side;')
p=dst/'NumiHumanPelvicContact.hpp'
for a,b in [('countTrace)*18u*16u','countTrace)*50u*16u'),('attempt<8;++attempt','attempt<24;++attempt'),('j<9;++j)for(unsigned b=0;b<2;++b)','j<25;++j)for(unsigned b=0;b<2;++b)'),('z[18*k+2*j+b]','z[50*k+2*j+b]')]:replace(p,a,b)
