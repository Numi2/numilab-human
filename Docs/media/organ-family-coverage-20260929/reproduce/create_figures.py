from pathlib import Path
import json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
root=Path(__file__).resolve().parent;out=Path('Docs/media/organ-family-coverage-20260929');out.mkdir(parents=True,exist_ok=True)
d=json.loads((root/'payload.final/source-organ-family-anatomy.manifest.json').read_text());selection=d['selection'];baseline=set(selection['baseline_source_members'])
bg='#101c2b';white='#f1f4f8';muted='#bac8d8';accent='#56d9c3'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':white,'axes.facecolor':bg,'figure.facecolor':bg})
fig=plt.figure(figsize=(16,9),dpi=120)
fig.text(.05,.94,'Numi Human | Closing omitted organ-family members',fontsize=25,weight='bold')
fig.text(.05,.89,'18 declared atlas families · 378 unique members · actual source-to-native geometry',fontsize=14,color=muted)
ax=fig.add_axes([.055,.225,.455,.615]);ax.axis('off')
rows=[[r['source_name'],str(len(set(r['expected_members']) & baseline)),str(len(r['expected_members']))] for r in selection['families']]
t=ax.table(cellText=rows,colLabels=['Declared source family','Before','Now'],cellLoc='left',loc='center',colWidths=[.74,.13,.13]);t.auto_set_font_size(False);t.set_fontsize(11.4);t.scale(1,1.55)
for (r,c),cell in t.get_celld().items():
 cell.set_facecolor('#244055' if r==0 else '#172838');cell.set_edgecolor('#446074');cell.set_text_props(color=white,weight='bold' if r==0 else 'normal')
fig.text(.55,.76,'77 omitted members added',fontsize=22,color=accent,weight='bold')
for y,text in zip([.66,.59,.52,.45],['18 organ components','40 vessel segments','15 ducts','4 cardiac cavity references']):fig.text(.55,y,text,fontsize=19)
fig.text(.55,.335,'310 → 387 retained surfaces',fontsize=20,weight='bold')
fig.text(.55,.275,'The previous 310 surfaces are byte-identical.',fontsize=13,color=muted)
fig.text(.05,.13,'All 387 surfaces checked in 3 poses × 4 viewing profiles. Shared members appear once in the payload.',fontsize=13,color=accent)
fig.text(.05,.075,'Family counts overlap. Complete atlas membership does not establish whole-body or clinical completeness.',fontsize=12,color=muted)
fig.savefig(out/'executive-organ-family-progress.png',dpi=120);plt.close(fig)
fig=plt.figure(figsize=(16,9),dpi=120)
fig.text(.05,.94,'Numi Human | Ducts and cavity references',fontsize=25,weight='bold')
fig.text(.05,.89,'Actual native captures · separate source types · 29 September 2026',fontsize=14,color=muted)
for i,(mask,title) in enumerate([(256,'Four cardiac cavity references'),(512,'Fifteen pancreatic / biliary ducts')]):
 src=next((root/f'final-native/projected-neutral-mask{mask}/views').glob('*-oblique.png'));shutil.copy2(src,out/f'native-neutral-mask{mask}-oblique.png')
 image=mpimg.imread(src)[360:870,340:710]
 ax=fig.add_axes([.065+i*.38,.26,.34,.565]);ax.imshow(image);ax.axis('off')
 fig.text(.235+i*.38,.22,title,fontsize=14,weight='bold',ha='center')
fig.text(.825,.72,'Reference',fontsize=17,color=accent,weight='bold');fig.text(.825,.66,'geometry',fontsize=17,color=accent,weight='bold')
fig.text(.825,.54,'No admitted',fontsize=14);fig.text(.825,.49,'blood / tissue',fontsize=14);fig.text(.825,.44,'volumes',fontsize=14)
fig.text(.05,.135,'Preserved source defects: FJ2404, FJ2405 and FJ2434. The right atrium / ventricle overlap remains.',fontsize=12.5)
fig.text(.05,.082,'Overlapping aggregate / descendant atlas meshes are retained. Connected lumens and mechanics remain unverified.',fontsize=12,color=muted)
fig.savefig(out/'executive-organ-family-types.png',dpi=120);plt.close(fig)
print(out)
