from pathlib import Path
import numpy as np,json
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');root=H/'tetgen-scale-relative-tolerance-059'
out=root/'exact-cell-orientation-061';out.mkdir(exist_ok=True)
def exact_det(p):
 ratios=[[float(v).as_integer_ratio() for v in row]for row in p]
 e=max(d.bit_length()-1 for row in ratios for n,d in row)
 a=[[n<<(e-(d.bit_length()-1)) for n,d in row]for row in ratios]
 u,v,w=[[a[j][k]-a[0][k]for k in range(3)]for j in (1,2,3)]
 d=u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])
 return d,e*3
reports={}
for role,run in [('fat','fat009-run001'),('skin','skin010-run001')]:
 path=root/run
 nodes=np.loadtxt(path/(role+'.1.node'),skiprows=1,comments='#');els=np.loadtxt(path/(role+'.1.ele'),skiprows=1,comments='#',dtype=np.int64)
 p=nodes[:,1:4];p32=p.astype(np.float32).astype(np.float64);t=els[:,1:5]-1
 q=p32[t];d=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0]);bad=np.flatnonzero(d<=0)
 rows=[]
 for i in bad:
  d32,e32=exact_det(p32[t[i]]);d64,e64=exact_det(p[t[i]])
  rows.append({'element_row':int(i),'element_id':int(els[i,0]),'node_ids':els[i,1:5].tolist(),'material':int(els[i,5]),'standard_f64_on_f32_det6':float(d[i]),'exact_f32_det6_numerator':str(d32),'exact_f32_det6_denominator_exp2':e32,'exact_f32_sign':int(d32>0)-int(d32<0),'exact_f64_sign':int(d64>0)-int(d64<0),'exact_f64_det6_numerator':str(d64),'exact_f64_det6_denominator_exp2':e64})
 reports[role]={'cells':len(t),'standard_nonpositive':len(bad),'exact_zero':sum(r['exact_f32_sign']==0 for r in rows),'exact_negative':sum(r['exact_f32_sign']<0 for r in rows),'standard_false_zero_or_sign_error':sum(int(np.sign(r['standard_f64_on_f32_det6']))!=r['exact_f32_sign']for r in rows),'rows':rows}
(out/'report.json').write_text(json.dumps({'status':'DIAGNOSTIC_EXACT_BINARY_RATIONAL_ORIENTATIONS','physical_admission':False,'runtime':0,'accepted_steps':0,'roles':reports},indent=2)+'\n')
print(json.dumps({r:{k:v for k,v in d.items()if k!='rows'}for r,d in reports.items()}))
