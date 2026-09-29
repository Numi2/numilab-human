"""Export all authored Z-Anatomy lung/pleural meshes and 27 thorax anchors.

Run in Blender with --disable-autoexec, then --output PATH --registration PATH.
Existing modifiers are retained; no repair or additional smoothing is applied.
"""
import bpy,json,hashlib,sys,argparse
from pathlib import Path
from mathutils import Vector
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--registration',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
reg=json.loads(args.registration.read_text())
mesh_names=[o.name for o in bpy.data.collections['Lungs'].objects if o.type=='MESH' and len(o.data.polygons)>0]
assert set(mesh_names)=={'Inferior lobe of left lung','Inferior lobe of right lung','Middle lobe of right lung','Superior lobe of left lung','Superior lobe of right lung'}
anchor_specs=[]
for a in reg['anchors']:
 s=a['source'];name=s['name']
 if ' rib' in name:
  side,num,_=name.split(' ');object_name=num.capitalize()+' rib.'+('r' if side=='right' else 'l')
 elif name in ['right clavicle','left clavicle']:object_name='Clavicle.'+('r' if name.startswith('right') else 'l')
 elif name=='body of sternum':object_name='Body of sternum'
 else:continue
 anchor_specs.append((object_name,s['member_id']))
deps=bpy.context.evaluated_depsgraph_get()
def export(name):
 o=bpy.data.objects[name];e=o.evaluated_get(deps);m=e.to_mesh()
 try:
  m.calc_loop_triangles();v=[list(e.matrix_world @ x.co) for x in m.vertices];f=[list(x.vertices) for x in m.loop_triangles]
  n=[Vector((0.,0.,0.)) for _ in v]
  for a,b,c in f:
   q=(Vector(v[b])-Vector(v[a])).cross(Vector(v[c])-Vector(v[a]))
   for i in [a,b,c]:n[i]+=q
  ns=[]
  for i,q in enumerate(n):
   if q.length<=1e-12:q=Vector(v[i])-sum((Vector(z) for z in v),Vector((0,0,0)))/len(v)
   assert q.length>1e-12,(name,i)
   ns.append(list(q.normalized()))
  modifiers=[{'name':s.name,'type':s.type,'show_viewport':s.show_viewport,'show_render':s.show_render,**({'levels':s.levels,'render_levels':s.render_levels} if s.type=='SUBSURF' else {}),**({'thickness':s.thickness,'offset':s.offset} if s.type=='SOLIDIFY' else {})} for s in o.modifiers]
  return {'object_name':name,'vertices_world_m':v,'triangles':f,'normals_world':ns,'polygons':[list(p.vertices) for p in m.polygons],'matrix_world':[list(row) for row in e.matrix_world],'modifiers':modifiers,'raw_vertex_count':len(o.data.vertices),'raw_polygon_count':len(o.data.polygons)}
 finally:e.to_mesh_clear()
organs=[export(n) for n in sorted(mesh_names)]+[export('Pleura')]
anchors=[]
for name,member in anchor_specs:
 r=export(name);r['bodyparts_member_id']=member;anchors.append(r)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'schema':'numi.human.zanatomy-thorax-blender-export.v1','source':{'blend_sha256':sha(bpy.data.filepath),'blend_file':Path(bpy.data.filepath).name,'blender_version':bpy.app.version_string,'exporter_sha256':sha(__file__),'scene_unit_scale':bpy.context.scene.unit_settings.scale_length,'coordinates':'evaluated Blender world metres','triangulation':'Blender calc_loop_triangles without additional modifiers, smoothing, decimation or geometry repair','scripts_autoexec':False,'registration_sha256':sha(args.registration)},'source_lung_mesh_members':sorted(mesh_names),'objects':organs,'registration_anchors':anchors}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,separators=(',',':'))+'\n')
print('exported',[(r['object_name'],len(r['vertices_world_m']),len(r['triangles']),r['modifiers']) for r in organs]);print('anchors',len(anchors))
