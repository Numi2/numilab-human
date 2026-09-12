"""Synthetic controls never claim case18 anatomy or physical simulation."""
import copy
from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
from numilab_human import cardiac_material_attribution as a
from numilab_human import cardiac_material_frames as f
from numilab_human.model import ImportError as HumanImportError
from tests.test_cardiac_material_frames import fixture as base_fixture

def fixture(asset, labels=(1,11)):
    parent,parent_sha,pins,_=base_fixture(asset)
    raw=struct.pack('<2I',*labels);(asset/'labels.u32le').write_bytes(raw)
    pins['labels.u32le']=(f.sha256(raw),[2],len(raw))
    m=f._read_json((asset/'manifest.json').read_bytes())
    m['buffers']['labels.u32le']['sha256']=pins['labels.u32le'][0]
    m['topology']['regional_cell_counts']={str(v):labels.count(v) for v in sorted(set(labels))}
    m['topology']['regional_geometric_volume_m3']={str(v):float(Fraction(labels.count(v),6)) for v in sorted(set(labels))}
    raw=f.canonical(m);(asset/'manifest.json').write_bytes(raw)
    c,_=a.load_config();c['asset_manifest_sha256']=f.sha256(raw)
    c['parent_source_config']['sha256']=parent_sha
    return c,parent,parent_sha,pins

def prepare(asset,output,contract,config_path=None):
    c,parent,parent_sha,pins=contract
    return a._prepare(asset,output,c,f.canonical(c),parent,parent_sha,pins,
        source_kind='synthetic_fixture_only',config_path=config_path)

class AttributionTests(unittest.TestCase):
    def test_pinned_sidecar_density_scope(self):
        c,raw=a.load_config();self.assertEqual(f.sha256(raw),a.CONFIG_SHA256)
        self.assertEqual(c['parent_source_config']['sha256'],f.wall.CONFIG_SHA256)
        self.assertEqual(c['asset_manifest_sha256'],'e8cb0391623577efc4eac04e5710cf7c9a4757614e09d936f5af3889c37d56f1')
        d=c['lv_geometric_mass_attribution']
        self.assertEqual(d['density_kg_per_m3'],d['source_density_value']*d['si_multiplier'])
        self.assertEqual(d['source_label'],1);self.assertFalse(d['mass_is_native_inertia'])
        self.assertTrue(all(x['inertial_density_kg_per_m3'] is None for x in c['passive_classes']))

    def test_exact_complete_label_lookup_with_unresolved_caps(self):
        c,_=a.load_config()
        self.assertEqual(a.source_class_map(c),{**{v:0 for v in (1,2)},
            **{v:1 for v in (3,4,18,19,20,21,22,23,24)},5:2,6:3,
            **{v:4 for v in (7,8,9,10)},**{v:a.UNRESOLVED for v in range(11,18)}})

    def test_duplicate_missing_unknown_classes_and_density_reject(self):
        original,_=a.load_config()
        changes=[lambda c:c['passive_classes'][1]['source_labels'].append(1),
            lambda c:c['passive_classes'][0]['source_labels'].remove(1),
            lambda c:c['passive_classes'][0]['source_labels'].append(25),
            lambda c:c['passive_classes'][0].update(class_id=True),
            lambda c:c['passive_classes'][1].update(class_id=0),
            lambda c:c['passive_classes'][0].update(inertial_density_kg_per_m3=1050),
            lambda c:c['assignment_policy'].update(class_id_is_native_material_index=True)]
        for change in changes:
            c=copy.deepcopy(original);change(c)
            with self.assertRaises(HumanImportError):a.source_class_map(c)

    def test_exact_mass_partial_map_identity_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=r/'asset';c=fixture(s);before={p.name:p.read_bytes() for p in s.iterdir()}
            result=prepare(s,r/'out',c);mass=result['lv_geometric_mass']
            self.assertEqual(struct.unpack('<2I',(r/'out'/a.BUFFER).read_bytes()),(0,a.UNRESOLVED))
            self.assertEqual(result['unresolved_cells'],1)
            self.assertEqual(mass['volume_exact'],a.exact_record(Fraction(1,6)))
            self.assertEqual(mass['mass_exact'],a.exact_record(Fraction(175)))
            self.assertEqual(mass['mass_kg'],175)
            self.assertFalse(result['qualification']['complete_passive_material_map'])
            self.assertFalse(result['qualification']['native_cooking_ready'])
            self.assertFalse(mass['blood_mass_partitioned'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in s.iterdir()})
            self.assertEqual(result['source_identity_sha256'],f.sha256(f.canonical(result['source_identity_record'])))

    def test_existing_output_has_no_staging_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=r/'asset';c=fixture(s);one=prepare(s,r/'out',c)
            before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (r/'out').iterdir()}
            with mock.patch.object(a.tempfile,'mkdtemp',side_effect=AssertionError('repeat staging')):
                self.assertEqual(one,prepare(s,r/'out',c))
            self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (r/'out').iterdir()})

    def test_complete_classes_and_false_flags_cannot_promote_native(self):
        for labels in ((1,11),(1,2)):
            with tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);s=r/'asset';result=prepare(s,r/'out',fixture(s,labels))
                result['qualification'].update(native_cooking_ready=True,native_inertial_density_map=True)
                result['unresolved_cells']=0
                with self.assertRaisesRegex(HumanImportError,'native cooking denied'):a.require_native_ready(result)

    def test_config_hash_duplicate_nonfinite_and_symlink_reject(self):
        c,raw=a.load_config()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';c['lv_geometric_mass_attribution']['density_kg_per_m3']=1000
            for value in (f.canonical(c),raw+b' ',b'{"schema":1,"schema":2}',b'{"x":NaN}',b'{"x":1e999}'):
                p.write_bytes(value)
                with self.assertRaises((HumanImportError,ValueError)):a.load_config(p)
            p.unlink();p.symlink_to(a.CONFIG)
            with self.assertRaises(HumanImportError):a.load_config(p)

    def test_public_route_rejects_fake_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=r/'asset';fixture(s)
            with self.assertRaises(HumanImportError):a.prepare_material_attribution(s,r/'out')
            self.assertFalse((r/'out').exists())

    def test_changed_raw_input_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=r/'asset';c=fixture(s)
            (s/'labels.u32le').write_bytes(struct.pack('<2I',1,2))
            with self.assertRaises(HumanImportError):prepare(s,r/'out',c)
            self.assertFalse((r/'out').exists())

    def test_rehashed_bad_geometry_or_label_rejects(self):
        cases=[('labels.u32le',struct.pack('<2I',1,25)),
            ('tetrahedra.u32le',struct.pack('<8I',0,2,1,3,0,2,1,4)),
            ('tetrahedra.u32le',struct.pack('<8I',0,0,1,3,0,2,1,4)),
            ('tetrahedra.u32le',struct.pack('<8I',0,1,2,999,0,2,1,4)),
            ('nodes.f64le',struct.pack('<15d',float('nan'),*([0.]*14)))]
        for name,raw in cases:
            with tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);s=r/'asset';c=fixture(s);config,_,_,pins=c
                (s/name).write_bytes(raw);pins[name]=(f.sha256(raw),pins[name][1],len(raw))
                m=f._read_json((s/'manifest.json').read_bytes());m['buffers'][name]['sha256']=pins[name][0]
                encoded=f.canonical(m);(s/'manifest.json').write_bytes(encoded);config['asset_manifest_sha256']=f.sha256(encoded)
                with self.subTest(name=name),self.assertRaises(HumanImportError):prepare(s,r/'out',c)
                self.assertFalse((r/'out').exists());self.assertFalse(list(r.glob('.cardiac-material-attribution-*')))

    def test_source_summary_drift_rejects(self):
        for key in ('regional_cell_counts','regional_geometric_volume_m3'):
            with tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);s=r/'asset';c=fixture(s)
                m=f._read_json((s/'manifest.json').read_bytes());m['topology'][key]['1']=999
                raw=f.canonical(m);(s/'manifest.json').write_bytes(raw);c[0]['asset_manifest_sha256']=f.sha256(raw)
                with self.assertRaisesRegex(HumanImportError,'summary mismatch'):prepare(s,r/'out',c)
                self.assertFalse((r/'out').exists())

    def test_input_mutation_after_consumption_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=r/'asset';c=fixture(s);original=a._write_or_compare
            def change(*args):
                result=original(*args);(s/'labels.u32le').write_bytes(struct.pack('<2I',1,2));return result
            with mock.patch.object(a,'_write_or_compare',side_effect=change),self.assertRaises(HumanImportError):prepare(s,r/'out',c)
            self.assertFalse((r/'out').exists())

    def test_config_mutation_during_conversion_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=r/'asset';c=fixture(s);p=r/'config.json';p.write_bytes(f.canonical(c[0]))
            original=a._write_or_compare
            def change(*args):
                result=original(*args);p.write_bytes(p.read_bytes()+b' ');return result
            with mock.patch.object(a,'_write_or_compare',side_effect=change),self.assertRaisesRegex(HumanImportError,'configuration changed'):
                prepare(s,r/'out',c,p)
            self.assertFalse((r/'out').exists())

    def test_changed_existing_output_not_overwritten(self):
        for name in (a.BUFFER,'manifest.json'):
            with tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);s=r/'asset';c=fixture(s);prepare(s,r/'out',c)
                p=r/'out'/name;changed=p.read_bytes()+b'x';p.write_bytes(changed)
                with self.assertRaises(HumanImportError):prepare(s,r/'out',c)
                self.assertEqual(p.read_bytes(),changed)

    def test_symlink_or_nested_output_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=r/'asset';c=fixture(s)
            with self.assertRaises(HumanImportError):prepare(s,s/'out',c)
            (r/'out').symlink_to(s,target_is_directory=True)
            with self.assertRaises(HumanImportError):prepare(s,r/'out',c)
            (s/'nodes.f64le').rename(r/'nodes');(s/'nodes.f64le').symlink_to(r/'nodes')
            with self.assertRaises(HumanImportError):prepare(s,r/'other',c)

    def test_cli_discovery_required_arguments(self):
        result=subprocess.run([str(a.ROOT/'.numi/commands/human-cardiac-material-attribution'),'--numi-describe'],
            text=True,capture_output=True,check=True)
        self.assertIn('unresolved',result.stdout)
        with self.assertRaises(SystemExit) as error:a.main([])
        self.assertEqual(error.exception.code,2)

if __name__=='__main__':unittest.main()
