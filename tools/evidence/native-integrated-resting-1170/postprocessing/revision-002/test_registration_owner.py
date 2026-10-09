import copy, importlib.util, json, unittest
from pathlib import Path
P=Path(__file__).with_name("analyze_final_pair.py")
s=importlib.util.spec_from_file_location("fixed_pair",P)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class RegistrationOwnerTests(unittest.TestCase):
    def setUp(self):
        self.source=json.loads(Path("/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/registration.json").read_text())["payload"]["source"]
    def test_sealed_human_owner(self):
        m.validate_registration_source(self.source)
        self.assertNotEqual(self.source["revision"],m.EXPECTED_REVISION)
    def test_lab_revision_cannot_impersonate_human(self):
        self.source["revision"]=m.EXPECTED_REVISION
        with self.assertRaisesRegex(ValueError,"Human revision"):m.validate_registration_source(self.source)
    def test_wrong_repository_rejected(self):
        self.source["repository"]=str(m.LAB)
        with self.assertRaisesRegex(ValueError,"Human repository"):m.validate_registration_source(self.source)
    def test_dirty_owner_rejected(self):
        self.source["status"]=" M x"
        with self.assertRaisesRegex(ValueError,"clean source"):m.validate_registration_source(self.source)
if __name__=="__main__":unittest.main()
