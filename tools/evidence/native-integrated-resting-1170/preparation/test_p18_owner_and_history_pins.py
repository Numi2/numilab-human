import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("prepare_final_plan_v018",ROOT/"prepare_final_plan-v018.py")
prep=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(prep)
class P18OwnerAndHistoryPins(unittest.TestCase):
 def test_terminal_capture_owner_is_exact_fixed_revision(self):
  self.assertEqual(prep.PREPARATION_OWNER_REV,"f4f1d1d4c35ea31bce7669c329a6071176c30afd")
  self.assertEqual(prep.PREPARATION_OWNER_SHA,"ef87b09a17b96c985d6f584932e806ba9e7c6a53785da32a37a56ac846a3ac5f")
  self.assertEqual(prep.PREPARATION_OWNER_TEST_SHA,"8ac11ae28bc3fd059103e8255ca4b7b94075fb0ddca8cecf300d3fe7b4397c8b")
  self.assertEqual(prep.sha(prep.OWNER),prep.PREPARATION_OWNER_SHA)
  self.assertEqual(prep.sha(prep.PREPARATION_OWNER_TEST),prep.PREPARATION_OWNER_TEST_SHA)
  head=subprocess.check_output(["git","-C",str(prep.PREPARATION_OWNER_ROOT),"rev-parse","HEAD"],text=True).strip()
  status=subprocess.check_output(["git","-C",str(prep.PREPARATION_OWNER_ROOT),"status","--porcelain=v1"],text=True)
  self.assertEqual(head,prep.PREPARATION_OWNER_REV); self.assertEqual(status,"")
 def test_full_q_reference_is_regenerated_1173_and_old_931_raw_run_is_not_required(self):
  self.assertEqual(prep.FULL_Q_SHA,"24312b80992534d2d52c68818a3472994e5a005b20a7b6262e6448fe6fc057b7")
  self.assertEqual(prep.TERMINAL_1173_VERIFICATION_SHA,"13755a72927eb6fe38cd98d1b5512daf6ecbe1faa859b11ad147d71e8367bd16")
  self.assertTrue(prep.FULL_Q.is_file()); self.assertEqual(prep.sha(prep.FULL_Q),prep.FULL_Q_SHA)
  report=json.loads(prep.FULL_Q_VERIFICATION.read_text())
  self.assertTrue(report["pass"]); self.assertEqual(report["terminal"]["accepted_step"],10000)
  self.assertFalse((prep.E/"native-terminal-cycle-931/run-metadata.json").exists())
  self.assertTrue(prep.TERMINAL_931_HISTORICAL_VERIFICATION.is_file())
  self.assertEqual(prep.sha(prep.TERMINAL_931_HISTORICAL_VERIFICATION),prep.TERMINAL_931_HISTORICAL_VERIFICATION_SHA)
 def test_917_history_is_pinned_but_removed_trial_is_not_reconstructed(self):
  self.assertEqual(str(prep.STUDY),"/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170")
  self.assertFalse(prep.STUDY.exists())
  for path,digest in prep.STUDY_917_HISTORY:
   self.assertTrue(path.is_file(),str(path)); self.assertEqual(prep.sha(path),digest,str(path))
  incident=json.loads(prep.STUDY_917_HISTORY[0][0].read_text())
  self.assertIn("Original raw trial files are now absent",incident["evidence_status"])
  old=Path("/Users/n/numi-human-resting-evidence-20261005/native-integrated-resting-study-917/trials/resting-baseline/output/scene")
  self.assertFalse(old.exists())
if __name__=="__main__": unittest.main()
