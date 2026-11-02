import unittest,json,copy
from pathlib import Path
from V6.experimental.v7_shadow_models import load_bundle,scores,H
O=Path(__file__).resolve().parent
class Adoption(unittest.TestCase):
 def test_restored_models_replay_original_scores(self):
  adoption=json.loads((O/'window37-model-adoption.json').read_text());p,models,packet=load_bundle(adoption,H(adoption));table=json.loads(Path(p['features']['path']).read_text());fs=sorted([f for f in table if f['signal_session']==p['offline_signal']],key=lambda f:f['stock_id']);self.assertEqual(H(scores(p,models,fs)),H(packet['scores']));self.assertEqual(len(packet['scores']),512)
 def test_adoption_changed_identity_rejected(self):
  a=json.loads((O/'window37-model-adoption.json').read_text());b=copy.deepcopy(a);b['run']['sha256']='0'*64
  with self.assertRaises(ValueError):load_bundle(b,H(b))
if __name__=='__main__':unittest.main()
