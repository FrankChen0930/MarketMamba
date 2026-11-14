"""Small tests for serialization and paired support; synthetic only."""
import sys,unittest,json,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_robustness_audit import leave_one
class ReplayAndSupport(unittest.TestCase):
 def test_json_object_key_roundtrip_identity(self):
  value={'daily_N':{20:1,21:2}};restored=json.loads(json.dumps(value))
  self.assertNotEqual(value,restored);self.assertEqual(canonical_hash(value),canonical_hash(restored))
 def test_below20_remains_undefined_and_paired(self):
  rows=[dict(stock_id='A',signal_session='2023-01-02',horizon=5,score=1.,label=.1,label_id='x',label_metadata_sha256='y')]
  q=leave_one(rows,rows,None,'2023-01-02',5,{'min_valid_n':20})
  self.assertIsNone(q['delta']);self.assertEqual(q['L0']['reasons'],['INSUFFICIENT_SUPPORT']);self.assertEqual(q['L0']['stock_ids'],['A'])
 def test_changed_label_or_duplicate_refuses(self):
  r=dict(stock_id='A',signal_session='2023-01-02',horizon=5,score=1.,label=.1,label_id='x',label_metadata_sha256='y')
  for a,b in [([r],[{**r,'label':.2}]),([r,r],[r,r])]:
   with self.assertRaises(ValueError):leave_one(a,b,None,'2023-01-02',5,{'min_valid_n':20})
 def test_replay_repair_did_not_change_calculation(self):
  repair=json.loads(Path(__file__).with_name('window27-replay-repair.json').read_text())
  old=Path(repair['original_calculation_source']['path']).read_text();new=Path(__file__).with_name('window27-diagnostic.py').read_text()
  body=lambda s:next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='calculate')
  self.assertEqual(ast.dump(body(old)),ast.dump(body(new)))
if __name__=='__main__':unittest.main()
