"""Adversarial metadata-only readiness tests; no data, weights or evaluation."""
import copy
from dataclasses import replace
from pathlib import Path
import unittest

from V6.experimental.v7_e1_readiness import audit, read_result_blob, candidate_config

ROOT = Path(__file__).resolve().parents[2]


class ReadinessTest(unittest.TestCase):
    def test_admitted_artifacts_integrity_is_not_execution_readiness(self):
        report = audit(ROOT)
        self.assertEqual(report['copied_artifacts_verified'], 7)
        self.assertEqual(report['formal_seeds'], [17, 29])
        self.assertEqual(report['features'], 48)
        self.assertFalse(report['execution_ready'])
        self.assertEqual(report['strict_phase0'], 'STOP')
        self.assertEqual(report['matrix_payload'], 'NOT_VERIFIED')
        self.assertEqual(report['checkpoint_payload'], 'NOT_LOADED')

    def test_tampered_copied_artifacts_are_rejected(self):
        for name in ('evaluation.json','report.md','run-summary.json',
                     'seed-17-result.json','seed-29-result.json',
                     'seed-17-checkpoint.json','seed-29-checkpoint.json'):
            def read(n):
                b=read_result_blob(ROOT,n)
                return b+b' ' if n==name else b
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit(ROOT,read_blob=read)

    def test_self_relabelled_manifest_cannot_become_authority(self):
        def read(n):
            b=read_result_blob(ROOT,n)
            return b.replace(b'PASS_REFRESH_HYPOTHESIS',b'NEW_APPROVED_RESULT') if n=='result-manifest.json' else b
        with self.assertRaises(ValueError): audit(ROOT,read_blob=read)

    def test_missing_artifact_fails_closed(self):
        def read(n):
            if n=='seed-29-checkpoint.json': raise FileNotFoundError(n)
            return read_result_blob(ROOT,n)
        with self.assertRaises(ValueError): audit(ROOT,read_blob=read)

    def test_candidate_split_or_feature_order_drift_rejected(self):
        cfg=candidate_config(ROOT)
        for bad in (replace(cfg,train_end='2025-11-19'),
                    replace(cfg,feature_order=tuple(reversed(cfg.feature_order)))):
            with self.subTest(config=bad.sha256), self.assertRaises(ValueError):
                audit(ROOT,config=bad)


if __name__=='__main__': unittest.main()
