import unittest

from V6.experimental.v7_evidence_tier import classify_evidence_tier, classify_from_publication

FULL = dict(
    health_state="healthy", publish_allowed=True, evidence_class="real_pit_oos",
    out_of_sample_validated=True, baseline_compared=True, pit_audited=True,
)


class EvidenceTierTest(unittest.TestCase):
    def test_all_conditions_verified(self):
        self.assertEqual(classify_evidence_tier(**FULL)["tier"], "VERIFIED")

    def test_default_is_research_only(self):
        result = classify_evidence_tier()
        self.assertEqual(result["tier"], "RESEARCH_ONLY")
        self.assertTrue(result["reasons"])
        self.assertEqual(len(result["required_for_verified"]), 5)

    def test_each_missing_condition_downgrades(self):
        for key, bad in (
            ("health_state", "degraded"), ("publish_allowed", None),
            ("evidence_class", "synthetic"), ("evidence_class", "Fixture"),
            ("evidence_class", None), ("out_of_sample_validated", "yes"),
            ("baseline_compared", None), ("pit_audited", 1),
        ):
            with self.subTest(key=key, bad=bad):
                self.assertEqual(
                    classify_evidence_tier(**{**FULL, key: bad})["tier"], "RESEARCH_ONLY")

    def test_blocked(self):
        self.assertEqual(classify_evidence_tier(**{**FULL, "health_state": "blocked"})["tier"], "BLOCKED")
        self.assertEqual(classify_evidence_tier(**{**FULL, "publish_allowed": False})["tier"], "BLOCKED")

    def test_publication_without_evidence_is_research_only(self):
        doc = {"state": "healthy", "publish_allowed": True}
        self.assertEqual(classify_from_publication(doc)["tier"], "RESEARCH_ONLY")
        doc["evidence"] = {"evidence_class": "real_pit_oos", "out_of_sample_validated": True,
                           "baseline_compared": True, "pit_audited": True}
        self.assertEqual(classify_from_publication(doc)["tier"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()


class AllowlistTests(unittest.TestCase):
    def test_unknown_or_synthetic_classes_never_verify(self):
        from V6.experimental.v7_evidence_tier import classify_evidence_tier
        for cls in ("demo", "mock", "SYNTHETIC_OR_FIXTURE_ONLY_NOT_INVESTMENT_EVIDENCE"):
            r = classify_evidence_tier(health_state="healthy", publish_allowed=True,
                evidence_class=cls, out_of_sample_validated=True,
                baseline_compared=True, pit_audited=True)
            self.assertEqual(r["tier"], "RESEARCH_ONLY", cls)
        r = classify_evidence_tier(health_state="healthy", publish_allowed=True,
            evidence_class="real_pit_oos", out_of_sample_validated=True,
            baseline_compared=True, pit_audited=True)
        self.assertEqual(r["tier"], "VERIFIED")
