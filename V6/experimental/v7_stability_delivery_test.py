import json
import unittest
import zipfile
from pathlib import Path


class StabilityDeliveryTests(unittest.TestCase):
    def test_colab_delivery_tracks_runtime_manifest_used_by_notebook(self):
        root = Path(__file__).resolve().parents[2]
        delivery = root / "deliveries" / "V7-Stability-20260915"
        archive = delivery / "MarketMamba-V7-stability.zip"
        required = "V6/experimental/v7_integrated_environment.json"
        notebook = json.loads(
            (delivery / "V7-穩定性與三窗續跑.ipynb").read_text(encoding="utf-8")
        )
        notebook_source = "".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn(required, notebook_source)
        with zipfile.ZipFile(archive) as package:
            self.assertIn(required, package.namelist())
            source_manifest = json.loads(package.read("experiment-source.json"))
        self.assertIn(required, source_manifest)


if __name__ == "__main__":
    unittest.main()
