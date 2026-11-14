"""CPU-only contract tests. SSD execution is always explicitly injected."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest import mock

import torch
from torch import nn
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from v7_integrated_config import get_test_config, validate_v6_feature_literal
from v7_integrated_data_quality import protocol_metadata, QualityPolicy, assess_prices, calendar_labels, validate_protocol, assess_training


def synthetic_metadata(dates, minimum=2, ids=('A','B')):
    days = sorted(set(str(pd.Timestamp(d).date()) for d in dates))
    return protocol_metadata({'trading_calendar':days,
        'provenance':'explicit synthetic expected calendar',
        'expected_universe':{day:list(ids) for day in days},
        'universe_provenance':'explicit synthetic fixed basket'}, QualityPolicy(minimum_usable_days=minimum))


def synthetic_ssd(x, dt, A, B, C, *, D, dt_bias, dt_softplus, chunk_size):
    """Differentiable CPU test double, deliberately not a runtime fallback."""
    assert dt_softplus
    rate = torch.nn.functional.softplus(dt + dt_bias).unsqueeze(-1)
    drive = (B.mean(-1) + C.mean(-1)).mean(-1, keepdim=True).unsqueeze(-1)
    state = torch.zeros_like(x[:, 0])
    outputs = []
    decay = torch.sigmoid(A).view(1, A.numel(), 1)
    for t in range(x.shape[1]):
        state = decay * state + rate[:, t] * (x[:, t] + drive[:, t])
        outputs.append(state + D.view(1, -1, 1) * x[:, t])
    return torch.stack(outputs, dim=1)


class RecordingGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, h, edge_index, edge_attr):
        self.last = (h.detach().clone(), edge_index.detach().clone(), edge_attr.detach().clone())
        if edge_index.numel():
            out = torch.zeros_like(h)
            out.index_add_(0, edge_index[1], h[edge_index[0]] * edge_attr[:, None])
            return h + out
        return h


class TestIntegratedCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch("torch.cuda.is_available", side_effect=AssertionError("CUDA probe forbidden")):
            cls.mod = importlib.import_module("v7_integrated_model")

    def make_model(self, seed=17):
        torch.manual_seed(seed)
        cfg = get_test_config()
        graph = RecordingGraph()
        model = self.mod.MarketMambaV7Integrated(cfg, ssd_fn=synthetic_ssd, graph_layer=graph)
        return model, graph

    def test_contract_backward_initialization_and_no_conv(self):
        model, _ = self.make_model()
        x = torch.randn(5, 7, 59, requires_grad=True)
        mask = torch.ones(5, 7, dtype=torch.bool)
        mask[3:] = False
        edges = torch.tensor([[0, 1, 1, 3], [1, 0, 3, 0]])
        attrs = torch.ones(4)
        y = model(x, edges, attrs, mask)
        self.assertEqual(tuple(y.shape), (5, 2))
        self.assertTrue(torch.equal(y[3:], torch.zeros_like(y[3:])))
        y[:3].sum().backward()
        self.assertIsNotNone(model.head_5d.weight.grad)
        self.assertIsNotNone(x.grad)
        blocks = [m for m in model.modules() if isinstance(m, self.mod.Mamba2NoConv)]
        self.assertEqual(len(blocks), 3)
        self.assertFalse(any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules()))
        for block in blocks:
            dt = torch.nn.functional.softplus(block.dt_bias.detach())
            self.assertTrue(bool(((dt >= model.config.dt_min) & (dt <= model.config.dt_max)).all()))
            A = -torch.exp(block.A_log.detach())
            self.assertTrue(bool(((A <= -model.config.a_min) & (A >= -model.config.a_max)).all()))
            self.assertTrue(getattr(block.A_log, "_no_weight_decay", False))
            self.assertTrue(getattr(block.D, "_no_weight_decay", False))
            self.assertTrue(getattr(block.dt_bias, "_no_weight_decay", False))

    def test_gate_is_applied_before_rms(self):
        norm = self.mod.GatedRMSNorm(3)
        x, z = torch.tensor([[1., 2., 4.]]), torch.tensor([[2., -1., .5]])
        gated = x * torch.nn.functional.silu(z)
        expected = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + norm.eps)
        self.assertTrue(torch.allclose(norm(x, z), expected))

    def test_grouped_rms_configuration_is_rejected(self):
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError, "n_groups"):
            replace(get_test_config(), n_groups=2).validate()

    def test_requires_explicit_ssd_and_never_auto_falls_back(self):
        cfg = get_test_config()
        model = self.mod.MarketMambaV7Integrated(cfg, graph_layer=RecordingGraph())
        with self.assertRaisesRegex(RuntimeError, "explicit SSD"):
            model(torch.randn(2, 3, 59), torch.empty(2, 0, dtype=torch.long), torch.empty(0), torch.ones(2, 3, dtype=torch.bool))

    def test_eligible_only_edge_remap_and_scatter_alignment(self):
        model, graph = self.make_model()
        model.eval()
        x = torch.randn(5, 6, 59)
        mask = torch.ones(5, 6, dtype=torch.bool)
        mask[[1, 3]] = False
        edges = torch.tensor([[0, 2, 4, 1, 4], [2, 4, 0, 2, 3]])
        attrs = torch.arange(1, 6, dtype=torch.float32)
        out = model(x, edges, attrs, mask)
        self.assertEqual(graph.last[0].shape[0], 3)
        self.assertTrue(torch.equal(graph.last[1], torch.tensor([[0, 1, 2], [1, 2, 0]])))
        self.assertTrue(torch.equal(graph.last[2], torch.tensor([1., 2., 3.])))
        self.assertTrue(torch.equal(out[[1, 3]], torch.zeros(2, 2)))
        self.assertGreater(float(out[[0, 2, 4]].detach().abs().sum()), 0.0)

    def test_only_real_timesteps_and_eligible_stocks_reach_scans(self):
        lengths = []
        def recording_ssd(x, dt, A, B, C, **kwargs):
            lengths.append(x.shape[1])
            return x
        torch.manual_seed(7)
        cfg = get_test_config()
        model = self.mod.MarketMambaV7Integrated(cfg, ssd_fn=recording_ssd,
                                                 graph_layer=RecordingGraph())
        mask = torch.tensor([[True, True, False, True], [False, False, False, False],
                             [True, False, True, False]])
        model(torch.randn(3, 4, 59), torch.empty(2, 0, dtype=torch.long),
              torch.empty(0), mask)
        self.assertEqual(lengths, [4, 1, 1])

    def test_equal_length_temporal_sequences_are_batched(self):
        calls = []
        def recording_ssd(x, dt, A, B, C, **kwargs):
            calls.append(tuple(x.shape[:2]))
            return x
        model = self.mod.MarketMambaV7Integrated(
            get_test_config(), ssd_fn=recording_ssd, graph_layer=RecordingGraph())
        model(torch.randn(8, 5, 59), torch.empty(2, 0, dtype=torch.long),
              torch.empty(0), torch.ones(8, 5, dtype=torch.bool))
        self.assertEqual(calls, [(8, 5), (1, 8), (1, 8)])

    def test_canonical_order_bidirectional_restore_and_reproducibility(self):
        ids = ["2330", "1101", "0050", "2317"]
        order, inverse = self.mod.canonical_stock_order(ids)
        self.assertEqual([ids[i] for i in order.tolist()], sorted(ids))
        values = torch.arange(4)
        self.assertTrue(torch.equal(values[order][inverse], values))
        trace = []
        def scan(seq):
            trace.append(seq.detach().clone())
            return seq + torch.arange(seq.shape[1])[None, :, None]
        inp = torch.arange(12).view(1, 4, 3).float()
        got = self.mod.bidirectional_scan(inp, scan)
        positions = torch.arange(4)[None, :, None]
        expected = .5 * (trace[0] + positions + torch.flip(trace[1] + positions, [1]))
        self.assertTrue(torch.equal(got, expected))
        m1, _ = self.make_model(91); m2, _ = self.make_model(91)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values())))
        sample = torch.randn(2, 3, 59)
        empty_edges = torch.empty(2, 0, dtype=torch.long)
        self.assertTrue(torch.equal(
            m1(sample, empty_edges, torch.empty(0), torch.ones(2, 3, dtype=torch.bool)),
            m2(sample, empty_edges, torch.empty(0), torch.ones(2, 3, dtype=torch.bool)),
        ))

    def test_bounded_fusion_weights(self):
        model, _ = self.make_model()
        branches = [torch.randn(3, 32) for _ in range(3)]
        _, weights = model.fusion(*branches)
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3)))
        self.assertGreaterEqual(float(weights.detach().min()), 1.0 - model.config.fusion_limit)
        self.assertLessEqual(float(weights.detach().max()), model.config.fusion_limit)
        _, weights = model.fusion(*branches)
        self.assertTrue(bool((weights >= 1 - model.config.fusion_limit).all()))
        self.assertTrue(bool((weights <= model.config.fusion_limit).all()))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3)))

    def test_temporal_padding_is_excluded_from_ssm_interactions(self):
        model, _ = self.make_model()
        model.eval()
        x = torch.randn(2, 5, 59)
        mask = torch.tensor([[True, False, True, False, True], [True, True, False, False, False]])
        altered = x.clone()
        altered[~mask] = 10_000 * torch.randn_like(altered[~mask])
        edges = torch.empty(2, 0, dtype=torch.long)
        attrs = torch.empty(0)
        self.assertTrue(torch.allclose(model(x, edges, attrs, mask), model(altered, edges, attrs, mask)))

    def test_no_conv_path_applies_activation_before_ssd(self):
        seen = []
        def recorder(x, dt, A, B, C, **kwargs):
            seen.append((x.detach(), B.detach(), C.detach()))
            return x
        cfg = get_test_config()
        block = self.mod.Mamba2NoConv(cfg, recorder)
        block(torch.randn(1, 3, cfg.d_model))
        for tensor in seen[0]:
            self.assertGreaterEqual(float(tensor.min()), -0.279)


class TestPreparationPackage(unittest.TestCase):
    def test_train_commands_forward_explicit_runtime_metadata_to_preflight(self):
        train = importlib.import_module("v7_integrated_train")
        probe = importlib.import_module("v7_integrated_probe")
        parser = train.build_parser()
        for command in ("preflight", "smoke", "train", "forecast"):
            args = parser.parse_args([command, "--runtime-metadata", "runtime.json"])
            self.assertEqual(args.runtime_metadata, Path("runtime.json"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); wheel = root / "mamba.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mamba_ssm-2.3.2.dist-info/METADATA",
                    "Metadata-Version: 2.1\nName: mamba-ssm\nVersion: 2.3.2.post1\n"
                    "Requires-Dist: torch\nRequires-Dist: einops\nRequires-Dist: transformers\n")
            import hashlib
            runtime = probe.RuntimeMetadata("3.11.9", "cp311", "2.6.0", "12.4", False, "linux-x86_64")
            manifest = {"mamba_ssm_pin": "2.3.2.post1", "wheel_candidates": [{
                "filename": wheel.name, "python_tag": "cp311", "torch": "2.6.0",
                "cuda": "12.4", "cxx11_abi": False, "platform": "linux_x86_64",
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}]}
            path = root / "manifest.json"; path.write_text(json.dumps(manifest))
            runtime_path = root / "runtime.json"; runtime_path.write_text(json.dumps(runtime.__dict__))
            args = parser.parse_args(["preflight", "--manifest", str(path),
                                      "--runtime-metadata", str(runtime_path)])
            with mock.patch.object(probe, "collect_runtime_metadata", side_effect=AssertionError("implicit metadata forbidden")):
                result = train.run(args)
            self.assertEqual(result["runtime"], runtime.__dict__)
            self.assertIn("transformers", result["wheel_requires_dist"])

    def test_import_safety_with_cuda_probe_forbidden(self):
        with mock.patch("torch.cuda.is_available", side_effect=AssertionError("CUDA probe forbidden")), \
             mock.patch("subprocess.run", side_effect=AssertionError("subprocess forbidden")):
            for name in ("v7_integrated_probe", "v7_integrated_train"):
                sys.modules.pop(name, None)
                importlib.import_module(name)

    def test_canonical_compaction_edge_remap_and_scatter(self):
        train = importlib.import_module("v7_integrated_train")
        x = torch.arange(4 * 2 * 3).view(4, 2, 3).float()
        mask = torch.tensor([[True, True], [False, False], [True, True], [True, True]])
        edges = torch.tensor([[0, 2, 3, 1], [2, 3, 0, 2]])
        attrs = torch.tensor([1., 2., 3., 4.])
        prepared = train.prepare_stock_batch(x, edges, attrs, mask, ["B", "X", "A", "C"])
        self.assertEqual(prepared.stock_ids, ("A", "B", "C"))
        self.assertTrue(torch.equal(prepared.edge_index, torch.tensor([[1, 0, 2], [0, 2, 1]])))
        canonical = torch.tensor([[10., 11.], [20., 21.], [30., 31.]])
        restored = prepared.scatter_to_original(canonical)
        self.assertTrue(torch.equal(restored, torch.tensor([[20., 21.], [0., 0.], [10., 11.], [30., 31.]])))

    def test_preflight_fails_before_compiler_or_install_subprocess(self):
        probe = importlib.import_module("v7_integrated_probe")
        manifest = {
            "schema_version": 1,
            "mamba_ssm_pin": "2.3.2.post1",
            "wheel_candidates": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch("subprocess.run", side_effect=AssertionError("must fail before subprocess")):
                with self.assertRaisesRegex(probe.PreflightError, "wheel metadata"):
                    probe.preflight(path)

    def test_official_release_asset_selection_normalizes_cuda_major_and_architecture(self):
        probe = importlib.import_module("v7_integrated_probe")
        assets = [{"name": name, "browser_download_url": "https://example.invalid/" + name}
                  for name in (
            "mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
            "mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_aarch64.whl",
            "mamba_ssm-2.3.2+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        )]
        for platform_name, expected in (
            ("linux-x86_64", "linux_x86_64"),
            ("linux-aarch64", "linux_aarch64"),
        ):
            runtime = probe.RuntimeMetadata("3.12.3", "cp312", "2.10.0", "12.8", True,
                                            platform_name)
            selected = probe.select_official_release_asset(runtime, assets)
            self.assertIn("2.3.2.post1+cu12", selected["name"])
            self.assertIn(expected, selected["name"])
        unsupported = probe.RuntimeMetadata("3.12.3", "cp312", "2.10.0", "12.8", True,
                                            "linux-riscv64")
        with self.assertRaisesRegex(probe.PreflightError, "platform"):
            probe.select_official_release_asset(unsupported, assets)

    def test_mocked_official_release_download_handles_both_linux_architectures(self):
        probe = importlib.import_module("v7_integrated_probe")
        wheel_metadata = ("Metadata-Version: 2.1\nName: mamba-ssm\nVersion: 2.3.2.post1\n"
                          "Requires-Dist: torch>=2.4\nRequires-Dist: einops>=0.7\n"
                          "Requires-Dist: transformers>=4.40; python_version >= '3.8'\n")
        for platform_name, wheel_platform in (("linux-x86_64", "linux_x86_64"),
                                               ("linux-aarch64", "linux_aarch64")):
            runtime = probe.RuntimeMetadata("3.12.3", "cp312", "2.10.0", "12.8", True,
                                            platform_name)
            filename = ("mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-"
                        f"cp312-cp312-{wheel_platform}.whl")
            release = {"assets": [{"name": filename,
                                    "browser_download_url": "https://example.invalid/" + filename}]}
            with self.subTest(platform=platform_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); manifest = root / "manifest.json"; runtime_path = root / "runtime.json"
                manifest.write_text(json.dumps({"schema_version": 1,
                    "mamba_ssm_pin": "2.3.2.post1", "wheel_candidates": []}))
                def fake_download(_url, target):
                    with zipfile.ZipFile(target, "w") as archive:
                        archive.writestr("mamba_ssm-2.3.2.post1.dist-info/METADATA", wheel_metadata)
                    return str(target), None
                response = io.BytesIO(json.dumps(release).encode())
                with mock.patch.object(probe, "collect_colab_runtime_metadata", return_value=runtime), \
                     mock.patch("urllib.request.urlopen", return_value=response), \
                     mock.patch("urllib.request.urlretrieve", side_effect=fake_download):
                    result = probe.setup_colab(manifest, runtime_path, root)
                self.assertEqual(result["wheel"]["platform"], wheel_platform)
                self.assertEqual(result["wheel_requires_dist"], ["einops", "torch", "transformers"])

    def test_wheel_metadata_parses_versioned_requirements_and_applicable_markers(self):
        probe = importlib.import_module("v7_integrated_probe")
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "mamba.whl"
            metadata = ("Metadata-Version: 2.1\nName: mamba-ssm\nVersion: 2.3.2.post1\n"
                        "Requires-Dist: torch>=2.4\nRequires-Dist: einops>=0.7\n"
                        "Requires-Dist: transformers; python_version >= '3.8'\n"
                        "Requires-Dist: fake-dep; python_version < '2'\n")
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mamba_ssm-2.3.2.post1.dist-info/METADATA", metadata)
            self.assertEqual(probe._wheel_requirements(wheel), {"torch", "einops", "transformers"})

    def test_prepare_source_bounds_keep_price_window_but_retain_context_history(self):
        train = importlib.import_module("v7_integrated_train")
        bounds = train.source_read_bounds("2022-01-01", "2024-04-30")
        self.assertEqual(bounds["prices"], ("2022-01-01", "2024-04-30"))
        for source in ("per", "market_value", "revenue", "financials", "balance_sheet", "cashflow", "dividend",
                       "macro", "fear_greed", "business_indicator", "fed_rate"):
            self.assertEqual(bounds[source], (None, "2024-04-30"))

    def test_prior_history_predicate_regression_keeps_context_not_sample_prices(self):
        train = importlib.import_module("v7_integrated_train")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = pd.DataFrame({"Date": pd.to_datetime(["2021-12-31", "2022-01-03"]),
                                 "stock_id": ["2330", "2330"], "value": [1.0, 2.0]})
            price_path, financial_path = root / "prices.parquet", root / "financials.parquet"
            rows.to_parquet(price_path); rows.to_parquet(financial_path)
            bounds = train.source_read_bounds("2022-01-01", "2024-04-30")
            prices = train._read_bounded_parquet(price_path, ["2330"], *bounds["prices"])
            financials = train._read_bounded_parquet(financial_path, ["2330"], *bounds["financials"])
            self.assertEqual(prices["value"].tolist(), [2.0])
            self.assertEqual(financials["value"].tolist(), [1.0, 2.0])

    def test_restricted_prepare_is_labeled_and_stale_twii_fails_active_rs(self):
        train = importlib.import_module("v7_integrated_train")
        self.assertEqual(train.preparation_artifact_kind(False, ["2330"], None, None),
                         "diagnostic-restricted-not-performance-evidence")
        self.assertEqual(train.preparation_artifact_kind(False, None, None, None),
                         "full-prepared-candidate")
        prices = pd.DataFrame({"Date": pd.to_datetime(["2024-04-29", "2024-04-30"])})
        macro = pd.DataFrame({"Date": pd.to_datetime(["2024-04-29", "2024-04-30"]),
                              "TWII_Close": [20000.0, np.nan], "VIX": [15.0, 15.0]})
        with self.assertRaisesRegex(ValueError, "active RS.*TWII"):
            train.require_twii_coverage_for_active_rs(prices, macro)

    def test_checkpoint_resume_provenance_in_memory(self):
        train = importlib.import_module("v7_integrated_train")
        state = {"weight": torch.tensor([1.0])}
        checkpoint = train.build_checkpoint(state, {"step": 4}, epoch=2, step=9,
                                            provenance={"protocol":"synthetic", "config_sha256": "abc", "data_sha256": "def"})
        stream = io.BytesIO()
        train.save_checkpoint(checkpoint, stream)
        stream.seek(0)
        loaded = train.load_checkpoint(stream, expected_provenance={"protocol":"synthetic", "config_sha256": "abc", "data_sha256": "def"})
        self.assertEqual((loaded["epoch"], loaded["step"]), (2, 9))
        with self.assertRaisesRegex(ValueError, "provenance"):
            train.load_checkpoint(io.BytesIO(stream.getvalue()), expected_provenance={"protocol":"synthetic", "config_sha256": "wrong"})

    def test_group_d_is_zeroed_without_changing_width(self):
        train = importlib.import_module("v7_integrated_train")
        x = torch.ones(2, 3, 59)
        got = train.zero_macro_group(x, get_test_config())
        self.assertEqual(got.shape, x.shape)
        self.assertTrue(torch.equal(got[..., :47], x[..., :47]))
        self.assertTrue(torch.equal(got[..., 47:], torch.zeros_like(got[..., 47:])))

    def test_rank_centered_weighted_short_loss(self):
        train = importlib.import_module("v7_integrated_train")
        labels = torch.tensor([[10., 4.], [30., 1.], [20., 7.]])
        targets = train.rank_center_targets(labels)
        self.assertTrue(torch.allclose(targets.mean(0), torch.zeros(2)))
        perfect = train.short_loss(targets, labels)
        reversed_loss = train.short_loss(torch.flip(targets, [0]), labels)
        self.assertLess(float(perfect), float(reversed_loss))

    def test_rank_ic_ignores_nan_and_constant_forecast(self):
        train = importlib.import_module("v7_integrated_train")
        pred = torch.tensor([[1., 2.], [float("nan"), 2.], [3., 2.]])
        labels = torch.tensor([[1., 3.], [2., 2.], [3., 1.]])
        with self.assertRaisesRegex(ValueError,'nonfinite predictions'):
            train.rank_ic_by_horizon(pred, labels)
        pred[1,0]=2.
        labels[1,0]=float('nan')
        metrics = train.rank_ic_by_horizon(pred, labels)
        self.assertAlmostEqual(metrics["rank_ic_5d"], 1.0, places=6)
        self.assertTrue(torch.isnan(torch.tensor(metrics["rank_ic_10d"])))

    def test_f20_is_configuration_not_forecast_tensor(self):
        train = importlib.import_module("v7_integrated_train")
        self.assertEqual(train.economic_f20_config(), {
            "N": 50, "buffer": 1.5, "frequency": 20, "weighting": "equal"})

    def test_no_weight_decay_parameter_groups(self):
        train = importlib.import_module("v7_integrated_train")
        model = importlib.import_module("v7_integrated_model").MarketMambaV7Integrated(
            get_test_config(), ssd_fn=synthetic_ssd, graph_layer=RecordingGraph())
        groups = train.adamw_parameter_groups(model, weight_decay=0.1)
        no_decay_ids = {id(p) for group in groups if group["weight_decay"] == 0 for p in group["params"]}
        marked_ids = {id(p) for p in model.parameters() if getattr(p, "_no_weight_decay", False)}
        self.assertEqual(no_decay_ids, marked_ids)

    def test_frozen_date_boundaries_reject_leakage_and_duplicates(self):
        train = importlib.import_module("v7_integrated_train")
        valid = {
            "train": ["2013-01-01", "2023-11-17"],
            "validation": ["2024-01-02", "2024-12-31"],
            "test": ["2025-02-12", "2025-03-03"],
        }
        train.validate_frozen_dates(valid)
        for invalid in (
            {**valid, "test": ["2024-12-31", "2025-03-03"]},
            {**valid, "validation": ["2024-01-02", "2024-01-02"]},
            {**valid, "train": ["2012-12-31", "2023-11-17"]},
        ):
            with self.assertRaises(ValueError):
                train.validate_frozen_dates(invalid)

    def test_resume_state_has_rng_scheduler_and_step_boundary(self):
        train = importlib.import_module("v7_integrated_train")
        state = train.capture_rng_state()
        checkpoint = train.build_checkpoint({}, {}, epoch=1, step=10, batch_index=3,
            provenance={"config": "x", "source": "y", "runtime": "z", "features": "f",
                        "kg": "k", "data": "d", "split": "s"}, scheduler_state={"last_epoch": 1}, rng_state=state)
        self.assertIn("rng_state", checkpoint)
        self.assertIn("scheduler_state", checkpoint)
        self.assertFalse(train.should_update(checkpoint["step"], 10))
        selected = train.build_checkpoint({}, {}, epoch=2, step=11, provenance={"config": "x"},
                                          best_validation_metric=.17,
                                          validation_metrics={"rank_ic_5d": .17})
        self.assertEqual(selected["best_validation_metric"], .17)
        self.assertEqual(selected["validation_metrics"]["rank_ic_5d"], .17)
        self.assertEqual(train.normalize_resume_position(3, 80, 80), (4, 0))
        self.assertEqual(train.normalize_resume_position(3, 81, 80), (4, 1))

    def test_daily_dataset_indexes_one_read_and_returns_chronological_last60(self):
        train = importlib.import_module("v7_integrated_train")
        features = list(train.V6_FEATURE_COLUMNS)
        dates = pd.date_range("2023-10-01", periods=70, freq="D")
        frame = pd.DataFrame({"Date": list(dates) * 2, "stock_id": ["B"] * 70 + ["A"] * 70,
            **{name: list(range(70)) * 2 for name in features},
            "Alpha_5d": [1.] * 140, "Alpha_10d": [2.] * 140})
        frame["observation_valid"] = True
        metadata = {**synthetic_metadata(dates), "feature_columns": features, "feature_fingerprint": "fp"}
        reads = []
        def reader(path, **kwargs):
            reads.append(kwargs)
            return frame.copy()
        requested = [dates[-2].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")]
        dataset = train.DailyCrossSectionDataset("features.parquet", requested, metadata,
                                                  parquet_reader=reader)
        self.assertEqual(reads, [])
        first, second = dataset[0], dataset[1]
        self.assertEqual(len(reads), 1)
        self.assertNotIn("filters", reads[0])
        self.assertEqual(first["stock_ids"], ("A", "B"))
        self.assertTrue(torch.equal(second["x"][0, :, 0], torch.arange(10, 70).float()))
        self.assertTrue(torch.equal(second["x"][..., -12:], torch.zeros_like(second["x"][..., -12:])))

    def test_daily_dataset_builds_shared_array_index_without_full_frame_scans(self):
        train = importlib.import_module("v7_integrated_train")
        features = list(train.V6_FEATURE_COLUMNS)
        dates = pd.date_range("2024-01-02", periods=120, freq="B")
        rows = [(day, f"S{stock:03d}") for stock in range(40) for day in dates]
        frame = pd.DataFrame(rows, columns=["Date", "stock_id"])
        for offset, name in enumerate(features):
            frame[name] = np.arange(len(frame), dtype=np.float32) + offset
        frame["Alpha_5d"], frame["Alpha_10d"] = 1.0, np.nan
        reads = []
        def reader(path, **kwargs):
            reads.append(path); return frame.copy()
        frame["observation_valid"] = True
        metadata = {**synthetic_metadata(dates), "feature_columns": features, "feature_fingerprint": "fp"}
        index = train.PreparedDataIndex.from_parquet("features.parquet", metadata,
                                                     parquet_reader=reader)
        left = train.DailyCrossSectionDataset("features.parquet", [str(dates[-2].date())], metadata,
                                               prepared_index=index)
        right = train.DailyCrossSectionDataset("features.parquet", [str(dates[-1].date())], metadata,
                                                prepared_index=index)
        before = index.search_operations
        sample = right[0]
        self.assertEqual(len(reads), 1)
        self.assertIs(left.prepared_index, right.prepared_index)
        self.assertEqual(index.search_operations - before, 40)
        self.assertFalse(any(isinstance(value, pd.DataFrame) for value in vars(index).values()))
        self.assertEqual(sample["x"].shape, (40, 60, 59))
        self.assertTrue(torch.isnan(sample["labels"][:, 1]).all())

    def test_feature_metadata_requires_frozen_v6_order(self):
        train = importlib.import_module("v7_integrated_train")
        actual = list(train.V6_FEATURE_COLUMNS)
        train.validate_feature_metadata({"feature_columns": actual, "feature_fingerprint": "x"}, get_test_config())
        with self.assertRaisesRegex(ValueError, "frozen V6"):
            train.validate_feature_metadata({"feature_columns": list(reversed(actual)), "feature_fingerprint": "x"}, get_test_config())

    def test_frozen_feature_literal_matches_v6_config_without_import(self):
        source = HERE.parent / "marketmamba" / "config.py"
        if not source.exists(): source=Path("/home/frank/projects/MarketMamba/.artifacts/v7-data-policy/workspace/V6/marketmamba/config.py")
        with mock.patch("pathlib.Path.mkdir", side_effect=AssertionError("config import side effect forbidden")):
            validate_v6_feature_literal(source)

    def test_real_portfolio_helper_adapter_writes_daily_and_charges_rebalances(self):
        train = importlib.import_module("v7_integrated_train")
        helper_path = Path(__file__).resolve().with_name("portfolio_lab.py")
        if not helper_path.exists(): helper_path=Path("/home/frank/projects/MarketMamba/.artifacts/v7-data-policy/workspace/V6/experimental/portfolio_lab.py")
        spec = importlib.util.spec_from_file_location("portfolio_lab", helper_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        stocks = [f"S{i:03d}" for i in range(100)]
        dates = pd.bdate_range("2025-01-02", periods=41)
        scores, market = [], []
        for di, day in enumerate(dates):
            for si, stock in enumerate(stocks):
                score = float(si if di < 20 else 99 - si)
                scores.append((day, stock, score, score))
                market.append((day, stock, 100.0 + di * (1 + si / 1000), 1000.0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sp = root / "scores.parquet"; mp = root / "market.parquet"
            pd.DataFrame(scores, columns=["Date", "stock_id", "Score_5d", "Score_10d"]).to_parquet(sp)
            pd.DataFrame(market, columns=["Date", "stock_id", "Close", "Volume"]).to_parquet(mp)
            with mock.patch.dict(sys.modules, {"portfolio_lab": module}):
                result = train.run_economic_replay(sp, mp, root / "experiment")
            daily = pd.read_parquet(root / "experiment" / "daily_returns.parquet")
            self.assertAlmostEqual(float(daily.iloc[0].transaction_cost), .0015)
            self.assertGreater(float(daily.iloc[20].transaction_cost), 0.0)
            self.assertAlmostEqual(float(daily.iloc[0].daily_return), -.0015)
            self.assertTrue((root / "experiment" / "summary.json").is_file())
            self.assertNotIn("_daily", result)
            with mock.patch.dict(sys.modules, {"portfolio_lab": module}):
                same = train.run_economic_replay(sp, mp, root / "experiment_same")
            same_daily = pd.read_parquet(root / "experiment_same" / "daily_returns.parquet")
            self.assertTrue(daily.daily_return.equals(same_daily.daily_return))
            self.assertEqual(result, same)

    def test_real_portfolio_adapter_accepts_ragged_scores_and_preserves_market_calendar(self):
        train = importlib.import_module("v7_integrated_train")
        helper_path = Path(__file__).resolve().with_name("portfolio_lab.py")
        if not helper_path.exists(): helper_path=Path("/home/frank/projects/MarketMamba/.artifacts/v7-data-policy/workspace/V6/experimental/portfolio_lab.py")
        spec = importlib.util.spec_from_file_location("portfolio_lab", helper_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        stocks, dates = [f"S{i:03d}" for i in range(100)], pd.bdate_range("2025-01-02", periods=41)
        market = pd.DataFrame([(d, s, 100.0) for d in dates for s in stocks],
                              columns=["Date", "stock_id", "Close"])
        scores = pd.DataFrame([(d, s, float(i) if di == 0 else float(99-i))
                               for di, d in enumerate(dates[::20]) for i, s in enumerate(stocks)],
                              columns=["Date", "stock_id", "Score_5d"])
        scores = scores[~((scores.Date == dates[40]) & (scores.stock_id == "S099"))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sp = root/"scores.parquet"; mp = root/"market.parquet"
            scores.to_parquet(sp); market.to_parquet(mp)
            with mock.patch.dict(sys.modules, {"portfolio_lab": module}):
                train.run_economic_replay(sp, mp, root/"out")
            daily = pd.read_parquet(root/"out"/"daily_returns.parquet")
            self.assertEqual(list(pd.to_datetime(daily.Date)), list(dates))
            self.assertAlmostEqual(float(daily.iloc[0].transaction_cost), .0015)
            self.assertAlmostEqual(float(daily.iloc[20].transaction_cost), .0030)

    def test_frozen_protocol_defaults_to_two_sets_and_optional_test_is_separate(self):
        train = importlib.import_module("v7_integrated_train")
        two = {"train": ["2013-01-02", "2023-11-17"],
               "validation": ["2024-01-02", "2026-06-02"]}
        train.validate_frozen_dates(two)
        parser = train.build_parser()
        self.assertEqual(parser.parse_args(["forecast"]).split, "validation")

    def test_training_budget_distinguishes_smoke_cap_from_full_epochs(self):
        train = importlib.import_module("v7_integrated_train")
        self.assertEqual(train.resolve_training_budget("smoke", epochs=20, batches_per_epoch=80,
                                                       max_steps=None, smoke_steps=2)["max_steps"], 2)
        full = train.resolve_training_budget("train", epochs=20, batches_per_epoch=80,
                                             max_steps=None, smoke_steps=2)
        self.assertEqual((full["epochs"], full["max_steps"]), (20, 1600))

    def test_bounded_training_events_preserve_exact_resume_state(self):
        train = importlib.import_module("v7_integrated_train")
        events = [train.training_event_due(step, checkpoint_interval=3, progress_interval=2,
                                           epoch_end=(step == 5), stopping=(step == 5))
                  for step in range(1, 6)]
        self.assertEqual([i + 1 for i, event in enumerate(events) if event["checkpoint"]], [3, 5])
        self.assertEqual([i + 1 for i, event in enumerate(events) if event["progress"]], [2, 4, 5])
        state = train.capture_rng_state()
        checkpoint = train.build_checkpoint({}, {}, epoch=0, batch_index=5, step=5,
            provenance={"control": "cpu"}, rng_state=state)
        self.assertEqual((checkpoint["epoch"], checkpoint["batch_index"], checkpoint["step"]),
                         (0, 5, 5))
        random_before = random.getstate()
        train.restore_rng_state(checkpoint["rng_state"])
        self.assertEqual(random.getstate(), state["python"])
        random.setstate(random_before)

    def test_graph_edge_list_conversion_validates_and_writes_csr(self):
        train = importlib.import_module("v7_integrated_train")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"graph.npz"; target = root/"csr.npz"
            np.savez(source, stock_ids=np.array(["A", "B", "C"]),
                     edge_index=np.array([[2, 0, 1], [0, 1, 2]], dtype=np.int32),
                     edge_attr=np.array([.3, .1, .2], dtype=np.float32))
            report = train.convert_knowledge_graph(source, target)
            with np.load(target, allow_pickle=False) as graph:
                self.assertEqual(graph["indptr"].tolist(), [0, 1, 2, 3])
                self.assertEqual(graph["indices"].tolist(), [1, 2, 0])
                self.assertEqual(graph["weights"].tolist(),
                                 [np.float32(.1), np.float32(.2), np.float32(.3)])
            self.assertEqual(report["edges"], 3)
            bad = root/"bad.npz"
            np.savez(bad, stock_ids=np.array(["A"]), edge_index=np.array([[0], [1]]),
                     edge_attr=np.array([1.], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "bounds"):
                train.convert_knowledge_graph(bad, root/"never.npz")

    def test_prepared_data_check_distinguishes_prices_and_allows_label_tail_nan(self):
        train = importlib.import_module("v7_integrated_train")
        features = list(train.V6_FEATURE_COLUMNS)
        dates = pd.DatetimeIndex(["2013-01-02", "2024-01-02", "2024-01-03"])
        frame = pd.DataFrame({"Date": dates, "stock_id": ["2330"]*3,
                              **{name: np.ones(3, np.float32) for name in features},
                              "Alpha_5d": [1., 2., np.nan], "Alpha_10d": [1., np.nan, np.nan]})
        market = pd.DataFrame({"Date": dates, "stock_id": ["2330"]*3,
                               "Close": [600., 601., 602.]})
        frame["observation_valid"] = True
        metadata = {**synthetic_metadata(dates,minimum=1,ids=("2330","2317")), "feature_columns": features, "feature_fingerprint": "fp",
                    "feature_order_sha256": train.canonical_fingerprint(features),
                    "raw_source_sha256": {"prices_raw.parquet": "abc"},
                    "artifact_kind": "diagnostic-not-performance-evidence"}
        frame = pd.concat([frame, frame.assign(stock_id="2317")],ignore_index=True).sort_values(["stock_id","Date"])
        market = pd.concat([market,market.assign(stock_id="2317")],ignore_index=True).sort_values(["stock_id","Date"])
        report = train.check_prepared_frames(frame, market, metadata,
                                              {"train": [str(dates[0].date())],
                                               "validation": [str(dates[1].date())]},
                                              graph_stock_ids=("2330","2317"))
        self.assertEqual(report["rows"], 6)
        frame.loc[0, "Close"] = np.nan
        with self.assertRaisesRegex(ValueError, "coverage outage"):
            train.check_prepared_frames(frame, market, metadata,
                                        {"train": [str(dates[0].date())],
                                         "validation": [str(dates[1].date())]},
                                        graph_stock_ids=("2330","2317"))

    def test_price_quality_audit_reports_ohlc_contradictions_before_prepare(self):
        train = importlib.import_module("v7_integrated_train")
        prices = pd.DataFrame({"Date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "stock_id": ["2330", "1240"], "Open": [100., 10.], "High": [101., 9.],
            "Low": [99., 8.], "Close": [100., 10.], "Volume": [1., 1.]})
        report = train.audit_selected_prices(prices)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["ohlc_contradictions"], 1)
        self.assertEqual(report["affected_stocks"], ["1240"])
        self.assertTrue(report["blocking"])

    def test_raw_calendar_audit_uses_union_not_prices_only(self):
        train = importlib.import_module("v7_integrated_train")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = root/"prices.parquet"; margin = root/"margin.parquet"
            pd.DataFrame({"Date": ["2026-09-01"], "stock_id": ["2330"]}).to_parquet(prices)
            pd.DataFrame({"date": pd.to_datetime(["2026-09-01", "2026-09-04"]),
                          "stock_id": ["2330", "2330"]}).to_parquet(margin)
            report = train.audit_raw_source_calendars({"prices": prices, "margin": margin})
        self.assertEqual(report["prices"]["dates"], 1)
        self.assertEqual(report["margin"]["dates"], 2)
        self.assertEqual(report["cross_source_calendar"]["dates"], 2)
        self.assertEqual(report["cross_source_calendar"]["date_max"], "2026-09-04")
        frames = {"prices": pd.DataFrame({"Date": pd.to_datetime(["2026-09-01"])}),
                  "margin": pd.DataFrame({"Date": pd.to_datetime(["2026-09-01", "2026-09-04"])})}
        self.assertEqual(train.cross_source_trading_calendar(frames),
                         ["2026-09-01", "2026-09-04"])

    def test_validation_metrics_are_averaged_per_date_not_globally(self):
        train = importlib.import_module("v7_integrated_train")
        daily = [{"rank_ic_5d": 1.0, "rank_ic_10d": np.nan},
                 {"rank_ic_5d": -0.5, "rank_ic_10d": 0.25}]
        got = train.aggregate_validation_metrics(daily)
        self.assertAlmostEqual(got["rank_ic_5d"], 0.25)
        self.assertAlmostEqual(got["rank_ic_10d"], 0.25)

    def test_predictive_comparison_requires_same_heldout_keys(self):
        import pandas as pd
        train = importlib.import_module("v7_integrated_train")
        candidate = pd.DataFrame({"Date": ["2025-01-02", "2025-01-02"], "stock_id": ["A", "B"],
            "Score_5d": [1., 2.], "Score_10d": [2., 1.], "Alpha_5d": [1., 3.], "Alpha_10d": [4., 2.]})
        baseline = candidate[["Date", "stock_id", "Score_5d", "Score_10d"]].copy()
        result = train.compare_predictive_frames(candidate, baseline)
        self.assertEqual(set(result), {"v7_integrated_candidate", "v2_kg_nomacro"})
        with self.assertRaisesRegex(ValueError, "identical"):
            train.compare_predictive_frames(candidate, baseline.iloc[:1])

    def test_economic_replay_rejects_live_output_target_before_import(self):
        train = importlib.import_module("v7_integrated_train")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("importlib.import_module", side_effect=AssertionError("must reject first")):
                with self.assertRaisesRegex(ValueError, "isolated"):
                    train.run_economic_replay(root/"scores.parquet", root/"market.parquet",
                                              root/"production"/"live")

    def test_economic_ties_follow_sorted_stock_first_rank(self):
        train = importlib.import_module("v7_integrated_train")
        stocks = [f"S{i:03d}" for i in reversed(range(60))]
        scores = pd.DataFrame({"Date": ["2025-01-02"]*60, "stock_id": stocks, "Score_5d": 1.0})
        market = pd.DataFrame({"Date": ["2025-01-02"]*60, "stock_id": stocks, "Close": 100.0})
        seen = []
        fake = type(sys)("portfolio_lab")
        def runner(mkt, rank, *args, **kwargs):
            seen.append(rank.copy()); return {"_daily": np.zeros(len(mkt.dates)), "n_days": len(mkt.dates)}
        fake.run_config = runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sp=root/"s.parquet"; mp=root/"m.parquet"
            scores.to_parquet(sp); market.to_parquet(mp)
            with mock.patch.dict(sys.modules, {"portfolio_lab": fake}):
                train.run_economic_replay(sp, mp, root/"out")
        self.assertEqual(seen[0].columns.tolist(), sorted(stocks))
        self.assertEqual(seen[0].iloc[0].tolist(), list(map(float, range(1, 61))))

    def test_economic_inputs_reject_duplicates_nonfinite_nonpositive_and_mismatch(self):
        train = importlib.import_module("v7_integrated_train")
        fake = type(sys)("portfolio_lab"); fake.run_config = lambda *a, **k: {"_daily": np.zeros(1)}
        base_scores = pd.DataFrame({"Date": ["2025-01-02"], "stock_id": ["A"], "Score_5d": [1.]})
        base_market = pd.DataFrame({"Date": ["2025-01-02"], "stock_id": ["A"], "Close": [100.]})
        cases = [
            (pd.concat([base_scores, base_scores]), base_market, "duplicate"),
            (base_scores.assign(Score_5d=np.inf), base_market, "finite"),
            (base_scores, base_market.assign(Close=0.), "positive"),
            (base_scores, base_market.assign(stock_id="B"), "missing scored-stock"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, (scores, market, message) in enumerate(cases):
                sp, mp = root/f"s{number}.parquet", root/f"m{number}.parquet"
                scores.to_parquet(sp); market.to_parquet(mp)
                with mock.patch.dict(sys.modules, {"portfolio_lab": fake}):
                    with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                        train.run_economic_replay(sp, mp, root/f"out{number}")


class TestCalendarPolicy(unittest.TestCase):
    def fixture(self, n=75):
        dates = pd.bdate_range('2023-01-02',periods=n)
        rows = [(day,stock,100.+i) for stock in ('A','B','C','D') for i,day in enumerate(dates)]
        prices = pd.DataFrame(rows,columns=['Date','stock_id','Close'])
        prices['Open']=prices.Close; prices['High']=prices.Close+1; prices['Low']=prices.Close-1; prices['Volume']=100.
        prices['source'] = ['archive','exchange']*(len(prices)//2)
        calendar = [str(day.date()) for day in dates]
        document = {'trading_calendar':calendar,'provenance':'independent synthetic oracle'}
        universe = {day:['A','B','C','D'] for day in calendar}
        document.update(expected_universe=universe, universe_provenance='explicit synthetic four-stock basket')
        return prices,document,universe

    def test_exact_calendar_endpoints_and_interior_gaps(self):
        prices,doc,universe = self.fixture(16)
        checked,report = assess_prices(prices,doc,expected_universe=universe)
        self.assertFalse(report['blocking'])
        labels = calendar_labels(checked,doc['trading_calendar'])
        a = labels[labels.stock_id=='A'].reset_index(drop=True)
        self.assertAlmostEqual(a.loc[0,'Alpha_5d'],.05)
        self.assertAlmostEqual(a.loc[0,'Alpha_10d'],.10)
        prices = prices[~((prices.stock_id=='A') & (prices.Date==pd.Timestamp(doc['trading_calendar'][3])))]
        checked,_ = assess_prices(prices,doc,expected_universe=universe)
        a = calendar_labels(checked,doc['trading_calendar'],False).query("stock_id=='A'").reset_index(drop=True)
        self.assertAlmostEqual(a.loc[0,'Alpha_5d'],.05)  # t+5, never sixth observed row
        self.assertTrue(np.isnan(a.loc[3,'Alpha_5d']))
        a = calendar_labels(checked,doc['trading_calendar']).query("stock_id=='A'").reset_index(drop=True)
        self.assertTrue(np.isnan(a.loc[0,'Alpha_5d']))
        self.assertAlmostEqual(a.loc[4,'Alpha_5d'],109/104-1)
        self.assertTrue(np.isnan(a.loc[11,'Alpha_5d']))

    def test_quarantine_sparse_outlier_whole_day_and_calendar_ambiguity(self):
        prices,doc,universe = self.fixture(12)
        prices.loc[0,'High']=1.
        checked,report = assess_prices(prices,doc,expected_universe=universe)
        self.assertFalse(checked.loc[0,'observation_valid'])
        self.assertEqual(checked.loc[0,'High'],1.)
        self.assertFalse(report['blocking'])
        self.assertIn('QUARANTINE',[e['severity'] for e in report['entries']])
        self.assertTrue(checked.loc[1:,'observation_valid'].all())
        absent = prices[prices.Date!=pd.Timestamp(doc['trading_calendar'][4])]
        _,report = assess_prices(absent,doc,expected_universe=universe)
        self.assertTrue(report['blocking'])
        entry = next(e for e in report['entries'] if e['reason_code']=='MAJOR_COVERAGE_OUTAGE')
        self.assertEqual(entry['dates'],[doc['trading_calendar'][4]])
        self.assertEqual(entry['denominator'],4)
        missing_universe = dict(doc); del missing_universe['expected_universe']
        with self.assertRaises(ValueError): assess_prices(prices,missing_universe)
        with self.assertRaisesRegex(ValueError,'calendar'): assess_prices(prices,{},expected_universe=universe)

    def test_sixty_calendar_positions_and_current_graph_membership(self):
        import v7_integrated_train as train
        prices,doc,universe = self.fixture()
        calendar=doc['trading_calendar']; dates=pd.to_datetime(calendar)
        frame = pd.DataFrame({'Date':list(dates)*2,'stock_id':['A']*75+['B']*75})
        for name in train.V6_FEATURE_COLUMNS: frame[name] = list(range(75))*2
        frame['Alpha_5d']=1.;frame['Alpha_10d']=np.nan;frame['observation_valid']=True
        frame=frame[~((frame.stock_id=='A') & (frame.Date==dates[30]))].copy()
        frame.loc[(frame.stock_id=='B') & (frame.Date==dates[-1]),'observation_valid']=False
        frame.loc[(frame.stock_id=='A') & (frame.Date==dates[0]),'observation_valid']=False
        metadata={**protocol_metadata(doc),'feature_columns':list(train.V6_FEATURE_COLUMNS),'feature_fingerprint':'test'}
        index=train.PreparedDataIndex.from_parquet('unused',metadata,parquet_reader=lambda *a,**k:frame)
        seen=[]
        dataset=train.DailyCrossSectionDataset('unused',[calendar[-1]],metadata,prepared_index=index,
            graph_provider=lambda ids:(seen.append(tuple(ids)) or torch.empty(2,0,dtype=torch.long),torch.empty(0)))
        sample=dataset[0]
        self.assertEqual(seen,[('A',)])
        self.assertEqual(sample['x'].shape,(1,60,59))
        self.assertTrue(sample['padding_mask'].all())
        self.assertFalse(sample['observation_mask'][0,15])
        self.assertEqual(sample['x'][0,14,0],29.)
        self.assertEqual(sample['x'][0,16,0],31.)
        values,labels,history,valid=index.window('A',calendar[4],60)
        self.assertEqual(len(values),5)
        self.assertTrue(history.all())  # quarantine must not shift stock's first history
        self.assertFalse(valid[0])
        with self.assertRaisesRegex(ValueError,'calendar'):
            train.PreparedDataIndex.from_parquet('unused',{'feature_columns':list(train.V6_FEATURE_COLUMNS),'feature_fingerprint':'old'},parquet_reader=lambda *a,**k:frame)
        incompatible=dict(metadata,protocol_fingerprint='old')
        with self.assertRaisesRegex(ValueError,'fingerprint'): validate_protocol(incompatible)

    def test_internal_hole_preserved_invalid_nan_invariance_and_padding(self):
        import v7_integrated_model as mod
        torch.manual_seed(19)
        lengths=[]
        def recording(*args,**kwargs):
            lengths.append(args[0].shape[1]);return synthetic_ssd(*args,**kwargs)
        graph=RecordingGraph()
        model=mod.MarketMambaV7Integrated(get_test_config(),ssd_fn=recording,graph_layer=graph).eval()
        x=torch.randn(3,60,59)
        history=torch.ones(3,60,dtype=torch.bool);history[0,:5]=False
        observed=history.clone();observed[0,30]=False;observed[2,-1]=False
        edges=torch.tensor([[0,1,2],[1,0,0]]);attrs=torch.ones(3)
        baseline=model(x,edges,attrs,history,observed)
        self.assertEqual(lengths[:2],[60,55])
        self.assertTrue(torch.equal(graph.last[1],torch.tensor([[0,1],[1,0]])))
        changed=x.clone();changed[~observed]=float('nan')
        self.assertTrue(torch.equal(baseline,model(changed,edges,attrs,history,observed)))
        changed[~observed]=1e30
        self.assertTrue(torch.equal(baseline,model(changed,edges,attrs,history,observed)))
        self.assertTrue(torch.equal(baseline[2],torch.zeros(2)))
        # Missing embedding participates at the internal calendar hole.
        model(x,edges,attrs,history,observed).sum().backward()
        self.assertIsNotNone(model.missing_observation.grad)

    def test_sparse_heads_steps_skips_and_nonfinite_failures(self):
        import v7_integrated_train as train
        model=nn.Linear(2,2);optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        features=torch.eye(2)
        labels=torch.tensor([[1.,float('nan')],[2.,float('nan')]])
        before=scheduler.last_epoch
        self.assertIsNotNone(train.training_step(model,optimizer,scheduler,model(features),labels))
        self.assertEqual(scheduler.last_epoch,before+1)
        params=[p.clone() for p in model.parameters()]
        self.assertIsNone(train.training_step(model,optimizer,scheduler,model(features),torch.full((2,2),float('nan'))))
        self.assertEqual(scheduler.last_epoch,before+1)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(params,model.parameters())))
        optimizer.zero_grad()
        self.assertIsNotNone(train.training_step(model,optimizer,scheduler,model(features),labels.flip(1)))
        self.assertEqual(scheduler.last_epoch,before+2)
        metric=train.rank_ic_by_horizon(torch.tensor([[1.,9.],[2.,8.]]),labels)
        self.assertAlmostEqual(metric['rank_ic_5d'],1.,places=6)
        self.assertTrue(np.isnan(metric['rank_ic_10d']))
        with self.assertRaisesRegex(ValueError,'nonfinite predictions'):
            train.rank_ic_by_horizon(torch.full((2,2),float('nan')),labels)
        optimizer.zero_grad()
        handle=model.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
        with self.assertRaisesRegex(ValueError,'nonfinite gradients'):
            train.training_step(model,optimizer,scheduler,model(features),labels)
        handle.remove()
        self.assertEqual(scheduler.last_epoch,before+2)
        optimizer.zero_grad()
        with self.assertRaisesRegex(ValueError,'nonfinite predictions'):
            train.training_step(model,optimizer,scheduler,torch.full((2,2),float('nan')),labels)
        with self.assertRaisesRegex(ValueError,'nonfinite loss'):
            train.training_step(model,optimizer,scheduler,torch.full((2,2),1e30,requires_grad=True),labels)
        self.assertEqual(scheduler.last_epoch,before+2)

    def test_gap_adapter_restarts_price_state_preserves_pit_context(self):
        import v7_integrated_train as train
        prices,doc,universe=self.fixture(12)
        checked,_=assess_prices(prices,doc,expected_universe=universe)
        checked.loc[(checked.stock_id=='A') & (checked.Date==pd.Timestamp(doc['trading_calendar'][3])),'observation_valid']=False
        data={name:None for name in train.RAW_SOURCES}
        data['prices']=checked
        data['revenue']=pd.DataFrame({'Date':pd.to_datetime(['2022-12-01','2023-01-03','2024-01-01']),
                                     'stock_id':['A','B','A'],'revenue':[42.,99.,500.]})
        calls=[]
        def helper(df_price,**kwargs):
            calls.append((df_price.copy(),kwargs))
            out=df_price.copy();out['rolling']=out.Close.diff()
            return out
        result=train.gap_safe_features(checked,data,doc['trading_calendar'],helper)
        after=result[(result.stock_id=='A') & (result.Date==pd.Timestamp(doc['trading_calendar'][4]))]
        self.assertTrue(after['rolling'].isna().all())
        a_calls=[kwargs for frame,kwargs in calls if frame.stock_id.iloc[0]=='A']
        self.assertEqual(len(a_calls),2)
        for kwargs in a_calls:
            self.assertEqual(kwargs['df_rev'].revenue.tolist(),[42.])
            self.assertIsNone(kwargs['df_fed_rate'])
        b=result[result.stock_id=='B'].sort_values('Date')
        self.assertTrue(np.allclose(b['rolling'].iloc[1:],1.))


    def test_actual_cpu_training_loop_no_target_run_and_resume_counters(self):
        import v7_integrated_train as train
        class TinyModel(nn.Module):
            def __init__(self,*a,**k):
                super().__init__();self.head=nn.Linear(59,2)
            def forward(self,x,edges,attrs,padding,observation=None): return self.head(x[:,-1])
        empty=torch.full((2,2),float('nan'))
        valid=torch.tensor([[float('nan'),1.],[float('nan'),2.]])
        x=torch.zeros(2,60,59);x[1,:,0]=1.
        def sample(labels):
            return {'x':x,'padding_mask':torch.ones(2,60,dtype=torch.bool),
                    'observation_mask':torch.ones(2,60,dtype=torch.bool),'stock_ids':('A','B'),
                    'edge_index':torch.empty(2,0,dtype=torch.long),'edge_attr':torch.empty(0),'labels':labels}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=train.build_parser().parse_args(['train','--device','cpu','--checkpoint',str(root/'c.pt'),
                    '--epochs','1','--max-steps','1'])
            args.kg=root/'unused';args.feature_parquet=root/'unused'
            graph=mock.Mock();graph.edges_for=lambda ids:None
            with mock.patch.object(train,'KnowledgeGraphCSR',return_value=graph), \
                 mock.patch.object(train.PreparedDataIndex,'from_parquet',return_value=object()), \
                 mock.patch.object(train,'MarketMambaV7Integrated',TinyModel), \
                 mock.patch.object(train,'official_ssd_callable',return_value=synthetic_ssd), \
                 mock.patch.object(train,'build_provenance',return_value={'protocol':'explicit-test'}), \
                 mock.patch.object(train,'DailyCrossSectionDataset',side_effect=[[sample(empty),sample(valid)],[sample(valid)]]):
                result=train._run_training(args,get_test_config(),{}, {'train':['a','b'],'validation':['v']})
            self.assertEqual(result['step'],1)
            self.assertEqual(result['skipped_batches'],1)
            self.assertEqual(result['head_target_counts'],[0,2])
            checkpoint=train.load_checkpoint(root/'c.pt',expected_provenance={'protocol':'explicit-test'})
            self.assertEqual(checkpoint['scheduler_state']['last_epoch'],1)
            self.assertEqual(checkpoint['training_counters']['skipped_batches'],1)
            self.assertEqual((checkpoint['epoch'],checkpoint['batch_index']),(1,0))
            args.checkpoint=root/'no.pt'
            with mock.patch.object(train,'KnowledgeGraphCSR',return_value=graph), \
                 mock.patch.object(train.PreparedDataIndex,'from_parquet',return_value=object()), \
                 mock.patch.object(train,'MarketMambaV7Integrated',TinyModel), \
                 mock.patch.object(train,'official_ssd_callable',return_value=synthetic_ssd), \
                 mock.patch.object(train,'build_provenance',return_value={'protocol':'explicit-test'}), \
                 mock.patch.object(train,'DailyCrossSectionDataset',side_effect=[[sample(empty),sample(empty)],[sample(valid)]]):
                with self.assertRaisesRegex(RuntimeError,'no-target whole run'):
                    train._run_training(args,get_test_config(),{}, {'train':['a','b'],'validation':['v']})
            self.assertFalse((root/'no.pt').exists())

    def test_protected_price_rolling_scaling_and_asof_with_manual_oracles(self):
        import ast
        import logging
        import v7_integrated_train as train
        base=Path('/home/frank/projects/MarketMamba/.artifacts/v7-data-policy/workspace/V6/marketmamba')
        local=HERE.parent/'marketmamba'
        if local.exists(): base=local
        config_tree=ast.parse((base/'config.py').read_text())
        groups=next(ast.literal_eval(node.value) for node in config_tree.body
                    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FEATURE_GROUPS' for t in node.targets))
        namespace={'pd':pd,'np':np,'logger':logging.getLogger('protected-cpu-test'),
                   'FEATURE_COLS':list(train.V6_FEATURE_COLUMNS),'FEATURE_GROUPS':groups,
                   'AVAIL_COLS':[], 'NEUTRALIZE_EXCLUDE':frozenset(), 'MACRO_TS_MIN_HIST':252,
                   'PROCESSED_DIR':base, 'SEQ_LEN':60}
        tree=ast.parse((base/'data/feature_engineer.py').read_text())
        # Execute protected function definitions only; never import side-effectful V6 config.
        functions=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)],type_ignores=[])
        exec(compile(functions,str(base/'data/feature_engineer.py'),'exec',flags=__import__('__future__').annotations.compiler_flag),namespace)
        prices,doc,universe=self.fixture(75)
        prices.Close += prices.stock_id.map({'A':0.,'B':100.,'C':200.,'D':300.})
        prices['Open']=prices.Close;prices['High']=prices.Close+1;prices['Low']=prices.Close-1
        checked,_=assess_prices(prices,doc,expected_universe=universe)
        data={name:None for name in train.RAW_SOURCES};data['prices']=checked
        def protected_price(df_price,**kwargs):
            return namespace['_add_price_momentum_features'](df_price.copy())
        healthy=train.gap_safe_features(checked,data,doc['trading_calendar'],protected_price)
        a=healthy[healthy.stock_id=='A'].reset_index(drop=True)
        self.assertAlmostEqual(a.loc[5,'Return_5d'],.05)
        self.assertAlmostEqual(a.loc[19,'MA_20'],109.5)
        checked.loc[(checked.stock_id=='A') & (checked.Date==pd.Timestamp(doc['trading_calendar'][30])),'observation_valid']=False
        gap=train.gap_safe_features(checked,data,doc['trading_calendar'],protected_price)
        row=gap[(gap.stock_id=='A') & (gap.Date==pd.Timestamp(doc['trading_calendar'][31]))].iloc[0]
        self.assertTrue(np.isnan(row.Return_5d))
        self.assertTrue(np.isnan(row.MA_20))
        # Hand-computed 1/99% winsorization and sample-standard-deviation scaling.
        frame=pd.DataFrame({'Date':pd.to_datetime(['2023-01-02']*4),'stock_id':['A','B','C','D']})
        for name in train.V6_FEATURE_COLUMNS: frame[name]=[100.,200.,300.,400.]
        scaled=namespace['clean_and_scale'](frame,macro_norm='ts',neutralize='none')
        values=np.array([103.,200.,300.,397.]);expected=(values-250.)/np.sqrt(((values-250.)**2).sum()/3)
        self.assertTrue(np.allclose(scaled.Close,expected))
        ref=pd.DataFrame({'Date':pd.to_datetime(['2022-11-01','2022-12-01']),
                          'stock_id':['A','A'],'revenue':[100.,150.]})
        context=pd.DataFrame({'Date':pd.to_datetime(['2023-01-10','2023-01-11','2023-02-01']), 'stock_id':['A']*3})
        asof=namespace['_merge_revenue'](context,ref)
        self.assertTrue(np.allclose(asof.Revenue_MoM,[0.,.5,.5]))


    def test_training_report_floor_leading_padding_and_old_checkpoint(self):
        import v7_integrated_train as train
        prices,doc,universe=self.fixture(12)
        checked,report=assess_prices(prices,doc,expected_universe=universe)
        labels=calendar_labels(checked,doc['trading_calendar'])
        frame=checked.merge(labels,on=['Date','stock_id'])
        assess_training(frame,[doc['trading_calendar'][0]],QualityPolicy(),report)
        self.assertTrue(report['blocking'])
        self.assertEqual(report['training_coverage']['usable_dates'],1)
        self.assertEqual(report['entries'][-1]['reason_code'],'INSUFFICIENT_USABLE_TRAINING_DATA')
        frame=frame[frame.Date >= pd.Timestamp(doc['trading_calendar'][4])].copy()
        for name in train.V6_FEATURE_COLUMNS: frame[name]=1.
        metadata={**protocol_metadata(doc),'feature_columns':list(train.V6_FEATURE_COLUMNS),'feature_fingerprint':'test'}
        index=train.PreparedDataIndex.from_parquet('unused',metadata,parquet_reader=lambda *a,**k:frame)
        dataset=train.DailyCrossSectionDataset('unused',[doc['trading_calendar'][5]],metadata,prepared_index=index)
        sample=dataset[0]
        self.assertEqual(sample['padding_mask'][0].tolist(),[False]*58+[True]*2)
        self.assertEqual(sample['observation_mask'][0].tolist(),[False]*58+[True]*2)
        stream=io.BytesIO()
        torch.save({'format':'marketmamba-v7-integrated-v2','provenance':{'protocol':'old'}},stream)
        stream.seek(0)
        with self.assertRaisesRegex(ValueError,'format'):
            train.load_checkpoint(stream,expected_provenance={'protocol':'explicit-test'})
        # Independent expected denominator: two absent stocks out of four is major.
        sparse=prices[~((prices.Date==pd.Timestamp(doc['trading_calendar'][0])) & prices.stock_id.isin(['A','B']))]
        _,report=assess_prices(sparse,doc,expected_universe=universe)
        self.assertTrue(report['blocking'])
        # Endpoint-only opt-in affects fingerprint and only the unavailable endpoint head.
        checked.loc[(checked.stock_id=='A') & (checked.Date==pd.Timestamp(doc['trading_calendar'][5])),'observation_valid']=False
        targets=calendar_labels(checked,doc['trading_calendar'],False).query("stock_id=='A'").reset_index(drop=True)
        self.assertTrue(np.isnan(targets.loc[0,'Alpha_5d']))
        self.assertAlmostEqual(targets.loc[0,'Alpha_10d'],.10)
        self.assertNotEqual(protocol_metadata(doc)['protocol_fingerprint'],protocol_metadata(doc,QualityPolicy(interior_gap_invalidates_target=False))['protocol_fingerprint'])


import copy
from v7_integrated_config import V6_FEATURE_COLUMNS
import v7_integrated_data_quality as quality
import v7_integrated_train as train
class Checks(unittest.TestCase):
 def document(self):
  days=['2023-01-03','2023-01-04','2024-02-01']
  return {'trading_calendar':days,'provenance':'independent synthetic calendar','expected_universe':{d:['A','B','C','D'] for d in days},'universe_provenance':'fixed synthetic basket'}
 def test_expected_universe_is_bound_to_protocol(self):
  meta=quality.protocol_metadata(self.document());changed=copy.deepcopy(meta)
  self.assertIn('expected_universe',meta)
  changed['expected_universe']['2023-01-03']=['A']
  with self.assertRaises(ValueError):quality.validate_protocol(changed)
 def test_effective_feature_outage_is_blocked(self):
  doc=self.document();meta=quality.protocol_metadata(doc);meta.update(feature_columns=list(V6_FEATURE_COLUMNS),feature_fingerprint='synthetic',raw_source_sha256={'fixture':'synthetic'},quality_report={'blocking':False})
  rows=[];prices=[]
  for day in doc['trading_calendar']:
   for sid in ['A','B','C','D']:
    row=dict.fromkeys(V6_FEATURE_COLUMNS,0.1);row.update(Date=day,stock_id=sid,observation_valid=not(day=='2024-02-01' and sid!='A'),Alpha_5d=.01,Alpha_10d=.02);rows.append(row);prices.append(dict(Date=day,stock_id=sid,Close=100.))
  features=pd.DataFrame(rows).sort_values(['stock_id','Date']);market=pd.DataFrame(prices).sort_values(['stock_id','Date'])
  with self.assertRaises(ValueError):
   train.check_prepared_frames(features,market,meta,{'train':['2023-01-03','2023-01-04'],'validation':['2024-02-01']},graph_stock_ids=['A','B','C','D'])
 def test_null_stock_key_blocks(self):
  doc={'trading_calendar':['2023-01-03'],'provenance':'synthetic','expected_universe':{'2023-01-03':['A']},'universe_provenance':'synthetic fixed basket'}
  frame=pd.DataFrame([dict(Date='2023-01-03',stock_id=sid,Open=100.,High=101.,Low=99.,Close=100.,Volume=10.) for sid in ['A',None]])
  try: _,report=quality.assess_prices(frame,doc,expected_universe=doc['expected_universe'])
  except ValueError:return
  self.assertTrue(report['blocking'])

class ContractReworkTests(unittest.TestCase):
 document = Checks.document
 def test_metadata_roundtrip_and_tampering(self):
  meta=quality.protocol_metadata(self.document())
  self.assertEqual(quality.validate_protocol(json.loads(json.dumps(meta)))[0],self.document()['trading_calendar'])
  for key,value in [('calendar_provenance','changed'),('universe_provenance','changed'),
                    ('selected_universe',['A']),('protocol_version','v7-calendar-v1'),
                    ('trading_calendar',['2023-01-03'])]:
   changed=copy.deepcopy(meta);changed[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):quality.validate_protocol(changed)
  for key in ['expected_universe','universe_provenance','calendar_provenance','quality_policy']:
   changed=copy.deepcopy(meta);del changed[key]
   with self.subTest(missing=key),self.assertRaises(ValueError):quality.validate_protocol(changed)

 def test_ambiguous_stock_keys_have_graded_block(self):
  doc=self.document();day=doc['trading_calendar'][0]
  for sid in [None,np.nan,'','   ',' A']:
   frame=pd.DataFrame([dict(Date=day,stock_id=x,Open=100.,High=101.,Low=99.,Close=100.,Volume=10.) for x in ['A',sid]])
   with self.subTest(sid=sid),self.assertRaises(quality.QualityBlocked) as caught:
    quality.assess_prices(frame,doc)
   self.assertTrue(caught.exception.report['blocking'])
   self.assertEqual(caught.exception.report['entries'][0]['reason_code'],'AMBIGUOUS_STOCK_KEY')
   self.assertIn('BLOCK',caught.exception.report['summary_zh_TW'])

 def test_declared_union_retains_early_rows_and_excludes_etf(self):
  doc=self.document();doc['expected_universe'][doc['trading_calendar'][0]]=['A']
  rows=pd.DataFrame({'Date':[doc['trading_calendar'][0]]*3,'stock_id':['A','D','ETF']})
  selected,audit=quality.select_declared_prices(rows,doc)
  self.assertEqual(selected.stock_id.tolist(),['A','D'])
  self.assertEqual(audit['excluded_rows'],1)
  self.assertEqual(audit['excluded_stocks'],['ETF'])
  self.assertEqual(audit['universe_provenance'],'fixed synthetic basket')

 def test_effective_coverage_boundaries_absent_dates_and_graph(self):
  doc=self.document();meta=quality.protocol_metadata(doc)
  meta.update(feature_columns=list(V6_FEATURE_COLUMNS),feature_fingerprint='synthetic',raw_source_sha256={'fixture':'synthetic'})
  rows=[];prices=[]
  for day in doc['trading_calendar']:
   for sid in ['A','B','C','D']:
    row=dict.fromkeys(V6_FEATURE_COLUMNS,.1);row.update(Date=day,stock_id=sid,observation_valid=True,Alpha_5d=.01,Alpha_10d=.02);rows.append(row)
    prices.append(dict(Date=day,stock_id=sid,Close=100.))
  frame=pd.DataFrame(rows).sort_values(['stock_id','Date']);market=pd.DataFrame(prices).sort_values(['stock_id','Date'])
  splits={'train':doc['trading_calendar'][:2],'validation':doc['trading_calendar'][2:]}
  kwargs=dict(graph_stock_ids=['A','B','C','D'])
  self.assertEqual(train.check_prepared_frames(frame,market,meta,splits,**kwargs)['rows'],12)
  validation=frame.Date.eq('2024-02-01')
  sparse=frame.copy();sparse.loc[validation & sparse.stock_id.eq('D'),'observation_valid']=False
  train.check_prepared_frames(sparse,market,meta,splits,**kwargs)
  sparse.loc[validation & sparse.stock_id.eq('C'),'observation_valid']=False
  with self.assertRaisesRegex(ValueError,'coverage'):train.check_prepared_frames(sparse,market,meta,splits,**kwargs)
  with self.assertRaisesRegex(ValueError,'coverage'):train.check_prepared_frames(frame[~validation],market,meta,splits,**kwargs)
  with self.assertRaisesRegex(ValueError,'graph'):train.check_prepared_frames(frame,market,meta,splits,graph_stock_ids=['A','B','C'])
  broken=frame.copy();broken.loc[validation & broken.stock_id.isin(['B','C','D']),'Open']=np.nan
  with self.assertRaises(ValueError):train.check_prepared_frames(broken,market,meta,splits,**kwargs)


if __name__ == "__main__":
    unittest.main()
