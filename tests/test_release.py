"""Behavioral checks for protocol isolation, discovery, scoring, and artifact safety."""

import tempfile
import unittest
import contextlib
import io
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from actiondiscovery.cli import main
from actiondiscovery.config import (
    Config,
    DiscoveryConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)
from actiondiscovery.data import FeatureDataset, Protocol
from actiondiscovery.discovery import DiscoveryState, ggsm, intervals, proposal_score
from actiondiscovery.engine import (
    evaluate_checkpoint,
    load_model,
    predict,
    seed_everything,
    train,
)
from actiondiscovery.loss import training_loss
from actiondiscovery.metrics import (
    IGNORED_LABEL,
    edit_score,
    evaluate,
    segment_counts,
    summarize_folds,
)
from actiondiscovery.model import MultiStageModel

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        metadata = self.root / "protocol"
        metadata.mkdir()
        (metadata / "mapping.txt").write_text("5 known\n8 hidden_one\n12 hidden_two\n")
        (metadata / "set1.actions.bundle").write_text("known\n")
        (metadata / "train.split1.bundle").write_text("train.txt\n")
        (metadata / "test.split1.bundle").write_text("test.txt\n")
        self.protocol = Protocol.load(metadata)
        self.config = Config(
            "protocol",
            "fixture",
            1,
            model=ModelConfig(4, 8, 2, 2, 0),
            training=TrainingConfig(epochs=1),
            discovery=DiscoveryConfig(max_components=3),
        )
        folder = self.root / "fixture"
        (folder / "features").mkdir(parents=True)
        (folder / "groundTruth").mkdir()
        rng = np.random.default_rng(42)
        for name in ["train", "test"]:
            np.save(
                folder / "features" / f"{name}.npy",
                rng.normal(size=(4, 24)).astype(np.float32),
            )
            (folder / "groundTruth" / f"{name}.txt").write_text(
                "known\n" * 8 + "hidden_one\n" * 8 + "hidden_two\n" * 8
            )
        self.training = FeatureDataset(
            self.config, self.protocol, self.root, 1, "train"
        )
        self.testing = FeatureDataset(self.config, self.protocol, self.root, 1, "test")
        seed_everything(0)
        torch.set_num_threads(1)

    def tearDown(self):
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            handler.close()
        self.temporary.cleanup()

    def test_paper_recipes_and_protocol_counts(self):
        for name, known in [
            ("breakfast", 33),
            ("fs_fine", 13),
            ("fs_coarse", 7),
            ("desktop_assembly", 17),
        ]:
            config = load_config(ROOT / "configs" / f"{name}.yaml")
            protocol = Protocol.load(ROOT / "protocols" / name)
            self.assertEqual(len(protocol.known_ids), known)
            self.assertEqual(config.training.temperature, 0.4)
            self.assertEqual(config.training.smoothing_weight, 0.15)
            self.assertEqual(config.discovery.alpha, 0.001)

    def test_unknown_labels_are_removed_from_training_targets(self):
        _, before = self.training.training_sample("train.txt")
        annotation = self.training.directory / "groundTruth/train.txt"
        annotation.write_text("known\n" * 8 + "hidden_two\n" * 16)
        _, after = self.training.training_sample("train.txt")
        torch.testing.assert_close(before, after)
        self.assertEqual(before.tolist(), [1] * 8 + [0] * 16)

    def test_unknown_configuration_fields_are_rejected(self):
        with self.assertRaises(TypeError):
            Config.from_dict(
                {
                    "protocol": "fixture",
                    "dataset_folder": "fixture",
                    "splits": 1,
                    "training": {"temperatur": 0.4},
                }
            )
        self.config.model.layers = 1.5
        with self.assertRaises(ValueError):
            self.config.validate()

    def test_single_frame_loss_and_model(self):
        model = MultiStageModel(self.config.model, 1)
        logits, embeddings = model(torch.ones(1, 4, 1))
        loss = training_loss(
            logits,
            embeddings,
            torch.zeros(1, 1, dtype=torch.long),
            self.config.training,
        )
        self.assertTrue(torch.isfinite(loss["loss"]))
        loss["loss"].backward()
        self.assertEqual(logits.shape, (2, 1, 2, 1))

    def test_padding_is_excluded_from_contrastive_loss(self):
        from actiondiscovery.loss import temporal_contrastive

        features = torch.randn(1, 8, 24)
        labels = torch.zeros(1, 24, dtype=torch.long)
        seed_everything(1)
        reference = temporal_contrastive(features, labels, self.config.training)
        seed_everything(1)
        padded = temporal_contrastive(
            torch.cat([features, torch.randn(1, 8, 6)], dim=2),
            torch.cat([labels, torch.full((1, 6), -100)], dim=1),
            self.config.training,
        )
        torch.testing.assert_close(reference, padded)

    def test_intervals_include_singletons_and_empty_sequences(self):
        self.assertEqual(intervals(np.array([1, 2, 2, 3])), [(0, 1), (1, 3), (3, 4)])
        self.assertEqual(intervals(np.array([])), [])
        self.assertEqual(proposal_score([(0, 1)], [(0, 1)], 0.001), 1.0)
        self.assertLess(
            proposal_score([(0, 4)], [(0, 1)], 0.1),
            proposal_score([(0, 4)], [(0, 1)], 0),
        )

    def test_no_known_prior_uses_backbone_intervals(self):
        embeddings = np.ones((3, 8))
        self.assertEqual(ggsm(embeddings, np.zeros(3), self.config.discovery), [(0, 3)])

    def test_empty_and_single_segment_discovery(self):
        empty = DiscoveryState.fit([], 8, self.config.discovery, 0)
        self.assertEqual(empty.centroids.shape, (0, 8))
        one = DiscoveryState.fit([np.ones(8)], 8, self.config.discovery, 0)
        self.assertEqual(one.centroids.shape, (1, 8))
        result = one.predict(
            np.ones((3, 8)), np.zeros(3, dtype=int), self.config.discovery, 1
        )
        self.assertEqual(result.tolist(), [2, 2, 2])

    def test_positive_bic_scores_still_select_a_valid_model(self):
        with patch("actiondiscovery.discovery.GaussianMixture") as mixture:
            mixture.return_value.bic.side_effect = [100.0, 50.0]
            state = DiscoveryState.fit(
                [np.zeros(8), np.ones(8)], 8, self.config.discovery, 0
            )
        self.assertEqual(len(state.centroids), 2)

    def test_dataset_level_assignment_is_shared_across_videos(self):
        predictions = {"one": np.array([1, 2, 3]), "two": np.array([2, 3])}
        targets = {"one": np.array([5, 8, 12]), "two": np.array([12, 8])}
        result = evaluate(predictions, targets, self.protocol, [])
        self.assertEqual(result["metrics"]["known"]["MoF"], 100)
        self.assertEqual(result["metrics"]["unknown"]["MoF"], 50)

    def test_perfect_and_empty_partition_metrics(self):
        result = evaluate(
            {"video": np.array([1, 2, 3])},
            {"video": np.array([5, 8, 12])},
            self.protocol,
            [],
        )
        self.assertEqual(result["metrics"]["unknown"]["F1@50"], 100)
        result = evaluate(
            {"video": np.array([1])}, {"video": np.array([5])}, self.protocol, []
        )
        self.assertEqual(result["metrics"]["unknown"]["frames"], 0)

    def test_unmatched_clusters_retain_false_positive_boundaries(self):
        result = evaluate(
            {"video": np.array([2, 2, 3, 3, 4, 5])},
            {"video": np.array([8, 8, 12, 12, 8, 8])},
            self.protocol,
            [],
        )
        # Two matched segments, two distinct unmatched predictions, one missed
        # ground-truth segment: F1 = 2*2 / (2*2 + 2 + 1).
        self.assertAlmostEqual(result["metrics"]["unknown"]["F1@50"], 400 / 7)

    def test_partition_scoring_preserves_timeline_and_penalizes_false_positives(self):
        result = evaluate(
            {"video": np.array([1, 1, 1, 1, 1, 1])},
            {"video": np.array([5, 5, 8, 8, 5, 5])},
            self.protocol,
            [],
        )
        known = result["metrics"]["known"]
        self.assertEqual(known["MoF"], 100)
        self.assertEqual(
            known["Edit"], 50
        )  # Two known segments, not one concatenated segment.
        self.assertAlmostEqual(known["F1@10"], 200 / 3)
        self.assertEqual(known["F1@50"], 0)  # Original interval IoU is 2/6, not 1.
        result = evaluate(
            {"video": np.array([1, 2, 1])},
            {"video": np.array([5, 8, 8])},
            self.protocol,
            [],
        )
        self.assertEqual(result["metrics"]["known"]["MoF"], 100)
        self.assertAlmostEqual(result["metrics"]["known"]["F1@50"], 200 / 3)

    def test_partition_labels_are_ignored_independently(self):
        result = evaluate(
            {"video": np.array([1, 2, 1])},
            {"video": np.array([5, 5, 5])},
            self.protocol,
            [],
        )
        known = result["metrics"]["known"]
        self.assertAlmostEqual(known["MoF"], 200 / 3)
        self.assertEqual(known["Edit"], 50)
        self.assertEqual(known["F1@50"], 0)

    def test_breakfast_skips_empty_targets_but_fs_counts_false_positives(self):
        predictions = {"present": np.array([1]), "absent": np.array([1])}
        targets = {"present": np.array([5]), "absent": np.array([8])}
        fs = evaluate(predictions, targets, self.protocol, [])
        breakfast = evaluate(
            predictions, targets, self.protocol, [], skip_empty_targets=True
        )
        self.assertAlmostEqual(fs["metrics"]["known"]["F1@50"], 200 / 3)
        self.assertEqual(fs["metrics"]["known"]["Edit"], 50)
        self.assertEqual(breakfast["metrics"]["known"]["F1@50"], 100)
        self.assertEqual(breakfast["metrics"]["known"]["Edit"], 100)
        for result in (fs, breakfast):
            self.assertEqual(result["metrics"]["known"]["MoF"], 100)
            self.assertEqual(result["metrics"]["known"]["frames"], 1)

    def test_ignored_background_and_unmatched_clusters(self):
        result = evaluate(
            {"video": np.array([1, 2, 1])},
            {"video": np.array([5, 8, 5])},
            self.protocol,
            ["known"],
        )
        self.assertEqual(result["metrics"]["known"]["frames"], 0)
        self.assertEqual(result["metrics"]["unknown"]["F1@50"], 100)
        # Both unknown clusters have no unknown ground-truth class to match.
        result = evaluate(
            {"video": np.array([2, 3])}, {"video": np.array([5, 5])}, self.protocol, []
        )
        self.assertEqual(result["metrics"]["unknown"]["Edit"], 0)
        self.assertEqual(result["metrics"]["unknown"]["F1@50"], 0)

    def test_empty_segment_metrics_are_finite(self):
        ignored = np.array([IGNORED_LABEL, IGNORED_LABEL])
        valid = np.array([8, 8])
        for predicted, expected, counts, edit in [
            (ignored, ignored, (0, 0, 0), 100),
            (valid, ignored, (0, 1, 0), 0),
            (ignored, valid, (0, 0, 1), 0),
        ]:
            self.assertEqual(
                segment_counts(predicted, expected, 0.5, (IGNORED_LABEL,)), counts
            )
            self.assertEqual(edit_score(predicted, expected, (IGNORED_LABEL,)), edit)

    def test_fold_summary_is_an_unweighted_mean_and_requires_metric_fields(self):
        first = evaluate(
            {"video": np.array([1])}, {"video": np.array([5])}, self.protocol, []
        )
        second = evaluate(
            {"video": np.array([2] * 9)},
            {"video": np.array([5] * 9)},
            self.protocol,
            [],
        )
        self.assertEqual(
            summarize_folds([first, second])["metrics"]["known"]["MoF"], 50
        )
        with self.assertRaises(ValueError):
            summarize_folds([{"metrics": first["metrics"]}])
        with self.assertRaises(ValueError):
            summarize_folds(
                [
                    {
                        "skip_empty_targets": False,
                        "metrics": {"known": first["metrics"]["known"]},
                    }
                ]
            )
        with self.assertRaises(ValueError):
            summarize_folds([])

    def test_missing_and_wrong_shape_inputs_fail(self):
        with self.assertRaises(FileNotFoundError):
            self.testing.features("missing.txt")
        np.save(self.testing.directory / "features/test.npy", np.zeros((3, 24)))
        with self.assertRaises(ValueError):
            self.testing.features("test.txt")

    def test_cli_overrides_and_unknown_flags(self):
        config_path = self.root / "config.yaml"
        import yaml

        config_path.write_text(yaml.safe_dump(self.config.to_dict()))
        with (
            patch("actiondiscovery.cli.train") as trainer,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    "--mode",
                    "train",
                    "--config",
                    str(config_path),
                    "--data-root",
                    str(self.root),
                    "--protocol-root",
                    str(self.root),
                    "--output-dir",
                    str(self.root / "cli"),
                    "--epochs",
                    "2",
                ]
            )
            self.assertEqual(trainer.call_args.args[0].training.epochs, 2)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            main(
                [
                    "--mode",
                    "train",
                    "--config",
                    str(config_path),
                    "--data-root",
                    str(self.root),
                    "--output-dir",
                    str(self.root / "other"),
                    "--unknown-option",
                ]
            )

    def test_end_to_end_artifacts_and_annotation_independence(self):
        output = self.root / "train_run"
        output.mkdir()
        checkpoint = train(self.config, self.training, torch.device("cpu"), output)
        evaluation = self.root / "evaluation"
        evaluation.mkdir()
        result = evaluate_checkpoint(
            self.config,
            self.protocol,
            self.training,
            self.testing,
            checkpoint,
            torch.device("cpu"),
            evaluation,
        )
        self.assertTrue((evaluation / "metrics.json").exists())
        self.assertTrue((evaluation / "discovery.pt").exists())
        self.assertIn("estimated_unknown_classes", result)
        cached_output = self.root / "cached"
        cached_output.mkdir()
        with patch(
            "actiondiscovery.engine.fit_discovery",
            side_effect=AssertionError("must reuse fitted state"),
        ):
            cached = evaluate_checkpoint(
                self.config,
                self.protocol,
                self.training,
                self.testing,
                checkpoint,
                torch.device("cpu"),
                cached_output,
                evaluation / "discovery.pt",
            )
        self.assertEqual(result, cached)
        self.config.seed = 1
        with self.assertRaises(ValueError):
            evaluate_checkpoint(
                self.config,
                self.protocol,
                self.training,
                self.testing,
                checkpoint,
                torch.device("cpu"),
                cached_output,
                evaluation / "discovery.pt",
            )
        self.config.seed = 0
        model = load_model(checkpoint, self.config, self.protocol, torch.device("cpu"))
        artifact = torch.load(evaluation / "discovery.pt", weights_only=True)
        state = DiscoveryState(artifact["centroids"].numpy(), artifact["bic"])
        before = predict(model, state, self.testing, self.config, torch.device("cpu"))
        (self.testing.directory / "groundTruth/test.txt").unlink()
        after = predict(model, state, self.testing, self.config, torch.device("cpu"))
        np.testing.assert_array_equal(before["test.txt"], after["test.txt"])
        self.config.model.hidden_dim = 9
        with self.assertRaises(ValueError):
            load_model(checkpoint, self.config, self.protocol, torch.device("cpu"))

    def test_genuine_gmm_is_bounded_by_available_samples(self):
        state = DiscoveryState.fit(
            [np.zeros(8), np.ones(8)], 8, DiscoveryConfig(max_components=20), 0
        )
        self.assertTrue(1 <= len(state.centroids) <= 2)
        self.assertTrue(all(np.isfinite(score) for score in state.bic.values()))

    def test_time_major_features_are_transposed_before_sampling(self):
        self.config.feature_layout = "TD"
        self.config.sampling_rate = 2
        data = np.arange(96, dtype=np.float32).reshape(24, 4)
        np.save(self.testing.directory / "features/test.npy", data)
        features, labels = self.testing.training_sample("test.txt")
        np.testing.assert_array_equal(features.numpy(), data.T[:, ::2])
        self.assertEqual(len(labels), 12)

    def test_dataset_rejects_duplicate_and_unsafe_video_names(self):
        manifest = self.protocol.directory / "test.split1.bundle"
        for content in ["test.txt\ntest.txt\n", "../test.txt\n"]:
            manifest.write_text(content)
            with self.assertRaises(ValueError):
                FeatureDataset(self.config, self.protocol, self.root, 1, "test")


if __name__ == "__main__":
    unittest.main()
