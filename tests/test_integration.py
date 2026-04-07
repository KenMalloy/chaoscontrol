"""Integration smoke test: all new features enabled simultaneously."""
import unittest
import torch
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.training import train_chaoscontrol_for_budget
from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
from chaoscontrol.data import build_lm_starts, choose_eval_starts


class TestFullStackIntegration(unittest.TestCase):
    """Smoke test: 3 training steps with every feature enabled."""

    def test_full_stack_trains_and_evals(self):
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=32, num_layers=2, ff_mult=2,
            a_mode="diag",
            rich_b_mode="nn", rich_b_bottleneck=16,
            outer_model_dim=16, outer_model_type="multislot",
            outer_max_slots=8, outer_compress_ratio=2,
            wernicke_enabled=True, wernicke_k_max=4, wernicke_window=4,
            wernicke_router="moe",
            semantic_tier_bases=4,
            typed_storage=True,
            typed_consolidation=True,
            compression_consequence=True,
            compression_selection="survival",
        )
        device = torch.device("cpu")

        # Fake tokens
        tokens = torch.randint(0, 256, (2000,))
        train_starts = build_lm_starts(1800, seq_len=32, stride=16)
        eval_starts = choose_eval_starts(
            build_lm_starts(200, seq_len=32, stride=16),
            batch_size=2, eval_batches=2, seed=42,
        )

        # Train with all features
        result = train_chaoscontrol_for_budget(
            model,
            train_tokens=tokens[:1800],
            train_starts=train_starts,
            seq_len=32,
            batch_size=2,
            device=device,
            param_dtype=torch.float32,
            budget_seconds=5.0,  # just a few steps
            base_lr=1e-3,
            weight_decay=0.01,
            grad_clip_norm=1.0,
            seed=42,
            metabolic_gate=True,
            metabolic_k=2,
            metabolic_threshold=0.05,  # low threshold so gate fires
            metabolic_mode="mcts",
            mcts_horizon=2,
            mcts_ucb_c=1.0,
            consolidation_write="full_sequence",
            latent_persistence=True,
            cfr_enabled=True,
        )

        assert result["steps"] > 0, "Should complete at least 1 step"
        assert "history" in result
        assert "spectral_snapshots" in result
        # Check no NaN in loss history
        for h in result["history"]:
            assert h["loss"] == h["loss"], f"NaN loss at step {h['step']}"

        # Eval with warmup
        eval_result = evaluate_chaoscontrol_bpb(
            model,
            tokens=tokens[1800:],
            eval_starts=eval_starts,
            batch_size=2,
            seq_len=32,
            device=device,
            metabolic_gate=True,
            metabolic_k=2,
            warmup=True,
        )
        assert "bpb" in eval_result
        assert eval_result["bpb"] == eval_result["bpb"], "NaN in eval bpb"
        assert eval_result["bpb"] > 0


    def test_step_interface_matches_forward_with_features(self):
        """step() should work on a model with Wernicke and memory."""
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=32, num_layers=2,
            a_mode="diag",
            outer_model_dim=16, outer_model_type="multislot",
            wernicke_enabled=True, wernicke_k_max=4, wernicke_window=4,
        )
        ids = torch.randint(0, 256, (2, 8))

        # Forward
        out = model(ids)
        assert out["logits"].shape == (2, 8, 256)

        # Step (skips Wernicke/memory by design — just tests it doesn't crash)
        state = model.init_state(2)
        for t in range(8):
            logits, hidden, state = model.step(ids[:, t:t+1], state)
        assert logits.shape == (2, 256)


    def test_eval_respects_metabolic_mode(self):
        """evaluate_chaoscontrol_bpb must dispatch to the correct gate
        function based on metabolic_mode and always produce bpb_gated."""
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=32, num_layers=2, ff_mult=2,
            a_mode="diag",
            outer_model_dim=16, outer_model_type="multislot",
            outer_max_slots=8, outer_compress_ratio=2,
        )
        device = torch.device("cpu")
        tokens = torch.randint(0, 256, (500,))
        eval_starts = choose_eval_starts(
            build_lm_starts(500, seq_len=32, stride=16),
            batch_size=2, eval_batches=2, seed=42,
        )

        results = {}
        for mode in ("fork", "monte_carlo", "mcts"):
            res = evaluate_chaoscontrol_bpb(
                model,
                tokens=tokens,
                eval_starts=eval_starts,
                batch_size=2,
                seq_len=32,
                device=device,
                metabolic_gate=True,
                metabolic_k=2,
                metabolic_mode=mode,
            )
            self.assertIn("bpb_gated", res, f"bpb_gated missing for mode={mode}")
            self.assertGreater(res["bpb_gated"], 0, f"bpb_gated <= 0 for mode={mode}")
            self.assertTrue(
                res["bpb_gated"] == res["bpb_gated"],
                f"NaN bpb_gated for mode={mode}",
            )
            results[mode] = res["bpb_gated"]

        # Different modes should generally give different gated bpb values
        # (not a hard assert — just sanity-check that at least two differ)
        vals = list(results.values())
        self.assertFalse(
            vals[0] == vals[1] == vals[2],
            "All three modes returned identical bpb_gated — dispatch may be broken",
        )


    def test_cfr_accumulates_meaningful_regret(self):
        """After training with CFR enabled, the regret table should have
        non-zero entries and at least one bucket with a non-uniform strategy."""
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=32, num_layers=2, ff_mult=2,
            a_mode="diag",
            rich_b_mode="nn", rich_b_bottleneck=16,
            outer_model_dim=16, outer_model_type="multislot",
            outer_max_slots=8, outer_compress_ratio=2,
            wernicke_enabled=True, wernicke_k_max=4, wernicke_window=4,
            wernicke_router="moe",
            semantic_tier_bases=4,
            typed_storage=True,
            typed_consolidation=True,
            compression_consequence=True,
            compression_selection="survival",
        )
        device = torch.device("cpu")

        tokens = torch.randint(0, 256, (2000,))
        train_starts = build_lm_starts(1800, seq_len=32, stride=16)

        result = train_chaoscontrol_for_budget(
            model,
            train_tokens=tokens[:1800],
            train_starts=train_starts,
            seq_len=32,
            batch_size=2,
            device=device,
            param_dtype=torch.float32,
            budget_seconds=8.0,
            base_lr=1e-3,
            weight_decay=0.01,
            grad_clip_norm=1.0,
            seed=42,
            metabolic_gate=True,
            metabolic_k=2,
            metabolic_threshold=0.05,
            metabolic_mode="mcts",
            mcts_horizon=2,
            mcts_ucb_c=1.0,
            consolidation_write="full_sequence",
            latent_persistence=True,
            cfr_enabled=True,
        )

        assert result["steps"] > 0, "Should complete at least 1 step"
        regret_table = result.get("regret_table")
        assert regret_table is not None, "regret_table should be returned when cfr_enabled"

        # Check that cumulative regret is not all zeros
        total_regret = regret_table.cumulative_regret.abs().sum().item()
        assert total_regret > 0, (
            "CFR regret table should have non-zero entries after training "
            f"(got total abs regret = {total_regret})"
        )

        # Check at least one bucket has a non-uniform strategy
        found_non_uniform = False
        for b in range(regret_table.n_buckets):
            strategy = regret_table.get_strategy(b)
            uniform_val = 1.0 / regret_table.n_actions
            if not all(abs(strategy[a].item() - uniform_val) < 1e-5 for a in range(regret_table.n_actions)):
                found_non_uniform = True
                break
        assert found_non_uniform, (
            "At least one bucket should have a non-uniform strategy after training"
        )


if __name__ == "__main__":
    unittest.main()
