"""Config-threading pin: every kwarg the CD runner expects must be a
parameter on `train_fast_for_budget`. If this fails, some config plumbing
got dropped on the floor between exp24 and the runner."""
import importlib.util
import inspect
from pathlib import Path


def _load_runner_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "experiments" / "23_fast_path" / "runner_fast_path.py"
    )
    spec = importlib.util.spec_from_file_location("runner_fast_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REQUIRED_CD_KEYS = {
    "criticality_distill_enabled",
    "criticality_distill_weight",
    "criticality_distill_budget_frac",
    "criticality_distill_critical_value",
    "criticality_distill_trace_half_life_steps",
    "criticality_distill_trace_ttl_steps",
    "criticality_distill_horizon_H",
    "criticality_distill_event_frac",
    "criticality_distill_seat_refresh_interval",
    "criticality_distill_min_weighted_events_per_layer",
    "criticality_distill_uniform_pressure",
    "criticality_distill_score_permute_before_topk",
    "criticality_distill_fixed_random_seats",
    "criticality_distill_num_layers",
    "criticality_distill_dim",
}
REQUIRED_LM_HEAD_KEYS = {
    "lm_head_backward_mode",
    "lm_head_emit_entropy",
}
REQUIRED_RARE_BUCKET_KEYS = {
    "rare_bucket_ce_enabled",
    "rare_bucket_ce_num_buckets",
    "rare_bucket_ce_token_frequencies",
    "rare_bucket_ce_eval_tokens",
    "rare_bucket_ce_eval_num_tokens",
}


def test_cd_kwargs_all_present_on_train_fast_for_budget():
    mod = _load_runner_module()
    sig = inspect.signature(mod.train_fast_for_budget)
    missing = [k for k in REQUIRED_CD_KEYS if k not in sig.parameters]
    assert not missing, f"missing CD kwargs: {missing}"


def test_lm_head_flag_kwargs_all_present_on_train_fast_for_budget():
    mod = _load_runner_module()
    sig = inspect.signature(mod.train_fast_for_budget)
    missing = [k for k in REQUIRED_LM_HEAD_KEYS if k not in sig.parameters]
    assert not missing, f"missing LM-head flag kwargs: {missing}"


def test_lm_head_emit_entropy_is_orthogonal_to_mode_name():
    mod = _load_runner_module()
    sig = inspect.signature(mod.train_fast_for_budget)
    # Defaults confirm no coupling: lm_head_backward_mode is a string default,
    # lm_head_emit_entropy is a bool default (False). They are independent
    # parameters — not a tuple, not derived from the same enum.
    assert sig.parameters["lm_head_backward_mode"].default == "fused"
    assert sig.parameters["lm_head_emit_entropy"].default is False


def test_rare_bucket_ce_kwargs_all_present_on_train_fast_for_budget():
    mod = _load_runner_module()
    sig = inspect.signature(mod.train_fast_for_budget)
    missing = [k for k in REQUIRED_RARE_BUCKET_KEYS if k not in sig.parameters]
    assert not missing, f"missing rare-bucket CE kwargs: {missing}"
