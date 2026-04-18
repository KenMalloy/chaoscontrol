from chaoscontrol.eval_stream.types import DocRecord, ChunkRecord, RunConfig


def test_docrecord_fields():
    rec = DocRecord(doc_id=0, tokens=[1, 2, 3], raw_bytes=10)
    assert rec.doc_id == 0
    assert len(rec.tokens) == 3
    assert rec.raw_bytes == 10


def test_chunkrecord_fields():
    """ChunkRecord is currently unused in the driver (per-doc logging only),
    but lives in __all__ and is kept for per-chunk logging work later. This
    test pins the field set so the shape doesn't drift silently.
    """
    rec = ChunkRecord(
        doc_id=7, chunk_idx=3, tokens=[10, 11, 12],
        loss_before_adapt=1.234, loss_after_adapt=1.100,
    )
    assert rec.doc_id == 7
    assert rec.chunk_idx == 3
    assert rec.tokens == [10, 11, 12]
    assert rec.loss_before_adapt == 1.234
    assert rec.loss_after_adapt == 1.100
    # loss_after_adapt allows None when no weight TTT was applied.
    rec_no_adapt = ChunkRecord(
        doc_id=0, chunk_idx=0, tokens=[1], loss_before_adapt=0.5,
        loss_after_adapt=None,
    )
    assert rec_no_adapt.loss_after_adapt is None


def test_runconfig_defaults():
    """All RunConfig defaults — load-bearing Param Golf contract values
    (warmup_steps=20, budget_seconds=600.0, max_docs=50_000) must not drift
    silently. A refactor that flips any of these breaks the eval contract."""
    cfg = RunConfig()
    # Axis 1
    assert cfg.adapt_set == "none"
    # Axis 2
    assert cfg.persistence_mode == "reset"
    # Axis 3
    assert cfg.delta_scale == 1.0
    assert cfg.log_a_shift == 0.0
    # Schedule
    assert cfg.chunk_size == 256
    assert cfg.steps_per_chunk == 1
    assert cfg.eval_lr == 0.064
    assert cfg.persistent_muon_moments is False
    assert cfg.warmup_steps == 20
    # Run
    assert cfg.seed == 0
    assert cfg.max_docs == 50_000
    assert cfg.budget_seconds == 600.0
    assert cfg.checkpoint_path == ""
    assert cfg.output_path == ""


def test_runconfig_override_fields():
    """RunConfig accepts keyword overrides on all fields."""
    cfg = RunConfig(
        adapt_set="log_a",
        persistence_mode="carry_state",
        delta_scale=2.0,
        log_a_shift=-0.5,
        chunk_size=64,
        steps_per_chunk=5,
        eval_lr=0.032,
        persistent_muon_moments=True,
        warmup_steps=10,
        seed=42,
        max_docs=100,
        budget_seconds=60.0,
        checkpoint_path="/tmp/ckpt.pt",
        output_path="/tmp/out.jsonl",
    )
    assert cfg.adapt_set == "log_a"
    assert cfg.persistence_mode == "carry_state"
    assert cfg.delta_scale == 2.0
    assert cfg.log_a_shift == -0.5
    assert cfg.chunk_size == 64
    assert cfg.steps_per_chunk == 5
    assert cfg.eval_lr == 0.032
    assert cfg.persistent_muon_moments is True
    assert cfg.warmup_steps == 10
    assert cfg.seed == 42
    assert cfg.max_docs == 100
    assert cfg.budget_seconds == 60.0
    assert cfg.checkpoint_path == "/tmp/ckpt.pt"
    assert cfg.output_path == "/tmp/out.jsonl"
