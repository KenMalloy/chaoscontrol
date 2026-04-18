from chaoscontrol.eval_stream.types import DocRecord, ChunkRecord, RunConfig


def test_docrecord_fields():
    rec = DocRecord(doc_id=0, tokens=[1, 2, 3], raw_bytes=10)
    assert rec.doc_id == 0
    assert len(rec.tokens) == 3


def test_runconfig_defaults():
    cfg = RunConfig()
    assert cfg.chunk_size == 256
    assert cfg.steps_per_chunk == 1
    assert cfg.adapt_set == "none"
    assert cfg.persistence_mode == "reset"
    assert cfg.delta_scale == 1.0
    assert cfg.log_a_shift == 0.0
    assert cfg.persistent_muon_moments is False
