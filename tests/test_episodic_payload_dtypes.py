"""Pin the dtype shapes/fields that Tasks 1.4 + 1.5 share."""
from __future__ import annotations

import numpy as np
import pytest

from chaoscontrol.episodic.payload_dtypes import (
    make_query_candidate_dtype,
    make_write_payload_dtype,
)


def test_write_payload_has_expected_fields_and_shapes():
    dt = make_write_payload_dtype(span_length=4, key_rep_dim=8)
    assert dt.names == ("key_fp", "key_rep", "value_tok_ids", "value_anchor_id")
    assert dt["key_fp"] == np.int64
    assert dt["key_rep"].subdtype == (np.dtype(np.float32), (8,))
    assert dt["value_tok_ids"].subdtype == (np.dtype(np.int64), (4,))
    assert dt["value_anchor_id"] == np.int64


def test_query_candidate_has_expected_fields_and_shapes():
    dt = make_query_candidate_dtype(key_rep_dim=8)
    assert dt.names == ("batch_index", "position", "pressure", "residual")
    assert dt["batch_index"] == np.int64
    assert dt["position"] == np.int64
    assert dt["pressure"] == np.float32
    assert dt["residual"].subdtype == (np.dtype(np.float32), (8,))


def test_write_payload_can_round_trip_through_zeros_slot():
    dt = make_write_payload_dtype(span_length=2, key_rep_dim=4)
    slot = np.zeros((), dtype=dt)
    slot["key_fp"] = 12345
    slot["key_rep"] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    slot["value_tok_ids"] = np.array([10, 20], dtype=np.int64)
    slot["value_anchor_id"] = 10
    assert int(slot["key_fp"]) == 12345
    assert (slot["key_rep"] == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (slot["value_tok_ids"] == np.array([10, 20])).all()
    assert int(slot["value_anchor_id"]) == 10


def test_query_candidate_can_round_trip_through_zeros_slot():
    dt = make_query_candidate_dtype(key_rep_dim=4)
    slot = np.zeros((), dtype=dt)
    slot["batch_index"] = 3
    slot["position"] = 17
    slot["pressure"] = 0.5
    slot["residual"] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert int(slot["batch_index"]) == 3
    assert int(slot["position"]) == 17
    assert float(slot["pressure"]) == pytest.approx(0.5)
    assert (slot["residual"] == np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)).all()


def test_dtype_factories_reject_nonpositive_dims():
    with pytest.raises(ValueError):
        make_write_payload_dtype(span_length=0, key_rep_dim=8)
    with pytest.raises(ValueError):
        make_write_payload_dtype(span_length=4, key_rep_dim=0)
    with pytest.raises(ValueError):
        make_query_candidate_dtype(key_rep_dim=0)
