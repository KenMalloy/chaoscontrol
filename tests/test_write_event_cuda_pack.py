"""CUDA WriteEvent pack kernel.

The production writer path builds fixed-size WriteEvent structs on GPU, then
copies a raw byte batch into pinned CPU staging for the shm-ring publisher. This
test is hardware-gated: local arm64/CPU runs skip it, while H100 pod builds with
CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=1 exercise the exact wire layout.
"""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext
from chaoscontrol.optim.episodic_writer import (
    fingerprint_tokens,
    tensor_fp16_to_u16_wire,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_pack_write_events_matches_wire_dict(tmp_path):
    if not bool(_ext.write_event_cuda_pack_available()):
        pytest.skip("CUDA WriteEvent pack kernel not compiled into extension")

    event_size = int(_ext.wire_event_sizes()["WriteEvent"])
    out = torch.empty((1, event_size), device="cuda", dtype=torch.uint8)
    input_ids = torch.tensor(
        [[101, 102, 103, 104, 105, 106, 107, 108]],
        device="cuda",
        dtype=torch.long,
    )
    target_ids = torch.tensor(
        [[201, 202, 203, 204, 205, 206, 207, 208]],
        device="cuda",
        dtype=torch.long,
    )
    hidden = torch.arange(8 * 256, device="cuda", dtype=torch.float32).reshape(
        1, 8, 256
    )
    pressure = torch.full((1, 8), 1.5, device="cuda", dtype=torch.float32)
    ce = torch.full((1, 8), 2.25, device="cuda", dtype=torch.float32)
    positions = torch.tensor([[0, 3]], device="cuda", dtype=torch.long)
    candidate_base = torch.tensor([7], device="cuda", dtype=torch.long)

    _ext.pack_write_events_cuda_(
        out,
        input_ids,
        target_ids,
        hidden,
        pressure,
        ce,
        positions,
        candidate_base,
        123,
        2,
        1,
        2,
        4,
        256,
    )
    raw = out.cpu()

    name = f"/cc_test_cuda_pack_{tmp_path.name}"
    try:
        _ext.ShmRingWriteEvent.unlink(name)
    except Exception:
        pass
    ring = _ext.ShmRingWriteEvent.create(name)
    try:
        stats = ring.push_batch_tensor(raw)
        assert dict(stats) == {"pushed": 1, "skipped": 0, "dropped": 0}
        event = ring.pop()
    finally:
        _ext.ShmRingWriteEvent.unlink(name)

    assert event["event_type"] == 1
    assert event["source_rank"] == 2
    assert event["write_bucket"] == 1
    assert event["candidate_id"] == (2 << 56) | 7
    assert event["gpu_step"] == 123
    assert event["key_fp"] == fingerprint_tokens(
        torch.tensor([102, 103], dtype=torch.long)
    )
    assert event["key_rep"] == tensor_fp16_to_u16_wire(
        hidden[0, 3].detach().cpu()
    )
    assert event["value_tok_ids"] == [204, 205, 206, 207]
    assert event["value_anchor_id"] == 204
    assert event["pressure_at_write"] == pytest.approx(1.5)
    assert event["pre_write_ce"] == pytest.approx(2.25)
