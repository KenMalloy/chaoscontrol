"""CRCT distributed coordination layer.

See docs/plans/2026-04-27-crct-distributed-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as _dist

__all__ = [
    "create_crct_process_groups",
    "WeightMailbox",
    "TeacherResultMailbox",
    "TeacherPayload",
    "Rank3MemoryCoprocessor",
    "rank3_coprocessor_loop",
    "fail_open_controller_term",
]


def create_crct_process_groups(world_size):
    """Create train, CRCT, and sync process groups in canonical order."""

    if int(world_size) < 4:
        return None, None, None
    train_pg = _dist.new_group(ranks=[0, 1, 2])
    crct_pg = _dist.new_group(ranks=[0, 1, 2, 3])
    sync_pg = _dist.new_group(ranks=[0, 3])
    return train_pg, crct_pg, sync_pg


def _flatten_trainable(model) -> torch.Tensor:
    params = [p.detach().reshape(-1) for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.empty(0)
    return torch.cat(params)


class _CompletedWork:
    def __init__(self, completed: bool = True) -> None:
        self._completed = bool(completed)

    def is_completed(self) -> bool:
        return self._completed

    def wait(self) -> None:
        self._completed = True


class WeightMailbox:  # noqa: D101 — fleshed out in Task 3
    def __init__(
        self,
        model,
        sync_pg,
        *,
        sync_interval_steps: int = 50,
        src_rank: int = 0,
        my_rank: int,
        broadcast_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.sync_pg = sync_pg
        self.sync_interval_steps = int(sync_interval_steps)
        self.src_rank = int(src_rank)
        self.my_rank = int(my_rank)
        self._broadcast = broadcast_fn or _dist.broadcast
        self._buffer = _flatten_trainable(model).clone()
        self._inflight: tuple[Any, int] | None = None
        self.snapshots_posted = 0
        self.snapshot_version = 0

    def post_weights(self, global_step: int) -> None:
        if self.my_rank != self.src_rank:
            return
        if self.sync_interval_steps <= 0:
            return
        if int(global_step) % self.sync_interval_steps != 0:
            return
        if self._inflight is not None and not self._inflight[0].is_completed():
            return
        flat = _flatten_trainable(self.model).to(device=self._buffer.device)
        if flat.numel() != self._buffer.numel():
            self._buffer = flat.clone()
        else:
            self._buffer.copy_(flat)
        work = self._broadcast(
            self._buffer,
            src=self.src_rank,
            group=self.sync_pg,
            async_op=True,
        )
        self.snapshots_posted += 1
        self._inflight = (work, self.snapshots_posted)

    def maybe_sync(self, global_step: int) -> None:
        del global_step
        if self.my_rank == self.src_rank:
            return
        if self._inflight is None:
            work = self._broadcast(
                self._buffer,
                src=self.src_rank,
                group=self.sync_pg,
                async_op=True,
            )
            self._inflight = (work, self.snapshot_version + 1)
            return
        work, version = self._inflight
        if not work.is_completed():
            return
        offset = 0
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            n = param.numel()
            param.data.copy_(self._buffer[offset:offset + n].view_as(param))
            offset += n
        self.snapshot_version = int(version)
        self._inflight = None


@dataclass
class TeacherPayload:
    step_id: int
    target: torch.Tensor
    conf: torch.Tensor
    loss_weight: float
    snapshot_version: int = 0


class TeacherResultMailbox:
    def __init__(
        self,
        crct_pg,
        *,
        my_rank: int,
        num_train_ranks: int,
        payload_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        queue_depth: int = 1,
        broadcast_fn: Callable[..., Any] | None = None,
        inline_score_fn: Callable[..., TeacherPayload | None] | None = None,
    ) -> None:
        if int(queue_depth) != 1:
            raise NotImplementedError(
                "queue_depth > 1 not in scope; see design §4"
            )
        self.crct_pg = crct_pg
        self.my_rank = int(my_rank)
        self.num_train_ranks = int(num_train_ranks)
        self.payload_shape = tuple(payload_shape)
        self.dtype = dtype
        self._target_buf = torch.zeros(self.payload_shape, dtype=dtype)
        self._conf_buf = torch.zeros(self.payload_shape[:2], dtype=dtype)
        self._meta_buf = torch.zeros(3, dtype=torch.float32)
        self._inflight: tuple[Any, TeacherPayload | None] | None = None
        self.posts_attempted = 0
        self.posts_dropped = 0
        if crct_pg is None:
            if inline_score_fn is None:
                raise ValueError("fallback requires inline_score_fn")
            self._inline_score = inline_score_fn
            self._broadcast = None
        else:
            self._inline_score = None
            self._broadcast = broadcast_fn or _dist.broadcast

    def post_result(
        self,
        *,
        step_id: int,
        target: torch.Tensor | None,
        conf: torch.Tensor | None,
        loss_weight: float,
        snapshot_version: int = 0,
    ) -> None:
        self.posts_attempted += 1
        if self.crct_pg is None:
            return
        if self._inflight is not None and not self._inflight[0].is_completed():
            self.posts_dropped += 1
            return
        if target is not None:
            self._target_buf.copy_(target.to(dtype=self.dtype))
        if conf is not None:
            self._conf_buf.copy_(conf.to(dtype=self.dtype))
        self._meta_buf[0] = float(loss_weight)
        self._meta_buf[1] = float(step_id)
        self._meta_buf[2] = float(snapshot_version)
        assert self._broadcast is not None
        work = self._broadcast(
            self._target_buf,
            src=3,
            group=self.crct_pg,
            async_op=True,
        )
        payload = TeacherPayload(
            step_id=int(step_id),
            target=self._target_buf,
            conf=self._conf_buf,
            loss_weight=float(loss_weight),
            snapshot_version=int(snapshot_version),
        )
        self._inflight = (work, payload)

    def try_get(self, step: int) -> TeacherPayload | None:
        del step
        if self.my_rank == 3 or self.crct_pg is None:
            return None
        if self._inflight is None:
            assert self._broadcast is not None
            work = self._broadcast(
                self._target_buf,
                src=3,
                group=self.crct_pg,
                async_op=True,
            )
            self._inflight = (work, None)
            return None
        work, payload = self._inflight
        if not work.is_completed():
            return None
        if payload is None:
            payload = TeacherPayload(
                step_id=int(self._meta_buf[1].item()),
                target=self._target_buf.clone(),
                conf=self._conf_buf.clone(),
                loss_weight=float(self._meta_buf[0].item()),
                snapshot_version=int(self._meta_buf[2].item()),
            )
        self._inflight = None
        return payload

    def try_get_with_input(
        self,
        *,
        step: int,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> TeacherPayload | None:
        if self.crct_pg is None:
            assert self._inline_score is not None
            return self._inline_score(input_ids, valid_mask, step)
        raise RuntimeError(
            "try_get_with_input is only valid for world_size < 4 fallback"
        )


class Rank3MemoryCoprocessor:
    def __init__(
        self,
        *,
        model_copy,
        weight_mailbox: WeightMailbox,
        result_mailbox: TeacherResultMailbox,
        score_fn: Callable[..., Any],
        episodic_drain_fn: Callable[[], Any],
        controller_tick_fn: Callable[[], Any],
        crct_pg,
        sync_pg,
        high_stream: Any | None = None,
        low_stream: Any | None = None,
        gather_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model_copy
        self.weight_mailbox = weight_mailbox
        self.result_mailbox = result_mailbox
        self.score_fn = score_fn
        self.episodic_drain_fn = episodic_drain_fn
        self.controller_tick_fn = controller_tick_fn
        self.crct_pg = crct_pg
        self.sync_pg = sync_pg
        self.high_stream = high_stream
        self.low_stream = low_stream
        self._gather = gather_fn or _dist.gather
        self._latest_batch = None
        self._inflight_score: Any | None = None
        self._pending_payload: TeacherPayload | None = None
        self.metrics = {
            "teacher_drops": 0,
            "scores_run": 0,
            "scores_reaped": 0,
        }

    def step_once(self, global_step: int) -> None:
        if (
            self._inflight_score is not None
            and self._inflight_score.is_completed()
        ):
            payload = self._pending_payload
            self.result_mailbox.post_result(
                step_id=payload.step_id if payload is not None else int(global_step),
                target=payload.target if payload is not None else None,
                conf=payload.conf if payload is not None else None,
                loss_weight=payload.loss_weight if payload is not None else 1.0,
                snapshot_version=(
                    payload.snapshot_version if payload is not None else 0
                ),
            )
            self._pending_payload = None
            self._inflight_score = None
            self.metrics["scores_reaped"] += 1

        batch = self._gather(group=self.crct_pg)
        self._latest_batch = batch
        self.episodic_drain_fn()
        self.controller_tick_fn()
        self.weight_mailbox.maybe_sync(global_step)

        if self._inflight_score is None and self._latest_batch is not None:
            result = self.score_fn(self.model, self._latest_batch)
            if isinstance(result, TeacherPayload):
                self._pending_payload = result
                self._inflight_score = _CompletedWork(True)
            elif isinstance(result, dict):
                self._pending_payload = TeacherPayload(
                    step_id=int(result.get("step_id", global_step)),
                    target=result["target"],
                    conf=result["conf"],
                    loss_weight=float(result.get("loss_weight", 1.0)),
                    snapshot_version=int(result.get("snapshot_version", 0)),
                )
                self._inflight_score = _CompletedWork(True)
            elif result is not None and hasattr(result, "is_completed"):
                self._inflight_score = result
            else:
                self._inflight_score = _CompletedWork(True)
            self.metrics["scores_run"] += 1
        elif self._inflight_score is not None:
            self.metrics["teacher_drops"] += 1


def rank3_coprocessor_loop(coprocessor, *, stop_flag, step_iter) -> None:
    """Drive a rank-3 coprocessor until ``stop_flag`` trips."""

    if coprocessor is None:
        return
    for step in step_iter:
        if stop_flag():
            break
        coprocessor.step_once(step)


def fail_open_controller_term(
    *,
    payload,
    controller_logits: torch.Tensor,
    lm_loss: torch.Tensor,
    lambda_ctrl: float,
    controller_loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compose LM loss with controller loss while preserving DDP grad sets."""

    if payload is None:
        return lm_loss + 0.0 * controller_logits.sum()
    return lm_loss + float(lambda_ctrl) * controller_loss_fn(
        payload,
        controller_logits,
    )
