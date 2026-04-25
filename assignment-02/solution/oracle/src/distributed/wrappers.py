"""
Reusable wrappers for:
  - per-parameter DDP (DDPWrapper)
  - flat-bucket DDP with comm/compute overlap (DDPBucketedWrapper)
  - ZeRO stage-1 sharded optimizer (ShardedOptimizer)

These are the module-level interfaces used by `tests/adapters.py`. The
per-rank training scripts in `src/distributed/ddp.py` and `src/distributed/zero1.py`
were written first and stay in place as benchmarks; the classes here are the
reusable equivalents.
"""

from __future__ import annotations

from typing import Iterable, Type

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# -----------------------------------------------------------------------------
# Per-parameter DDP
# -----------------------------------------------------------------------------


class DDPWrapper(torch.nn.Module):
    """
    Individual-parameter DDP.

    On __init__, every parameter is broadcast from rank 0 so all ranks start
    from the same weights. During backward, a post-accumulate-grad hook
    launches an async all-reduce (AVG) for each parameter as soon as its
    gradient is ready. `finish_gradient_synchronization()` waits on all
    outstanding handles, which must be called before optimizer.step().
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._handles: list[dist.Work] = []

        if dist.is_available() and dist.is_initialized():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

            for p in self.module.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, p: torch.nn.Parameter):
        if p.grad is None:
            return
        handle = dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
        self._handles.append(handle)

    def finish_gradient_synchronization(self):
        for h in self._handles:
            h.wait()
        self._handles.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# -----------------------------------------------------------------------------
# Bucketed DDP (flat-bucket overlap)
# -----------------------------------------------------------------------------


class DDPBucketedWrapper(torch.nn.Module):
    """
    Flat-bucket DDP. Parameters are grouped into buckets sized up to
    `bucket_size_mb` megabytes. When every parameter in a bucket has its
    gradient ready, the bucket's gradients are flattened into a single tensor
    and all-reduced asynchronously. `finish_gradient_synchronization()` waits
    on all outstanding bucket handles, then unflattens the synced gradients
    back into each parameter.

    Buckets are formed by iterating parameters in reverse order (the order
    gradients become available in backward). Parameters with `requires_grad=False`
    are skipped entirely.
    """

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        # Initial broadcast so every rank starts from rank 0's weights.
        if dist.is_available() and dist.is_initialized():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

        # Group params into buckets (reversed = backward order).
        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        bucket_size_bytes = bucket_size_mb * 1024 * 1024

        self._buckets: list[list[torch.nn.Parameter]] = []
        curr_bucket: list[torch.nn.Parameter] = []
        curr_size = 0.0
        for p in reversed(trainable_params):
            p_size = p.numel() * p.element_size()
            if curr_bucket and curr_size + p_size > bucket_size_bytes:
                self._buckets.append(curr_bucket)
                curr_bucket = [p]
                curr_size = p_size
            else:
                curr_bucket.append(p)
                curr_size += p_size
        if curr_bucket:
            self._buckets.append(curr_bucket)

        # Param → bucket-index lookup.
        self._param_to_bucket: dict[torch.nn.Parameter, int] = {}
        for b_idx, bucket in enumerate(self._buckets):
            for p in bucket:
                self._param_to_bucket[p] = b_idx

        # Per-bucket ready-counter, pending async handle, and flattened tensor.
        self._ready_counts: list[int] = [0] * len(self._buckets)
        self._pending: list[
            tuple[dist.Work, torch.Tensor, list[torch.nn.Parameter]] | None
        ] = [None] * len(self._buckets)

        # Register hooks.
        if dist.is_available() and dist.is_initialized():
            for p in trainable_params:
                p.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, p: torch.nn.Parameter):
        if p.grad is None:
            return
        b_idx = self._param_to_bucket[p]
        self._ready_counts[b_idx] += 1
        if self._ready_counts[b_idx] == len(self._buckets[b_idx]):
            bucket_params = self._buckets[b_idx]
            grads = [bp.grad for bp in bucket_params]
            flat = _flatten_dense_tensors(grads)
            handle = dist.all_reduce(flat, op=dist.ReduceOp.AVG, async_op=True)
            self._pending[b_idx] = (handle, flat, bucket_params)

    def finish_gradient_synchronization(self):
        for b_idx, pending in enumerate(self._pending):
            if pending is None:
                continue
            handle, flat, bucket_params = pending
            handle.wait()
            grads = [bp.grad for bp in bucket_params]
            unflat = _unflatten_dense_tensors(flat, grads)
            for bp, new_grad in zip(bucket_params, unflat):
                bp.grad = new_grad
        # Reset for next step.
        self._ready_counts = [0] * len(self._buckets)
        self._pending = [None] * len(self._buckets)

    def on_train_batch_start(self):
        """Reset per-step bucket state. Called before each batch's forward."""
        self._ready_counts = [0] * len(self._buckets)
        self._pending = [None] * len(self._buckets)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# -----------------------------------------------------------------------------
# ZeRO-1 sharded optimizer
# -----------------------------------------------------------------------------


class ShardedOptimizer(torch.optim.Optimizer):
    """
    ZeRO stage-1 sharded optimizer.

    Each parameter is assigned to one rank (round-robin by parameter index).
    Every rank instantiates the underlying optimizer over ALL parameters (so
    `state_dict` / `add_param_group` stay compatible), but during `step()`
    only the locally-owned parameters are updated; then every parameter is
    broadcast from its owning rank so all ranks end up with the same weights.

    Gradient all-reduce is NOT done here — callers are expected to sync
    gradients before calling step() (e.g. via a DDP wrapper).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        param_list = [p for p in params]

        self._world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        self._rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )

        # Round-robin assignment: param i is owned by rank (i % world_size).
        self._owner_of: dict[torch.nn.Parameter, int] = {}
        self._local_params: list[torch.nn.Parameter] = []
        for i, p in enumerate(param_list):
            owner = i % self._world_size
            self._owner_of[p] = owner
            if owner == self._rank:
                self._local_params.append(p)

        # Inner optimizer on LOCAL params only. Falls back to a zero-param
        # optimizer if this rank owns nothing (shouldn't happen with rr).
        if self._local_params:
            self._inner = optimizer_cls(self._local_params, **kwargs)
        else:
            self._inner = None

        # Store everything for param_groups compatibility.
        self._all_params = param_list
        # Call Optimizer.__init__ with all params so external code iterating
        # param_groups sees the full set.
        super().__init__(param_list, {**kwargs})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._inner is not None:
            self._inner.step()

        if dist.is_available() and dist.is_initialized() and self._world_size > 1:
            for p in self._all_params:
                owner = self._owner_of[p]
                dist.broadcast(p.data, src=owner)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        # Zero ALL grads — each rank runs forward/backward on its own local
        # data shard and must clear every grad between steps.
        super().zero_grad(set_to_none=set_to_none)
