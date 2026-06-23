"""Point-to-point communication primitives: isend, irecv, P2POp, batch_isend_irecv."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup


def isend(
    tensor: Tensor,
    dst: int,
    group: ProcessGroup | None = None,
) -> Any:
    """Send *tensor* asynchronously to rank *dst*."""
    return torch_dist.isend(tensor, dst, group=group)


def irecv(
    tensor: Tensor,
    src: int,
    group: ProcessGroup | None = None,
) -> Any:
    """Receive into *tensor* asynchronously from rank *src*."""
    return torch_dist.irecv(tensor, src, group=group)


def P2POp(
    op: Any,
    tensor: Tensor,
    peer: int,
    group: ProcessGroup | None = None,
) -> Any:
    """Create a point-to-point operation descriptor for batch operations."""
    return torch_dist.P2POp(op, tensor, peer, group=group)


def batch_isend_irecv(p2p_ops: list) -> list:
    """Batch multiple point-to-point send/recv operations."""
    return torch_dist.batch_isend_irecv(p2p_ops)
