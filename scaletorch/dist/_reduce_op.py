"""Shared reduce-op string-to-enum mapping."""

from __future__ import annotations

from torch import distributed as torch_dist

_REDUCE_OP_MAPPINGS = {
    'sum': torch_dist.ReduceOp.SUM,
    'product': torch_dist.ReduceOp.PRODUCT,
    'min': torch_dist.ReduceOp.MIN,
    'max': torch_dist.ReduceOp.MAX,
    'band': torch_dist.ReduceOp.BAND,
    'bor': torch_dist.ReduceOp.BOR,
    'bxor': torch_dist.ReduceOp.BXOR,
}


def _get_reduce_op(name: str) -> torch_dist.ReduceOp:
    """Converts a string operation name to a ``torch.distributed.ReduceOp``."""
    name_lower = name.lower()
    if name_lower not in _REDUCE_OP_MAPPINGS:
        raise ValueError(
            f'reduce op should be one of {list(_REDUCE_OP_MAPPINGS.keys())}, '
            f'but got {name}')
    return _REDUCE_OP_MAPPINGS[name_lower]
