"""Distributed communication primitives: collectives, P2P ops, and utilities.

Sub-modules
-----------
- ``collective_ops`` – tensor-level collectives (all_reduce, all_gather, …)
- ``p2p_ops``        – point-to-point send / recv helpers
- ``object_ops``     – picklable-object broadcast / gather
- ``gather_utils``   – distributed result collection (CPU tmpdir & GPU paths)
- ``utils``          – init, rank queries, device helpers
"""

from scaletorch.dist._reduce_op import _get_reduce_op  # noqa: F401

from scaletorch.dist.collective_ops import (  # noqa: F401
    all_gather,
    all_reduce,
    all_reduce_dict,
    all_reduce_params,
    all_to_all,
    broadcast,
    gather,
    reduce,
    reduce_scatter,
    scatter,
    sync_random_seed,
)

from scaletorch.dist.p2p_ops import (  # noqa: F401
    P2POp,
    batch_isend_irecv,
)

from scaletorch.dist.object_ops import (  # noqa: F401
    all_gather_object,
    broadcast_object_list,
    gather_object,
)

from scaletorch.dist.gather_utils import (  # noqa: F401
    collect_results,
    collect_results_cpu,
    collect_results_gpu,
)

from scaletorch.dist.utils import (  # noqa: F401
    barrier,
    cleanup_dist,
    destroy_group,
    get_rank,
    get_world_size,
    init_dist,
    is_distributed,
    new_group,
)

__all__ = [
    "_get_reduce_op",
    "all_gather",
    "all_gather_object",
    "all_reduce",
    "all_reduce_dict",
    "all_reduce_params",
    "all_to_all",
    "barrier",
    "batch_isend_irecv",
    "broadcast",
    "broadcast_object_list",
    "cleanup_dist",
    "collect_results",
    "collect_results_cpu",
    "collect_results_gpu",
    "destroy_group",
    "gather",
    "gather_object",
    "get_rank",
    "get_world_size",
    "init_dist",
    "is_distributed",
    "new_group",
    "P2POp",
    "reduce",
    "reduce_scatter",
    "scatter",
    "sync_random_seed",
]
