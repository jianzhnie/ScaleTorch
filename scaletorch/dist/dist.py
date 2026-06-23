"""Backward-compatible re-export shim.

All public symbols that were previously importable from ``scaletorch.dist.dist``
are re-exported here so that any stale ``from scaletorch.dist.dist import …``
(e.g. in third-party code or cached bytecode) keeps working.

New code should import from ``scaletorch.dist`` (the package) directly.
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
from scaletorch.dist.gather_utils import (  # noqa: F401
    collect_results,
    collect_results_cpu,
    collect_results_gpu,
)
from scaletorch.dist.object_ops import (  # noqa: F401
    all_gather_object,
    broadcast_object_list,
    gather_object,
)
from scaletorch.dist.p2p_ops import (  # noqa: F401
    P2POp,
    batch_isend_irecv,
    irecv,
    isend,
)
