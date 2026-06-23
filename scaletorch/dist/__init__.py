"""Distributed communication primitives: collectives, P2P ops, and utilities.

Sub-modules
-----------
- ``collective_ops`` – tensor-level collectives (all_reduce, all_gather, …)
- ``p2p_ops``        – point-to-point send / recv helpers
- ``object_ops``     – picklable-object broadcast / gather
- ``gather_utils``   – distributed result collection (CPU tmpdir & GPU paths)
- ``utils``          – init, rank queries, device helpers
"""

# -- shared helper -----------------------------------------------------------
from scaletorch.dist._reduce_op import _get_reduce_op  # noqa: F401

# -- tensor-level collectives ------------------------------------------------
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

# -- point-to-point ----------------------------------------------------------
from scaletorch.dist.p2p_ops import (  # noqa: F401
    P2POp,
    batch_isend_irecv,
    irecv,
    isend,
)

# -- object-level communication ----------------------------------------------
from scaletorch.dist.object_ops import (  # noqa: F401
    all_gather_object,
    broadcast_object_list,
    gather_object,
)

# -- result collection -------------------------------------------------------
from scaletorch.dist.gather_utils import (  # noqa: F401
    collect_results,
    collect_results_cpu,
    collect_results_gpu,
)

# -- environment / init / rank utilities -------------------------------------
from scaletorch.dist.utils import (  # noqa: F401
    barrier,
    cast_data_device,
    cleanup_dist,
    destroy_group,
    get_backend,
    get_comm_device,
    get_data_device,
    get_default_group,
    get_dist_info,
    get_local_group,
    get_local_rank,
    get_local_size,
    get_rank,
    get_world_size,
    infer_launcher,
    init_dist,
    init_local_group,
    is_distributed,
    is_main_process,
    master_only,
    new_group,
)

__all__ = [
    # collectives
    'all_gather',
    'all_reduce',
    'all_reduce_dict',
    'all_reduce_params',
    'all_to_all',
    'broadcast',
    'gather',
    'reduce',
    'reduce_scatter',
    'scatter',
    'sync_random_seed',
    # p2p
    'P2POp',
    'batch_isend_irecv',
    'irecv',
    'isend',
    # object ops
    'all_gather_object',
    'broadcast_object_list',
    'gather_object',
    # gather utils
    'collect_results',
    'collect_results_cpu',
    'collect_results_gpu',
    # reduce-op helper
    '_get_reduce_op',
    # env / init / rank utilities
    'barrier',
    'cast_data_device',
    'cleanup_dist',
    'destroy_group',
    'get_backend',
    'get_comm_device',
    'get_data_device',
    'get_default_group',
    'get_dist_info',
    'get_local_group',
    'get_local_rank',
    'get_local_size',
    'get_rank',
    'get_world_size',
    'infer_launcher',
    'init_dist',
    'init_local_group',
    'is_distributed',
    'is_main_process',
    'master_only',
    'new_group',
]
