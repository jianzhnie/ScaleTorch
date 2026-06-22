"""Process Group Manager for 4D parallelism (DP, PP, CP, TP)."""

from __future__ import annotations

import contextlib
import os

import torch
from torch.distributed import ProcessGroup

from scaletorch.dist import (
    destroy_group,
    get_rank,
    get_world_size,
    is_distributed,
    new_group,
)


class ProcessGroupManager:
    """
    Manages process groups for different parallelism strategies in distributed training.

    This class creates and manages process groups for Tensor Parallelism (TP), Context Parallelism (CP),
    Pipeline Parallelism (PP), and Data Parallelism (DP). It organizes processes in a 4D grid
    structure and provides easy access to relevant process group information.

    Attributes:
        global_rank (int): Global rank of the current process
        world_size (int): Total number of processes in the world
        local_rank (int): Local rank of the current process
        dp_rank (int): Data parallelism rank
        pp_rank (int): Pipeline parallelism rank
        cp_rank (int): Context parallelism rank
        tp_rank (int): Tensor parallelism rank
    """

    def __init__(
        self, tp_size: int, cp_size: int, pp_size: int, dp_size: int, ep_size: int = 1
    ) -> None:
        """
        Initialize the ProcessGroupManager.

        Args:
            tp_size (int): Size of tensor parallelism dimension
            cp_size (int): Size of context parallelism dimension
            pp_size (int): Size of pipeline parallelism dimension
            dp_size (int): Size of data parallelism dimension
            ep_size (int): Size of expert parallelism dimension (default: 1)

        Raises:
            RuntimeError: If distributed training is not initialized
            ValueError: If world_size doesn't equal tp_size * cp_size * pp_size * dp_size * ep_size
        """
        # Check if distributed training is initialized
        if not is_distributed():
            raise RuntimeError(
                "Distributed training must be initialized before creating ProcessGroupManager"
            )

        # Validate parallelism sizes
        for size, name in [
            (tp_size, "tp_size"),
            (cp_size, "cp_size"),
            (pp_size, "pp_size"),
            (dp_size, "dp_size"),
            (ep_size, "ep_size"),
        ]:
            if size <= 0:
                raise ValueError(f"{name} must be positive, got {size}")

        self.global_rank: int = get_rank()
        self.world_size: int = get_world_size()
        self.local_rank: int = int(
            os.environ.get("LOCAL_RANK", self.global_rank % self.world_size)
        )

        # Validate world size matches the product of all parallelism dimensions
        expected_world_size = tp_size * cp_size * pp_size * dp_size * ep_size
        if self.world_size != expected_world_size:
            raise ValueError(
                f"World size ({self.world_size}) != TP ({tp_size}) * CP ({cp_size}) * PP ({pp_size}) * DP ({dp_size}) * EP ({ep_size}) = {expected_world_size}. "
                f"Please check your distributed training setup and ensure the total number of processes matches the product of all parallelism dimensions."
            )

        # Create 5D grid: [DP, PP, CP, EP, TP]
        self.grid: torch.Tensor = torch.arange(self.world_size).view(
            dp_size, pp_size, cp_size, ep_size, tp_size
        )

        # Compute position via arithmetic (avoids O(world_size) tensor scan)
        remainder = self.global_rank
        self.tp_rank: int = remainder % tp_size
        remainder //= tp_size
        self.ep_rank: int = remainder % ep_size
        remainder //= ep_size
        self.cp_rank: int = remainder % cp_size
        remainder //= cp_size
        self.pp_rank: int = remainder % pp_size
        self.dp_rank: int = remainder // pp_size

        # Create process groups for different parallelism strategies
        self._create_process_groups(tp_size, cp_size, pp_size, dp_size, ep_size)

        # Initialize group IDs and properties
        self._initialize_group_properties()

    def _create_parallel_groups(
        self,
        rank_lists: list[list[int]],
    ) -> tuple[list[ProcessGroup], ProcessGroup | None]:
        """Create process groups and select the one for current rank."""
        groups = []
        for ranks in rank_lists:
            groups.append(new_group(ranks=ranks))
        my_group = None
        for ranks, group in zip(rank_lists, groups, strict=False):
            if self.global_rank in ranks:
                my_group = group
                break
        return groups, my_group

    def _create_process_groups(
        self, tp_size: int, cp_size: int, pp_size: int, dp_size: int, ep_size: int = 1
    ) -> None:
        """Create process groups for different parallelism strategies."""
        # Tensor Parallelism groups: same DP, PP, CP, EP
        tp_rank_lists = [
            self.grid[d, p, c, e, :].tolist()
            for d in range(dp_size)
            for p in range(pp_size)
            for c in range(cp_size)
            for e in range(ep_size)
        ]
        self._tp_groups, self.tp_group = self._create_parallel_groups(tp_rank_lists)

        # Context Parallelism groups: same DP, PP, EP, TP
        cp_rank_lists = [
            self.grid[d, p, :, e, t].tolist()
            for d in range(dp_size)
            for p in range(pp_size)
            for e in range(ep_size)
            for t in range(tp_size)
        ]
        self._cp_groups, self.cp_group = self._create_parallel_groups(cp_rank_lists)

        # Pipeline Parallelism groups: same DP, CP, EP, TP
        pp_rank_lists = [
            self.grid[d, :, c, e, t].tolist()
            for d in range(dp_size)
            for c in range(cp_size)
            for e in range(ep_size)
            for t in range(tp_size)
        ]
        self._pp_groups, self.pp_group = self._create_parallel_groups(pp_rank_lists)

        # Expert Parallelism groups: same DP, PP, CP, TP (vary EP)
        ep_rank_lists = [
            self.grid[d, p, c, :, t].tolist()
            for d in range(dp_size)
            for p in range(pp_size)
            for c in range(cp_size)
            for t in range(tp_size)
        ]
        self._ep_groups, self.ep_group = self._create_parallel_groups(ep_rank_lists)

        # Data Parallelism groups: same PP, CP, EP, TP
        dp_rank_lists = [
            self.grid[:, p, c, e, t].tolist()
            for p in range(pp_size)
            for c in range(cp_size)
            for e in range(ep_size)
            for t in range(tp_size)
        ]
        self._dp_groups, self.dp_group = self._create_parallel_groups(dp_rank_lists)

        # Context + Data Parallelism groups: same PP, EP, TP
        cp_dp_rank_lists = [
            self.grid[:, p, :, e, t].flatten().tolist()
            for p in range(pp_size)
            for e in range(ep_size)
            for t in range(tp_size)
        ]
        self._cp_dp_groups, self.cp_dp_group = self._create_parallel_groups(
            cp_dp_rank_lists
        )

        # Pipeline + Data Parallelism groups: same CP, EP, TP
        pp_dp_rank_lists = [
            self.grid[:, :, c, e, t].flatten().tolist()
            for c in range(cp_size)
            for e in range(ep_size)
            for t in range(tp_size)
        ]
        self._pp_dp_groups, self.pp_dp_group = self._create_parallel_groups(
            pp_dp_rank_lists
        )

    def _initialize_group_properties(self) -> None:
        """Initialize group IDs and properties for all parallelism strategies."""
        # Group IDs for current process
        self.tp_group_ids: list[int] = self.grid[
            self.dp_rank, self.pp_rank, self.cp_rank, self.ep_rank, :
        ].tolist()
        self.cp_group_ids: list[int] = self.grid[
            self.dp_rank, self.pp_rank, :, self.ep_rank, self.tp_rank
        ].tolist()
        self.pp_group_ids: list[int] = self.grid[
            self.dp_rank, :, self.cp_rank, self.ep_rank, self.tp_rank
        ].tolist()
        self.ep_group_ids: list[int] = self.grid[
            self.dp_rank, self.pp_rank, self.cp_rank, :, self.tp_rank
        ].tolist()
        self.dp_group_ids: list[int] = self.grid[
            :, self.pp_rank, self.cp_rank, self.ep_rank, self.tp_rank
        ].tolist()
        self.cp_dp_group_ids: list[int] = (
            self.grid[:, self.pp_rank, :, self.ep_rank, self.tp_rank].flatten().tolist()
        )
        self.pp_dp_group_ids: list[int] = (
            self.grid[:, :, self.cp_rank, self.ep_rank, self.tp_rank].flatten().tolist()
        )

        # Tensor Parallelism properties
        self.tp_world_size: int = get_world_size(group=self.tp_group)
        self.tp_first_rank: int = self.tp_group_ids[0]
        self.tp_last_rank: int = self.tp_group_ids[-1]

        # Context Parallelism properties
        self.cp_world_size: int = get_world_size(group=self.cp_group)
        self.cp_first_rank: int = self.cp_group_ids[0]
        self.cp_last_rank: int = self.cp_group_ids[-1]
        self.cp_send_rank: int = self.cp_group_ids[
            (self.cp_rank + 1) % self.cp_world_size
        ]
        self.cp_recv_rank: int = self.cp_group_ids[
            (self.cp_rank - 1) % self.cp_world_size
        ]

        # Expert Parallelism properties
        self.ep_world_size: int = (
            get_world_size(group=self.ep_group) if self.ep_group else 1
        )
        self.ep_first_rank: int = (
            self.ep_group_ids[0] if self.ep_group_ids else self.global_rank
        )
        self.ep_last_rank: int = (
            self.ep_group_ids[-1] if self.ep_group_ids else self.global_rank
        )

        # Pipeline Parallelism properties
        self.pp_world_size: int = get_world_size(group=self.pp_group)
        self.pp_first_rank: int = self.pp_group_ids[0]
        self.pp_last_rank: int = self.pp_group_ids[-1]
        self.pp_is_first_stage: bool = self.pp_rank == 0
        self.pp_is_last_stage: bool = self.pp_rank == self.pp_world_size - 1

        # Next and previous ranks in pipeline
        if self.pp_rank == self.pp_world_size - 1:
            self.pp_next_rank: int | None = None
        else:
            self.pp_next_rank = int(
                self.grid[
                    self.dp_rank,
                    self.pp_rank + 1,
                    self.cp_rank,
                    self.ep_rank,
                    self.tp_rank,
                ].item()
            )

        if self.pp_rank == 0:
            self.pp_prev_rank: int | None = None
        else:
            self.pp_prev_rank = int(
                self.grid[
                    self.dp_rank,
                    self.pp_rank - 1,
                    self.cp_rank,
                    self.ep_rank,
                    self.tp_rank,
                ].item()
            )

        # Data Parallelism properties
        self.dp_world_size: int = get_world_size(group=self.dp_group)
        self.dp_first_rank: int = self.dp_group_ids[0]
        self.dp_last_rank: int = self.dp_group_ids[-1]

        # Context + Data Parallelism properties
        self.cp_dp_world_size: int = get_world_size(group=self.cp_dp_group)

        # Pipeline + Data Parallelism properties
        self.pp_dp_world_size: int = get_world_size(group=self.pp_dp_group)

    def get_info(self) -> str:
        """
        Get comprehensive information about the current process configuration.

        Returns:
            str: Formatted string containing process group information
        """
        info = [
            f"Process Group Manager Info (Rank {self.global_rank}):",
            f"  Grid Position: DP={self.dp_rank}, PP={self.pp_rank}, CP={self.cp_rank}, TP={self.tp_rank}",
            f"  Tensor Parallelism: world_size={self.tp_world_size}, rank={self.tp_rank}",
            f"  Context Parallelism: world_size={self.cp_world_size}, rank={self.cp_rank}",
            f"  Pipeline Parallelism: world_size={self.pp_world_size}, rank={self.pp_rank}",
            f"  Data Parallelism: world_size={self.dp_world_size}, rank={self.dp_rank}",
            "  Pipeline Stage: "
            + (
                "First"
                if self.pp_is_first_stage
                else "Last"
                if self.pp_is_last_stage
                else "Middle"
            ),
        ]

        if self.pp_next_rank is not None:
            info.append(f"  Pipeline Next Rank: {self.pp_next_rank}")
        if self.pp_prev_rank is not None:
            info.append(f"  Pipeline Previous Rank: {self.pp_prev_rank}")

        return "\n".join(info)

    def __str__(self) -> str:
        """Return a compact string representation of the process configuration."""
        return f"TP({self.tp_world_size})-CP({self.cp_world_size})-PP({self.pp_world_size})-DP({self.dp_world_size})-Rank({self.global_rank})"

    def __repr__(self) -> str:
        """Return a detailed string representation of the ProcessGroupManager."""
        return (
            f"ProcessGroupManager(global_rank={self.global_rank}, world_size={self.world_size}, "
            f"tp_rank={self.tp_rank}, cp_rank={self.cp_rank}, pp_rank={self.pp_rank}, dp_rank={self.dp_rank})"
        )

    def cleanup(self) -> None:
        """Destroy all created process groups and reset the manager."""
        for group in (
            self._tp_groups
            + self._cp_groups
            + self._pp_groups
            + self._ep_groups
            + self._dp_groups
            + self._cp_dp_groups
            + self._pp_dp_groups
        ):
            with contextlib.suppress(Exception):
                destroy_group(group)


class ProcessGroupManagerProxy:
    """Proxy that always delegates to the current global instance."""

    _instance: ProcessGroupManager | None = None

    def __getattr__(self, name):
        if self._instance is None:
            raise AttributeError(
                f"Process group manager not initialized (accessing .{name})"
            )
        return getattr(self._instance, name)

    def __bool__(self):
        return self._instance is not None

    def __eq__(self, other):
        if other is None:
            return self._instance is None
        return self._instance == other

    def __ne__(self, other):
        return not self.__eq__(other)


# Global process group manager proxy - supports `from ... import process_group_manager as pgm`
process_group_manager = ProcessGroupManagerProxy()


def setup_process_group_manager(
    tp_size: int, cp_size: int, pp_size: int, dp_size: int, ep_size: int = 1
) -> ProcessGroupManager:
    """
    Set up the global process group manager.

    Args:
        tp_size (int): Size of tensor parallelism dimension
        cp_size (int): Size of context parallelism dimension
        pp_size (int): Size of pipeline parallelism dimension
        dp_size (int): Size of data parallelism dimension
        ep_size (int): Size of expert parallelism dimension

    Returns:
        ProcessGroupManager: The created process group manager instance
    """
    instance = ProcessGroupManager(tp_size, cp_size, pp_size, dp_size, ep_size)
    process_group_manager._instance = instance
    return instance


def get_process_group_manager() -> ProcessGroupManager | None:
    """
    Get the global process group manager instance.

    Returns:
        Optional[ProcessGroupManager]: The process group manager instance, or None if not set up
    """
    return process_group_manager
