"""
Process Group Manager for Distributed Training.

This module provides a comprehensive manager for handling different types of parallelism
in distributed training environments, including Tensor Parallelism (TP), Context Parallelism (CP),
Pipeline Parallelism (PP), and Data Parallelism (DP).
"""

import os
from typing import List, Optional

import torch
import torch.distributed as dist


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

    def __init__(self, tp_size: int, cp_size: int, pp_size: int,
                 dp_size: int) -> None:
        """
        Initialize the ProcessGroupManager.

        Args:
            tp_size (int): Size of tensor parallelism dimension
            cp_size (int): Size of context parallelism dimension
            pp_size (int): Size of pipeline parallelism dimension
            dp_size (int): Size of data parallelism dimension

        Raises:
            RuntimeError: If distributed training is not initialized
            ValueError: If world_size doesn't equal tp_size * cp_size * pp_size * dp_size
        """
        # Check if distributed training is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                'Distributed training must be initialized before creating ProcessGroupManager'
            )

        # Validate parallelism sizes
        for size, name in [(tp_size, 'tp_size'), (cp_size, 'cp_size'),
                           (pp_size, 'pp_size'), (dp_size, 'dp_size')]:
            if size <= 0:
                raise ValueError(f'{name} must be positive, got {size}')

        self.global_rank: int = dist.get_rank()
        self.world_size: int = dist.get_world_size()
        self.local_rank: int = int(
            os.environ.get('LOCAL_RANK', self.global_rank % self.world_size))

        # Validate world size matches the product of all parallelism dimensions
        expected_world_size = tp_size * cp_size * pp_size * dp_size
        if self.world_size != expected_world_size:
            raise ValueError(
                f'World size ({self.world_size}) != TP ({tp_size}) * CP ({cp_size}) * PP ({pp_size}) * DP ({dp_size}) = {expected_world_size}. '
                f'Please check your distributed training setup and ensure the total number of processes matches the product of all parallelism dimensions.'
            )

        # Create 4D grid: [DP, PP, CP, TP]
        self.grid: torch.Tensor = torch.arange(self.world_size).view(
            dp_size, pp_size, cp_size, tp_size)

        # Find the position of the current process in the grid
        position = (self.grid == self.global_rank).nonzero().flatten()
        if len(position) != 4:
            raise RuntimeError(
                f'Could not find rank {self.global_rank} in the process grid')

        self.dp_rank: int = int(position[0].item())
        self.pp_rank: int = int(position[1].item())
        self.cp_rank: int = int(position[2].item())
        self.tp_rank: int = int(position[3].item())

        # Create process groups for different parallelism strategies
        self._create_process_groups(tp_size, cp_size, pp_size, dp_size)

        # Initialize group IDs and properties
        self._initialize_group_properties()

        # Store reference to world group
        self.world_group = dist.group.WORLD

    def _create_process_groups(self, tp_size: int, cp_size: int, pp_size: int,
                               dp_size: int) -> None:
        """
        Create process groups for different parallelism strategies.

        Args:
            tp_size (int): Size of tensor parallelism dimension
            cp_size (int): Size of context parallelism dimension
            pp_size (int): Size of pipeline parallelism dimension
            dp_size (int): Size of data parallelism dimension
        """
        # Create group objects for each logical group and select the one
        # that contains the current rank. PyTorch expects a ranks list per
        # call to dist.new_group, not a list-of-lists.

        # Tensor Parallelism groups: processes with same DP, PP, CP ranks
        self._tp_groups: List[dist.ProcessGroup] = []
        tp_rank_lists: List[List[int]] = []
        for d in range(dp_size):
            for p in range(pp_size):
                for c in range(cp_size):
                    ranks = self.grid[d, p, c, :].tolist()
                    tp_rank_lists.append(ranks)
                    self._tp_groups.append(dist.new_group(ranks=ranks))
        # pick current rank's tp_group
        for ranks, group in zip(tp_rank_lists, self._tp_groups):
            if self.global_rank in ranks:
                self.tp_group = group
                break

        # Context Parallelism groups: processes with same DP, PP, TP ranks
        self._cp_groups: List[dist.ProcessGroup] = []
        cp_rank_lists: List[List[int]] = []
        for d in range(dp_size):
            for p in range(pp_size):
                for t in range(tp_size):
                    ranks = self.grid[d, p, :, t].tolist()
                    cp_rank_lists.append(ranks)
                    self._cp_groups.append(dist.new_group(ranks=ranks))
        for ranks, group in zip(cp_rank_lists, self._cp_groups):
            if self.global_rank in ranks:
                self.cp_group = group
                break

        # Pipeline Parallelism groups: processes with same DP, CP, TP ranks
        self._pp_groups: List[dist.ProcessGroup] = []
        pp_rank_lists: List[List[int]] = []
        for d in range(dp_size):
            for c in range(cp_size):
                for t in range(tp_size):
                    ranks = self.grid[d, :, c, t].tolist()
                    pp_rank_lists.append(ranks)
                    self._pp_groups.append(dist.new_group(ranks=ranks))
        for ranks, group in zip(pp_rank_lists, self._pp_groups):
            if self.global_rank in ranks:
                self.pp_group = group
                break

        # Data Parallelism groups: processes with same PP, CP, TP ranks
        self._dp_groups: List[dist.ProcessGroup] = []
        dp_rank_lists: List[List[int]] = []
        for p in range(pp_size):
            for c in range(cp_size):
                for t in range(tp_size):
                    ranks = self.grid[:, p, c, t].tolist()
                    dp_rank_lists.append(ranks)
                    self._dp_groups.append(dist.new_group(ranks=ranks))
        for ranks, group in zip(dp_rank_lists, self._dp_groups):
            if self.global_rank in ranks:
                self.dp_group = group
                break

        # Context + Data Parallelism groups: processes with same PP, TP ranks
        self._cp_dp_groups: List[dist.ProcessGroup] = []
        cp_dp_rank_lists: List[List[int]] = []
        for p in range(pp_size):
            for t in range(tp_size):
                ranks = self.grid[:, p, :, t].flatten().tolist()
                cp_dp_rank_lists.append(ranks)
                self._cp_dp_groups.append(dist.new_group(ranks=ranks))
        for ranks, group in zip(cp_dp_rank_lists, self._cp_dp_groups):
            if self.global_rank in ranks:
                self.cp_dp_group = group
                break

        # Pipeline + Data Parallelism groups: processes with same CP, TP ranks
        self._pp_dp_groups: List[dist.ProcessGroup] = []
        pp_dp_rank_lists: List[List[int]] = []
        for c in range(cp_size):
            for t in range(tp_size):
                ranks = self.grid[:, :, c, t].flatten().tolist()
                pp_dp_rank_lists.append(ranks)
                self._pp_dp_groups.append(dist.new_group(ranks=ranks))
        for ranks, group in zip(pp_dp_rank_lists, self._pp_dp_groups):
            if self.global_rank in ranks:
                self.pp_dp_group = group
                break

    def _initialize_group_properties(self) -> None:
        """Initialize group IDs and properties for all parallelism strategies."""
        # Group IDs for current process
        self.tp_group_ids: List[int] = self.grid[self.dp_rank, self.pp_rank,
                                                 self.cp_rank, :].tolist()
        self.cp_group_ids: List[int] = self.grid[self.dp_rank, self.pp_rank, :,
                                                 self.tp_rank].tolist()
        self.pp_group_ids: List[int] = self.grid[self.dp_rank, :, self.cp_rank,
                                                 self.tp_rank].tolist()
        self.dp_group_ids: List[int] = self.grid[:, self.pp_rank, self.cp_rank,
                                                 self.tp_rank].tolist()
        self.cp_dp_group_ids: List[int] = self.grid[:, self.pp_rank, :,
                                                    self.tp_rank].flatten(
                                                    ).tolist()
        self.pp_dp_group_ids: List[int] = self.grid[:, :, self.cp_rank,
                                                    self.tp_rank].flatten(
                                                    ).tolist()

        # Tensor Parallelism properties
        self.tp_world_size: int = dist.get_world_size(group=self.tp_group)
        self.tp_first_rank: int = self.tp_group_ids[0]
        self.tp_last_rank: int = self.tp_group_ids[-1]

        # Context Parallelism properties
        self.cp_world_size: int = dist.get_world_size(group=self.cp_group)
        self.cp_first_rank: int = self.cp_group_ids[0]
        self.cp_last_rank: int = self.cp_group_ids[-1]
        self.cp_send_rank: int = self.cp_group_ids[(self.cp_rank + 1) %
                                                   self.cp_world_size]
        self.cp_recv_rank: int = self.cp_group_ids[(self.cp_rank - 1) %
                                                   self.cp_world_size]

        # Pipeline Parallelism properties
        self.pp_world_size: int = dist.get_world_size(group=self.pp_group)
        self.pp_first_rank: int = self.pp_group_ids[0]
        self.pp_last_rank: int = self.pp_group_ids[-1]
        self.pp_is_first_stage: bool = self.pp_rank == 0
        self.pp_is_last_stage: bool = self.pp_rank == self.pp_world_size - 1

        # Next and previous ranks in pipeline
        if self.pp_rank == self.pp_world_size - 1:
            self.pp_next_rank: Optional[int] = None
        else:
            self.pp_next_rank = int(self.grid[self.dp_rank, self.pp_rank + 1,
                                              self.cp_rank,
                                              self.tp_rank].item())

        if self.pp_rank == 0:
            self.pp_prev_rank: Optional[int] = None
        else:
            self.pp_prev_rank = int(self.grid[self.dp_rank, self.pp_rank - 1,
                                              self.cp_rank,
                                              self.tp_rank].item())

        # Data Parallelism properties
        self.dp_world_size: int = dist.get_world_size(group=self.dp_group)
        self.dp_first_rank: int = self.dp_group_ids[0]
        self.dp_last_rank: int = self.dp_group_ids[-1]

        # Context + Data Parallelism properties
        self.cp_dp_world_size: int = dist.get_world_size(
            group=self.cp_dp_group)

        # Pipeline + Data Parallelism properties
        self.pp_dp_world_size: int = dist.get_world_size(
            group=self.pp_dp_group)

    def get_info(self) -> str:
        """
        Get comprehensive information about the current process configuration.

        Returns:
            str: Formatted string containing process group information
        """
        info = [
            f'Process Group Manager Info (Rank {self.global_rank}):',
            f'  Grid Position: DP={self.dp_rank}, PP={self.pp_rank}, CP={self.cp_rank}, TP={self.tp_rank}',
            f'  Tensor Parallelism: world_size={self.tp_world_size}, rank={self.tp_rank}',
            f'  Context Parallelism: world_size={self.cp_world_size}, rank={self.cp_rank}',
            f'  Pipeline Parallelism: world_size={self.pp_world_size}, rank={self.pp_rank}',
            f'  Data Parallelism: world_size={self.dp_world_size}, rank={self.dp_rank}',
            f"  Pipeline Stage: {'First' if self.pp_is_first_stage else 'Last' if self.pp_is_last_stage else 'Middle'}",
        ]

        if self.pp_next_rank is not None:
            info.append(f'  Pipeline Next Rank: {self.pp_next_rank}')
        if self.pp_prev_rank is not None:
            info.append(f'  Pipeline Previous Rank: {self.pp_prev_rank}')

        return '\n'.join(info)

    def __str__(self) -> str:
        """Return a compact string representation of the process configuration."""
        return f'TP({self.tp_world_size})-CP({self.cp_world_size})-PP({self.pp_world_size})-DP({self.dp_world_size})-Rank({self.global_rank})'

    def __repr__(self) -> str:
        """Return a detailed string representation of the ProcessGroupManager."""
        return (
            f'ProcessGroupManager(global_rank={self.global_rank}, world_size={self.world_size}, '
            f'tp_rank={self.tp_rank}, cp_rank={self.cp_rank}, pp_rank={self.pp_rank}, dp_rank={self.dp_rank})'
        )


# Global process group manager instance
process_group_manager: Optional[ProcessGroupManager] = None


def setup_process_group_manager(tp_size: int, cp_size: int, pp_size: int,
                                dp_size: int) -> ProcessGroupManager:
    """
    Set up the global process group manager.

    Args:
        tp_size (int): Size of tensor parallelism dimension
        cp_size (int): Size of context parallelism dimension
        pp_size (int): Size of pipeline parallelism dimension
        dp_size (int): Size of data parallelism dimension

    Returns:
        ProcessGroupManager: The created process group manager instance

    Raises:
        RuntimeError: If distributed training is not initialized

    Example:
        >>> setup_process_group_manager(tp_size=2, cp_size=2, pp_size=2, dp_size=2)
        ProcessGroupManager instance
    """
    global process_group_manager
    process_group_manager = ProcessGroupManager(tp_size, cp_size, pp_size,
                                                dp_size)
    return process_group_manager


def get_process_group_manager() -> Optional[ProcessGroupManager]:
    """
    Get the global process group manager instance.

    Returns:
        Optional[ProcessGroupManager]: The process group manager instance, or None if not set up
    """
    return process_group_manager
