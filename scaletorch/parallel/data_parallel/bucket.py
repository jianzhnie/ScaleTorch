"""
Gradient Bucketing for Distributed Data Parallel Training.

This module provides efficient gradient synchronization through bucketing mechanisms,
reducing communication overhead by grouping parameters and synchronizing their gradients
in batches rather than individually.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn


class Bucket:
    """
    Manages gradient synchronization for a group of parameters.

    A bucket collects gradients from multiple parameters and performs a single
    all-reduce operation to synchronize them across processes, improving
    communication efficiency in distributed training.

    Attributes:
        params (set): Set of parameters assigned to this bucket
        params_with_grad_ready (set): Parameters with gradients ready for synchronization
        grad_data (torch.Tensor): Tensor storing gradients for all parameters in the bucket
        process_group (torch.distributed.ProcessGroup): Process group for gradient synchronization
        process_group_size (int): Number of processes in the process group
        handle (Optional[torch.distributed.Work]): Handle for the async all-reduce operation
    """

    def __init__(self, params: List[nn.Parameter], grad_data: torch.Tensor,
                 process_group: dist.ProcessGroup) -> None:
        """
        Initialize a Bucket instance.

        Args:
            params: List of parameters assigned to this bucket
            grad_data: Tensor to store the gradients for this bucket
            process_group: Process group used for synchronizing gradients

        Raises:
            ValueError: If params list is empty or grad_data is not a tensor
        """
        if not params:
            raise ValueError('Parameter list cannot be empty')
        if not isinstance(grad_data, torch.Tensor):
            raise ValueError('grad_data must be a torch.Tensor')

        self.params: set[nn.Parameter] = set(params)
        self.params_with_grad_ready: set[nn.Parameter] = set()
        self.grad_data: torch.Tensor = grad_data
        self.process_group: dist.ProcessGroup = process_group
        self.process_group_size: int = dist.get_world_size(
            group=self.process_group)
        self.handle: Optional[dist.Work] = None

        self.reset()

    def sync_gradient(self) -> None:
        """
        Launch an asynchronous all-reduce operation to synchronize gradients across processes.

        The gradients are averaged by dividing by the process group size before synchronization.

        Raises:
            RuntimeError: If a synchronization operation is already in progress
        """
        if self.handle is not None:
            raise RuntimeError(
                'Cannot start new synchronization while previous one is in progress'
            )

        # Average gradients across processes
        self.grad_data.div_(self.process_group_size)

        # Launch asynchronous all-reduce operation
        self.handle = dist.all_reduce(self.grad_data,
                                      group=self.process_group,
                                      async_op=True)

    def reset(self) -> None:
        """
        Reset the bucket to its initial state.

        Typically called after gradient synchronization is finished to prepare
        for the next iteration.
        """
        self.handle = None
        self.params_with_grad_ready.clear()
        self.grad_data.zero_()

    def wait(self) -> None:
        """
        Wait for the all-reduce operation to complete.

        Raises:
            RuntimeError: If no synchronization operation is in progress
        """
        if self.handle is None:
            raise RuntimeError('No synchronization operation in progress')

        self.handle.wait()
        self.handle = None

    def is_synchronization_complete(self) -> bool:
        """
        Check if gradient synchronization is complete.

        Returns:
            bool: True if all parameters have ready gradients, False otherwise
        """
        return len(self.params_with_grad_ready) == len(self.params)

    def mark_param_as_ready(self, param: nn.Parameter) -> None:
        """
        Mark a parameter as ready for gradient synchronization.

        Launches synchronization when all parameters in the bucket have their gradients ready.

        Args:
            param: Parameter to mark as ready

        Raises:
            ValueError: If parameter is not in this bucket or already marked as ready
        """
        if param not in self.params:
            raise ValueError(f'Parameter {param} is not in this bucket')
        if param in self.params_with_grad_ready:
            raise ValueError(f'Parameter {param} is already marked as ready')

        self.params_with_grad_ready.add(param)

        # Launch synchronization when all parameters are ready
        if self.is_synchronization_complete():
            self.sync_gradient()


class BucketManager:
    """
    Manages multiple buckets for efficient gradient synchronization.

    Divides model parameters into buckets based on size constraints and coordinates
    gradient synchronization across all buckets.

    Attributes:
        params (List[nn.Parameter]): List of model parameters
        device (torch.device): Device where parameters and gradients reside
        buckets (List[Bucket]): List of buckets for gradient synchronization
        process_group (dist.ProcessGroup): Process group for gradient synchronization
        process_group_size (int): Number of processes in the process group
        params_to_bucket_location (Dict[nn.Parameter, Tuple[int, int, int]]):
            Mapping from parameters to their bucket locations (start, end, bucket_idx)
        bucket_size (int): Maximum size of each bucket in terms of gradient elements
        bucket_sizes (Optional[List[int]]): Actual sizes of each bucket
        grad_data_list (List[torch.Tensor]): List of gradient tensors, one per bucket
        grad_type (torch.dtype): Data type of gradients
    """

    def __init__(self,
                 params: List[nn.Parameter],
                 process_group: dist.ProcessGroup,
                 bucket_size: int,
                 grad_type: torch.dtype = torch.float32) -> None:
        """
        Initialize the BucketManager.

        Args:
            params: List of model parameters
            process_group: Process group used for gradient synchronization
            bucket_size: Maximum size of each bucket in terms of gradient elements
            grad_type: Data type of gradients, defaults to torch.float32

        Raises:
            ValueError: If params list is empty or bucket_size is not positive
        """
        if not params:
            raise ValueError('Parameter list cannot be empty')
        if bucket_size <= 0:
            raise ValueError('Bucket size must be positive')

        self.params: List[nn.Parameter] = list(params)
        self.device: torch.device = (self.params[0].device
                                     if self.params[0].is_cuda else
                                     torch.device('cpu'))
        self.buckets: List[Bucket] = []
        self.process_group: dist.ProcessGroup = process_group
        self.process_group_size: int = dist.get_world_size(
            group=self.process_group)
        self.params_to_bucket_location: Dict[nn.Parameter, Tuple[int, int,
                                                                 int]] = {}
        self.bucket_size: int = bucket_size
        self.bucket_sizes: Optional[List[int]] = None
        self.grad_data_list: List[torch.Tensor] = []
        self.grad_type: torch.dtype = grad_type

        self._initialize_buckets()

    def _initialize_buckets(self) -> None:
        """
        Divide model parameters into buckets for gradient synchronization based on bucket size.

        This method implements a greedy algorithm that assigns parameters to buckets
        while respecting the bucket size constraint.
        """
        current_bucket_size: int = 0
        current_bucket_idx: int = 0

        # Assign parameters to buckets using greedy approach
        for param in self.params:
            if not param.requires_grad:
                continue

            param_size: int = param.numel()

            # Start new bucket if current parameter doesn't fit
            if current_bucket_size > 0 and current_bucket_size + param_size > self.bucket_size:
                current_bucket_idx += 1
                current_bucket_size = 0

            # Assign parameter to current bucket
            start_idx: int = current_bucket_size
            end_idx: int = current_bucket_size + param_size
            self.params_to_bucket_location[param] = (start_idx, end_idx,
                                                     current_bucket_idx)
            current_bucket_size += param_size

        # Calculate final bucket sizes and organize parameters
        self._finalize_bucket_creation()

    def _finalize_bucket_creation(self) -> None:
        """
        Finalize bucket creation by calculating sizes and creating Bucket objects.
        """
        # Determine number of buckets needed
        num_buckets: int = max(
            location[2]
            for location in self.params_to_bucket_location.values()) + 1

        # Initialize bucket tracking structures
        bucket_sizes: List[int] = [0] * num_buckets
        buckets_to_params: List[List[nn.Parameter]] = [
            [] for _ in range(num_buckets)
        ]

        # Organize parameters by bucket
        for param, (start, end,
                    bucket_idx) in self.params_to_bucket_location.items():
            bucket_sizes[bucket_idx] = max(bucket_sizes[bucket_idx], end)
            buckets_to_params[bucket_idx].append(param)

        # Create gradient tensors and Bucket objects
        for i in range(num_buckets):
            grad_tensor: torch.Tensor = torch.zeros(bucket_sizes[i],
                                                    dtype=self.grad_type,
                                                    device=self.device)
            self.grad_data_list.append(grad_tensor)
            self.buckets.append(
                Bucket(buckets_to_params[i], grad_tensor, self.process_group))

        # Create gradient views for each parameter
        self._create_gradient_views()

    def _create_gradient_views(self) -> None:
        """
        Create gradient views for each parameter pointing to the appropriate bucket location.

        This allows parameters to directly use the bucket gradient memory for gradient
        accumulation, eliminating the need for additional copies.
        """
        for param in self.params:
            if not param.requires_grad:
                continue

            start_idx, end_idx, bucket_idx = self.params_to_bucket_location[
                param]
            param.main_grad = self._get_view_from_tensor(
                self.grad_data_list[bucket_idx], param.shape, start_idx,
                end_idx)

    def _get_view_from_tensor(self, tensor: torch.Tensor, shape: torch.Size,
                              start: int, end: int) -> torch.Tensor:
        """
        Create a view of the given tensor with the specified shape from start to end indices.

        Args:
            tensor: Source tensor
            shape: Desired shape for the view
            start: Starting index in the tensor
            end: Ending index in the tensor

        Returns:
            torch.Tensor: View of the tensor with specified shape

        Raises:
            ValueError: If the shape doesn't match the specified indices
        """
        expected_size: int = torch.Size(shape).numel()
        actual_size: int = end - start

        if expected_size != actual_size:
            raise ValueError(
                f'Shape {shape} expects {expected_size} elements, '
                f'but indices [{start}, {end}) provide {actual_size} elements')

        return tensor[start:end].view(shape)

    def reset(self) -> None:
        """Reset all buckets by clearing gradients and internal states."""
        for bucket in self.buckets:
            bucket.reset()

    def wait(self) -> None:
        """Wait for all buckets to complete their gradient synchronization."""
        for bucket in self.buckets:
            bucket.wait()

    def is_synchronization_complete(self) -> bool:
        """
        Check if gradient synchronization is complete for all buckets.

        Returns:
            bool: True if all buckets have completed synchronization, False otherwise
        """
        return all(bucket.is_synchronization_complete()
                   for bucket in self.buckets)

    def mark_param_as_ready(self, param: nn.Parameter) -> None:
        """
        Mark a parameter's gradient as ready for synchronization.

        Args:
            param: Parameter to mark as ready

        Raises:
            ValueError: If parameter is not found in any bucket
        """
        if param not in self.params_to_bucket_location:
            raise ValueError(f'Parameter {param} not found in any bucket')

        bucket_idx: int = self.params_to_bucket_location[param][2]
        self.buckets[bucket_idx].mark_param_as_ready(param)

    def get_bucket_info(self) -> str:
        """
        Get detailed information about bucket configuration.

        Returns:
            str: Formatted string containing bucket information
        """
        info_lines: List[str] = [
            'BucketManager Info:',
            f'  Total Parameters: {len(self.params)}',
            f'  Parameters with Grad: {sum(1 for p in self.params if p.requires_grad)}',
            f'  Number of Buckets: {len(self.buckets)}',
            f'  Bucket Size Limit: {self.bucket_size}',
            f'  Process Group Size: {self.process_group_size}',
            f'  Device: {self.device}',
        ]

        for i, (bucket, size) in enumerate(
                zip(self.buckets, [len(b.params) for b in self.buckets])):
            info_lines.append(f'  Bucket {i}: {size} parameters')

        return '\n'.join(info_lines)

    def __str__(self) -> str:
        """Return a compact string representation of the BucketManager."""
        return (f'BucketManager(buckets={len(self.buckets)}, '
                f'params={len(self.params)}, device={self.device})')

    def __repr__(self) -> str:
        """Return a detailed string representation of the BucketManager."""
        return (
            f'BucketManager(params={len(self.params)}, '
            f'buckets={len(self.buckets)}, bucket_size={self.bucket_size}, '
            f'device={self.device}, grad_type={self.grad_type})')
