"""
Data Parallelism implementations for distributed training.

This module provides two implementations of data parallelism:
1. DataParallelNaive: Simple gradient synchronization using all-reduce
2. DataParallelBucket: Efficient gradient synchronization using bucketing to reduce communication overhead

Both implementations support gradient accumulation through the `no_sync` context manager.
"""

import contextlib
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Variable

from scaletorch.parallel.data_parallel.bucket import BucketManager
from scaletorch.parallel.pg_manager import process_group_manager as pgm


class DataParallelNaive(nn.Module):
    """
    Naive Data Parallelism implementation for educational purposes.

    This implementation provides a simple approach to gradient synchronization
    using all-reduce operations. While not used in production due to performance
    limitations, it serves as a good starting point for understanding data parallelism.

    Features:
    - Simple all-reduce gradient synchronization
    - Gradient accumulation support via `no_sync` context manager
    - Compatible with context + data parallel groups

    Note:
        This implementation is primarily for educational purposes and understanding
        the basic concepts of data parallelism.
    """

    def __init__(self, module: nn.Module) -> None:
        """
        Initialize the DataParallel wrapper for a given module.

        Args:
            module: The model to be wrapped for data parallelism

        Raises:
            RuntimeError: If process group manager is not initialized
        """
        super().__init__()

        self.module: nn.Module = module
        self.require_backward_grad_sync: bool = True  # Whether to synchronize gradients during backward pass
        self.register_backward_hook(self._allreduce_grads)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """
        Forward pass through the wrapped module.

        Args:
            *inputs: Input tensors
            **kwargs: Additional keyword arguments

        Returns:
            Module output
        """
        return self.module(*inputs, **kwargs)

    def register_backward_hook(self, hook: callable) -> None:
        """
        Register a backward hook for all parameters that require gradients.

        Args:
            hook: Hook function to be called during backward pass
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)

    def _allreduce_grads(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Perform all-reduce operation to synchronize gradients across processes.

        This method averages gradients across all processes in the context + data parallel group.

        Args:
            grad: Gradient tensor to be synchronized

        Returns:
            Synchronized gradient tensor
        """
        if self.require_backward_grad_sync:
            # Synchronize gradients across context + data parallel processes
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.cp_dp_group)
            grad.div_(pgm.cp_dp_world_size)

        return grad

    @contextlib.contextmanager
    def no_sync(self) -> None:
        """
        Context manager to temporarily disable gradient synchronization.

        This is useful for gradient accumulation where you want to perform
        multiple backward passes without synchronizing gradients in between.

        Example:
            >>> with model.no_sync():
            ...     for micro_batch in micro_batches:
            ...         loss = model(micro_batch)
            ...         loss.backward()
            >>> # Final backward pass with synchronization
            >>> loss = model(final_batch)
            >>> loss.backward()
        """
        saved_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = saved_require_backward_grad_sync


class DataParallelBucket(nn.Module):
    """
    Efficient Data Parallelism implementation using gradient bucketing.

    This implementation groups gradients into buckets to reduce communication overhead
    during gradient synchronization. It's suitable for production use and provides
    better performance than the naive implementation, especially for large models.

    Features:
    - Gradient bucketing for reduced communication overhead
    - Mixed precision training support
    - Gradient accumulation support via `no_sync` context manager
    - Automatic gradient view management

    Args:
        module: The model to be parallelized
        bucket_cap_mb: Maximum size of each gradient synchronization bucket in megabytes
        grad_type: Data type of gradients (default: torch.float32)
    """

    def __init__(self,
                 module: nn.Module,
                 bucket_cap_mb: int = 25,
                 grad_type: torch.dtype = torch.float32) -> None:
        """
        Initialize the DataParallelBucket module.

        Args:
            module: The model to be parallelized
            bucket_cap_mb: Maximum size of each gradient synchronization bucket in megabytes
            grad_type: Data type of gradients

        Raises:
            RuntimeError: If process group manager is not initialized
            ValueError: If bucket_cap_mb is not positive
        """
        super().__init__()

        if bucket_cap_mb <= 0:
            raise ValueError(
                f'bucket_cap_mb must be positive, got {bucket_cap_mb}')

        self.module: nn.Module = module
        self.require_backward_grad_sync: bool = True
        self._post_backward_callback_set: bool = False
        self.grad_type: torch.dtype = grad_type

        # Calculate bucket size in number of elements
        grad_size_bytes: int = 2 if grad_type == torch.bfloat16 else 4
        bucket_size: int = bucket_cap_mb * 1024 * 1024 // grad_size_bytes

        # Initialize bucket manager for gradient synchronization
        self.bucket_manager = BucketManager(module.parameters(),
                                            pgm.cp_dp_group, bucket_size,
                                            grad_type)

        self.register_backward_hook()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """
        Forward pass through the wrapped module.

        Args:
            *inputs: Input tensors
            **kwargs: Additional keyword arguments

        Returns:
            Module output
        """
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor,
                 output_tensor_grad: torch.Tensor) -> torch.Tensor:
        """
        Custom backward pass for the module.

        Args:
            input_tensor: Input tensor
            output_tensor: Output tensor
            output_tensor_grad: Gradient of the output tensor

        Returns:
            Gradient of the input tensor
        """
        return self.module.backward(input_tensor, output_tensor,
                                    output_tensor_grad)

    def register_backward_hook(self) -> None:
        """
        Register backward hooks for manual gradient accumulation and synchronization.

        This method sets up hooks that:
        1. Accumulate gradients into main gradient storage
        2. Handle mixed precision training
        3. Coordinate gradient synchronization through buckets

        The gradient accumulation functions are stored to prevent them from going out of scope.

        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []

        for param in self.module.parameters():
            if param.requires_grad:
                # Expand parameter to access gradient function
                param_tmp = param.expand_as(param)
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]

                # Register hook for gradient accumulation and synchronization
                hook = self._make_param_hook(param, self.bucket_manager)
                grad_acc_fn.register_hook(hook)
                self.grad_accs.append(grad_acc_fn)

    def _make_param_hook(self, param: nn.Parameter,
                         bucket_manager: BucketManager) -> callable:
        """
        Create a hook for parameter-specific gradient accumulation and synchronization.

        Args:
            param: Parameter to create hook for
            bucket_manager: Bucket manager for gradient synchronization

        Returns:
            Hook function for gradient processing
        """

        def param_hook(*unused: Any) -> None:
            """
            Hook called after gradient is ready.

            Performs the following operations:
            1. Accumulates gradient into main gradient storage
            2. Clears parameter gradient to prevent double accumulation
            3. Sets up post-backward callback for synchronization
            4. Marks parameter as ready for bucket synchronization
            """
            if param.requires_grad and param.grad is not None:
                # Accumulate gradients into main gradient storage
                param.main_grad.add_(param.grad.data)
                param.grad = None  # Clear parameter gradient

                # Skip gradient synchronization during gradient accumulation
                if self.require_backward_grad_sync:
                    # Add post-backward callback (only once per backward pass)
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(
                            self._post_backward)
                        self._post_backward_callback_set = True

                    # Mark parameter as ready for bucket synchronization
                    bucket_manager.mark_param_as_ready(param)

        return param_hook

    @contextlib.contextmanager
    def no_sync(self) -> None:
        """
        Context manager to temporarily disable gradient synchronization.

        This is useful for gradient accumulation where you want to perform
        multiple backward passes without synchronizing gradients in between.

        Example:
            >>> with model.no_sync():
            ...     for micro_batch in micro_batches:
            ...         loss = model(micro_batch)
            ...         loss.backward()
            >>> # Final backward pass with synchronization
            >>> loss = model(final_batch)
            >>> loss.backward()
        """
        saved_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = saved_require_backward_grad_sync

    def _post_backward(self) -> None:
        """
        Post-backward callback to finalize gradient synchronization.

        This method:
        1. Waits for all bucket synchronizations to complete
        2. Copies synchronized gradients back to parameters
        3. Resets synchronization state

        Called after backward pass and before optimizer step.
        """
        # Wait for all bucket synchronizations to complete
        self.bucket_manager.wait()
        self._post_backward_callback_set = False

        # Copy synchronized gradients back to parameters
        for param in self.module.parameters():
            if param.requires_grad and hasattr(param, 'main_grad'):
                # Convert to parameter's data type for optimizer compatibility
                param.grad = param.main_grad.to(param.dtype)

    def reset(self) -> None:
        """
        Reset the bucket manager and clear gradients in the model.

        This method should be called when you need to clear all gradients,
        typically at the beginning of a new training iteration.
        """
        self.bucket_manager.reset()

        # Clear parameter gradients
        for param in self.module.parameters():
            param.grad = None

    def get_bucket_info(self) -> str:
        """
        Get detailed information about bucket configuration and status.

        Returns:
            String containing bucket manager information
        """
        return self.bucket_manager.get_bucket_info()

    def __str__(self) -> str:
        """Return string representation of the DataParallelBucket module."""
        return (f'DataParallelBucket(module={self.module.__class__.__name__}, '
                f'buckets={len(self.bucket_manager.buckets)}, '
                f'grad_type={self.grad_type})')

    def __repr__(self) -> str:
        """Return detailed string representation of the DataParallelBucket module."""
        return (f'DataParallelBucket(module={self.module}, '
                f'bucket_manager={self.bucket_manager}, '
                f'require_sync={self.require_backward_grad_sync})')
