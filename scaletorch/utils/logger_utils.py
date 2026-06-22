"""Colored distributed-aware logging with rank information."""

from __future__ import annotations

import logging
import os
import sys
from logging import Formatter, LogRecord
from pathlib import Path
from typing import ClassVar

import torch.distributed as dist
from colorama import Fore, Style

logger_initialized: dict[str, bool] = {}


class ColorfulFormatter(Formatter):
    """Formatter that adds ANSI color codes and rank information to log messages."""

    COLORS: ClassVar[dict[str, str]] = {
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.LIGHTGREEN_EX,
    }

    def format(self, record: LogRecord) -> str:
        # Add rank information to the record
        record.rank = self._get_rank()
        record.is_main = record.rank == 0

        # Format the log message
        log_message = super().format(record)

        # Add color based on log level
        return self.COLORS.get(record.levelname, "") + log_message + Fore.RESET

    def _get_rank(self) -> int:
        return _get_distributed_rank()


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    log_level: int = logging.INFO,
    file_mode: str = "w",
    force_main_process: bool = False,
) -> logging.Logger:
    """Create or retrieve a logger with optional file output and distributed-aware log levels."""
    if file_mode not in ("w", "a"):
        raise ValueError("file_mode must be either 'w' or 'a'")

    # Get or create logger instance
    logger = logging.getLogger(name)

    # Return existing logger if already initialized
    if name in logger_initialized:
        return logger

    # Check if parent logger is already initialized
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # Get current rank safely
    rank = _get_distributed_rank()
    is_main_process = rank == 0

    # Fix PyTorch DDP duplicate logging issue
    # Clear existing handlers to prevent duplicate logging
    if logger.handlers:
        logger.handlers.clear()

    # Only configure handlers for main process or if explicitly requested
    if is_main_process or not force_main_process:
        # Initialize handlers list
        handlers = []

        # Add StreamHandler for main process only
        if is_main_process:
            stream_handler = logging.StreamHandler(sys.stdout)
            handlers.append(stream_handler)

        # Add FileHandler for rank 0 process if log_file is specified
        if is_main_process and log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(str(log_file), file_mode))

        # Configure formatter with rank information
        if is_main_process:
            fmt = "%(asctime)s - [Rank %(rank)d] - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
        else:
            fmt = (
                "%(asctime)s - [Rank %(rank)d] - %(name)s - %(levelname)s - %(message)s"
            )

        formatter = ColorfulFormatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

        # Apply configuration to all handlers
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level if is_main_process else logging.ERROR)
            logger.addHandler(handler)

    # Set logger level based on rank and configuration
    if force_main_process:
        logger.setLevel(
            log_level if is_main_process else logging.CRITICAL + 1
        )  # Disable logging for non-main processes
    else:
        logger.setLevel(log_level if is_main_process else logging.ERROR)

    # Mark logger as initialized
    logger_initialized[name] = True

    return logger


def _get_distributed_rank() -> int:
    """Return the current distributed rank, falling back to the RANK env var or 0."""
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass

    # Fallback to environment variables
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)

    return 0


def get_outdir(path: str, *paths, inc: bool = False) -> str:
    """Create and return an output directory. If inc=True, append an incrementing suffix to avoid collisions."""
    outdir = os.path.join(path, *paths)
    os.makedirs(outdir, exist_ok=True)
    if not inc:
        return outdir

    for count in range(1, 100):
        outdir_inc = f"{outdir}-{count}"
        if not os.path.exists(outdir_inc):
            os.makedirs(outdir_inc)
            return outdir_inc
    raise RuntimeError("Failed to create unique output directory after 100 attempts")
