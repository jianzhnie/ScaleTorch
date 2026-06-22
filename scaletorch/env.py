"""Centralized environment variable names and default values for ScaleTorch.

All environment variable strings used across the codebase should be defined here
to avoid typos, enable auto-complete, and provide a single source of truth.
"""

# Feature toggles
ENV_FLASH_ATTENTION: str = "FLASH_ATTEN"  # Note: legacy spelling preserved
ENV_CONTEXT_PARALLEL: str = "CONTEXT_PARALLEL"
ENV_SEQUENCE_PARALLEL: str = "SEQUENCE_PARALLEL"
ENV_VERBOSE: str = "VERBOSE"

# Data / dtype
ENV_DTYPE: str = "DTYPE"

# Distributed
ENV_RANK: str = "RANK"
ENV_LOCAL_RANK: str = "LOCAL_RANK"
ENV_WORLD_SIZE: str = "WORLD_SIZE"
ENV_MASTER_ADDR: str = "MASTER_ADDR"
ENV_MASTER_PORT: str = "MASTER_PORT"

# Misc
ENV_PRINT_LOCK: str = "SCALETORCH_PRINT_LOCK"

# Default values
DEFAULT_DTYPE: str = "bfloat16"
DEFAULT_FLASH_ATTENTION_ENABLED: str = "1"
DEFAULT_PARALLELISM_DISABLED: str = "0"
