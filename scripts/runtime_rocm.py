"""Boot the isolated single-GPU ROCm runtime without modifying its packages."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any


def bootstrap_torch(runtime_site: Path) -> Any:
    if not runtime_site.is_dir():
        raise RuntimeError(f"runtime site does not exist: {runtime_site}")
    sys.path.insert(0, str(runtime_site))
    import torch

    # The validated portable Torch build omits distributed c10d. Transformers
    # imports optional FSDP symbols eagerly, while this runner is single-GPU.
    try:
        import torch.distributed.fsdp  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        composable = types.ModuleType("torch.distributed._composable")
        fsdp_v2 = types.ModuleType("torch.distributed._composable.fsdp")
        fsdp_v1 = types.ModuleType("torch.distributed.fsdp")

        def unavailable(*_: object, **__: object) -> None:
            raise RuntimeError("FSDP is unavailable in this single-GPU runtime")

        class CPUOffloadPolicy:
            pass

        class MixedPrecisionPolicy:
            def __init__(self, **_: object) -> None:
                pass

        class FullyShardedDataParallel:
            pass

        for module, name, value in (
            (fsdp_v2, "fully_shard", unavailable),
            (composable, "fsdp", fsdp_v2),
            (fsdp_v1, "CPUOffloadPolicy", CPUOffloadPolicy),
            (fsdp_v1, "MixedPrecisionPolicy", MixedPrecisionPolicy),
            (fsdp_v1, "FullyShardedDataParallel", FullyShardedDataParallel),
        ):
            setattr(module, name, value)
        sys.modules["torch.distributed._composable"] = composable
        sys.modules["torch.distributed._composable.fsdp"] = fsdp_v2
        sys.modules["torch.distributed.fsdp"] = fsdp_v1
    return torch
