# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.utils.benchmark as benchmark
from torcheval.metrics import Metric
from torcheval.metrics.toolkit import sync_and_compute


class InputType(Enum):
    # classification
    BINARY_LABEL = "binary_label"
    BINARY_LOGIT = "binary_logit"
    BINARY_PROB = "binary_prob"
    MULTICLASS_LABEL = "multiclass_label"
    MULTICLASS_LOGIT = "multiclass_logit"
    MULTICLASS_PROB = "multiclass_prob"
    MULTILABEL = "multilabel"
    # regression/aggregation
    SINGLE_VALUE = "single_output"
    MULTI_DIM = "multi_dim"


def generate_random_input(
    input_type: InputType,
    device: torch.device,
    num_batches: int,
    batch_size: int,
    num_classes: Optional[int] = None,
    num_dim: Optional[int] = None,
) -> torch.Tensor:
    if input_type == InputType.BINARY_LABEL:
        return torch.randint(high=2, size=(num_batches, batch_size), device=device)
    elif input_type == InputType.BINARY_LOGIT:
        return torch.randn(num_batches, batch_size, device=device)
    elif input_type == InputType.BINARY_PROB:
        return torch.rand(num_batches, batch_size, device=device)
    elif input_type == InputType.MULTICLASS_LABEL:
        return torch.randint(
            high=num_classes, size=(num_batches, batch_size), device=device
        )
    elif input_type == InputType.MULTICLASS_LOGIT:
        return torch.randn(num_batches, batch_size, num_classes, device=device)
    elif input_type == InputType.MULTICLASS_PROB:
        return torch.rand(num_batches, batch_size, num_classes, device=device)
    elif input_type == InputType.MULTILABEL:
        return torch.randint(
            high=num_classes, size=(num_batches, batch_size, num_classes), device=device
        )
    elif input_type == InputType.SINGLE_VALUE:
        return torch.rand(num_batches, batch_size, device=device)
    elif input_type == InputType.MULTI_DIM:
        return torch.rand(num_batches, batch_size, num_dim, device=device)


### MAIN BENCHMARK FUNCTION
def get_benchmark_stats(
    metric: Metric,
    num_batches: int,
    compute_interval: int,
    update_kwargs: Dict[str, Any],
    use_torchmetrics: bool = False,
):
    metric.reset()
    get_gpu_memory(
        metric, num_batches, compute_interval, update_kwargs, use_torchmetrics
    )
    metric.reset()
    get_time(metric, num_batches, compute_interval, update_kwargs, use_torchmetrics)


def get_time(
    metric: Metric,
    num_batches: int,
    compute_interval: int,
    update_kwargs: Dict[str, Any],
    use_torchmetrics: bool = False,
):
    timer = benchmark.Timer(
        setup="from __main__ import run_metric_computation",
        stmt="run_metric_computation(metric, num_batches, compute_interval, update_kwargs, use_torchmetrics)",
        globals={
            "metric": metric,
            "num_batches": num_batches,
            "compute_interval": compute_interval,
            "update_kwargs": update_kwargs,
            "use_torchmetrics": use_torchmetrics,
        },
    )
    result = timer.timeit(3)
    rank_0_print(f"Average time spent of 20 runs: {result.mean}")


def get_gpu_memory(
    metric: Metric,
    num_batches: int,
    compute_interval: int,
    update_kwargs: Dict[str, Any],
    use_torchmetrics: bool = False,
):
    from __main__ import run_metric_computation

    torch.cuda.empty_cache()
    run_metric_computation(
        metric, num_batches, compute_interval, update_kwargs, use_torchmetrics
    )
    rank_0_print("memory usage on rank 0...")
    rank_0_print(torch.cuda.memory_summary(abbreviated=False))


### UTILITIES
def setup_distributed() -> torch.device:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return device


def rank_0_print(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg)
