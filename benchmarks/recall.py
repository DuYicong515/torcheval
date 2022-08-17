# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import uuid
from typing import Any, Callable, Dict

import torch
import torch.distributed.launcher as pet
from torcheval.metrics import BinaryRecall, Metric, MulticlassRecall
from torcheval.metrics.toolkit import sync_and_compute
from torchmetrics import Recall

from .utils import (
    generate_random_input,
    get_benchmark_stats,
    InputType,
    rank_0_print,
    setup_distributed,
)


### INPUT GENERATION
def generate_update_args_binary(
    num_batches: int, batch_size: int, device: torch.device
) -> Dict[str, Any]:
    input = generate_random_input(
        InputType.BINARY_PROB, device, num_batches, batch_size
    )
    target = generate_random_input(
        InputType.BINARY_LABEL, device, num_batches, batch_size
    )
    return {"input": input, "target": target}


def generate_update_args_multiclass(
    num_batches: int, batch_size: int, device: torch.device, num_classes: int
) -> Dict[str, Any]:
    input = generate_random_input(
        InputType.MULTICLASS_LOGIT,
        device,
        num_batches,
        batch_size,
        num_classes=num_classes,
    )
    target = generate_random_input(
        InputType.MULTICLASS_LABEL,
        device,
        num_batches,
        batch_size,
        num_classes=num_classes,
    )
    return {"input": input, "target": target}


### METRIC COMPUTATIONS
def run_metric_computation(
    metric: Metric,
    num_batches: int,
    compute_interval: int,
    update_kwargs: Dict[str, Any],
    use_torchmetrics: bool = False,
):
    for batch_idx in range(num_batches):
        metric.update(
            update_kwargs["input"][batch_idx], update_kwargs["target"][batch_idx]
        )
        if (batch_idx + 1) % compute_interval == 0:
            if use_torchmetrics:
                metric.compute()
            else:
                sync_and_compute(metric)


def main(num_batches, batch_size, compute_interval, num_classes, use_torchmetrics):
    device = setup_distributed()

    rank_0_print(f"Benchmark BinaryPrecision...")
    if use_torchmetrics:
        metric = Recall().to(device)
    else:
        metric = BinaryRecall(device=device)
    update_kwargs = generate_update_args_binary(num_batches, batch_size, device)
    get_benchmark_stats(
        metric, num_batches, compute_interval, update_kwargs, use_torchmetrics
    )

    rank_0_print(f"Benchmark MulticlassRecall...")
    if use_torchmetrics:
        metric = Recall().to(device)
    else:
        metric = MulticlassRecall(device=device)
    update_kwargs = generate_update_args_multiclass(
        num_batches, batch_size, device, num_classes
    )
    get_benchmark_stats(
        metric, num_batches, compute_interval, update_kwargs, use_torchmetrics
    )


if __name__ == "__main__":
    print("Benchmark recall metrics...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-batches", type=int, default=int(10_000))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--compute_interval", type=int, default=100)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--use-pet", action="store_true", default=False)
    parser.add_argument("--use-torchmetrics", action="store_true", default=False)

    args = parser.parse_args()

    if args.use_pet:
        lc = pet.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=args.num_processes,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=1,
        )
        pet.elastic_launch(lc, entrypoint=main)(
            args.num_batches,
            args.batch_size,
            args.compute_interval,
            args.num_classes,
            args.use_torchmetrics,
        )
    else:
        main(
            args.num_batches,
            args.batch_size,
            args.compute_interval,
            args.num_classes,
            args.use_torchmetrics,
        )
