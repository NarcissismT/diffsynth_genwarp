#!/usr/bin/env python3
"""Exercise the production NCCL/rank-isolation path, optionally failing one rank."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--failure-token", default="")
    args = parser.parse_args()

    if args.expected_world_size < 1:
        raise ValueError("--expected-world-size must be positive")
    if args.fail_rank < -1:
        raise ValueError("--fail-rank must be -1 or a non-negative rank")
    if args.fail_rank >= args.expected_world_size:
        raise ValueError("--fail-rank must be smaller than the world size")
    if args.fail_rank >= 0 and not args.failure_token:
        raise ValueError("--failure-token is required for intentional failure")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "isolated worker must see exactly one CUDA device; "
            f"observed={torch.cuda.device_count()}"
        )
    if int(os.environ.get("LOCAL_RANK", "-1")) != 0:
        raise RuntimeError("isolated worker LOCAL_RANK must be logical rank 0")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("isolated worker current CUDA device must be 0")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices or "," in visible_devices or visible_devices == "-1":
        raise RuntimeError(
            "isolated worker must have exactly one CUDA_VISIBLE_DEVICES token; "
            f"observed={visible_devices!r}"
        )
    for name in ("LOCAL_WORLD_SIZE", "D2R_LAUNCH_LOCAL_WORLD_SIZE"):
        if int(os.environ.get(name, "-1")) != args.expected_world_size:
            raise RuntimeError(
                f"{name} must preserve the launch world size "
                f"{args.expected_world_size}; observed={os.environ.get(name)!r}"
            )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"world size mismatch: expected={args.expected_world_size}, got={world_size}"
        )
    if int(os.environ.get("RANK", "-1")) != rank:
        raise RuntimeError("environment RANK disagrees with the initialized process group")
    physical_local_rank = int(os.environ.get("D2R_PHYSICAL_LOCAL_RANK", "-1"))
    record = torch.tensor(
        [rank, physical_local_rank, torch.cuda.device_count(), torch.cuda.current_device()],
        dtype=torch.int64,
        device="cuda:0",
    )
    gathered = [torch.empty_like(record) for _ in range(world_size)]
    dist.all_gather(gathered, record)
    rows = [tensor.cpu().tolist() for tensor in gathered]
    if sorted(row[0] for row in rows) != list(range(world_size)):
        raise RuntimeError(f"global ranks are incomplete or duplicated: {rows}")
    if sorted(row[1] for row in rows) != list(range(world_size)):
        raise RuntimeError(f"physical local ranks are incomplete or duplicated: {rows}")
    if any(row[2:] != [1, 0] for row in rows):
        raise RuntimeError(f"one or more ranks were not mapped to logical cuda:0: {rows}")
    if rank == 0:
        print(
            "D2R_DDP_ISOLATION_OK "
            + json.dumps({"world_size": world_size, "ranks": rows}, sort_keys=True),
            flush=True,
        )

    dist.barrier()
    if rank == args.fail_rank:
        print(f"D2R_EXPECTED_RANK_FAILURE {args.failure_token}", flush=True)
        raise RuntimeError(f"intentional DDP failure: {args.failure_token}")
    if args.fail_rank >= 0:
        # The failing rank never enters this collective.  torchrun must notice
        # and terminate these peers instead of leaving the Slurm step hung.
        dist.barrier()
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
