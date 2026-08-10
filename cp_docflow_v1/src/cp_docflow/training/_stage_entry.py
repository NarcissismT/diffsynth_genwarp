from __future__ import annotations

import argparse

from ..config import load_config
from ..train_full import canonical_stage_name, train


def run_stage(expected_stage: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Train canonical DocGrid-Flow stage: {expected_stage}"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parent-checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    actual = canonical_stage_name(config.get("train", {}).get("stage", "full_page"))
    if actual != expected_stage:
        parser.error(
            f"config train.stage={actual!r}; this entry requires {expected_stage!r}"
        )
    train(
        config,
        resume=args.resume,
        output_dir_override=args.output_dir,
        seed_override=args.seed,
        parent_checkpoint_override=args.parent_checkpoint,
    )
