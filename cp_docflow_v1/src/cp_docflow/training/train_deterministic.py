"""Stage 1: deterministic CNN/FPN coarse/confidence and convex upsampling."""

from ._stage_entry import run_stage


def main() -> None:
    run_stage("coarse")


if __name__ == "__main__":
    main()

