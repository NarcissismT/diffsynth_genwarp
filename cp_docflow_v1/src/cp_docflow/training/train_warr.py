"""Stage 2: H/V structure pathway, WARR, and convex upsampling."""

from ._stage_entry import run_stage


def main() -> None:
    run_stage("warr")


if __name__ == "__main__":
    main()

