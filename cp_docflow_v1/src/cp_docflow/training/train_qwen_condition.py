"""Stage 4: frozen-Qwen DPT/FPN adapter and gated feature fusion."""

from ._stage_entry import run_stage


def main() -> None:
    run_stage("qwen")


if __name__ == "__main__":
    main()

