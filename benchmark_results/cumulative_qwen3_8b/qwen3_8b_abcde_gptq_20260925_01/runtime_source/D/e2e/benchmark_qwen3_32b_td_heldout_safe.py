"""Run TD32 heldout evaluation with safe power-of-two HadK restoration."""
from e2e.qwen3_32b_td_numeric import install_safe_collector

install_safe_collector()

from e2e.benchmark_qwen3_32b_td_heldout import main


if __name__ == "__main__":
    main()
