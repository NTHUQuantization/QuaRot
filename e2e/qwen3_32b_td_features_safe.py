"""Run TD32 S1/S2 extraction with safe power-of-two HadK restoration."""
from e2e.qwen3_32b_td_numeric import install_safe_collector

install_safe_collector()

from e2e.qwen3_32b_td_features import main


if __name__ == "__main__":
    main()
