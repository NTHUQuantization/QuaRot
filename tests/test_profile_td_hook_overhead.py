import pytest

from e2e.pard2 import DEFAULT_TARGET, DEFAULT_TOKENIZER
from e2e.profile_td_hook_overhead import (
    CASES,
    interleaved_order,
    parser,
    summarize,
)


def test_profile_parser_has_short_safe_defaults():
    args = parser().parse_args(["--output", "result.json"])
    assert args.target == str(DEFAULT_TARGET)
    assert args.tokenizer == str(DEFAULT_TOKENIZER)
    assert args.iterations == 5
    assert args.warmups == 2
    assert args.output == "result.json"
    assert args.context_length == 32
    assert args.max_preexisting_gib == 1.0


@pytest.mark.parametrize("flag", ("--iterations", "--warmups"))
def test_profile_parser_rejects_non_positive_counts(flag):
    with pytest.raises(SystemExit):
        parser().parse_args(["--output", "result.json", flag, "0"])


def test_interleaving_rotates_case_order_and_summary_reports_cv():
    assert interleaved_order(0) == CASES
    assert interleaved_order(1) == CASES[1:] + CASES[:1]
    assert interleaved_order(2) == CASES[2:] + CASES[:2]
    result = summarize([1.0, 2.0, 3.0])
    assert result["median_ms"] == 2.0
    assert result["mean_ms"] == 2.0
    assert result["cv_percent"] == 50.0
