import argparse

import pytest

from e2e.benchmark_downstream import PAPER_RESULTS, metric_value, parse_tasks


def test_metric_value_uses_paper_metric():
    value, metric = metric_value(
        "piqa", {"acc,none": 0.1, "acc_norm,none": 0.75})
    assert metric == "acc_norm,none"
    assert value == 0.75


def test_parse_tasks_rejects_unknown_task():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_tasks("piqa,not_a_task")


def test_paper_averages_match_reported_table():
    reference = sum(PAPER_RESULTS["reference"].values()) / 6
    int4 = sum(PAPER_RESULTS["int4"].values()) / 6
    assert reference == pytest.approx(0.69815)
    assert int4 == pytest.approx(0.6563833333333333)
