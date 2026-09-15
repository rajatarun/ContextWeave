from __future__ import annotations

import logging
import sys
from pathlib import Path

src_root = str(Path(__file__).resolve().parents[1] / "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from shared.demo_logging import (
    DEMO_LEVEL,
    demo_for,
    demo_if,
    demo_step,
    demo_strategy_choice,
    resolve_log_level,
)


def test_resolve_demo_level():
    assert resolve_log_level("DEMO") == DEMO_LEVEL


def test_demo_if_and_demo_for_emit_messages(caplog):
    logger = logging.getLogger("demo-test")
    logger.setLevel(DEMO_LEVEL)

    with caplog.at_level(DEMO_LEVEL):
        demo_if(logger, "x > 0", True)
        demo_if(logger, "x > 0", False)
        demo_for(logger, "items", 1, 3)
        demo_step(logger, "test step")
        demo_strategy_choice(logger, "hybrid", 0.92)

    # This used to assert `len(message.split()) == 10` for all five records and
    # then look for phrases like "condition succeeded" and "else branch chosen".
    # None of that was ever true: the messages are 7, 7, 8, 4 and 3 words, and
    # those phrases appear nowhere in shared/demo_logging.py. The test was
    # written against an imagined API, and a word count is not a property worth
    # asserting anyway -- it fails on a rewording that changes nothing and
    # passes on a message that says the wrong thing.
    #
    # What matters is checked instead, in two parts.
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 5

    # 1. The rendered text a human reads: the right marker, and the values
    #    actually interpolated into it.
    assert messages[0] == "[CHECK] x > 0 → YES, proceeding"
    assert messages[1] == "[CHECK] x > 0 → NO, skipping"
    assert messages[2] == "[LOOP] items — item 1 of 3"
    assert messages[3] == "[STEP] test step"
    assert messages[4] == "[ROUTE] strategy=hybrid confidence=0.92"

    # 2. The structured fields, which are the real contract -- a log aggregator
    #    queries these, not the prose. `extra=` keys land as record attributes.
    assert (caplog.records[0].condition, caplog.records[0].branch,
            caplog.records[0].condition_result) == ("x > 0", "if", "satisfied")
    assert (caplog.records[1].condition, caplog.records[1].branch,
            caplog.records[1].condition_result) == ("x > 0", "else", "not satisfied")
    assert (caplog.records[2].iterator, caplog.records[2].iteration_index,
            caplog.records[2].iteration_total) == ("items", 1, 3)
    assert caplog.records[3].step == "test step"
    assert (caplog.records[4].strategy, caplog.records[4].strategy_confidence) == ("hybrid", 0.92)


def test_demo_for_without_a_total_omits_it(caplog):
    """The no-total branch of demo_for had no coverage at all.

    It renders a different message and sets iteration_total to None, so a
    regression there would have gone unnoticed.
    """
    logger = logging.getLogger("demo-test-no-total")
    logger.setLevel(DEMO_LEVEL)

    with caplog.at_level(DEMO_LEVEL):
        demo_for(logger, "items", 2)

    assert caplog.records[0].getMessage() == "[LOOP] items — item 2"
    assert caplog.records[0].iteration_total is None
