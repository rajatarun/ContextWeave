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

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages[0].split()) == 10
    assert len(messages[1].split()) == 10
    assert len(messages[2].split()) == 10
    assert len(messages[3].split()) == 10
    assert len(messages[4].split()) == 10
    assert "condition succeeded" in messages[0]
    assert "else branch chosen" in messages[1]
    assert "for loop iterates item" in messages[2]
    assert "selected strategy recorded" in messages[4]
