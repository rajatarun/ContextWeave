from __future__ import annotations

import logging


DEMO_LEVEL = 15


def install_demo_level() -> None:
    """Register a DEMO logging level and logger.demo(...) helper."""
    if logging.getLevelName(DEMO_LEVEL) != "DEMO":
        logging.addLevelName(DEMO_LEVEL, "DEMO")

    if not hasattr(logging.Logger, "demo"):
        def demo(self: logging.Logger, message: str, *args, **kwargs) -> None:
            if self.isEnabledFor(DEMO_LEVEL):
                self._log(DEMO_LEVEL, message, args, **kwargs)

        setattr(logging.Logger, "demo", demo)


def resolve_log_level(level_name: str | None, default: str = "INFO") -> int:
    """Resolve LOG_LEVEL text to an integer logging level, including DEMO."""
    install_demo_level()
    candidate = (level_name or default).upper()
    if candidate == "DEMO":
        return DEMO_LEVEL
    return logging._nameToLevel.get(candidate, logging.INFO)


def demo_if(logger: logging.Logger, condition_desc: str, result: bool) -> None:
    if result:
        logger.demo(
            "[CHECK] %s → YES, proceeding",
            condition_desc,
            extra={"condition": condition_desc, "branch": "if", "condition_result": "satisfied"},
        )
        return
    logger.demo(
        "[CHECK] %s → NO, skipping",
        condition_desc,
        extra={"condition": condition_desc, "branch": "else", "condition_result": "not satisfied"},
    )


def demo_for(logger: logging.Logger, iterator_desc: str, index: int, total: int | None = None) -> None:
    if total is not None:
        logger.demo(
            "[LOOP] %s — item %d of %d",
            iterator_desc,
            index,
            total,
            extra={"iterator": iterator_desc, "iteration_index": index, "iteration_total": total},
        )
    else:
        logger.demo(
            "[LOOP] %s — item %d",
            iterator_desc,
            index,
            extra={"iterator": iterator_desc, "iteration_index": index, "iteration_total": None},
        )


def demo_step(logger: logging.Logger, step_desc: str) -> None:
    logger.demo(
        "[STEP] %s",
        step_desc,
        extra={"step": step_desc},
    )


def demo_strategy_choice(logger: logging.Logger, strategy: str, confidence: float) -> None:
    logger.demo(
        "[ROUTE] strategy=%s confidence=%.2f",
        strategy,
        confidence,
        extra={"strategy": strategy, "strategy_confidence": confidence},
    )
