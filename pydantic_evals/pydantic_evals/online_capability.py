"""Online evaluation capability for pydantic-ai agents.

Provides an `OnlineEvaluation` capability that attaches evaluators to agent runs,
dispatching them asynchronously in the background after each run completes.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from .evaluators.context import EvaluatorContext
from .evaluators.evaluator import Evaluator
from .online import (
    DEFAULT_CONFIG,
    EVALUATION_DISABLED,
    OnlineEvalConfig,
    OnlineEvaluator,
    SpanReference,
    dispatch_async,
    dispatch_evaluators,
    resolve_sample_rate_field,
    should_evaluate,
)
from .otel._context_subtree import context_subtree

__all__ = ('OnlineEvaluation',)


def _parse_traceparent(traceparent: str | None) -> SpanReference | None:
    """Parse a W3C traceparent string into a SpanReference.

    Format: ``00-{trace_id}-{span_id}-{flags}``
    Returns None if the string is missing, malformed, or has zero IDs.
    """
    if traceparent is None:
        return None
    parts = traceparent.split('-')
    if len(parts) != 4:
        return None
    trace_id, span_id = parts[1], parts[2]
    if not trace_id or trace_id == '0' * 32:
        return None
    if not span_id or span_id == '0' * 16:
        return None
    return SpanReference(trace_id=trace_id, span_id=span_id)


@dataclass
class OnlineEvaluation(AbstractCapability[AgentDepsT]):
    """Capability that runs online evaluators on agent run results.

    Dispatches evaluators asynchronously in the background after each run completes.
    Non-blocking - the agent run returns immediately and evaluators run concurrently.

    Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.online import OnlineEvalConfig, CallbackSink
    from pydantic_evals.online_capability import OnlineEvaluation

    class IsHelpful(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return len(str(ctx.output)) > 10

    agent = Agent(
        'test',
        capabilities=[
            OnlineEvaluation(
                evaluators=[IsHelpful()],
                config=OnlineEvalConfig(default_sink=CallbackSink(lambda r, f, c: print(r))),
            ),
        ],
    )
    ```
    """

    evaluators: Sequence[Evaluator | OnlineEvaluator]
    """Evaluators to run after each agent run."""

    config: OnlineEvalConfig | None = None
    """Optional config override. Defaults to the global ``DEFAULT_CONFIG``."""

    name: str | None = None
    """Optional name for the EvaluatorContext. Defaults to the agent run's ``run_id``."""

    _online_evaluators: list[OnlineEvaluator] = field(init=False, repr=False)
    _resolved_config: OnlineEvalConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._online_evaluators = [
            e if isinstance(e, OnlineEvaluator) else OnlineEvaluator(evaluator=e) for e in self.evaluators
        ]
        self._resolved_config = self.config if self.config is not None else DEFAULT_CONFIG

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        config = self._resolved_config

        if not config.enabled or EVALUATION_DISABLED.get():
            return await handler()

        sampled = [
            oe
            for oe in self._online_evaluators
            if should_evaluate(resolve_sample_rate_field(oe, config), config.enabled)
        ]
        if not sampled:
            return await handler()

        with context_subtree() as span_tree:
            t0 = time.perf_counter()
            result = await handler()
            duration = time.perf_counter() - t0

        usage = result.usage()
        metrics: dict[str, int | float] = {}
        if usage.requests:
            metrics['requests'] = usage.requests
        if usage.input_tokens:
            metrics['input_tokens'] = usage.input_tokens
        if usage.output_tokens:
            metrics['output_tokens'] = usage.output_tokens
        if usage.tool_calls:
            metrics['tool_calls'] = usage.tool_calls

        metadata: dict[str, Any] | None = None
        if config.metadata or ctx.metadata:
            metadata = {**(config.metadata or {}), **(ctx.metadata or {})}

        context = EvaluatorContext(
            name=self.name or ctx.run_id or 'agent',
            inputs=ctx.prompt,
            output=result.output,
            expected_output=None,
            metadata=metadata,
            duration=duration,
            _span_tree=span_tree,
            attributes={},
            metrics=metrics,
        )

        span_reference = _parse_traceparent(result._traceparent(required=False))  # pyright: ignore[reportPrivateUsage]

        dispatch_async(dispatch_evaluators(sampled, context, span_reference, config))

        return result
