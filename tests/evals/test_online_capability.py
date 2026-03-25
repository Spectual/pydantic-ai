"""Tests for OnlineEvaluation capability — agent integration for online evaluators."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel
    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorContext, EvaluatorFailure
    from pydantic_evals.evaluators.evaluator import EvaluatorOutput
    from pydantic_evals.online import (
        OnlineEvalConfig,
        OnlineEvaluator,
        SpanReference,
        disable_evaluation,
        wait_for_evaluations,
    )
    from pydantic_evals.online_capability import (
        OnlineEvaluation,
        _parse_traceparent as _parse_traceparent,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')


if TYPE_CHECKING or imports_successful():

    @dataclass
    class AlwaysTrue(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return True

    @dataclass
    class OutputEquals(Evaluator):
        value: Any

        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return ctx.output == self.value

    @dataclass
    class FailingEvaluator(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            raise ValueError('Simulated evaluator failure')

    class Collector:
        def __init__(self) -> None:
            self.calls: list[
                tuple[list[EvaluationResult[Any]], list[EvaluatorFailure], EvaluatorContext[Any, Any, Any]]
            ] = []

        async def __call__(
            self,
            results: Sequence[EvaluationResult[Any]],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
        ) -> None:
            self.calls.append((list(results), list(failures), context))

    class SpanCaptureSink:
        def __init__(self) -> None:
            self.submissions: list[tuple[list[EvaluationResult[Any]], SpanReference | None]] = []

        async def submit(
            self,
            *,
            results: Sequence[EvaluationResult[Any]],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
            span_reference: SpanReference | None,
        ) -> None:
            self.submissions.append((list(results), span_reference))


@pytest.mark.anyio
async def test_basic_dispatch():
    """OnlineEvaluation dispatches evaluators after agent.run()."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    result = await agent.run('hello')
    await wait_for_evaluations()

    assert result.output == 'success (no tool calls)'
    assert len(collector.calls) == 1
    results, failures, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert len(failures) == 0
    assert ctx.output == 'success (no tool calls)'


@pytest.mark.anyio
async def test_evaluator_context_fields():
    """EvaluatorContext is populated with correct agent run data."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, metadata={'env': 'test'})

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(evaluators=[AlwaysTrue()], config=config, name='my-agent'),
        ],
    )

    result = await agent.run('what is 2+2?')
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.name == 'my-agent'
    assert ctx.inputs == 'what is 2+2?'
    assert ctx.output == result.output
    assert ctx.expected_output is None
    assert ctx.duration > 0
    assert ctx.metadata is not None
    assert ctx.metadata['env'] == 'test'


@pytest.mark.anyio
async def test_usage_metrics():
    """Token usage from the agent run appears in EvaluatorContext metrics."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.metrics.get('requests', 0) > 0


@pytest.mark.anyio
async def test_sampling_zero_rate():
    """Evaluators with sample_rate=0.0 are never dispatched."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(
                evaluators=[OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.0)],
                config=config,
            ),
        ],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_disable_evaluation():
    """disable_evaluation() context manager prevents dispatch."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    with disable_evaluation():
        await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_config_disabled():
    """Config with enabled=False prevents dispatch."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, enabled=False)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    result = await agent.run('hello')
    await wait_for_evaluations()

    assert result.output == 'success (no tool calls)'
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_multiple_evaluators():
    """Multiple evaluators all dispatch concurrently."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(
                evaluators=[AlwaysTrue(), OutputEquals(value='success (no tool calls)')],
                config=config,
            ),
        ],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 2
    assert all(len(r) == 1 for r, _, _ in collector.calls)
    assert all(r[0].value is True for r, _, _ in collector.calls)


@pytest.mark.anyio
async def test_failing_evaluator_does_not_crash_agent():
    """Evaluator exceptions don't crash the agent run."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(evaluators=[FailingEvaluator()], config=config),
        ],
    )

    result = await agent.run('hello')
    await wait_for_evaluations()

    assert result.output == 'success (no tool calls)'
    assert len(collector.calls) == 1
    results, failures, _ = collector.calls[0]
    assert len(results) == 0
    assert len(failures) == 1


@pytest.mark.anyio
async def test_gate_prevents_evaluation():
    """Gated evaluators only run when the gate returns True."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(
                evaluators=[OnlineEvaluator(evaluator=AlwaysTrue(), gate=lambda _: False)],
                config=config,
            ),
        ],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_gate_allows_evaluation():
    """Gated evaluators run when the gate returns True."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(
                evaluators=[OnlineEvaluator(evaluator=AlwaysTrue(), gate=lambda _: True)],
                config=config,
            ),
        ],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_metadata_merging():
    """Config metadata and run metadata are merged."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, metadata={'config_key': 'config_val'})

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    await agent.run('hello', metadata={'run_key': 'run_val'})
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.metadata is not None
    assert ctx.metadata['config_key'] == 'config_val'
    assert ctx.metadata['run_key'] == 'run_val'


@pytest.mark.anyio
async def test_name_defaults_to_run_id():
    """EvaluatorContext name defaults to run_id when no name is provided."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    # run_id is a UUID string, so it should be non-empty and not 'agent'
    assert ctx.name is not None
    assert len(ctx.name) > 0


def test_parse_traceparent_valid():
    """_parse_traceparent parses a valid W3C traceparent string."""
    tp = '00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01'
    ref = _parse_traceparent(tp)
    assert ref is not None
    assert ref.trace_id == '0af7651916cd43dd8448eb211c80319c'
    assert ref.span_id == 'b7ad6b7169203331'


def test_parse_traceparent_none():
    assert _parse_traceparent(None) is None


def test_parse_traceparent_malformed():
    assert _parse_traceparent('not-a-traceparent') is None


def test_parse_traceparent_zero_trace_id():
    tp = '00-00000000000000000000000000000000-b7ad6b7169203331-01'
    assert _parse_traceparent(tp) is None


def test_parse_traceparent_zero_span_id():
    tp = '00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01'
    assert _parse_traceparent(tp) is None


@pytest.mark.anyio
async def test_default_config_fallback():
    """OnlineEvaluation uses DEFAULT_CONFIG when no config is provided."""
    collector = Collector()

    # Create a config that we'll set as defaults - but we won't pass it to OnlineEvaluation
    from pydantic_evals.online import configure

    configure(default_sink=collector)
    try:
        agent = Agent(
            TestModel(),
            capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()])],
        )

        await agent.run('hello')
        await wait_for_evaluations()

        assert len(collector.calls) == 1
    finally:
        configure(default_sink=None)
