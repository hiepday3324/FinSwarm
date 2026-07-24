from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import pyarrow.parquet as pq
import yaml

from puppy.chat import (
    ChatOpenAICompatible,
    SanitizedChatBackendError,
    StructuredCompletionResult,
)
from puppy.common.schemas import CausalRunContext, MarketStep, NewsEvent, PriceBar
from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.normalization import (
    FILING_SCHEMA,
    NEWS_SCHEMA,
    PRICE_SCHEMA,
    write_records,
)
from puppy.fusion.text_encoder import TextStreamEncoder
from puppy.embedding import SentenceTransformerMemoryEmb, build_memory_embedding
from puppy.graph.graph_dataset import CANONICAL_FEATURE_NAMES
from puppy.swarm.causal_agent_factory import (
    CAUSAL_PROMPT_VERSION,
    _build_backend_from_environment,
    build_causal_agent_bundle,
)
from puppy.swarm.causal_orchestrator import CausalOrchestrator
from puppy.swarm.legacy_finmem_adapter import (
    ADAPTER_VERSION,
    CausalDecisionFailure,
    _stable_hash,
)
from puppy.swarm.psychology import build_psychology_state
from puppy.swarm.sector_agent import SectorAgent
from puppy.reflection import CausalDecisionResponse
from puppy.training.artifact_generator import generate_training_artifacts
from puppy.training.artifact_validation import validate_training_artifacts
from puppy.training.config import TrainingConfig
from puppy.training.decision_journal import JOURNAL_DIRECTORY
from puppy.training.pipeline import verify_causal_inputs


ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc


class _EmptyBrain:
    def __init__(self, namespace: str, embedding_config: dict | None = None) -> None:
        self.namespace = namespace
        self.emb_config = dict(embedding_config or {})
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

    def add_memory_short(self, *args, **kwargs) -> None:
        return None

    def add_memory_mid(self, *args, **kwargs) -> None:
        return None

    def add_memory_long(self, *args, **kwargs) -> None:
        return None

    def add_memory_reflection(self, *args, **kwargs) -> None:
        return None


def _memory_factory(config: dict, namespace: str) -> _EmptyBrain:
    return _EmptyBrain(
        namespace,
        config["agent"]["agent_1"]["embedding"]["detail"],
    )


def _memory_index(prompt: str, value: int) -> int:
    return value if f"{value}." in prompt else -1


def _valid_response(prompt: str, decision: str = "BUY") -> str:
    return json.dumps(
        {
            "investment_decision": decision,
            "summary_reason": "Synthetic causal parser fixture.",
            "short_memory_index": [
                {"memory_index": _memory_index(prompt, 100000)}
            ],
            "middle_memory_index": [
                {"memory_index": _memory_index(prompt, 200000)}
            ],
            "long_memory_index": [
                {"memory_index": _memory_index(prompt, 300000)}
            ],
            "reflection_memory_index": [
                {"memory_index": _memory_index(prompt, 400000)}
            ],
        }
    )


class _FakeBackend:
    model = "mock-openai-model"
    provider = "injected_mock"
    service_provider = "injected_mock"
    base_url_identity = "mock-base-url-v1"
    is_mock = True

    def __init__(self, responses: list[str] | None = None, identity: str = "mock-backend-v1") -> None:
        self.identity = identity
        self.responses = list(responses or [])
        self.requests: list[str] = []
        self.api_call_count = 0
        self.retry_count = 0
        self.last_response_hash: str | None = None
        self.last_returned_model: str | None = self.model

    @property
    def public_config(self) -> dict:
        return {
            "provider": self.provider,
            "interface": self.provider,
            "base_url": "mock://offline/chat/completions",
            "model": self.model,
            "temperature": 0,
            "backend_identity": self.identity,
        }

    def guardrail_endpoint(self):
        def endpoint(prompt: str, **kwargs) -> str:
            self.requests.append(str(prompt))
            self.api_call_count += 1
            response = self.responses.pop(0) if self.responses else _valid_response(str(prompt))
            self.last_response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
            return response

        return endpoint


class _InterruptingBackend(_FakeBackend):
    def __init__(self, successful_calls_before_failure: int) -> None:
        super().__init__()
        self.successful_calls_before_failure = successful_calls_before_failure
        self.successful_calls = 0

    def guardrail_endpoint(self):
        def endpoint(prompt: str, **kwargs) -> str:
            self.requests.append(str(prompt))
            self.api_call_count += 1
            if self.successful_calls >= self.successful_calls_before_failure:
                raise RuntimeError("synthetic interrupted rollout")
            self.successful_calls += 1
            response = _valid_response(str(prompt))
            self.last_response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
            return response

        return endpoint


class _FakeParsedCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeOpenAIClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = _FakeParsedCompletions(outcomes)
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )


def _gemini_completion(
    parsed: object | None,
    *,
    content: str | None = None,
    finish_reason: str = "stop",
    returned_model: str = "gemini-3.5-flash-2026-01",
):
    return SimpleNamespace(
        model=returned_model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(parsed=parsed, content=content),
            )
        ],
    )


def _gemini_backend(outcomes: list[object]) -> tuple[ChatOpenAICompatible, _FakeOpenAIClient]:
    client = _FakeOpenAIClient(outcomes)
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    backend = ChatOpenAICompatible(
        end_point=base_url + "chat/completions",
        base_url=base_url,
        model="gemini-3.5-flash",
        api_key="synthetic-gemini-key",
        require_api_key=True,
        provider="openai_compatible",
        service_provider="google_gemini",
        other_parameters={"temperature": 0},
        openai_client=client,
    )
    return backend, client


def _parsed_decision(
    decision: str,
    *,
    short_ids: list[int] | None = None,
) -> CausalDecisionResponse:
    return CausalDecisionResponse(
        investment_decision=decision,
        summary_reason=f"Structured {decision} fixture.",
        short_memory_index=[
            {"memory_index": value} for value in (short_ids or [])
        ],
        middle_memory_index=[],
        long_memory_index=[],
        reflection_memory_index=[],
    )


class _FakeTransformerBackend:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_texts(self, texts, *, batch_size: int = 32) -> np.ndarray:
        values = list(texts)
        self.calls.append(values)
        matrix = np.zeros((len(values), self.dimension), dtype=np.float32)
        for index, text in enumerate(values):
            matrix[index, index % self.dimension] = 1.0
            matrix[index, -1] = float(len(text) + 1)
        return matrix


def _local_memory_embedding() -> SentenceTransformerMemoryEmb:
    return SentenceTransformerMemoryEmb(
        _FakeTransformerBackend.model_name,
        transformer_backend=_FakeTransformerBackend(),
    )


def _bundle_with_real_memory() -> tuple[dict, _FakeBackend, SentenceTransformerMemoryEmb]:
    backend = _FakeBackend()
    memory_embedding = _local_memory_embedding()
    bundle = build_causal_agent_bundle(
        str(ROOT / "config/swarm/experiment_v1.yaml"),
        chat_backend=backend,
        memory_embedding_backend=memory_embedding,
        encoder=_encoder(),
        environ={
            "OPENAI_API_KEY": "ollama",
            "FINMEM_MEMORY_EMBEDDING_PROVIDER": "sentence_transformers",
            "FINMEM_MEMORY_EMBEDDING_MODEL": _FakeTransformerBackend.model_name,
        },
        sleep_func=lambda _: None,
    )
    return bundle, backend, memory_embedding


def _encoder() -> TextStreamEncoder:
    def embed(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return np.asarray([(byte / 127.5) - 1.0 for byte in digest[:8]], dtype=np.float32)

    return TextStreamEncoder(
        embedding_func=embed,
        embedding_model="mock-frozen-semantic-encoder",
    )


def _bundle(backend: _FakeBackend | None = None) -> tuple[dict, _FakeBackend]:
    backend = backend or _FakeBackend()
    bundle = build_causal_agent_bundle(
        str(ROOT / "config/swarm/experiment_v1.yaml"),
        chat_backend=backend,
        memory_factory=_memory_factory,
        encoder=_encoder(),
        environ={},
        sleep_func=lambda _: None,
    )
    return bundle, backend


def _market_step(symbols: tuple[str, ...] = ("TSLA",)) -> MarketStep:
    date = dt.date(2026, 1, 5)
    cutoff = dt.datetime(2026, 1, 5, 21, 10, tzinfo=UTC)
    return MarketStep(
        date=date,
        symbols=list(symbols),
        prices={
            symbol: PriceBar(
                symbol=symbol,
                date=date,
                open=100.0 + index,
                high=102.0 + index,
                low=99.0 + index,
                close=101.0 + index,
                volume=1000,
            )
            for index, symbol in enumerate(symbols)
        },
        news={symbol: [] for symbol in symbols},
        filings={symbol: [] for symbol in symbols},
        as_of=cutoff,
    )


def _run_context(step: MarketStep, prompt_version: str = CAUSAL_PROMPT_VERSION) -> CausalRunContext:
    return CausalRunContext(
        run_id="factory-test",
        config_path="config/swarm/experiment_v1.yaml",
        current_date=step.date,
        current_cutoff=step.as_of,
        bootstrap=False,
        prompt_version=prompt_version,
        encoder_version="mock-frozen-semantic-encoder",
        adapter_version=ADAPTER_VERSION,
        git_commit="test",
        metadata={"generation_parameters": {"temperature": 0.0}},
    )


def _decide(bundle: dict, symbol: str, step: MarketStep, run: CausalRunContext):
    agent = bundle["agents"][symbol]
    agent.causal_adapter.advance_memory_clock(step.date, run)
    return CausalOrchestrator(bundle["encoder"]).decide(
        agent=agent,
        market_step=step,
        run_context=run,
        expected_outcome_date=step.date + dt.timedelta(days=1),
        split="train",
        psychology_state=build_psychology_state(None),
    )


class CausalAgentFactoryTest(unittest.TestCase):
    def test_local_memory_embedding_is_float32_384_and_never_builds_openai(self) -> None:
        transformer = _FakeTransformerBackend()
        with patch("puppy.embedding.OpenAIEmbeddings") as openai_embeddings:
            embedding = build_memory_embedding(
                {
                    "provider": "sentence_transformers",
                    "embedding_model": transformer.model_name,
                },
                transformer_backend=transformer,
            )
            values = embedding(["first", "second"])
        openai_embeddings.assert_not_called()
        self.assertEqual(values.shape, (2, 384))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(embedding.get_embedding_dimension(), 384)
        with self.assertRaisesRegex(ValueError, "qwen3:4b"):
            SentenceTransformerMemoryEmb(
                "qwen3:4b",
                transformer_backend=transformer,
            )

    def test_factory_keeps_memory_stores_and_faiss_indexes_separate(self) -> None:
        bundle, _, memory_embedding = _bundle_with_real_memory()
        brains = [agent.llm_agent.brain for agent in bundle["agents"].values()]
        stores = [brain.short_term_memory for brain in brains]
        self.assertEqual(len({id(brain) for brain in brains}), 3)
        self.assertEqual(len({id(store) for store in stores}), 3)
        self.assertEqual({id(store.emb_func) for store in stores}, {id(memory_embedding)})
        self.assertTrue(
            all(
                brain.emb_config["provider"] == "sentence_transformers"
                for brain in brains
            )
        )
        indexes = []
        for symbol, store in zip(("TSLA", "NVDA", "AAPL"), stores):
            store.add_new_symbol(symbol)
            index = store.universe[symbol]["index"]
            self.assertEqual(index.d, memory_embedding.get_embedding_dimension())
            indexes.append(index)
        self.assertEqual(len({id(index) for index in indexes}), 3)

    def test_ollama_key_does_not_route_memory_to_openai_and_ingest_reaches_llm(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "ollama"}, clear=False):
            with patch("puppy.embedding.OpenAIEmbeddings") as openai_embeddings:
                bundle, backend, memory_embedding = _bundle_with_real_memory()
                step = _market_step()
                step.news["TSLA"] = [
                    NewsEvent(
                        event_id="offline-news-1",
                        symbol="TSLA",
                        published_at=step.as_of - dt.timedelta(hours=2),
                        available_at=step.as_of - dt.timedelta(hours=2),
                        text="Causally available local embedding fixture.",
                    )
                ]
                run = _run_context(step)
                agent = bundle["agents"]["TSLA"]
                agent.causal_adapter.advance_memory_clock(step.date, run)
                refs = agent.causal_adapter.ingest_market_events(step, run)
                result = CausalOrchestrator(bundle["encoder"]).decide(
                    agent=agent,
                    market_step=step,
                    run_context=run,
                    expected_outcome_date=step.date + dt.timedelta(days=1),
                    split="train",
                    psychology_state=build_psychology_state(None),
                )
        openai_embeddings.assert_not_called()
        self.assertEqual(len(refs), 1)
        self.assertEqual(len(backend.requests), 1)
        self.assertTrue(result.output.parse_ok)
        self.assertEqual(result.output.signal.symbol, "TSLA")
        index = agent.llm_agent.brain.short_term_memory.universe["TSLA"]["index"]
        self.assertEqual(index.d, 384)
        self.assertGreater(len(memory_embedding.transformer_backend.calls), 0)

    def test_factory_builds_three_isolated_nonbootstrap_agents(self) -> None:
        bundle, backend = _bundle()
        self.assertEqual(tuple(bundle["agents"]), ("TSLA", "NVDA", "AAPL"))
        self.assertTrue(all(isinstance(agent, SectorAgent) for agent in bundle["agents"].values()))
        self.assertTrue(all(agent.causal_adapter.bootstrap is False for agent in bundle["agents"].values()))
        self.assertEqual(len({agent.memory_namespace for agent in bundle["agents"].values()}), 3)
        self.assertEqual(len({id(agent.llm_agent.brain) for agent in bundle["agents"].values()}), 3)
        self.assertEqual({id(agent.llm_agent.chat_backend) for agent in bundle["agents"].values()}, {id(backend)})
        self.assertIsInstance(bundle["encoder"], TextStreamEncoder)
        self.assertEqual(bundle["prompt_version"], CAUSAL_PROMPT_VERSION)
        for agent in bundle["agents"].values():
            self.assertEqual(
                agent.llm_agent.brain.emb_config["provider"],
                "sentence_transformers",
            )
            self.assertEqual(
                agent.llm_agent.brain.emb_config["embedding_model"],
                "sentence-transformers/all-MiniLM-L6-v2",
            )

    def test_missing_credential_fails_before_backend_construction(self) -> None:
        with patch("puppy.swarm.causal_agent_factory.ChatOpenAICompatible") as constructor:
            with self.assertRaisesRegex(RuntimeError, "FINMEM_LLM_API_KEY"):
                build_causal_agent_bundle(
                    str(ROOT / "config/swarm/experiment_v1.yaml"),
                    memory_factory=_memory_factory,
                    encoder=_encoder(),
                    environ={},
                )
            constructor.assert_not_called()

    def test_openai_key_fallback_is_not_serialized(self) -> None:
        secret = "unit-test-secret-never-log"
        bundle = build_causal_agent_bundle(
            str(ROOT / "config/swarm/experiment_v1.yaml"),
            memory_factory=_memory_factory,
            encoder=_encoder(),
            environ={
                "OPENAI_API_KEY": secret,
                "FINMEM_LLM_MODEL": "unit-model",
                "FINMEM_LLM_REQUESTS_PER_MINUTE": "30",
            },
            sleep_func=lambda _: None,
        )
        serialized = json.dumps(bundle["provenance"], sort_keys=True)
        serialized += json.dumps(
            [agent.llm_agent.chat_config_save for agent in bundle["agents"].values()],
            sort_keys=True,
        )
        self.assertNotIn(secret, serialized)
        self.assertNotIn("api_key", serialized.lower())

    def test_mocked_one_agent_decision_uses_strict_production_path(self) -> None:
        bundle, backend = _bundle()
        step = _market_step()
        result = _decide(bundle, "TSLA", step, _run_context(step))
        self.assertIn(str(result.output.signal.decision), {"buy", "hold", "sell"})
        self.assertIn(result.output.signal.action_signal, {-1, 0, 1})
        self.assertTrue(result.output.parse_ok)
        self.assertTrue(result.output.decision_id)
        self.assertTrue(result.output.context_hash)
        self.assertEqual(result.output.prompt_version, CAUSAL_PROMPT_VERSION)
        self.assertFalse(bundle["agents"]["TSLA"].causal_adapter.bootstrap)
        request = backend.requests[0].lower()
        self.assertNotIn("future_record", request)
        self.assertNotIn("direction_label", request)
        self.assertNotIn("outcome_id", request)
        self.assertTrue(result.output.metadata["synthetic"])

    def test_invalid_response_never_defaults_to_hold(self) -> None:
        backend = _FakeBackend(responses=["not-json", "still-not-json"])
        bundle, _ = _bundle(backend)
        step = _market_step()
        with self.assertRaisesRegex(CausalDecisionFailure, "response_hash"):
            _decide(bundle, "TSLA", step, _run_context(step))
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(bundle["agents"]["TSLA"].causal_adapter.decision_stats["failed_decision_count"], 1)

    def test_context_cache_and_cache_identity_contract(self) -> None:
        bundle, backend = _bundle()
        step = _market_step()
        run = _run_context(step)
        agent = bundle["agents"]["TSLA"]
        agent.causal_adapter.advance_memory_clock(step.date, run)
        orchestrator = CausalOrchestrator(bundle["encoder"])
        first = orchestrator.decide(
            agent=agent,
            market_step=step,
            run_context=run,
            expected_outcome_date=step.date + dt.timedelta(days=1),
            split="train",
            psychology_state=build_psychology_state(None),
        )
        second = orchestrator.decide(
            agent=agent,
            market_step=step,
            run_context=run,
            expected_outcome_date=step.date + dt.timedelta(days=1),
            split="train",
            psychology_state=build_psychology_state(None),
        )
        self.assertEqual(len(backend.requests), 1)
        self.assertFalse(first.output.metadata["cache_hit"])
        self.assertTrue(second.output.metadata["cache_hit"])
        self.assertEqual(first.output.metadata["cache_key"], second.output.metadata["cache_key"])
        self.assertEqual(first.output.metadata["backend_identity"], backend.identity)
        self.assertEqual(
            first.output.metadata["cache_key"],
            _stable_hash(
                backend.service_provider,
                backend.provider,
                backend.base_url_identity,
                backend.model,
                agent.llm_agent.causal_schema_version,
                run.prompt_version,
                "TSLA",
                first.context.context_hash,
                {"temperature": 0.0, "reasoning_effort": "low"},
            ),
        )

    def test_mocked_three_agent_day_has_no_mutable_cross_contamination(self) -> None:
        bundle, backend = _bundle()
        step = _market_step(("TSLA", "NVDA", "AAPL"))
        run = _run_context(step)
        outputs = []
        for symbol in ("TSLA", "NVDA", "AAPL"):
            outputs.append(_decide(bundle, symbol, step, run).output)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(backend.requests), 3)
        self.assertEqual({output.signal.symbol for output in outputs}, {"TSLA", "NVDA", "AAPL"})
        self.assertTrue(all(output.parse_ok for output in outputs))
        self.assertEqual(len({id(agent.causal_adapter._records) for agent in bundle["agents"].values()}), 3)


class GeminiStructuredOutputTest(unittest.TestCase):
    def test_environment_key_precedence_and_official_gemini_base_url(self) -> None:
        fake_backend = _FakeBackend()
        with patch(
            "puppy.swarm.causal_agent_factory.ChatOpenAICompatible",
            return_value=fake_backend,
        ) as constructor:
            _build_backend_from_environment(
                {
                    "FINMEM_LLM_PROVIDER": "openai_compatible",
                    "FINMEM_LLM_BASE_URL": (
                        "https://generativelanguage.googleapis.com/v1beta/openai/"
                    ),
                    "FINMEM_LLM_MODEL": "gemini-3.5-flash",
                    "FINMEM_LLM_API_KEY": "preferred-finmem-key",
                    "OPENAI_API_KEY": "fallback-key",
                },
                sleep_func=lambda _: None,
            )
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "preferred-finmem-key")
        self.assertEqual(
            kwargs["base_url"],
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.assertEqual(kwargs["model"], "gemini-3.5-flash")
        self.assertEqual(kwargs["service_provider"], "google_gemini")
        self.assertNotIn("preferred-finmem-key", json.dumps(fake_backend.public_config))

    def test_parsed_schema_reaches_agent_output_without_free_form_parsing(self) -> None:
        backend, client = _gemini_backend(
            [_gemini_completion(_parsed_decision("BUY"), content="SELL free-form")]
        )
        bundle, _ = _bundle(backend)  # type: ignore[arg-type]
        step = _market_step()
        result = _decide(bundle, "TSLA", step, _run_context(step))
        call = client.completions.calls[0]
        self.assertIs(call["response_format"], CausalDecisionResponse)
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(call["reasoning_effort"], "low")
        self.assertEqual(str(result.output.signal.decision), "buy")
        self.assertEqual(result.output.signal.action_signal, 1)
        provenance = result.output.metadata["causal_provenance"]
        self.assertEqual(provenance["provider"], "google_gemini")
        self.assertEqual(provenance["interface"], "openai_compatible")
        self.assertEqual(provenance["requested_model"], "gemini-3.5-flash")
        self.assertEqual(provenance["returned_model"], "gemini-3.5-flash-2026-01")
        self.assertEqual(provenance["schema_version"], "causal-decision-response-v1")

    def test_buy_hold_sell_map_to_action_signals(self) -> None:
        for decision, action in (("BUY", 1), ("HOLD", 0), ("SELL", -1)):
            with self.subTest(decision=decision):
                backend, _ = _gemini_backend(
                    [_gemini_completion(_parsed_decision(decision))]
                )
                bundle, _ = _bundle(backend)  # type: ignore[arg-type]
                step = _market_step()
                result = _decide(bundle, "TSLA", step, _run_context(step))
                self.assertEqual(result.output.signal.action_signal, action)
                self.assertEqual(str(result.output.signal.decision), decision.lower())

    def test_malformed_first_response_retries_same_schema_then_succeeds(self) -> None:
        backend, client = _gemini_backend(
            [
                _gemini_completion(None, content="BUY free-form prose"),
                _gemini_completion(_parsed_decision("HOLD")),
            ]
        )
        bundle, _ = _bundle(backend)  # type: ignore[arg-type]
        step = _market_step()
        result = _decide(bundle, "TSLA", step, _run_context(step))
        self.assertEqual(len(client.completions.calls), 2)
        self.assertTrue(
            all(
                call["response_format"] is CausalDecisionResponse
                for call in client.completions.calls
            )
        )
        self.assertEqual(str(result.output.signal.decision), "hold")
        self.assertEqual(
            result.output.metadata["causal_provenance"]["retry_count"], 1
        )

    def test_invalid_decision_and_two_free_form_responses_never_become_hold(self) -> None:
        invalid = {
            "investment_decision": "MAYBE",
            "summary_reason": "invalid",
            "short_memory_index": [],
            "middle_memory_index": [],
            "long_memory_index": [],
            "reflection_memory_index": [],
        }
        for outcomes, category in (
            (
                [_gemini_completion(invalid), _gemini_completion(invalid)],
                "invalid_decision_enum",
            ),
            (
                [
                    _gemini_completion(None, content="BUY"),
                    _gemini_completion(None, content="SELL"),
                ],
                "structured_parse_failed",
            ),
        ):
            with self.subTest(category=category):
                backend, _ = _gemini_backend(outcomes)
                bundle, _ = _bundle(backend)  # type: ignore[arg-type]
                step = _market_step()
                with self.assertRaises(CausalDecisionFailure) as raised:
                    _decide(bundle, "TSLA", step, _run_context(step))
                self.assertEqual(raised.exception.category, category)
                self.assertNotIn("synthetic-gemini-key", str(raised.exception))

    def test_invalid_memory_ids_are_rejected(self) -> None:
        invalid = _parsed_decision("BUY", short_ids=[999999])
        backend, _ = _gemini_backend(
            [_gemini_completion(invalid), _gemini_completion(invalid)]
        )
        bundle, _ = _bundle(backend)  # type: ignore[arg-type]
        step = _market_step()
        with self.assertRaises(CausalDecisionFailure) as raised:
            _decide(bundle, "TSLA", step, _run_context(step))
        self.assertEqual(raised.exception.category, "invalid_memory_id")

    def test_gemini_and_ollama_cache_identities_do_not_collide(self) -> None:
        gemini, _ = _gemini_backend([])
        ollama = ChatOpenAICompatible(
            end_point="http://localhost:11434/v1/chat/completions",
            base_url="http://localhost:11434/v1/",
            model="qwen3:4b",
            api_key="ollama",
            provider="openai_compatible",
            service_provider="ollama",
            openai_client=_FakeOpenAIClient([]),
        )
        self.assertNotEqual(gemini.base_url_identity, ollama.base_url_identity)
        self.assertNotEqual(gemini.identity, ollama.identity)


class ChatRetryTest(unittest.TestCase):
    @staticmethod
    def _response(status: int, content: str = '{"status":"ok"}') -> httpx.Response:
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        payload = {"choices": [{"message": {"content": content}}]}
        return httpx.Response(status, request=request, json=payload)

    def test_transient_retry_is_bounded_and_temperature_is_zero(self) -> None:
        responses = [self._response(429), self._response(500), self._response(200)]
        sleeps: list[float] = []
        payloads: list[dict] = []

        def post(*args, **kwargs):
            payloads.append(kwargs["json"])
            return responses.pop(0)

        backend = ChatOpenAICompatible(
            "https://example.test/v1/chat/completions",
            model="unit-model",
            other_parameters={"temperature": 0},
            api_key="fake-test-key",
            require_api_key=True,
            max_retries=2,
            sleep_func=sleeps.append,
            http_post=post,
        )
        self.assertEqual(backend.guardrail_endpoint()("fixture"), '{"status":"ok"}')
        self.assertEqual(backend.api_call_count, 3)
        self.assertEqual(backend.retry_count, 2)
        self.assertEqual(sleeps, [0.5, 1.0])
        self.assertTrue(all(payload["temperature"] == 0 for payload in payloads))

    def test_authentication_error_is_not_retried_or_leaked(self) -> None:
        calls = 0

        def post(*args, **kwargs):
            nonlocal calls
            calls += 1
            return self._response(401)

        backend = ChatOpenAICompatible(
            "https://example.test/v1/chat/completions",
            model="unit-model",
            api_key="fake-test-key-never-print",
            require_api_key=True,
            max_retries=5,
            sleep_func=lambda _: None,
            http_post=post,
        )
        with self.assertRaises(SanitizedChatBackendError) as raised:
            backend.guardrail_endpoint()("fixture")
        self.assertEqual(calls, 1)
        self.assertNotIn("fake-test-key", str(raised.exception))


class MockArtifactRolloutTest(unittest.TestCase):
    def test_mock_rollout_is_temporally_valid_but_never_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, dates = _synthetic_repository(root, 6)
            bundle = build_causal_agent_bundle(
                str(config_path),
                chat_backend=_FakeBackend(),
                memory_factory=_memory_factory,
                encoder=_encoder(),
                environ={},
                sleep_func=lambda _: None,
            )
            result = generate_training_artifacts(
                str(config_path),
                start=dates[0].isoformat(),
                end=dates[4].isoformat(),
                output_root=root / "data/training",
                causal_agents=bundle["agents"],
                encoder=bundle["encoder"],
                prompt_version=bundle["prompt_version"],
                generation_provenance=bundle["provenance"],
            )
            artifact = root / "data/training/finswarm_v1"
            manifest = json.loads((artifact / "data_manifest.json").read_text())
            self.assertEqual(result["temporal_gate"], "PASS")
            self.assertEqual(manifest["action_source"], "mock_causal_sector_agent")
            self.assertFalse(manifest["external_llm_used"])
            self.assertFalse(manifest["final_training_eligible"])
            self.assertTrue(manifest["generation_provenance"]["synthetic"])
            self.assertEqual(manifest["decision_quality"]["total_decisions"], 15)
            report = validate_training_artifacts(
                str(config_path), "finswarm_v1", require_final=False
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(
                tuple(json.loads((artifact / "feature_schema.json").read_text())["graph_feature_names"]),
                CANONICAL_FEATURE_NAMES,
            )
            with self.assertRaisesRegex(ValueError, "final_action_contract"):
                validate_training_artifacts(str(config_path), "finswarm_v1")


class PersistentPartialResumeTest(unittest.TestCase):
    def _bundle_for(self, config_path: Path, backend: _FakeBackend) -> dict:
        return build_causal_agent_bundle(
            str(config_path),
            chat_backend=backend,
            memory_factory=_memory_factory,
            encoder=_encoder(),
            environ={},
            sleep_func=lambda _: None,
        )

    @staticmethod
    def _run(
        config_path: Path,
        dates: list[dt.date],
        output_root: Path,
        bundle: dict,
        *,
        resume: bool = False,
        force_llm: bool = False,
    ) -> dict:
        return generate_training_artifacts(
            str(config_path),
            start=dates[0].isoformat(),
            end=dates[3].isoformat(),
            output_root=output_root,
            causal_agents=bundle["agents"],
            encoder=bundle["encoder"],
            prompt_version=bundle["prompt_version"],
            generation_provenance=bundle["provenance"],
            resume=resume,
            force_llm=force_llm,
        )

    def _assert_artifacts_equal(self, expected: Path, actual: Path) -> None:
        for name in (
            "text_samples.parquet",
            "agent_outputs.parquet",
            "observed_features.parquet",
            "outcomes.parquet",
            "pending_decisions.parquet",
            "reflection_updates.parquet",
        ):
            self.assertTrue(
                pq.read_table(expected / name).equals(pq.read_table(actual / name)),
                name,
            )
        with np.load(expected / "graph_samples.npz", allow_pickle=False) as left:
            with np.load(actual / "graph_samples.npz", allow_pickle=False) as right:
                self.assertEqual(left.files, right.files)
                for field in left.files:
                    self.assertTrue(np.array_equal(left[field], right[field]), field)
        for name in (
            "data_manifest.json",
            "feature_schema.json",
            "temporal_quality.json",
            "rollout_manifest.json",
            "state/checkpoint.json",
        ):
            self.assertEqual(
                json.loads((expected / name).read_text(encoding="utf-8")),
                json.loads((actual / name).read_text(encoding="utf-8")),
                name,
            )

    @staticmethod
    def _agent_state(bundle: dict) -> str:
        state = {}
        for symbol, agent in bundle["agents"].items():
            adapter = agent.causal_adapter
            state[symbol] = {
                "records": {
                    layer: [record.__dict__ for record in records]
                    for layer, records in adapter._records.items()
                },
                "seen_source_ids": sorted(adapter._seen_source_ids),
                "advanced_dates": sorted(adapter._advanced_dates),
                "decay_count": adapter.decay_count,
                "feedback_count": adapter.feedback_count,
                "decision_stats": adapter.decision_stats,
                "brain_step_count": agent.llm_agent.brain.step_count,
                "outputs": {
                    date.isoformat(): output.model_dump(mode="json")
                    for date, output in sorted(agent.outputs.items())
                },
            }
        return json.dumps(state, sort_keys=True, default=str)

    def _assert_resume_after(
        self,
        successful_decisions: int,
        *,
        expected_first_symbol: str,
        expected_completed_symbols: list[str],
        expected_last_completed_index: int | None,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, dates = _synthetic_repository(root, 5)
            baseline_root = root / "baseline"
            resumed_root = root / "resumed"

            baseline_backend = _FakeBackend()
            baseline_bundle = self._bundle_for(config_path, baseline_backend)
            baseline_result = self._run(
                config_path, dates, baseline_root, baseline_bundle
            )

            interrupted_backend = _InterruptingBackend(successful_decisions)
            interrupted_bundle = self._bundle_for(config_path, interrupted_backend)
            with self.assertRaisesRegex(
                CausalDecisionFailure, "Causal decision failed"
            ):
                self._run(config_path, dates, resumed_root, interrupted_bundle)

            partial = resumed_root / "finswarm_v1"
            journal_dir = partial / "state" / JOURNAL_DIRECTORY
            self.assertEqual(len(list(journal_dir.glob("*.json"))), successful_decisions)
            checkpoint = json.loads(
                (partial / "state/checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint["completed_symbols_for_current_date"],
                expected_completed_symbols,
            )
            expected_last_completed = (
                dates[expected_last_completed_index].isoformat()
                if expected_last_completed_index is not None
                else None
            )
            self.assertEqual(
                checkpoint["last_fully_completed_date"], expected_last_completed
            )

            resumed_backend = _FakeBackend()
            resumed_bundle = self._bundle_for(config_path, resumed_backend)
            resumed_result = self._run(
                config_path,
                dates,
                resumed_root,
                resumed_bundle,
                resume=True,
                force_llm=True,
            )
            total_decisions = 12
            self.assertEqual(
                len(resumed_backend.requests), total_decisions - successful_decisions
            )
            self.assertIn(expected_first_symbol, resumed_backend.requests[0])
            self.assertEqual(len(list(journal_dir.glob("*.json"))), total_decisions)
            self.assertEqual(
                baseline_result["artifact_counts"], resumed_result["artifact_counts"]
            )

            baseline = baseline_root / "finswarm_v1"
            self._assert_artifacts_equal(baseline, partial)
            self.assertEqual(
                self._agent_state(baseline_bundle), self._agent_state(resumed_bundle)
            )
            expected_hashes = sorted(
                json.loads(path.read_text(encoding="utf-8"))["record_hash"]
                for path in (baseline / "state" / JOURNAL_DIRECTORY).glob("*.json")
            )
            actual_hashes = sorted(
                json.loads(path.read_text(encoding="utf-8"))["record_hash"]
                for path in journal_dir.glob("*.json")
            )
            self.assertEqual(expected_hashes, actual_hashes)
            decision_ids = [
                row["decision_id"]
                for row in pq.read_table(partial / "agent_outputs.parquet").to_pylist()
            ]
            self.assertEqual(len(decision_ids), len(set(decision_ids)))

            idempotent_backend = _FakeBackend()
            idempotent_bundle = self._bundle_for(config_path, idempotent_backend)
            idempotent = self._run(
                config_path,
                dates,
                resumed_root,
                idempotent_bundle,
                resume=True,
                force_llm=True,
            )
            self.assertEqual(idempotent_backend.requests, [])
            self.assertEqual(idempotent["resume"]["duplicated_rows"], 0)

    def test_resume_after_tsla_reuses_tsla_and_starts_with_nvda(self) -> None:
        self._assert_resume_after(
            1,
            expected_first_symbol="NVDA",
            expected_completed_symbols=["TSLA"],
            expected_last_completed_index=None,
        )

    def test_resume_after_nvda_reuses_two_symbols_and_starts_with_aapl(self) -> None:
        self._assert_resume_after(
            2,
            expected_first_symbol="AAPL",
            expected_completed_symbols=["TSLA", "NVDA"],
            expected_last_completed_index=None,
        )

    def test_resume_after_aapl_starts_at_next_date_tsla(self) -> None:
        self._assert_resume_after(
            3,
            expected_first_symbol="TSLA",
            expected_completed_symbols=["TSLA", "NVDA", "AAPL"],
            expected_last_completed_index=0,
        )

    def test_resume_after_multiple_dates_reuses_all_completed_decisions(self) -> None:
        self._assert_resume_after(
            7,
            expected_first_symbol="NVDA",
            expected_completed_symbols=["TSLA"],
            expected_last_completed_index=1,
        )


class Stage6ReadinessTest(unittest.TestCase):
    def test_stage6_accepts_only_validated_real_action_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts"
            artifact.mkdir()
            for name in (
                "text_samples.parquet",
                "agent_outputs.parquet",
                "observed_features.parquet",
                "outcomes.parquet",
                "pending_decisions.parquet",
                "reflection_updates.parquet",
                "graph_samples.npz",
                "feature_schema.json",
            ):
                (artifact / name).write_bytes(b"fixture")
            (artifact / "temporal_quality.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "action_audit": {"final_training_eligible": True},
                    }
                ),
                encoding="utf-8",
            )
            (artifact / "rollout_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "temporal_gate": "PASS",
                        "final_training_eligible": True,
                        "full_interval": True,
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "bootstrap": False,
                "bootstrap_policy": None,
                "action_source": "real_causal_sector_agent",
                "external_llm_used": True,
                "final_training_eligible": True,
                "full_interval": True,
                "generation_provenance": {"synthetic": False},
            }
            (artifact / "data_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            config = _training_config(root, artifact)
            self.assertEqual(verify_causal_inputs(config)["status"], "PASS")
            manifest["generation_provenance"]["synthetic"] = True
            (artifact / "data_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "rejects bootstrap"):
                verify_causal_inputs(config)

    def test_runbook_contains_no_literal_secret(self) -> None:
        text = (ROOT / "docs/openai_causal_rollout_runbook.md").read_text(
            encoding="utf-8"
        )
        self.assertIn('FINMEM_LLM_MODEL="gemini-3.5-flash"', text)
        self.assertNotIn("sk-", text.lower())


def _training_config(root: Path, artifact: Path) -> TrainingConfig:
    return TrainingConfig(
        config_path=root / "training.yaml",
        repo_root=root,
        run_id="fixture",
        seed=42,
        experiment_config=root / "experiment.yaml",
        artifact_dir=artifact,
        split_manifest=root / "split.json",
        model_dir=root / "model",
        text_embedding={},
        split={},
        labels={},
        text_model={},
        graph_model={},
        fusion_model={},
        calibration={},
        raw={},
    )


def _trading_dates(count: int) -> list[dt.date]:
    dates = []
    value = dt.date(2026, 1, 2)
    while len(dates) < count:
        if value.weekday() < 5:
            dates.append(value)
        value += dt.timedelta(days=1)
    return dates


def _synthetic_repository(root: Path, count: int) -> tuple[Path, list[dt.date]]:
    dates = _trading_dates(count)
    config_dir = root / "config/swarm"
    config_dir.mkdir(parents=True)
    shutil.copyfile(ROOT / "config/swarm/model_config.yaml", config_dir / "model_config.yaml")
    shutil.copyfile(ROOT / "config/swarm/training_v1.yaml", config_dir / "training_v1.yaml")
    shutil.copyfile(ROOT / "config/tsla_gpt_config.toml", root / "config/tsla_gpt_config.toml")
    split_path = root / "data/manifests/price_split_manifest.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        json.dumps(
            {
                "splits": {
                    "warmup": {"dates": [date.isoformat() for date in dates]},
                    "train": {"dates": []},
                    "validation": {"dates": []},
                    "test": {"dates": []},
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = config_dir / "experiment_v1.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"run_id": "finswarm_v1"},
                "assets": {"symbols": ["TSLA", "NVDA", "AAPL"]},
                "time": {
                    "timezone": "America/New_York",
                    "decision_cutoff": "16:10:00",
                },
                "data": {
                    "start": dates[0].isoformat(),
                    "end": dates[-1].isoformat(),
                    "split_manifest": "data/manifests/price_split_manifest.json",
                    "raw": {
                        "prices": "data/raw/prices",
                        "news": "data/raw/news",
                        "filings": "data/raw/filings",
                    },
                    "canonical": {
                        "prices": "data/parquet/prices.parquet",
                        "news": "data/parquet/news.parquet",
                        "filings": "data/parquet/filings.parquet",
                    },
                    "duckdb": "data/duckdb/finswarm.db",
                },
                "normalization": {
                    "filings": {
                        "include_8k_exhibits": ["EX-99", "EX-99.1", "EX-99.2"]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    prices = []
    for symbol_index, symbol in enumerate(("TSLA", "NVDA", "AAPL")):
        for index, date in enumerate(dates):
            price = 100.0 + symbol_index * 10 + index
            prices.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": price,
                    "high": price + 2,
                    "low": price - 1,
                    "close": price + 1,
                    "adj_close": None,
                    "volume": 1000 + index,
                    "source": "synthetic",
                    "provider_record_id": f"{symbol}-{date}",
                    "metadata": "{}",
                }
            )
    parquet = root / "data/parquet"
    write_records(parquet / "prices.parquet", prices, PRICE_SCHEMA)
    write_records(parquet / "news.parquet", [], NEWS_SCHEMA)
    write_records(parquet / "filings.parquet", [], FILING_SCHEMA)
    store = DuckDBMarketStore(str(root / "data/duckdb/finswarm.db"))
    store.connect()
    store.ingest_prices(prices)
    store.close()
    return config_path, dates


if __name__ == "__main__":
    unittest.main()
