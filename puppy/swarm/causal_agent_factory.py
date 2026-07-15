from __future__ import annotations

import copy
import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import toml
import yaml

from puppy.agent import LLMAgent
from puppy.chat import ChatOpenAICompatible, MissingChatCredentialError
from puppy.data_engine.normalization import load_stage4_config
from puppy.embedding import build_memory_embedding
from puppy.fusion.text_encoder import TextStreamEncoder

from .legacy_finmem_adapter import LegacyFinMemAdapter
from .sector_agent import SectorAgent


CAUSAL_PROMPT_VERSION = "finmem-causal-buy-hold-sell-v1"
REQUIRED_SYMBOLS = ("TSLA", "NVDA", "AAPL")


def _required_secret(environ: Mapping[str, str]) -> str:
    key = environ.get("FINMEM_LLM_API_KEY") or environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingChatCredentialError(
            "Missing FINMEM_LLM_API_KEY (OPENAI_API_KEY is the supported fallback)"
        )
    return key


def _integer_setting(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    raw = environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _endpoint_from_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return (
        normalized
        if normalized.endswith("/chat/completions")
        else normalized + "/chat/completions"
    )


def _build_backend_from_environment(
    environ: Mapping[str, str],
    *,
    sleep_func: Callable[[float], None],
) -> ChatOpenAICompatible:
    api_key = _required_secret(environ)
    provider = environ.get("FINMEM_LLM_PROVIDER", "openai_compatible").strip()
    if provider != "openai_compatible":
        raise ValueError("FINMEM_LLM_PROVIDER must be 'openai_compatible'")
    model = environ.get("FINMEM_LLM_MODEL", "").strip()
    if not model:
        raise ValueError("FINMEM_LLM_MODEL is required")
    base_url = environ.get("FINMEM_LLM_BASE_URL", "https://api.openai.com/v1").strip()
    if not base_url:
        raise ValueError("FINMEM_LLM_BASE_URL is required")
    timeout = _integer_setting(
        environ, "FINMEM_LLM_TIMEOUT_SECONDS", 120, minimum=1
    )
    max_retries = _integer_setting(environ, "FINMEM_LLM_MAX_RETRIES", 5)
    requests_per_minute = _integer_setting(
        environ, "FINMEM_LLM_REQUESTS_PER_MINUTE", 30, minimum=1
    )
    service_provider = (
        "google_gemini"
        if "generativelanguage.googleapis.com" in base_url.lower()
        else "openai_compatible"
    )
    backend = ChatOpenAICompatible(
        end_point=_endpoint_from_base_url(base_url),
        model=model,
        system_message="Return only valid JSON for the FinMem causal decision schema.",
        other_parameters={"temperature": 0},
        api_key=api_key,
        require_api_key=True,
        provider=provider,
        service_provider=service_provider,
        base_url=base_url,
        timeout_seconds=timeout,
        max_retries=max_retries,
        requests_per_minute=requests_per_minute,
        sleep_func=sleep_func,
    )
    backend.is_mock = False
    return backend


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required factory configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Factory configuration must be a mapping: {path}")
    return value


def _symbol_config(
    template: dict[str, Any],
    *,
    symbol: str,
    agent_id: str,
    backend: Any,
    memory_embedding_config: Mapping[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config["general"]["agent_name"] = agent_id
    config["general"]["trading_symbol"] = symbol
    character = str(config["general"]["character_string"])
    config["general"]["character_string"] = (
        character.replace("TSLA", symbol).replace("Tesla", symbol)
        + f"\nRuntime identity: {agent_id}. Trade only {symbol}."
    )
    config["chat"]["model"] = str(getattr(backend, "model", "injected-mock"))
    public = getattr(backend, "public_config", {})
    endpoint = getattr(backend, "end_point", None)
    if endpoint:
        config["chat"]["end_point"] = str(endpoint)
    elif isinstance(public, Mapping) and public.get("base_url"):
        config["chat"]["end_point"] = str(public["base_url"])
    config["chat"]["temperature"] = 0
    config["agent"]["agent_1"]["embedding"]["detail"] = dict(
        memory_embedding_config
    )
    return config


def _memory_embedding_config(
    environ: Mapping[str, str],
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    provider = environ.get(
        "FINMEM_MEMORY_EMBEDDING_PROVIDER", "sentence_transformers"
    ).strip().lower()
    configured_model = environ.get("FINMEM_MEMORY_EMBEDDING_MODEL", "").strip()
    if provider == "sentence_transformers":
        model = configured_model or str(training_config["text_embedding"]["model_name"])
        return {
            "provider": provider,
            "embedding_model": model,
            "batch_size": int(training_config["text_embedding"].get("batch_size", 32)),
            "normalize": bool(training_config["text_embedding"].get("normalize", True)),
            "local_files_only": bool(
                training_config["text_embedding"].get("local_files_only", False)
            ),
            "device": "cpu",
        }
    if provider == "openai":
        return {
            "provider": provider,
            "embedding_model": configured_model or "text-embedding-ada-002",
            "chunk_size": 5000,
            "verbose": False,
        }
    raise ValueError(
        "FINMEM_MEMORY_EMBEDDING_PROVIDER must be 'sentence_transformers' "
        "or explicit legacy 'openai'"
    )


def _null_logger(agent_id: str) -> logging.Logger:
    logger = logging.getLogger(f"puppy.causal_factory.{agent_id}")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


def build_causal_agent_bundle(
    config_path: str,
    *,
    chat_backend: Any | None = None,
    memory_factory: Callable[[dict[str, Any], str], Any] | None = None,
    memory_embedding_backend: Any | None = None,
    encoder: TextStreamEncoder | Any | None = None,
    environ: Mapping[str, str] | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Build the one canonical three-agent causal production composition.

    Optional arguments exist only for offline dependency injection. Any injected
    chat backend is treated as synthetic and can never make Stage 5 final-training
    eligible.
    """

    stage4 = load_stage4_config(config_path)
    if tuple(stage4.symbols) != REQUIRED_SYMBOLS:
        raise ValueError(
            "Canonical causal factory requires symbols in order TSLA, NVDA, AAPL"
        )
    repo_root = stage4.repo_root
    model_config = _load_yaml(repo_root / "config/swarm/model_config.yaml")
    training_config = _load_yaml(repo_root / "config/swarm/training_v1.yaml")
    template_path = repo_root / "config/tsla_gpt_config.toml"
    template = toml.load(template_path)

    runtime_environ = os.environ if environ is None else environ
    injected_backend = chat_backend is not None
    backend = chat_backend or _build_backend_from_environment(
        runtime_environ,
        sleep_func=sleep_func,
    )
    if not callable(getattr(backend, "guardrail_endpoint", None)):
        raise TypeError("Injected chat backend must expose guardrail_endpoint()")
    try:
        backend.is_mock = injected_backend
    except (AttributeError, TypeError):
        if injected_backend and not bool(getattr(backend, "is_mock", False)):
            raise TypeError("Injected chat backend must expose is_mock=True")

    if encoder is None:
        text = training_config["text_embedding"]
        if model_config.get("text_encoder", {}).get("provider") != "pretrained":
            raise ValueError("Canonical TextStreamEncoder requires pretrained provider")
        encoder = TextStreamEncoder.from_transformer(
            str(text["model_name"]),
            max_length=int(text.get("max_length", 512)),
            normalize=bool(text.get("normalize", True)),
            local_files_only=bool(text.get("local_files_only", False)),
            device="cpu",
        )

    memory_config = _memory_embedding_config(runtime_environ, training_config)
    shared_memory_embedding = memory_embedding_backend
    if memory_factory is None and shared_memory_embedding is None:
        transformer_backend = getattr(encoder, "transformer_backend", None)
        if (
            transformer_backend is not None
            and str(getattr(transformer_backend, "model_name", ""))
            != str(memory_config["embedding_model"])
        ):
            transformer_backend = None
        shared_memory_embedding = build_memory_embedding(
            memory_config,
            transformer_backend=transformer_backend,
        )

    agents: dict[str, SectorAgent] = {}
    namespaces: set[str] = set()
    for symbol in REQUIRED_SYMBOLS:
        agent_id = f"sector_{symbol.lower()}"
        namespace = f"{stage4.run_id}/{agent_id}/{symbol.lower()}"
        if namespace in namespaces:
            raise AssertionError(f"Duplicate causal memory namespace: {namespace}")
        namespaces.add(namespace)
        agent_config = _symbol_config(
            template,
            symbol=symbol,
            agent_id=agent_id,
            backend=backend,
            memory_embedding_config=memory_config,
        )
        brain = (
            memory_factory(agent_config, namespace)
            if memory_factory is not None
            else None
        )
        llm_agent = LLMAgent.from_config(
            agent_config,
            brain_db=brain,
            chat_backend=backend,
            memory_embedding_backend=shared_memory_embedding,
            logger=_null_logger(agent_id) if injected_backend else None,
        )
        adapter = LegacyFinMemAdapter(
            llm_agent=llm_agent,
            agent_id=agent_id,
            symbol=symbol,
            bootstrap=False,
        )
        agent = SectorAgent(
            agent_id=agent_id,
            symbol=symbol,
            llm_agent=llm_agent,
            causal_adapter=adapter,
        )
        agent.memory_namespace = namespace
        agent.runtime_identity = (
            f"{agent_id}:{symbol}:"
            f"{getattr(backend, 'identity', type(backend).__name__)}"
        )
        agents[symbol] = agent

    if len({id(agent.llm_agent.brain) for agent in agents.values()}) != len(agents):
        raise RuntimeError("Causal agents must not share a mutable BrainDB instance")
    raw_backend_config = getattr(backend, "public_config", {})
    raw_backend_config = (
        dict(raw_backend_config) if isinstance(raw_backend_config, Mapping) else {}
    )
    backend_config = {
        key: raw_backend_config[key]
        for key in (
            "provider",
            "interface",
            "base_url",
            "model",
            "timeout_seconds",
            "max_retries",
            "requests_per_minute",
            "temperature",
            "backend_identity",
        )
        if key in raw_backend_config
    }
    return {
        "agents": agents,
        "encoder": encoder,
        "prompt_version": CAUSAL_PROMPT_VERSION,
        "provenance": {
            "mode": "synthetic_mock" if injected_backend else "real_openai_compatible",
            "synthetic": injected_backend,
            "external_llm_expected": not injected_backend,
            "provider": backend_config.get("provider", "injected_mock"),
            "model": str(getattr(backend, "model", "injected-mock")),
            "backend_identity": str(getattr(backend, "identity", "injected-mock")),
            "backend_config": backend_config,
            "memory_embedding": {
                "provider": memory_config["provider"],
                "model": memory_config["embedding_model"],
                "dimension": (
                    int(shared_memory_embedding.get_embedding_dimension())
                    if shared_memory_embedding is not None
                    else None
                ),
                "shared_read_only_encoder": shared_memory_embedding is not None,
            },
        },
    }
