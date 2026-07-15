from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from abc import ABC
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx


def build_llama2_prompt(messages: list[dict[str, str]]) -> str:
    start_prompt = "<s>[INST] "
    end_prompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(
                f" [/INST] {message['content'].strip()}</s><s>[INST] "
            )
    return start_prompt + "".join(conversation) + end_prompt


class LongerThanContextError(Exception):
    pass


class MissingChatCredentialError(RuntimeError):
    """Raised before transport construction when a required credential is absent."""


class SanitizedChatBackendError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class StructuredCompletionError(RuntimeError):
    def __init__(
        self,
        category: str,
        *,
        response_hash: str = "unavailable",
    ) -> None:
        super().__init__(f"Structured completion failed: {category}")
        self.category = category
        self.response_hash = response_hash


@dataclass(frozen=True)
class StructuredCompletionResult:
    parsed: Any | None
    returned_model: str | None
    response_hash: str
    finish_reason: str | None = None
    content_present: bool = False


class ChatOpenAICompatible(ABC):
    """OpenAI-compatible Guardrails endpoint with bounded, observable retries.

    The API key is intentionally kept outside ``public_config`` and is never
    included in backend identity, exceptions, cache keys, or persisted state.
    """

    def __init__(
        self,
        end_point: str,
        model: str = "gemini-pro",
        system_message: str = "You are a helpful assistant.",
        other_parameters: Mapping[str, Any] | None = None,
        *,
        api_key: str | None = None,
        require_api_key: bool = False,
        provider: str = "openai_compatible",
        service_provider: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 600.0,
        max_retries: int = 0,
        requests_per_minute: int = 0,
        sleep_func: Callable[[float], None] = time.sleep,
        clock_func: Callable[[], float] = time.monotonic,
        http_post: Callable[..., httpx.Response] = httpx.post,
        openai_client: Any | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if require_api_key and not resolved_key:
            raise MissingChatCredentialError(
                "Missing FINMEM_LLM_API_KEY (OPENAI_API_KEY is the supported fallback)"
            )
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute must be non-negative")

        self.end_point = end_point
        self.model = model
        self.system_message = system_message
        self.provider = provider
        self.service_provider = service_provider or provider
        self.base_url = (base_url or self._base_url_from_endpoint(end_point)).rstrip(
            "/"
        ) + "/"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.requests_per_minute = int(requests_per_minute)
        self.other_parameters = dict(other_parameters or {})
        self._api_key = resolved_key
        self._sleep = sleep_func
        self._clock = clock_func
        self._http_post = http_post
        self._openai_client = openai_client
        self._request_times: deque[float] = deque()
        self.api_call_count = 0
        self.retry_count = 0
        self.last_response_hash: str | None = None
        self.last_returned_model: str | None = None

        if self.model.startswith("gemini-pro"):
            proc_result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
            )
            access_token = proc_result.stdout.strip()
            self.headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
        elif self.model.startswith("tgi"):
            self.headers = {"Content-Type": "application/json"}
        else:
            self.headers = {
                "Authorization": f"Bearer {resolved_key or '-'}",
                "Content-Type": "application/json",
            }

    @property
    def identity(self) -> str:
        identity = json.dumps(
            {
                "provider": self.service_provider,
                "interface": self.provider,
                "base_url_identity": self.base_url_identity,
                "model": self.model,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _base_url_from_endpoint(end_point: str) -> str:
        normalized = end_point.rstrip("/")
        suffix = "/chat/completions"
        return normalized[: -len(suffix)] if normalized.endswith(suffix) else normalized

    @property
    def base_url_identity(self) -> str:
        value = json.dumps(
            {
                "provider": self.service_provider,
                "base_url": self.base_url.rstrip("/"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @property
    def public_config(self) -> dict[str, Any]:
        return {
            "provider": self.service_provider,
            "interface": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "requests_per_minute": self.requests_per_minute,
            "temperature": self.other_parameters.get("temperature"),
            "backend_identity": self.identity,
        }

    def parse_response(self, response: httpx.Response) -> str:
        response_out = response.json()
        if self.model.startswith("gemini-pro"):
            return str(response_out["candidates"][0]["content"]["parts"][0]["text"])
        if self.model.startswith("tgi"):
            return str(response_out["generated_text"])
        self.last_returned_model = str(response_out.get("model") or self.model)
        return str(response_out["choices"][0]["message"]["content"])

    def _get_openai_client(self) -> Any:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
        parse_method = getattr(
            getattr(
                getattr(
                    getattr(self._openai_client, "beta", None),
                    "chat",
                    None,
                ),
                "completions",
                None,
            ),
            "parse",
            None,
        )
        if not callable(parse_method):
            raise StructuredCompletionError("structured_parse_failed")
        return self._openai_client

    @staticmethod
    def _structured_error_category(error: Exception) -> str:
        status_code = getattr(error, "status_code", None)
        if status_code in {401, 403}:
            return "authentication_failed"
        if status_code == 429:
            return "rate_limited"
        name = type(error).__name__.lower()
        if "authentication" in name or "permission" in name:
            return "authentication_failed"
        if "ratelimit" in name:
            return "rate_limited"
        return "structured_parse_failed"

    def structured_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_format: type[Any],
        temperature: float = 0.0,
        reasoning_effort: str = "low",
    ) -> StructuredCompletionResult:
        """Invoke the official OpenAI-compatible parsed response interface."""

        self._wait_for_rate_limit()
        self.api_call_count += 1
        try:
            completion = self._get_openai_client().beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            choice = completion.choices[0]
            message = choice.message
            parsed = getattr(message, "parsed", None)
            returned_model = str(getattr(completion, "model", None) or self.model)
            finish_reason = getattr(choice, "finish_reason", None)
            if parsed is not None:
                payload = (
                    parsed.model_dump_json()
                    if hasattr(parsed, "model_dump_json")
                    else json.dumps(parsed, sort_keys=True, default=str)
                )
            else:
                payload = str(getattr(message, "content", None) or finish_reason or "")
            response_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            self.last_response_hash = response_hash
            self.last_returned_model = returned_model
            return StructuredCompletionResult(
                parsed=parsed,
                returned_model=returned_model,
                response_hash=response_hash,
                finish_reason=str(finish_reason) if finish_reason is not None else None,
                content_present=bool(getattr(message, "content", None)),
            )
        except StructuredCompletionError:
            raise
        except Exception as exc:
            raise StructuredCompletionError(
                self._structured_error_category(exc),
                response_hash=self.last_response_hash or "unavailable",
            ) from exc

    def _wait_for_rate_limit(self) -> None:
        if self.requests_per_minute <= 0:
            return
        now = self._clock()
        while self._request_times and now - self._request_times[0] >= 60.0:
            self._request_times.popleft()
        if len(self._request_times) >= self.requests_per_minute:
            delay = max(0.0, 60.0 - (now - self._request_times[0]))
            self._sleep(delay)
            now = self._clock()
            while self._request_times and now - self._request_times[0] >= 60.0:
                self._request_times.popleft()
        self._request_times.append(self._clock())

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code <= 599

    def _post_with_retry(self, *, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            self.api_call_count += 1
            try:
                response = self._http_post(
                    self.end_point,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code == 422 and "must have less than" in exc.response.text:
                    raise LongerThanContextError from exc
                retryable = self._retryable_status(status_code)
                if not retryable or attempt >= self.max_retries:
                    raise SanitizedChatBackendError(
                        f"OpenAI-compatible request failed with HTTP {status_code}",
                        status_code=status_code,
                        retryable=retryable,
                    ) from exc
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError) as exc:
                if attempt >= self.max_retries:
                    raise SanitizedChatBackendError(
                        "OpenAI-compatible request failed after a temporary transport error",
                        retryable=True,
                    ) from exc
            self.retry_count += 1
            self._sleep(min(30.0, 0.5 * (2**attempt)))
        raise AssertionError("bounded retry loop exhausted unexpectedly")

    def guardrail_endpoint(self) -> Callable[..., str]:
        def endpoint(input: str, **_: Any) -> str:
            input_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant only capable of communicating "
                        "with valid JSON, and no other text."
                    ),
                },
                {"role": "user", "content": str(input)},
            ]
            if self.model.startswith("gemini-pro"):
                payload = {
                    "contents": {
                        "role": "USER",
                        "parts": {"text": input_messages[1]["content"]},
                    },
                    "generation_config": {
                        "temperature": 0.2,
                        "top_p": 0.1,
                        "top_k": 16,
                        "max_output_tokens": 2048,
                        "candidate_count": 1,
                        "stop_sequences": [],
                    },
                    "safety_settings": {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_LOW_AND_ABOVE",
                    },
                }
            elif self.model.startswith("tgi"):
                payload = {
                    "inputs": build_llama2_prompt(input_messages),
                    "parameters": {
                        "do_sample": True,
                        "top_p": 0.6,
                        "temperature": 0.8,
                        "top_k": 50,
                        "max_new_tokens": 256,
                        "repetition_penalty": 1.03,
                        "stop": ["</s>"],
                    },
                }
            else:
                payload = {"model": self.model, "messages": input_messages}
                payload.update(self.other_parameters)
            response = self._post_with_retry(payload=payload)
            parsed = self.parse_response(response)
            self.last_response_hash = hashlib.sha256(parsed.encode("utf-8")).hexdigest()
            return parsed

        return endpoint
