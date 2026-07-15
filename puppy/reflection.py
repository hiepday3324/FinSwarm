# sourcery skip: dont-import-test-modules
from rich import print
import hashlib
import json
import logging
import guardrails as gd
from datetime import date
from enum import Enum
from .run_type import RunMode
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from httpx import HTTPStatusError
try:
    from guardrails.validators import ValidChoices
except ImportError:  # Guardrails 0.5+ removed this legacy export.
    class ValidChoices:  # type: ignore[no-redef]
        def __init__(self, choices, on_fail=None):
            self.choices = tuple(choices)
            self.on_fail = on_fail
from typing import List, Callable, Dict, Union, Any, Tuple
from .chat import (
    LongerThanContextError,
    StructuredCompletionError,
    StructuredCompletionResult,
)
from .prompts import (
    short_memory_id_desc,
    mid_memory_id_desc,
    long_memory_id_desc,
    reflection_memory_id_desc,
    train_prompt,
    train_memory_id_extract_prompt,
    train_trade_reason_summary,
    train_investment_info_prefix,
    test_prompt,
    test_trade_reason_summary,
    test_memory_id_extract_prompt,
    test_invest_action_choice,
    test_investment_info_prefix,
    test_sentiment_explanation,
    test_momentum_explanation,
)


CAUSAL_DECISION_SCHEMA_VERSION = "causal-decision-response-v1"


class CausalDecisionValue(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class CausalMemoryIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_index: int


class CausalDecisionResponse(BaseModel):
    """Single structured response schema for all causal LLM decisions."""

    model_config = ConfigDict(extra="forbid")

    investment_decision: CausalDecisionValue
    summary_reason: str = Field(min_length=1)
    short_memory_index: list[CausalMemoryIndex]
    middle_memory_index: list[CausalMemoryIndex]
    long_memory_index: list[CausalMemoryIndex]
    reflection_memory_index: list[CausalMemoryIndex]

    @field_validator("summary_reason")
    @classmethod
    def _nonblank_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("summary_reason must not be blank")
        return normalized


class CausalDecisionParseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str = "structured_parse_failed",
        response_hash: str = "unavailable",
        requested_model: str | None = None,
        returned_model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.response_hash = response_hash
        self.requested_model = requested_model
        self.returned_model = returned_model


def _validation_category(error: ValidationError) -> str:
    errors = error.errors()
    if any(item.get("type") == "missing" for item in errors):
        return "missing_required_field"
    if any(tuple(item.get("loc", ())) == ("investment_decision",) for item in errors):
        return "invalid_decision_enum"
    return "structured_parse_failed"


def _validate_cited_memory_ids(
    response: CausalDecisionResponse,
    allowed_memory_ids: dict[str, list[int]],
) -> None:
    for field, allowed in allowed_memory_ids.items():
        allowed_set = set(allowed)
        cited = getattr(response, field)
        if any(item.memory_index not in allowed_set for item in cited):
            raise CausalDecisionParseError(
                "Structured causal decision cited an ineligible memory ID",
                category="invalid_memory_id",
            )


def _response_to_legacy_boundary(
    response: CausalDecisionResponse,
    *,
    retry_count: int,
    response_hash: str,
    provenance: dict[str, Any],
) -> Dict[str, Any]:
    result = {
        "investment_decision": response.investment_decision.value.lower(),
        "summary_reason": response.summary_reason,
        "short_memory_index": [item.model_dump() for item in response.short_memory_index],
        "middle_memory_index": [item.model_dump() for item in response.middle_memory_index],
        "long_memory_index": [item.model_dump() for item in response.long_memory_index],
        "reflection_memory_index": [
            item.model_dump() for item in response.reflection_memory_index
        ],
        "_causal_parse_retries": retry_count,
        "_causal_response_hash": response_hash,
        "_causal_provenance": provenance,
    }
    return _delete_placeholder_info(result)


def _raw_output_hashes(guard: Any) -> list[str]:
    try:
        raw_outputs = list(guard.history[0].raw_outputs)
    except (AttributeError, IndexError, TypeError):
        return []
    return [hashlib.sha256(str(value).encode("utf-8")).hexdigest() for value in raw_outputs]


def _find_exception(error: BaseException, exception_type: type[BaseException]) -> BaseException | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, exception_type):
            return current
        current = current.__cause__ or current.__context__
    return None


def _strict_causal_response(
    *,
    endpoint_func: Callable[[str], str],
    prompt: str,
    investment_info: str,
    allowed_memory_ids: dict[str, list[int]],
) -> Dict[str, Any]:
    schema_instruction = (
        "Return one JSON object with investment_decision (BUY, HOLD, or SELL), "
        "summary_reason, short_memory_index, middle_memory_index, "
        "long_memory_index, and reflection_memory_index. Each memory field must "
        "be a list of objects containing memory_index. No additional text."
    )
    frozen_prompt = (
        prompt.replace("${investment_info}", investment_info)
        .replace("${gr.complete_json_suffix_v2}", schema_instruction)
    )
    last_hash = "unavailable"
    for attempt in range(2):
        request = frozen_prompt
        if attempt:
            request += (
                "\nThe previous response was invalid. Use the exact JSON schema and "
                "an explicit BUY, HOLD, or SELL value."
            )
        raw = endpoint_func(request)
        last_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        try:
            parsed = CausalDecisionResponse.model_validate_json(raw)
            _validate_cited_memory_ids(parsed, allowed_memory_ids)
            return _response_to_legacy_boundary(
                parsed,
                retry_count=attempt,
                response_hash=last_hash,
                provenance={
                    "schema_version": CAUSAL_DECISION_SCHEMA_VERSION,
                    "interface": "legacy_json_compatibility",
                },
            )
        except (ValidationError, CausalDecisionParseError):
            continue
    raise CausalDecisionParseError(
        "Causal LLM response did not match the strict BUY/HOLD/SELL schema",
        response_hash=last_hash,
    )


def _structured_causal_response(
    *,
    structured_endpoint: Callable[..., StructuredCompletionResult],
    prompt: str,
    investment_info: str,
    allowed_memory_ids: dict[str, list[int]],
    requested_model: str,
    prompt_version: str,
    provider: str,
    interface: str,
) -> Dict[str, Any]:
    eligible_ids = {
        field: sorted({value for value in values if value != -1})
        for field, values in allowed_memory_ids.items()
    }
    schema_instruction = (
        "Return only the structured causal decision response. Decision must be "
        "exactly BUY, HOLD, or SELL. Cite only IDs from these eligible memory "
        f"layers: {json.dumps(eligible_ids, sort_keys=True)}. Use an empty list "
        "when a layer has no eligible IDs."
    )
    frozen_prompt = (
        prompt.replace("${investment_info}", investment_info)
        .replace("${gr.complete_json_suffix_v2}", schema_instruction)
    )
    base_messages = [
        {
            "role": "system",
            "content": (
                "You are a causal financial decision agent. Use only the frozen "
                "context and the eligible memory IDs supplied by the user."
            ),
        },
        {"role": "user", "content": frozen_prompt},
    ]
    last_category = "structured_parse_failed"
    last_hash = "unavailable"
    last_returned_model: str | None = None
    for attempt in range(2):
        messages = list(base_messages)
        if attempt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Schema correction: return every required field, one valid "
                        "decision enum, and only eligible memory IDs. Keep the same "
                        "frozen context."
                    ),
                }
            )
        try:
            completion = structured_endpoint(
                messages=messages,
                response_format=CausalDecisionResponse,
                temperature=0.0,
                reasoning_effort="low",
            )
            last_hash = completion.response_hash
            last_returned_model = completion.returned_model
            if completion.parsed is None:
                finish = str(completion.finish_reason or "").lower()
                if any(marker in finish for marker in ("safety", "blocked", "content_filter")):
                    last_category = "safety_blocked"
                elif completion.content_present:
                    last_category = "structured_parse_failed"
                else:
                    last_category = "empty_response"
                continue
            try:
                parsed = CausalDecisionResponse.model_validate(completion.parsed)
            except ValidationError as exc:
                last_category = _validation_category(exc)
                continue
            try:
                _validate_cited_memory_ids(parsed, eligible_ids)
            except CausalDecisionParseError as exc:
                last_category = exc.category
                continue
            return _response_to_legacy_boundary(
                parsed,
                retry_count=attempt,
                response_hash=last_hash,
                provenance={
                    "provider": provider,
                    "interface": interface,
                    "requested_model": requested_model,
                    "returned_model": last_returned_model,
                    "prompt_version": prompt_version,
                    "schema_version": CAUSAL_DECISION_SCHEMA_VERSION,
                    "temperature": 0.0,
                    "reasoning_effort": "low",
                    "response_hash": last_hash,
                    "retry_count": attempt,
                },
            )
        except StructuredCompletionError as exc:
            last_category = exc.category
            last_hash = exc.response_hash
            if last_category in {"authentication_failed", "rate_limited"}:
                break
    raise CausalDecisionParseError(
        "Gemini structured causal response failed validation",
        category=last_category,
        response_hash=last_hash,
        requested_model=requested_model,
        returned_model=last_returned_model,
    )


def _train_memory_factory(memory_layer: str, id_list: List[int]):
    class Memory(BaseModel):
        memory_index: int = Field(
            ...,
            description=train_memory_id_extract_prompt.format(
                memory_layer=memory_layer
            ),
            validators=[ValidChoices(id_list, on_fail="reask")],  # type: ignore
        )

    return Memory


def _test_memory_factory(memory_layer: str, id_list: List[int]):
    class Memory(BaseModel):
        memory_index: int = Field(
            ...,
            description=test_memory_id_extract_prompt.format(memory_layer=memory_layer),
            validators=[ValidChoices(id_list)],  # type: ignore
        )

    return Memory


# train + test reflection model
def _train_reflection_factory(
    short_id_list: List[int],
    mid_id_list: List[int],
    long_id_list: List[int],
    reflection_id_list: List[int],
):
    LongMem = _train_memory_factory("long-level", long_id_list)
    MidMem = _train_memory_factory("mid-level", mid_id_list)
    ShortMem = _train_memory_factory("short-level", short_id_list)
    ReflectionMem = _train_memory_factory("reflection-level", reflection_id_list)

    class InvestInfo(BaseModel):
        if reflection_id_list:
            reflection_memory_index: List[ReflectionMem] = Field(
                ...,
                description=reflection_memory_id_desc,
            )
        if long_id_list:
            long_memory_index: List[LongMem] = Field(
                ...,
                description=long_memory_id_desc,
            )
        if mid_id_list:
            middle_memory_index: List[MidMem] = Field(
                ...,
                description=mid_memory_id_desc,
            )
        if short_id_list:
            short_memory_index: List[ShortMem] = Field(
                ...,
                description=short_memory_id_desc,
            )
        summary_reason: str = Field(
            ...,
            description=train_trade_reason_summary,
        )

    return InvestInfo


def _test_reflection_factory(
    short_id_list: List[int],
    mid_id_list: List[int],
    long_id_list: List[int],
    reflection_id_list: List[int],
):
    LongMem = _test_memory_factory("long-level", long_id_list)
    MidMem = _test_memory_factory("mid-level", mid_id_list)
    ShortMem = _test_memory_factory("short-level", short_id_list)
    ReflectionMem = _test_memory_factory("reflection-level", reflection_id_list)

    class InvestInfo(BaseModel):
        investment_decision: str = Field(
            ...,
            description=test_invest_action_choice,
            validators=[ValidChoices(choices=["buy", "sell", "hold"])],  # type: ignore
        )
        summary_reason: str = Field(
            ...,
            description=test_trade_reason_summary,
        )
        if short_id_list:
            short_memory_index: List[ShortMem] = Field(
                ...,
                description=short_memory_id_desc,
            )
        if mid_id_list:
            middle_memory_index: List[MidMem] = Field(
                ...,
                description=mid_memory_id_desc,
            )
        if long_id_list:
            long_memory_index: List[LongMem] = Field(
                ...,
                description=long_memory_id_desc,
            )
        if reflection_id_list:
            reflection_memory_index: List[ReflectionMem] = Field(
                ...,
                description=reflection_memory_id_desc,
            )

    return InvestInfo


def _format_memories(
    short_memory: Union[List[str], None] = None,
    short_memory_id: Union[List[int], None] = None,
    mid_memory: Union[List[str], None] = None,
    mid_memory_id: Union[List[int], None] = None,
    long_memory: Union[List[str], None] = None,
    long_memory_id: Union[List[int], None] = None,
    reflection_memory: Union[List[str], None] = None,
    reflection_memory_id: Union[List[int], None] = None,
) -> Tuple[
    List[str],
    List[int],
    List[str],
    List[int],
    List[str],
    List[int],
    List[str],
    List[int],
]:
    # add placeholder information if not memory is available
    # each memory has a duplicate because guardrails::ValidChoices does not support single choice
    if (short_memory is None) or len(short_memory) == 0:
        short_memory = ["No short-term information.", "No short-term information."]
        short_memory_id = [-1, -1]
    elif len(short_memory) == 1:
        short_memory = [short_memory[0], short_memory[0]]
        short_memory_id = [short_memory_id[0], short_memory_id[0]]  # type: ignore
    if (mid_memory is None) or len(mid_memory) == 0:
        mid_memory = ["No mid-term information.", "No mid-term information."]
        mid_memory_id = [-1, -1]
    elif len(mid_memory) == 1:
        mid_memory = [mid_memory[0], mid_memory[0]]
        mid_memory_id = [mid_memory_id[0], mid_memory_id[0]]  # type: ignore
    if (long_memory is None) or len(long_memory) == 0:
        long_memory = ["No long-term information.", "No long-term information."]
        long_memory_id = [-1, -1]
    elif len(long_memory) == 1:
        long_memory = [long_memory[0], long_memory[0]]
        long_memory_id = [long_memory_id[0], long_memory_id[0]]  # type: ignore
    if (reflection_memory is None) or len(reflection_memory) == 0:
        reflection_memory = [
            "No reflection-term information.",
            "No reflection-term information.",
        ]
        reflection_memory_id = [-1, -1]
    elif len(reflection_memory) == 1:
        reflection_memory = [reflection_memory[0], reflection_memory[0]]
        reflection_memory_id = [reflection_memory_id[0], reflection_memory_id[0]]  # type: ignore

    return (
        short_memory,
        short_memory_id,
        mid_memory,
        mid_memory_id,
        long_memory,
        long_memory_id,
        reflection_memory,
        reflection_memory_id,
    )


def _delete_placeholder_info(validated_output: Dict[str, Any]) -> Dict[str, Any]:
    if "reflection_memory_index" in validated_output and (
        (validated_output["reflection_memory_index"])
        and (validated_output["reflection_memory_index"][0]["memory_index"] == -1)
    ):
        del validated_output["reflection_memory_index"]
    if "long_memory_index" in validated_output and (
        (validated_output["long_memory_index"])
        and (validated_output["long_memory_index"][0]["memory_index"] == -1)
    ):
        del validated_output["long_memory_index"]
    if "middle_memory_index" in validated_output and (
        (validated_output["middle_memory_index"])
        and (validated_output["middle_memory_index"][0]["memory_index"] == -1)
    ):
        del validated_output["middle_memory_index"]
    if "short_memory_index" in validated_output and (
        (validated_output["short_memory_index"])
        and (validated_output["short_memory_index"][0]["memory_index"] == -1)
    ):
        del validated_output["short_memory_index"]

    return validated_output


def _add_momentum_info(momentum: int, investment_info: str) -> str:
    if momentum == -1:
        investment_info += (
            "The cumulative return of past 3 days for this stock is negative."
        )

    elif momentum == 0:
        investment_info += (
            "The cumulative return of past 3 days for this stock is zero."
        )

    elif momentum == 1:
        investment_info += (
            "The cumulative return of past 3 days for this stock is positive."
        )

    return investment_info


def _train_response_model_invest_info(
    cur_date: date,
    symbol: str,
    future_record: Dict[str, float | str],
    short_memory: List[str],
    short_memory_id: List[int],
    mid_memory: List[str],
    mid_memory_id: List[int],
    long_memory: List[str],
    long_memory_id: List[int],
    reflection_memory: List[str],
    reflection_memory_id: List[int],
):
    # pydantic reflection model
    response_model = _train_reflection_factory(
        short_id_list=short_memory_id,
        mid_id_list=mid_memory_id,
        long_id_list=long_memory_id,
        reflection_id_list=reflection_memory_id,
    )
    # investment info + memories
    investment_info = train_investment_info_prefix.format(
        cur_date=cur_date, symbol=symbol, future_record=future_record
    )
    if short_memory:
        investment_info += "The short-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
        )
        investment_info += "\n\n"
    if mid_memory:
        investment_info += "The mid-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
        )
        investment_info += "\n\n"
    if long_memory:
        investment_info += "The long-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
        )
        investment_info += "\n\n"
    if reflection_memory:
        investment_info += "The reflection-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(reflection_memory_id, reflection_memory)]
        )
        investment_info += "\n\n"

    return response_model, investment_info


def _test_response_model_invest_info(
    cur_date: date,
    symbol: str,
    short_memory: List[str],
    short_memory_id: List[int],
    mid_memory: List[str],
    mid_memory_id: List[int],
    long_memory: List[str],
    long_memory_id: List[int],
    reflection_memory: List[str],
    reflection_memory_id: List[int],
    momentum: Union[int, None] = None,
    build_response_model: bool = True,
):
    # pydantic reflection model
    response_model = (
        _test_reflection_factory(
            short_id_list=short_memory_id,
            mid_id_list=mid_memory_id,
            long_id_list=long_memory_id,
            reflection_id_list=reflection_memory_id,
        )
        if build_response_model
        else None
    )
    # investment info + memories
    investment_info = test_investment_info_prefix.format(
        symbol=symbol, cur_date=cur_date
    )
    if short_memory:
        investment_info += "The short-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
        )
        investment_info += test_sentiment_explanation
        investment_info += "\n\n"
    if mid_memory:
        investment_info += "The mid-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
        )
        investment_info += "\n\n"
    if long_memory:
        investment_info += "The long-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
        )
        investment_info += "\n\n"
    if reflection_memory:
        investment_info += "The reflection-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(reflection_memory_id, reflection_memory)]
        )
        investment_info += "\n\n"
    if momentum:
        investment_info += test_momentum_explanation
        investment_info = _add_momentum_info(momentum, investment_info)

    return response_model, investment_info


def trading_reflection(
    cur_date: date,
    endpoint_func: Callable[[str], str],
    symbol: str,
    run_mode: RunMode,
    logger: logging.Logger,
    momentum: Union[int, None] = None,
    future_record: Union[Dict[str, float | str], None] = None,
    short_memory: Union[List[str], None] = None,
    short_memory_id: Union[List[int], None] = None,
    mid_memory: Union[List[str], None] = None,
    mid_memory_id: Union[List[int], None] = None,
    long_memory: Union[List[str], None] = None,
    long_memory_id: Union[List[int], None] = None,
    reflection_memory: Union[List[str], None] = None,
    reflection_memory_id: Union[List[int], None] = None,
    strict_output: bool = False,
    structured_endpoint: Callable[..., StructuredCompletionResult] | None = None,
    requested_model: str | None = None,
    prompt_version: str | None = None,
    provider: str = "unknown",
    interface: str = "unknown",
) -> Dict[str, Any]:
    # format memories
    (
        short_memory,
        short_memory_id,
        mid_memory,
        mid_memory_id,
        long_memory,
        long_memory_id,
        reflection_memory,
        reflection_memory_id,
    ) = _format_memories(
        short_memory=short_memory,
        short_memory_id=short_memory_id,
        mid_memory=mid_memory,
        mid_memory_id=mid_memory_id,
        long_memory=long_memory,
        long_memory_id=long_memory_id,
        reflection_memory=reflection_memory,
        reflection_memory_id=reflection_memory_id,
    )

    if run_mode == RunMode.Train:
        response_model, investment_info = _train_response_model_invest_info(
            cur_date=cur_date,
            symbol=symbol,
            future_record=future_record,  # type: ignore
            short_memory=short_memory,
            short_memory_id=short_memory_id,
            mid_memory=mid_memory,
            mid_memory_id=mid_memory_id,
            long_memory=long_memory,
            long_memory_id=long_memory_id,
            reflection_memory=reflection_memory,
            reflection_memory_id=reflection_memory_id,
        )
        cur_prompt = train_prompt
    else:
        response_model, investment_info = _test_response_model_invest_info(
            cur_date=cur_date,
            symbol=symbol,
            short_memory=short_memory,
            short_memory_id=short_memory_id,
            mid_memory=mid_memory,
            mid_memory_id=mid_memory_id,
            long_memory=long_memory,
            long_memory_id=long_memory_id,
            reflection_memory=reflection_memory,
            reflection_memory_id=reflection_memory_id,
            momentum=momentum,
            build_response_model=not strict_output,
        )
        cur_prompt = test_prompt

    if strict_output:
        allowed_memory_ids = {
            "short_memory_index": list(short_memory_id),
            "middle_memory_index": list(mid_memory_id),
            "long_memory_index": list(long_memory_id),
            "reflection_memory_index": list(reflection_memory_id),
        }
        if structured_endpoint is not None:
            return _structured_causal_response(
                structured_endpoint=structured_endpoint,
                prompt=cur_prompt,
                investment_info=investment_info,
                allowed_memory_ids=allowed_memory_ids,
                requested_model=requested_model or "unknown",
                prompt_version=prompt_version or "unknown",
                provider=provider,
                interface=interface,
            )
        return _strict_causal_response(
            endpoint_func=endpoint_func,
            prompt=cur_prompt,
            investment_info=investment_info,
            allowed_memory_ids=allowed_memory_ids,
        )

    # prompt + validated output
    guard = gd.Guard.from_pydantic(
        output_class=response_model, prompt=cur_prompt, num_reasks=1
    )

    try:
        # , validated_output
        validated_outcomes = guard(
            endpoint_func,
            prompt_params={"investment_info": investment_info},
        )
        response_hashes = _raw_output_hashes(guard)
        if not strict_output:
            logger.info("Guardrails Raw LLM Outputs")
            for i, o in enumerate(guard.history[0].raw_outputs):
                logger.info(f"Reask {i}")
                logger.info(o)
                logger.info("\n\n")
        # print(guard.history.last.tree)
        if (validated_outcomes.validated_output is None) or (
            not isinstance(validated_outcomes.validated_output, dict)
        ):
            logger.info(f"reflection failed for {symbol}")
            if strict_output:
                raise CausalDecisionParseError(
                    "Causal LLM response did not match the strict BUY/HOLD/SELL schema",
                    response_hash=response_hashes[-1] if response_hashes else "unavailable",
                )
            if run_mode == RunMode.Train:
                return {"summary_reason": validated_outcomes.__dict__['reask'].__dict__['fail_results'][0].__dict__['error_message'], "short_memory_index": None, "middle_memory_index": None, "long_memory_index": None, "reflection_memory_index": None}
            else:
                return {"investment_decision" : "hold", "summary_reason": validated_outcomes.__dict__['reask'].__dict__['fail_results'][0].__dict__['error_message'], "short_memory_index": None, "middle_memory_index": None, "long_memory_index": None, "reflection_memory_index": None}
        result = _delete_placeholder_info(validated_outcomes.validated_output)
        if strict_output:
            result["_causal_parse_retries"] = max(0, len(response_hashes) - 1)
            result["_causal_response_hash"] = (
                response_hashes[-1] if response_hashes else "unavailable"
            )
        return result

    except Exception as e:
        if _find_exception(e, LongerThanContextError) is not None:
            raise LongerThanContextError from e
        if strict_output:
            from .chat import SanitizedChatBackendError

            backend_error = _find_exception(e, SanitizedChatBackendError)
            if backend_error is not None:
                raise backend_error
            if isinstance(e, CausalDecisionParseError):
                raise
            response_hashes = _raw_output_hashes(guard)
            raise CausalDecisionParseError(
                "Causal LLM response could not be parsed",
                response_hash=response_hashes[-1] if response_hashes else "unavailable",
            ) from e
        logger.info("Wrong again!!!!!")
        logger.error(e)
        return _delete_placeholder_info({})
