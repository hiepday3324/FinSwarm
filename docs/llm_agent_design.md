# LLM Agent Design for FINMEM Swarm

This document describes the MVP upgrade path for Tien's FINMEM Swarm work.

## Goal

Keep the original FinMem `LLMAgent` as the single-agent baseline, then wrap it
with `SectorAgent` so the Swarm system can consume stable artifacts instead of
internal mutable objects.

## Artifact Contracts

The Swarm layer communicates through Pydantic schemas in `puppy.common.schemas`:

- `AgentContext`: memory, shared context, shadow state, and psychology state.
- `AgentSignal`: local `buy`/`hold`/`sell` decision and cited memory.
- `AgentOutput`: full SectorAgent output including raw LLM reflection.
- `MemoryShareRoute` and `MemoryShareEvent`: graph-triggered context sharing.
- `DebateRequest`, `DebateTranscript`, and `DebateVerdict`: conflict handling.

## Runtime Flow

1. `SectorAgent.step()` calls the original `LLMAgent.step()`.
2. `SectorAgent` converts the raw reflection into `AgentOutput`.
3. `ContextBuilder` creates deterministic text context for text features.
4. `TextStreamEncoder` converts `AgentContext` into `h_text`.
5. `AllocatorAgent` aggregates local signals without managing real capital.
6. `MemorySharingService` injects graph-routed shared context.
7. `DebateLoop` and `DebateJudge` handle grey-zone or text/graph conflicts.

## Guardrails

- Do not rewrite `LLMAgent`, `reflection.py`, or `prompts.py` in the MVP.
- Treat SectorAgent decisions as local signals, not executable orders.
- Keep shared memory in external context before promoting it into durable memory.
- Do not use future returns or future shadow portfolio state in current context.
