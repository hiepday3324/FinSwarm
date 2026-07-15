from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


STATUS_BADGES = (
    "IN-SAMPLE TRAIN REPLAY",
    "E4 DEMO-FIT",
    "MODEL FROZEN",
    "NO LIVE TRAINING",
    "TRAIN PARTITION ONLY",
    "NO VALIDATION ACCESS",
    "NO TEST ACCESS",
)

PIPELINE_STEPS = (
    "1. Historical News",
    "2. Sector Agents",
    "3. Text Stream",
    "4. GAT Attention",
    "5. Scalar Fusion",
    "6. Capital Allocation",
)

DISCLAIMER = (
    "This E4 demo-fit model was trained on the same historical train period being "
    "replayed. The interface demonstrates FinSwarm's information flow, Sector Agent "
    "reasoning, Text scoring, dynamic GAT attention and capital allocation. It is not "
    "an out-of-sample performance, profitability or trading-alpha claim."
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #172126;
          --muted: #627078;
          --line: #d9e0df;
          --paper: #ffffff;
          --wash: #f5f7f6;
          --teal: #0f766e;
          --coral: #d95d39;
          --gold: #c08b2d;
        }
        .stApp { background: #eef2f1; color: var(--ink); }
        .block-container { max-width: 1460px; padding-top: 1.1rem; padding-bottom: 2.2rem; }
        h1, h2, h3 { letter-spacing: 0 !important; color: var(--ink); }
        h2 { font-size: 1.45rem !important; margin-top: 1rem !important; }
        h3 { font-size: 1.02rem !important; }
        .hero-line { height: 5px; background: var(--teal); border-right: 180px solid var(--coral); margin-bottom: 16px; }
        .replay-title { font-size: 2.05rem; line-height: 1.1; font-weight: 760; margin: 0; }
        .replay-subtitle { color: var(--muted); font-size: 1rem; margin: 7px 0 12px; }
        .badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }
        .status-badge { border: 1px solid var(--line); border-radius: 4px; background: var(--paper); padding: 5px 8px; font-size: 0.70rem; font-weight: 750; color: var(--ink); }
        .status-badge:nth-child(1), .status-badge:nth-child(3) { border-color: var(--teal); color: var(--teal); }
        .status-badge:nth-child(2), .status-badge:nth-child(4) { border-color: var(--coral); color: var(--coral); }
        .pipeline-strip { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 5px; margin: 10px 0 18px; }
        .pipeline-step { background: var(--paper); border-top: 3px solid var(--teal); border-left: 1px solid var(--line); border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 9px 6px; text-align: center; font-size: 0.78rem; font-weight: 700; min-height: 44px; }
        .pipeline-step:nth-child(2), .pipeline-step:nth-child(5) { border-top-color: var(--coral); }
        .pipeline-step:nth-child(3), .pipeline-step:nth-child(6) { border-top-color: var(--gold); }
        .section-band { background: var(--paper); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; margin: 7px 0 10px; }
        .agent-summary { background: var(--paper); border: 1px solid var(--line); border-radius: 6px; padding: 12px; min-height: 238px; }
        .agent-summary.selected { border: 2px solid var(--teal); padding: 11px; }
        .agent-symbol { font-size: 1.2rem; font-weight: 780; }
        .action-buy { color: var(--teal); font-weight: 800; }
        .action-hold { color: var(--gold); font-weight: 800; }
        .action-sell { color: var(--coral); font-weight: 800; }
        .small-label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; font-weight: 750; margin-top: 7px; }
        .small-value { color: var(--ink); font-size: 0.88rem; }
        .news-meta { color: var(--muted); font-size: 0.76rem; line-height: 1.45; }
        .disclaimer { background: #fff8e8; border-left: 5px solid var(--gold); padding: 11px 14px; margin: 8px 0 14px; color: #4e4020; }
        .disclaimer strong { display: block; color: #7a5d10; margin-bottom: 3px; }
        .equation { background: #172126; color: #ffffff; padding: 12px 14px; border-radius: 5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88rem; }
        [data-testid="stMetric"] { background: var(--paper); border: 1px solid var(--line); padding: 8px 10px; border-radius: 6px; }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * { color: var(--muted) !important; }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * { color: var(--ink) !important; }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] * { color: var(--ink) !important; font-weight: 700; }
        .stButton > button { background: var(--ink); color: #ffffff !important; border: 1px solid var(--ink); }
        .stButton > button:hover { background: var(--teal); color: #ffffff !important; border-color: var(--teal); }
        .stButton > button *,
        .stButton > button:hover * { color: #ffffff !important; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }
        @media (max-width: 900px) {
          .pipeline-strip { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          .replay-title { font-size: 1.55rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    badges = "".join(
        f'<span class="status-badge">{html.escape(value)}</span>'
        for value in STATUS_BADGES
    )
    st.markdown(
        f"""
        <div class="hero-line"></div>
        <h1 class="replay-title">FinSwarm &mdash; E4 Historical Decision Replay</h1>
        <p class="replay-subtitle">Frozen-model replay of news, Agent reasoning, dynamic Graph attention and capital allocation</p>
        <div class="badge-row">{badges}</div>
        """,
        unsafe_allow_html=True,
    )
    render_disclaimer()


def render_pipeline_strip() -> None:
    content = "".join(
        f'<div class="pipeline-step">{html.escape(step)}</div>'
        for step in PIPELINE_STEPS
    )
    st.markdown(f'<div class="pipeline-strip">{content}</div>', unsafe_allow_html=True)


def render_disclaimer() -> None:
    st.markdown(
        f'<div class="disclaimer"><strong>IN-SAMPLE HISTORICAL TRAIN REPLAY</strong>{html.escape(DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )


def action_class(action: str) -> str:
    normalized = str(action).lower()
    return normalized if normalized in {"buy", "hold", "sell"} else "hold"


def agent_summary_html(
    *,
    symbol: str,
    action: str,
    reason: str,
    psychology: str,
    news_count: int,
    memory_count: int,
    text_probability: float,
    text_delta: float,
    selected: bool,
) -> str:
    safe_reason = html.escape(reason[:310] + ("..." if len(reason) > 310 else ""))
    selected_class = " selected" if selected else ""
    action_name = action_class(action)
    return f"""
    <div class="agent-summary{selected_class}">
      <div class="agent-symbol">{html.escape(symbol)} Sector Agent</div>
      <div class="action-{action_name}">{html.escape(action.upper())}</div>
      <div class="small-label">Confidence</div><div class="small-value">Unavailable in artifact</div>
      <div class="small-label">Concise reasoning</div><div class="small-value">{safe_reason}</div>
      <div class="small-label">Psychology</div><div class="small-value">{html.escape(psychology)}</div>
      <div class="small-label">Context counts</div><div class="small-value">{news_count} news &nbsp; | &nbsp; {memory_count} memories</div>
      <div class="small-label">Text Stream predictive score</div><div class="small-value">{text_probability:.3f} ({text_delta:+.3f} vs previous)</div>
    </div>
    """


def selected_head_matrix(output: dict, label: str):
    if label == "Mean across heads":
        return output["attention_mean"]
    index = int(label.split()[-1]) - 1
    return output["attention_heads"][index]


def badges_inline(values: Iterable[str]) -> str:
    return "".join(
        f'<span class="status-badge">{html.escape(value)}</span>' for value in values
    )
