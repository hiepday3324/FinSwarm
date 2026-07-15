from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.allocator import ACTION_MULTIPLIERS, allocate
from demo.components import (
    agent_summary_html,
    inject_css,
    render_disclaimer,
    render_header,
    render_pipeline_strip,
    selected_head_matrix,
)
from demo.constants import DEFAULT_BUNDLE_DIR, DISPLAY_ORDER, NODE_ORDER, UNAVAILABLE
from demo.data_loader import ReplayStore
from demo.graph_visualization import (
    edge_records,
    largest_attention_changes,
    make_attention_heatmaps,
    make_network_figure,
)
from demo.inference import FrozenReplayEngine


st.set_page_config(
    page_title="FinSwarm E4 Historical Decision Replay",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading frozen E4 demo-fit bundle...")
def load_runtime() -> tuple[ReplayStore, FrozenReplayEngine]:
    return ReplayStore(DEFAULT_BUNDLE_DIR), FrozenReplayEngine(DEFAULT_BUNDLE_DIR)


def _symbol_index(symbol: str) -> int:
    return NODE_ORDER.index(symbol)


def _display_values(values: np.ndarray) -> dict[str, float]:
    return {symbol: float(values[_symbol_index(symbol)]) for symbol in DISPLAY_ORDER}


def _initialize_state(store: ReplayStore) -> None:
    scenarios = store.scenarios()
    default_index = store.date_index(scenarios[0]["date"]) if scenarios else len(store.dates) - 1
    defaults = {
        "date_index": default_index,
        "date_selector": store.dates[default_index].isoformat(),
        "scenario_selector": "Manual train date",
        "playing": False,
        "selected_symbol": None,
        "selected_news_group": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    pending_index = st.session_state.pop("pending_date_index", None)
    if pending_index is not None:
        bounded = max(0, min(int(pending_index), len(store.dates) - 1))
        st.session_state.date_index = bounded
        st.session_state.date_selector = store.dates[bounded].isoformat()


def _controls(store: ReplayStore, engine: FrozenReplayEngine) -> tuple[str, float]:
    scenarios = store.scenarios()
    scenario_by_label = {
        f"{item['central_symbol']}-centred | {item['date']}": item for item in scenarios
    }

    def scenario_changed() -> None:
        selected = scenario_by_label.get(st.session_state.scenario_selector)
        if selected is None:
            return
        index = store.date_index(selected["date"])
        st.session_state.date_index = index
        st.session_state.date_selector = store.dates[index].isoformat()
        st.session_state.playing = False
        st.session_state.selected_symbol = selected["central_symbol"]
        st.session_state.selected_news_group = None

    def date_changed() -> None:
        st.session_state.date_index = store.date_index(st.session_state.date_selector)
        st.session_state.playing = False
        st.session_state.scenario_selector = "Manual train date"
        st.session_state.selected_news_group = None

    def move(step: int) -> None:
        index = max(0, min(st.session_state.date_index + step, len(store.dates) - 1))
        st.session_state.date_index = index
        st.session_state.date_selector = store.dates[index].isoformat()
        st.session_state.playing = False
        st.session_state.scenario_selector = "Manual train date"
        st.session_state.selected_news_group = None

    def set_playing(value: bool) -> None:
        st.session_state.playing = value and st.session_state.date_index < len(store.dates) - 1

    scenario_options = ["Manual train date", *scenario_by_label]
    control_row = st.columns([2.0, 1.65, 0.75, 0.62, 0.62, 0.62, 1.35, 1.35])
    with control_row[0]:
        st.selectbox(
            "Curated scenario",
            scenario_options,
            key="scenario_selector",
            on_change=scenario_changed,
        )
    with control_row[1]:
        st.selectbox(
            "Train date",
            [value.isoformat() for value in store.dates],
            key="date_selector",
            on_change=date_changed,
        )
    with control_row[2]:
        st.write("")
        st.button("Previous", on_click=move, args=(-1,), width="stretch")
    with control_row[3]:
        st.write("")
        st.button("Play", on_click=set_playing, args=(True,), width="stretch")
    with control_row[4]:
        st.write("")
        st.button("Pause", on_click=set_playing, args=(False,), width="stretch")
    with control_row[5]:
        st.write("")
        st.button("Next", on_click=move, args=(1,), width="stretch")
    with control_row[6]:
        head_options = [
            "Mean across heads",
            *[f"Head {index + 1}" for index in range(engine.graph_model.num_heads)],
        ]
        head_label = st.selectbox("Attention head", head_options)
    with control_row[7]:
        capital = float(
            st.number_input(
                "Portfolio capital",
                min_value=0.0,
                max_value=100_000_000.0,
                value=1_000_000.0,
                step=50_000.0,
                format="%.0f",
            )
        )
    return head_label, capital


def _render_news_and_graph(
    store: ReplayStore,
    output: dict[str, Any],
    previous: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    head_label: str,
    previous_date: str | None,
) -> None:
    date = output["date"]
    news = store.news_for_date(date)
    selected_news = st.session_state.selected_news_group
    selected_symbols: set[str] = set()
    if selected_news is not None:
        match = next((item for item in news if item["news_group_id"] == selected_news), None)
        if match is not None:
            selected_symbols.update(match["agents_received"])
    if st.session_state.selected_symbol:
        selected_symbols.add(st.session_state.selected_symbol)

    current_attention = selected_head_matrix(output, head_label)
    previous_attention = selected_head_matrix(previous, head_label)
    probabilities = _display_values(output["fusion_probability"])
    actions = {symbol: str(records[symbol]["decision"]) for symbol in DISPLAY_ORDER}

    left, right = st.columns([0.82, 1.58], gap="large")
    with left:
        st.subheader("Historical News")
        with st.container(height=515, border=False):
            if not news:
                st.info("No causally included news events are stored for this replay date.")
            for index, item in enumerate(news):
                active = item["news_group_id"] == selected_news
                label = ("Selected | " if active else "") + str(item["headline"])
                if st.button(
                    label,
                    key=f"news_{date}_{index}_{item['news_group_id']}",
                    width="stretch",
                ):
                    st.session_state.selected_news_group = (
                        None if active else item["news_group_id"]
                    )
                    st.session_state.selected_symbol = None
                    st.rerun()
                routed = ", ".join(item["agents_received"]) or UNAVAILABLE
                not_routed = ", ".join(
                    symbol for symbol in DISPLAY_ORDER if symbol not in item["agents_received"]
                ) or "None"
                published = item["published_at"].isoformat() if item["published_at"] else UNAVAILABLE
                cutoff = str(item.get("decision_cutoff") or UNAVAILABLE)
                causal = "Causally available" if item["causally_available"] else "Unavailable at cutoff"
                st.markdown(
                    f"""
                    <div class="section-band">
                      <strong>{item['source']}</strong><br>
                      <span class="news-meta">Published: {published}<br>
                      Decision cutoff: {cutoff}<br>
                      Related symbols: {', '.join(item['related_symbols']) or UNAVAILABLE}<br>
                      Routed to: {routed}<br>Not routed to: {not_routed}<br>
                      Status: {causal}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    with right:
        st.subheader("GAT Attention Network")
        figure = make_network_figure(
            current_attention,
            previous_attention,
            NODE_ORDER,
            probabilities=probabilities,
            actions=actions,
            selected_symbols=selected_symbols,
            head_label=head_label,
            replay_date=date,
            previous_date=previous_date,
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        st.caption(
            "Rows are targets; columns are sources. Each displayed alpha is a "
            "sample-specific attention coefficient from frozen trained GAT parameters."
        )


def _agent_context_sections(
    store: ReplayStore,
    date: str,
    record: dict[str, Any],
) -> None:
    context = record["context_snapshot"].get("agent_context", {})
    event_ids = set(record["news_event_ids"])
    included_news = [
        item
        for item in store.news_for_date(date)
        if event_ids.intersection(item["event_ids"])
    ]
    memories = record["memory_refs"]
    with st.expander("Agent context and provenance", expanded=False):
        st.markdown("**Included News**")
        if included_news:
            for item in included_news:
                st.markdown(f"- {item['headline']} ({item['source']})")
        else:
            st.write("No included news stored for this Agent/date.")
        st.markdown("**Retrieved Memory**")
        if memories:
            for item in memories[:12]:
                layer = item.get("layer", UNAVAILABLE)
                preview = str(item.get("text_preview") or UNAVAILABLE)
                st.markdown(f"- `{layer}` | {preview[:320]}")
        else:
            st.write("No retrieved memory stored.")
        st.markdown("**Psychology**")
        st.json(record["psychology_state"] or {"status": UNAVAILABLE}, expanded=False)
        st.markdown("**Prompt**")
        st.warning("Exact final LLM prompt: Unavailable in artifact")
        st.caption("Stored canonical Text Stream input preview")
        st.code(str(record["text_input"])[:5000], language="text")
        st.markdown("**Full Reasoning**")
        st.write(record["reason"])
        st.markdown("**Raw AgentOutput**")
        st.json(record["raw_agent_output"], expanded=False)
        st.markdown("**Provenance**")
        st.json(record["causal_provenance"], expanded=False)


def _render_agents(
    store: ReplayStore,
    output: dict[str, Any],
    previous: dict[str, Any] | None,
    records: dict[str, dict[str, Any]],
) -> None:
    st.subheader("Sector Agents")
    columns = st.columns(3, gap="medium")
    for column, symbol in zip(columns, DISPLAY_ORDER):
        node = _symbol_index(symbol)
        current_probability = float(output["text_probability"][node])
        prior_probability = (
            float(previous["text_probability"][node]) if previous is not None else current_probability
        )
        record = records[symbol]
        psychology = record["psychology_state"] or {}
        psychology_name = str(
            psychology.get("state", psychology.get("psychology", UNAVAILABLE))
        )
        with column:
            if st.button(
                f"Select {symbol}",
                key=f"agent_select_{output['date']}_{symbol}",
                width="stretch",
            ):
                st.session_state.selected_symbol = (
                    None if st.session_state.selected_symbol == symbol else symbol
                )
                st.session_state.selected_news_group = None
                st.rerun()
            st.markdown(
                agent_summary_html(
                    symbol=symbol,
                    action=str(record["decision"]),
                    reason=str(record["reason"]),
                    psychology=psychology_name,
                    news_count=len(record["news_event_ids"]),
                    memory_count=len(record["memory_refs"]),
                    text_probability=current_probability,
                    text_delta=current_probability - prior_probability,
                    selected=st.session_state.selected_symbol == symbol,
                ),
                unsafe_allow_html=True,
            )
            _agent_context_sections(store, output["date"], record)


def _render_stream_outputs(
    output: dict[str, Any],
    previous: dict[str, Any] | None,
) -> None:
    text_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    for symbol in DISPLAY_ORDER:
        node = _symbol_index(symbol)
        text_probability = float(output["text_probability"][node])
        graph_probability = float(output["graph_probability"][node])
        previous_text = (
            float(previous["text_probability"][node]) if previous is not None else np.nan
        )
        previous_graph = (
            float(previous["graph_probability"][node]) if previous is not None else np.nan
        )
        text_rows.append(
            {
                "Symbol": symbol,
                "Text probability": text_probability,
                "Text logit": float(output["text_logits"][node]),
                "Previous probability": previous_text,
                "Probability delta": text_probability - previous_text,
                "Score status": "in_sample_demo",
            }
        )
        graph_rows.append(
            {
                "Symbol": symbol,
                "Graph probability": graph_probability,
                "Graph logit": float(output["graph_logits"][node]),
                "Previous probability": previous_graph,
                "Probability delta": graph_probability - previous_graph,
                "Message passing": "full topology",
            }
        )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Text Stream predictive score")
        st.dataframe(
            pd.DataFrame(text_rows).set_index("Symbol").style.format(
                {
                    "Text probability": "{:.3f}",
                    "Text logit": "{:+.3f}",
                    "Previous probability": "{:.3f}",
                    "Probability delta": "{:+.3f}",
                }
            ),
            width="stretch",
        )
    with right:
        st.subheader("Graph Stream output")
        st.caption("Produced after cross-stock message passing.")
        st.dataframe(
            pd.DataFrame(graph_rows).set_index("Symbol").style.format(
                {
                    "Graph probability": "{:.3f}",
                    "Graph logit": "{:+.3f}",
                    "Previous probability": "{:.3f}",
                    "Probability delta": "{:+.3f}",
                }
            ),
            width="stretch",
        )


def _render_attention_detail(
    output: dict[str, Any],
    previous: dict[str, Any],
    *,
    head_label: str,
    previous_date: str | None,
) -> None:
    current_matrix = selected_head_matrix(output, head_label)
    previous_matrix = selected_head_matrix(previous, head_label)
    st.subheader("Attention Shift Between Replay Dates")
    st.plotly_chart(
        make_attention_heatmaps(
            previous_matrix, current_matrix, NODE_ORDER, head_label=head_label
        ),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption("Rows: targets | Columns: sources | Softmax normalization axis: sources")
    changes = largest_attention_changes(
        edge_records(
            current_matrix,
            previous_matrix,
            NODE_ORDER,
            head_label=head_label,
            replay_date=output["date"],
            previous_date=previous_date,
        )
    )
    table = pd.DataFrame(
        [
            {
                "Edge": f"{item['source']} -> {item['target']}",
                "Previous": item["previous_alpha"],
                "Current": item["current_alpha"],
                "Delta": item["delta"],
                "Absolute delta": item["absolute_delta"],
                "Head": item["attention_head"],
            }
            for item in changes
        ]
    )
    st.dataframe(
        table.style.format(
            {"Previous": "{:.4f}", "Current": "{:.4f}", "Delta": "{:+.4f}", "Absolute delta": "{:.4f}"}
        ),
        width="stretch",
        hide_index=True,
    )


def _feature_frame(output: dict[str, Any]) -> pd.DataFrame:
    rows = {
        symbol: output["raw_graph_features"][_symbol_index(symbol)]
        for symbol in DISPLAY_ORDER
    }
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=list(output["graph_feature_names"])
    )


def _render_graph_features(
    output: dict[str, Any],
    previous: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    st.subheader("Graph Input Features")
    current = pd.DataFrame(
        [output["raw_graph_features"][_symbol_index(symbol)] for symbol in DISPLAY_ORDER],
        index=DISPLAY_ORDER,
        columns=schema["feature_names"],
    )
    before = pd.DataFrame(
        [previous["raw_graph_features"][_symbol_index(symbol)] for symbol in DISPLAY_ORDER],
        index=DISPLAY_ORDER,
        columns=schema["feature_names"],
    )
    delta = (current - before).abs()
    largest = (
        delta.stack()
        .sort_values(ascending=False)
        .head(6)
        .reset_index()
        .rename(columns={"level_0": "Symbol", "level_1": "Feature", 0: "Absolute change"})
    )
    st.dataframe(
        largest.style.format({"Absolute change": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )
    blocks = schema["feature_blocks"]
    feature_groups = (
        ("Market features", list(blocks["market_features"])),
        (
            "Agent, Text and shadow priors",
            list(blocks["action_signal"])
            + list(blocks["text_score"])
            + list(blocks["prior_shadow_features"]),
        ),
        ("Psychology", list(blocks["prior_psychology_features"])),
    )
    for title, names in feature_groups:
        st.markdown(f"**{title}**")
        local_delta = delta[names]
        threshold = float(np.quantile(local_delta.to_numpy(), 0.80))

        def highlight(_: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                np.where(
                    local_delta.to_numpy() >= threshold,
                    "background-color: #fff0d1; color: #493711",
                    "",
                ),
                index=current.index,
                columns=names,
            )

        st.dataframe(
            current[names].style.apply(highlight, axis=None).format("{:+.4f}"),
            width="stretch",
        )


def _render_fusion_and_allocation(
    engine: FrozenReplayEngine,
    output: dict[str, Any],
    previous: dict[str, Any] | None,
    records: dict[str, dict[str, Any]],
    previous_records: dict[str, dict[str, Any]] | None,
    *,
    capital: float,
) -> None:
    coefficients = engine.coefficients
    st.subheader("Scalar Fusion")
    st.markdown(
        f"""
        <div class="equation">fusion_logit = {coefficients['intercept']:+.6f}
        + ({coefficients['weight_text']:+.6f} * text_logit)
        + ({coefficients['weight_graph']:+.6f} * graph_logit)</div>
        """,
        unsafe_allow_html=True,
    )
    current_rows = {row["symbol"]: row for row in engine.fusion_decomposition(output)}
    prior_rows = (
        {row["symbol"]: row for row in engine.fusion_decomposition(previous)}
        if previous is not None
        else {}
    )
    fusion_table: list[dict[str, Any]] = []
    for symbol in DISPLAY_ORDER:
        row = current_rows[symbol]
        previous_probability = (
            float(prior_rows[symbol]["fusion_probability"]) if symbol in prior_rows else np.nan
        )
        fusion_table.append(
            {
                "Symbol": symbol,
                "Intercept": row["intercept_contribution"],
                "Text term": row["text_contribution"],
                "Graph term": row["graph_contribution"],
                "Fusion logit": row["fusion_logit"],
                "Fusion probability": row["fusion_probability"],
                "Previous probability": previous_probability,
                "Probability delta": row["fusion_probability"] - previous_probability,
            }
        )
    st.dataframe(
        pd.DataFrame(fusion_table).set_index("Symbol").style.format(
            {
                "Intercept": "{:+.4f}",
                "Text term": "{:+.4f}",
                "Graph term": "{:+.4f}",
                "Fusion logit": "{:+.4f}",
                "Fusion probability": "{:.3f}",
                "Previous probability": "{:.3f}",
                "Probability delta": "{:+.3f}",
            }
        ),
        width="stretch",
    )

    actions = {symbol: str(records[symbol]["decision"]) for symbol in DISPLAY_ORDER}
    probabilities = {
        symbol: float(output["fusion_probability"][_symbol_index(symbol)])
        for symbol in DISPLAY_ORDER
    }
    allocation = allocate(actions, probabilities, capital=capital)
    if previous is not None:
        if previous_records is None:
            raise ValueError("Previous replay output requires aligned Agent records")
        previous_probabilities = {
            symbol: float(previous["fusion_probability"][_symbol_index(symbol)])
            for symbol in DISPLAY_ORDER
        }
        previous_actions = {
            symbol: str(previous_records[symbol]["decision"])
            for symbol in DISPLAY_ORDER
        }
        previous_allocation = allocate(
            previous_actions, previous_probabilities, capital=capital
        )
    else:
        previous_allocation = allocation

    st.subheader("Rule-based demo allocation")
    st.caption("Not a trained Master Allocator")
    assets = [*DISPLAY_ORDER, "Cash"]
    current_weights = [
        *[float(allocation["weights"][symbol]) for symbol in DISPLAY_ORDER],
        float(allocation["cash_weight"]),
    ]
    previous_weights = [
        *[float(previous_allocation["weights"][symbol]) for symbol in DISPLAY_ORDER],
        float(previous_allocation["cash_weight"]),
    ]
    chart = go.Figure()
    chart.add_trace(
        go.Bar(
            x=assets,
            y=previous_weights,
            name="Previous date",
            marker_color="#AAB4B3",
            text=[f"{value:.1%}" for value in previous_weights],
            textposition="outside",
            textfont={"color": "#172126"},
        )
    )
    chart.add_trace(
        go.Bar(
            x=assets,
            y=current_weights,
            name="Current date",
            marker_color=["#0F766E", "#D95D39", "#C08B2D", "#39464D"],
            text=[f"{value:.1%}" for value in current_weights],
            textposition="outside",
            textfont={"color": "#172126"},
        )
    )
    chart.update_layout(
        barmode="group",
        height=340,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"tickfont": {"color": "#627078"}},
        yaxis={
            "tickformat": ".0%",
            "range": [0.0, max(0.55, max(current_weights) * 1.2)],
            "tickfont": {"color": "#627078"},
            "gridcolor": "#E4E9E8",
        },
        legend={"orientation": "h", "y": 1.08, "font": {"color": "#172126"}},
        font={"color": "#172126"},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    previous_weight_map = dict(zip(assets, previous_weights))
    allocation_rows: list[dict[str, Any]] = []
    for row in allocation["rows"]:
        allocation_rows.append(
            {
                "Symbol": row.symbol,
                "Agent action": row.action.upper(),
                "Fusion probability": row.fusion_probability,
                "Action multiplier": row.action_multiplier,
                "Raw score": row.raw_score,
                "Final weight": row.final_weight,
                "Dollar allocation": row.dollar_allocation,
                "Previous weight": previous_weight_map[row.symbol],
                "Weight delta": row.final_weight - previous_weight_map[row.symbol],
            }
        )
    allocation_rows.append(
        {
            "Symbol": "Cash",
            "Agent action": "N/A",
            "Fusion probability": np.nan,
            "Action multiplier": np.nan,
            "Raw score": np.nan,
            "Final weight": allocation["cash_weight"],
            "Dollar allocation": allocation["cash_dollars"],
            "Previous weight": previous_weight_map["Cash"],
            "Weight delta": allocation["cash_weight"] - previous_weight_map["Cash"],
        }
    )
    st.dataframe(
        pd.DataFrame(allocation_rows).set_index("Symbol").style.format(
            {
                "Fusion probability": "{:.3f}",
                "Action multiplier": "{:.1f}",
                "Raw score": "{:.4f}",
                "Final weight": "{:.1%}",
                "Dollar allocation": "${:,.0f}",
                "Previous weight": "{:.1%}",
                "Weight delta": "{:+.1%}",
            },
            na_rep="N/A",
        ),
        width="stretch",
    )


def main() -> None:
    inject_css()
    store, engine = load_runtime()
    _initialize_state(store)
    render_header()
    head_label, capital = _controls(store, engine)
    render_pipeline_strip()

    index = int(st.session_state.date_index)
    date = store.dates[index]
    previous_index = index - 1 if index > 0 else index
    previous_date = store.dates[index - 1].isoformat() if index > 0 else None
    output = engine.infer_date(index)
    previous_for_attention = engine.infer_date(previous_index)
    previous_output = previous_for_attention if index > 0 else None
    records = {row["symbol"]: row for row in store.records_for_date(date)}
    previous_records = (
        {
            row["symbol"]: row
            for row in store.records_for_date(store.dates[index - 1])
        }
        if index > 0
        else None
    )
    expected_ids = store.index["decision_ids_by_date"][date.isoformat()]
    if output["decision_ids"].astype(str).tolist() != expected_ids:
        raise RuntimeError("Replay date panels are not aligned to the same decision IDs")

    top_metrics = st.columns(4)
    top_metrics[0].metric("Replay date", date.isoformat())
    top_metrics[1].metric("Previous train date", previous_date or UNAVAILABLE)
    top_metrics[2].metric("Aligned decisions", len(output["decision_ids"]))
    top_metrics[3].metric(
        "Model state",
        "FROZEN" if engine.state_hashes() == engine.startup_state_hashes else "CHANGED",
    )

    _render_news_and_graph(
        store,
        output,
        previous_for_attention,
        records,
        head_label=head_label,
        previous_date=previous_date,
    )
    _render_agents(store, output, previous_output, records)
    _render_stream_outputs(output, previous_output)
    _render_attention_detail(
        output,
        previous_for_attention,
        head_label=head_label,
        previous_date=previous_date,
    )
    _render_graph_features(output, previous_for_attention, store.feature_schema)
    _render_fusion_and_allocation(
        engine,
        output,
        previous_output,
        records,
        previous_records,
        capital=capital,
    )
    render_disclaimer()
    st.caption(
        "Frozen model state hashes | Text: "
        f"{engine.startup_state_hashes['text_state_sha256'][:12]} | GAT: "
        f"{engine.startup_state_hashes['gat_state_sha256'][:12]} | Scalar Fusion: "
        f"{engine.startup_state_hashes['scalar_fusion_state_sha256'][:12]}"
    )

    if st.session_state.playing:
        if index >= len(store.dates) - 1:
            st.session_state.playing = False
        else:
            time.sleep(2.0)
            st.session_state.pending_date_index = index + 1
            st.rerun()


if __name__ == "__main__":
    main()
