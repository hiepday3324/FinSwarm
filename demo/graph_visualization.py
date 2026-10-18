from __future__ import annotations

import math
from itertools import permutations
from typing import Any, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODE_CURRENT = "Current Attention"
MODE_DELTA = "Delta vs Previous"
MODE_DOMINANT = "Dominant Path"
DISPLAY_MODES = (MODE_CURRENT, MODE_DELTA, MODE_DOMINANT)
DELTA_NEUTRAL_THRESHOLD = 0.02

NODE_POSITIONS = {
    "AAPL": (-0.86, 0.64),
    "NVDA": (0.86, 0.64),
    "TSLA": (0.0, -0.68),
}
ACTION_COLORS = {"BUY": "#0F766E", "HOLD": "#B98218", "SELL": "#C2412D"}
EDGE_INCREASE = "#138A63"
EDGE_DECREASE = "#C2412D"
EDGE_STABLE = "#8B989D"
EDGE_DIM = "#B6C0C3"


def _as_matrix(values: np.ndarray, symbols: Sequence[str], *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    expected = (len(symbols), len(symbols))
    if matrix.shape != expected:
        raise ValueError(f"{name} attention matrix must have shape {expected}")
    return matrix


def attention_trend(delta: float, threshold: float = DELTA_NEUTRAL_THRESHOLD) -> str:
    if delta > threshold:
        return "Increased"
    if delta < -threshold:
        return "Decreased"
    return "Stable"


def edge_color(delta: float, threshold: float = DELTA_NEUTRAL_THRESHOLD) -> str:
    trend = attention_trend(delta, threshold=threshold)
    if trend == "Increased":
        return EDGE_INCREASE
    if trend == "Decreased":
        return EDGE_DECREASE
    return EDGE_STABLE


def edge_records(
    current: np.ndarray,
    previous: np.ndarray,
    symbols: Iterable[str],
    *,
    head_label: str,
    replay_date: str,
    previous_date: str | None,
) -> list[dict[str, Any]]:
    names = tuple(symbols)
    current_values = _as_matrix(current, names, name="Current")
    previous_values = _as_matrix(previous, names, name="Previous")
    records: list[dict[str, Any]] = []
    for target_index, target in enumerate(names):
        for source_index, source in enumerate(names):
            before = float(previous_values[target_index, source_index])
            now = float(current_values[target_index, source_index])
            delta = now - before
            percentage = None if abs(before) < 1e-9 else 100.0 * delta / abs(before)
            records.append(
                {
                    "source": source,
                    "target": target,
                    "edge": (source, target),
                    "previous_alpha": before,
                    "current_alpha": now,
                    "delta": delta,
                    "absolute_delta": abs(delta),
                    "percentage_delta": percentage,
                    "trend": attention_trend(delta),
                    "is_self_loop": source == target,
                    "attention_head": head_label,
                    "replay_date": replay_date,
                    "previous_replay_date": previous_date,
                }
            )
    for rank, item in enumerate(
        sorted(records, key=lambda value: value["current_alpha"], reverse=True), start=1
    ):
        item["current_rank"] = rank
    for rank, item in enumerate(
        sorted(records, key=lambda value: value["absolute_delta"], reverse=True), start=1
    ):
        item["absolute_delta_rank"] = rank
    return records


def largest_attention_changes(
    records: list[dict[str, Any]], limit: int = 3
) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: item["absolute_delta"], reverse=True)[:limit]


def attention_views(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before = np.asarray(previous, dtype=np.float32)
    now = np.asarray(current, dtype=np.float32)
    if before.shape != now.shape:
        raise ValueError("Previous and current attention matrices must align")
    return before, now, now - before


def dominant_attention_path(
    current: np.ndarray,
    symbols: Iterable[str],
) -> dict[str, Any]:
    names = tuple(symbols)
    matrix = _as_matrix(current, names, name="Current")
    index = {symbol: offset for offset, symbol in enumerate(names)}
    best: dict[str, Any] | None = None
    for length in (2, 3):
        for path in permutations(names, length):
            values: list[float] = []
            edges: list[tuple[str, str]] = []
            for source, target in zip(path[:-1], path[1:]):
                value = float(matrix[index[target], index[source]])
                values.append(value)
                edges.append((source, target))
            score = float(np.prod(values, dtype=np.float64))
            candidate = {
                "symbols": list(path),
                "edges": edges,
                "coefficients": values,
                "score": score,
            }
            if best is None or (score, " -> ".join(path)) > (
                float(best["score"]),
                " -> ".join(best["symbols"]),
            ):
                best = candidate
    if best is None:
        raise ValueError("Dominant path requires at least two distinct symbols")
    return best


def _rounded_rect_path(
    x: float,
    y: float,
    width: float,
    height: float,
    radius: float,
) -> str:
    left, right = x - width / 2, x + width / 2
    top, bottom = y + height / 2, y - height / 2
    r = min(radius, width / 2, height / 2)
    return (
        f"M {left + r},{top} L {right - r},{top} "
        f"Q {right},{top} {right},{top - r} L {right},{bottom + r} "
        f"Q {right},{bottom} {right - r},{bottom} L {left + r},{bottom} "
        f"Q {left},{bottom} {left},{bottom + r} L {left},{top - r} "
        f"Q {left},{top} {left + r},{top} Z"
    )


def _quadratic_points(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    count: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, count)
    x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
    y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
    return x, y


def _cross_edge_points(source: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    sx, sy = NODE_POSITIONS[source]
    tx, ty = NODE_POSITIONS[target]
    dx, dy = tx - sx, ty - sy
    length = max(math.hypot(dx, dy), 1e-9)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    card_radius = 0.31
    start = (sx + ux * card_radius, sy + uy * card_radius)
    end = (tx - ux * card_radius, ty - uy * card_radius)
    direction_sign = 1.0 if source < target else -1.0
    bend = 0.18 * direction_sign
    control = ((sx + tx) / 2 + px * bend, (sy + ty) / 2 + py * bend)
    return _quadratic_points(start, control, end)


def _self_edge_points(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = NODE_POSITIONS[symbol]
    angle = np.linspace(math.pi * 0.04, math.pi * 1.96, 40)
    offset_y = 0.30 if symbol != "TSLA" else -0.30
    center = (x, y + offset_y)
    return center[0] + 0.24 * np.cos(angle), center[1] + 0.15 * np.sin(angle)


def _hex_to_rgba(color: str, opacity: float) -> str:
    color = color.lstrip("#")
    red, green, blue = (int(color[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{max(0.0, min(1.0, opacity)):.3f})"


def edge_visual_attributes(
    record: dict[str, Any],
    *,
    mode: str,
    path_edges: set[tuple[str, str]] | None = None,
    selected_symbols: set[str] | None = None,
) -> dict[str, Any]:
    if mode not in DISPLAY_MODES:
        raise ValueError(f"Unsupported influence-map mode: {mode}")
    selected = set(selected_symbols or ())
    path = set(path_edges or ())
    source = str(record["source"])
    target = str(record["target"])
    connected_to_selected = not selected or source in selected or target in selected
    on_path = (source, target) in path
    is_self_loop = bool(record["is_self_loop"])
    coefficient = float(record["current_alpha"])
    delta = float(record["delta"])
    magnitude = coefficient if mode != MODE_DELTA else float(record["absolute_delta"])
    width = 1.15 + (7.2 * magnitude)
    if mode == MODE_DELTA:
        width = 0.9 + (18.0 * magnitude)
    if is_self_loop:
        width *= 0.52
    highlighted = connected_to_selected
    if mode == MODE_DOMINANT:
        highlighted = on_path
    opacity = 0.22 + min(0.70, 0.78 * coefficient)
    if mode == MODE_DELTA and record["trend"] == "Stable":
        opacity *= 0.52
    if not highlighted:
        opacity *= 0.28
    if is_self_loop:
        opacity *= 0.70
    color = edge_color(delta)
    if not highlighted:
        color = EDGE_DIM
    return {
        "width": max(0.65, min(7.5, width)),
        "color": color,
        "opacity": max(0.08, min(0.95, opacity)),
        "dash": "solid" if highlighted else "dot",
        "highlighted": highlighted,
        "on_path": on_path,
    }


def top_influence_rows(
    records: list[dict[str, Any]],
    *,
    mode: str,
    path_edges: set[tuple[str, str]] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if mode == MODE_CURRENT:
        ordered = sorted(records, key=lambda item: item["current_alpha"], reverse=True)
    elif mode == MODE_DELTA:
        ordered = sorted(records, key=lambda item: item["absolute_delta"], reverse=True)
    elif mode == MODE_DOMINANT:
        path = set(path_edges or ())
        ordered = sorted(
            records,
            key=lambda item: (
                0 if item["edge"] in path else 1,
                -float(item["current_alpha"]),
            ),
        )
    else:
        raise ValueError(f"Unsupported influence-map mode: {mode}")
    return ordered[:limit]


def attention_summary(
    records: list[dict[str, Any]],
    *,
    mode: str,
    path: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    strongest = max(records, key=lambda item: item["current_alpha"])
    increases = [item for item in records if item["delta"] > DELTA_NEUTRAL_THRESHOLD]
    decreases = [item for item in records if item["delta"] < -DELTA_NEUTRAL_THRESHOLD]
    stable_count = sum(item["trend"] == "Stable" for item in records)
    largest_up = max(increases, key=lambda item: item["delta"], default=None)
    largest_down = min(decreases, key=lambda item: item["delta"], default=None)
    cross_edges = [item for item in records if not item["is_self_loop"]]
    strongest_cross = max(cross_edges, key=lambda item: item["current_alpha"])
    statements = [
        f"{strongest['target']} receives the highest current attention from {strongest['source']}.",
        (
            f"{strongest_cross['source']} -> {strongest_cross['target']} is the strongest "
            "cross-stock edge."
        ),
    ]
    if largest_up is not None:
        statements.append(
            f"{largest_up['source']} -> {largest_up['target']} increased the most versus the previous date."
        )
    if largest_down is not None:
        statements.append(
            f"{largest_down['source']} -> {largest_down['target']} decreased the most versus the previous date."
        )
    if mode == MODE_DOMINANT:
        statements = [
            *statements[:3],
            f"The dominant attention path is {' -> '.join(path['symbols'])}.",
        ]
    else:
        statements = statements[:4]
    stats = {
        "strongest_current_edge": f"{strongest['source']} -> {strongest['target']}",
        "largest_increase": (
            "None"
            if largest_up is None
            else f"{largest_up['source']} -> {largest_up['target']} ({largest_up['delta']:+.3f})"
        ),
        "largest_decrease": (
            "None"
            if largest_down is None
            else f"{largest_down['source']} -> {largest_down['target']} ({largest_down['delta']:+.3f})"
        ),
        "stable_edges": stable_count,
        "highlighted_path_edges": len(path["edges"]) if mode == MODE_DOMINANT else 0,
    }
    return statements, stats


def _edge_hover(record: dict[str, Any]) -> str:
    percentage = record["percentage_delta"]
    percentage_text = "Unavailable" if percentage is None else f"{float(percentage):+.1f}%"
    return (
        f"Source: {record['source']}<br>"
        f"Target: {record['target']}<br>"
        f"Previous alpha: {record['previous_alpha']:.4f}<br>"
        f"Current alpha: {record['current_alpha']:.4f}<br>"
        f"Delta: {record['delta']:+.4f}<br>"
        f"Absolute delta: {record['absolute_delta']:.4f}<br>"
        f"Percentage delta: {percentage_text}<br>"
        f"Selected attention head: {record['attention_head']}<br>"
        f"Previous replay date: {record['previous_replay_date'] or 'No previous eligible replay date'}<br>"
        f"Current replay date: {record['replay_date']}<br>"
        f"Current attention rank: {record['current_rank']}<br>"
        f"Absolute delta rank: {record['absolute_delta_rank']}<extra></extra>"
    )


def _legend_text(mode: str) -> str:
    if mode == MODE_DELTA:
        return (
            "Green edge = attention increased<br>"
            "Red edge = attention decreased<br>"
            "Grey edge = stable attention<br>"
            "Edge thickness = absolute attention change<br>"
            "Loop = self-attention"
        )
    if mode == MODE_DOMINANT:
        return (
            "Solid highlighted edges = dominant attention path<br>"
            "Faded edges = other attention relationships<br>"
            "Edge color = attention shift<br>"
            "Loop = self-attention"
        )
    return (
        "Green edge = attention increased<br>"
        "Red edge = attention decreased<br>"
        "Grey edge = stable attention<br>"
        "Edge thickness = current attention strength<br>"
        "Dashed or faded edge = non-highlighted edge<br>"
        "Loop = self-attention"
    )


def make_network_figure(
    current: np.ndarray,
    previous: np.ndarray,
    symbols: Iterable[str],
    *,
    text_probabilities: dict[str, float],
    graph_probabilities: dict[str, float],
    fusion_probabilities: dict[str, float],
    fusion_deltas: dict[str, float],
    actions: dict[str, str],
    selected_symbols: set[str] | None,
    head_label: str,
    replay_date: str,
    previous_date: str | None,
    mode: str = MODE_CURRENT,
    show_edge_values: bool = False,
) -> go.Figure:
    names = tuple(symbols)
    records = edge_records(
        current,
        previous,
        names,
        head_label=head_label,
        replay_date=replay_date,
        previous_date=previous_date,
    )
    path = dominant_attention_path(current, names)
    path_edges = set(path["edges"]) if mode == MODE_DOMINANT else set()
    selected = set(selected_symbols or ())
    figure = go.Figure()
    for record in records:
        source = str(record["source"])
        target = str(record["target"])
        visual = edge_visual_attributes(
            record,
            mode=mode,
            path_edges=path_edges,
            selected_symbols=selected,
        )
        x, y = (
            _self_edge_points(source)
            if source == target
            else _cross_edge_points(source, target)
        )
        rgba = _hex_to_rgba(str(visual["color"]), float(visual["opacity"]))
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={
                    "color": rgba,
                    "width": visual["width"],
                    "dash": visual["dash"],
                },
                hovertemplate=_edge_hover(record),
                showlegend=False,
                meta={
                    "kind": "edge",
                    "source": source,
                    "target": target,
                    "is_self_loop": bool(record["is_self_loop"]),
                    "width": visual["width"],
                    "color": visual["color"],
                    "highlighted": visual["highlighted"],
                    "on_path": visual["on_path"],
                },
            )
        )
        arrow_index = -3
        tail_index = -9
        figure.add_annotation(
            x=float(x[arrow_index]),
            y=float(y[arrow_index]),
            ax=float(x[tail_index]),
            ay=float(y[tail_index]),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=0.85,
            arrowwidth=max(0.8, min(3.2, float(visual["width"]) * 0.48)),
            arrowcolor=rgba,
            opacity=float(visual["opacity"]),
        )
        if show_edge_values:
            label_index = len(x) // 2
            if source == target:
                text_position = "top center" if source != "TSLA" else "bottom center"
            elif source < target:
                text_position = "top center"
            else:
                text_position = "bottom center"
            figure.add_trace(
                go.Scatter(
                    x=[float(x[label_index])],
                    y=[float(y[label_index])],
                    mode="markers+text",
                    marker={"size": 22, "color": "rgba(255,255,255,0.01)"},
                    text=[f"{record['current_alpha']:.3f}<br>{record['delta']:+.3f}"],
                    textposition=text_position,
                    textfont={"size": 10, "color": "#172126"},
                    hovertemplate=_edge_hover(record),
                    showlegend=False,
                    meta={"kind": "edge_label", "source": source, "target": target},
                )
            )

    for symbol in names:
        x, y = NODE_POSITIONS[symbol]
        action = str(actions[symbol]).upper()
        selected_node = symbol in selected
        dimmed = bool(selected) and symbol not in selected
        width = 0.56 if not selected_node else 0.62
        height = 0.39 if not selected_node else 0.43
        border_width = 3 if selected_node else 1.5
        fill = "#FFFFFF" if not dimmed else "#F3F6F5"
        accent = ACTION_COLORS.get(action, ACTION_COLORS["HOLD"])
        border = accent if not selected_node else "#172126"
        figure.add_shape(
            type="path",
            path=_rounded_rect_path(x, y, width, height, 0.055),
            fillcolor=fill,
            line={"color": border, "width": border_width},
            layer="above",
        )
        figure.add_shape(
            type="rect",
            x0=x - width / 2,
            x1=x - width / 2 + 0.045,
            y0=y - height / 2 + 0.02,
            y1=y + height / 2 - 0.02,
            fillcolor=accent,
            line={"width": 0},
            layer="above",
        )
        opacity = 0.45 if dimmed else 1.0
        figure.add_annotation(
            x=x,
            y=y,
            text=(
                f"<b>{symbol}</b><br>"
                f"<span style='color:{accent}'><b>{action}</b></span><br>"
                f"Text&nbsp;&nbsp;{text_probabilities[symbol]:.3f}<br>"
                f"Graph&nbsp;{graph_probabilities[symbol]:.3f}<br>"
                f"Fusion {fusion_probabilities[symbol]:.3f}<br>"
                f"Delta&nbsp;{fusion_deltas[symbol]:+.3f}"
            ),
            showarrow=False,
            align="left",
            font={"size": 11, "color": "#172126"},
            opacity=opacity,
            bgcolor="rgba(255,255,255,0)",
        )
        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker={"size": 120, "color": "rgba(255,255,255,0.01)"},
                hovertemplate=(
                    f"{symbol}<br>Agent action: {action}<br>"
                    f"Text probability: {text_probabilities[symbol]:.4f}<br>"
                    f"Graph probability: {graph_probabilities[symbol]:.4f}<br>"
                    f"Fusion probability: {fusion_probabilities[symbol]:.4f}<br>"
                    f"Fusion delta: {fusion_deltas[symbol]:+.4f}<extra></extra>"
                ),
                showlegend=False,
                meta={"kind": "node", "symbol": symbol, "selected": selected_node},
            )
        )

    path_label = " -> ".join(path["symbols"])
    path_terms = " x ".join(
        f"alpha({source} -> {target})" for source, target in path["edges"]
    )
    figure.add_annotation(
        x=0.0,
        y=-1.05,
        text=(
            f"<b>Dominant attention path:</b> {path_label}<br>"
            f"<b>Path score:</b> {path_terms} = {path['score']:.4f}"
        ),
        showarrow=False,
        font={"size": 12, "color": "#172126"},
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="#D9E0DF",
        borderwidth=1,
        align="center",
    )
    figure.add_annotation(
        x=-1.28,
        y=1.08,
        xanchor="left",
        yanchor="top",
        text=_legend_text(mode),
        showarrow=False,
        align="left",
        font={"size": 10, "color": "#39464D"},
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#D9E0DF",
        borderwidth=1,
        borderpad=6,
    )
    previous_label = previous_date or "No previous eligible replay date"
    figure.update_layout(
        height=560,
        margin={"l": 8, "r": 8, "t": 66, "b": 10},
        title={
            "text": (
                "Decision Influence Map - Frozen E4 GAT"
                f"<br><sup>Previous: {previous_label} | Current: {replay_date} | {head_label}</sup>"
            ),
            "x": 0.02,
            "font": {"size": 18, "color": "#172126"},
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F5F7F6",
        xaxis={"visible": False, "range": [-1.42, 1.42], "fixedrange": True},
        yaxis={"visible": False, "range": [-1.16, 1.18], "fixedrange": True},
        hoverlabel={"bgcolor": "#172126", "font": {"color": "#FFFFFF"}},
    )
    return figure


def make_attention_heatmaps(
    previous: np.ndarray,
    current: np.ndarray,
    symbols: Iterable[str],
    *,
    head_label: str,
) -> go.Figure:
    names = tuple(symbols)
    before, now, delta = attention_views(previous, current)
    delta_bound = max(float(np.max(np.abs(delta))), 0.05)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Previous date", "Current date", "Delta"),
        horizontal_spacing=0.035,
    )
    for column, (matrix, colorscale, zmin, zmax, show_scale) in enumerate(
        (
            (before, "YlGnBu", 0.0, 1.0, False),
            (now, "YlGnBu", 0.0, 1.0, False),
            (delta, "RdBu", -delta_bound, delta_bound, True),
        ),
        start=1,
    ):
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=names,
                y=names,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                reversescale=column == 3,
                text=np.vectorize(
                    lambda value: f"{value:+.3f}" if column == 3 else f"{value:.3f}"
                )(matrix),
                texttemplate="%{text}",
                textfont={"size": 10, "color": "#172126"},
                hovertemplate=(
                    "Source: %{x}<br>Target: %{y}<br>Value: %{z:.4f}<extra></extra>"
                ),
                showscale=show_scale,
                colorbar={"title": "Delta", "len": 0.82} if show_scale else None,
            ),
            row=1,
            col=column,
        )
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(tickfont={"color": "#627078", "size": 10})
    figure.update_yaxes(tickfont={"color": "#627078", "size": 10})
    figure.update_annotations(font={"color": "#172126", "size": 12})
    figure.update_layout(
        height=276,
        margin={"l": 18, "r": 20, "t": 52, "b": 18},
        title={
            "text": f"Attention heatmaps | {head_label}",
            "x": 0.01,
            "font": {"color": "#172126", "size": 14},
        },
        font={"color": "#172126"},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hoverlabel={"bgcolor": "#172126", "font": {"color": "#FFFFFF"}},
    )
    return figure
