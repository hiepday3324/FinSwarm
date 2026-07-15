from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


NODE_POSITIONS = {
    "AAPL": (-0.92, 0.72),
    "NVDA": (0.92, 0.72),
    "TSLA": (0.0, -0.82),
}
NODE_COLORS = {"AAPL": "#0F766E", "NVDA": "#D95D39", "TSLA": "#C08B2D"}


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
    current_values = np.asarray(current, dtype=np.float64)
    previous_values = np.asarray(previous, dtype=np.float64)
    expected = (len(names), len(names))
    if current_values.shape != expected or previous_values.shape != expected:
        raise ValueError(f"Attention matrices must have shape {expected}")
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
                    "previous_alpha": before,
                    "current_alpha": now,
                    "delta": delta,
                    "absolute_delta": abs(delta),
                    "percentage_delta": percentage,
                    "attention_head": head_label,
                    "replay_date": replay_date,
                    "previous_replay_date": previous_date,
                }
            )
    return records


def largest_attention_changes(records: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
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


def _quadratic_points(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    count: int = 28,
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
    radius = 0.17
    start = (sx + ux * radius, sy + uy * radius)
    end = (tx - ux * radius, ty - uy * radius)
    direction_sign = 1.0 if source < target else -1.0
    bend = 0.14 * direction_sign
    control = ((sx + tx) / 2 + px * bend, (sy + ty) / 2 + py * bend)
    return _quadratic_points(start, control, end)


def _self_edge_points(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = NODE_POSITIONS[symbol]
    angle = np.linspace(math.pi * 0.12, math.pi * 1.88, 32)
    center_y = y + (0.23 if symbol != "TSLA" else -0.23)
    return x + 0.22 * np.cos(angle), center_y + 0.16 * np.sin(angle)


def make_network_figure(
    current: np.ndarray,
    previous: np.ndarray,
    symbols: Iterable[str],
    *,
    probabilities: dict[str, float],
    actions: dict[str, str],
    selected_symbols: set[str] | None,
    head_label: str,
    replay_date: str,
    previous_date: str | None,
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
    selected = set(selected_symbols or ())
    figure = go.Figure()
    for record in records:
        source = str(record["source"])
        target = str(record["target"])
        coefficient = float(record["current_alpha"])
        emphasized = not selected or source in selected or target in selected
        color = "#0F766E" if emphasized else "#B8C1C5"
        opacity = (0.42 + 0.55 * coefficient) if emphasized else 0.20
        rgba = (
            f"rgba(15,118,110,{opacity:.3f})"
            if emphasized
            else f"rgba(184,193,197,{opacity:.3f})"
        )
        x, y = (
            _self_edge_points(source)
            if source == target
            else _cross_edge_points(source, target)
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": rgba, "width": 1.2 + 7.0 * coefficient},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        arrow_index = -2
        tail_index = -7
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
            arrowsize=1.05,
            arrowwidth=max(1.0, 1.0 + 3.0 * coefficient),
            arrowcolor=color,
            opacity=opacity,
        )
        if source == target:
            label_index = len(x) // 2
            text_position = "top center"
        elif source < target:
            label_index = int(round((len(x) - 1) * 0.36))
            text_position = "top center"
        else:
            label_index = int(round((len(x) - 1) * 0.64))
            text_position = "bottom center"
        percentage = record["percentage_delta"]
        percentage_text = (
            "Unavailable"
            if percentage is None
            else f"{float(percentage):+.1f}%"
        )
        hover = (
            f"Source: {source}<br>Target: {target}<br>"
            f"Previous alpha: {record['previous_alpha']:.4f}<br>"
            f"Current alpha: {record['current_alpha']:.4f}<br>"
            f"Delta: {record['delta']:+.4f}<br>"
            f"Percentage delta: {percentage_text}<br>"
            f"Attention head: {head_label}<br>Replay date: {replay_date}<br>"
            f"Previous replay date: {previous_date or 'Unavailable'}<extra></extra>"
        )
        figure.add_trace(
            go.Scatter(
                x=[float(x[label_index])],
                y=[float(y[label_index])],
                mode="markers+text",
                marker={"size": 22, "color": "rgba(255,255,255,0.01)"},
                text=[f"alpha {coefficient:.3f}"],
                textposition=text_position,
                textfont={"size": 10, "color": "#39464D"},
                hovertemplate=hover,
                showlegend=False,
            )
        )

    node_x = [NODE_POSITIONS[symbol][0] for symbol in names]
    node_y = [NODE_POSITIONS[symbol][1] for symbol in names]
    node_sizes = [64 if symbol in selected else 54 for symbol in names]
    line_widths = [5 if symbol in selected else 2 for symbol in names]
    text = [
        f"<b>{symbol}</b><br>{actions[symbol].upper()}<br>p={probabilities[symbol]:.3f}"
        for symbol in names
    ]
    hover = [
        f"{symbol}<br>Agent action: {actions[symbol].upper()}<br>"
        f"Fusion probability: {probabilities[symbol]:.4f}<extra></extra>"
        for symbol in names
    ]
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker={
                "size": node_sizes,
                "color": [NODE_COLORS[symbol] for symbol in names],
                "line": {"color": "#FFFFFF", "width": line_widths},
            },
            text=text,
            textposition="middle center",
            textfont={"size": 12, "color": "#FFFFFF"},
            hovertemplate=hover,
            showlegend=False,
        )
    )
    figure.update_layout(
        height=515,
        margin={"l": 10, "r": 10, "t": 38, "b": 10},
        title={
            "text": f"Sample-specific attention coefficients | {head_label}",
            "x": 0.02,
            "font": {"size": 17, "color": "#172126"},
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F5F7F6",
        xaxis={"visible": False, "range": [-1.35, 1.35], "fixedrange": True},
        yaxis={"visible": False, "range": [-1.20, 1.20], "fixedrange": True},
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
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Previous date", "Current date", "Attention delta"),
        horizontal_spacing=0.08,
    )
    for column, (matrix, colorscale, zmin, zmax) in enumerate(
        (
            (before, "Viridis", 0.0, 1.0),
            (now, "Viridis", 0.0, 1.0),
            (delta, "RdBu", -max(float(np.max(np.abs(delta))), 1e-4), max(float(np.max(np.abs(delta))), 1e-4)),
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
                text=np.vectorize(lambda value: f"{value:+.3f}" if column == 3 else f"{value:.3f}")(matrix),
                texttemplate="%{text}",
                hovertemplate=(
                    "Source: %{x}<br>Target: %{y}<br>Value: %{z:.4f}<extra></extra>"
                ),
                showscale=column == 3,
            ),
            row=1,
            col=column,
        )
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(tickfont={"color": "#627078"})
    figure.update_yaxes(tickfont={"color": "#627078"})
    figure.update_annotations(font={"color": "#172126", "size": 13})
    figure.update_layout(
        height=350,
        margin={"l": 28, "r": 22, "t": 72, "b": 35},
        title={
            "text": f"Attention heatmaps | {head_label}",
            "x": 0.01,
            "font": {"color": "#172126", "size": 16},
        },
        font={"color": "#172126"},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hoverlabel={"bgcolor": "#172126", "font": {"color": "#FFFFFF"}},
    )
    return figure
