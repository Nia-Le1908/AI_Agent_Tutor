"""
Learning analytics dashboard for AI Tutor.

Charts:
1. Session summary metrics (total, correct, accuracy, streak).
2. Pie chart: correct/incorrect ratio by subject.
3. Radar chart: accuracy by subject (strengths / weaknesses).
4. Line chart: cumulative accuracy over time.
5. Bar chart: average score by difficulty.

This module renders; it owns no SQL and no DB connections. Every number arrives
from :mod:`sqlite_manager`, and each chart builder is a pure function over plain
dicts/lists, which keeps them testable without a database or a Streamlit runtime.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from sqlite_manager import (
    get_difficulty_scores,
    get_progress_timeline,
    get_user_stats,
    get_weak_topics,
)

NO_DATA_LABEL = "Chưa có dữ liệu"

# Chart palette (Plotly defaults, named so the same colours are reused).
COLOR_ACCENT = "#636EFA"
COLOR_POSITIVE = "#00CC96"
COLOR_NEGATIVE = "#EF553B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_timestamp(raw: Any) -> str:
    """Render a stored ISO timestamp for an axis label, tolerating odd values."""
    text = str(raw)
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text


def _empty_figure(title: str, message: str = NO_DATA_LABEL) -> go.Figure:
    """Placeholder figure used whenever a chart has no data to show."""
    figure = go.Figure()
    figure.update_layout(
        title=title,
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )
    return figure


# ---------------------------------------------------------------------------
# Chart builders (pure)
# ---------------------------------------------------------------------------
def build_subject_pie(subject_stats: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Donut chart of correct vs incorrect attempts per subject.

    Args:
        subject_stats: mapping of subject -> {"correct": float, "incorrect": float}.
    """
    labels: List[str] = []
    values: List[float] = []

    for subject, stats in subject_stats.items():
        correct = int(stats.get("correct") or 0)
        incorrect = int(stats.get("incorrect") or 0)
        if correct > 0:
            labels.append(f"{subject} - Đúng")
            values.append(correct)
        if incorrect > 0:
            labels.append(f"{subject} - Sai")
            values.append(incorrect)

    if not values:
        labels, values = [NO_DATA_LABEL], [1]

    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.35,
                sort=False,
                textinfo="label+percent",
            )
        ]
    )
    figure.update_layout(title="Tỷ lệ Đúng/Sai theo môn")
    return figure


def build_weak_topic_radar(subject_stats: Dict[str, Dict[str, float]]) -> go.Figure:
    """Radar chart of accuracy per subject; the polygon is closed on itself."""
    subjects = list(subject_stats)
    if not subjects:
        return _empty_figure("Điểm mạnh / Điểm yếu theo môn")

    accuracies = [float(subject_stats[s].get("accuracy", 0.0)) * 100 for s in subjects]

    figure = go.Figure(
        data=[
            go.Scatterpolar(
                r=accuracies + [accuracies[0]],
                theta=subjects + [subjects[0]],
                fill="toself",
                name="Độ chính xác (%)",
                line={"color": COLOR_ACCENT, "width": 2},
                fillcolor="rgba(99, 110, 250, 0.25)",
            )
        ]
    )
    figure.update_layout(
        title="Điểm mạnh / Điểm yếu theo môn",
        polar={"radialaxis": {"visible": True, "range": [0, 100], "ticksuffix": "%"}},
        showlegend=False,
    )
    return figure


def build_progress_line(timeline: List[Dict[str, Any]]) -> go.Figure:
    """
    Cumulative-accuracy line chart over attempts.

    Cumulative rather than per-attempt accuracy: a single answer is a noisy signal,
    while the running curve shows the longitudinal trend the student cares about.
    """
    if not timeline:
        return _empty_figure("Tiến trình học tập")

    x_vals: List[str] = []
    y_vals: List[float] = []
    total = 0
    correct = 0

    for row in timeline:
        total += 1
        correct += 1 if int(row.get("is_correct") or 0) == 1 else 0
        x_vals.append(format_timestamp(row.get("ts")))
        y_vals.append((correct / total) * 100.0)

    figure = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name="Độ chính xác tích lũy",
                line={"width": 3, "color": COLOR_POSITIVE},
            )
        ]
    )
    figure.update_layout(
        title="Tiến trình học tập",
        xaxis_title="Thời gian",
        yaxis_title="Độ chính xác (%)",
        yaxis={"range": [0, 100]},
    )
    return figure


def build_difficulty_bar(difficulty_rows: List[Dict[str, Any]]) -> go.Figure:
    """Bar chart of average score per difficulty level, with attempt counts."""
    if not difficulty_rows:
        return _empty_figure("Điểm trung bình theo độ khó")

    x_vals: List[str] = []
    y_vals: List[float] = []
    text_vals: List[str] = []

    for row in difficulty_rows:
        average = float(row.get("avg_score") or 0.0) * 100.0
        attempts = int(row.get("total_attempts") or 0)
        x_vals.append(f"Độ khó {int(row['difficulty'])}")
        y_vals.append(average)
        text_vals.append(f"{average:.1f}% ({attempts} câu)")

    figure = go.Figure(
        data=[
            go.Bar(
                x=x_vals,
                y=y_vals,
                text=text_vals,
                textposition="outside",
                marker_color=COLOR_NEGATIVE,
            )
        ]
    )
    figure.update_layout(
        title="Điểm trung bình theo độ khó",
        xaxis_title="Độ khó",
        yaxis_title="Điểm trung bình (%)",
        yaxis={"range": [0, 100]},
    )
    return figure


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_summary_metrics(stats: Dict[str, Any]) -> None:
    """Render the four headline numbers above the charts."""
    columns = st.columns(4)
    columns[0].metric("📝 Tổng câu đã làm", stats.get("total_attempted", 0))
    columns[1].metric("✅ Số câu đúng", stats.get("total_correct", 0))
    columns[2].metric("🎯 Độ chính xác", f"{stats.get('accuracy', 0.0):.1f}%")
    columns[3].metric("🔥 Chuỗi đúng hiện tại", f"{stats.get('current_streak', 0)} câu")


def render_dashboard(uid: Optional[int]) -> None:
    """
    Render the full analytics dashboard for one user.

    Args:
        uid: user id in the users table. Invalid ids show a hint instead of raising,
            because this runs inside a Streamlit rerun where an exception would kill
            the whole page.
    """
    if not isinstance(uid, int) or uid <= 0:
        st.warning("Vui lòng chọn người dùng hợp lệ trước khi mở thống kê.")
        return

    st.subheader("📊 Phân tích học tập")

    render_summary_metrics(get_user_stats(uid))
    st.markdown("---")

    # One query feeds both the pie and the radar chart.
    subject_stats = get_weak_topics(uid)
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(build_subject_pie(subject_stats), width="stretch")
    with col_right:
        st.plotly_chart(build_weak_topic_radar(subject_stats), width="stretch")

    col_bottom_left, col_bottom_right = st.columns(2)
    with col_bottom_left:
        st.plotly_chart(build_progress_line(get_progress_timeline(uid)), width="stretch")
    with col_bottom_right:
        st.plotly_chart(build_difficulty_bar(get_difficulty_scores(uid)), width="stretch")
