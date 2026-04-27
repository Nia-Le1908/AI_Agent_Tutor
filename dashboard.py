"""
Dashboard module for AI Tutor V5.1 (Phase 4).

This file is intentionally independent from app.py so charts can be reused in
other entry points (for example a dedicated analytics page later).

Charts:
1. Session summary metrics (total, accuracy, streak).
2. Pie chart: correct/incorrect ratio by subject.
3. Radar chart: weak-topic accuracy by subject.
4. Line chart: progress over time.
5. Bar chart: average score by difficulty.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH
from sqlite_manager import get_user_stats, get_weak_topics


def _get_connection() -> sqlite3.Connection:
    """Create a SQLite connection with dict-like row access enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _fetch_subject_stats(uid: int) -> List[sqlite3.Row]:
    """Aggregate correct/incorrect counts by subject for one user."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                q.subject AS subject,
                SUM(CASE WHEN h.is_correct = 1 THEN 1 ELSE 0 END) AS correct_count,
                SUM(CASE WHEN h.is_correct = 0 THEN 1 ELSE 0 END) AS incorrect_count,
                COUNT(*) AS total_count
            FROM history h
            INNER JOIN questions q ON q.id = h.qid
            WHERE h.uid = ?
            GROUP BY q.subject
            ORDER BY q.subject ASC
            """,
            (uid,),
        ).fetchall()
        return rows
    finally:
        conn.close()


def _fetch_timeline(uid: int) -> List[sqlite3.Row]:
    """Fetch attempts ordered by timestamp for progress timeline chart."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                h.timestamp AS ts,
                h.is_correct AS is_correct
            FROM history h
            WHERE h.uid = ?
            ORDER BY h.timestamp ASC, h.rowid ASC
            """,
            (uid,),
        ).fetchall()
        return rows
    finally:
        conn.close()


def _fetch_difficulty_scores(uid: int) -> List[sqlite3.Row]:
    """Fetch correctness grouped by difficulty for average-score bar chart."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                q.difficulty AS difficulty,
                AVG(CASE WHEN h.is_correct = 1 THEN 1.0 ELSE 0.0 END) AS avg_score,
                COUNT(*) AS total_attempts
            FROM history h
            INNER JOIN questions q ON q.id = h.qid
            WHERE h.uid = ?
            GROUP BY q.difficulty
            ORDER BY q.difficulty ASC
            """,
            (uid,),
        ).fetchall()
        return rows
    finally:
        conn.close()


def _build_subject_pie(subject_rows: List[sqlite3.Row]) -> go.Figure:
    """Create a pie chart representing correct/incorrect ratio per subject."""
    labels: List[str] = []
    values: List[int] = []

    for row in subject_rows:
        subject = str(row["subject"])
        correct = int(row["correct_count"] or 0)
        incorrect = int(row["incorrect_count"] or 0)

        if correct > 0:
            labels.append(f"{subject} - Đúng")
            values.append(correct)
        if incorrect > 0:
            labels.append(f"{subject} - Sai")
            values.append(incorrect)

    if not values:
        labels = ["Chưa có dữ liệu"]
        values = [1]

    fig = go.Figure(
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
    fig.update_layout(title="Tỷ lệ Đúng/Sai theo môn")
    return fig


def _build_weak_topic_radar(uid: int) -> go.Figure:
    """Create a radar chart showing accuracy per subject for weak-topic visualization."""
    weak_data = get_weak_topics(uid)

    if not weak_data:
        fig = go.Figure()
        fig.update_layout(title="Điểm mạnh / Điểm yếu theo môn", annotations=[
            {"text": "Chưa có dữ liệu", "xref": "paper", "yref": "paper",
             "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 16}}
        ])
        return fig

    subjects = list(weak_data.keys())
    accuracies = [weak_data[s]["accuracy"] * 100 for s in subjects]

    # Close the radar polygon
    subjects_closed = subjects + [subjects[0]]
    accuracies_closed = accuracies + [accuracies[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=accuracies_closed,
                theta=subjects_closed,
                fill="toself",
                name="Accuracy %",
                line={"color": "#636EFA", "width": 2},
                fillcolor="rgba(99, 110, 250, 0.25)",
            )
        ]
    )
    fig.update_layout(
        title="Điểm mạnh / Điểm yếu theo môn",
        polar={"radialaxis": {"visible": True, "range": [0, 100], "suffix": "%"}},
        showlegend=False,
    )
    return fig


def _build_progress_line(timeline_rows: List[sqlite3.Row]) -> go.Figure:
    """
    Create progress line chart from cumulative accuracy over attempts.

    Why cumulative accuracy:
    - Smooths noisy single-attempt outcomes.
    - Better reflects longitudinal progress.
    """
    x_vals: List[str] = []
    y_vals: List[float] = []

    total = 0
    correct = 0

    for row in timeline_rows:
        total += 1
        if int(row["is_correct"]) == 1:
            correct += 1

        ts_raw = str(row["ts"])
        try:
            ts_display = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            ts_display = ts_raw

        x_vals.append(ts_display)
        y_vals.append((correct / total) * 100.0)

    if not x_vals:
        x_vals = ["Chưa có dữ liệu"]
        y_vals = [0.0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name="Cumulative Accuracy",
                line={"width": 3, "color": "#00CC96"},
            )
        ]
    )
    fig.update_layout(
        title="Tiến trình học tập",
        xaxis_title="Thời gian",
        yaxis_title="Độ chính xác (%)",
        yaxis={"range": [0, 100]},
    )
    return fig


def _build_difficulty_bar(diff_rows: List[sqlite3.Row]) -> go.Figure:
    """Create bar chart of average correctness by difficulty level."""
    x_vals: List[str] = []
    y_vals: List[float] = []
    text_vals: List[str] = []

    for row in diff_rows:
        difficulty = int(row["difficulty"])
        avg_score = float(row["avg_score"] or 0.0) * 100.0
        total_attempts = int(row["total_attempts"] or 0)

        x_vals.append(f"Level {difficulty}")
        y_vals.append(avg_score)
        text_vals.append(f"{avg_score:.1f}% ({total_attempts} câu)")

    if not x_vals:
        x_vals = ["Chưa có dữ liệu"]
        y_vals = [0.0]
        text_vals = ["0.0%"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_vals,
                y=y_vals,
                text=text_vals,
                textposition="outside",
                marker_color="#EF553B",
            )
        ]
    )
    fig.update_layout(
        title="Điểm trung bình theo độ khó",
        xaxis_title="Độ khó",
        yaxis_title="Điểm trung bình (%)",
        yaxis={"range": [0, 100]},
    )
    return fig


def render_dashboard(uid: int) -> None:
    """
    Render the full analytics dashboard for a user.

    Args:
        uid: user id in the users table.
    """
    if not isinstance(uid, int) or uid <= 0:
        st.warning("Please select a valid user before opening the dashboard.")
        return

    st.subheader("📊 Phân tích học tập")

    # --- Session summary metrics ---
    stats = get_user_stats(uid)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📝 Tổng câu đã làm", stats["total_attempted"])
    m2.metric("✅ Số câu đúng", stats["total_correct"])
    m3.metric("🎯 Độ chính xác", f"{stats['accuracy']:.1f}%")
    m4.metric("🔥 Streak hiện tại", f"{stats['current_streak']} câu")

    st.markdown("---")

    # --- Charts ---
    subject_rows = _fetch_subject_stats(uid)
    timeline_rows = _fetch_timeline(uid)
    diff_rows = _fetch_difficulty_scores(uid)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(_build_subject_pie(subject_rows), width="stretch")

    with col2:
        st.plotly_chart(_build_weak_topic_radar(uid), width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(_build_progress_line(timeline_rows), width="stretch")

    with col4:
        st.plotly_chart(_build_difficulty_bar(diff_rows), width="stretch")
