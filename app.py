"""
Main Streamlit app for AI Tutor V5.1 (Phase 4).

Key requirement handled in this file:
- Streamlit reruns on every interaction, so we persist critical app context in
  st.session_state to avoid resets.

Session keys used:
- user_id: current logged-in user in DB
- user_name: display name
- current_level: adaptive learning level [1..5]
- chat_history: list of chat messages
- current_question: currently displayed exercise dict
- selected_answer: currently selected option label (A/B/C/D)
- last_feedback: result of latest submission
"""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any, Dict, List
from generator import generate, generate_batch
import streamlit as st

from adaptive_logic import get_next_difficulty
from config import DB_PATH
from controller import chat
from dashboard import render_dashboard
from init_db import initialize_database
from sqlite_manager import get_all_subjects, get_question_by_diff, get_questions_filtered, save_history


def _get_connection() -> sqlite3.Connection:
    """Create a DB connection with safe defaults."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn



def _safe_init_db() -> None:
    """Initialize DB if not ready yet; show a clear user-facing status."""
    try:
        initialize_database()
    except Exception as exc:
        st.error(f"Failed to initialize database: {exc}")
        st.stop()


def _ensure_session_state() -> None:
    """Create all required session keys once to survive Streamlit reruns."""
    defaults = {
        "user_id": None,
        "user_name": "",
        "current_level": 1,
        "chat_history": [],
        "current_question": None,
        "selected_answer": "",
        "last_feedback": "",
        "subject_filter": "Tất cả",
        "skip_answered": True,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_or_create_user(name: str) -> int:
    """
    Resolve user id by exact name, or create new user with default level.

    This keeps onboarding friction low for demo/testing usage.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT id, level FROM users WHERE name = ? ORDER BY id ASC LIMIT 1",
            (name.strip(),),
        ).fetchone()
        if row is not None:
            st.session_state["current_level"] = int(row["level"])
            return int(row["id"])

        cursor = conn.execute(
            "INSERT INTO users (name, level) VALUES (?, ?)",
            (name.strip(), 1),
        )
        conn.commit()
        st.session_state["current_level"] = 1
        return int(cursor.lastrowid)
    finally:
        conn.close()


def _load_random_question_for_current_level() -> None:
    """
    Load one question from DB based on current adaptive level,
    with optional subject filter and answered-question exclusion.
    """
    level = int(st.session_state["current_level"])
    subject = st.session_state.get("subject_filter", "Tất cả")
    subject_param = None if subject == "Tất cả" else subject
    exclude_uid = None
    if st.session_state.get("skip_answered", False) and st.session_state["user_id"]:
        exclude_uid = int(st.session_state["user_id"])

    questions = get_questions_filtered(
        level=level, subject=subject_param, exclude_uid=exclude_uid
    )

    if not questions:
        st.session_state["current_question"] = None
        subject_info = f" cho môn '{subject}'" if subject_param else ""
        st.session_state["last_feedback"] = (
            f"Không tìm thấy câu hỏi ở độ khó {level}{subject_info}. "
            "Hãy thểm câu hỏi vào Database hoặc đổi bộ lọc."
        )
        return

    st.session_state["current_question"] = random.choice(questions)
    st.session_state["selected_answer"] = ""
    st.session_state["last_feedback"] = ""


def _parse_options(options_json: str) -> Dict[str, str]:
    """
    Parse options JSON string from DB into {A, B, C, D} mapping.

    Falls back to empty dict if malformed so the UI can fail gracefully.
    """
    try:
        parsed = json.loads(options_json)
    except Exception:
        return {}

    if isinstance(parsed, dict):
        return {
            "A": str(parsed.get("A", "")).strip(),
            "B": str(parsed.get("B", "")).strip(),
            "C": str(parsed.get("C", "")).strip(),
            "D": str(parsed.get("D", "")).strip(),
        }

    if isinstance(parsed, list) and len(parsed) == 4:
        return {
            "A": str(parsed[0]).strip(),
            "B": str(parsed[1]).strip(),
            "C": str(parsed[2]).strip(),
            "D": str(parsed[3]).strip(),
        }

    return {}


def _render_sidebar() -> None:
    """Render user setup controls and keep selected user in session state."""
    st.sidebar.title("AI Tutor V5.1")
    st.sidebar.caption("Personalized learning assistant")

    name_input = st.sidebar.text_input(
        "Student name",
        value=st.session_state["user_name"],
        max_chars=100,
    )

    if st.sidebar.button("Start / Load User", width="stretch"):
        if not name_input.strip():
            st.sidebar.warning("Please enter a valid student name.")
        else:
            uid = _get_or_create_user(name_input)
            st.session_state["user_id"] = uid
            st.session_state["user_name"] = name_input.strip()
            st.sidebar.success(f"Active user ID: {uid}")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Current level: {st.session_state['current_level']}")
    st.sidebar.write(f"User ID: {st.session_state['user_id']}")


def _render_chat_tab() -> None:
    """Render chatbot interface with persistent chat history."""
    st.subheader("Chat Tutor")

    if not st.session_state["user_id"]:
        st.info("Create or load a user from the sidebar first.")
        return

    # Replay chat history on every rerun.
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask your tutor anything...")

    if user_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = chat(user_prompt)
                except Exception as exc:
                    answer = f"Error while generating answer: {exc}"
                st.markdown(answer)

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})


def _render_exercise_tab() -> None:
    """Render adaptive exercise UI with subject filter and smart question loading."""
    st.subheader("📝 Luyện tập")

    if not st.session_state["user_id"]:
        st.info("Hãy tạo hoặc chọn user ở sidebar trước.")
        return

    # --- Filter controls ---
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])

    with col_filter1:
        subjects = get_all_subjects()
        subject_options = ["Tất cả"] + subjects
        selected_subject = st.selectbox(
            "📚 Môn học:",
            options=subject_options,
            index=subject_options.index(st.session_state.get("subject_filter", "Tất cả"))
            if st.session_state.get("subject_filter", "Tất cả") in subject_options
            else 0,
            key="subject_select",
        )
        st.session_state["subject_filter"] = selected_subject

    with col_filter2:
        st.metric("🎯 Độ khó hiện tại", st.session_state["current_level"])

    with col_filter3:
        skip = st.checkbox("❌ Bỏ câu đã làm đúng", value=st.session_state.get("skip_answered", True), key="skip_chk")
        st.session_state["skip_answered"] = skip

    if st.button("🔄 Tải câu hỏi mới", width="stretch"):
        _load_random_question_for_current_level()

    question = st.session_state["current_question"]

    if not question:
        if st.session_state["last_feedback"]:
            st.warning(st.session_state["last_feedback"])
        else:
            st.info("Click 'Load Question' to start practicing.")
        return

    st.markdown(f"**Question #{question['id']}**")
    st.write(question["content"])

    options_map = _parse_options(question.get("options", ""))
    if not options_map or not all(options_map.values()):
        st.error("This question has invalid options format in database.")
        return

    option_labels = [f"{key}. {value}" for key, value in options_map.items()]

    selected = st.radio(
        "Choose your answer:",
        options=option_labels,
        index=None,
        key="exercise_option_radio",
    )

    if st.button("Submit Answer", type="primary", width="stretch"):
        if not selected:
            st.warning("Please select an answer before submitting.")
            return

        chosen_letter = selected.split(".", 1)[0].strip().upper()
        correct_letter = str(question["answer"]).strip().upper()
        is_correct = chosen_letter == correct_letter

        try:
            save_history(
                uid=int(st.session_state["user_id"]),
                qid=int(question["id"]),
                is_correct=is_correct,
            )

            # Update adaptive difficulty immediately after each answer.
            new_level = get_next_difficulty(int(st.session_state["user_id"]))
            st.session_state["current_level"] = int(new_level)
        except Exception as exc:
            st.error(f"Failed to save result: {exc}")
            return

        if is_correct:
            st.success("Correct!")
        else:
            st.error(f"Incorrect. Correct answer is {correct_letter}.")

        explanation = str(question.get("explanation", "")).strip()
        if explanation:
            st.info(f"Explanation: {explanation}")


def _render_dashboard_tab() -> None:
    """Render learning analytics charts using Plotly."""
    st.subheader("Dashboard")

    if not st.session_state["user_id"]:
        st.info("Create or load a user from the sidebar first.")
        return

    render_dashboard(int(st.session_state["user_id"]))


def _render_admin_panel() -> None:
    """Render admin panel for generating multiple questions via Ollama."""
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Admin Panel (Tạo bài tập)")
    new_topic = st.sidebar.text_input("Nhập chủ đề muốn AI tạo:")
    new_diff = st.sidebar.slider("Chọn độ khó:", 1, 5, 1)
    num_questions = st.sidebar.slider("Số lượng câu hỏi:", 2, 10, 3)

    if st.sidebar.button("Sinh câu hỏi & Lưu vào DB", width="stretch"):
        if not new_topic:
            st.sidebar.warning("Vui lòng nhập chủ đề trước!")
            return

        progress_bar = st.sidebar.progress(
            0, text=f"🤖 Ollama đang tạo {num_questions} câu hỏi..."
        )

        try:
            questions = generate_batch(
                topic=new_topic, difficulty=new_diff, count=num_questions
            )
            progress_bar.progress(70, text="💾 Đang lưu vào Database...")

            saved = 0
            conn = _get_connection()
            try:
                for q in questions:
                    conn.execute(
                        """
                        INSERT INTO questions (content, difficulty, subject, options, answer, explanation)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            q.get("content"),
                            q.get("difficulty"),
                            q.get("subject"),
                            json.dumps(q.get("options"), ensure_ascii=False),
                            q.get("answer"),
                            q.get("explanation"),
                        ),
                    )
                    saved += 1
                conn.commit()
            finally:
                conn.close()

            progress_bar.progress(100, text="✅ Hoàn tất!")
            st.sidebar.success(
                f"✅ Đã tạo và lưu {saved}/{num_questions} câu hỏi "
                f"về '{new_topic}' (độ khó {new_diff})!"
            )
            if saved < num_questions:
                st.sidebar.info(
                    f"ℹ️ {num_questions - saved} câu không đạt validation đã bị bỏ qua."
                )
        except Exception as e:
            progress_bar.progress(100, text="❌ Lỗi!")
            st.sidebar.error(f"❌ Lỗi: {e}")


def main() -> None:
    """Application entrypoint."""
    st.set_page_config(
        page_title="AI Tutor V5.1",
        page_icon="🎓",
        layout="wide",
    )

    _safe_init_db()
    _ensure_session_state()
    _render_sidebar()
    _render_admin_panel()

    st.title("AI Tutor V5.1")

    tab_chat, tab_exercise, tab_dashboard = st.tabs(["Chat", "Exercise", "Dashboard"])

    with tab_chat:
        _render_chat_tab()

    with tab_exercise:
        _render_exercise_tab()

    with tab_dashboard:
        _render_dashboard_tab()


if __name__ == "__main__":
    main()
