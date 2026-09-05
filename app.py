"""
Main Streamlit app for AI Tutor.

Streamlit reruns this script on every interaction, so all context that must
survive a rerun lives in ``st.session_state`` under the keys declared in
:data:`SESSION_DEFAULTS`.

Layering rules for this file:
- no SQL and no direct DB connections (repository functions do that);
- no grading or difficulty rules (:func:`controller.record_answer` owns them);
- no JSON parsing of stored options (:mod:`schemas` owns that).

The small pure helpers at the top of this module are intentionally free of
Streamlit calls so they can be unit-tested without a script runtime.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import streamlit as st

import schemas
from config import MAX_BATCH_SIZE
from controller import chat, record_answer
from generator import generate_batch
from dashboard import render_dashboard
from init_db import initialize_database
from logging_setup import configure_logging
from sqlite_manager import (
    get_all_subjects,
    get_or_create_user,
    get_questions_filtered,
    insert_questions,
)
from validation import require_level

ALL_SUBJECTS_LABEL = "Tất cả"
USER_NAME_MAX_LENGTH = 100

# Every session key the app relies on, with its initial value.
SESSION_DEFAULTS: Dict[str, Any] = {
    "user_id": None,
    "user_name": "",
    "current_level": 1,
    "chat_history": [],
    "current_question": None,
    "selected_answer": "",
    "last_feedback": "",
    "subject_filter": ALL_SUBJECTS_LABEL,
    "skip_answered": True,
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def parse_selected_option(label: str | None) -> str:
    """Extract the option letter from a rendered "A. text" radio label."""
    if not label:
        return ""
    return label.split(".", 1)[0].strip().upper()


def build_no_question_message(level: int, subject: Optional[str]) -> str:
    """Explain why the exercise tab has nothing to show, in user-facing Vietnamese."""
    subject_info = f" cho môn '{subject}'" if subject else ""
    return (
        f"Không tìm thấy câu hỏi ở độ khó {level}{subject_info}. "
        "Hãy thêm câu hỏi vào cơ sở dữ liệu hoặc đổi bộ lọc."
    )


def select_question(questions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick one question at random, or None when the pool is empty."""
    if not questions:
        return None
    return random.choice(questions)


def subject_options(subjects: List[str]) -> List[str]:
    """Options for the subject filter, with the 'all' entry first."""
    return [ALL_SUBJECTS_LABEL, *subjects]


def subject_selection_index(options: List[str], selected: str) -> int:
    """
    Index of the stored filter inside the option list.

    Falls back to "all" when the stored value no longer exists (e.g. the last
    question of that subject was deleted), which would otherwise crash the selectbox.
    """
    try:
        return options.index(selected)
    except ValueError:
        return 0


def _ensure_session_state() -> None:
    """Create every required session key once so values survive Streamlit reruns."""
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _current_user_id() -> Optional[int]:
    user_id = st.session_state.get("user_id")
    return int(user_id) if user_id else None


def _selected_subject() -> Optional[str]:
    subject = st.session_state.get("subject_filter", ALL_SUBJECTS_LABEL)
    return None if subject == ALL_SUBJECTS_LABEL else subject


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------
def _load_random_question_for_current_level() -> None:
    """
    Load one question for the current adaptive level, honouring the UI filters.

    Results are written to session state (including the "nothing found" message)
    rather than rendered here, so the render functions stay the only output path.
    """
    level = require_level(st.session_state["current_level"], "current_level")
    subject = _selected_subject()

    exclude_uid = None
    if st.session_state.get("skip_answered", False):
        exclude_uid = _current_user_id()

    question = select_question(
        get_questions_filtered(level=level, subject=subject, exclude_uid=exclude_uid)
    )

    if question is None:
        st.session_state["current_question"] = None
        st.session_state["last_feedback"] = build_no_question_message(level, subject)
        return

    st.session_state["current_question"] = question
    st.session_state["selected_answer"] = ""
    st.session_state["last_feedback"] = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    """Render user setup controls and keep the selected user in session state."""
    st.sidebar.title("AI Tutor")
    st.sidebar.caption("Trợ lý học tập cá nhân hóa")

    name_input = st.sidebar.text_input(
        "Tên học viên",
        value=st.session_state["user_name"],
        max_chars=USER_NAME_MAX_LENGTH,
    )

    if st.sidebar.button("Tạo / Tải người dùng", width="stretch"):
        if not name_input.strip():
            st.sidebar.warning("Vui lòng nhập tên học viên hợp lệ.")
        else:
            user_id, level = get_or_create_user(name_input)
            st.session_state["user_id"] = user_id
            st.session_state["user_name"] = name_input.strip()
            st.session_state["current_level"] = level
            st.sidebar.success(f"Mã người dùng đang sử dụng: {user_id}")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Độ khó hiện tại: {st.session_state['current_level']}")
    st.sidebar.write(f"Mã người dùng: {st.session_state['user_id']}")


def _render_admin_panel() -> None:
    """Render the admin controls that generate questions with the LLM."""
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Quản trị tạo bài tập")

    new_topic = st.sidebar.text_input("Nhập chủ đề muốn AI tạo:")
    new_diff = st.sidebar.slider("Chọn độ khó:", 1, 5, 1)
    num_questions = st.sidebar.slider("Số lượng câu hỏi:", 2, MAX_BATCH_SIZE, 3)

    if not st.sidebar.button("Sinh câu hỏi và lưu", width="stretch"):
        return

    if not new_topic.strip():
        st.sidebar.warning("Vui lòng nhập chủ đề trước!")
        return

    progress = st.sidebar.progress(0, text=f"🤖 AI đang tạo {num_questions} câu hỏi...")
    try:
        questions = generate_batch(topic=new_topic, difficulty=new_diff, count=num_questions)
        progress.progress(70, text="💾 Đang lưu vào cơ sở dữ liệu...")

        saved = insert_questions(questions)
        progress.progress(100, text="✅ Hoàn tất!")

        st.sidebar.success(
            f"✅ Đã tạo và lưu {saved}/{num_questions} câu hỏi "
            f"về '{new_topic}' (độ khó {new_diff})!"
        )
        if saved < len(questions):
            st.sidebar.info(
                f"ℹ️ {len(questions) - saved} câu không đạt kiểm tra định dạng đã bị bỏ qua."
            )
    except Exception as exc:  # noqa: BLE001 - a failed generation must not kill the rerun
        progress.progress(100, text="❌ Lỗi!")
        st.sidebar.error(f"❌ Lỗi: {exc}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def _render_chat_tab() -> None:
    """Render the chatbot with persistent history."""
    st.subheader("Trợ lý trò chuyện")

    if not _current_user_id():
        st.info("Hãy tạo hoặc tải người dùng từ thanh bên trước.")
        return

    # Replay history first: Streamlit shows widgets in call order.
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Nhập câu hỏi cho trợ lý...")
    if not user_prompt:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                answer = chat(user_prompt)
            except Exception as exc:  # noqa: BLE001 - keep the chat usable on failure
                answer = f"Lỗi khi tạo câu trả lời: {exc}"
            st.markdown(answer)

    st.session_state["chat_history"].append({"role": "assistant", "content": answer})


def _render_exercise_filters() -> None:
    """Render subject filter, level indicator, and the answered-question toggle."""
    col_subject, col_level, col_skip = st.columns([2, 2, 1])

    with col_subject:
        options = subject_options(get_all_subjects())
        index = subject_selection_index(options, st.session_state.get("subject_filter", ALL_SUBJECTS_LABEL))
        selected = st.selectbox("📚 Môn học:", options=options, index=index, key="subject_select")
        st.session_state["subject_filter"] = selected

    with col_level:
        st.metric("🎯 Độ khó hiện tại", st.session_state["current_level"])

    with col_skip:
        st.session_state["skip_answered"] = st.checkbox(
            "❌ Bỏ câu đã làm đúng",
            value=st.session_state.get("skip_answered", True),
            key="skip_chk",
        )


def _render_question_answer_form(question: Dict[str, Any], options_map: Dict[str, str]) -> None:
    """Render the radio group and handle submission/grading for one question."""
    labels = [f"{letter}. {text}" for letter, text in options_map.items()]

    selected = st.radio("Chọn đáp án của bạn:", options=labels, index=None, key="exercise_option_radio")

    if not st.button("Nộp đáp án", type="primary", width="stretch"):
        return

    if not selected:
        st.warning("Vui lòng chọn một đáp án trước khi nộp.")
        return

    try:
        result = record_answer(
            uid=_current_user_id(),
            question=question,
            selected_answer=parse_selected_option(selected),
        )
    except Exception as exc:  # noqa: BLE001 - show the failure, keep the page alive
        st.error(f"Không thể lưu kết quả: {exc}")
        return

    st.session_state["current_level"] = result["new_level"]

    if result["is_correct"]:
        st.success("Chính xác!")
    else:
        st.error(f"Chưa đúng. Đáp án đúng là {result['correct_answer']}.")

    if result["explanation"]:
        st.info(f"Giải thích: {result['explanation']}")


def _render_exercise_tab() -> None:
    """Render adaptive practice with subject filter and smart question loading."""
    st.subheader("📝 Luyện tập")

    if not _current_user_id():
        st.info("Hãy tạo hoặc chọn người dùng ở thanh bên trước.")
        return

    _render_exercise_filters()

    if st.button("🔄 Tải câu hỏi mới", width="stretch"):
        _load_random_question_for_current_level()

    question = st.session_state["current_question"]
    if not question:
        feedback = st.session_state["last_feedback"]
        if feedback:
            st.warning(feedback)
        else:
            st.info("Bấm 'Tải câu hỏi mới' để bắt đầu luyện tập.")
        return

    st.markdown(f"**Câu hỏi #{question['id']}**")
    st.write(question["content"])

    options_map = schemas.parse_options(question.get("options", ""))
    if not schemas.is_option_set_complete(options_map):
        st.error("Câu hỏi này có định dạng đáp án không hợp lệ trong cơ sở dữ liệu.")
        return

    _render_question_answer_form(question, options_map)


def _render_dashboard_tab() -> None:
    """Render learning analytics for the current user."""
    st.subheader("Thống kê học tập")

    if not _current_user_id():
        st.info("Hãy tạo hoặc tải người dùng từ thanh bên trước.")
        return

    render_dashboard(_current_user_id())


def _safe_init_db() -> None:
    """Initialize the database, stopping with a readable message when it fails."""
    try:
        initialize_database()
    except Exception as exc:  # noqa: BLE001 - a broken DB must not show a stack trace
        st.error(f"Không thể khởi tạo cơ sở dữ liệu: {exc}")
        st.stop()


def main() -> None:
    """Application entry point."""
    configure_logging()

    st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="wide")

    _safe_init_db()
    _ensure_session_state()
    _render_sidebar()
    _render_admin_panel()

    st.title("AI Tutor")

    tab_chat, tab_exercise, tab_dashboard = st.tabs(
        ["Trò chuyện", "Luyện tập", "Thống kê"]
    )

    with tab_chat:
        _render_chat_tab()
    with tab_exercise:
        _render_exercise_tab()
    with tab_dashboard:
        _render_dashboard_tab()


if __name__ == "__main__":
    main()
