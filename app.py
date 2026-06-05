import streamlit as st
import os
import html
import shutil
import tempfile
import pickle
import random
from datetime import datetime
from dotenv import load_dotenv
from utils.pdf_reader import extract_pages
from utils.chunker import split_pages_into_chunks
from utils.embedder import build_faiss_index, load_faiss_index, retrieve_top_chunks
from utils.qa_engine import generate_answer
from utils.quiz_generator import generate_quiz
from utils.flashcard_generator import generate_flashcards

load_dotenv()
st.set_page_config(page_title="Clarix", page_icon="💡", layout="wide")
SESSIONS_FILE = "Clarix_sessions.pkl"

def save_sessions(sessions):
    to_save = {}
    for sid, s in sessions.items():
        to_save[sid] = {
            "name": s["name"], "pages": s["pages"],
            "created": s["created"], "chat": s["chat"],
            "chunks": s["chunks"]
        }
    with open(SESSIONS_FILE, "wb") as f:
        pickle.dump(to_save, f)

def load_sessions():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "rb") as f:
            saved = pickle.load(f)
    except Exception:
        return {}
    sessions = {}
    for sid, s in saved.items():
        try:
            save_dir = f"faiss_{sid}"
            if os.path.exists(f"{save_dir}/index.faiss"):
                index, chunks = load_faiss_index(save_dir)
            else:
                index, chunks = build_faiss_index(s["chunks"], save_dir=save_dir)
            sessions[sid] = {**s, "index": index, "chunks": chunks}
        except Exception as e:
            print(f"Error loading session {sid}: {e}")
    return sessions

def safe(v):  return html.escape(str(v), quote=True)
def nl2br(v): return safe(v).replace("\n", "<br>")

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
is_dark = st.session_state["theme"] == "dark"

# ── Color System ──────────────────────────────────────────────
if is_dark:
    bg           = "#111827"
    card         = "#1f2937"
    card2        = "#374151"
    txt          = "#f9fafb"
    muted        = "#9ca3af"
    border       = "#374151"
    active       = "#1e3a5f"
    inp_bg       = "#1f2937"
    msg_user     = "#1d4ed8"
    msg_user_txt = "#ffffff"
    msg_bot      = "#1f2937"
    msg_bot_txt  = "#f9fafb"
    msg_bot_bdr  = "#374151"
else:
    bg           = "#f8fafc"
    card         = "#ffffff"
    card2        = "#eff6ff"
    txt          = "#0f172a"
    muted        = "#64748b"
    border       = "#e2e8f0"
    active       = "#eff6ff"
    inp_bg       = "#ffffff"
    msg_user     = "#2563eb"
    msg_user_txt = "#ffffff"
    msg_bot      = "#ffffff"
    msg_bot_txt  = "#0f172a"
    msg_bot_bdr  = "#e2e8f0"

# Sidebar always dark navy
s_bg    = "#0f172a"
s_hover = "#1e3a5f"
s_bdr   = "#1e293b"
s_muted = "#94a3b8"

accent  = "#3b82f6"
accent2 = "#6366f1"
accent3 = "#06b6d4"
accent4 = "#10b981"
danger  = "#ef4444"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu,footer,header{{visibility:hidden}}
*{{box-sizing:border-box}}
html,body,.stApp{{
    font-family:'Inter',sans-serif !important;
    background:{bg} !important;
    color:{txt} !important;
    overflow-x:hidden
}}
.main .block-container{{
    overflow-x:hidden;
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1080px !important;
}}

/* ━━ SIDEBAR ━━ */
section[data-testid="stSidebar"]{{
    background:{s_bg} !important;
    border-right:1px solid {s_bdr} !important;
    min-width:260px !important;
    max-width:260px !important;
    transform:none !important;
}}
section[data-testid="stSidebar"] *{{
    color:#ffffff !important;
    font-family:'Inter',sans-serif !important;
}}
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"]{{display:none !important}}
section[data-testid="stSidebar"] .stButton>button{{
    background:transparent !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:8px !important;
    font-size:13px !important;
    font-weight:500 !important;
    text-align:left !important;
    padding:8px 10px !important;
    transition:background .15s !important;
    box-shadow:none !important;
    width:100% !important;
}}
section[data-testid="stSidebar"] .stButton>button:hover{{
    background:{s_hover} !important;
    color:#ffffff !important;
}}

/* ━━ MAIN BUTTONS ━━ */
.stButton>button{{
    background:{card} !important;
    color:{txt} !important;
    border:1.5px solid {border} !important;
    border-radius:10px !important;
    font-size:13px !important;
    font-weight:600 !important;
    font-family:'Inter',sans-serif !important;
    transition:all .15s !important;
    padding:8px 16px !important;
}}
.stButton>button:hover{{
    background:{active} !important;
    border-color:{accent} !important;
    color:{accent} !important;
}}
.stButton>button[kind="primary"]{{
    background:{accent} !important;
    color:#fff !important;
    border-color:{accent} !important;
}}
.stButton>button[kind="primary"]:hover{{
    background:{accent2} !important;
    border-color:{accent2} !important;
    color:#fff !important;
}}
.stButton>button:disabled{{opacity:.35 !important}}

/* ━━ TABS ━━ */
.stTabs [data-baseweb="tab-list"]{{
    background:transparent !important;
    border-bottom:2px solid {border} !important;
    gap:0 !important;
}}
.stTabs [data-baseweb="tab"]{{
    color:{muted} !important;
    font-size:14px;
    font-weight:600;
    padding:10px 24px;
    background:transparent !important;
    border-bottom:3px solid transparent;
    margin-bottom:-2px;
    font-family:'Inter',sans-serif !important;
}}
.stTabs [aria-selected="true"]{{
    color:{accent} !important;
    border-bottom:3px solid {accent} !important;
    background:transparent !important;
}}

/* ━━ CHAT INPUT ━━ */
[data-testid="stChatInput"]{{
    background:{inp_bg} !important;
    border:1.5px solid {border} !important;
    border-radius:14px !important;
}}
.stChatInput textarea{{
    background:{inp_bg} !important;
    color:{txt} !important;
    border:none !important;
    font-size:14px !important;
    font-family:'Inter',sans-serif !important;
}}
.stChatInput textarea::placeholder{{
    color:{muted} !important;
    opacity:1 !important;
}}
[data-testid="stChatInputSubmitButton"]{{
    background:{accent} !important;
    border-radius:8px !important;
    opacity:1 !important;
}}
[data-testid="stChatInputSubmitButton"]:hover{{
    background:{accent2} !important;
}}
[data-testid="stChatMessage"]{{
    background:transparent !important;
    border:none !important;
    padding:0 !important;
}}
[data-testid="stChatMessage"] img,
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"]{{display:none !important}}

/* ━━ FILE UPLOADER — fix overlap ━━ */
[data-testid="stFileUploader"]{{width:100% !important}}
[data-testid="stFileUploaderDropzone"]{{
    background:{inp_bg} !important;
    border:2px dashed {accent}88 !important;
    border-radius:12px !important;
    padding:20px 16px !important;
    text-align:center !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:center !important;
    justify-content:center !important;
    gap:4px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover{{
    border-color:{accent} !important;
    background:{active} !important;
}}
/* Hide internal duplicate markdown text */
[data-testid="stFileUploaderDropzone"] div[data-testid="stMarkdownContainer"]{{
    display:none !important;
}}
[data-testid="stFileUploaderDropzone"] span{{
    color:{muted} !important;
    font-size:13px !important;
    font-family:'Inter',sans-serif !important;
}}
[data-testid="stFileUploaderDropzone"] small{{
    color:{muted} !important;
    font-size:11px !important;
    display:block !important;
    margin-top:2px !important;
}}
[data-testid="stFileUploaderDropzone"] button{{
    background:{accent} !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:8px !important;
    font-size:13px !important;
    font-weight:600 !important;
    padding:8px 20px !important;
    margin-top:6px !important;
    font-family:'Inter',sans-serif !important;
    cursor:pointer !important;
}}
[data-testid="stFileUploaderDropzone"] button:hover{{
    background:{accent2} !important;
}}

/* ━━ TYPOGRAPHY ━━ */
p,li,label,span,div{{
    font-family:'Inter',sans-serif !important;
    color:{txt} !important;
}}
h1,h2,h3,h4{{
    font-family:'Inter',sans-serif !important;
    color:{txt} !important;
}}
div[data-testid="stMarkdownContainer"] p{{color:{txt} !important}}
.stRadio label{{color:{txt} !important}}

/* ━━ MISC ━━ */
.stProgress>div>div{{
    background:linear-gradient(90deg,{accent},{accent2}) !important;
    border-radius:10px !important;
}}
[data-testid="stTooltipIcon"],[role="tooltip"]{{display:none !important}}
.stSlider [data-testid="stTickBar"]{{display:none}}
hr{{border-color:{border} !important}}
</style>""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state["sessions"] = load_sessions()
for k, v in [
    ("active_session", None), ("show_uploader", True),
    ("auto_question", None),  ("flashcards", []),
    ("current_card", 0),      ("card_flipped", False),
    ("cards_studied", set())
]:
    if k not in st.session_state:
        st.session_state[k] = v

def reset_fc():
    st.session_state.update({
        "flashcards": [], "current_card": 0,
        "card_flipped": False, "cards_studied": set()
    })

def process_upload(f):
    sid = f"doc_{datetime.now().strftime('%H%M%S')}"
    with st.spinner("Processing your document..."):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(f.name)[1]
        ) as tmp:
            tmp.write(f.read()); p = tmp.name
        pages  = extract_pages(p)
        chunks = split_pages_into_chunks(pages)
        index, chunks = build_faiss_index(chunks, save_dir=f"faiss_{sid}")
        os.unlink(p)
    st.session_state["sessions"][sid] = {
        "name": f.name, "pages": len(pages),
        "chunks": chunks, "index": index,
        "chat": [], "created": datetime.now().strftime("%b %d, %H:%M")
    }
    save_sessions(st.session_state["sessions"])
    st.session_state.update({
        "active_session": sid,
        "show_uploader": False,
        "auto_question": None
    })
    reset_fc(); st.rerun()

def delete_session(sid):
    if sid in st.session_state["sessions"]:
        del st.session_state["sessions"][sid]
    faiss_dir = f"faiss_{sid}"
    if os.path.exists(faiss_dir):
        shutil.rmtree(faiss_dir)
    if st.session_state["active_session"] == sid:
        st.session_state.update({
            "active_session": None, "show_uploader": False
        })
        reset_fc()
    save_sessions(st.session_state["sessions"])

# ━━━━━━━━━━━━━━━━ SIDEBAR ━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 12px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">
        <div style="width:30px;height:30px;border-radius:8px;
        background:linear-gradient(135deg,{accent},{accent2});
        display:flex;align-items:center;justify-content:center;
        font-size:14px;font-weight:900;color:#fff;flex-shrink:0">💡</div>
        <div>
          <div style="font-size:16px;font-weight:800;color:#f8fafc;
          letter-spacing:-0.3px;line-height:1.1">Clarix</div>
          <div style="font-size:10px;color:{s_muted};margin-top:1px">
          Document Q&A</div>
        </div>
      </div>
    </div>
    <div style="height:1px;background:{s_bdr};margin:0 16px 12px"></div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([5, 1])
    with c1:
        if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
            st.session_state.update({
                "active_session": None,
                "show_uploader": False,
                "auto_question": None
            })
            reset_fc(); st.rerun()
    with c2:
        if st.button("🌙" if not is_dark else "☀️", key="theme_btn"):
            st.session_state["theme"] = "dark" if not is_dark else "light"
            st.rerun()

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.session_state["sessions"]:
        st.markdown(
            f"<p style='font-size:10px;color:{s_muted};margin:12px 0 6px 4px;"
            f"font-weight:700;letter-spacing:0.1em'>RECENTS</p>",
            unsafe_allow_html=True
        )
        for sid, s in reversed(list(st.session_state["sessions"].items())):
            is_act  = sid == st.session_state["active_session"]
            name    = s["name"][:24]+"…" if len(s["name"])>24 else s["name"]
            bg_item = s_hover if is_act else "transparent"
            bl_clr  = accent  if is_act else "transparent"

            st.markdown(
                f"<div style='background:{bg_item};border-left:3px solid {bl_clr};"
                f"border-radius:8px;margin-bottom:2px'>",
                unsafe_allow_html=True
            )
            col_name, col_del = st.columns([10, 2])
            with col_name:
                if st.button(f"📄  {name}", key=f"card_{sid}",
                             use_container_width=True):
                    st.session_state.update({
                        "active_session": sid,
                        "show_uploader": False,
                        "auto_question": None
                    })
                    reset_fc(); st.rerun()
            with col_del:
                if st.button("✕", key=f"del_{sid}"):
                    delete_session(sid); st.rerun()

            st.markdown(
                f"<p style='font-size:10px;color:{s_muted};"
                f"margin:-2px 0 8px 12px;line-height:1.4'>"
                f"{s['created']} · {s['pages']}p · "
                f"{len(s['chat'])} Q&As</p></div>",
                unsafe_allow_html=True
            )

    st.markdown(f"""
    <div style="position:absolute;bottom:0;left:0;right:0;
    padding:14px 18px;border-top:1px solid {s_bdr};
    background:{s_bg}">
      <div style="display:flex;gap:14px">
        <span style="font-size:11px;color:{s_muted}">💬 Chat</span>
        <span style="font-size:11px;color:{s_muted}">🧠 Quiz</span>
        <span style="font-size:11px;color:{s_muted}">🃏 Cards</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━ WELCOME ━━━━━━━━━━━━━━━━
if not st.session_state["active_session"]:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:36px">
          <div style="width:56px;height:56px;border-radius:16px;
          margin:0 auto 16px;
          background:linear-gradient(135deg,{accent},{accent2});
          display:flex;align-items:center;justify-content:center;
          font-size:24px;color:#fff;
          box-shadow:0 8px 24px {accent}44">💡</div>
          <h1 style="font-size:28px;font-weight:800;color:{txt};
          margin:0 0 10px;letter-spacing:-.5px;line-height:1.2">
          Hi there! 👋</h1>
          <p style="font-size:14px;color:{muted};line-height:1.7;
          margin:0 auto;max-width:300px">
          Upload a document and ask questions,
          generate quizzes, or create flashcards.
          </p>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for cw, icon, title, sub, clr in [
            (c1, "💬", "Ask Questions", "Chat with your PDF",  accent),
            (c2, "🧠", "Generate Quiz", "Test your knowledge", accent2),
            (c3, "🃏", "Flashcards",    "Study smarter",       accent3),
        ]:
            with cw:
                st.markdown(f"""
                <div style="background:{card};
                border:1.5px solid {border};
                border-top:3px solid {clr};
                border-radius:12px;
                padding:18px 12px 14px;
                text-align:center;
                margin-bottom:20px">
                  <div style="font-size:22px;margin-bottom:8px">{icon}</div>
                  <div style="font-size:12px;font-weight:700;
                  color:{txt};margin-bottom:3px">{title}</div>
                  <div style="font-size:11px;color:{muted}">{sub}</div>
                </div>""", unsafe_allow_html=True)

        # Upload box — clean, no overlap
        st.markdown(f"""
        <div style="background:{card};border:1.5px solid {border};
        border-radius:14px;overflow:hidden;margin-top:4px">
          <div style="padding:18px 20px 14px;
          border-bottom:1px solid {border};text-align:center">
            <div style="font-size:24px;margin-bottom:8px">📂</div>
            <div style="font-size:14px;font-weight:700;
            color:{txt};margin-bottom:3px">Upload your document</div>
            <div style="font-size:12px;color:{muted}">
            PDF or TXT &nbsp;·&nbsp; Max 200 MB</div>
          </div>
          <div style="padding:10px 16px 14px">
        """, unsafe_allow_html=True)
        mf = st.file_uploader("", type=["pdf", "txt"],
                               label_visibility="collapsed", key="main_up")
        st.markdown("</div></div>", unsafe_allow_html=True)
        if mf: process_upload(mf)

# ━━━━━━━━━━━━━━━━ CHAT SCREEN ━━━━━━━━━━━━━━━━
else:
    session = st.session_state["sessions"][
        st.session_state["active_session"]
    ]
    sn = safe(session["name"])
    tab1, tab2, tab3 = st.tabs([
        "💬  Chat", "🧠  Generate Quiz", "🃏  Flashcards"
    ])

    with tab1:
        st.markdown(f"""
        <div style="background:{card};border:1.5px solid {border};
        border-radius:12px;padding:12px 18px;margin-bottom:16px;
        display:flex;align-items:center;gap:12px">
          <div style="background:linear-gradient(135deg,{accent},{accent2});
          border-radius:8px;width:38px;height:38px;
          display:flex;align-items:center;justify-content:center;
          font-size:16px;flex-shrink:0">📄</div>
          <div style="min-width:0;flex:1">
            <div style="font-size:14px;font-weight:700;color:{txt};
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
            {sn}</div>
            <div style="font-size:11px;color:{muted};margin-top:2px">
            {session['pages']} pages · {len(session['chat'])} questions asked
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state["auto_question"]:
            aq = st.session_state["auto_question"]
            st.session_state["auto_question"] = None
            with st.spinner("Thinking..."):
                top = retrieve_top_chunks(aq, session["index"], session["chunks"])
                res = generate_answer(aq, top)
            session["chat"].append({"question": aq, "answer": res["answer"]})
            save_sessions(st.session_state["sessions"]); st.rerun()

        if not session["chat"]:
            st.markdown(f"""
            <div style="text-align:center;padding:48px 20px 28px">
              <div style="font-size:32px;margin-bottom:14px;opacity:.3">💡</div>
              <div style="font-size:18px;font-weight:700;color:{txt};
              margin-bottom:8px">Ask me anything</div>
              <div style="font-size:13px;color:{muted}">
              about <b style="color:{txt}">{sn}</b></div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            for cw, key, lbl, q in [
                (c1,"p1","📝 Summarize","Summarize this document"),
                (c2,"p2","🔑 Key topics","What are the key topics?"),
                (c3,"p3","💡 Main concept","Explain the main concept"),
            ]:
                with cw:
                    if st.button(lbl, use_container_width=True, key=key):
                        st.session_state["auto_question"] = q; st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)

        for chat in session["chat"]:
            uq = nl2br(chat.get("question", ""))
            ab = nl2br(chat.get("answer", ""))

            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;
            margin-bottom:12px;gap:10px;align-items:flex-end">
              <div style="background:{msg_user};color:{msg_user_txt};
              border-radius:18px 18px 4px 18px;
              padding:11px 15px;max-width:68%;
              font-size:14px;line-height:1.65">
              {uq}</div>
              <div style="width:32px;height:32px;min-width:32px;
              border-radius:50%;flex-shrink:0;
              background:linear-gradient(135deg,{accent},{accent2});
              display:flex;align-items:center;justify-content:center;
              font-size:11px;font-weight:700;color:#fff">You</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;
            margin-bottom:20px;gap:10px;align-items:flex-start">
              <div style="width:32px;height:32px;min-width:32px;
              border-radius:50%;flex-shrink:0;
              background:linear-gradient(135deg,{accent2},{accent3});
              display:flex;align-items:center;justify-content:center;
              font-size:13px;color:#fff;font-weight:800;margin-top:2px">💡</div>
              <div style="background:{msg_bot};color:{msg_bot_txt};
              border-radius:4px 18px 18px 18px;
              padding:13px 17px;max-width:76%;
              font-size:14px;line-height:1.75;
              border:1px solid {msg_bot_bdr}">
              {ab}</div>
            </div>""", unsafe_allow_html=True)

        question = st.chat_input(
            f"Ask anything about {session['name']}..."
        )
        if question:
            with st.spinner("Thinking..."):
                top = retrieve_top_chunks(
                    question, session["index"], session["chunks"]
                )
                res = generate_answer(question, top)
            session["chat"].append({
                "question": question, "answer": res["answer"]
            })
            save_sessions(st.session_state["sessions"]); st.rerun()

    with tab2:
        st.markdown(f"""
        <div style="background:{card};border:1.5px solid {border};
        border-left:4px solid {accent2};border-radius:12px;
        padding:16px 20px;margin-bottom:24px">
          <h3 style="color:{txt};margin:0 0 4px;font-size:16px;font-weight:700">
          🧠 Quiz Generator</h3>
          <p style="color:{muted};font-size:13px;margin:0;font-weight:400">
          Auto-generate multiple choice questions from your document</p>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1: num_q = st.slider("Number of questions", 3, 10, 5)
        with c2:
            st.write(""); st.write("")
            gen = st.button("🎯 Generate", type="primary",
                            use_container_width=True)

        if gen:
            with st.spinner("Generating quiz..."):
                plain = [
                    c["text"] if isinstance(c, dict) else c
                    for c in session["chunks"]
                ]
                quiz = generate_quiz(plain, num_questions=num_q)
            if not quiz:
                st.error("Failed to generate. Please try again.")
            else:
                st.success(f"✅ {len(quiz)} questions generated!")
                st.divider()
                answers = []
                for i, q in enumerate(quiz):
                    st.markdown(f"""
                    <div style="background:{card};
                    border:1.5px solid {border};
                    border-left:4px solid {accent2};
                    border-radius:10px;padding:14px 18px;
                    margin-bottom:10px">
                    <p style="font-size:14px;font-weight:600;
                    color:{txt};margin:0;line-height:1.5">
                    Q{i+1}. {safe(q['question'])}</p>
                    </div>""", unsafe_allow_html=True)
                    ans = st.radio("", q["options"], key=f"q_{i}",
                                   label_visibility="collapsed")
                    answers.append({"user": ans, "correct": q["answer"]})
                    st.write("")

                if st.button("✅ Submit Answers", type="primary"):
                    score = sum(
                        1 for a in answers if a["user"]==a["correct"]
                    )
                    pct = int(score/len(quiz)*100)
                    clr = (accent4 if pct>=70
                           else "#f59e0b" if pct>=40 else danger)
                    st.balloons()
                    st.markdown(f"""
                    <div style="background:{card};
                    border:2px solid {clr}44;
                    border-radius:14px;padding:32px;
                    text-align:center;margin:16px 0">
                      <div style="font-size:48px;font-weight:800;
                      color:{clr};line-height:1">
                      {score}/{len(quiz)}</div>
                      <div style="font-size:14px;color:{muted};
                      margin-top:10px;font-weight:500">
                      {pct}% — {
                        'Excellent! 🎉' if pct>=70
                        else 'Good try! 💪' if pct>=40
                        else 'Keep studying! 📚'
                      }</div>
                    </div>""", unsafe_allow_html=True)
                    for i, a in enumerate(answers):
                        if a["user"]==a["correct"]:
                            st.success(f"Q{i+1}: ✅ Correct!")
                        else:
                            st.error(
                                f"Q{i+1}: ❌ Wrong! "
                                f"Correct: {a['correct']}"
                            )

    with tab3:
        st.markdown(f"""
        <div style="background:{card};border:1.5px solid {border};
        border-left:4px solid {accent3};border-radius:12px;
        padding:16px 20px;margin-bottom:24px">
          <h3 style="color:{txt};margin:0 0 4px;font-size:16px;font-weight:700">
          🃏 Flashcard Generator</h3>
          <p style="color:{muted};font-size:13px;margin:0;font-weight:400">
          Generate interactive study flashcards from your document</p>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1: num_cards = st.slider("Number of flashcards", 5, 20, 10)
        with c2:
            st.write(""); st.write("")
            gc = st.button("✨ Generate", type="primary",
                           use_container_width=True)

        if gc:
            with st.spinner("Generating flashcards..."):
                plain = [
                    c["text"] if isinstance(c, dict) else c
                    for c in session["chunks"]
                ]
                cards = generate_flashcards(plain, num_cards=num_cards)
            if not cards: st.error("Failed. Try again!")
            else:
                st.session_state.update({
                    "flashcards": cards, "current_card": 0,
                    "card_flipped": False, "cards_studied": set()
                }); st.rerun()

        if st.session_state["flashcards"]:
            cards   = st.session_state["flashcards"]
            idx     = st.session_state["current_card"]
            flipped = st.session_state["card_flipped"]
            studied = st.session_state["cards_studied"]
            fc      = cards[idx]

            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
            align-items:center;margin-bottom:8px">
              <span style="font-size:13px;color:{muted};font-weight:600">
              Card {idx+1} of {len(cards)}</span>
              <span style="font-size:13px;color:{accent4};font-weight:600">
              ✅ {len(studied)} studied</span>
            </div>""", unsafe_allow_html=True)
            st.progress(len(studied)/len(cards))
            st.markdown("<br>", unsafe_allow_html=True)

            if not flipped:
                st.markdown(f"""
                <div style="background:{card};
                border:1.5px solid {border};
                border-top:4px solid {accent3};
                border-radius:16px;padding:52px 44px;
                text-align:center;min-height:220px;
                display:flex;flex-direction:column;
                justify-content:center;align-items:center">
                  <div style="font-size:10px;font-weight:700;
                  color:{accent3};letter-spacing:.2em;
                  margin-bottom:20px">QUESTION</div>
                  <div style="font-size:18px;font-weight:600;
                  color:{txt};line-height:1.6;max-width:600px">
                  {safe(fc['front'])}</div>
                  <div style="font-size:12px;color:{muted};margin-top:24px">
                  Click Show Answer to reveal</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:{active};
                border:1.5px solid {accent}44;
                border-top:4px solid {accent};
                border-radius:16px;padding:52px 44px;
                text-align:center;min-height:220px;
                display:flex;flex-direction:column;
                justify-content:center;align-items:center">
                  <div style="font-size:10px;font-weight:700;
                  color:{accent};letter-spacing:.2em;
                  margin-bottom:20px">ANSWER</div>
                  <div style="font-size:16px;font-weight:400;
                  color:{txt};line-height:1.75;max-width:640px">
                  {nl2br(fc['back'])}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("← Prev", use_container_width=True,
                             disabled=idx==0):
                    st.session_state["current_card"]=idx-1
                    st.session_state["card_flipped"]=False; st.rerun()
            with c2:
                if not flipped:
                    if st.button("Show Answer", use_container_width=True,
                                 type="primary"):
                        st.session_state["card_flipped"]=True
                        st.session_state["cards_studied"].add(idx); st.rerun()
                else:
                    if st.button("Hide", use_container_width=True):
                        st.session_state["card_flipped"]=False; st.rerun()
            with c3:
                if st.button("Next →", use_container_width=True,
                             disabled=idx==len(cards)-1):
                    st.session_state["current_card"]=idx+1
                    st.session_state["card_flipped"]=False; st.rerun()
            with c4:
                if st.button("🔀 Shuffle", use_container_width=True):
                    random.shuffle(st.session_state["flashcards"])
                    st.session_state.update({
                        "current_card":0,"card_flipped":False,
                        "cards_studied":set()
                    }); st.rerun()

            if len(studied)==len(cards):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:{'#052e16' if is_dark else '#f0fdf4'};
                border:1.5px solid {accent4}55;
                border-radius:12px;padding:28px;text-align:center">
                  <div style="font-size:32px;margin-bottom:10px">🎉</div>
                  <div style="font-size:17px;font-weight:700;
                  color:{accent4};margin-bottom:4px">All cards studied!</div>
                  <div style="font-size:13px;color:{muted}">
                  You reviewed all {len(cards)} flashcards.</div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Start Over", use_container_width=True):
                    st.session_state.update({
                        "current_card":0,"card_flipped":False,
                        "cards_studied":set()
                    }); st.rerun()