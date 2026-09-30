"""Streamlit operations dashboard for the customer-support NLP system."""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from src.service import analyse_ticket

st.set_page_config(page_title="ResolveAI | Support Intelligence", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background:linear-gradient(120deg,#F7F9FC 0%,#FDFEFF 45%,#F1F5FF 100%); color:#101828; }
[data-testid="stSidebar"] { background:#101828; } [data-testid="stSidebar"] * { color:#F9FAFB !important; }
.block-container { max-width:1400px; padding-top:2.1rem; padding-bottom:3rem; }
.hero { background:linear-gradient(108deg,#101828 0%,#183B75 58%,#2153B8 100%); padding:2.1rem 2.3rem; border-radius:24px; color:#fff; box-shadow:0 22px 55px rgba(21,94,239,.20); margin-bottom:1.6rem; }
.eyebrow { font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:#B2CCFF; font-weight:500; }
.hero h1 { font-size:2.45rem; line-height:1.1; margin:.48rem 0 .65rem; letter-spacing:-.045em; color:#fff; }.hero p { margin:0; color:#D0D5DD; font-size:.98rem; max-width:690px; }
.hero-badge { display:inline-block; font-family:'DM Mono',monospace; font-size:.72rem; color:#B9F8F8; background:rgba(11,165,164,.18); border:1px solid rgba(170,255,250,.25); border-radius:99px; padding:.4rem .65rem; }
.section-label { font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.11em; color:#475467; text-transform:uppercase; font-weight:500; margin-bottom:.25rem; }
.info-card { background:rgba(255,255,255,.92); border:1px solid #E4E7EC; border-radius:18px; padding:1.35rem; box-shadow:0 8px 22px rgba(16,24,40,.045); min-height:148px; }.info-card h3 { font-size:1rem; margin:0 0 .55rem; color:#101828; }.info-card p,.info-card li { font-size:.84rem; color:#667085; line-height:1.6; }.info-card ul { margin:.25rem 0 0; padding-left:1.1rem; }
.metric-card { background:#fff; border:1px solid #E4E7EC; border-radius:16px; padding:.95rem 1.1rem; box-shadow:0 8px 20px rgba(16,24,40,.04); }.metric-card .label { color:#667085; font-family:'DM Mono',monospace; font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; }.metric-card .value { color:#101828; font-size:1.4rem; font-weight:800; letter-spacing:-.035em; margin-top:.25rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.status-critical { color:#B42318; background:#FEF3F2; border:1px solid #FECDCA; }.status-high { color:#B54708; background:#FFFAEB; border:1px solid #FEDF89; }.status-medium { color:#175CD3; background:#EFF8FF; border:1px solid #B2DDFF; }.status-low { color:#027A48; background:#ECFDF3; border:1px solid #ABEFC6; }.status-chip { border-radius:999px; padding:.28rem .62rem; font-family:'DM Mono',monospace; font-size:.71rem; font-weight:600; }
.evidence { background:#F8FAFC; border:1px solid #E4E7EC; border-left:4px solid #155EEF; padding:1rem 1.1rem; border-radius:10px; color:#344054; line-height:1.55; }.queue-title { font-size:1.05rem; font-weight:800; color:#fff; }
.stButton>button { border-radius:10px; padding:.65rem 1.1rem; font-weight:700; border:0; background:#155EEF; box-shadow:0 7px 14px rgba(21,94,239,.22); }.stButton>button:hover { background:#004EEB; border:0; }.stTextArea textarea { border-radius:14px!important; border:1px solid #D0D5DD!important; background:#fff!important; font-size:.98rem!important; } div[data-testid="stTabs"] button { font-weight:700; }
</style>
""", unsafe_allow_html=True)

api_url = os.getenv("API_URL", "http://127.0.0.1:8000")


def submit(message: str) -> dict:
    try:
        response = requests.post(f"{api_url}/tickets/analyse", json={"message": message, "save": True}, timeout=5)
        response.raise_for_status()
        result = response.json()
        result["execution_mode"] = "FastAPI + persistent ticket store"
        return result
    except requests.RequestException:
        result = analyse_ticket(message)
        result["execution_mode"] = "Local development fallback"
        return result


def ticket_queue() -> list[dict]:
    try:
        response = requests.get(f"{api_url}/tickets?limit=20", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def status_html(priority: str) -> str:
    return f'<span class="status-chip status-{priority.lower()}">{priority.upper()} PRIORITY</span>'


with st.sidebar:
    st.markdown("<div class='eyebrow'>ResolveAI workspace</div><div class='queue-title'>Support-agent queue</div>", unsafe_allow_html=True)
    tickets = ticket_queue()
    st.metric("Tickets in view", len(tickets))
    if tickets:
        st.dataframe(pd.DataFrame(tickets)[["intent", "priority", "department", "status"]], hide_index=True, use_container_width=True)
    else:
        st.caption("No persisted tickets yet. Start FastAPI to activate the queue.")
    st.divider(); st.caption("Safety mode: sensitive, uncertain, and human-requested cases always escalate.")

st.markdown("""<section class="hero"><span class="hero-badge">● LIVE TRIAGE WORKFLOW</span><h1>Turn customer friction into<br/>a clear next action.</h1><p>Classify intent, assess risk, retrieve policy evidence, and route each ticket—with humans retained for sensitive decisions.</p></section>""", unsafe_allow_html=True)
triage_tab, queue_tab, about_tab = st.tabs(["✦ Analyse ticket", "▦ Agent workspace", "⌘ System design"])

with triage_tab:
    input_col, guide_col = st.columns([1.75, .85], gap="large")
    with input_col:
        st.markdown("<div class='section-label'>Customer message</div>", unsafe_allow_html=True)
        message = st.text_area("Describe the issue", label_visibility="collapsed", height=176, placeholder="Example: My order ORD-12345 has not arrived and I need it urgently.")
        analyse = st.button("Analyse & create ticket  →", type="primary")
        st.caption("No refund, delivery, or account facts are invented. High-risk and uncertain tickets go to people.")
    with guide_col:
        st.markdown('<div class="info-card"><h3>What the system checks</h3><ul><li>Complaint intent and confidence</li><li>Sentiment and operational entities</li><li>Fraud and urgency signals</li><li>Grounded policy evidence</li></ul></div>', unsafe_allow_html=True)
    if analyse:
        try:
            result = submit(message)
            st.markdown("<br/><div class='section-label'>Decision summary</div>", unsafe_allow_html=True)
            entries = [("Intent", result["intent"].replace("_", " ").title()), ("Priority", result["priority"].title()), ("Route", result["department"]), ("Confidence", f"{result['confidence']:.0%}")]
            for column, (label, value) in zip(st.columns(4, gap="small"), entries):
                column.markdown(f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<br/>{status_html(result['priority'])}", unsafe_allow_html=True)
            if result["escalate_to_human"]: st.error("Human review required · " + " · ".join(result["escalation_reasons"]))
            else: st.success("Ready for automatic specialist routing.")
            reasoning_col, response_col = st.columns(2, gap="large")
            with reasoning_col:
                st.markdown("#### Why this route")
                for item in result["explanation"]: st.write("• " + item)
                st.markdown("#### Signal profile")
                st.write(f"**Sentiment:** {result['sentiment'].title()} &nbsp; · &nbsp; **Score:** {result['sentiment_score']:.2f}")
                st.bar_chart(pd.DataFrame(result["top_intents"]).set_index("intent"), horizontal=True)
            with response_col:
                st.markdown(f"#### Policy-backed next step · `{result['retrieval_backend']}`")
                st.markdown(f"<div class='evidence'>{result['suggested_response']['answer']}</div>", unsafe_allow_html=True)
                st.markdown("#### Retrieved evidence")
                st.dataframe(pd.DataFrame(result["retrieved_sources"])[["id", "title", "score"]], hide_index=True, use_container_width=True)
            with st.expander("Operational entities and audit details"):
                st.json(result["entities"]); st.caption(result["execution_mode"])
                if "ticket" in result: st.code(result["ticket"]["id"], language=None)
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))

with queue_tab:
    st.markdown("<div class='section-label'>Ticket operations</div><h2 style='margin-top:0'>Human review queue</h2>", unsafe_allow_html=True)
    tickets = ticket_queue()
    if tickets:
        frame = pd.DataFrame(tickets)
        col1, col2, col3 = st.columns(3)
        col1.metric("Critical tickets", int((frame.priority == "critical").sum())); col2.metric("Human escalations", int(frame.escalated.sum())); col3.metric("Open tickets", int((frame.status == "open").sum()))
        st.dataframe(frame[["id", "created_at", "intent", "priority", "department", "status", "customer_message"]], hide_index=True, use_container_width=True)
    else: st.info("The queue is empty. Submit a ticket while FastAPI is running, then refresh this tab.")

with about_tab:
    st.markdown("<div class='section-label'>NLP engineering scope</div><h2 style='margin-top:0'>Transparent by design</h2>", unsafe_allow_html=True)
    cards = [("01 / Understand", "TF-IDF + Logistic Regression establishes a measured baseline; DistilBERT is available as a separate experiment."), ("02 / Decide", "Priority combines intent, confidence, risk language, and explicit escalation rules. Sentiment is never the sole determinant."), ("03 / Ground", "FAQ/policy retrieval finds evidence before a suggested response is drafted. Unsupported or sensitive cases are escalated.")]
    for column, (title, body) in zip(st.columns(3), cards): column.markdown(f"<div class='info-card'><h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)
