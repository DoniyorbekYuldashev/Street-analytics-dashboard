import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import json
import re
from pathlib import Path
from groq import Groq

st.set_page_config(
    page_title="Street Analytics Dashboard",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path(__file__).parent / "cv_analytics.db"

# ── API key ───────────────────────────────────────────────────────────────────
_secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

with st.sidebar:
    st.title("⚙️ Settings")
    if _secret_key:
        st.success("✅ API key loaded")
        api_key = _secret_key
    else:
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        st.markdown("**Get free key:** [console.groq.com](https://console.groq.com)")
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Model: `llama-3.3-70b-versatile`")

# ── DB ────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM people_tracks", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    return df

def run_sql(query):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df

# ── LLM ──────────────────────────────────────────────────────────────────────
SCHEMA = """
Table: people_tracks
Columns: id, track_id, label TEXT ('male','female','child'), timestamp DATETIME
Dates available: 2026-05-28, 2026-05-29, 2026-05-30, 2026-05-31
"""

SYSTEM_PROMPT = f"""You are a data analyst for a street CV analytics system.
Schema: {SCHEMA}

Respond ONLY with raw JSON (no markdown):
{{
  "answer_type": "chart" | "text",
  "sql": "<valid SQLite query or empty>",
  "chart_type": "bar" | "pie" | "line" | "table" | "",
  "chart_title": "<title>",
  "text_answer": "<short answer under 20 words>"
}}

Rules:
- Count/stat questions → answer_type="text", run sql, put real number in text_answer
- Distribution/trend/comparison → answer_type="chart"
- ALWAYS convert 12h to 24h: 5pm=17, 6pm=18, 7pm=19, 8pm=20, 9am=9, 10am=10
- Hour filter: strftime('%H', timestamp) = '17'  (zero-padded string)
- Range: strftime('%H', timestamp) BETWEEN '17' AND '19'
- For "which day" questions: GROUP BY DATE(timestamp)
- For hourly trend: GROUP BY strftime('%H', timestamp)
- chart_type="line" for time/hour trends, "bar" for comparisons, "pie" for proportions
"""

def ask_llm(question, api_key):
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

# ── Chart renderer ────────────────────────────────────────────────────────────
COLOR_MAP = {"male": "#4C9BE8", "female": "#E87C9B", "child": "#F5C842"}

def render_chart(df, chart_type, title):
    if df is None or df.empty:
        st.warning("No data for this query.")
        return
    cols = df.columns.tolist()
    try:
        if chart_type == "pie" and len(cols) >= 2:
            fig = px.pie(df, names=cols[0], values=cols[1], title=title,
                         hole=0.35, color=cols[0], color_discrete_map=COLOR_MAP)
            fig.update_traces(textinfo="percent+label", textposition="inside")
        elif chart_type == "line" and len(cols) >= 2:
            fig = px.line(df, x=cols[0], y=cols[1], title=title, markers=True,
                          color=cols[2] if len(cols) > 2 else None,
                          color_discrete_map=COLOR_MAP)
            fig.update_layout(xaxis_title=cols[0], yaxis_title=cols[1])
        elif chart_type == "table":
            st.dataframe(df, use_container_width=True, height=400)
            return
        else:  # bar default
            if len(cols) >= 3:
                fig = px.bar(df, x=cols[0], y=cols[1], color=cols[2], title=title,
                             barmode="group", color_discrete_map=COLOR_MAP)
            else:
                fig = px.bar(df, x=cols[0], y=cols[1], title=title,
                             color=cols[0], color_discrete_map=COLOR_MAP)
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.dataframe(df, use_container_width=True)
        st.caption(f"Chart fallback: {e}")

# ── Load data ─────────────────────────────────────────────────────────────────
df_all = load_data()

st.markdown("## 🚶 Street People Analytics")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

col_viz, col_chat = st.columns([1.6, 1], gap="large")

# ── LEFT: Charts ──────────────────────────────────────────────────────────────
with col_viz:
    if st.session_state.last_result is None:
        st.subheader("📊 Overview")

        c1, c2 = st.columns(2)

        with c1:
            # Pie: overall gender split (exclude may28 raw day)
            df_synth = df_all[df_all["date"] != "2026-05-28"]
            counts = df_synth.groupby("label").size().reset_index(name="count")
            fig = px.pie(counts, names="label", values="count",
                         title="Gender Distribution", hole=0.35,
                         color="label", color_discrete_map=COLOR_MAP)
            fig.update_traces(textinfo="percent+label", textposition="inside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Bar: daily totals by gender
            daily = df_synth.groupby(["date","label"]).size().reset_index(name="count")
            fig = px.bar(daily, x="date", y="count", color="label",
                         barmode="group", title="Daily Count by Gender",
                         color_discrete_map=COLOR_MAP)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
                              xaxis_title="Date", yaxis_title="People")
            st.plotly_chart(fig, use_container_width=True)

        # Line: hourly traffic all days
        hourly = df_synth.groupby(["hour","label"]).size().reset_index(name="count")
        fig = px.line(hourly, x="hour", y="count", color="label",
                      title="Hourly Traffic by Gender (All Days Combined)",
                      markers=True, color_discrete_map=COLOR_MAP)
        fig.update_layout(xaxis=dict(tickmode="linear", dtick=1, title="Hour of Day"),
                          yaxis_title="People Count",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA")
        st.plotly_chart(fig, use_container_width=True)

    else:
        res = st.session_state.last_result
        st.subheader("📈 Query Result")
        if res["type"] == "text":
            st.markdown(f"<div style='font-size:2rem;font-weight:700;padding:2rem 0'>{res['answer']}</div>",
                        unsafe_allow_html=True)
        elif res["type"] == "chart":
            render_chart(res["df"], res["chart_type"], res["chart_title"])

        if st.button("← Back to Overview", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()

# ── RIGHT: Chat ───────────────────────────────────────────────────────────────
with col_chat:
    st.subheader("💬 Ask the Data")

    chat_box = st.container(height=480)
    with chat_box:
        if not st.session_state.messages:
            st.caption("Ask anything about the street data…")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("e.g. How many people at 5pm? Which day was busiest?")

    if user_input:
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Thinking…"):
                try:
                    result = ask_llm(user_input, api_key)

                    if result["answer_type"] == "text":
                        answer = result.get("text_answer", "")
                        if result.get("sql"):
                            try:
                                df_res = run_sql(result["sql"])
                                if not df_res.empty:
                                    val = df_res.iloc[0, 0]
                                    answer = re.sub(r'\b\d+\b', str(val), answer, count=1) if answer else str(val)
                            except Exception:
                                pass
                        answer = str(answer) if answer else "No data found."
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.last_result = {"type": "text", "answer": answer}

                    else:
                        sql = result.get("sql", "")
                        if sql:
                            df_res = run_sql(sql)
                            title = result.get("chart_title", user_input)
                            st.session_state.last_result = {
                                "type": "chart", "df": df_res,
                                "chart_type": result.get("chart_type", "bar"),
                                "chart_title": title,
                            }
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"📊 **{title}**"})
                        else:
                            st.session_state.messages.append(
                                {"role": "assistant", "content": "Couldn't generate a chart for that."})

                except json.JSONDecodeError:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "⚠️ Couldn't parse response. Try rephrasing."})
                except Exception as e:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ {e}"})

            st.rerun()
