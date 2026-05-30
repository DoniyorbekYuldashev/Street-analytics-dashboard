import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json
import re
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Street Analytics Dashboard",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path(__file__).parent / "cv_analytics.db"

# ── API key: secrets → sidebar fallback ──────────────────────────────────────
_secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

with st.sidebar:
    st.title("⚙️ Settings")
    if _secret_key:
        st.success("✅ API key loaded from secrets")
        api_key = _secret_key
    else:
        api_key = st.text_input("Groq API Key", type="password", placeholder="Paste your key here…")
        st.markdown(
            "**Get a free key:**  \n"
            "[console.groq.com →](https://console.groq.com)"
        )
    st.markdown("---")
    st.caption("Data: `cv_analytics.db`  \nModel: `llama-3.3-70b-versatile`")

# ── DB helpers ────────────────────────────────────────────────────────────────
@st.cache_data
def load_overview():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM people_tracks", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    return df

def run_sql(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df

# ── Schema string for the LLM ─────────────────────────────────────────────────
SCHEMA = """
Table: people_tracks
Columns:
  id         INTEGER  (primary key)
  track_id   INTEGER  (unique person track within a session)
  label      TEXT     (values: 'male', 'female', 'child')
  timestamp  DATETIME (format: 'YYYY-MM-DD HH:MM:SS')

Available dates in data: 2026-05-28, 2026-05-29, 2026-05-30, 2026-05-31
"""

SYSTEM_PROMPT = f"""You are a data analyst assistant for a computer-vision street analytics system.
The SQLite database schema is:
{SCHEMA}

Your job:
1. Understand the user's question.
2. Decide if a SQL query + chart is needed, or if a short direct answer suffices.
3. Respond with ONLY valid JSON in this exact format:

{{
  "answer_type": "chart" | "text",
  "sql": "<SQL query or empty string>",
  "chart_type": "bar" | "pie" | "line" | "table" | "",
  "chart_title": "<title or empty>",
  "text_answer": "<short direct answer if answer_type=text, or empty>"
}}

Rules:
- For counting/stat questions (how many, total, percent at a time range) → answer_type="text", sql with aggregate query, put the result in text_answer.
- For distribution/comparison questions → answer_type="chart".
- chart_type="table" for raw data requests.
- NEVER wrap JSON in markdown code blocks. Output raw JSON only.
- Keep text_answer under 30 words.
- ALWAYS use 24-hour format for time filtering. Convert AM/PM to 24h: 5pm=17, 6pm=18, 7pm=19, 8pm=20, 9am=09, etc.
- Use strftime('%H', timestamp) for hour filtering. strftime returns zero-padded strings like '17', '18'.
- For a single hour like "at 5pm": WHERE strftime('%H', timestamp) = '17'
- For a range like "5pm to 7pm": WHERE strftime('%H', timestamp) BETWEEN '17' AND '19'
- Always run the SQL and return the count in text_answer for counting questions.
"""

# ── Groq call ────────────────────────────────────────────────────────────────
def ask_gemini(user_question: str, api_key: str) -> dict:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

# ── Chart renderer ────────────────────────────────────────────────────────────
def render_chart(df: pd.DataFrame, chart_type: str, title: str):
    if df.empty:
        st.warning("Query returned no data.")
        return
    cols = df.columns.tolist()
    if chart_type == "pie" and len(cols) >= 2:
        fig = px.pie(df, names=cols[0], values=cols[1], title=title,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "bar" and len(cols) >= 2:
        fig = px.bar(df, x=cols[0], y=cols[1], title=title, color=cols[0],
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "line" and len(cols) >= 2:
        fig = px.line(df, x=cols[0], y=cols[1], title=title, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "table":
        st.dataframe(df, use_container_width=True)
    else:
        # Fallback: try bar
        if len(cols) >= 2:
            fig = px.bar(df, x=cols[0], y=cols[1], title=title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

# ── Main layout ───────────────────────────────────────────────────────────────
df_all = load_overview()

# Title
st.markdown("## 🚶 Street People Analytics")
st.markdown("---")

# ── Session state for chat ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None  # {"type": "chart"|"text", "df": df, ...}

# ── Two-column layout: left=viz, right=chat ───────────────────────────────────
col_viz, col_chat = st.columns([1.6, 1], gap="large")

# ── LEFT: Visualization panel ─────────────────────────────────────────────────
with col_viz:
    if st.session_state.last_result is None:
        # Default view: overview pie + daily bar
        st.subheader("📊 Overview")

        sub1, sub2 = st.columns(2)

        # Pie: gender distribution (all days)
        with sub1:
            gender_counts = df_all.groupby("label").size().reset_index(name="count")
            color_map = {"male": "#4C9BE8", "female": "#E87C9B", "child": "#F5C842"}
            fig_pie = px.pie(
                gender_counts,
                names="label",
                values="count",
                title="Gender Distribution (All Days)",
                color="label",
                color_discrete_map=color_map,
                hole=0.35,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Bar: people per day per gender
        with sub2:
            daily = df_all.groupby(["date", "label"]).size().reset_index(name="count")
            daily["date"] = daily["date"].astype(str)
            fig_bar = px.bar(
                daily,
                x="date",
                y="count",
                color="label",
                barmode="group",
                title="Daily Count by Gender",
                color_discrete_map=color_map,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Line: hourly traffic (all days combined)
        hourly = df_all.groupby("hour").size().reset_index(name="count")
        fig_line = px.line(
            hourly,
            x="hour",
            y="count",
            title="Hourly Traffic Pattern (All Days)",
            markers=True,
            labels={"hour": "Hour of Day", "count": "People Count"},
        )
        fig_line.update_xaxes(tickmode="linear", dtick=1)
        st.plotly_chart(fig_line, use_container_width=True)

    else:
        # Show result from last query
        res = st.session_state.last_result
        st.subheader("📈 Query Result")
        if res["type"] == "text":
            st.markdown(f"### {res['answer']}")
        elif res["type"] == "chart":
            render_chart(res["df"], res["chart_type"], res["chart_title"])

        if st.button("← Back to Overview", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()

# ── RIGHT: Chat panel ─────────────────────────────────────────────────────────
with col_chat:
    st.subheader("💬 Ask the Data")

    # Chat history display
    chat_container = st.container(height=460)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("e.g. How many people were at 5pm–6pm?")

    if user_input:
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("Thinking…"):
                try:
                    result = ask_gemini(user_input, api_key)  # uses Groq internally

                    if result["answer_type"] == "text":
                        answer = result.get("text_answer", "")
                        # Always run SQL if present and inject the real number
                        if result.get("sql"):
                            try:
                                df_res = run_sql(result["sql"])
                                if not df_res.empty:
                                    val = df_res.iloc[0, 0]
                                    # Replace placeholder numbers in answer with real value
                                    if answer:
                                        import re as _re
                                        answer = _re.sub(r'\b\d+\b', str(val), answer, count=1)
                                    else:
                                        answer = str(val)
                            except Exception as ex:
                                answer = answer or f"SQL error: {ex}"
                        answer_str = str(answer) if answer else "No data found."
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer_str}
                        )
                        st.session_state.last_result = {"type": "text", "answer": answer_str}

                    elif result["answer_type"] == "chart":
                        sql = result.get("sql", "")
                        if sql:
                            df_res = run_sql(sql)
                            st.session_state.last_result = {
                                "type": "chart",
                                "df": df_res,
                                "chart_type": result.get("chart_type", "bar"),
                                "chart_title": result.get("chart_title", user_input),
                            }
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"📊 Chart ready: **{result.get('chart_title', user_input)}**"}
                            )
                        else:
                            st.session_state.messages.append(
                                {"role": "assistant", "content": "Couldn't generate a chart for that."}
                            )

                except json.JSONDecodeError as e:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ Couldn't parse response. Try rephrasing."}
                    )
                except Exception as e:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ Error: {e}"}
                    )
                    if not api_key:
                        st.error("Please enter your Groq API key in the sidebar.")

            st.rerun()
