"""
app.py

Streamlit frontend for the SCOTUS RAG API (api.py). Talks to the FastAPI
backend over HTTP only -- no direct use of the retrieval/generation pipeline.

Run with:
    streamlit run app.py
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Indexed case titles, as built by case_store.build_case_index(). The /ask API
# only takes {question, access_key} -- it has no explicit case-pin parameter --
# so selecting a case here biases routing by folding the case name into the
# question text sent to the backend; resolve_case()'s lexical matching then
# favors that case. It is a nudge, not a guarantee.
CASES = [
    "Brown v. Board of Education",
    "Gideon v. Wainwright",
    "Marbury v. Madison",
    "Meyer v. Nebraska",
    "Miranda v. Arizona",
    "United States v. James Daniel Good Real Property",
]
ANY_CASE = "Any / Let the system decide"

st.set_page_config(page_title="SCOTUS Legal Aid", page_icon="⚖️")

st.title("SCOTUS Legal Aid")
st.caption(
    "Retrieval-augmented Q&A over a small, fixed set of indexed U.S. Supreme Court opinions."
)
st.caption(
    "Not legal advice, not a comprehensive legal database, and not a reflection of current law beyond these opinions."
)

col1, col2 = st.columns(2)
with col1:
    st.caption("In scope, e.g.:")
    st.markdown(
        "- What liberty interest did *Meyer v. Nebraska* recognize?\n"
        "- Why did the Court strike down the Nebraska statute?\n"
        "- What right did *Gideon v. Wainwright* establish?\n"
        "- What test did the Court apply in *Miranda v. Arizona*?"
    )
with col2:
    st.caption("Out of scope, e.g.:")
    st.markdown(
        "- Current or pending Supreme Court cases\n"
        "- General legal advice for your own situation\n"
        "- Cases outside the 6 indexed opinions\n"
        "- Non-legal questions"
    )

with st.sidebar:
    st.subheader("Options")
    selected_case = st.selectbox("Case", [ANY_CASE] + CASES)
    access_key = st.text_input("Access key (optional)", type="password")

if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

with st.form("ask_form"):
    question = st.text_input("Your question")
    submitted = st.form_submit_button("Ask")

if submitted:
    st.session_state.last_response = None
    st.session_state.last_error = None
    st.session_state.suggested_questions = []

    if not question.strip():
        st.session_state.last_error = "Please enter a question."
    else:
        sent_question = question.strip()
        if selected_case != ANY_CASE:
            sent_question = f"Regarding {selected_case}: {sent_question}"

        payload = {"question": sent_question}
        if access_key.strip():
            payload["access_key"] = access_key.strip()

        with st.spinner("Researching..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
            except requests.exceptions.RequestException as e:
                st.session_state.last_error = f"Could not reach the backend at {BACKEND_URL}: {e}"
            else:
                if resp.status_code == 200:
                    st.session_state.last_response = resp.json()
                else:
                    try:
                        body = resp.json()
                    except ValueError:
                        body = {}
                    if resp.status_code == 429:
                        retry_after = body.get("retry_after", "a while")
                        st.session_state.last_error = (
                            f"Rate limited. Try again in {retry_after} seconds."
                        )
                    elif resp.status_code == 400:
                        st.session_state.last_error = (
                            f"Out of scope: {body.get('detail', 'question rejected.')}"
                        )
                        st.session_state.suggested_questions = body.get("suggestions") or []
                    elif resp.status_code == 503:
                        st.session_state.last_error = "Daily usage limit reached. Try again tomorrow."
                    else:
                        st.session_state.last_error = f"Backend error ({resp.status_code}): {body}"

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    if st.session_state.suggested_questions:
        st.write("Try one of these instead:")
        for q in st.session_state.suggested_questions:
            st.write(f"- {q}")

if st.session_state.last_response:
    data = st.session_state.last_response

    case_name = data.get("case_name")
    routing_method = data.get("routing_method") or "unknown"

    if routing_method == "meta":
        st.caption("ℹ️ About this app")
    elif routing_method == "overview":
        st.caption("ℹ️ Case overview")
    else:
        st.caption(f"Case: {case_name or 'Unresolved'} | Routing: {routing_method}")
        if not case_name:
            st.info(
                "This question didn't match a specific indexed case. "
                "The answer below is based on a broader search across the corpus "
                "and may be less precise."
            )

    if data.get("faithful") is False:
        reason = data.get("flagged_reason") or "Potential faithfulness issue detected."
        st.warning(reason)

    st.write(data.get("answer", ""))

    sources = data.get("sources") or []
    deduped_sources = list(dict.fromkeys(sources))
    with st.expander(f"Sources ({len(deduped_sources)})"):
        if deduped_sources:
            for src in deduped_sources:
                st.write(f"- {src}")
        else:
            st.write("No sources returned.")
