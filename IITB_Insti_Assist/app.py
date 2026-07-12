import streamlit as st
from rag import ask


st.set_page_config(
    page_title="IITB Insti-Assist",
    page_icon="🎓",
    layout="centered"
)
st.title("🎓 IITB Insti-Assist")

st.write(
    "Ask questions about IIT Bombay using official institute documents."
)

st.divider()

with st.form("question_form", clear_on_submit=True):

    question = st.text_input(
        "Ask a question:",
        placeholder="e.g. What is CPI?"
    )

    submitted = st.form_submit_button("Ask")
if submitted:

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching official documents..."):

            answer, sources = ask(question)
        st.subheader("Answer")

        if len(sources) == 0:
            st.warning(answer)
        else:
            st.success(answer)

        if len(sources) > 0:

            st.divider()
            st.subheader("Sources")

            shown = set()

            for s in sources:

                key = (s["source"], s["page"])

                if key not in shown:

                    st.markdown(
                        f"📄 **{s['source']}** (Page {s['page']})"
                    )

                    shown.add(key)