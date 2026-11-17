
import streamlit as st

from app.chains.qa_chain import answer_question


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="ResearchIQ",
    page_icon="📚",
    layout="wide",
)

# =========================
# APPLICATION HEADER
# =========================

st.title("📚 ResearchIQ")
st.caption(
    "AI-powered research assistant for understanding "
    "academic papers, figures, charts, and diagrams."
)

st.divider()

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("ResearchIQ")

    st.markdown(
        """
        **Capabilities**

        - Research paper question answering
        - Figure and diagram understanding
        - Chart description analysis
        - Source-based answers
        """
    )

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by LangChain, ChromaDB, Docling and Groq.")


# =========================
# CHAT HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# DISPLAY CHAT HISTORY
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("View Retrieved Sources"):
                    for index, document in enumerate(sources, start=1):

                        metadata = document.metadata

                        source = metadata.get(
                            "source", "Unknown source"
                        )

                        pages = metadata.get(
                            "page_numbers", []
                        )

                        if pages:
                            page_text = ", ".join(
                                str(page) for page in pages
                            )
                        else:
                            page_text = str(
                                metadata.get(
                                    "page_start", "Unknown"
                                )
                            )

                        st.markdown(
                            f"**Source {index}: {source}**"
                        )
                        st.caption(f"Page: {page_text}")

                        st.write(document.page_content)

                        if index < len(sources):
                            st.divider()


# =========================
# USER INPUT
# =========================

question = st.chat_input(
    "Ask anything about your research papers..."
)


# =========================
# QUESTION ANSWERING
# =========================

if question:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Analyzing research papers..."):

            try:
                result = answer_question(question)

                answer = result["answer"]
                sources = result["source_documents"]

                st.markdown(answer)

                if sources:
                    with st.expander("View Retrieved Sources"):

                        for index, document in enumerate(
                            sources, start=1
                        ):

                            metadata = document.metadata

                            source = metadata.get(
                                "source", "Unknown source"
                            )

                            pages = metadata.get(
                                "page_numbers", []
                            )

                            if pages:
                                page_text = ", ".join(
                                    str(page) for page in pages
                                )
                            else:
                                page_text = str(
                                    metadata.get(
                                        "page_start", "Unknown"
                                    )
                                )

                            st.markdown(
                                f"**Source {index}: {source}**"
                            )

                            st.caption(
                                f"Page: {page_text}"
                            )

                            st.write(document.page_content)

                            if index < len(sources):
                                st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                error_message = (
                    f"Something went wrong: {str(error)}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
