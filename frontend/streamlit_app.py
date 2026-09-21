import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Research Paper RAG", page_icon="")
st.title("Research Paper RAG")
st.caption("Upload a research paper, then ask questions about it — answers are grounded only in what you upload.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload a paper")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None and st.button("Index this paper"):
        with st.spinner("Extracting, chunking, and embedding..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
                response.raise_for_status()
                data = response.json()
                st.success(
                    f"Indexed '{data['filename']}' — "
                    f"{data['pages_extracted']} pages, {data['chunks_created']} chunks."
                )
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Is `uvicorn app.main:app --reload` running?")
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except ValueError:
                    detail = exc.response.text or f"Server returned {exc.response.status_code} with no body — check the uvicorn terminal for a traceback."
                st.error(f"Upload failed: {detail}")


for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            with st.expander(f"Sources ({len(entry['sources'])})"):
                for source in entry["sources"]:
                    st.markdown(
                        f"**{source['filename']}**, page {source['page_number']} "
                        f"(similarity: {source['similarity_score']})"
                    )
                    st.text(source["chunk_text"][:300] + "...")

question = st.chat_input("Ask a question about your uploaded papers...")

if question:
    st.session_state.chat_history.append({"question": question, "answer": "", "sources": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and asking Grok..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat", json={"question": question}, timeout=60
                )
                response.raise_for_status()
                data = response.json()
                st.write(data["answer"])

                if data["sources"]:
                    with st.expander(f"Sources ({len(data['sources'])})"):
                        for source in data["sources"]:
                            st.markdown(
                                f"**{source['filename']}**, page {source['page_number']} "
                                f"(similarity: {source['similarity_score']})"
                            )
                            st.text(source["chunk_text"][:300] + "...")

                st.session_state.chat_history[-1]["answer"] = data["answer"]
                st.session_state.chat_history[-1]["sources"] = data["sources"]
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Is `uvicorn app.main:app --reload` running?")
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except ValueError:
                    detail = exc.response.text or f"Server returned {exc.response.status_code} with no body — check the uvicorn terminal for a traceback."
                st.error(f"Request failed: {detail}")
