import os
import json
import streamlit as st
from dotenv import load_dotenv
import chromadb
from groq import Groq
from openai import OpenAI


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATAFOLDER = os.path.join(os.path.dirname(__file__), "_data")
print(DATAFOLDER)

st.set_page_config(
    page_title="NovaTech Assistant",
    layout="centered"
)
st.title("NovaTech AI assistant")
st.caption("Ask anything about the company policy details")
st.divider()


def load_rag():
    if not os.path.exists(DATAFOLDER):
        st.error(F"_data folder not found in path {DATAFOLDER}")
        st.stop()

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    chroma = chromadb.EphemeralClient()

    try:
        chroma.delete_collection('docs')
    except Exception:
        pass

    collection = chroma.create_collection('docs')

    chunks = []
    ids = []
    metadatas = []
    chunk_id = 0

    for filename in sorted(os.listdir(DATAFOLDER)):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(DATAFOLDER, filename), "r") as f:
            text = f.read()

            paragraph = text.strip().split("\n\n")

            for para in paragraph:
                para = para.strip()

                if len(para) < 50:
                    continue

                if para.startswith("=="):
                    continue

                chunks.append(para)
                ids.append(f"chunk_{chunk_id}")
                metadatas.append({"sources": filename, "text": para})
                chunk_id = chunk_id + 1
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

    return collection, client, chunk_id


with st.spinner(text="Document loading and chunking are in progress..."):
    collection, client, total_chunks = load_rag()

st.success(f"Ready document chunks indexed")

st.divider()

# =====================================================
# CHAT HISTORY
# ====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


def ask_rag(question: str) -> dict:

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    chunks = results["documents"][0]
    sources = [m["sources"] for m in results["metadatas"][0]]
    context = "\n\n".join(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "you are a helful assitant to answer company policy questions based on context"
                "If the context does not have answeer to provide answer and tell not have enough information"
                "dont create answers"

            )
        },
        {
            "role": "user",
            "content": f"Context: {context}, Question: {question}"
        }
    ]

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    print(f"Answer: {answer}")
    print(f"{"==" * 60}")

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "chunks": list(zip(chunks, sources))
    }


question = st.chat_input("Ask about Novatech company or polices")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        result = ask_rag(question)

        st.markdown(result["answer"])
        sources = result.get("sources", [])

        if isinstance(sources, list):
            st.caption("Sources: " + ", ".join(map(str, sources)))
        else:
            st.caption(f"Sources: {sources}")

        with st.expander("View retried document chunks"):
            for i, (chunk_text, source_file) in enumerate(result["chunks"], 1):
                st.markdown(f"** chunck {i} - '{source_file}'**")
                st.info(chunk_text)

        st.session_state.messages.append({
            "role": "assistant",
            "answer": result['answer'],
            "sources": result['sources'],
            "chunks": result['chunks']
        })
