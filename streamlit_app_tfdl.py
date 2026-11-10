
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# ENVIRONMENT CONFIGURATION
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Check your .env file.")
    st.stop()


# =====================================================
# STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NovaTech Assistant",
    layout="centered"
)

st.title("NovaTech AI Assistant")
st.caption("Ask anything about company policy details")
st.divider()


# =====================================================
# GROQ CLIENT
# =====================================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "qwen/qwen3.8-27b"

DATAFOLDER = Path(__file__).parent / "_data"


# =====================================================
# LOAD DOCUMENTS AND CREATE TF-IDF VECTORS
# =====================================================

@st.cache_resource
def load_rag():

    if not DATAFOLDER.exists():
        st.error(f"_data folder not found: {DATAFOLDER}")
        st.stop()

    chunks = []
    metadatas = []

    chunk_id = 0

    # Read all TXT files
    for filename in sorted(os.listdir(DATAFOLDER)):

        if not filename.lower().endswith(".txt"):
            continue

        file_path = DATAFOLDER / filename

        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()

        paragraphs = document_text.strip().split("\n")

        for para in paragraphs:

            para = para.strip()

            if len(para) < 50:
                continue

            if para.startswith("=="):
                continue

            chunks.append(para)

            metadatas.append({
                "source": filename,
                "chunk_id": chunk_id
            })

            chunk_id += 1

    if not chunks:
        st.error("No valid document chunks found in _data.")
        st.stop()

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    # Convert documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return chunks, metadatas, vectorizer, tfidf_matrix


# =====================================================
# INITIALIZE TF-IDF
# =====================================================

with st.spinner("Loading documents and creating TF-IDF vectors..."):

    chunks, metadatas, vectorizer, tfidf_matrix = load_rag()

st.success(f"Ready! Indexed {len(chunks)} document chunks.")

st.caption(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

st.divider()


# =====================================================
# RAG QUESTION ANSWERING
# =====================================================

def ask_rag(question: str):

    # Convert user question into TF-IDF vector
    query_vector = vectorizer.transform([question])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Get top 3 matching document chunks
    top_indices = similarities.argsort()[::-1][:3]

    # Ignore irrelevant results
    relevant_indices = [
        i for i in top_indices
        if similarities[i] > 0
    ]

    if not relevant_indices:

        return (
            "I could not find relevant information in the company documents.",
            []
        )

    # Retrieve matching chunks
    retrieved_chunks = [
        chunks[i] for i in relevant_indices
    ]

    sources = list(dict.fromkeys(
        metadatas[i]["source"]
        for i in relevant_indices
    ))

    # Prepare context
    context = "\n\n".join(retrieved_chunks)

    messages = [

        {
            "role": "system",
            "content": """
            You are a helpful NovaTech company policy assistant.

            Answer questions using ONLY the provided context.

            Rules:
            1. Do not invent information.
            2. Do not use outside knowledge.
            3. If the context does not contain the answer,
               clearly say there is not enough information.
            4. Provide clear and concise answers.
            """
        },

        {
            "role": "user",
            "content": f"""
            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            """
        }

    ]

    # Generate response using Groq
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return answer, sources


# =====================================================
# CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            st.caption(
                "Sources: " + ", ".join(message["sources"])
            )


# =====================================================
# CHAT INPUT
# =====================================================

question = st.chat_input("Ask a company policy question...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching company policies..."):

            try:

                answer, sources = ask_rag(question)

                st.markdown(answer)

                if sources:
                    st.caption(
                        "Sources: " + ", ".join(sources)
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:

                st.error(f"Error generating response: {e}")
