
import os
from pathlib import Path

import streamlit as st
import chromadb

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "_data"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

TOP_K = 3

st.set_page_config(
    page_title="NovaTech AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# 2. PAGE HEADER
# ============================================================

st.title("NovaTech AI Assistant")

st.caption(
    "Ask questions about NovaTech company policies, "
    "employee benefits, leave, attendance and work policies."
)

st.divider()


# ============================================================
# 3. VALIDATE CONFIGURATION
# ============================================================

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is missing. "
        "Add it to your .env file or deployment environment variables."
    )
    st.stop()

if not DATA_FOLDER.exists():
    st.error(f"Policy documents folder not found: {DATA_FOLDER}")
    st.stop()


# ============================================================
# 4. DOCUMENT CHUNKING
# ============================================================

def load_documents():

    documents = []
    metadatas = []
    ids = []

    chunk_id = 0

    for file_path in sorted(DATA_FOLDER.glob("*.txt")):

        try:
            text = file_path.read_text(encoding="utf-8")

        except Exception as e:
            print(f"Could not read {file_path.name}: {e}")
            continue

        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if len(paragraph) < 50:
                continue

            if paragraph.startswith("=="):
                continue

            documents.append(paragraph)

            metadatas.append({
                "sources": file_path.name
            })

            ids.append(f"chunk_{chunk_id}")

            chunk_id += 1

    return documents, metadatas, ids


# ============================================================
# 5. INITIALIZE CHROMADB AND GEMINI
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag(api_key, model_name):

    # Initialize Gemini API
    gemini_client = genai.Client(api_key=api_key)

    # Initialize ChromaDB
    chroma_client = chromadb.EphemeralClient()

    collection = chroma_client.get_or_create_collection(
        name="novatech_documents",
        metadata={"hnsw:space": "cosine"}
    )

    # Load documents only if collection is empty
    if collection.count() == 0:

        documents, metadatas, ids = load_documents()

        if not documents:
            raise ValueError(
                "No valid text chunks found in the _data folder."
            )

        # Add in batches
        batch_size = 100

        for start in range(0, len(documents), batch_size):

            end = start + batch_size

            collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )

    return collection, gemini_client


# ============================================================
# 6. LOAD RAG COMPONENTS
# ============================================================

try:

    with st.spinner(
        "Loading company policies and initializing Gemini..."
    ):

        collection, gemini_client = initialize_rag(
            GEMINI_API_KEY,
            GEMINI_MODEL
        )

    st.success(
        f"Knowledge base ready! "
        f"{collection.count()} document chunks indexed."
    )

except Exception as e:

    st.error(f"RAG initialization failed: {str(e)}")
    st.stop()


st.divider()


# ============================================================
# 7. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(question):

    if collection.count() == 0:
        return [], []

    results = collection.query(
        query_texts=[question],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0] or []
    metadata = results["metadatas"][0] or []

    sources = [
        item.get("sources", "Unknown source")
        if item else "Unknown source"
        for item in metadata
    ]

    return chunks, sources


# ============================================================
# 8. GENERATE ANSWER USING GEMINI
# ============================================================

def ask_gemini(question, chunks, sources):

    if not chunks:

        return (
            "I could not find relevant information in "
            "the available company policy documents."
        )

    context = ""

    for index, (chunk, source) in enumerate(
        zip(chunks, sources), 1
    ):

        context += f"""
        DOCUMENT CHUNK {index}
        SOURCE: {source}

        CONTENT:
        {chunk}

        -------------------------
        """

    prompt = f"""
    You are NovaTech's company policy assistant.

    Answer employee questions using ONLY the policy context
    provided below.

    STRICT RULES:

    1. Do not invent company policies.
    2. Do not use outside knowledge to answer policy questions.
    3. If the answer is not available in the context,
       clearly state:
       "I could not find sufficient information in the
       available NovaTech policy documents."
    4. Provide a clear and helpful answer.
    5. Use bullet points when appropriate.
    6. When useful, mention the policy document filename.
    7. Do not claim that a policy says something unless
       the retrieved context supports it.

    COMPANY POLICY CONTEXT:

    {context}

    EMPLOYEE QUESTION:

    {question}

    ANSWER:
    """

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1500
        )
    )

    if not response.text:
        return "Gemini returned an empty response. Please try again."

    return response.text.strip()


# ============================================================
# 9. COMPLETE RAG PIPELINE
# ============================================================

def ask_rag(question):

    chunks, sources = retrieve_documents(question)

    answer = ask_gemini(
        question,
        chunks,
        sources
    )

    return {
        "answer": answer,
        "sources": sorted(set(sources)),
        "chunks": list(zip(chunks, sources))
    }


# ============================================================
# 10. CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous conversation

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("sources"):

                st.caption(
                    "Sources: "
                    + ", ".join(message["sources"])
                )

            if message.get("chunks"):

                with st.expander(
                    "View retrieved document chunks"
                ):

                    for index, item in enumerate(
                        message["chunks"], 1
                    ):

                        chunk_text, source_file = item

                        st.markdown(
                            f"**Chunk {index} - {source_file}**"
                        )

                        st.info(chunk_text)


# ============================================================
# 11. USER CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about NovaTech company policies..."
)


if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching policies and asking Gemini..."):

                result = ask_rag(question)

            st.markdown(result["answer"])

            if result["sources"]:

                st.caption(
                    "Sources: "
                    + ", ".join(result["sources"])
                )

            else:

                st.warning("No source documents were retrieved.")

            with st.expander("View retrieved document chunks"):

                if result["chunks"]:

                    for index, (chunk, source) in enumerate(
                        result["chunks"], 1
                    ):

                        st.markdown(
                            f"**Chunk {index} - {source}**"
                        )

                        st.info(chunk)

                else:

                    st.write("No relevant chunks found.")

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "chunks": result["chunks"]
            })

        except Exception as e:

            st.error(f"Error generating response: {str(e)}")

            st.info(
                "Check your Gemini API key, model availability "
                "and ChromaDB configuration."
            )


# ============================================================
# 12. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("NovaTech RAG Information")

    st.metric(
        "Indexed document chunks",
        collection.count()
    )

    st.metric(
        "Retrieved chunks per question",
        TOP_K
    )

    st.write(f"**Gemini model:** {GEMINI_MODEL}")

    st.divider()

    if st.button("Clear Chat History", use_container_width=True):

        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Google Gemini + ChromaDB + Streamlit")
