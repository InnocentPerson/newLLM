# app.py
import os
import uuid
import streamlit as st
import torch

# PDF / DOCX reading
from pypdf import PdfReader   # More reliable than PyPDF2
import docx2txt

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama


# -----------------------------------------------------------
# ✅ SETTINGS
# -----------------------------------------------------------

DB_BASE_PATH = "D:/OLLAMA_MAIN/db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"📌 Running on: **{DEVICE.upper()}**")


# -----------------------------------------------------------
# 🔥 HELPER FUNCTIONS
# -----------------------------------------------------------

def load_text_from_file(uploaded_file):
    """Reads text from PDF, TXT, DOCX."""
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    elif file_type == "pdf":
        pdf = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages)

    elif file_type == "docx":
        text = docx2txt.process(uploaded_file)

    else:
        st.error("Unsupported file type!")
        st.stop()

    if len(text.strip()) == 0:
        st.error("❌ No readable text found! This PDF may be scanned or image-based.")
        st.stop()

    return text


def split_into_chunks(text):
    """Splits text into manageable 1000-character chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    chunks = [c for c in chunks if c.strip()]

    if len(chunks) == 0:
        st.error("❌ No chunks could be created. Document may be empty.")
        st.stop()

    return chunks


def create_vector_db(chunks):
    """Creates Chroma DB with safe unique folder (avoids WinError 32)."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # safer and stable
    )

    # 🔍 Test embedding
    st.write("🔍 Testing embeddings…")
    try:
        test_vec = embeddings.embed_documents([chunks[0]])
        st.write(f"✔ Embedding OK — vector size = {len(test_vec[0])}")
    except Exception as e:
        st.error(f"❌ Embedding generation failed: {e}")
        st.stop()

    # 🔥 Create unique session folder to avoid file lock issues
    unique_db_path = os.path.join(DB_BASE_PATH, f"session_{uuid.uuid4()}")
    os.makedirs(unique_db_path, exist_ok=True)

    st.write(f"📁 Using DB folder: {unique_db_path}")

    # Create vector DB
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=unique_db_path
    )

    return db


def build_qa_chain(db):
    """Creates QA chain using Ollama."""
    llm = Ollama(model="mistral", temperature=0.3)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=False
    )
    return qa


# -----------------------------------------------------------
# 🚀 STREAMLIT UI
# -----------------------------------------------------------

st.title("📚 Chat With Your File (Local LLM + ChromaDB)")
st.write("Upload a document and ask anything about it!")

uploaded_file = st.file_uploader("Upload PDF / TXT / DOCX", type=["txt", "pdf", "docx"])

if "history" not in st.session_state:
    st.session_state.history = []


if uploaded_file:

    # STEP 1: READ FILE
    text = load_text_from_file(uploaded_file)

    # STEP 2: SPLIT INTO CHUNKS
    chunks = split_into_chunks(text)
    st.success(f"📌 Document split into **{len(chunks)} chunks**")

    # STEP 3: CREATE VECTOR DB
    with st.spinner("🔧 Creating vector database..."):
        db = create_vector_db(chunks)

    # STEP 4: INIT QA SYSTEM
    with st.spinner("🤖 Initializing AI model..."):
        qa = build_qa_chain(db)

    # STEP 5: CHAT UI
    st.subheader("💬 Ask Questions About Your Document")

    user_q = st.text_input("Your question:")

    if st.button("Submit"):
        if user_q:
            with st.spinner("Thinking..."):
                answer = qa.run(user_q)

            st.session_state.history.append(("🧑 You", user_q))
            st.session_state.history.append(("🤖 AI", answer))

            st.success(answer)

    # STEP 6: DISPLAY CHAT HISTORY
    if st.session_state.history:
        st.subheader("📜 Chat History")
        for sender, msg in st.session_state.history:
            st.markdown(f"**{sender}:** {msg}")

else:
    st.info("📄 Upload a file to begin.")
