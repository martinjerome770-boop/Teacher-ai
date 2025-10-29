
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import pyttsx3

st.title("📚 AI Teacher")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

uploaded_file = st.file_uploader("Upload your notes (PDF)", type=["pdf"])

if uploaded_file is not None:
    pdf = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

    embeddings = model.encode([text])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success("✅ Notes uploaded and processed!")

    question = st.text_input("Ask something from your notes:")

    if question:
        q_emb = model.encode([question])
        D, I = index.search(q_emb, 1)
        answer = text

        st.write("🧑‍🏫 Answer:", answer[:500] + "...")

        engine = pyttsx3.init()
        engine.say(answer[:200])
        engine.runAndWait()
