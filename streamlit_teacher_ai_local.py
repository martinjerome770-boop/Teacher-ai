import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from gtts import gTTS
import tempfile

st.title("📚 AI Teacher - Voice Enabled")

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# gTTS speak function
def speak(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# File Upload Section
uploaded_file = st.file_uploader("Upload your notes (PDF)", type=["pdf"])

if uploaded_file is not None:
    pdf = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    embeddings = model.encode([text])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success("✅ Notes uploaded and processed!")

    question = st.text_input("Ask something from your notes:")

    if question:
        # Generate answer
        q_emb = model.encode([question])
        D, I = index.search(q_emb, 1)
        answer = text

        # Convert to exam-focused explanation
        explanation = f"""
🎓 *Exam-Ready Explanation*  
Here is the important, scoring answer:

{answer[:800]}...
"""

        st.write(explanation)

        # Voice Output
        speak(explanation)
        
