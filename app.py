import os
import tempfile
import time
import nltk
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import google.generativeai as genai

# ------------------------------------------------------------
# 🧠 NLTK Setup
# ------------------------------------------------------------
for pkg in ["punkt", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(pkg if "/" in pkg else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ------------------------------------------------------------
# 🔑 Load Environment Variables
# ------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in environment!")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment!")

genai.configure(api_key=GOOGLE_API_KEY)

# ------------------------------------------------------------
# 🤖 Custom Gemini Embeddings Wrapper (Fixed + Optimized)
# ------------------------------------------------------------
class GeminiEmbeddings:
    def __init__(self, model_name="models/embedding-001"):
        self.model_name = model_name

    def embed_documents(self, texts):
        """Embed multiple documents into vectors"""
        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(model=self.model_name, content=text.page_content)
                embeddings.append(result["embedding"])
                time.sleep(0.2)  # avoid rate limit / memory spikes
            except Exception as e:
                print("⚠️ Embedding failed for a chunk:", e)
                embeddings.append([0.0] * 768)  # fallback vector
        return embeddings

    def embed_query(self, text):
        """Embed a single query string"""
        try:
            result = genai.embed_content(model=self.model_name, content=text)
            return result["embedding"]
        except Exception as e:
            print("⚠️ Query embedding failed:", e)
            return [0.0] * 768

    def __call__(self, text):
        """Allow FAISS/LangChain to call this object directly."""
        return self.embed_query(text)

# ------------------------------------------------------------
# ⚙️ Global Configuration
# ------------------------------------------------------------
VECTORSTORE_PATH = "./faiss_index"
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

vectorstore = None
qa_chain = None

# ------------------------------------------------------------
# 🧩 Flask Setup
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# 💬 Load ChatGroq Model
# ------------------------------------------------------------
def make_chatgroq_model():
    return ChatGroq(
        model="groq/compound-mini",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

# ------------------------------------------------------------
# 📄 Process Uploaded PDF → Create FAISS Vectorstore
# ------------------------------------------------------------
def process_document(file):
    global vectorstore, qa_chain

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        print(f"📄 Processing document: {file.filename}")
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No text extracted from the PDF")

        # Reduced chunk size for memory optimization
        text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        print(f"🔍 Split into {len(texts)} chunks.")

        embeddings = GeminiEmbeddings()
        print("⚙️ Generating embeddings...")

        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)

        llm = make_chatgroq_model()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )

        print("✅ Document processed and vectorstore created successfully.")
        os.unlink(tmp_path)
        return True

    except Exception as e:
        print("❌ Error processing document:", e)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False

# ------------------------------------------------------------
# ❓ Query the Vectorstore
# ------------------------------------------------------------
def query_document(question: str):
    global qa_chain, vectorstore
    print("❓ Received question:", question)

    if qa_chain is None:
        try:
            print("♻️ Reloading FAISS vectorstore...")
            embeddings = GeminiEmbeddings()
            vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            llm = make_chatgroq_model()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True,
            )
        except Exception as e:
            print("⚠️ Reload failed:", e)
            return "Please upload a document first.", None

    try:
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "No answer found")
        src_docs = result.get("source_documents", [])
        source = src_docs[0].metadata.get("source", "Unknown") if src_docs else "N/A"
        print("💡 Answer generated successfully.")
        return answer, source

    except Exception as e:
        print("❌ Error during query:", e)
        return f"Error: {e}", None

# ------------------------------------------------------------
# 🌐 Flask Routes
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Vishal's Multimodal AI Flask API with Gemini + Groq!"})

@app.route("/upload_doc", methods=["POST"])
def upload_doc():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file."}), 400

    success = process_document(file)
    if success:
        return jsonify({"message": "✅ PDF processed successfully! You can now ask questions."})
    else:
        return jsonify({"error": "Failed to process document."}), 500

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400

    answer, source = query_document(question)
    return jsonify({"answer": answer, "source": source or "No source found"})

# ------------------------------------------------------------
# 🚀 Run Flask App
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
