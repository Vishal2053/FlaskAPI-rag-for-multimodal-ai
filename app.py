import os
import tempfile
import nltk
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ------------------------------------------------------------
# NLTK Setup (quietly downloads required resources)
# ------------------------------------------------------------
for pkg in ["punkt", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(pkg if "/" in pkg else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in environment!")

# ------------------------------------------------------------
# Global settings
# ------------------------------------------------------------
VECTORSTORE_PATH = "./faiss_index"
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

vectorstore = None
qa_chain = None

# ------------------------------------------------------------
# Flask setup
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# Helper: Load LLM
# ------------------------------------------------------------
def make_chatgroq_model():
    return ChatGroq(
        model="groq/compound-mini",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

# ------------------------------------------------------------
# Process uploaded PDF -> FAISS Vectorstore
# ------------------------------------------------------------
def process_document(file):
    global vectorstore, qa_chain

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        print("📄 Processing document:", file.filename)
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No text extracted from the PDF")

        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print(f"🔍 Split into {len(texts)} chunks.")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("⚙️ Generating embeddings...")

        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)

        llm = make_chatgroq_model()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )

        print("✅ Document processed and vectorstore created.")
        os.unlink(tmp_path)
        return True

    except Exception as e:
        print("❌ Error processing document:", e)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False

# ------------------------------------------------------------
# Query the Vectorstore
# ------------------------------------------------------------
def query_document(question: str):
    global qa_chain, vectorstore
    print("❓ Question received:", question)

    if qa_chain is None:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
        print("💡 Answer generated:", result)
        answer = result.get("result", "No answer found")
        src_docs = result.get("source_documents", [])
        source = src_docs[0].metadata.get("source", "Unknown") if src_docs else "N/A"
        return answer, source
    except Exception as e:
        print("❌ Error during query:", e)
        return f"Error: {e}", None

# ------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Vishal's PDF Chat Flask API! Upload a document and ask questions."})

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
# Run Flask
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
 