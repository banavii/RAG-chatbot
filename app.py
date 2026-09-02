import os
import gradio as gr

from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# --------------------------------------------------
# Global vector store
# --------------------------------------------------

vectorstore = None


# --------------------------------------------------
# Process uploaded PDF
# --------------------------------------------------

def upload_pdf(pdf_file):

    global vectorstore

    if pdf_file is None:
        return "❌ Please upload a PDF first."

    try:

        # Get uploaded PDF path
        pdf_path = pdf_file

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split PDF into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        # Create Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )

        # Store chunks in Chroma
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="rag_chatbot_documents"
        )

        return (
            f"✅ PDF processed successfully!\n\n"
            f"📄 Pages: {len(documents)}\n"
            f"🧩 Chunks created: {len(chunks)}"
        )

    except Exception as e:

        return f"❌ Error processing PDF:\n{str(e)}"


# --------------------------------------------------
# Ask question
# --------------------------------------------------

def chat(question):

    global vectorstore

    if vectorstore is None:
        return "❌ Please upload and process a PDF first."

    if not question.strip():
        return "❌ Please enter a question."

    try:

        # Retrieve relevant chunks
        results = vectorstore.similarity_search(
            question,
            k=3
        )

        # Combine retrieved chunks
        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        # Ask Gemini using retrieved context
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"""
You are a helpful RAG chatbot.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the uploaded PDF."

Do not make up information.

Context:
{context}

Question:
{question}
"""
        )

        # Get unique source pages
        pages = sorted(
            set(
                doc.metadata.get("page", 0) + 1
                for doc in results
            )
        )

        sources = "\n".join(
            [f"- 📄 Page {page}" for page in pages]
        )

        # Final answer
        return f"""
### 🤖 RAG Answer

{response.text}

### 📚 Sources

{sources}
"""

    except Exception as e:

        return f"❌ Error:\n{str(e)}"


# --------------------------------------------------
# Gradio Interface
# --------------------------------------------------

with gr.Blocks(
    title="RAG Chatbot"
) as app:

    gr.Markdown(
        """
# 📚 RAG Chatbot

Upload a PDF and ask questions about its content.

The chatbot uses **Gemini Embeddings + ChromaDB + Gemini** 
to retrieve relevant information and generate answers.
"""
    )

    # -----------------------------
    # PDF Upload Section
    # -----------------------------

    pdf = gr.File(
        label="📄 Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    upload_button = gr.Button(
        "📥 Process PDF"
    )

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    upload_button.click(
        upload_pdf,
        inputs=pdf,
        outputs=status
    )

    # -----------------------------
    # Question Section
    # -----------------------------

    gr.Markdown("## 💬 Ask a Question")

    question = gr.Textbox(
        label="Ask a question",
        placeholder="What is the Waterfall model?"
    )

    ask_button = gr.Button(
        "🤖 Ask"
    )

    answer = gr.Markdown(
        label="RAG Answer"
    )

    ask_button.click(
        chat,
        inputs=question,
        outputs=answer
    )


# --------------------------------------------------
# Launch application
# --------------------------------------------------

app.launch()