import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from google import genai
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# Gemini
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Store uploaded documents
vectorstore = None


def upload_pdf(pdf):
    global vectorstore

    if pdf is None:
        return "Please upload a PDF."

    # Load PDF
    loader = PyPDFLoader(pdf)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # Create Chroma database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="uploaded_documents"
    )

    return f"✅ PDF uploaded! Created {len(chunks)} chunks."


def chat(question):
    if vectorstore is None:
        return "Please upload a PDF first."

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        result.page_content for result in results
    )

    prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I couldn't find this information in the provided document."
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# Gradio UI
with gr.Blocks(title="RAG Chatbot") as app:

    gr.Markdown("# 📚 RAG Chatbot")
    gr.Markdown("Upload a PDF and ask questions about it.")

    pdf = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    upload_button = gr.Button("📥 Process PDF")
    status = gr.Textbox(label="Status")

    upload_button.click(
        upload_pdf,
        inputs=pdf,
        outputs=status
    )

    question = gr.Textbox(
        label="Ask a question",
        placeholder="What is the Waterfall model?"
    )

    ask_button = gr.Button("🤖 Ask")

    answer = gr.Markdown(label="RAG Answer")

    ask_button.click(
        chat,
        inputs=question,
        outputs=answer
    )


app.launch()