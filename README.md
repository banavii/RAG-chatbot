# 📚 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and ask questions about its content.

The application retrieves relevant information from the uploaded document using vector similarity search and uses Google's Gemini model to generate a grounded answer.

## 🚀 Features

- 📄 Upload PDF documents
- ✂️ Split documents into smaller chunks
- 🧠 Generate embeddings using Gemini
- 🗄️ Store embeddings in ChromaDB
- 🔍 Retrieve relevant document chunks
- 🤖 Generate answers using Gemini
- 💻 Simple Gradio web interface

## 🛠️ Technologies Used

- Python
- Gradio
- LangChain
- Google Gemini
- Gemini Embeddings
- ChromaDB
- PyPDF

## 🔄 How It Works

```text
PDF
 ↓
PDF Text Extraction
 ↓
Text Chunking
 ↓
Gemini Embeddings
 ↓
ChromaDB
 ↓
User Question
 ↓
Similarity Search
 ↓
Relevant Chunks
 ↓
Gemini
 ↓
Generated Answer

