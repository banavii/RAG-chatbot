from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Find all PDFs
pdf_files = [
    file for file in os.listdir("documents")
    if file.lower().endswith(".pdf")
]

print("PDFs found:", pdf_files)

# 2. Load all PDFs
documents = []

for pdf in pdf_files:
    loader = PyPDFLoader(os.path.join("documents", pdf))
    documents.extend(loader.load())

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# 4. Create Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 5. Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="unit1_documents",
    persist_directory="./chroma_db"
)

print("Documents stored in Chroma!")