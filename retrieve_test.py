from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = Chroma(
    collection_name="unit1_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

question = input("Ask something: ")

results = vectorstore.similarity_search(
    question,
    k=2
)

print("\nRelevant chunks:\n")

for i, result in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(result.page_content)
    print()