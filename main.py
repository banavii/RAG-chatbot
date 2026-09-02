from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = Chroma(
    collection_name="unit1_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

question = input("Ask something: ")

results = vectorstore.similarity_search_with_score(
    question,
    k=3
)

# Only use results that are sufficiently relevant
relevant_results = [
    result for result, score in results
    if score < 1.0
]

if not relevant_results:
    print("\nRAG Answer:")
    print("I couldn't find this information in the provided documents.")
    exit()



context = "\n\n".join(
    result.page_content for result in relevant_results
)

prompt = f"""
You are a study assistant.

Answer the question ONLY using the information in the provided context.

If the answer cannot be found in the context, say:
"I couldn't find this information in the provided documents."

Do not use your general knowledge to answer.

Context:
{context}

Question:
{question}
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
)

print("\nRAG Answer:")
print(response.text)

