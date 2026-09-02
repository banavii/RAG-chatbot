from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

question = input("Ask something: ")

response = llm.invoke(question)

print("\nGemini:", response.content)