from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

text = "Machine learning allows computers to learn from data."

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)

vector = result.embeddings[0].values

print("Original text:")
print(text)

print("\nEmbedding:")
print(vector)

print("\nNumber of dimensions:", len(vector))