import chromadb

client = chromadb.Client()

collection = client.create_collection(name="test_collection")

collection.add(
    documents=[
        "Machine learning allows computers to learn from data.",
        "Python is a popular programming language.",
        "Chroma is a vector database used for storing embeddings."
    ],
    ids=["ml", "python", "chroma"]
)

results = collection.query(
    query_texts=["How do computers learn?"],
    n_results=1
)

print(results["documents"])