from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("UNIT1 TEST.pdf")

documents = loader.load()

print("Number of pages:", len(documents))

print("\nFirst page:")
print(documents[0].page_content[:1000])