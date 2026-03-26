import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the knowledge base JSON file
with open("cdev_knowledge_base.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert each chunk into a LangChain Document object
docs = [
    Document(
        page_content=item["text"],
        metadata=item["metadata"]
    )
    for item in data
]

# Load HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings and store them in Chroma
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Done! {len(docs)} chunks indexed.")