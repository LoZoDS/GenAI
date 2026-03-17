import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

with open("cdev_knowledge_base.json", "r") as f:
    data = json.load(f)

docs = [
    Document(
        page_content=item["text"],
        metadata=item["metadata"]
    )
    for item in data
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Done! {len(docs)} chunks indexed.")