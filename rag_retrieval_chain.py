from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize the Chroma vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  
)

# Define a natural language query
query = "How often should a newborn be fed?"

# Retrieve the most relevant documents based on the query
docs = retriever.invoke(query)

# print title what is following being to print
print("Documents related to the query:")

for doc in docs:
    print(doc.page_content)


print()
print()
print("Step 1: importing libraries Done")

print("Step 2: creating Hugging Face pipeline...")
hf_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=128,
    temperature=0.1
)

# Remove inherited/default max_length if present
if hasattr(hf_pipeline.model, "generation_config"):
    hf_pipeline.model.generation_config.max_length = None

print("Step 2 done.")


print("Step 3: wrapping pipeline for LangChain...")
llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("Step 3 done.")


print("Step 4: creating prompt template...")
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant specialized in early childhood development.

RULES:
- Answer only using the provided context.
- Do not use outside knowledge.
- If the answer is not in the context, say exactly:
  "I do not know based on the provided information."
- Keep answers clear, short, and user-friendly.
- Do not diagnose.
- Do not provide medical advice.
- Do not recommend treatment, medication, or emergency action as a professional instruction.
- Do not claim certainty about a child's condition.
- If the question is medical or safety-related, provide only general informational guidance from the context and remind the user to consult a healthcare professional.

DISCLAIMER:
This chatbot provides general information only and is not a doctor or medical professional.
For medical concerns, diagnosis, or treatment, consult a qualified healthcare provider.

Context:
{context}

Question:
{question}
""")
print("Step 4 done.")


print("Step 5: defining document formatter...")
def format_docs(docs):
    print(f"format_docs: received {len(docs)} docs")
    return "\n\n".join(doc.page_content for doc in docs)
print("Step 5 done.")


print("Step 6: building RAG chain...")
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
print("Step 6 done.")


print("Step 7: preparing query...")
query = "How often should a newborn be fed?"
print("Query:", query)
print("Step 7 done.")


print("Step 8: invoking RAG chain...")
answer = rag_chain.invoke(query)
print("Step 8 done.")


print("Step 9: final answer:")
print(answer)

