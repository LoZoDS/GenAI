from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts import SYSTEM_PROMPT, FALLBACKS
from safety import classify_question, validate_answer

# 1. Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load Chroma vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 4. Load language model
hf_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=160,
    temperature=0.1,
    return_full_text=False
)

if hasattr(hf_pipeline.model, "generation_config"):
    hf_pipeline.model.generation_config.max_length = None

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 5. Build prompt + chain
prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
rag_chain = prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_context(query: str):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    return docs, context

def extract_sources(docs):
    sources = []

    for doc in docs:
        metadata = doc.metadata or {}
        sources.append({
            "source": metadata.get("source", "Unknown"),
            "age": metadata.get("age", ""),
            "category": metadata.get("category", ""),
            "chunk_index": metadata.get("chunk_index", "")
        })

    return sources

def ask_chatbot(query: str) -> dict:
    try:
        safety_label = classify_question(query)

        if safety_label == "urgent":
            return {
                "answer": FALLBACKS["urgent"],
                "sources": [],
                "status": "fallback",
                "safety_label": "urgent"
            }

        if safety_label == "diagnostic":
            return {
                "answer": FALLBACKS["diagnostic"],
                "sources": [],
                "status": "fallback",
                "safety_label": "diagnostic"
            }

        if safety_label == "out_of_scope":
            return {
                "answer": FALLBACKS["out_of_scope"],
                "sources": [],
                "status": "fallback",
                "safety_label": "out_of_scope"
            }

        docs, context = retrieve_context(query)

        if not docs or not context.strip():
            return {
                "answer": FALLBACKS["no_evidence"],
                "sources": [],
                "status": "fallback",
                "safety_label": "no_evidence"
            }

        raw_answer = rag_chain.invoke({
            "context": context,
            "question": query
        })

        final_answer = validate_answer(raw_answer)
        sources = extract_sources(docs)

        return {
            "answer": final_answer,
            "sources": sources,
            "status": "ok",
            "safety_label": "in_scope"
        }

    except Exception:
        return {
            "answer": FALLBACKS["error"],
            "sources": [],
            "status": "error",
            "safety_label": "error"
        }