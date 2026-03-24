from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
    search_type="similarity",
    search_kwargs={"k": 5}
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
    # model="Qwen/Qwen2.5-0.5B-Instruct",
    # model="microsoft/Phi-3-mini-4k-instruct",
    model="Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens=200,
    temperature=0.2,
    device="mps",
    return_full_text=False,
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
- Answer using only the provided context.
- Do not use outside knowledge.
- If the answer is directly stated, answer it.
- If the context contains related information, provide a general answer based on it.
- If there is no useful information, say exactly:
  "I do not know based on the provided information."
- Keep the answer to a maximum of 4 sentences.
- Do not add extra sentences or continue the conversation. 
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
    prompt
    | llm
    | StrOutputParser()
)
print("Step 6 done.")


def is_medical_question(query: str) -> bool:
    medical_keywords = [
        "diagnose", "diagnosis", "treatment", "medicine", "medication",
        "dose", "fever", "rash", "vomiting", "infection", "emergency",
        "serious", "urgent", "hospital", "dehydration", "sick", "ill"
    ]
    query_lower = query.lower()
    return any(word in query_lower for word in medical_keywords)


def add_disclaimer(answer: str, query: str) -> str:
    disclaimer = (
        "\n\nNote: This chatbot provides general information only and is not a medical professional. "
        "For diagnosis, treatment, or urgent concerns, consult a qualified healthcare provider."
    )

    if is_medical_question(query):
        return answer + disclaimer

    return answer


def enforce_safety(answer: str) -> str:
    if not answer:
        return answer

    unsafe_phrases = [
        # "your child has",
        # "your baby has",
        # "this means your child has",
        # "this means your baby has",
        "the diagnosis is",
        "i diagnose",
        "you should give medication",
        "administer",
        "prescribe",
        "definitely has",
        "certainly has"
    ]

    answer_lower = answer.lower()

    for phrase in unsafe_phrases:
        if phrase in answer_lower:
            return (
                "I can only provide general information from the available context. "
                "I cannot diagnose or provide medical advice. "
                "Please consult a qualified healthcare professional."
            )

    return answer


def fallback_response() -> str:
    return (
        "I could not find enough relevant information in the knowledge base to answer that safely. "
        "Please rephrase your question or consult a qualified healthcare professional."
    )


def postprocess_answer(answer: str, query: str) -> str:
    if answer is None:
        print("postprocess_answer: answer is None")
        return fallback_response()

    answer = answer.strip()

    if answer == "":
        print("postprocess_answer: answer is empty string")
        return fallback_response()

    answer = enforce_safety(answer)
    answer = add_disclaimer(answer, query)

    return answer

def retrieve_context(query: str, retriever):
    docs = retriever.invoke(query)
    return docs, format_docs(docs)

def clean_answer(raw_answer: str) -> str:
    if not raw_answer:
        return ""

    text = raw_answer.strip()

    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    stop_phrases = [
        "If you have any other questions",
        "If you need further assistance",
        "feel free to ask",
        "I'm here to help",
        "I’m here to help",
        "Please let me know",
        "Stop",
        "RULES:",
    ]
    for phrase in stop_phrases:
        if phrase in text:
            text = text.split(phrase)[0].strip()

    fallback = "I do not know based on the provided information."
    if fallback in text and text != fallback:
        parts = text.split(fallback, 1)
        if len(parts) > 1 and len(parts[1].strip()) > 20:
            text = parts[1].strip()
        else:
            text = fallback

    sentences = text.split(". ")
    text = ". ".join(sentences[:4]).strip()

    if text.endswith("I do not know based on the"):
        text = text.split("I do not know")[0].strip()

    if text and not text.endswith(".") and "." in text:
        text = text[:text.rfind(".") + 1]

    return text

def ask_chatbot(query: str, retriever, rag_chain) -> str:
    print("Retrieving documents...")
    docs, context = retrieve_context(query, retriever)
    print(f"Retrieved {len(docs)} documents.")

    if len(docs) == 0:
        return fallback_response()

    print("Generating answer...")
    raw_answer = rag_chain.invoke({
        "context": context,
        "question": query
    })

    cleaned = clean_answer(raw_answer)
    final_answer = postprocess_answer(cleaned, query)

    sentences = final_answer.split(". ")
    final_answer = ". ".join(sentences[:4]).strip()

    if final_answer and not final_answer.endswith(".") and "." in final_answer:
        final_answer = final_answer[:final_answer.rfind(".") + 1]

    return final_answer


# -----------------------------
# Simple chat interface
# -----------------------------
def chat_loop(retriever, rag_chain):
    print("Early Childhood Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() == "exit":
            print("Bot: Goodbye.")
            break

        if query == "":
            print("Bot: Please enter a question.")
            continue

        answer = ask_chatbot(query, retriever, rag_chain)
        print("Bot:", answer)
        print()

# Run
if __name__ == "__main__":
    chat_loop(retriever, rag_chain)
