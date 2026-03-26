import uuid
import streamlit as st
from chat_service import ask_chatbot
from prompts import DISCLAIMER


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Early Childhood Development Chatbot",
    layout="wide"
)


# -----------------------------
# Helper functions
# -----------------------------
def create_new_chat():
    chat_id = str(uuid.uuid4())
    new_chat = {
        "id": chat_id,
        "title": "New chat",
        "messages": []
    }
    st.session_state.chats.append(new_chat)
    st.session_state.current_chat_id = chat_id


def get_current_chat():
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.current_chat_id:
            return chat
    return None


def auto_title_from_query(query: str) -> str:
    words = query.strip().split()
    if not words:
        return "New chat"

    title = " ".join(words[:6]).strip()
    if len(words) > 6:
        title += "..."
    return title


def render_sources(sources):
    if not sources:
        return

    with st.expander("Sources"):
        for src in sources:
            source_name = src.get("source", "Unknown")
            age = src.get("age", "")
            category = src.get("category", "")
            chunk_index = src.get("chunk_index", "")

            line = f"- {source_name}"
            if age:
                line += f" | age: {age}"
            if category:
                line += f" | category: {category}"
            if chunk_index != "":
                line += f" | chunk: {chunk_index}"

            st.write(line)


def process_query(query: str):
    current_chat = get_current_chat()
    if current_chat is None:
        return

    # Add user message
    current_chat["messages"].append({
        "role": "user",
        "content": query
    })

    # Auto-title only if chat is still unnamed
    if current_chat["title"] == "New chat":
        current_chat["title"] = auto_title_from_query(query)

    # Generate response
    result = ask_chatbot(query)

    current_chat["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "status": result.get("status", ""),
        "safety_label": result.get("safety_label", "")
    })


# -----------------------------
# Session state initialization
# -----------------------------
if "chats" not in st.session_state:
    st.session_state.chats = []
    first_chat_id = str(uuid.uuid4())
    st.session_state.chats.append({
        "id": first_chat_id,
        "title": "New chat",
        "messages": []
    })
    st.session_state.current_chat_id = first_chat_id

if "current_chat_id" not in st.session_state:
    if st.session_state.chats:
        st.session_state.current_chat_id = st.session_state.chats[0]["id"]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot provides general information about early childhood development "
        "based on the project knowledge base."
    )
    st.write(
        "It is for informational purposes only and does not replace professional "
        "medical or developmental assessment."
    )

    st.divider()

    if st.button("➕ Start new chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.subheader("Chats")

    # Show chats in reverse order (newest first)
    for chat in reversed(st.session_state.chats):
        label = chat["title"]

        if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
            st.session_state.current_chat_id = chat["id"]
            st.rerun()

    st.divider()

    current_chat = get_current_chat()
    if current_chat:
        st.subheader("Rename current chat")
        new_title = st.text_input(
            "Chat title",
            value=current_chat["title"],
            key=f"rename_input_{current_chat['id']}"
        )

        if st.button("Save title", use_container_width=True):
            cleaned_title = new_title.strip()
            if cleaned_title:
                current_chat["title"] = cleaned_title
                st.rerun()


# -----------------------------
# Main area
# -----------------------------
st.title("Early Childhood Development Information Chatbot")
st.caption(DISCLAIMER)

st.markdown("### Suggested questions")

col1, col2 = st.columns(2)

with col1:
    if st.button("What are common language milestones at 2 years?", use_container_width=True):
        st.session_state.pending_prompt = "What are common language milestones at 2 years?"

    if st.button("How do children develop social interaction?", use_container_width=True):
        st.session_state.pending_prompt = "How do children develop social interaction?"

with col2:
    if st.button("How do toddlers learn through play?", use_container_width=True):
        st.session_state.pending_prompt = "How do toddlers learn through play?"

    if st.button("What are common emotional development milestones?", use_container_width=True):
        st.session_state.pending_prompt = "What are common emotional development milestones?"


# -----------------------------
# Query handling
# -----------------------------
chat_input_value = st.chat_input("Ask a question about early childhood development...")

query_to_process = None

if st.session_state.pending_prompt:
    query_to_process = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif chat_input_value:
    query_to_process = chat_input_value

if query_to_process:
    process_query(query_to_process)


# -----------------------------
# Render current chat
# -----------------------------
current_chat = get_current_chat()

if current_chat:
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("sources"):
                render_sources(msg["sources"])