import streamlit as st
import os
from faq import ingest_faq_data, faq_chain, faqs_path
from sql import sql_chain
from router import build_router

api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
model = st.secrets.get("GROQ_MODEL") or os.environ.get("GROQ_MODEL", "llama3-8b-8192")

@st.cache_resource
def initialize():
    ingest_faq_data(faqs_path)
    # build_router() explicitly adds routes so the index is marked ready
    return build_router()

router = initialize()

def ask(query):
    result = router(query)
    route = result.name
    print(f"Query: {query} → Route: {route}")
    if route == 'faq':
        return faq_chain(query)
    elif route == 'sql':
        return sql_chain(query)
    else:
        return faq_chain(query)  # safe fallback

st.title("TrovaAI")

query = st.chat_input("Write your query")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = ask(query)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})