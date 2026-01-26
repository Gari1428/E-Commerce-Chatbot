import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas as pd
import streamlit as st

faqs_path = Path(__file__).parent / "resources" / "faqs.csv"
collection_name_faq = 'faqs'

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

@st.cache_resource
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    return Groq(api_key=api_key)

def ingest_faq_data(path: str):
    chroma_client = get_chroma_client()
    ef = get_embedding_function()
    
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        st.info("Ingesting FAQ data into Chromadb...")
        collection = chroma_client.create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )
        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(docs))]
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
        st.success(f"FAQ Data successfully ingested!")

def get_relevant_qa(query):
    chroma_client = get_chroma_client()
    ef = get_embedding_function()
    collection = chroma_client.get_collection(
        name=collection_name_faq,
        embedding_function=ef
    )
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result

def generate_answer(query, context):
    groq_client = get_groq_client()
    prompt = f'''Given the following context and question, generate answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.
    
    CONTEXT: {context}
    
    QUESTION: {query}
    '''
    chat_completion = groq_client.chat.completions.create(
        model=os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768'),
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )
    return chat_completion.choices[0].message.content

def faq_chain(query):
    result = get_relevant_qa(query)
    context = "".join([r.get('answer') for r in result['metadatas'][0]])
    answer = generate_answer(query, context)
    return answer
