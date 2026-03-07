import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    GROQ_MODEL = st.secrets.get("GROQ_MODEL") or os.environ.get("GROQ_MODEL", "llama3-8b-8192")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")


faqs_path = "resources/faqs.csv"
chroma_client = chromadb.PersistentClient(path="/tmp/chroma")
groq_client = Groq(api_key=GROQ_API_KEY)
collection_name_faq = 'faqs'

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )

def ingest_faq_data(path: str):
    print("Ingesting FAQ data into Chromadb...")
    collection = chroma_client.get_or_create_collection(
        name=collection_name_faq,
        embedding_function=ef
    )
    df = pandas.read_csv(path)
    docs = df['question'].to_list()
    metadata = [{'answer': ans} for ans in df['answer'].to_list()]
    ids = [f"id_{i}" for i in range(len(docs))]
    # upsert instead of add — safe to call every time, no duplicates
    collection.upsert(
        documents=docs,
        metadatas=metadata,
        ids=ids
    )
    print(f"FAQ Data successfully ingested into Chroma collection: {collection_name_faq}")

def get_relevant_qa(query):
    collection = chroma_client.get_or_create_collection(
        name=collection_name_faq,
        embedding_function=ef
    )
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result

def generate_answer(query, context):
    prompt = f'''Given the following context and question, generate answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.
    
    CONTEXT: {context}
    
    QUESTION: {query}
    '''
    chat_completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
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
    print("Context:", context)
    answer = generate_answer(query, context)
    return answer

     
if __name__ == '__main__':
    ingest_faq_data(faqs_path)
    query = "what's your policy on defective products?"
    query = "Do you take cash as a payment option?"
    answer = faq_chain(query)
    print(answer)