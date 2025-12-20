import streamlit as st

from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


MONGO_URI = st.secrets['MONGO_URI']
DB_NAME = "vector_store_database"
COLLECTION_NAME = "embedding_stream"
ATLAS_VECTOR_SEARCH = "vector_index_ghw"
API_KEY = st.secrets['GEMINI_API_KEY']

def get_vector_store():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = MongoDBAtlasVectorSearch(
        collection = collection,
        embedding = embeddings,
        index_name = "vector_index_ghw"
    )

    return vector_store

def ingest_text(text_context):
    vector_store = get_vector_store()
    docs = Document(page_content = text_context)
    vector_store.add_documents([docs])
    
    return True

def get_rag_response(query):
    vector_store = get_vector_store()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs = {'k':3}) 

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer: \n\n{context}"),
        ("human", "{question}")
    ])
                                
    chain =RunnableParallel({"context": retriever, "question": RunnablePassthrough()}).assign(
        answer = prompt | llm | StrOutputParser())
    
    result = chain.invoke(query)

    return {
        "answer": result['answer'],
        "sources": result['context']
    }