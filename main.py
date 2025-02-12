import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load Hugging Face API Token
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

if not HF_TOKEN:
    raise RuntimeError("Missing Hugging Face API token. Please set HF_TOKEN in your environment variables.")

# Initialize LLM
def load_llm():
    try:
        return HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            task="text-generation",
            temperature=0.3,  # Lower temperature for factual responses
            model_kwargs={"max_length": 512}  # Removed incorrect token key
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise RuntimeError("Failed to load the AI model.")

# Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    logger.error(f"Error loading FAISS database: {e}")
    db = None  # Allow API to function even if FAISS fails

# Strict Answering Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
You are an AI assistant for Mizoram Police, designed to provide accurate and professional responses.

- **Use ONLY the provided context** to answer.  
- **If the context lacks relevant information, respond with:**  
  "I'm sorry, but I don't have enough information to answer that. Please contact Mizoram Police for further assistance."
- **DO NOT generate numbers, make assumptions, or fabricate information.**  
- **Ensure responses are clear, concise, and well-structured.**  

Context: {context}  
Question: {question}  

Answer:
"""

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Setup FAISS Retriever (if database loaded)
retriever = db.as_retriever(search_kwargs={'k': 3}) if db else None

# Create Retrieval QA Chain
qa_chain = None
if retriever:
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Request Model
class QueryRequest(BaseModel):
    question: str

# API Endpoint
@app.post("/query")
def query(data: QueryRequest):
    if not qa_chain:
        raise HTTPException(status_code=500, detail="The AI system is temporarily unavailable. Please try again later.")

    try:
        response = qa_chain.invoke({'query': data.question})
        retrieved_docs = response.get("source_documents", [])
        answer = response.get("result", "").strip()

        # Prevent invalid or nonsensical responses
        if not retrieved_docs or not answer or len(answer) > 500 or answer.isdigit():
            return {"answer": "I'm sorry, but I don't have enough information to answer that. Please contact Mizoram Police for further assistance."}

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
