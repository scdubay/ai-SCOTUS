from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import requests
from typing import List
import os

# Optional but recommended
os.environ["USER_AGENT"] = "Mozilla/5.0"

# Custom Ollama embeddings class
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "llama3:2", endpoint: str = "http://localhost:11434/api/embeddings"):
        self.model = model
        self.endpoint = endpoint

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> List[float]:
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

# --- Step 1: Load webpage content ---
url = "https://supreme.justia.com/cases-by-topic/abortion-reproductive-rights/"
loader = WebBaseLoader(url)
documents = loader.load()

# --- Step 2: Split into manageable chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# --- Step 3: Generate embeddings via Ollama ---
embeddings = OllamaEmbeddings(model="llama3.2")

# --- Step 4: Create and persist FAISS vector store ---
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("rag_vectorstore_ollama")

print("✅ Vector store created and saved locally as 'rag_vectorstore_ollama'")
