from fastapi import FastAPI 
from pydantic import BaseModel
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# === Constants ===
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2"

# === Custom Ollama embedding class ===
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = EMBED_MODEL, endpoint: str = "http://localhost:11434/api/embeddings"):
        self.model = model
        self.endpoint = endpoint

    def embed_query(self, text):
        return self._embed(text)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def _embed(self, text):
        response = requests.post(self.endpoint, json={"model": self.model, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]

# === Load your vector store ===
embeddings = OllamaEmbeddings()
vectorstore = FAISS.load_local("rag_vectorstore_justia6", embeddings, allow_dangerous_deserialization=True)

# === FastAPI app ===
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/rag-query")
def rag_query(req: PromptRequest):
    query = req.prompt

    # Step 1: Retrieve top-k relevant docs
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 2: Construct final prompt for Ollama
    full_prompt = f"""You are a legal analyst assistant. Use the following legal case text to answer the user's question.

Context:
{context}

Question: {query}

Answer:"""

    # Step 3: Send to Ollama
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": GEN_MODEL,
            "prompt": full_prompt,
            "stream": False
        })
        response.raise_for_status()
        return {
            "answer": response.json()["response"],
            "sources": [doc.metadata.get("source", "unknown") for doc in docs]
        }
    except requests.RequestException as e:
        return {"error": f"Failed to generate response: {str(e)}"}

# Optional: health check
@app.get("/health")
def health():
    return {"status": "ok"}
