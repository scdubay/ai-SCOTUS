from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import requests
from typing import List
import argparse

# Custom Ollama embeddings class
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text", endpoint: str = "http://localhost:11434/api/embeddings"):
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

def load_vector_store(path: str, model_name: str) -> FAISS:
    embeddings = OllamaEmbeddings(model=model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def search_vector_store(query: str, k: int, path: str, model_name: str):
    vectorstore = load_vector_store(path, model_name)
    return vectorstore.similarity_search_with_score(query, k=k)

def print_results(results):
    print(f"\nFound {len(results)} relevant documents:\n")
    for i, (doc, score) in enumerate(results):
        print(f"RESULT {i+1} [Similarity: {score:.4f}]")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content}...")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Query a FAISS vector store")
    parser.add_argument("query", help="The search query")
    parser.add_argument("-k", type=int, default=10, help="Number of results to return (default: 3)")
    parser.add_argument("--path", default="rag_vectorstore_justia", help="Path to the vector store")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama model name (default: nomic-embed-text)")
    
    args = parser.parse_args()
    
    print(f"🔎 Searching for: '{args.query}'")
    results = search_vector_store(args.query, k=args.k, path=args.path, model_name=args.model)
    print_results(results)

if __name__ == "__main__":
    main()
