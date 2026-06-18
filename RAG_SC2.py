from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import os
import time
from typing import List, Set

# Set user agent
os.environ["USER_AGENT"] = "Mozilla/5.0"
HEADERS = {"User-Agent": os.environ["USER_AGENT"]}

# Ollama Embeddings
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

# Basic recursive crawler to collect page content
# Basic recursive crawler to collect page content and source URLs
def crawl_site(base_url: str, max_pages: int = 500, delay: float = 1.0) -> tuple[List[str], List[str]]:
    visited: Set[str] = set()
    to_visit: List[str] = [base_url]
    texts: List[str] = []
    sources: List[str] = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue
        try:
            print(f"🔍 Visiting: {url}")
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            visited.add(url)

            soup = BeautifulSoup(res.text, "html.parser")
            page_text = soup.get_text(separator="\n", strip=True)
            texts.append(page_text)
            sources.append(url)

            # Discover and queue additional links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)

                if (
                    "https://supreme.justia.com/cases/federal/" in full_url or
                    full_url.startswith(base_url)
                ) and full_url.startswith("https://supreme.justia.com") and full_url not in visited:
                    print(f"  ➕ Queued: {full_url}")
                    to_visit.append(full_url)

            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ Skipping {url}: {e}")

    print(f"\n✅ Finished crawling {len(texts)} pages.")
    return texts, sources


# Step 1: Crawl site for content
base_url = "https://supreme.justia.com/cases-by-topic/"
pages_text = crawl_site(base_url, max_pages=100)

# Step 2: Convert to LangChain documents
from langchain_core.documents import Document
documents = [Document(page_content=page) for page in pages_text]

# Step 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Step 4: Embed with Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Step 5: Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("rag_vectorstore_justia")

print("✅ Vector store created from recursively crawled pages.")
