import os
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Set, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# === Setup ===
os.environ["USER_AGENT"] = "Mozilla/5.0"
HEADERS = {"User-Agent": os.environ["USER_AGENT"]}

# === Ollama Embeddings ===
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text", endpoint: str = "http://localhost:11434/api/embeddings"):
        self.model = model
        self.endpoint = endpoint

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, texts: List[str]):
        return [self._embed(t) for t in texts]

    def _embed(self, text: str):
        response = requests.post(self.endpoint, json={"model": self.model, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]

# === Metadata extractor ===
def extract_case_metadata(html: str, url: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    citation_match = re.search(r"\b\d{1,3}\sU\.S\.\s\d{1,4}\b", text)
    citation = citation_match.group(0) if citation_match else "Unknown"

    docket_match = re.search(r"(Docket|No\.):\s*([\w\-]+)", text)
    docket = docket_match.group(2) if docket_match else "Unknown"

    date_match = re.search(r"Decided:\s*([A-Za-z]+\s\d{1,2},\s\d{4})", text)
    date_decided = date_match.group(1) if date_match else "Unknown"

    author_match = re.search(r"(Opinion (delivered by|by)|Justice\s[A-Z][a-z]+)", text)
    opinion_author = author_match.group(0) if author_match else "Unknown"

    dissent_matches = re.findall(r"(Dissenting|Dissent)\s(opinion\s)?(by\sJustice\s)?([A-Z][a-z]+)", text)
    dissent_authors = list(set(match[3] for match in dissent_matches)) if dissent_matches else []

    concur_matches = re.findall(r"(Concurring|Concurrence)\s(opinion\s)?(by\sJustice\s)?([A-Z][a-z]+)", text)
    concurrence_authors = list(set(match[3] for match in concur_matches)) if concur_matches else []

    return Document(
        page_content=text,
        metadata={
            "title": title,
            "citation": citation,
            "docket": docket,
            "date_decided": date_decided,
            "opinion_author": opinion_author,
            "dissent_authors": dissent_authors,
            "concurrence_authors": concurrence_authors,
            "source": url
        }
    )

# === Crawl and collect cases ===
def crawl_site(base_url: str, max_pages: int = 500, delay: float = 1.0) -> List[Document]:
    visited: Set[str] = set()
    to_visit: List[str] = [base_url]
    documents: List[Document] = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue
        try:
            print(f"🔍 Visiting: {url}")
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            visited.add(url)

            # Case page — extract content and metadata
            if "https://supreme.justia.com/cases/federal/" in url:
                doc = extract_case_metadata(res.text, url)
                documents.append(doc)

            # Discover and queue more links
            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"])
                if (
                    full_url.startswith(base_url) or
                    full_url.startswith("https://supreme.justia.com/cases/federal/")
                ) and full_url not in visited:
                    to_visit.append(full_url)

            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ Error with {url}: {e}")

    print(f"✅ Finished crawling {len(documents)} case pages.")
    return documents

# === Main process ===
def main():
    base_url = "https://supreme.justia.com/cases-by-topic/"
    print(f"🚀 Starting crawl from: {base_url}")
    documents = crawl_site(base_url, max_pages=500)

    print("🔪 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print("🔐 Creating vector store with Ollama embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("rag_vectorstore_justia_new")

    print("✅ Vector store saved as 'rag_vectorstore_justia_new'")

if __name__ == "__main__":
    main()
