import os
import csv
import re
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import requests

# === Configuration ===
VECTORSTORE_PATH = "rag_vectorstore_justia7"
CSV_EXPORT = True
CSV_FILENAME = "vectorstore_metadata.csv"

# === Embedding Class (Same as Main Script) ===
class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", endpoint: str = "http://localhost:11434/api/embeddings"):
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

# === Load Vector Store ===
print(f"📥 Loading vector store from: {VECTORSTORE_PATH}")
embeddings = OllamaEmbeddings()
vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
all_docs = list(vectorstore.docstore._dict.values())
docs = [doc for doc in all_docs if re.match(r"https://supreme\.justia\.com/cases/federal/us/\d+/\d+/?$", doc.metadata.get("source", ""))]
print(f"✅ Filtered to {len(docs)} real case documents.")
print(f"✅ Loaded {len(docs)} documents from FAISS.")

# === Basic Metadata Stats ===
print("\n📊 --- Basic Stats ---")
sections = [doc.metadata.get("section", "unknown") for doc in docs]
titles = set(doc.metadata.get("title", "unknown") for doc in docs)
citations = set(doc.metadata.get("citation") for doc in docs if "citation" in doc.metadata)
related_docs = [doc for doc in docs if doc.metadata.get("relation")]

print(f"📁 Total documents: {len(docs)}")
print(f"📘 Unique titles: {len(titles)}")
print(f"📚 Unique citations (cases): {len(citations)}")
print(f"🔗 Related documents: {len(related_docs)}")

# === Section breakdown ===
section_counts = Counter(sections)
print("\n📂 Documents by section:")
for section, count in section_counts.items():
    print(f"  - {section}: {count}")

# === Related Document Types ===
rel_type_counts = Counter(doc.metadata.get("document_type", "unknown") for doc in related_docs)
print("\n🔎 Related document types:")
for doc_type, count in rel_type_counts.items():
    print(f"  - {doc_type}: {count}")

# === Sample Documents ===
print("\n🧪 Sample documents:")
for doc in docs[:3]:
    print("-----")
    print("Title:", doc.metadata.get("title", "N/A"))
    print("Citation:", doc.metadata.get("citation", "N/A"))
    print("Section:", doc.metadata.get("section", "N/A"))
    print("Source:", doc.metadata.get("source", "N/A"))
    print("Preview:", doc.page_content[:300].replace("\n", " "), "...\n")

# === Optional CSV Export ===
if CSV_EXPORT:
    print(f"\n📝 Exporting metadata to: {CSV_FILENAME}")
    with open(CSV_FILENAME, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set(k for doc in docs for k in doc.metadata.keys())))
        writer.writeheader()
        for doc in docs:
            writer.writerow(doc.metadata)
    print("✅ Metadata export complete.")

# === Done ===
print("\n🏁 Vector store validation complete.")
