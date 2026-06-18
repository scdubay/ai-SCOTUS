import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_PATH = "rag_vectorstore_courtlistener_demo"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2"


class OllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model: str = EMBED_MODEL,
        endpoint: str = "http://localhost:11434/api/embed",
    ):
        self.model = model
        self.endpoint = endpoint

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, texts: List[str]):
        return [self._embed(t) for t in texts]

    def _embed(self, text: str):
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if "embeddings" in data and data["embeddings"]:
            return data["embeddings"][0]

        raise RuntimeError(f"No embedding returned: {data.keys()}")


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
    You are a legal research assistant analyzing U.S. Supreme Court materials.

    Answer the question using only the provided context.

    Rules:
    - Identify the case or cases used.
    - Distinguish majority reasoning from dissenting reasoning if the context suggests a dissent.
    - Do not rely on general legal knowledge unless it is supported by the context.
    - If the retrieved context is incomplete, say what appears incomplete.
    - Cite source numbers like [Source 1], [Source 2].

    Question:
    {question}

    Context:
    {context}

    Answer:
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def format_context(docs):
    parts = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata

        parts.append(
            f"""
SOURCE {i}
Case: {meta.get("case_title")}
Citation: {meta.get("citation")}
Document Type: {meta.get("document_type") or meta.get("section")}
Chunk: {meta.get("chunk_id")} of {meta.get("total_chunks")}
Source URL: {meta.get("source")}

Text:
{doc.page_content}
"""
        )

    return "\n\n".join(parts)

def rerank_docs(question: str, docs):
    question_terms = [t.lower() for t in question.split() if len(t) > 3]

    scored = []

    for doc in docs:
        text = doc.page_content.lower()
        meta = doc.metadata

        score = 0

        # Basic lexical relevance
        for term in question_terms:
            if term in text:
                score += 1

        # Prefer substantive chunks over headers / captions
        if len(text) > 800:
            score += 3
        if len(text) < 300:
            score -= 5

        # Prefer majority / court opinion when available
        opinion_role = (
            meta.get("opinion_role")
            or meta.get("section")
            or meta.get("document_type")
            or ""
        ).lower()

        if "majority" in opinion_role or "court" in opinion_role:
            score += 5
        if "syllabus" in opinion_role:
            score += 2
        if "dissent" in opinion_role:
            score -= 3
        if "concurrence" in opinion_role or "concurring" in opinion_role:
            score -= 1

        # Penalize repeated caption/header chunks
        caption_terms = [
            "supreme court of the united states",
            "on writ of certiorari",
            "argued",
            "decided",
        ]
        caption_hits = sum(1 for phrase in caption_terms if phrase in text[:500])
        if caption_hits >= 3:
            score -= 5

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored]

def main():
    print("🔎 Loading vector store...")

    embeddings = OllamaEmbeddings()

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("✅ Vector store loaded.")
    print("Ask a question about the indexed Supreme Court cases.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question> ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        if not question:
            continue

        print("\n🔍 Retrieving relevant chunks...")
        raw_docs = vectorstore.similarity_search(question, k=20)
        docs = rerank_docs(question, raw_docs)[:8]
        context = format_context(docs)

        print("🤖 Generating answer...\n")
        answer = generate_answer(question, context)

        print("ANSWER")
        print("=" * 80)
        print(answer)

        print("\nSOURCES")
        print("=" * 80)

        for i, doc in enumerate(docs, start=1):
            
            meta = doc.metadata
            print(f"[{i}] {meta.get('case_title')} | {meta.get('citation')}")
            print(f"    Chunk: {meta.get('chunk_id')} of {meta.get('total_chunks')}")
            print(f"    URL: {meta.get('source')}")

            preview = doc.page_content[:300].replace("\n", " ")
            print(f"    Preview: {preview}...")

            print()


if __name__ == "__main__":
    main()