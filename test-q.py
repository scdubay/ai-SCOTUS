import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from query_demo_clean import (
    VECTORSTORE_PATH,
    EMBED_MODEL,
    OLLAMA_EMBED_ENDPOINT,
    OLLAMA_GENERATE_ENDPOINT,
    rerank_docs,
    format_context,
)


MODELS_TO_TEST = [
    "llama3.2",
    "llama3.1:8b",
    "qwen3:8b",
]

TESTS = [
    {
        "case": "Meyer v. Nebraska",
        "question": "What liberty interest did Meyer v. Nebraska recognize under the Fourteenth Amendment?",
        "must_include": [
            "Fourteenth Amendment",
            "liberty",
            "teach",
            "parents",
            "education",
        ],
    },
    {
        "case": "United States v. James Daniel Good Real Property",
        "question": "What balancing test did the Court apply in United States v. James Daniel Good Real Property?",
        "must_include": [
            "Mathews v. Eldridge",
            "private interest",
            "risk of erroneous deprivation",
            "Government's interest",
        ],
    },
]

OUTPUT_DIR = Path("data/evals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = EMBED_MODEL, endpoint: str = OLLAMA_EMBED_ENDPOINT):
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

        if "embedding" in data:
            return data["embedding"]

        raise RuntimeError(f"No embedding returned: {data.keys()}")


def generate_answer(model: str, question: str, context: str) -> str:
    prompt = f"""
You are a legal research assistant analyzing U.S. Supreme Court materials.

Answer the question using only the provided context.

Rules:
- Identify the case or cases used.
- Distinguish majority/court reasoning from dissenting or concurring reasoning when the source context supports that distinction.
- Do not rely on general legal knowledge unless it is supported by the provided context.
- If the retrieved context is incomplete or mixed, say so.
- If the context does not clearly state the answer, say the context is insufficient.
- Cite source numbers like [Source 1], [Source 2].
- Keep the answer concise but legally precise.

Question:
{question}

Context:
{context}

Answer:
""".strip()

    response = requests.post(
        OLLAMA_GENERATE_ENDPOINT,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json().get("response", "").strip()


def score_answer(answer: str, must_include: List[str]) -> Tuple[int, List[str]]:
    answer_lower = answer.lower()

    missing = [
        term for term in must_include
        if term.lower() not in answer_lower
    ]

    score = len(must_include) - len(missing)
    return score, missing


def source_summary(docs: List[Document]):
    rows = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        rows.append(
            {
                "source_number": i,
                "case_title": meta.get("case_title"),
                "citation": meta.get("citation"),
                "document_type": meta.get("document_type"),
                "opinion_role": meta.get("opinion_role"),
                "section_label": meta.get("section_label"),
                "chunk_id": meta.get("chunk_id"),
                "total_chunks": meta.get("total_chunks"),
                "source_url": meta.get("source"),
                "preview": doc.page_content[:400].replace("\n", " "),
            }
        )

    return rows


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = OUTPUT_DIR / f"model_eval_results_{timestamp}.json"

    print("🔎 Loading vector store...")
    embeddings = OllamaEmbeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("✅ Vector store loaded.")

    all_results = []

    for model in MODELS_TO_TEST:
        print(f"\n==============================")
        print(f"🧠 Testing model: {model}")
        print(f"==============================")

        os.environ["OLLAMA_GEN_MODEL"] = model

        for test in TESTS:
            question = test["question"]
            print(f"\nQuestion: {question}")

            raw_docs = vectorstore.similarity_search_with_score(question, k=30)
            docs = rerank_docs(question, raw_docs)[:8]
            context = format_context(docs)

            try:
                answer = generate_answer(model, question, context)
                score, missing = score_answer(answer, test["must_include"])

                result = {
                    "model": model,
                    "case": test["case"],
                    "question": question,
                    "must_include": test["must_include"],
                    "score": score,
                    "max_score": len(test["must_include"]),
                    "missing_terms": missing,
                    "answer": answer,
                    "sources": source_summary(docs),
                }

                print(f"Score: {score}/{len(test['must_include'])}")
                if missing:
                    print(f"Missing: {', '.join(missing)}")
                else:
                    print("✅ Required terms present")

            except Exception as exc:
                result = {
                    "model": model,
                    "case": test["case"],
                    "question": question,
                    "error": str(exc),
                }
                print(f"❌ Error: {exc}")

            all_results.append(result)

    output_file.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n✅ Saved eval results to: {output_file}")


if __name__ == "__main__":
    main()