from langchain_community.vectorstores import FAISS

from query_demo_clean import VECTORSTORE_PATH, OllamaEmbeddings
from case_store import build_case_index


def main():
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        OllamaEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    case_index = build_case_index(vectorstore)

    for case_title, record in sorted(case_index.items()):
        print("\n" + "=" * 100)
        print(f"CASE: {case_title}")
        print(f"CITATION: {record.get('citation')}")
        print(f"TOTAL CHARS: {record.get('total_chars')}")
        print(f"OPINIONS: {len(record.get('opinions', []))}")

        for i, op in enumerate(record.get("opinions", []), start=1):
            preview = op["text"][:250].replace("\n", " ")
            print("-" * 100)
            print(f"Opinion {i}")
            print(f"  key:       {op.get('opinion_key')}")
            print(f"  role:      {op.get('role')}")
            print(f"  author:    {op.get('author')}")
            print(f"  chunks:    {op.get('n_chunks')}")
            print(f"  chars:     {op.get('char_len')}")
            print(f"  source:    {op.get('source')}")
            print(f"  preview:   {preview}")


if __name__ == "__main__":
    main()