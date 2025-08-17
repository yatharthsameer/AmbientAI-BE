import requests
from datasets import load_dataset
import itertools

BASE_URL = "http://localhost:8000"


def ingest_medquad(batch_size: int = 500):
    """Ingest keivalya/MedQuad-MedicalQnADataset into RAG QA."""
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")

    def row_to_pair(r):
        q = r.get("question") or r.get("Question")
        a = r.get("answer") or r.get("Answer")
        if not q or not a:
            return None
        return {
            "question": q.strip(),
            "answer": a.strip(),
            "specialty": r.get("specialty") or r.get("topic") or r.get("disease") or None,
            "source": "MedQuAD",
        }

    all_pairs = []
    for split in ds:
        for r in ds[split]:
            p = row_to_pair(r)
            if p:
                all_pairs.append(p)

    def chunks(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, size))
            if not batch:
                break
            yield batch

    added = 0
    for batch in chunks(all_pairs, batch_size):
        resp = requests.post(f"{BASE_URL}/api/v1/rag/qa:ingest", json={"pairs": batch}, timeout=120)
        resp.raise_for_status()
        added += resp.json().get("added", 0)
    print(f"Ingested {added} QA pairs from MedQuAD")


if __name__ == "__main__":
    ingest_medquad()