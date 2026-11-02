"""Part D: policy chunking, TF-IDF retrieval, and grounded answers."""

from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
POL = ROOT / "data" / "policies"


# ---------------------------------------------------------------------------
# Policy retrieval
# ---------------------------------------------------------------------------

class PolicyRAG:
    """Build a TF-IDF index over policy chunks and answer grounded questions."""

    def __init__(self, chunk_size=40, overlap=10):
        """Load policy text, split it into overlapping chunks, and build an index."""
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []

        for path in sorted(POL.glob("*.txt")):
            words = path.read_text(encoding="utf-8").split()
            step = max(1, chunk_size - overlap)

            for start in range(0, len(words), step):
                chunk = words[start : start + chunk_size]
                self.chunks.append(
                    {
                        "source": path.name,
                        "text": " ".join(chunk),
                    }
                )

                if start + chunk_size >= len(words):
                    break

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=3):
        """Return the top-k policy chunks ranked by cosine similarity."""
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        indices = scores.argsort()[::-1][:k]

        return [
            {
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            }
            for index in indices
        ]

    def answer(self, question):
        """Ask the offline mock model to answer using retrieved policy context only."""
        hits = self.retrieve(question, 3)
        context = "\n".join(
            f"[{hit['source']}]\n{hit['text']}" for hit in hits
        )

        response = mock_llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "POLICY ASSISTANT: answer only from supplied context. "
                        "If not present, say I could not find that information "
                        "in the policy documents."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\nQUESTION:\n{question}"
                    ),
                },
            ]
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": list(
                dict.fromkeys(hit["source"] for hit in hits)
            ),
            "retrieved": hits,
        }


# ---------------------------------------------------------------------------
# Policy retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate(rag):
    """Evaluate document retrieval, expected facts, and fallback responses."""
    evaluation_data = pd.read_csv(
        ROOT / "data" / "raw" / "policy_eval.csv",
        keep_default_na=False,
    )
    results = []

    for _, row in evaluation_data.iterrows():
        answer_result = rag.answer(row.question)
        expected_doc = row.expected_doc
        expected_fact = row.expected_fact

        results.append(
            {
                "question": row.question,
                "expected_doc": expected_doc,
                "retrieved_doc": (
                    expected_doc in answer_result["sources"]
                    if expected_doc != "NONE"
                    else False
                ),
                "fact_found": (
                    expected_fact in answer_result["answer"]
                    if expected_fact
                    else False
                ),
                "fallback": answer_result["answer"].startswith(
                    "I could not find"
                ),
            }
        )

    return pd.DataFrame(results)
