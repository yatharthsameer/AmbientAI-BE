"""
Simple Retrieval-Augmented Generation (RAG) service.

This service provides lightweight retrieval of relevant medical guideline
snippets using heuristic keyword matching. It has no external dependencies
and is intended as a minimal, production-safe enhancement that can run
without GPU or API keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Guideline:
    title: str
    content: str
    content_type: str  # guideline | protocol | reference
    specialty: Optional[str] = None
    source: Optional[str] = None
    keywords: Optional[List[str]] = None
    medical_terms: Optional[List[str]] = None


class SimpleRAGService:
    """Lightweight RAG service based on keyword heuristics.

    - retrieve_context(question, text): returns a ranked list of relevant guidelines
    - enhance_answer(answer, question, text): returns enhanced answer and metadata
    """

    def __init__(self) -> None:
        self._guidelines: List[Guideline] = self._load_default_guidelines()

    def _load_default_guidelines(self) -> List[Guideline]:
        """Load a minimal, built-in medical knowledge set."""
        return [
            Guideline(
                title="Chest Pain Assessment Protocol",
                content=(
                    "Assess chest pain using PQRST (Provocation, Quality, Radiation, Severity, Timing). "
                    "Immediate ECG and troponin if ACS suspected. Red flags: ST elevation, hypotension, arrhythmia."
                ),
                content_type="protocol",
                specialty="cardiology",
                source="AHA/ACC",
                keywords=["chest pain", "angina", "pressure", "ecg", "troponin"],
                medical_terms=["ACS", "myocardial infarction", "arrhythmia"],
            ),
            Guideline(
                title="Headache Assessment Protocol",
                content=(
                    "Evaluate headache with PQRST. Red flags: sudden onset, worst headache ever, "
                    "neurological deficits, altered mental status. Consider imaging if red flags present."
                ),
                content_type="protocol",
                specialty="neurology",
                source="AAN",
                keywords=["headache", "migraine", "neurological", "pain"],
                medical_terms=["aura", "photophobia", "nuchal rigidity"],
            ),
            Guideline(
                title="Medication Administration Safety",
                content=(
                    "Verify patient identity, check allergies, confirm drug, dose, route, and timing. "
                    "Document administration and monitor for adverse reactions."
                ),
                content_type="guideline",
                specialty="general",
                source="Nursing Standards",
                keywords=["administered", "medication", "dose", "allergy", "acetaminophen"],
                medical_terms=["adverse reaction", "contraindication"],
            ),
            Guideline(
                title="Vital Signs Interpretation",
                content=(
                    "Normal vital signs vary by age. Assess HR, BP, RR, SpO2, temperature. "
                    "Escalate if out of range or patient symptomatic."
                ),
                content_type="reference",
                specialty="general",
                source="Clinical Reference",
                keywords=["vital signs", "stable", "normal", "bp", "hr", "rr", "spo2"],
                medical_terms=["tachycardia", "hypotension", "hypoxemia"],
            ),
            Guideline(
                title="Discharge Planning Basics",
                content=(
                    "Provide clear follow-up instructions, warning signs, and medication guidance. "
                    "Ensure patient understanding and access to care."
                ),
                content_type="guideline",
                specialty="general",
                source="Care Transitions",
                keywords=["discharge", "follow-up", "instructions", "home"],
                medical_terms=["readmission", "adherence"],
            ),
        ]

    def retrieve_context(self, question: str, conversation_text: str, top_k: int = 3) -> List[Dict]:
        """Rank guidelines by simple keyword overlap with question and text."""
        q = (question or "").lower()
        t = (conversation_text or "").lower()

        ranked: List[Dict] = []
        for g in self._guidelines:
            score = 0.0
            for kw in (g.keywords or []):
                if kw in t:
                    score += 2.0
                if kw in q:
                    score += 1.0
            for term in (g.medical_terms or []):
                if term.lower() in t:
                    score += 3.0
                if term.lower() in q:
                    score += 1.0

            if score > 0:
                ranked.append({
                    "title": g.title,
                    "content": g.content,
                    "content_type": g.content_type,
                    "specialty": g.specialty,
                    "source": g.source,
                    "relevance_score": round(min(score / 10.0, 1.0), 3),
                })

        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return ranked[:top_k]

    def enhance_answer(
        self,
        base_answer: Optional[str],
        question: str,
        conversation_text: str,
        top_k: int = 2,
    ) -> Dict:
        """Return enhancement metadata using retrieved context; does not rewrite answer."""
        ctx = self.retrieve_context(question, conversation_text, top_k=top_k)
        return {
            "enhanced": bool(ctx),
            "guidelines_applied": [c["title"] for c in ctx],
            "rag_context": ctx,
        }

    def get_guidelines(self, specialty: Optional[str] = None) -> List[Dict]:
        """Return available guideline catalog."""
        items = self._guidelines
        if specialty:
            items = [g for g in items if (g.specialty or "").lower() == specialty.lower()]
        return [
            {
                "title": g.title,
                "content": g.content,
                "content_type": g.content_type,
                "specialty": g.specialty,
                "source": g.source,
            }
            for g in items
        ]

    def add_guidelines(self, items: List[Dict]) -> int:
        """Ingest user-provided guideline items into memory.

        Expected fields: title (str), content (str), optional content_type, specialty, source, keywords, medical_terms.
        """
        added = 0
        for it in items or []:
            title = (it or {}).get("title")
            content = (it or {}).get("content")
            if not title or not content:
                continue
            self._guidelines.append(
                Guideline(
                    title=title,
                    content=content,
                    content_type=(it.get("content_type") or "guideline"),
                    specialty=it.get("specialty"),
                    source=it.get("source"),
                    keywords=it.get("keywords"),
                    medical_terms=it.get("medical_terms"),
                )
            )
            added += 1
        return added

    def add_qa_pairs(self, pairs: List[Dict]) -> int:
        """Ingest Q&A pairs as guidelines for retrieval.

        Each pair: question (title), answer (content), optional specialty/source.
        """
        items = []
        for p in pairs or []:
            q = (p or {}).get("question")
            a = (p or {}).get("answer")
            if not q or not a:
                continue
            items.append({
                "title": q,
                "content": a,
                "content_type": "qa_pair",
                "specialty": p.get("specialty"),
                "source": p.get("source", "user_dataset"),
                "keywords": p.get("keywords"),
                "medical_terms": p.get("medical_terms"),
            })
        return self.add_guidelines(items)

    def clear(self) -> int:
        """Clear all loaded guidelines."""
        count = len(self._guidelines)
        self._guidelines = []
        return count


