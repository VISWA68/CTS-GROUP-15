import re
import json
from typing import List, Dict
from difflib import SequenceMatcher

class DeduplicationAgent:
    def __init__(self, similarity_threshold: float = 0.85, min_score: float = 0.5):
        """
        similarity_threshold: text similarity cutoff for duplicate removal
        min_score: filter out low confidence retrievals
        """
        self.similarity_threshold = similarity_threshold
        self.min_score = min_score

    def _normalize_text(self, text: str) -> str:
        """Basic normalization: lowercase, remove punctuation/extra spaces."""
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    def _is_similar(self, t1: str, t2: str) -> bool:
        """Check if two texts are near-duplicates."""
        return SequenceMatcher(None, t1, t2).ratio() >= self.similarity_threshold

    def deduplicate(self, retrievals: List[Dict]) -> Dict:
        seen = []
        curated = []

        for r in sorted(retrievals, key=lambda x: x.get("score", 0), reverse=True):
            if r.get("score", 0) < self.min_score:
                continue  # drop low-confidence
            norm = self._normalize_text(r["text"])

            if any(self._is_similar(norm, s) for s in seen):
                continue  # duplicate, skip

            seen.append(norm)
            curated.append({
                "text": r["text"],
                "citation": r.get("citation"),
                "score": r.get("score")
            })

        return {"curated_retrievals": curated}
