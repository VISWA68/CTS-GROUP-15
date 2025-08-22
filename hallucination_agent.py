from typing import List, Dict
from db import SampleMedicalDB

class HallucinationDetector:
    def __init__(self):
        self.db = SampleMedicalDB()

    def verify_locally(self, text: str, context_chunks: List[Dict]) -> str:
        """Keep only words that appear in the DB"""
        if not context_chunks:
            return ""
        verified_words = []
        db_words = set()
        for chunk in context_chunks:
            db_words.update(chunk['text'].split())
        for w in text.split():
            if w in db_words:
                verified_words.append(w)
        return " ".join(verified_words)

    def detect_hallucination(self, text: str, citation: str, score: float) -> Dict:
        context_chunks = self.db.search(text)
        verified_text = self.verify_locally(text, context_chunks)
        status = "valid" if verified_text == text else "invalid"
        return {
            "text": verified_text,
            "citation": citation,
            "score": score,
            "status": status
        }