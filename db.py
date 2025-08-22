from typing import List, Dict

class SampleMedicalDB:
    def __init__(self):
        self.documents = [
            "Humira recommended dosage is 40mg every other week, subcutaneous injection.",
            "Metformin dosage typically starts at 500mg twice daily for type 2 diabetes management.",
            "Aspirin is used for pain relief at doses of 325-650mg every 4-6 hours and for cardiovascular protection at 81mg daily."
        ]
        self.metadatas = [{"source": "sample_db", "id": i} for i in range(len(self.documents))]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_words = set(query.lower().split())
        results = []
        for i, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
            if overlap > 0.05:
                results.append({"text": doc, "metadata": self.metadatas[i], "similarity": overlap})
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]