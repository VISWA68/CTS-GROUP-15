import os
import re
import json
import argparse
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()  # Load .env into environment variables
import chromadb


# -----------------------------
# Schema
# -----------------------------

class Intent(str, Enum):
    DOSAGE = "dosage"
    CONTRAINDICATION = "contraindication"
    INTERACTION = "interaction"
    GENERAL_INFO = "general_info"
    SIDE_EFFECT = "side_effect"
    USAGE = "usage"

@dataclass
class ClassifiedQuery:
    intents: List[Intent]   # changed to list for multi-intent
    entities: List[str]
    raw_query: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["intents"] = [i.value for i in self.intents]
        return d

# -----------------------------
# Agent
# -----------------------------

class QueryClassifierAgent:
    def __init__(
        self,
        use_llm: Optional[bool] = None,
        gemini_model: str = "gemini-1.5-flash",
        drug_lexicon: Optional[List[str]] = None,
    ):
        if use_llm is None:
            use_llm = bool(os.getenv("GEMINI_API_KEY"))
        self.use_llm = use_llm
        self.gemini_model = gemini_model
        self.drug_lexicon = set([d.strip().lower() for d in (drug_lexicon or []) if d.strip()])

        # Import Gemini if available
        self.llm = None
        if self.use_llm:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
            self.llm = genai.GenerativeModel(gemini_model)

        
        self.intent_keywords = {
            Intent.DOSAGE: [
                "dose", "dosage", "dosing", "how much", "mg", "milligram", "tablet", "capsule",
                "once daily", "twice daily", "three times daily", "qd", "bid", "tid", "qid",
                "strength", "starting dose", "initial dose", "loading dose", "maintenance dose",
                "maximum dose", "max dose", "usual dose", "recommended dose", "dose adjustment",
                "dose reduction", "dose increase", "dose escalation", "titrate", "titration",
                "low dose", "high dose", "standard dose", "weight-based dose", "per kg",
                "per day", "per week", "weekly dose", "daily dose", "oral dose", "IV dose",
                "subcutaneous dose", "intravenous dose", "single dose", "split dose", "half dose"
            ]
            ,
            Intent.CONTRAINDICATION: [
                "contraindication", "contraindications", "contraindicated", 
                "who should not", "who must not", "avoid in", "should not take", 
                "not take", "not recommended", "not suitable", "unsuitable", 
                "unsafe", "risk factors", "risk of use", "dangerous in", 
                "do not use", "not advised", "must not be used", 
                "black box warning", "boxed warning", "serious warning", 
                "warning", "precaution", "precautions", "special warning", 
                "restricted use", "safety warning", "harmful for", 
                "caution in", "not approved for", "use with caution", 
                "contraindicated with", "adverse risk", "serious risks", 
                "patients with", "avoid if", "not for children", 
                "pregnancy warning", "not for pregnant", "not for elderly", 
                "liver disease warning", "kidney disease warning", 
                "heart condition warning", "should be avoided"
            ],
            Intent.INTERACTION: [
                "interaction", "interactions", "interact", "interacts", "interacting", 
                "together", "taken together", "take together", 
                "with", "combine", "combination", "co-administer", "coadministration", 
                "use with", "taken with", "using with", "given with", "when used with", 
                "drug interaction", "drug-drug interaction", "drug interaction check", "ddi", 
                "drug-food interaction", "food interaction", "meal interaction", 
                "alcohol interaction", "drink with", "alcohol use", "alcohol warning", 
                "mix with", "mixed with", "combined with", 
                "supplement interaction", "herbal interaction", "vitamin interaction", 
                "contraindicated with", "should not be taken with", 
                "avoid taking with", "cannot take with", "not with", "dangerous with", 
                "use caution with", "affects metabolism of", "affects absorption", 
                "synergistic effect", "antagonistic effect", "cross reaction"
            ],
                Intent.SIDE_EFFECT: [
                "side effect", "side effects", "adverse effect", "adverse effects",
                "adverse reaction", "adverse reactions", "bad reaction", "bad reactions",
                "toxicity", "toxic effect", "complication", "complications",
                "negative effect", "harmful effect", "serious effect", "unwanted effect",
                "common side effects", "serious side effects", "long term effects",
                "short term effects", "rare side effects"
            ],
                Intent.USAGE: [
                "what is", "used for", "indication", "indications", "purpose",
                "why take", "treats", "treatment for", "used in", "helps with",
                "benefits", "use of", "role of", "prescribed for"
            ]
                    }

    # --------- Public API ---------

    def classify(self, query: str) -> ClassifiedQuery:
        query = (query or "").strip()
        if not query:
            return ClassifiedQuery(intents=[], entities=[], raw_query=query)  # no intents on empty input


        # Rule-based always runs
        rule_intents, rule_entities = self._classify_rule_based(query)

        # LLM optional
        llm_intents, llm_entities = [], []
        if self.use_llm and self.llm:
            llm_intents, llm_entities = self._classify_with_gemini(query)

        # Merge
        intents = self._unique_preserve_order(rule_intents + llm_intents)
        entities = self._unique_preserve_order(rule_entities + llm_entities)

        # Add spaCy NER
        entities += self._extract_entities_spacy(query)
        entities = self._unique_preserve_order(entities)

        # Remove stop words from final entities
        stop = set([
            "can","could","would","should","may","might","will","shall","must",
            "what","which","who","whom","whose","when","where","why","how",
            "a","an","the","and","or","in","on","at","to","for","of","by","with",
            "about","between","into","through","during","before","after","above",
            "below","from","over","under","again","further",
            "is","are","was","were","be","being","been","do","does","did","doing",
            "have","has","had","having","use","using","used","take","taken",
            "tablet","capsule","pill","injection","dose","dosage","mg","milligram",
            "strength","daily","weekly","treatment","therapy","medication","drug",
            "medicine","patient","condition","disease","doctor",
            # Add generic LLM hallucination words here
            "list", "drugbrandnames", "conditions", "patientgroups", "precautions", "explain"
        ])
        entities = [e for e in entities if e.lower() not in stop]

        return ClassifiedQuery(intents=intents, entities=entities, raw_query=query)

    # --------- Rule-based path ---------

    def _classify_rule_based(self, query: str) -> tuple[List[Intent], List[str]]:
        q = query.lower()
        found_intents = []

        for intent, kws in self.intent_keywords.items():
            if any(kw in q for kw in kws):
                found_intents.append(intent)

        if not found_intents:
            found_intents = [Intent.GENERAL_INFO]

        entities = self._extract_entities_rule_based(query)
        return found_intents, entities

    def _extract_entities_rule_based(self, query: str) -> List[str]:
        entities: List[str] = []

        # Boost with lexicon
        for drug in self.drug_lexicon:
            pattern = r"\b" + re.escape(drug) + r"\b"
            if re.search(pattern, query, flags=re.IGNORECASE):
                entities.append(drug)

        # Tokenization
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", query)

        # Smarter stoplist
        stop = set([
            # ---- Question / Auxiliary words ----
            "can","could","would","should","may","might","will","shall","must",
            "what","which","who","whom","whose","when","where","why","how",
            "is","are","was","were","be","being","been","do","does","did","doing",
            "have","has","had","having",

            # ---- Articles, Conjunctions, Prepositions ----
            "a","an","the","and","or","but","if","because","as","while","of",
            "in","on","at","to","for","by","with","about","between","into",
            "through","during","before","after","above","below","from","over",
            "under","again","further","than","then","once","so","that","this",
            "these","those","there","here","it","its","their","them","they",
            "you","your","we","our","us",

            # ---- Generic medical words (not entities) ----
            "tablet","tablets","capsule","capsules","pill","pills","injection",
            "injections","syrup","dose","doses","dosage","dosing","mg","milligram",
            "milligrams","strength","daily","weekly","monthly","treatment","therapy",
            "medicine","medication","drug","drugs","pharmaceutical","pharmacy",
            "patient","patients","condition","conditions","disease","diseases",
            "disorder","disorders","doctor","doctors","nurse","nurses","health",
            "healthcare","hospital","clinic","symptom","symptoms","side","effects",
            "effect","reaction","reactions","precaution","precautions","warning",
            "warnings","contraindication","contraindications",

            # ---- LLM hallucination / placeholder terms ----
            "list","information","details","explain","describe","overview",
            "drugbrandnames","drugbrand names","patientgroups","patient groups",
            "conditions","diseases","precautionary","usage","guidelines","instruction",
            "instructions","indication","indications","administration","general","note",
            "notes","advice","advisory"
        ])

        # Common drug suffixes to improve detection
        drug_suffixes = ("mab","nib","vir","olol","pril","sartan","dipine","azole")

        # Candidate extraction
        
        for t in tokens:
            if t.lower() in stop:
                continue
            if t[0].isupper():
                if any(t.lower().endswith(s) for s in drug_suffixes) or len(t) > 3:
                    entities.append(t)

        # Contextual extraction (e.g., after 'with/and/between')
        m = re.search(r"(?:with|and|between)\s+([A-Za-z][A-Za-z\-']+)", query, flags=re.IGNORECASE)
        if m:
            entities.append(m.group(1))

        # Final cleanup
        entities = [self._clean_ent(e) for e in entities if len(e) > 2]
        return self._unique_preserve_order(entities)


    # --------- LLM path (Gemini) ---------

    def _classify_with_gemini(self, query: str) -> tuple[List[Intent], List[str]]:
        prompt = f"""
        You are a medical query classifier.
        Query: "{query}"
        Classify ALL applicable intents into: dosage, contraindication, interaction, general_info.
        Extract entities (drug/brand names, conditions, patient groups).
        Respond STRICTLY in JSON format: {{"intents": ["..."], "entities": ["..."]}}
        """

        response = self.llm.generate_content(prompt)
        content = response.text.strip()

        try:
            data = json.loads(self._extract_json(content))
            intents = [Intent(i.lower()) for i in data.get("intents", []) if i.lower() in [x.value for x in Intent]]
            if not intents:
                intents = [Intent.GENERAL_INFO]
            entities = [self._clean_ent(e) for e in data.get("entities", []) if isinstance(e, str)]
            return intents, entities
        except Exception:
            return [Intent.GENERAL_INFO], []

    # --------- spaCy entity extraction ---------

    def _extract_entities_spacy(self, query: str) -> List[str]:
        try:
            import spacy
            nlp = spacy.load("en_core_sci_sm")
            doc = nlp(query)
            ents = [ent.text for ent in doc.ents if ent.label_ in ["DRUG", "CHEMICAL", "DISEASE", "CONDITION"]]
            return [self._clean_ent(e) for e in ents]
        except Exception:
            return []

    # -----------------------------
    # Helpers
    # -----------------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return m.group(0) if m else text

    @staticmethod
    def _clean_ent(e: str) -> str:
        e = re.sub(r"[^A-Za-z0-9\-\s']", "", e).strip()
        return e if not e else e[0].upper() + e[1:]

    @staticmethod
    def _unique_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            xl = x.lower()
            if xl not in seen:
                seen.add(xl)
                out.append(x)
        return out

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Query Classifier Agent - Enhanced")
    parser.add_argument("--query", help="User query text")
    parser.add_argument("--drug-lexicon", default=None, help="Path to a newline-separated list of drug names")
    parser.add_argument("--no-llm", action="store_true", help="Force rule-based mode")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model to use")

    args = parser.parse_args()

    

    # Auto-load lexicon (default = drug.txt in same folder)
    lex = None
    lexicon_path = args.drug_lexicon or "drug.txt"
    if os.path.exists(lexicon_path):
        with open(lexicon_path, "r", encoding="utf-8") as f:
            lex = [line.strip() for line in f if line.strip()]
        

    agent = QueryClassifierAgent(use_llm=not args.no_llm, gemini_model=args.model, drug_lexicon=lex)


    # Interactive mode if no query
    if not args.query:
        while True:
            q = input("\nEnter query (or 'exit'): ")
            if q.lower() in ["exit", "quit"]:
                break
            if not q.strip():     # 👈 skip empty input
                continue
            result = agent.classify(q)
            print(json.dumps(result.to_dict(), indent=2))

            for ent in result.entities:
                print(f"\n🔎 Searching DB for: {ent}")
                deduper = DeduplicationAgent()
                db_results = search_db(ent)
                cleaned = deduper.deduplicate(db_results.get("retrievals", []))
                print(json.dumps(cleaned, indent=2))


    else:
        result = agent.classify(args.query)
        print(json.dumps(result.to_dict(), indent=2))

        for ent in result.entities:
            print(f"\n🔎 Searching DB for: {ent}")
            deduper = DeduplicationAgent()
            db_results = search_db(ent)
            cleaned = deduper.deduplicate(db_results.get("retrievals", []))
            print(json.dumps(cleaned, indent=2))


def search_db(entity: str, top_k: int = 5):
    """
    Search ChromaDB for relevant chunks for the given entity (drug name).
    Returns top_k results with metadata, citation, and score.
    """
    client = chromadb.PersistentClient(path="chroma_store")

    collection = client.get_or_create_collection("drug_chunks")

    results = collection.query(
        query_texts=[entity],
        n_results=top_k,
    )

    retrievals = []
    for i in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][i]
        retrievals.append({
            "text": results["documents"][0][i],
            "metadata": metadata,
            "citation": f"{metadata.get('source_file')} (Section: {metadata.get('section')}, Page: {metadata.get('page')})",
            "raw_distance": results["distances"][0][i],
            "score": 1 / (1 + results["distances"][0][i])  # simple normalization
        })

    return {"retrievals": retrievals}

from difflib import SequenceMatcher

class DeduplicationAgent:
    def __init__(self, similarity_threshold: float = 0.85, min_score: float = 0.2):
        """
        similarity_threshold: cutoff for duplicate removal
        min_score: filter out low-confidence retrievals
        """
        self.similarity_threshold = similarity_threshold
        self.min_score = min_score

    def _normalize_text(self, text: str) -> str:
        """Basic normalization: lowercase + remove punctuation."""
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
                continue  # duplicate found

            seen.append(norm)
            curated.append({
                "text": r["text"],
                "citation": r.get("citation"),
                "score": r.get("score")
            })

        return {"curated_retrievals": curated}

if __name__ == "__main__":
    main()
