# 🧠 Multi-Agent Medical Query Processing System

This project implements a **modular multi-agent system** for classifying, retrieving, and deduplicating medical queries.  
It uses **rule-based NLP + Gemini LLM (optional)** for classification, **ChromaDB** for retrieval, and a **deduplication agent** to refine results.

---

## 🚀 Features
1. **Query Classifier Agent**  
   - Identifies user query intents:  
     - `dosage`  
     - `contraindication`  
     - `interaction`  
     - `general_info`  
     - `side_effect`  
     - `usage`  
   - Extracts drug entities using:
     - Rule-based keyword search  
     - Drug lexicon  
     - spaCy (optional biomedical NER)  
     - Gemini API (if enabled)  

2. **Retriever Agent**  
   - Searches **ChromaDB** vector store for relevant chunks.  
   - Hybrid: semantic + keyword search.  
   - Attaches metadata (PDF source, page, section, chunk IDs).  

3. **Deduplication Agent**  
   - Removes near-duplicate results.  
   - Filters low-confidence retrievals.  
   - Returns a curated set of unique, high-score chunks.  

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **ChromaDB** – Vector database  
- **Google Gemini API** (optional, fallback to rule-based)  
- **spaCy (SciSpacy model)** – Medical NER (optional)  
- **dotenv** – For API key management  

---

## 📂 Project Structure
```
├── query_classifier_agent.py   # Main script (Classifier + Retriever + Deduper)
├── db.py                       # Utility to view & inspect ChromaDB
├── drug.txt                    # Lexicon of known drug names
├── chroma_store/               # Persistent ChromaDB storage
├── .env                        # Store API keys securely
└── README.md                   # Documentation
```

---

## 🔑 Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-repo/med-agent-system.git
cd med-agent-system
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` should contain:
```
chromadb
python-dotenv
google-generativeai
spacy
scispacy
```

### 3️⃣ Setup Environment Variables
Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
```

### 4️⃣ Setup Drug Lexicon
Edit `drug.txt` and add drug names (one per line):
```
Humira
Rinvoq
Aspirin
Azathioprine
```

### 5️⃣ (Optional) Setup spaCy Medical Model
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
```

---

## ▶️ Usage

### **Interactive Mode**
```bash
python query_classifier_agent.py
```

Example:
```
Enter query (or 'exit'): What’s the dosage of Humira?
{
  "intents": ["dosage"],
  "entities": ["Humira"],
  "raw_query": "What’s the dosage of Humira?"
}

🔎 Searching DB for: Humira
{
  "curated_retrievals": [
    {
      "text": "Humira recommended dosage is 40mg every other week, subcutaneous injection.",
      "citation": "humira.pdf (Section: Dosing, Page: 35)",
      "score": 0.49
    }
  ]
}
```

### **Single Query Mode**
```bash
python query_classifier_agent.py --query "Can Rinvoq be taken with Humira?" --drug-lexicon drug.txt
```

---

## 🧩 Modules Overview

### 1️⃣ Query Classifier Agent
- Rule-based keyword detection  
- Drug lexicon lookup  
- Gemini API (multi-intent classification)  
- SpaCy NER (optional)  

### 2️⃣ Retriever Agent
- Queries **ChromaDB** with entity names  
- Returns top-k results with citations  

### 3️⃣ Deduplication Agent
- Removes duplicates via similarity check  
- Filters by confidence score  
- Outputs curated retrievals  

---

## 📝 Sample Queries & Outputs

**Input**:  
```
Can Rinvoq be taken with Humira?
```

**Output**:
```json
{
  "intents": ["interaction"],
  "entities": ["Rinvoq", "Humira"],
  "raw_query": "Can Rinvoq be taken with Humira?"
}
```

**Retrievals**:
```json
{
  "curated_retrievals": [
    {
      "text": "Humira recommended dosage is 40mg every other week, subcutaneous injection.",
      "citation": "humira.pdf (Section: Dosing, Page: 35)",
      "score": 0.49
    }
  ]
}
```

---

## ⚠️ Notes
- Gemini free-tier is limited (50 req/day). Falls back to **rule-based mode** if quota exceeded.  
- ChromaDB must be pre-populated with chunks (see `db.py`).  
- This is **not medical advice** — intended for research & prototyping only.  

---

## 📌 Next Steps
- ✅ Multi-intent support  
- ✅ Deduplication agent  
- 🔲 Answer synthesis agent (to generate natural-language answers)  
- 🔲 Evaluation on real-world medical queries  
- 🔲 Web/Streamlit frontend for interactive use  

---

👨‍💻 **Author**: Your Name  
📅 **Last Updated**: August 2025  
