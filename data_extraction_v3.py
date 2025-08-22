
import streamlit as st
import os
import re
import hashlib
import json
import sqlite3
import time
import redis
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from functools import wraps

# PDF processing imports
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import camelot
import pandas as pd

# ML/AI imports
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================
# CONFIGURATION
# =====================================
DATA_DIR = Path(".")
PDF_DIR = DATA_DIR / "pdfs"
ASSETS_DIR = DATA_DIR / "pdf_assets"
LOG_DB = DATA_DIR / "ingest_logs.db"
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_store")

# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.6

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# =====================================
# UTILITY FUNCTIONS
# =====================================
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def retry(max_tries=3, backoff=1.0, exceptions=(Exception,)):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = backoff
            for attempt in range(1, max_tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_tries:
                        raise
                    logger.warning(f"{func.__name__} failed (attempt {attempt}/{max_tries}): {e}")
                    time.sleep(delay)
                    delay *= 2
            return wrapper
        return deco
    return deco

# =====================================
# AGENT 1: DATA INGESTION
# =====================================
class DataIngestionAgent:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client, self.collection = self._setup_chroma()
        self.conn = self._init_log_db()
        self.dedupe_hashes = self._load_dedupe_hashes()
        ensure_dir(ASSETS_DIR)
    
    def _setup_chroma(self):
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection("drug_pdfs")
        return client, collection
    
    def _init_log_db(self):
        conn = sqlite3.connect(LOG_DB)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ingests (
            id TEXT PRIMARY KEY,
            file TEXT,
            file_hash TEXT,
            pages INTEGER,
            chunks INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            meta JSON
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dedupe_hashes (
            hash TEXT PRIMARY KEY,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        return conn
    
    def _load_dedupe_hashes(self) -> set:
        cur = self.conn.cursor()
        cur.execute("SELECT hash FROM dedupe_hashes")
        return {row[0] for row in cur.fetchall()}
    
    def _fix_hyphenation_and_linebreaks(self, text: str) -> str:
        text = re.sub(r"(\w+)-\n(\w+)", lambda m: m.group(1) + m.group(2), text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    
    def _detect_headings(self, page: fitz.Page) -> List[Tuple[int, str]]:
        text = page.get_text("text")
        headings = []
        lines = text.splitlines()
        idx = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                idx += len(line) + 1
                continue
            
            # Detect numbered headings and all-caps sections
            if (re.match(r'^\d+(\.\d+)*\s+[A-Z][A-Z0-9 \-\,\(\)\/]+$', stripped) or
                (len(stripped) >= 10 and stripped.upper() == stripped and 
                 sum(c.isalpha() for c in stripped) > 4)):
                headings.append((idx, stripped))
            
            idx += len(line) + 1
        
        return headings
    
    def _assign_section(self, page_text: str, chunk_start_idx: int, headings: List[Tuple[int, str]]) -> str:
        section = "Unknown"
        for idx, h in headings:
            if idx <= chunk_start_idx:
                section = h
            else:
                break
        return section
    
    def _extract_pdf_assets(self, pdf_path: Path, temp_image_dir: Path) -> Dict[str, Any]:
        ensure_dir(temp_image_dir)
        doc = fitz.open(str(pdf_path))
        page_texts = []
        
        for pno in range(len(doc)):
            page = doc[pno]
            page_num = pno + 1
            raw_text = page.get_text("text")
            text = raw_text.strip()
            
            # OCR fallback for pages with little text
            if not text or len(text) < 30:
                try:
                    pix = page.get_pixmap(dpi=200)
                    mode = "RGBA" if pix.alpha else "RGB"
                    from PIL import Image
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    text = ocr_text
                except Exception as e:
                    logger.warning(f"OCR fallback failed: {e}")
            
            page_texts.append({"page": page_num, "text": text})
        
        return {"page_texts": page_texts}
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return []
        
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        chunks = []
        cur_sentences = []
        cur_len = 0
        cur_start = 0
        char_cursor = 0
        
        def emit_current():
            nonlocal cur_sentences, cur_len, cur_start
            if not cur_sentences:
                return
            chunk_text = " ".join(s.strip() for s in cur_sentences).strip()
            chunks.append({"chunk": chunk_text, "start_char": cur_start})
            
            if CHUNK_OVERLAP > 0:
                tail = []
                acc = 0
                for s in reversed(cur_sentences):
                    tail.insert(0, s)
                    acc += len(s) + 1
                    if acc >= CHUNK_OVERLAP:
                        break
                cur_sentences = tail
                cur_len = sum(len(s) + 1 for s in cur_sentences)
                cur_start = max(0, char_cursor - cur_len)
            else:
                cur_sentences = []
                cur_len = 0
        
        for sent in sentences:
            if not sent.strip():
                char_cursor += len(sent) + 1
                continue
            
            sent_len = len(sent) + 1
            if cur_len + sent_len <= CHUNK_MAX_CHARS or not cur_sentences:
                if not cur_sentences:
                    cur_start = char_cursor
                cur_sentences.append(sent)
                cur_len += sent_len
            else:
                emit_current()
                if cur_len + sent_len <= CHUNK_MAX_CHARS:
                    cur_sentences.append(sent)
                    cur_len += sent_len
            
            char_cursor += sent_len
        
        if cur_sentences:
            emit_current()
        
        return chunks
    
    def ingest_pdf(self, pdf_path: Path, drug_name: str):
        logger.info(f"Processing {pdf_path} as {drug_name}")
        assets_dir = ASSETS_DIR / pdf_path.stem
        
        # Check if already ingested
        fh = file_sha1(pdf_path)
        cur = self.conn.cursor()
        cur.execute("SELECT file_hash FROM ingests WHERE file = ?", (str(pdf_path.name),))
        row = cur.fetchone()
        if row and row[0] == fh:
            logger.info(f"File {pdf_path.name} already ingested - skipping")
            return
        
        extracted = self._extract_pdf_assets(pdf_path, assets_dir)
        all_chunks, all_metadatas, ids = [], [], []
        total_chunks = 0
        
        with fitz.open(str(pdf_path)) as doc:
            for p in extracted["page_texts"]:
                page_num = p["page"]
                page_text = self._fix_hyphenation_and_linebreaks(p["text"])
                page = doc[page_num - 1]
                headings = self._detect_headings(page)
                
                page_chunks = self._chunk_text(page_text)
                for ci, cobj in enumerate(page_chunks):
                    chunk_text = cobj["chunk"]
                    start_char = cobj["start_char"]
                    section = self._assign_section(page_text, start_char, headings)
                    
                    h = sha1(chunk_text)
                    if h in self.dedupe_hashes:
                        continue
                    
                    self.dedupe_hashes.add(h)
                    doc_id = f"{pdf_path.stem}~{page_num}~{ci}~{uuid4().hex[:8]}"
                    
                    metadata = {
                        "drug": drug_name,
                        "source_file": str(pdf_path.name),
                        "page": page_num,
                        "section": section,
                        "chunk_start": start_char,
                        "chunk_id": doc_id,
                        "type": "text_chunk"
                    }
                    
                    all_chunks.append(chunk_text)
                    all_metadatas.append(metadata)
                    ids.append(doc_id)
                    total_chunks += 1
        
        # Batch embed and store
        if all_chunks:
            for i in range(0, len(all_chunks), BATCH_SIZE):
                batch_texts = all_chunks[i:i + BATCH_SIZE]
                batch_ids = ids[i:i + BATCH_SIZE]
                batch_metas = all_metadatas[i:i + BATCH_SIZE]
                
                embeddings = self.embedder.encode(batch_texts, show_progress_bar=False)
                embeddings_list = [emb.tolist() for emb in embeddings]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metas,
                    embeddings=embeddings_list
                )
        
        # Log ingestion
        ingest_id = sha1(str(pdf_path.resolve()) + fh)
        cur.execute("""
        INSERT OR REPLACE INTO ingests (id, file, file_hash, pages, chunks, meta) 
        VALUES (?, ?, ?, ?, ?, ?)
        """, (ingest_id, str(pdf_path.name), fh, len(extracted["page_texts"]), 
              total_chunks, json.dumps({"drug": drug_name})))
        self.conn.commit()
        
        logger.info(f"Stored {total_chunks} chunks from {pdf_path.name}")

# =====================================
# AGENT 2: RETRIEVAL AND QUERY ROUTING
# =====================================
class RetrievalQueryRoutingAgent:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract drug names and intent from query using Gemini"""
        prompt = f"""
        Analyze this medical query and extract:
        1. Drug names mentioned (Humira, Rinvoq, Skyrizi, etc.)
        2. Intent/topic (dosage, side effects, interactions, contraindications, etc.)
        
        Query: "{query}"
        
        Respond in JSON format:
        {{
            "drugs": ["drug1", "drug2"],
            "intent": "primary_topic"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip())
            return result
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {"drugs": [], "intent": "general"}
    
    def vector_search(self, query: str, top_k: int = 8, filter_metadata: Optional[dict] = None) -> List[Dict]:
        """Perform vector similarity search"""
        query_embedding = self.embedder.encode([query])[0].tolist()
        
        where = filter_metadata if filter_metadata else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        retrievals = []
        for doc, meta, dist in zip(docs, metas, distances):
            similarity = max(0.0, 1.0 - dist) if dist is not None else 0.0
            citation = f"{meta.get('source_file', '')} (Page: {meta.get('page', '')}, Section: {meta.get('section', '')})"
            
            retrievals.append({
                "text": doc,
                "metadata": meta,
                "citation": citation,
                "similarity": similarity,
                "distance": dist
            })
        
        return retrievals

# =====================================
# AGENT 3: REASONING AND DOMAIN AGENT
# =====================================
class ReasoningDomainAgent:
    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def assess_chunk_relevance(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Assess relevance of chunks to query and filter by confidence threshold"""
        filtered_chunks = []
        
        for chunk in chunks:
            relevance_score = self._calculate_relevance(query, chunk["text"])
            chunk["relevance_score"] = relevance_score
            
            if relevance_score >= CONFIDENCE_THRESHOLD:
                filtered_chunks.append(chunk)
        
        # Sort by relevance score descending
        filtered_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return filtered_chunks
    
    def _calculate_relevance(self, query: str, chunk_text: str) -> float:
        """Calculate relevance score between query and chunk"""
        prompt = f"""
        Rate the relevance of this text chunk to the user query on a scale of 0.0 to 1.0.
        
        Query: "{query}"
        
        Text Chunk: "{chunk_text[:1000]}..."
        
        Consider:
        - Direct topical relevance
        - Presence of specific drug names mentioned in query
        - Relevance to medical context
        
        Respond with only a decimal number between 0.0 and 1.0:
        """
        
        try:
            response = self.model.generate_content(prompt)
            score_text = response.text.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            # Fallback: simple keyword matching
            return self._simple_relevance_score(query, chunk_text)
    
    def _simple_relevance_score(self, query: str, chunk_text: str) -> float:
        """Simple fallback relevance scoring"""
        query_words = set(query.lower().split())
        chunk_words = set(chunk_text.lower().split())
        intersection = query_words & chunk_words
        union = query_words | chunk_words
        
        if not union:
            return 0.0
        
        jaccard_score = len(intersection) / len(union)
        return min(1.0, jaccard_score * 2)  # Scale up for better discrimination
    
    def check_relationships_exist(self, retrievals: List[Dict], query: str) -> bool:
        """Check if retrievals contain sufficient information to answer query"""
        if not retrievals:
            return False
        
        # Simple heuristic: if we have high-confidence retrievals, relationships exist
        avg_relevance = sum(r.get("relevance_score", 0) for r in retrievals) / len(retrievals)
        return avg_relevance >= CONFIDENCE_THRESHOLD

# =====================================
# AGENT 4: ORCHESTRATION & ANSWER GENERATION
# =====================================
class AnswerGenerationAgent:
    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def generate_final_response(self, query: str, history: List[Dict], retrievals: List[Dict]) -> Dict[str, Any]:
        """Generate final structured response using Gemini"""
        if not retrievals:
            return {
                "short_answer": "I don't have sufficient information to answer your query.",
                "confidence_score": 0.0,
                "citations": [],
                "reasoning": "No relevant information found in the knowledge base."
            }
        
        # Build context from retrievals
        context_parts = []
        citations = []
        
        for i, retrieval in enumerate(retrievals):
            context_parts.append(f"Source {i+1}: {retrieval['text']}")
            citations.append({
                "source": retrieval["metadata"].get("source_file", ""),
                "page_reference": str(retrieval["metadata"].get("page", "")),
                "section_id": retrieval["metadata"].get("section", ""),
                "relevance_score": retrieval.get("relevance_score", 0.0)
            })
        
        context = "\n\n".join(context_parts)
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])  # Last 5 messages
        
        prompt = f"""
        You are a medical information assistant. Provide a structured response based ONLY on the provided context.
        
        Conversation History:
        {history_text}
        
        Retrieved Context:
        {context}
        
        User Query: {query}
        
        Your response must be in JSON format with these fields:
        - "short_answer": Direct, concise answer (string)
        - "confidence_score": Your confidence in the answer from 0.0 to 1.0 (float)
        - "reasoning": Brief explanation of your reasoning (string)
        
        Guidelines:
        - Use ONLY information from the provided context
        - Be concise and medically accurate
        - If information is insufficient, state so clearly
        - Include specific details like dosages, side effects when available
        - Maintain professional medical tone
        
        JSON Response:
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            final_response = json.loads(response_text)
            final_response["citations"] = citations
            
            return final_response
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "short_answer": "I encountered an error while processing your query. Please try rephrasing your question.",
                "confidence_score": 0.0,
                "citations": citations,
                "reasoning": f"Error in answer generation: {str(e)}"
            }

# =====================================
# AGENT 5: SESSION MANAGEMENT
# =====================================
class SessionManagementAgent:
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis_client.ping()  # Test connection
            self.use_redis = True
        except:
            logger.warning("Redis not available, using session state fallback")
            self.use_redis = False
    
    def get_history(self, session_id: str) -> List[Dict]:
        if self.use_redis:
            try:
                history_json = self.redis_client.get(session_id)
                return json.loads(history_json) if history_json else []
            except:
                return []
        else:
            # Fallback to Streamlit session state
            return st.session_state.get(f"history_{session_id}", [])
    
    def update_history(self, session_id: str, role: str, content: str):
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        
        if self.use_redis:
            try:
                self.redis_client.set(session_id, json.dumps(history))
            except:
                pass  # Silent fail
        else:
            st.session_state[f"history_{session_id}"] = history

# =====================================
# MAIN ORCHESTRATOR
# =====================================
class DrugChatbotOrchestrator:
    def __init__(self):
        # Initialize agents
        st.cache_resource.clear()  # Clear any cached resources
        
        with st.spinner("Initializing chatbot components..."):
            self.ingestion_agent = DataIngestionAgent()
            self.retrieval_agent = RetrievalQueryRoutingAgent(
                self.ingestion_agent.collection, 
                self.ingestion_agent.embedder
            )
            self.reasoning_agent = ReasoningDomainAgent()
            self.answer_agent = AnswerGenerationAgent()
            self.session_agent = SessionManagementAgent()
    
    def ingest_pdfs(self):
        """Ingest PDFs if they exist"""
        pdf_files = {
            "Humira": "humira.pdf",
            "Rinvoq": "rinvoq_pi.pdf", 
            "Skyrizi": "skyrizi_pi.pdf"
        }
        
        for drug_name, filename in pdf_files.items():
            pdf_path = PDF_DIR / filename
            if pdf_path.exists():
                try:
                    self.ingestion_agent.ingest_pdf(pdf_path, drug_name)
                    st.success(f"✅ Ingested {filename}")
                except Exception as e:
                    st.error(f"❌ Failed to ingest {filename}: {e}")
            else:
                st.warning(f"⚠️ {filename} not found in {PDF_DIR}")
    
    def process_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """Main orchestration logic"""
        # Step 1: Entity Recognition
        entities = self.retrieval_agent.extract_entities(query)
        
        # Step 2: Vector Search
        filter_metadata = None
        if entities.get("drugs"):
            # Filter by drug if specific drugs mentioned
            drug_filter = {"drug": {"$in": entities["drugs"]}}
            filter_metadata = drug_filter
        
        retrievals = self.retrieval_agent.vector_search(
            query, top_k=8, filter_metadata=filter_metadata
        )
        
        # Step 3: Reasoning and Filtering
        filtered_retrievals = self.reasoning_agent.assess_chunk_relevance(query, retrievals)
        
        # Step 4: Check if sufficient information exists
        has_sufficient_info = self.reasoning_agent.check_relationships_exist(filtered_retrievals, query)
        
        if not has_sufficient_info:
            return {
                "short_answer": "I don't have sufficient relevant information to answer your query based on the available documents.",
                "confidence_score": 0.0,
                "citations": [],
                "reasoning": "Insufficient relevant information in knowledge base.",
                "entities_found": entities
            }
        
        # Step 5: Generate Final Answer
        history = self.session_agent.get_history(session_id)
        response = self.answer_agent.generate_final_response(query, history, filtered_retrievals[:4])
        response["entities_found"] = entities
        
        return response

# =====================================
# STREAMLIT UI
# =====================================
def main():
    st.set_page_config(page_title="Drug Information Chatbot", page_icon="💊", layout="wide")
    
    st.title("💊 Drug Information Chatbot")
    st.caption("RAG-powered medical information assistant using Gemini AI")
    
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}_{uuid4().hex[:8]}"
    
    if "orchestrator" not in st.session_state:
        try:
            st.session_state.orchestrator = DrugChatbotOrchestrator()
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {e}")
            return
    
    orchestrator = st.session_state.orchestrator
    
    # Sidebar for system controls
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("🔄 Ingest PDFs"):
            with st.spinner("Ingesting PDF documents..."):
                orchestrator.ingest_pdfs()
        
        st.header("Configuration")
        st.info(f"**Model:** {GEMINI_MODEL}")
        st.info(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD}")
        st.info(f"**Session ID:** {st.session_state.session_id}")
        
        if st.button("🗑️ Clear History"):
            orchestrator.session_agent.update_history(st.session_state.session_id, "system", "History cleared")
            st.rerun()
    
    # Main chat interface
    history = orchestrator.session_agent.get_history(st.session_state.session_id)
    
    # Display chat history
    for message in history:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Query input
    if query := st.chat_input("Ask about drug information (e.g., 'What are the side effects of Humira?')"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        orchestrator.session_agent.update_history(st.session_state.session_id, "user", query)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                response = orchestrator.process_query(query, st.session_state.session_id)
            
            # Format response
            formatted_response = f"""
            **Answer:** {response.get('short_answer', 'No answer available')}
            
            **Confidence:** {response.get('confidence_score', 0.0):.2f}
            
            **Reasoning:** {response.get('reasoning', 'No reasoning provided')}
            """
            
            if response.get('citations'):
                formatted_response += "\n\n**Sources:**\n"
                for i, citation in enumerate(response['citations'], 1):
                    formatted_response += f"{i}. {citation.get('source', 'Unknown')} "
                    formatted_response += f"(Page: {citation.get('page_reference', 'N/A')}, "
                    formatted_response += f"Section: {citation.get('section_id', 'N/A')}, "
                    formatted_response += f"Relevance: {citation.get('relevance_score', 0.0):.2f})\n"
            
            if response.get('entities_found'):
                with st.expander("🔍 Entity Analysis"):
                    st.json(response['entities_found'])
            
            st.markdown(formatted_response)
        
        # Update history with response
        orchestrator.session_agent.update_history(
            st.session_state.session_id, 
            "assistant", 
            formatted_response
        )

if __name__ == "__main__":
    main()