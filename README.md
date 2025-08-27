# Drug Information Chatbot - Multi-Agent RAG System

A sophisticated RAG-based Drug Information Chatbot built with **Crew AI** for multi-agent orchestration. This system provides personalized medical information retrieval using advanced PDF processing, vector search, and AI-powered reasoning.

## 🏗️ Architecture Overview

### Multi-Agent System (Crew AI)
The chatbot employs 5 specialized agents working in orchestration:

1. **🔄 Data Ingestion Agent**: PDF processing, chunking, and vector storage
2. **🔍 Retrieval Query Routing Agent**: Entity extraction and vector similarity search
3. **🧠 Reasoning Domain Agent**: Relevance scoring and confidence filtering
4. **💬 Answer Generation Agent**: Structured response generation with citations
5. **📋 Session Management Agent**: Chat history and user context management

### Real-Time Agent Monitoring
- Live agent status updates in the UI
- Visual progress tracking through the multi-agent pipeline
- Color-coded confidence indicators
- Detailed process analysis with expandable sections

## 🚀 Key Features

### Advanced PDF Processing
- **Multi-format extraction**: Text, tables (Camelot), figures, and images
- **OCR capabilities**: Pytesseract for image-to-text conversion
- **Intelligent chunking**: Sentence-based with configurable overlap
- **Heading detection**: Font-based and pattern-based section identification
- **Deduplication**: SHA1-based content deduplication
- **Asset management**: Organized storage of extracted tables and figures

### Personalized Query Enhancement
- **Smart detection**: Identifies personalized queries ("Can I use...", "Should I take...")
- **Context integration**: Age, weight, symptoms automatically appended to search
- **Enhanced retrieval**: User context improves vector search relevance

### Multi-API Key Support
- **Automatic rotation**: Seamless switching between multiple Gemini API keys
- **Quota management**: Handles rate limits and resource exhaustion
- **Resilient processing**: Continues operation with remaining keys

### Vector Search & Filtering
- **Sentence Transformers**: all-MiniLM-L6-v2 for embeddings
- **ChromaDB**: Persistent vector storage with cosine similarity
- **Drug-specific filtering**: ChromaDB metadata filtering by drug names
- **Confidence thresholding**: Configurable relevance scoring (default: 0.6)

## 📁 Project Structure

```
d:\CTS-GROUP-15\
├── test.py                 # Main application file
├── personalized_phrases.py # Personalization detection phrases
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── pdfs/                  # Source PDF documents
│   ├── humira.pdf
│   ├── rinvoq_pi.pdf
│   └── skyrizi_pi.pdf
├── pdf_assets/            # Extracted tables, figures, images
├── chroma_store/          # Vector database persistence
└── ingest_logs.db         # SQLite ingestion tracking
```

## ⚙️ Configuration

### Environment Variables
```env
# Primary API key
GOOGLE_API_KEY=your_primary_key

# Additional keys for rotation (optional)
GOOGLE_API_KEY_1=your_backup_key_1
GOOGLE_API_KEY_2=your_backup_key_2
# ... up to GOOGLE_API_KEY_9

# Redis (optional - falls back to Streamlit session state)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Model Configuration
```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP = 200
CONFIDENCE_THRESHOLD = 0.6
```

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CTS-GROUP-15
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Create .env file with your API keys
   echo "GOOGLE_API_KEY=your_key_here" > .env
   ```

4. **Prepare PDF documents**
   ```bash
   # Place your medical PDFs in the pdfs/ directory
   mkdir pdfs
   # Copy your drug information PDFs
   ```

5. **Run the application**
   ```bash
   streamlit run test.py
   ```

## 📖 Usage Guide

### Initial Setup
1. **Ingest PDFs**: Click "🔄 Ingest PDFs" in the sidebar
2. **Add user information**: Fill out age, weight, symptoms for personalized responses
3. **Upload custom PDFs**: Use the file uploader for additional documents

### Query Types
- **General inquiries**: "What are the side effects of Humira?"
- **Personalized questions**: "Can I use Rinvoq?" (uses your profile)
- **Comparative analysis**: "Compare Humira and Skyrizi side effects"
- **Specific searches**: "Skyrizi dosage for psoriasis"

### Understanding Responses
- **Confidence scores**: 0.8+ (High), 0.6-0.8 (Medium), <0.6 (Low)
- **Citations**: Source file, page, section, and relevance score
- **Agent tracking**: Real-time status of which agent is processing

## 🔍 Agent Workflow

### 1. Query Processing Pipeline
```
User Query → Personalization Check → Entity Extraction → Vector Search → Relevance Filtering → Answer Generation
```

### 2. Real-Time Status Updates
- 🤖 **Initialization**: Multi-agent system startup
- 🔍 **Retrieval**: Entity extraction and vector search
- 🎯 **Drug Detection**: Specific medications identified
- 👤 **Personalization**: User context integration
- 📊 **Reasoning**: Chunk analysis and scoring
- 🧠 **Validation**: Information sufficiency check
- 💬 **Generation**: Structured response creation
- 🎉 **Completion**: Final confidence assessment

### 3. Fallback Mechanisms
- **API key rotation**: Automatic switching on quota exceeded
- **Graceful degradation**: Fallback processing on errors
- **Simple relevance scoring**: Jaccard similarity backup

## 📊 System Monitoring

### Sidebar Information Panel
- **Model configuration**: Current Gemini model
- **API status**: Available keys and current active key
- **System metrics**: Confidence threshold, session ID
- **User profile**: Age, weight, symptoms, uploaded files

### Debug Analysis (Expandable)
1. **Entity Extraction**: Identified drugs and intent
2. **Vector Search Results**: Retrieved chunks with similarity scores
3. **Filtered Retrievals**: Relevance-scored and filtered results
4. **Answer Generation**: Confidence and reasoning details

## 🎯 Personalization Features

### Automatic Detection
The system automatically detects personalized queries using phrases from `personalized_phrases.py`:
- "Can I use..."
- "Should I take..."
- "Is it safe for me..."
- "Based on my..."

### Context Enhancement
When personalized queries are detected:
1. User profile (age, weight, symptoms) is appended to the search query
2. Enhanced query improves vector search relevance
3. User context is passed to the answer generation agent
4. Responses are tailored to the individual's profile

## 🔧 Technical Implementation

### Vector Database (ChromaDB)
- **Persistent storage**: Local ChromaDB instance
- **Metadata filtering**: Drug-specific search filtering
- **Cosine similarity**: Distance-based relevance scoring
- **Batch processing**: Efficient embedding storage

### PDF Processing Pipeline
1. **Text extraction**: PyMuPDF for primary content
2. **Table extraction**: Camelot for structured data
3. **Image processing**: OCR with Pytesseract
4. **Asset organization**: Systematic file management
5. **Deduplication**: Content-based hash checking

### Session Management
- **Redis integration**: Optional persistent storage
- **Streamlit fallback**: Local session state backup
- **History tracking**: Multi-turn conversation support
- **Context preservation**: User profile persistence

## 🚨 Error Handling

### API Management
- Multiple key rotation on quota exceeded
- Graceful degradation with fallback scoring
- Comprehensive error logging and user feedback

### Data Processing
- Robust PDF parsing with fallback mechanisms
- OCR error handling for corrupted images
- Metadata sanitization for ChromaDB compatibility

## 🔮 Future Enhancements

- **Multi-language support**: Extend beyond English
- **Advanced medical NLP**: Disease-specific entity recognition
- **Integration APIs**: EHR and pharmacy system connections
- **Mobile responsiveness**: Enhanced UI/UX for mobile devices
- **Batch processing**: Multiple document ingestion workflows

## 📝 Dependencies

### Core Libraries
```
streamlit              # Web UI framework
crewai                # Multi-agent orchestration
chromadb              # Vector database
sentence-transformers # Text embeddings
google-generativeai   # Gemini AI API
```

### PDF Processing
```
PyMuPDF               # PDF text extraction
pytesseract           # OCR processing
camelot-py            # Table extraction
pdf2image             # PDF to image conversion
Pillow                # Image processing
```

### Data & Storage
```
redis                 # Session management
sqlite3               # Ingestion logging
pandas                # Data manipulation
```


