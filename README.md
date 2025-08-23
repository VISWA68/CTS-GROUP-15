# Drug Information Chatbot (RAG + Crew AI)

This project is a complete RAG-based Drug Information Chatbot that uses Crew AI for agent orchestration. It integrates PDF ingestion, retrieval, reasoning, and answer generation using Gemini AI, with confidence-based filtering and improved PDF extraction (tables, figures, OCR, heading detection).

## Features
- Modular agent-based architecture (Crew AI)
- PDF ingestion with table, figure, and OCR extraction
- Vector search and entity extraction
- Reasoning and confidence-based filtering
- Structured answer generation using Gemini AI
- Session management with Redis or Streamlit
- Streamlit UI with agent status display and current running agent indicator

## How to Run
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Set your `GOOGLE_API_KEY` in a `.env` file:
	```env
	GOOGLE_API_KEY=your_google_api_key_here
	```
3. Place your PDF files in the `pdfs/` directory.
4. Run the app:
	```bash
	streamlit run app.py
	```

## Requirements
See `requirements.txt` for all dependencies.

## Agent Orchestration
Agents are managed using Crew AI:
- **Ingestion Agent**: Extracts and chunks data from PDFs
- **Retrieval Agent**: Performs vector search and entity extraction
- **Reasoning Agent**: Assesses relevance and filters chunks
- **Answer Agent**: Generates structured medical answers
- **Session Agent**: Manages user session and chat history

## UI Features
- Displays current running agent during query processing
- Expandable analysis process with agent details
- Chat history and session management
