# 💡 Clarix — Document Q&A Application

A document-based question answering application built for the SenzMate internship assignment. Upload any PDF or TXT file and ask questions about its content using AI.

---

##  Features

### Core Features (Assignment Requirements)
- **Document Upload** — Upload PDF or TXT files up to 200MB
- **Text Processing** — Automatic text extraction and intelligent chunking
- **Question Input** — Natural language chat interface to ask questions
- **Answer Generation** — Accurate AI-powered answers using RAG pipeline
- **User Interface** — Clean and intuitive Streamlit web interface

### Bonus Features
- **Quiz Generator** — Auto-generate MCQ questions from document content
- **Flashcard Generator** — Create interactive study flashcards
- **Session History** — Uploaded documents and chats saved automatically
- **Dark / Light Mode** — Toggle between themes

---

##  Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq API (LLaMA 3.3 70B) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| PDF Parsing | PyMuPDF (fitz) |
| Text Chunking | tiktoken |

---

##  Project Structure
clarix/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── README.md                 # Documentation
└── utils/
├── init.py
├── pdf_reader.py         # PDF and TXT text extraction
├── chunker.py            # Text chunking with overlap
├── embedder.py           # FAISS vector index and retrieval
├── qa_engine.py          # Answer generation using Groq LLM
├── quiz_generator.py     # Bonus: MCQ quiz generation
└── flashcard_generator.py  # Bonus: Flashcard generation

---

##  Setup Instructions

### 1. Clone the repository
git clone https://github.com/YOURUSERNAME/clarix-doc-qa.git
cd clarix-doc-qa

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set up environment variables
Create a `.env` file in the root folder:
GROQ_API_KEY=your_groq_api_key_here

Get your free Groq API key at: https://console.groq.com

### 5. Run the application
streamlit run app.py

Open your browser at http://localhost:8501

---

##  Requirements
streamlit
groq
python-dotenv
PyMuPDF
faiss-cpu
sentence-transformers
tiktoken

---

##  How It Works

1. **Upload** — User uploads a PDF or TXT document
2. **Extract** — PyMuPDF extracts text from the document
3. **Chunk** — Text is split into overlapping chunks (300 tokens, 50 overlap)
4. **Embed** — SentenceTransformers converts chunks to vector embeddings
5. **Index** — FAISS stores embeddings for fast similarity search
6. **Query** — User asks a question about the document
7. **Retrieve** — Top 5 most relevant chunks retrieved via FAISS
8. **Answer** — Groq LLaMA 3.3 70B generates accurate answer

---

##  Evaluation Criteria

-  Functionality — Document upload, Q&A working correctly
-  Code Quality — Modular structure, commented code
-  User Interface — Clean professional Streamlit UI
-  Documentation — README with full setup instructions

---
## Important Note

To run this application you need a free Groq API key.

1. Go to https://console.groq.com
2. Sign up and create an API key
3. Create a `.env` file in the project root
4. Add this line:
GROQ_API_KEY=your_groq_api_key_here

The API is completely free to use.

##  Author

**Kajitha Jeyakanthan**
Internship Application — SenzMate