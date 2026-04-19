# AI Real Estate Copilot

Minimal Streamlit application for real estate document analysis with AI.

## Features
✅ Upload multiple images (PNG, JPG, JPEG)
✅ Upload PDF and text documents
✅ Display uploaded images in grid layout
✅ Extract text from documents
✅ Vector storage with FAISS database
✅ Natural language question answering
✅ Show relevant document sources
✅ Clean, simple interface

## Setup Instructions

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure API Key**
```bash
cp .env.example .env
```
Edit `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

3. **Run the application**
```bash
streamlit run app.py
```

## How to Use
1. Upload images using the image uploader in sideba
2. Upload PDF or text files using the document uploader
3. Click "Process Documents" to extract text and build vector database
4. Ask any question about your uploaded documents in the input box
5. View answers along with relevant document sections

## Technology Stack
- **Streamlit** - Web UI framework
- **LangChain** - LLM orchestration
- **FAISS** - Vector database
- **OpenAI GPT-3.5 Turbo** - LLM for answering questions
- **OpenAI Embeddings** - Text vectorization
- **PyPDF2** - PDF text extraction
