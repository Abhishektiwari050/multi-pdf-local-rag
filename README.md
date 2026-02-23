# Multi-PDF Local RAG Assistant

This application allows you to upload multiple PDF research papers and ask questions about their content. The system leverages **PyMuPDF** to extract text and bounding boxes locally at zero cost, and uses **Google Gemini (2.5 Flash)** to generate accurate answers heavily grounded in visual evidence.

## Features

- **Zero-Cost Local Extraction**: Parses PDFs locally using PyMuPDF (no Landing AI or paid external document parsing APIs needed).
- **Multiple PDF Support**: Upload several PDF files simultaneously.
- **Question Answering**: Ask questions about the content of your PDFs across multiple documents.
- **Visual Evidence**: See exactly where in the document the answers come from. The app dynamically renders the specific PDF pages and draws green bounding boxes around the extracted evidence.
- **Reasoning Transparency**: Includes detailed reasoning explaining how the LLM derived its answer.
- **Chat History**: Maintains your conversation history for easy reference within the session.
- **Modern Sidebar UI**: A clean, streamlined Streamlit interface utilizing native chat components.

## Architecture & Data Flow

1. **PDF Upload**: Users upload PDFs via the Streamlit sidebar.
2. **Local Processing**: `local-app.py` uses PyMuPDF (`fitz`) to iterate through every page, extracting text blocks and their normalized bounding box coordinates.
3. **Session Storage**: The raw PDF bytes and extracted chunk data are cached in the Streamlit session state to prevent reprocessing.
4. **Querying**: The user submits a question. The application bundles the question and the extracted JSON chunks and sends a prompt to the Google Gemini API.
5. **JSON Response**: Gemini is instructed to return a strict JSON schema containing the `answer`, `reasoning`, and an array of `best_chunks` (specifying the exact file, page, and coordinates used).
6. **Visual Grounding**: The application correlates the `best_chunks` back to the original PDF pages, uses OpenCV to draw bounding rectangles, and displays the annotated image directly in the chat interface.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Abhishektiwari050/multi-pdf-local-rag.git
cd multi-pdf-local-rag
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

3. Configure your API Keys:
Create a `.env` file in the project root directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run local-app.py
```

2. Open the provided URL in your web browser (typically `http://localhost:8501`).
3. Upload one or more PDF research papers using the sidebar.
4. The system will precompute the evidence from the PDFs locally.
5. Enter your question in the chat input box.
6. View the answer, and expand the "Show Reasoning & Details" accordion to see exactly where the assistant pulled the information from!

## Dependencies

- `streamlit`: Web application framework and UI
- `google-genai`: Official Google Gemini SDK for LLM interactions
- `PyMuPDF (fitz)`: Fast, local PDF parsing and rendering
- `opencv-python`: Image processing and bounding box drawing
- `python-dotenv`: Environment variable management
- `numpy` & `Pillow`: Image array handling

