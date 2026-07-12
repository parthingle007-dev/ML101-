# IITB Insti-Assist

A simple RAG (Retrieval-Augmented Generation) chatbot that answers questions about IIT Bombay using official institute documents.

This project was made as the final assignment for the Learners' Space NLP course.

---

## Features

- Answers questions using IIT Bombay documents
- Uses FAISS for document retrieval
- Uses Sentence Transformers for embeddings
- Uses Gemini API to generate answers
- Shows the document and page used for the answer
- Says **"I don't know based on the available documents."** when the answer is not present

---

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Gemini API
- pdfplumber

---

## Project Structure

```
.
├── app.py                # Streamlit UI
├── rag.py                # Retrieval + Gemini
├── build_index.py        # Builds FAISS index
├── data/                 # IITB PDF documents
├── faiss_index/          # Generated FAISS index
├── requirements.txt
└── README.md
```

---

## How it Works

1. Read all PDF documents.
2. Split them into small text chunks.
3. Convert each chunk into an embedding.
4. Store embeddings in a FAISS index.
5. When a question is asked:
   - Embed the question.
   - Retrieve the most similar chunks.
   - Send those chunks to Gemini.
   - Display the answer and source documents.

---

## Documents Used

The assistant uses official IIT Bombay documents such as:

- UG Rule Book
- Academic Calendar
- UG New Entrants Guide
- MCM Scholarship Guide
- Free Messing Guidelines
- Scholarship Documents
- Grading Statistics Report

---

## Setup

Clone the repository

```bash
git clone <repo-link>
cd IITB-Insti-Assist
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
GEMINI_API_KEY=your_api_key
```

Build the FAISS index

```bash
python build_index.py
```

Run the app

```bash
streamlit run app.py
```

---

## Example

**Question**

```
What is a minor?
```

**Answer**

The chatbot retrieves the relevant document, generates an answer from it, and shows the source PDF and page number.

---

## Limitations

- Works only on the documents added to the knowledge base.
- Some answers may be missed if the retrieval step does not find the correct chunk.
- PDF formatting can affect text extraction.
- The chatbot cannot answer questions outside the available documents.

---

## Future Improvements

- Better chunking strategy.
- Use a stronger embedding model.
- Add reranking for better retrieval.
- Support more IIT Bombay documents.
- Deploy the application online.

---

## Author

Parth Ingle  
Second Year Undergraduate  
IIT Bombay
