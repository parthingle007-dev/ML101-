import os
import pickle
import faiss
import numpy as np
import pdfplumber

from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
INDEX_FOLDER = "faiss_index"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 250

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

documents = []

print("\nReading PDFs...\n")

for filename in os.listdir(DATA_FOLDER):

    if filename.endswith(".pdf"):

        path = os.path.join(DATA_FOLDER, filename)

        print(f"Reading {filename}")

        with pdfplumber.open(path) as pdf:

            for page_num, page in enumerate(pdf.pages):

                text = page.extract_text()

                if text:

                    documents.append({
                        "text": text,
                        "source": filename,
                        "page": page_num + 1
                    })

print(f"\nLoaded {len(documents)} pages.")

print("\nChunking documents...\n")

chunks = []

for doc in documents:

    paragraphs = doc["text"].split("\n\n")

    for para in paragraphs:

        para = para.strip()

        if len(para) < 40:
            continue

        if len(para) <= CHUNK_SIZE:

            chunks.append({
                "text": para,
                "source": doc["source"],
                "page": doc["page"]
            })

        else:

            start = 0

            while start < len(para):

                end = start + CHUNK_SIZE

                chunk = para[start:end]

                chunks.append({
                    "text": chunk,
                    "source": doc["source"],
                    "page": doc["page"]
                })

                start += CHUNK_SIZE - CHUNK_OVERLAP

print(f"Created {len(chunks)} chunks.")

print("\nGenerating embeddings...\n")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype("float32")

faiss.normalize_L2(embeddings)

print("\nEmbeddings generated.")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

os.makedirs(INDEX_FOLDER, exist_ok=True)

faiss.write_index(
    index,
    os.path.join(INDEX_FOLDER, "index.faiss")
)

with open(
    os.path.join(INDEX_FOLDER, "metadata.pkl"),
    "wb"
) as f:

    pickle.dump(chunks, f)

print("\n==============================")
print("Index built successfully!")
print("==============================")
print(f"Documents : {len(documents)}")
print(f"Chunks    : {len(chunks)}")
print(f"Vectors   : {index.ntotal}")
print(f"Saved to  : {INDEX_FOLDER}")