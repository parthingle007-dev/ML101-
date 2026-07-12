import os
import pickle
import faiss
import google.generativeai as genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

llm = genai.GenerativeModel("gemini-flash-latest")

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index("faiss_index/index.faiss")

with open("faiss_index/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


SIMILARITY_THRESHOLD = 0.45


def ask(question, k=10):


    query_embedding = embed_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)


    similarities, indices = index.search(query_embedding, k)

    best_similarity = float(similarities[0][0])

    print(f"\nBest Similarity: {best_similarity:.3f}")

    if best_similarity < SIMILARITY_THRESHOLD:
        return (
            "I don't know based on the available documents.",
            []
        )

    retrieved_chunks = []
    sources = []

    for similarity, idx in zip(similarities[0], indices[0]):

        if idx == -1:
            continue

        if similarity < 0.35:
            continue

        chunk = metadata[idx]


        if len(chunk["text"].strip()) < 40:
            continue

    retrieved_chunks.append(chunk["text"])

    sources.append({
        "source": chunk["source"],
        "page": chunk["page"],
        "similarity": float(similarity)
    })

    if not retrieved_chunks:
        return (
            "I don't know based on the available documents.",
            []
        )

    context = "\n\n".join(retrieved_chunks)
    print("\n====================")
    print(context)
    print("====================\n")


    prompt = f"""
You are IITB Insti-Assist.

Answer ONLY using the provided context.

Rules:

1. Never use outside knowledge.

2. If the answer is not explicitly present in the context, reply EXACTLY:

I don't know based on the available documents.

3. Do not guess.

4. Do not infer.

5. Keep the answer concise (2-5 sentences).

6. If multiple retrieved chunks disagree, mention the ambiguity instead of choosing one.

Context:
{context}

Question:
{question}

Answer:
"""


    response = llm.generate_content(prompt)

    answer = response.text.strip()

    if not answer:
        return (
            "I don't know based on the available documents.",
            []
        )

    unsure_phrases = [
        "not enough information",
        "insufficient information",
        "cannot determine",
        "does not mention",
        "not mentioned",
        "not provided",
        "unable to determine"
    ]

    if any(p in answer.lower() for p in unsure_phrases):
        return (
            "I don't know based on the available documents.",
            []
        )

    return answer, sources


if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (type exit to quit): ")

        if question.lower() == "exit":
            break

        answer, sources = ask(question)

        print("\nAnswer:\n")
        print(answer)

        if sources:

            print("\nSources Used:")

            seen = set()

            for s in sources:

                key = (s["source"], s["page"])

                if key not in seen:

                    print(
                        f"• {s['source']} (Page {s['page']}) "
                        f"[Similarity: {s['similarity']:.3f}]"
                    )

                    seen.add(key)

        else:

            print("\nNo supporting document found.")