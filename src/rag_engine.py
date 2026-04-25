import faiss
import pickle
import numpy as np
import os
from src.embedding_model import get_embedding
from src.dataset_loader import load_cases, detect_domain_from_text
from src.llm_client import generate_llm_response

BASE_DIR = os.path.dirname(__file__)

print("Loading embedding model...")

index_path = os.path.join(BASE_DIR, "legal_data/legal_index.faiss")
text_path = os.path.join(BASE_DIR, "legal_data/legal_texts.pkl")

if os.path.exists(index_path) and os.path.exists(text_path):
    print("Loading existing FAISS database...")
    index = faiss.read_index(index_path)
    with open(text_path, "rb") as f:
        cases = pickle.load(f)
else:
    print("Creating FAISS database...")
    cases = load_cases()[:8000]
    texts = [case["text"] for case in cases]

    def batch_embeddings(texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings.extend(get_embedding(batch))
        return embeddings

    embeddings = batch_embeddings(texts)
    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 40
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, index_path)
    with open(text_path, "wb") as f:
        pickle.dump(cases, f)

print("Total cases loaded:", len(cases))

def classify_query(query):
    q = query.lower()
    if "section" in q:
        return "legal_section"
    elif "case" in q:
        return "case_lookup"
    return "general"


def ask(question):
    """
    Answer a legal question using RAG + LLM.
    """

    category = detect_domain_from_text(question)
    docs = retrieve_relevant_laws(question, category=category)

    if not docs:
        return "No relevant legal cases found."

    context = "\n\n".join([doc["content"] for doc in docs[:3]])

    prompt = f"""You are a professional legal assistant.

STRICT RULES:
- Answer ONLY using the provided context
- If answer is not present, say: "Not found in provided case data"
- Do NOT assume or fabricate legal facts

Use:
- Headings
- Bullet points

Context:
{context}

Question:
{question}

Give a helpful, easy-to-understand legal answer."""

    answer = generate_llm_response(prompt)
    return answer


def retrieve_relevant_laws(query_text, k=10, category=None):
    if not query_text or not query_text.strip():
        return []

    query_text = query_text[:1500]
    try:
        query_embedding = get_embedding(query_text)
        if query_embedding is None:
            return []
        query_embedding_array = np.array([query_embedding]).astype("float32")
    except Exception as e:
        print("Embedding error:", e)
        return []

    try:
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = 50
        distances, indices = index.search(query_embedding_array, k)
    except Exception as e:
        print("FAISS search error:", e)
        return []

    results = []
    seen_titles = set()
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(cases):
            continue

        case = cases[idx]
        text = case.get("text", "")
        if not text:
            continue

        distance = distances[0][i]
        similarity = round(100 / (1 + distance), 2)
        if category and case.get("category"):
            if case.get("category") == category:
                similarity += 10
            else:
                similarity -= 5

        if similarity < 20:
            continue

        title_line = ""
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 10:
                title_line = line[:120]
                break
        if not title_line:
            title_line = text[:80]

        if title_line in seen_titles:
            continue
        seen_titles.add(title_line)

        confidence = min(100, similarity * 1.2)

        results.append({
            "title": title_line,
            "case_name": title_line,
            "content": text,
            "summary": text[:300],
            "main_category": case.get("category", "Supreme Court Judgment"),
            "similarity": similarity,
            "confidence": confidence,
            "_id": None,
        })

    print(f"Retrieved {len(results)} results for query (category={category})")

    results.sort(key=lambda x: x["similarity"], reverse=True)

    top_results = results[:20]

    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    top_results = sorted(
        top_results,
        key=lambda x: cosine_similarity(
            query_embedding,
            get_embedding(x["content"])
        ),
        reverse=True
    )

    return top_results[:k]