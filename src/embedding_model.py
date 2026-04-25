from sentence_transformers import SentenceTransformer

model = SentenceTransformer("law-ai/InLegalBERT")

def get_embedding(text):
    return model.encode(text, normalize_embeddings=True)