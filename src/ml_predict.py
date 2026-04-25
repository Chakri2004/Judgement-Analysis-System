import pickle
from src.embedding_model import get_embedding

model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

def predict_domain_ml(text):
    emb = get_embedding(text)
    pred = model.predict([emb])
    return label_encoder.inverse_transform(pred)[0]