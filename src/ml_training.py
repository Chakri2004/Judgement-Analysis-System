import numpy as np
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import top_k_accuracy_score

from xgboost import XGBClassifier

from src.dataset_loader import load_cases
from src.legal_analyzer import detect_domain_from_text
from src.law_domains import LAW_DOMAIN_MAP, KEYWORD_DOMAIN_MAP
from src.embedding_model import get_embedding


# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
cases = load_cases()

allowed_domains = [
    "Civil Law",
    "Contract Law",
    "Criminal Law",
    "Family Law",
    "Property Law"
]

# =========================
# FILTER DATA
# =========================
print("Filtering data...")
texts = []

for case in cases:
    text = case.get("text", "")

    if not text or len(text) < 50:
        continue

    domain = detect_domain_from_text(
        text,
        LAW_DOMAIN_MAP,
        KEYWORD_DOMAIN_MAP
    )

    if domain not in allowed_domains:
        continue

    texts.append((text, domain))

print("Total filtered samples:", len(texts))


# =========================
# BATCH EMBEDDINGS
# =========================
batch_size = 32
X = []
labels = []

print("Generating embeddings (batch mode)...")

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]

    batch_texts = [t[0] for t in batch]
    batch_domains = [t[1] for t in batch]

    embeddings = get_embedding(batch_texts)

    X.extend(embeddings)
    labels.extend(batch_domains)

    if i % 5000 == 0:
        print(f"Processed {i} samples...")

# Convert to numpy (memory optimized)
X = np.array(X, dtype=np.float32)

print("Final dataset size:", len(X))


# =========================
# ENCODING
# =========================
le = LabelEncoder()
y = le.fit_transform(labels)


# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

weights = {i: class_weights[i] for i in range(len(class_weights))}
sample_weights = [weights[label] for label in y_train]


# =========================
# MODEL
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss'
)


# =========================
# TRAINING
# =========================
print("Training model...")
start = time.time()

model.fit(X_train, y_train, sample_weight=sample_weights)

end = time.time()

print("Training time:", round(end - start, 2), "seconds")


# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)


# =========================
# METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

top3 = top_k_accuracy_score(
    y_test,
    y_pred_prob,
    k=3,
    labels=range(len(le.classes_))
)

print("\n===== RESULTS =====")
print("Accuracy:", round(acc * 100, 2), "%")
print("F1 Score:", round(f1, 3))
print("Top-3 Accuracy:", round(top3, 3))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# =========================
# SAVE MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("\nModel saved successfully!")


# =========================
# CONFUSION MATRIX
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.savefig("confusion_matrix.png")
plt.show()


# =========================
# ROC CURVE
# =========================
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))

plt.figure(figsize=(10, 8))

for i in range(len(le.classes_)):
    if np.sum(y_test_bin[:, i]) == 0:
        continue

    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{le.classes_[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()