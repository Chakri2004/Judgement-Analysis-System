import os
import fitz
import re


def chunk_text(text, max_tokens=300):
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_tokens:
            current += " " + para
        else:
            chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    return chunks


def extract_sections(text):
    """
    Extract legal section references (IPC / CrPC / POCSO etc.)
    """
    matches = []
    patterns = [
        r"section\s+\d+[a-z]*",
        r"\b\d+\s*ipc\b",
        r"\b\d+\s*crpc\b",
        r"\b\d+\s*cpc\b",
        r"u/s\s*\d+",
    ]

    for pattern in patterns:
        matches.extend(re.findall(pattern, text.lower()))

    return list(set(matches))


def detect_domain_from_text(text):
    """
    FIX: Uses weighted scoring instead of first-keyword-wins.
    Multi-word phrases score 2x single words.
    Minimum score of 2 required to avoid noise matches.
    Uses the central KEYWORD_DOMAIN_MAP from law_domains.py.
    """
    from src.law_domains import KEYWORD_DOMAIN_MAP

    text_lower = text.lower()

    # Act-based detection (highest priority, most specific)
    act_patterns = {
        "Child Protection (POCSO)": r"\b(POCSO|Protection of Children from Sexual Offences Act)\b",
        "Cyber Law": r"\b(IT Act|Information Technology Act)\b",
        "Banking Law": r"\bNegotiable Instruments Act\b",
        "Consumer Protection": r"\bConsumer Protection Act\b",
        "Family Law": r"\bHindu Marriage Act\b",
        "Criminal Law": r"\b(IPC|Indian Penal Code|Bharatiya Nyaya Sanhita)\b",
        "Civil Law": r"\b(CPC|Code of Civil Procedure)\b",
        "Company Law": r"\bCompanies Act\b",
        "Tax Law": r"\bIncome Tax Act\b",
        "Arbitration Law": r"\bArbitration and Conciliation Act\b",
        "Constitutional Law": r"\bConstitution of India\b",
        "Labour Law": r"\bIndustrial Disputes Act\b",
        "Property Law": r"\bTransfer of Property Act\b",
        "Contract Law": r"\bIndian Contract Act\b",
        "Corporate Law": r"\b(IBC|Insolvency and Bankruptcy Code)\b",
        "Environmental Law": r"\bEnvironment Protection Act\b",
        "Insurance Law": r"\bInsurance Act\b",
        "Competition Law": r"\bCompetition Act\b",
        "Intellectual Property Law": r"\b(Copyright Act|Trademark Act|Patents Act)\b",
    }

    for domain, pattern in act_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return domain

    # Weighted keyword scoring
    domain_scores = {}
    for domain, keywords in KEYWORD_DOMAIN_MAP.items():
        score = 0
        for word in keywords:
            if word in text_lower:
                # Multi-word phrases are more specific — weight them higher
                score += 2 if " " in word else 1
        if score > 0:
            domain_scores[domain] = score

    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        # Minimum threshold to avoid noise
        if domain_scores[best_domain] >= 2:
            return best_domain

    return "General Law"


def load_cases():
    """
    Load Supreme Court judgments from data directory.
    Each PDF is chunked and domain-tagged.
    """
    base_path = "data/supreme_court_judgments"

    if not os.path.exists(base_path):
        print(f"Warning: {base_path} not found. No cases loaded.")
        return []

    cases = []

    for year in os.listdir(base_path):
        year_path = os.path.join(base_path, year)
        if not os.path.isdir(year_path):
            continue

        for file in os.listdir(year_path):
            if not file.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(year_path, file)
            try:
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        text += page_text
                doc.close()

                if len(text) < 200:
                    continue

                chunks = chunk_text(text)
                for chunk in chunks:
                    if len(chunk) < 100:
                        continue
                    cases.append({
                        "text": chunk,
                        "sections": extract_sections(chunk),
                        "category": detect_domain_from_text(chunk),
                        "source_file": file,
                        "year": year,
                    })

            except Exception as e:
                print(f"Skipped {file}: {e}")

    print(f"\nTotal chunks created: {len(cases)}")

    if cases:
        print("\nSample case:")
        print(f"  Category: {cases[0].get('category')}")
        print(f"  Text preview: {cases[0]['text'][:100]}")

    return cases