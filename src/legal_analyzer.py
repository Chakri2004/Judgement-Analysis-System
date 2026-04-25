from src.case_extractor import extract_text_from_pdf
from src.rag_engine import retrieve_relevant_laws
import re
import pandas as pd

try:
    ipc_data = pd.read_csv("data/ipc_sections.csv")
except Exception:
    ipc_data = pd.DataFrame()


def extract_case_name(text):
    """
    Try to detect case name from judgment text
    """
    lines = text.split("\n")
    for line in lines[:50]:
        line_lower = line.lower()
        if " vs " in line_lower or " v. " in line_lower or " vs. " in line_lower:
            return line.strip()
        if "petitioner" in line_lower and "respondent" in line_lower:
            return line.strip()
    return "Case Name Not Found"


def analyze_case(pdf_path):
    """
    FIX: Removed nested return bug — output is always returned at the end.
    Removed duplicate LAW_DOMAIN_MAP — now uses law_domains.py only.
    """
    case_text = extract_text_from_pdf(pdf_path)
    case_name = extract_case_name(case_text)

    results = retrieve_relevant_laws(case_text, k=5)
    ipc_sections = detect_ipc_sections(case_text)
    issues = extract_legal_issues(case_text)

    # Import domain from central source only
    from src.law_domains import LAW_DOMAIN_MAP, KEYWORD_DOMAIN_MAP
    domain = detect_domain_from_text(case_text, LAW_DOMAIN_MAP, KEYWORD_DOMAIN_MAP)

    sections = detect_sections(case_text)
    citations = extract_case_citations(case_text)

    if case_name == "Case Name Not Found" and len(results) > 0:
        case_name = results[0]["title"]

    detected_acts = []
    legal_provisions = []
    similar_cases = []

    for r in results:
        text = r["content"].lower()
        if "ipc" in text:
            detected_acts.append("Indian Penal Code")
        if "civil procedure" in text:
            detected_acts.append("Code of Civil Procedure")
        if "section" in text:
            legal_provisions.append(r["title"])
        similar_cases.append(r["title"])

    output = {
        "domain": domain,
        "case_name": case_name,
        "citations": citations,
        "detected_acts": (
            ["Indian Penal Code"] if ipc_sections
            else list(set(detected_acts))
        ),
        "legal_provisions": (
            sections if sections
            else (ipc_sections if ipc_sections else legal_provisions[:3])
        ),
        "legal_issues": (
            issues if issues
            else [
                "Legal dispute between parties",
                "Application of law",
                "Judicial interpretation required"
            ]
        ),
        "similar_judgements": similar_cases[:3]
    }

    return output


def detect_ipc_sections(text):
    detected_sections = []
    text_lower = text.lower()

    if ipc_data.empty:
        return []

    ipc_map = {
        str(row["Offense"]).lower(): str(row["Section"])
        for _, row in ipc_data.iterrows()
        if "Offense" in ipc_data.columns and "Section" in ipc_data.columns
    }
    for offense, section in ipc_map.items():
        if offense and offense in text_lower:
            detected_sections.append(section)

    return list(set(detected_sections))[:3]


def extract_case_citations(text):
    """
    Extract common Indian case citation formats
    """
    patterns = [
        r"\(\d{4}\)\s*\d+\s*SCC\s*\d+",
        r"AIR\s*\d{4}\s*SC\s*\d+",
        r"\d{4}\s*SCC\s*Online\s*\d+",
        r"\(\d{4}\)\s*\d+\s*SCR\s*\d+"
    ]
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)

    return list(set(citations))[:5]


def extract_legal_issues(text):
    issues = []
    text_lower = text.lower()

    issue_map = [
        (["property", "land", "title", "sale deed"], "Property ownership dispute"),
        (["sale deed"], "Validity of sale deed"),
        (["breach of contract", "contract breach"], "Breach of contract"),
        (["murder", "kill", "culpable homicide"], "Intent to cause death"),
        (["fraud", "cheating", "dishonest"], "Financial fraud or cheating"),
        (["assault", "grievous hurt", "hurt"], "Physical assault"),
        (["divorce", "matrimonial", "maintenance"], "Matrimonial dispute"),
        (["cheque bounce", "dishonour", "section 138"], "Cheque dishonour"),
        (["consumer", "deficiency", "service deficiency"], "Consumer deficiency of service"),
        (["pocso", "minor", "child abuse"], "Child sexual offence (POCSO)"),
        (["arbitration", "award"], "Arbitration dispute"),
        (["tax", "income tax", "gst"], "Tax dispute"),
        (["dispute"], "Legal dispute between parties"),
    ]

    for keywords, label in issue_map:
        if any(k in text_lower for k in keywords):
            if label not in issues:
                issues.append(label)
        if len(issues) >= 3:
            break

    return issues[:3]


def detect_domain_from_text(text, law_domain_map, keyword_domain_map):
    """
    FIX: Uses weighted scoring same as app.py classify_main_category.
    No more first-match-wins — uses the central maps from law_domains.py.
    """
    text_lower = text.lower()

    # Act-based detection first (highest priority)
    act_patterns = {
        "POCSO Act": r"\b(POCSO|Protection of Children from Sexual Offences Act)\b",
        "Information Technology Act": r"\b(IT Act|Information Technology Act)\b",
        "Negotiable Instruments Act": r"\bNegotiable Instruments Act\b",
        "Consumer Protection Act": r"\bConsumer Protection Act\b",
        "Hindu Marriage Act": r"\bHindu Marriage Act\b",
        "Indian Penal Code": r"\b(IPC|Indian Penal Code)\b",
        "Code of Civil Procedure": r"\b(CPC|Code of Civil Procedure)\b",
        "Companies Act": r"\bCompanies Act\b",
        "Income Tax Act": r"\bIncome Tax Act\b",
        "Arbitration and Conciliation Act": r"\bArbitration and Conciliation Act\b",
        "Constitution of India": r"\b(Constitution of India|Article \d+)\b",
        "Industrial Disputes Act": r"\bIndustrial Disputes Act\b",
        "Transfer of Property Act": r"\bTransfer of Property Act\b",
        "Indian Contract Act": r"\bIndian Contract Act\b",
        "Insolvency and Bankruptcy Code": r"\b(IBC|Insolvency and Bankruptcy Code)\b",
    }

    import re as _re
    for act, pattern in act_patterns.items():
        if _re.search(pattern, text, _re.IGNORECASE):
            domain = law_domain_map.get(act)
            if domain:
                return domain

    domain_scores = {}
    for domain, keywords in keyword_domain_map.items():
        score = 0
        for word in keywords:
            if word in text_lower:
                score += 2 if " " in word else 1
        if score > 0:
            domain_scores[domain] = score

    if domain_scores:
        best = max(domain_scores, key=domain_scores.get)
        if domain_scores[best] >= 2:
            return best

    return "General Law"

def detect_sections(text):
    sections = re.findall(r"section\s+\d+[a-z]*", text.lower())
    formatted_sections = []
    for s in sections:
        num = s.split()[1]
        formatted_sections.append(f"Section {num}")
    return list(set(formatted_sections))[:3]