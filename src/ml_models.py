import re
import spacy

# Load spacy model once at module level
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spacy en_core_web_sm model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


def extract_case_name(text):
    """
    Extract case name from judgment text.
    Looks for party vs party patterns in first 50 lines.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    appellant = None
    respondent_name = None

    for i, line in enumerate(lines):
        if "appellant" in line.lower() and i > 0:
            appellant = lines[i - 1]
        if "respondent" in line.lower() and i > 0:
            respondent_name = lines[i - 1]
        if appellant and respondent_name:
            case = f"{appellant.title()} vs {respondent_name.title()}"
            case = re.sub(r"\(.*?slp.*?\)", "", case, flags=re.I)
            return case.strip()

    for i, line in enumerate(lines):
        if line.lower() == "versus" and i > 0 and i < len(lines) - 1:
            party1 = lines[i - 1]
            party2 = lines[i + 1]
            case = f"{party1.title()} vs {party2.title()}"
            case = re.sub(r"\(.*?slp.*?\)", "", case, flags=re.I)
            return case.strip()

    # Search first 50 lines only for vs pattern
    for line in lines[:50]:
        match = re.search(r'(.+?)\s+vs?\.?\s+(.+)', line, re.IGNORECASE)
        if match:
            p1 = match.group(1).strip()
            p2 = match.group(2).strip()
            # Skip if it looks like a section reference
            if any(x in p1.lower() for x in ["section", "article", "order", "rule"]):
                continue
            if len(p1) > 3 and len(p2) > 3:
                case = f"{p1.title()} vs {p2.title()}"
                case = re.sub(r"\(.*?slp.*?\)", "", case, flags=re.I)
                return case.strip()

    return "Unknown Case"


def extract_court_name(text):
    """
    Extract court name from judgment text.
    Returns the most specific court found.
    """
    courts = [
        "NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION",
        "STATE CONSUMER DISPUTES REDRESSAL COMMISSION",
        "DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION",
        "NATIONAL COMPANY LAW TRIBUNAL",
        "DEBT RECOVERY TRIBUNAL",
        "INCOME TAX APPELLATE TRIBUNAL",
        "CENTRAL ADMINISTRATIVE TRIBUNAL",
        "ARMED FORCES TRIBUNAL",
        "SUPREME COURT OF INDIA",
        "DELHI HIGH COURT",
        "BOMBAY HIGH COURT",
        "MADRAS HIGH COURT",
        "KARNATAKA HIGH COURT",
        "CALCUTTA HIGH COURT",
        "ALLAHABAD HIGH COURT",
        "KERALA HIGH COURT",
        "GUJARAT HIGH COURT",
        "RAJASTHAN HIGH COURT",
        "PUNJAB AND HARYANA HIGH COURT",
        "ANDHRA PRADESH HIGH COURT",
        "TELANGANA HIGH COURT",
        "HIGH COURT",
        "SESSIONS COURT",
        "FAMILY COURT",
        "DISTRICT COURT",
        "TRIBUNAL",
    ]
    text_upper = text.upper()
    for court in courts:
        if court in text_upper:
            return court.title()
    return "Court Not Identified"


def extract_judge_name(text):
    """
    Extract judge name(s) from judgment text.
    Handles Hon'ble, Justice, Coram patterns.
    """
    patterns = [
        r"hon'?ble\s+mr\.?\s+justice\s+([A-Za-z\.\s]+)",
        r"hon'?ble\s+ms\.?\s+justice\s+([A-Za-z\.\s]+)",
        r"hon'?ble\s+mrs\.?\s+justice\s+([A-Za-z\.\s]+)",
        r"hon'?ble\s+dr\.?\s+justice\s+([A-Za-z\.\s]+)",
        r"justice\s+([A-Za-z\.\s]{4,40}?)(?:\s*,|\s*\n|\s*and\s|\s*$)",
        r"coram\s*[:\-]?\s*([A-Za-z\.\s,]{4,60}?)(?:\n|$)",
        r"before\s*[:\-]?\s*(?:hon'?ble\s+)?([A-Za-z\.\s,]{4,60}?)(?:\n|$)",
    ]
    judges = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            name = m.strip()
            name = re.sub(r"\s+", " ", name)
            name = re.sub(r"[,;]+$", "", name)
            # Filter noise
            if len(name) < 4 or len(name) > 60:
                continue
            if any(x in name.lower() for x in ["section", "order", "rule", "act", "page"]):
                continue
            judges.append(name.title())

    judges = list(dict.fromkeys(judges))  # deduplicate preserving order
    # Clean unwanted words
    cleaned_judges = []
    for name in judges:
        name = re.sub(r"\b(justice|hon'?ble|judgment)\b", "", name, flags=re.I)
        name = name.replace("J.", "")
        name = name.replace("J", "")
        name = re.sub(r"\s+", " ", name).strip()

        # Remove repeated long names
        parts = name.split()
        if len(parts) > 2:
            name = " ".join(parts[:2])

        if len(name) >= 4:
            cleaned_judges.append(name)

    # Final deduplication
    cleaned_judges = list(dict.fromkeys(cleaned_judges))

    if cleaned_judges:
        return cleaned_judges[0]

    return "Judge Not Identified"


def extract_legal_sections(text):
    """
    Extract legal section references from text.
    Returns sorted unique list.
    """
    pattern = r"(Section\s+\d+[A-Za-z]*\s*(?:IPC|CrPC|BNSS|CPC|POCSO|IT\s*Act)?|Article\s+\d+[A-Za-z]*|Rule\s+\d+[A-Za-z]*)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    sections = []
    for m in matches:
        if isinstance(m, tuple):
            sections.append(m[0].strip())
        else:
            sections.append(m.strip())
    sections = list(set(sections))
    return sorted(sections)


def extract_parties(case_name_line):
    """
    FIX: Now only called on the case name line, not full text.
    Extracts petitioner and respondent from 'X vs Y' format.
    Cleans up common artifacts like case numbers, SLP references.
    """
    if not case_name_line or case_name_line in ["Unknown Case", "Case Name Not Found"]:
        return None, None

    # Remove common artifacts before extracting
    clean = case_name_line
    clean = re.sub(r"\(arising out of.*?\)", "", clean, flags=re.I)
    clean = re.sub(r"SLP\s*\(.*?\)\s*No\.?\s*[\d/]+", "", clean, flags=re.I)
    clean = re.sub(r"Civil Appeal No\.?\s*[\d/]+", "", clean, flags=re.I)
    clean = re.sub(r"Criminal Appeal No\.?\s*[\d/]+", "", clean, flags=re.I)
    clean = re.sub(r"Writ Petition\s*\(.*?\)\s*No\.?\s*[\d/]+", "", clean, flags=re.I)
    clean = re.sub(r"\s+", " ", clean).strip()

    match = re.search(r"(.+?)\s+vs?\.?\s+(.+)", clean, re.IGNORECASE)
    if match:
        petitioner = match.group(1).strip().title()
        respondent = match.group(2).strip().title()

        # Validate: must be meaningful names (not just numbers/symbols)
        if len(petitioner) > 2 and len(respondent) > 2:
            # Remove trailing artifacts
            petitioner = re.sub(r"[\(\[].*$", "", petitioner).strip()
            respondent = re.sub(r"[\(\[].*$", "", respondent).strip()
            return petitioner, respondent

    return None, None


def extract_dates(text):
    """
    Extract important dates from legal document.
    Returns up to 5 unique dates found.
    """
    patterns = [
        r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{1,2}-\d{1,2}-\d{4}",
    ]
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)

    # Deduplicate preserving order
    seen = set()
    unique_dates = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique_dates.append(d)

    return unique_dates[:5]


def extract_entities(text):
    """
    Extract named entities from legal document using spaCy NER.
    Returns persons, organizations, locations, dates.
    Falls back to empty dict if spaCy not available.
    """
    if nlp is None:
        return {"persons": [], "organizations": [], "locations": [], "dates": []}

    # Use only first 5000 chars for speed
    doc = nlp(text[:5000])

    entities = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "dates": []
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
        elif ent.label_ == "GPE":
            entities["locations"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)

    # Deduplicate and limit each category
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))[:5]

    return entities