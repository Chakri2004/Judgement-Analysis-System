import os
import sys
import pdfplumber
import re
import json
import numpy as np
from src.llm_client import generate_llm_response
from src.embedding_model import get_embedding
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_required, current_user, logout_user, login_user
from bson.objectid import ObjectId
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv
from src.ml_predict import predict_domain_ml
from auth.auth_routes import auth, User
from src.database import save_prediction, get_all_documents, users_collection, predictions_collection
from src.rag_engine import retrieve_relevant_laws, ask
from werkzeug.security import check_password_hash, generate_password_hash
from src.ml_models import extract_case_name, extract_judge_name, extract_parties, extract_dates, extract_entities
from src.legal_analyzer import extract_case_citations
from src.law_domains import LAW_DOMAIN_MAP, KEYWORD_DOMAIN_MAP
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

load_dotenv()

app = Flask(
    __name__,
    template_folder="src/templates",
    static_folder="src/static"
)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret")

app.register_blueprint(auth)
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = None
login_manager.init_app(app)

# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def clean_mongo_data(data):
    if isinstance(data, dict):
        return {key: clean_mongo_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_mongo_data(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, datetime):
        return data.strftime("%Y-%m-%d %H:%M")
    else:
        return data

@app.context_processor
def inject_case_data():
    if not current_user.is_authenticated:
        return dict(caseData=[])
    docs = list(
        predictions_collection
        .find({"user_id": current_user.id})
        .sort("_id", -1)
    )
    docs = clean_mongo_data(docs)
    search_data = []
    for d in docs:
        if d.get("type") == "note":
            search_data.append({
                "_id": d["_id"],
                "type": "note",
                "title": d.get("case_name", "Note"),
                "main_category": "Note",
                "status": "Saved Note",
                "timestamp": ""
            })
        else:
            search_data.append({
                "_id": d["_id"],
                "type": "case",
                "title": d.get("case_name", "Case"),
                "main_category": d.get("main_category", ""),
                "status": d.get("status", "Case"),
                "timestamp": d.get("timestamp", "")
            })
    return dict(caseData=search_data)

@login_manager.user_loader
def load_user(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(user)
    return None

# ─────────────────────────────────────────────
# DOCUMENT TYPE DETECTION
# ─────────────────────────────────────────────

def detect_document_type(text):
    text_lower = text.lower()
    if "press release" in text_lower or "stakeholders consultation" in text_lower:
        return "Press Release"
    if "an act to" in text_lower and "be it enacted" in text_lower:
        return "Statute"
    if "versus" in text_lower or " v " in text_lower or " vs " in text_lower:
        return "Judgment"
    if "petitioner" in text_lower and "respondent" in text_lower:
        return "Petition"
    if "agreement" in text_lower or "contract" in text_lower:
        return "Contract"
    if "order" in text_lower and "court" in text_lower:
        return "Court Order"
    return "Legal Document"

# ─────────────────────────────────────────────
# DOMAIN CLASSIFICATION — fixed priority + no keyword bleed
# ─────────────────────────────────────────────

def classify_main_category(text):
    text_lower = text.lower()
    detected_acts = detect_acts(text)

    # Priority order: specific acts win over broad keywords
    priority_order = [
        "POCSO Act",
        "Protection of Children from Sexual Offences Act",
        "Information Technology Act",
        "Negotiable Instruments Act",
        "Insolvency and Bankruptcy Code",
        "Consumer Protection Act",
        "Hindu Marriage Act",
        "Hindu Succession Act",
        "Industrial Disputes Act",
        "Indian Contract Act",
        "Transfer of Property Act",
        "Indian Penal Code",
        "Code of Criminal Procedure",
        "Code of Civil Procedure",
        "Companies Act",
        "Income Tax Act",
        "Environment Protection Act",
        "Arbitration and Conciliation Act",
        "Competition Act",
        "Insurance Act",
        "Copyright Act",
        "Trademark Act",
        "Patents Act",
        "Constitution of India",
    ]

    for act in priority_order:
        if act in detected_acts and act in LAW_DOMAIN_MAP:
            return LAW_DOMAIN_MAP[act]

    # Weighted keyword scoring — prevents single-word false matches
    domain_scores = {}
    for domain, keywords in KEYWORD_DOMAIN_MAP.items():
        score = 0
        for word in keywords:
            if word in text_lower:
                # Multi-word phrases score higher
                score += 2 if " " in word else 1
        if score > 0:
            domain_scores[domain] = score

    if domain_scores:
        # Must have at least score of 2 to avoid noise
        best = max(domain_scores, key=domain_scores.get)
        if domain_scores[best] >= 2:
            return best

    # LLM fallback
    prompt = f"""
You are an expert Indian legal document classifier.

Classify the document into exactly ONE of these domains:
Criminal Law, Civil Law, Constitutional Law, Administrative Law,
Corporate Law, Labour Law, Intellectual Property Law, Tax Law,
Family Law, Child Protection (POCSO), Cyber Law, Consumer Protection,
Environmental Law, Property Law, Banking Law, Insurance Law,
Arbitration Law, Contract Law, Company Law, Competition Law

Return ONLY the domain name. No explanation.

Document (first 3000 chars):
{text[:3000]}
"""
    try:
        category = generate_llm_response(prompt).strip()
        # Validate it's a known domain
        all_domains = list(set(LAW_DOMAIN_MAP.values()))
        for d in all_domains:
            if d.lower() in category.lower():
                return d
        return category if category else "General Law"
    except Exception as e:
        print("Classification Error:", e)
        return "General Law"

# ─────────────────────────────────────────────
# EXPLAINABILITY
# ─────────────────────────────────────────────

def generate_explanation(domain, acts, text, confidence=75):
    text_lower = text.lower()
    matched_keywords = []
    domain_keywords = KEYWORD_DOMAIN_MAP.get(domain, [])
    for word in domain_keywords:
        if word in text_lower:
            matched_keywords.append(word)
    if acts:
        matched_keywords.extend(acts[:2])
    if not matched_keywords:
        matched_keywords = ["legal terminology", "contextual patterns"]

    explanation = (
        f"This case is classified under {domain} because the system detected "
        f"domain-specific indicators such as {', '.join(matched_keywords[:5])}. "
        f"Additionally, relevant legal acts like "
        f"{', '.join(acts[:2]) if acts else 'general legal provisions'} "
        f"support this classification."
    )

    if confidence >= 80:
        explanation += "\n\nThe model has high confidence due to strong keyword and semantic alignment."
    elif confidence >= 60:
        explanation += "\n\nThe model has moderate confidence based on partial keyword and contextual matches."
    else:
        explanation += "\n\nThe confidence is lower due to limited domain-specific indicators."

    return explanation.strip(), matched_keywords[:5]

# ─────────────────────────────────────────────
# SECTION EXTRACTION & NORMALIZATION
# ─────────────────────────────────────────────

def extract_legal_sections(text):
    sections = []
    patterns = [
        r"section\s+(\d{1,3}[a-z]?(?:\(\d+\))?)\b",
        r"sections?\s+(\d{1,3})\s*(?:and|,)\s*(\d{1,3})",
        r"u/s\s+(\d{1,3})",
        r"\b(\d{1,3})\s*ipc\b",
        r"\b(\d{1,3})\s*crpc\b",
        r"\b(\d{1,3})\s*cpc\b"
    ]
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                for m in match:
                    if m:
                        sections.append({"section": f"Section {m}", "title": f"Section {m}"})
            else:
                sections.append({"section": f"Section {match}", "title": f"Section {match}"})

    sections = list({s["section"]: s for s in sections}.values())
    sections = sorted(sections, key=lambda x: x["section"])
    return sections[:5]

def clean_sections(section_list):
    cleaned = []
    for s in section_list:
        s = s.strip()
        if "read with" in s.lower():
            parts = re.findall(r"Section\s+\d+[A-Za-z]*", s, re.IGNORECASE)
            cleaned.extend(parts)
        else:
            cleaned.append(s)
    final = []
    for sec in cleaned:
        sec = re.sub(r"Sections?", "Section", sec, flags=re.IGNORECASE)
        sec = re.sub(r"\s+", " ", sec)
        final.append(sec.strip())
    return sorted(list(set(final)))

def normalize_sections(sections):
    normalized = set()
    for s in sections:
        s = s.lower()
        match = re.search(r"section\s+(\d+[a-z]*)", s)
        if not match:
            continue
        number = match.group(1)
        if "ipc" in s:
            normalized.add(f"Section {number} IPC")
        elif "crpc" in s:
            normalized.add(f"Section {number} CrPC")
        else:
            normalized.add(f"Section {number}")
    return sorted(list(normalized))

def build_subdivisions_from_sections(sections):
    subdivisions = []
    for s in sections:
        title = s["title"].lower()
        if "assault" in title:
            severity = "High"
        elif "punishment" in title:
            severity = "High"
        elif "procedure" in title:
            severity = "Medium"
        elif "definition" in title:
            severity = "Low"
        else:
            severity = "Medium"
        subdivisions.append({
            "title": f"Legal Provision: {s['section']}",
            "law": s["section"],
            "severity": severity,
            "explanation": f"This provision explains {s['title']} under the Act."
        })
    return subdivisions

# ─────────────────────────────────────────────
# ACT DETECTION
# ─────────────────────────────────────────────

def detect_acts(text):
    acts = []
    act_patterns = {
        "Indian Penal Code": r"\b(IPC|I\.P\.C\.|Indian Penal Code)\b",
        "Code of Criminal Procedure": r"\b(CrPC|Cr\.P\.C\.|Code of Criminal Procedure)\b",
        "Code of Civil Procedure": r"\b(CPC|C\.P\.C\.|Code of Civil Procedure)\b",
        "POCSO Act": r"\b(POCSO|Protection of Children from Sexual Offences Act)\b",
        "Information Technology Act": r"\b(IT Act|Information Technology Act|IT Act 2000)\b",
        "Indian Contract Act": r"\b(Contract Act|Indian Contract Act)\b",
        "Transfer of Property Act": r"\bTransfer of Property Act\b",
        "Domestic Violence Act": r"\b(DV Act|Domestic Violence Act)\b",
        "Hindu Marriage Act": r"\bHindu Marriage Act\b",
        "Hindu Succession Act": r"\bHindu Succession Act\b",
        "Consumer Protection Act": r"\bConsumer Protection Act\b",
        "Negotiable Instruments Act": r"\bNegotiable Instruments Act\b",
        "Companies Act": r"\bCompanies Act\b",
        "Insolvency and Bankruptcy Code": r"\b(IBC|Insolvency and Bankruptcy Code)\b",
        "NDPS Act": r"\bNDPS\b",
        "Motor Vehicles Act": r"\bMotor Vehicles Act\b",
        "Income Tax Act": r"\bIncome Tax Act\b",
        "Environment Protection Act": r"\bEnvironment Protection Act\b",
        "Arbitration and Conciliation Act": r"\bArbitration and Conciliation Act\b",
        "Competition Act": r"\bCompetition Act\b",
        "Insurance Act": r"\bInsurance Act\b",
        "Copyright Act": r"\bCopyright Act\b",
        "Trademark Act": r"\bTrademark Act\b",
        "Patents Act": r"\bPatents Act\b",
        "Constitution of India": r"\b(Constitution of India|Article \d+)\b",
        "Industrial Disputes Act": r"\bIndustrial Disputes Act\b",
        "Factories Act": r"\bFactories Act\b",
        "Minimum Wages Act": r"\bMinimum Wages Act\b",
    }
    for act, pattern in act_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            acts.append(act)
    return list(set(acts))

# ─────────────────────────────────────────────
# COURT DECISION EXTRACTION
# ─────────────────────────────────────────────

OUTCOME_PATTERNS = {
    "Sentence Suspended": r"sentence\s+(?:shall\s+be\s+)?suspended",
    "Bail Granted": r"bail\s+(?:is\s+)?granted",
    "Bail Rejected": r"bail\s+(?:is\s+)?rejected",
    "Appeal Allowed": r"appeal\s+(?:is\s+)?allowed|appeal\s+succeeds",
    "Appeal Dismissed": r"appeal\s+(?:is\s+)?dismissed|appeal\s+fails",
    "Petition Allowed": r"petition\s+(?:is\s+)?allowed",
    "Petition Dismissed": r"petition\s+(?:is\s+)?dismissed",
    "Conviction Set Aside": r"conviction\s+(?:is\s+)?set\s+aside",
    "Conviction Upheld": r"conviction\s+(?:is\s+)?upheld",
    "Case Remanded": r"matter\s+is\s+remanded|case\s+is\s+remanded",
    "Judgment Set Aside": r"judgment\s+.*set\s+aside",
    "Decree Set Aside": r"decree\s+.*set\s+aside",
    "Complaint Allowed": r"complaint\s+(?:is\s+)?allowed",
    "Complaint Dismissed": r"complaint\s+(?:is\s+)?dismissed",
    "Order Upheld": r"order\s+(?:is\s+)?upheld",
    "Compensation Awarded": r"compensation\s+(?:of\s+)?(?:rs\.?|₹)?\s*\d+",
    "Acquitted": r"\bacquitt?ed\b",
    "Convicted": r"\bconvict(?:ed|ion\s+confirmed)\b",
}

def predict_case_outcome(text):
    text_lower = text.lower()
    for outcome, pattern in OUTCOME_PATTERNS.items():
        if re.search(pattern, text_lower):
            return outcome
    return "Outcome Not Detected"

def extract_court_decision(text):
    return predict_case_outcome(text)

# ─────────────────────────────────────────────
# COURT & JUDGE
# ─────────────────────────────────────────────

def detect_court(text):
    patterns = [
        r"supreme court of india",
        r"high court of [a-z\s]+",
        r"district court",
        r"session[s]? court",
        r"court of the civil judge",
        r"national consumer disputes redressal commission",
        r"state consumer disputes redressal commission",
        r"district consumer disputes redressal commission",
        r"family court",
        r"tribunal",
    ]
    text_lower = text.lower()
    for p in patterns:
        match = re.search(p, text_lower)
        if match:
            return match.group().title()
    return "Court Not Identified"

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def generate_case_summary(text):
    text_lower = text.lower()
    if "be it enacted" in text_lower or "an act to" in text_lower:
        prompt = f"""Summarize the following Indian legal statute.
Include: Purpose of the Act, Key provisions, What offences it regulates, Why the law exists.
Keep summary between 80-120 words.

Document:
{text[:3000]}"""
    else:
        prompt = f"""Summarize the following Indian legal case.
Include: Background, Key legal issue, Relevant law, Court reasoning.
Keep summary between 80-120 words. Use only information from the document.

Case:
{text[:3000]}"""
    try:
        summary = generate_llm_response(prompt) or "Summary unavailable."
        summary = summary.strip()
        summary = re.sub(r"[*#]", "", summary)
        summary = re.sub(r"\s+", " ", summary)
        return summary
    except Exception as e:
        print("Summary error:", e)
        return "Summary could not be generated."

# ─────────────────────────────────────────────
# RISK & ARGUMENTS
# ─────────────────────────────────────────────

def predict_legal_risk(text, main_category):
    prompt = f"""Evaluate the legal risk of the following case.

Provide:
Risk Level: Low / Medium / High

Strengths:
- bullet points

Weaknesses:
- bullet points

Probability of Success: percentage
Keep answer under 120 words.

Case Category: {main_category}
{text[:3000]}"""
    try:
        return generate_llm_response(prompt)
    except:
        return "Risk analysis could not be generated."

def generate_legal_arguments(text, main_category):
    prompt = f"""You are a senior Indian lawyer.
Based on the following case document generate:

Petitioner Arguments:
- Provide 3 strong legal arguments.

Respondent Arguments:
- Provide 3 strong legal arguments.

Use bullet points.
Case Category: {main_category}
Document:
{text[:3000]}"""
    try:
        return generate_llm_response(prompt)
    except:
        return "Legal arguments could not be generated."

# ─────────────────────────────────────────────
# TIMELINE & AMOUNTS
# ─────────────────────────────────────────────

def extract_case_timeline(text):
    timeline = {}
    patterns = {
        "Incident Date": r"(incident|offence|crime)\s+(?:occurred\s+on|dated)\s+([A-Za-z0-9,\-\s]+)",
        "FIR Date": r"(fir\s+(?:registered|lodged)\s+on)\s+([A-Za-z0-9,\-\s]+)",
        "Charge Sheet Filed": r"(charge\s+sheet\s+(?:filed|submitted)\s+on)\s+([A-Za-z0-9,\-\s]+)",
        "Trial Court Judgment": r"(trial\s+court\s+(?:judgment|order)\s+dated)\s+([A-Za-z0-9,\-\s]+)",
        "High Court Decision": r"(high\s+court\s+(?:judgment|order)\s+dated)\s+([A-Za-z0-9,\-\s]+)",
        "Supreme Court Decision": r"(supreme\s+court\s+(?:judgment|order)\s+dated)\s+([A-Za-z0-9,\-\s]+)"
    }
    text_lower = text.lower()
    for label, pattern in patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            timeline[label] = match.group(2).strip()
    return timeline

def extract_monetary_amounts(text):
    """
    FIX: Indian currency uses mixed comma grouping:
         Rs.34,50,00,000 — first group 2 digits, rest 2 digits (lakhs/crores)
         Old regex stopped at Rs.34 or Rs.6 — now captures full amount.
    Patterns in priority order — most specific first.
    """
    patterns = [
        # ₹ symbol with Indian grouping  e.g. ₹7,16,41,493 or ₹ 34,50,00,000
        r"₹\s?\d{1,3}(?:,\d{2,3})+",
        # Rs. / Rs with Indian grouping  e.g. Rs.7,16,41,493
        r"Rs\.?\s?\d{1,3}(?:,\d{2,3})+",
        # Plain ₹ followed by digits (fallback)
        r"₹\s?\d+",
        # Plain Rs. fallback
        r"Rs\.?\s?\d+",
        # digit + rupees  e.g. 500 rupees
        r"\d[\d,]+\s?rupees",
    ]

    seen = set()
    amounts = []

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for amt in matches:
            # Normalize prefix to Rs.
            normalized = re.sub(r"^₹\s?", "Rs.", amt.strip())
            normalized = re.sub(r"^[Rr][Ss]\.?\s?", "Rs.", normalized)
            normalized = normalized.strip()

            # Skip amounts that are clearly incomplete
            # (single number under 100 with no commas — likely a section number)
            digits_only = re.sub(r"[^\d]", "", normalized)
            if len(digits_only) < 4:
                continue

            # Skip if we already have this value (dedup)
            if normalized in seen:
                continue

            seen.add(normalized)
            amounts.append(normalized)

    # Sort by numeric value descending so largest amounts appear first
    def numeric_val(s):
        digits = re.sub(r"[^\d]", "", s)
        return int(digits) if digits else 0

    amounts.sort(key=numeric_val, reverse=True)
    return amounts[:6]

# ─────────────────────────────────────────────
# SECTION EXPLANATION
# ─────────────────────────────────────────────

def explain_sections_with_ai(section_list, text, acts):
    explanations = []
    for sec in section_list[:5]:
        if isinstance(sec, dict):
            sec = sec.get("section")
        prompt = f"""Explain the Indian legal provision {sec}.
Possible Acts mentioned in the document: {acts}
If the section belongs to one of these Acts, explain it.
Provide: 1. Act name 2. What the section defines 3. Punishment if applicable 4. Example
Keep explanation short.

Document context:
{text[:1200]}"""
        try:
            explanation = generate_llm_response(prompt).strip()
            if not explanation or len(explanation) < 10:
                explanation = f"{sec} is a legal provision under Indian law."
        except:
            explanation = f"{sec} is a legal provision under Indian law."
        explanations.append({"section": sec, "explanation": explanation})
    return explanations

# ─────────────────────────────────────────────
# DOMAIN SUBDIVISIONS
# ─────────────────────────────────────────────

DOMAIN_SUBDIVISIONS = {
    "Criminal Law": [
        {"title": "Cheating and Fraud", "explanation": "Cases involving deception, financial fraud, or dishonest inducement of property under provisions like Section 420 IPC."},
        {"title": "Forgery and Fake Documents", "explanation": "Cases involving creation or use of forged documents under provisions such as Sections 465–471 IPC."},
        {"title": "Criminal Conspiracy", "explanation": "Situations where multiple persons agree to commit an illegal act, punishable under Section 120B IPC."},
        {"title": "Bail Applications", "explanation": "Legal proceedings where an accused person seeks temporary release from custody pending investigation or trial."},
        {"title": "Suspension of Sentence", "explanation": "Applications filed before appellate courts requesting suspension of conviction or imprisonment during appeal."},
        {"title": "Assault and Violence", "explanation": "Cases involving physical harm, intimidation, or violent offences including assault and grievous hurt."}
    ],
    "Property Law": [
        {"title": "Partition of Property", "explanation": "Disputes among family members or co-owners regarding division and separate possession of property."},
        {"title": "Specific Performance of Agreement", "explanation": "Cases seeking enforcement of agreements for sale of property under the Specific Relief Act."},
        {"title": "Sale Deed Validity", "explanation": "Disputes questioning whether a sale deed was legally executed or validly transferred ownership."},
        {"title": "Ownership and Title Disputes", "explanation": "Cases determining the lawful ownership or title of land or immovable property."},
        {"title": "Transfer of Property", "explanation": "Issues relating to legal transfer of ownership under the Transfer of Property Act."},
        {"title": "Injunction Orders", "explanation": "Court orders preventing parties from selling, altering, or interfering with property rights."}
    ],
    "Civil Law": [
        {"title": "Contract Disputes", "explanation": "Cases involving breach or enforcement of agreements between parties."},
        {"title": "Property Ownership Disputes", "explanation": "Civil disputes determining lawful ownership or possession of land or property."},
        {"title": "Recovery of Money", "explanation": "Cases filed to recover unpaid debts, loans, or contractual payments."},
        {"title": "Injunction Proceedings", "explanation": "Civil remedies where courts order a party to stop or perform a particular act."},
        {"title": "Specific Performance", "explanation": "Cases seeking court enforcement of contractual obligations."}
    ],
    "Child Protection (POCSO)": [
        {"title": "Penetrative Sexual Assault", "explanation": "Offences defined under Sections 3 and 4 of the POCSO Act involving sexual assault against minors."},
        {"title": "Aggravated Sexual Assault", "explanation": "Serious offences involving abuse of authority, multiple offenders, or severe harm to a child."},
        {"title": "Sexual Harassment of Child", "explanation": "Acts involving inappropriate touching or harassment under Sections 11 and 12 of the POCSO Act."},
        {"title": "Child Pornography", "explanation": "Offences related to using minors for pornographic purposes under Sections 13–15."},
        {"title": "Trial in Special Court", "explanation": "Procedures for child-friendly trial conducted by Special Courts under Sections 28–38."}
    ],
    "Banking Law": [
        {"title": "Loan Default", "explanation": "Failure of a borrower to repay a loan according to agreed terms."},
        {"title": "Cheque Bounce", "explanation": "Dishonour of cheque under provisions like Section 138 of the Negotiable Instruments Act."},
        {"title": "Bank Fraud", "explanation": "Fraudulent transactions involving banking institutions."},
        {"title": "Recovery Proceedings", "explanation": "Legal proceedings initiated by banks for loan recovery."}
    ],
    "Family Law": [
        {"title": "Divorce", "explanation": "Legal proceedings seeking dissolution of marriage."},
        {"title": "Child Custody", "explanation": "Disputes over guardianship and custody of children."},
        {"title": "Maintenance", "explanation": "Claims for financial support between spouses."},
        {"title": "Adoption", "explanation": "Legal adoption of a child."},
        {"title": "Inheritance", "explanation": "Disputes relating to succession of property."}
    ],
    "Cyber Law": [
        {"title": "Online Fraud", "explanation": "Fraud committed using digital platforms."},
        {"title": "Identity Theft", "explanation": "Unauthorized use of personal identity information."},
        {"title": "Data Breach", "explanation": "Unauthorized access to confidential digital data."},
        {"title": "Cyber Harassment", "explanation": "Harassment conducted through electronic communication."}
    ],
    "Corporate Law": [
        {"title": "Corporate Fraud", "explanation": "Fraudulent activities conducted by company officials."},
        {"title": "Shareholder Disputes", "explanation": "Disputes between company shareholders."},
        {"title": "Company Mismanagement", "explanation": "Improper management of company affairs."},
        {"title": "Corporate Insolvency", "explanation": "Proceedings under insolvency and bankruptcy law."}
    ],
    "Company Law": [
        {"title": "Director Liability", "explanation": "Disputes over director duties and liabilities under Companies Act."},
        {"title": "Shareholder Rights", "explanation": "Cases involving minority shareholder protection."},
        {"title": "Corporate Governance", "explanation": "Disputes over proper management and governance of companies."}
    ],
    "Consumer Protection": [
        {"title": "Defective Product", "explanation": "Consumer complaints regarding faulty products."},
        {"title": "Service Deficiency", "explanation": "Failure to provide promised services."},
        {"title": "Medical Negligence", "explanation": "Negligence by medical professionals causing harm."},
        {"title": "Unfair Trade Practice", "explanation": "Deceptive or unfair practices by traders or service providers."},
        {"title": "Real Estate Delay", "explanation": "Developer's failure to deliver possession within agreed time."}
    ],
    "Labour Law": [
        {"title": "Wrongful Termination", "explanation": "Illegal termination of employment."},
        {"title": "Wage Disputes", "explanation": "Disputes related to payment of wages."},
        {"title": "Industrial Disputes", "explanation": "Disputes between workers and employers."}
    ],
    "Constitutional Law": [
        {"title": "Fundamental Rights Violation", "explanation": "Cases challenging violation of rights under Part III of the Constitution."},
        {"title": "Writ Petition", "explanation": "Petitions filed under Article 32 or 226 seeking constitutional remedies."},
        {"title": "Constitutional Validity", "explanation": "Challenges to the constitutional validity of statutes or government actions."}
    ],
    "Tax Law": [
        {"title": "Income Tax Dispute", "explanation": "Disputes over income tax assessment or recovery."},
        {"title": "GST Dispute", "explanation": "Cases involving Goods and Services Tax."},
        {"title": "Tax Evasion", "explanation": "Criminal proceedings for tax evasion."}
    ],
    "Arbitration Law": [
        {"title": "Arbitral Award Challenge", "explanation": "Applications to set aside or enforce arbitral awards."},
        {"title": "Arbitration Agreement", "explanation": "Disputes over validity or scope of arbitration agreements."}
    ],
    "Insurance Law": [
        {"title": "Claim Repudiation", "explanation": "Insurer's refusal to honor a valid insurance claim."},
        {"title": "Policy Dispute", "explanation": "Disputes over interpretation of insurance policy terms."}
    ],
    "Environmental Law": [
        {"title": "Pollution", "explanation": "Cases involving environmental pollution or damage."},
        {"title": "Forest & Wildlife", "explanation": "Cases under Forest Conservation or Wildlife Protection Acts."}
    ],
    "Intellectual Property Law": [
        {"title": "Copyright Infringement", "explanation": "Unauthorized use of copyrighted works."},
        {"title": "Trademark Dispute", "explanation": "Disputes over trademark ownership or infringement."},
        {"title": "Patent Dispute", "explanation": "Cases involving patent validity or infringement."}
    ],
    "Contract Law": [
        {"title": "Breach of Contract", "explanation": "Failure to perform contractual obligations."},
        {"title": "Specific Performance", "explanation": "Court enforcement of contract terms."},
        {"title": "Contract Validity", "explanation": "Disputes over whether a contract is valid and enforceable."}
    ],
    "Administrative Law": [
        {"title": "Government Order Challenge", "explanation": "Challenges to orders passed by administrative or regulatory authorities."},
        {"title": "Quasi-Judicial Decision", "explanation": "Review of decisions made by quasi-judicial bodies."}
    ],
}

SECTION_SUBDIVISION_MAP = {
    "120B": {"title": "Criminal Conspiracy", "explanation": "Agreement between two or more persons to commit an illegal act under Section 120B IPC."},
    "406": {"title": "Criminal Breach of Trust", "explanation": "Misappropriation of property entrusted to a person under Section 406 IPC."},
    "409": {"title": "Breach of Trust by Public Servant", "explanation": "Serious offence where a public servant dishonestly misappropriates entrusted property."},
    "420": {"title": "Cheating and Fraud", "explanation": "Dishonest inducement causing delivery of property under Section 420 IPC."},
    "430": {"title": "Mischief Causing Damage", "explanation": "Intentional damage to property or public infrastructure."},
    "3": {"title": "Penetrative Sexual Assault", "explanation": "Offence involving sexual penetration against a minor under Section 3 of the POCSO Act."},
    "4": {"title": "Punishment for Penetrative Sexual Assault", "explanation": "Punishment for penetrative sexual assault under Section 4 of the POCSO Act."},
    "7": {"title": "Sexual Assault on Child", "explanation": "Sexual assault without penetration under Section 7 of the POCSO Act."},
    "8": {"title": "Punishment for Sexual Assault", "explanation": "Punishment prescribed under Section 8 of the POCSO Act."},
    "138": {"title": "Cheque Dishonour", "explanation": "Dishonour of cheque for insufficiency of funds under Section 138 of the Negotiable Instruments Act."},
    "13": {"title": "Grounds for Divorce", "explanation": "Grounds for divorce under Section 13 of the Hindu Marriage Act."},
    "125": {"title": "Maintenance", "explanation": "Provision for maintenance of wife, children and parents under Section 125 CrPC."},
}

def generate_subdivisions_from_sections(sections):
    subdivisions = []
    for sec in sections:
        number = re.findall(r"\d+", sec)
        if not number:
            continue
        number = number[0]
        if number in SECTION_SUBDIVISION_MAP:
            info = SECTION_SUBDIVISION_MAP[number]
            subdivisions.append({
                "title": info["title"],
                "law": f"Section {number}",
                "severity": "High",
                "explanation": info["explanation"]
            })
    return subdivisions

# ─────────────────────────────────────────────
# DOMAIN-AWARE LLM SUBDIVISION ANALYSIS — fixed for all domains
# ─────────────────────────────────────────────

def analyze_subdivisions_llm(text, main_category):
    detected_sections = [s["section"] for s in extract_legal_sections(text)]
    retrieved_laws = retrieve_relevant_laws(text, k=5)
    law_context = "\n\n".join(
        [f"Case {i+1}:\n{law['content'][:400]}"
         for i, law in enumerate(retrieved_laws[:3])]
    )

    # Domain-specific guidance injected into prompt
    domain_hints = {
        "Consumer Protection": "Focus on deficiency of service, unfair trade practices, compensation awarded, possession delays, product defects.",
        "Family Law": "Focus on divorce grounds, maintenance amounts, child custody, alimony, matrimonial rights.",
        "Child Protection (POCSO)": "Focus on POCSO sections, nature of offence against child, bail/conviction status.",
        "Banking Law": "Focus on cheque bounce (Section 138 NI Act), loan defaults, recovery proceedings.",
        "Cyber Law": "Focus on IT Act sections, nature of cyber offence, digital evidence.",
        "Tax Law": "Focus on tax assessment, demand raised, grounds of appeal, tax amounts.",
        "Labour Law": "Focus on termination legality, wages due, reinstatement, industrial dispute.",
        "Corporate Law": "Focus on company mismanagement, shareholder disputes, IBC proceedings.",
        "Constitutional Law": "Focus on fundamental rights, writ type, article violated, relief sought.",
        "Criminal Law": "Focus on IPC/BNS sections, nature of offence, bail status, sentence.",
        "Property Law": "Focus on title dispute, sale deed, possession, partition, injunction.",
        "Environmental Law": "Focus on pollution type, environmental damage, regulatory violation.",
        "Arbitration Law": "Focus on arbitral award, grounds for challenge, Section 34/37 proceedings.",
        "Insurance Law": "Focus on claim type, repudiation reason, policy terms, compensation.",
        "Intellectual Property Law": "Focus on IP type (copyright/trademark/patent), infringement nature.",
        "Contract Law": "Focus on breach type, specific performance, damages claimed.",
        "Administrative Law": "Focus on government order challenged, authority, administrative action.",
        "Civil Law": "Focus on civil dispute type, parties, relief sought, procedural stage.",
        "Competition Law": "Focus on anti-competitive conduct, CCI proceedings, market dominance.",
        "Company Law": "Focus on Companies Act sections, director liability, NCLT proceedings.",
    }

    domain_context = domain_hints.get(main_category, "Extract all relevant legal issues from the document.")

    GENERIC_TITLES = [
        "legal issue", "legal matter", "legal provision",
        "legal case", "legal dispute", "general matter"
    ]

    prompt = f"""You are a senior Indian legal analyst specializing in {main_category}.

DOMAIN FOCUS: {domain_context}

STRICT RULES:
1. Extract legal provisions ONLY if explicitly present in the document.
2. Do NOT infer or guess sections not present in the text.
3. Generate meaningful issue titles specific to {main_category} — NEVER use generic titles like "Legal Issue".
4. Base all reasoning strictly on the provided document.
5. Return ONLY raw JSON — no markdown, no explanation outside JSON.

Detected Sections in Document: {detected_sections}
Main Category: {main_category}

Relevant Case Law Context:
{law_context}

Return this exact JSON structure:
{{
  "main_category": "{main_category}",
  "nature_of_dispute": "Clear 2-3 sentence summary of the specific dispute in this document",
  "legal_provisions": ["Only provisions explicitly mentioned in the document"],
  "subdivisions": [
    {{
      "title": "Specific descriptive title relevant to {main_category} (e.g., 'Cheque Dishonour Under Section 138', 'Custody Dispute Post Divorce', 'Service Deficiency by Builder'). NEVER use 'Legal Issue'.",
      "law": "Exact section from document OR 'Not specified in document'",
      "severity": "Low/Medium/High",
      "explanation": "Clear legal reasoning explaining this issue and its relevance to the case"
    }}
  ],
  "confidence": 85
}}

Case Document:
{text[:4000]}"""

    try:
        output = generate_llm_response(prompt)
        print("LLM OUTPUT:", output[:500])
        match = re.search(r"\{[\s\S]*\}", output)
        if match:
            analysis = json.loads(match.group().strip())
        else:
            raise ValueError("Invalid JSON returned by LLM")

        # Normalize sections
        all_sections = analysis.get("legal_provisions", []) + detected_sections
        cleaned = clean_sections(all_sections)
        analysis["legal_provisions"] = list(set(normalize_sections(cleaned)))

        # Fix generic subdivision titles
        for sub in analysis.get("subdivisions", []):
            title = sub.get("title", "").lower()
            if title in GENERIC_TITLES or len(title) < 5:
                law = sub.get("law", "")
                if "pocso" in law.lower():
                    sub["title"] = "Sexual Offence Against Minor"
                elif "420" in law:
                    sub["title"] = "Cheating and Fraud"
                elif "120b" in law.lower():
                    sub["title"] = "Criminal Conspiracy"
                elif "138" in law:
                    sub["title"] = "Cheque Dishonour"
                elif "13" in law and "marriage" in law.lower():
                    sub["title"] = "Divorce Petition"
                elif "consumer" in main_category.lower():
                    sub["title"] = "Consumer Dispute — Service Deficiency"
                elif "family" in main_category.lower():
                    sub["title"] = "Matrimonial Dispute"
                elif "property" in main_category.lower():
                    sub["title"] = "Property Title Dispute"
                else:
                    sub["title"] = f"{main_category} — Legal Issue"

        return analysis

    except Exception as e:
        print("RAG Analysis Error:", e)
        # Graceful fallback with domain-appropriate defaults
        fallback_subdivisions = []
        for issue in DOMAIN_SUBDIVISIONS.get(main_category, [])[:3]:
            fallback_subdivisions.append({
                "title": issue["title"],
                "law": "Based on case context",
                "severity": "Medium",
                "explanation": issue["explanation"]
            })
        if not fallback_subdivisions:
            fallback_subdivisions = [{
                "title": f"{main_category} — Legal Matter",
                "law": "Not specified in document",
                "severity": "Medium",
                "explanation": "The system could not extract structured legal issues. The document relates to a legal matter under " + main_category + "."
            }]
        return {
            "main_category": main_category,
            "nature_of_dispute": f"This document appears to involve a {main_category} matter.",
            "legal_provisions": detected_sections[:5],
            "subdivisions": fallback_subdivisions,
            "confidence": 70
        }

# ─────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def find_similar_cases(query_embedding, category, top_k=5):
    all_docs = list(
        predictions_collection.find(
            {"embedding": {"$exists": True}, "main_category": category},
            {"embedding": 1, "case_name": 1, "main_category": 1}
        ).limit(500)
    )
    scores = []
    for doc in all_docs:
        emb = doc.get("embedding")
        if not emb or not query_embedding:
            continue
        if len(emb) != len(query_embedding):
            continue
        score = cosine_similarity(query_embedding, emb)
        if score >= 0.98 or score < 0.50:
            continue
        doc["similarity"] = round(score * 100, 2)
        scores.append((score, doc))
    scores.sort(reverse=True, key=lambda x: x[0])
    seen = set()
    results = []
    for score, doc in scores:
        name = doc.get("case_name")
        if name in seen:
            continue
        seen.add(name)
        results.append(doc)
        if len(results) >= top_k:
            break
    return results

# ─────────────────────────────────────────────
# ENTITY CLEANING
# ─────────────────────────────────────────────

def clean_entities(entities):
    if not entities:
        return {}
    blacklist = {
        "schedule", "gazette", "miscellaneous", "section", "sec",
        "court", "case", "act", "law", "order", "assault", "child",
        "clause", "rule", "subsection", "article", "chapter", "part",
        "date", "year", "ors", "and ors", "bhartiya", "page", "crl",
        "c.a.", "slp", "lrs", "anr", "respondent", "appellant",
        "civil judge", "high court", "supreme court", "leave",
        "leave granted", "leave to appeal", "j.", "justice",
        "inter alia", "designation", "special courts"
    }
    cleaned_persons = []
    for p in entities.get("persons", []):
        p_clean = p.strip()
        if len(p_clean) < 3:
            continue
        if p_clean.lower() in blacklist:
            continue
        if p_clean.lower().startswith("section"):
            continue
        if " v. " in p_clean.lower():
            continue
        if any(char.isdigit() for char in p_clean):
            continue
        if len(p_clean.split()) > 4:
            continue
        if p_clean.isupper():
            continue
        if "." in p_clean and len(p_clean) < 5:
            continue
        if "—" in p_clean or "(" in p_clean:
            continue
        cleaned_persons.append(p_clean)

    cleaned_orgs = []
    for o in entities.get("organizations", []):
        o_clean = o.strip()
        if len(o_clean) < 3:
            continue
        if o_clean.lower() in blacklist:
            continue
        if o_clean.lower().startswith("section"):
            continue
        if "page" in o_clean.lower():
            continue
        if o_clean.lower() in ["ipc", "crpc", "cpc"]:
            continue
        if re.search(r"page\s*\d+", o_clean.lower()):
            continue
        cleaned_orgs.append(o_clean)

    entities["persons"] = list(set(cleaned_persons))
    entities["organizations"] = list(set(cleaned_orgs))
    entities.pop("locations", None)
    return entities

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return redirect(url_for("auth.login"))

@app.route("/home")
def main_home():
    return render_template("index.html")

@app.route("/careers")
def careers():
    return render_template("footer/careers/overview.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("footer/careers/how_it_works.html")

@app.route("/api-access")
def api_access():
    return render_template("footer/resources/api_access.html")

@app.route("/footer/privacy")
def footer_privacy():
    return render_template("footer/privacy.html")

@app.route("/footer/terms")
def footer_terms():
    return render_template("footer/terms.html")

@app.route("/footer/security")
def footer_security():
    return render_template("footer/security.html")

@app.route("/direct-login", methods=["POST"])
def direct_login():
    email = request.form.get("email")
    password = request.form.get("password")
    user = users_collection.find_one({"email": email})
    if user and check_password_hash(user["password"], password):
        login_user(User(user))
        return redirect("/home")
    flash("Invalid email or password", "danger")
    return redirect(url_for("auth.login"))

@app.route("/skip-login")
def skip_login():
    logout_user()
    return redirect("/home")

# ─────────────────────────────────────────────
# MAIN ANALYZE ROUTE — all bugs fixed
# ─────────────────────────────────────────────

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        return render_template("analyze.html")

    text = ""
    source = "Manual Text Entry"

    if request.form.get("judgement"):
        text = request.form["judgement"]
    elif request.files.get("pdf_file"):
        pdf = request.files["pdf_file"]
        source = "PDF Upload"
        try:
            with pdfplumber.open(pdf) as pdf_doc:
                pages = [page.extract_text() or "" for page in pdf_doc.pages]
                text = "\n".join(pages)[:10000]
        except Exception as e:
            print("PDF extraction error:", e)
            flash("Could not read PDF. Please try again.", "danger")
            return redirect(url_for("analyze"))

    if not text.strip():
        return redirect(url_for("analyze"))

    # ── Step 1: Detect document type
    doc_type = detect_document_type(text)

    # ── Step 2: Classify domain
    rule_domain = classify_main_category(text)
    try:
        ml_domain = predict_domain_ml(text)
    except Exception:
        ml_domain = "General Law"

    main_category = rule_domain if rule_domain != "General Law" else ml_domain

    # ── Step 3: Extract case metadata — always needed
    if doc_type == "Statute":
        case_name = "Statutory Legal Document"
        petitioner = None
        respondent = None
    else:
        case_name = extract_case_name(text)
        if "appeal no" in case_name.lower() or case_name == "Unknown Case":
            lines = text.split("\n")
            for line in lines[:40]:
                line_clean = line.strip()
                if len(line_clean) < 10:
                    continue
                if (" v " in line_clean.lower() or " vs " in line_clean.lower()):
                    if any(x in line_clean.lower() for x in ["arising out of", "slp"]):
                        continue
                    case_name = line_clean[:120]
                    break
        case_name = re.sub(r"\(arising.*?\)", "", case_name, flags=re.I).strip()
        case_name = re.sub(r"Civil Appeal No.*", "", case_name, flags=re.I).strip()
        case_name = re.sub(r"SLP.*No.*", "", case_name, flags=re.I).strip()
        if not case_name:
            case_name = "Unknown Case"
        petitioner, respondent = extract_parties(case_name)

    # FIX: court and judge always set regardless of doc_type
    court = detect_court(text)
    judge = extract_judge_name(text)

    # ── Step 4: Extract all features
    citations = extract_case_citations(text)
    dates = extract_dates(text)
    timeline = extract_case_timeline(text)
    amounts = extract_monetary_amounts(text)
    entities = clean_entities(extract_entities(text) or {})
    detected_acts = detect_acts(text)

    # ── Step 5: Run LLM subdivision analysis
    if doc_type == "Statute":
        raw_sections = extract_legal_sections(text)
        subdivisions = build_subdivisions_from_sections(raw_sections)
        analysis = {
            "main_category": main_category,
            "nature_of_dispute": "Statutory legal framework.",
            "legal_provisions": [s["section"] for s in raw_sections],
            "subdivisions": subdivisions,
            "confidence": 95
        }
    else:
        analysis = analyze_subdivisions_llm(text, main_category)

    # ── Step 6: Build final sections & subdivisions
    sections = normalize_sections(
        clean_sections(
            analysis.get("legal_provisions", []) +
            [s["section"] for s in extract_legal_sections(text)]
        )
    )

    subdivisions = analysis.get("subdivisions", [])

    # Fallback chain: section map → domain defaults → generic
    if not subdivisions:
        subdivisions = generate_subdivisions_from_sections(sections)

    if not subdivisions:
        domain_issues = DOMAIN_SUBDIVISIONS.get(main_category, [])
        subdivisions = [
            {
                "title": issue["title"],
                "law": "Based on case context",
                "severity": "Medium",
                "explanation": issue["explanation"]
            }
            for issue in domain_issues[:4]
        ]

    if not subdivisions:
        subdivisions = [{
            "title": f"{main_category} — Legal Matter",
            "law": "Not specified in document",
            "severity": "Medium",
            "explanation": f"No explicit statutory provisions found. Document relates to {main_category}."
        }]

    # ── Step 7: Confidence & explanation
    confidence_raw = str(analysis.get("confidence", "85"))
    match = re.search(r"\d+", confidence_raw)
    confidence = int(match.group()) if match else 85

    explanation, keywords = generate_explanation(main_category, detected_acts, text, confidence)

    # ── Step 8: Court decision
    if doc_type == "Statute":
        court_decision = "Not applicable (statutory document)"
    else:
        court_decision = predict_case_outcome(text)

    # ── Step 9: Summary, risk, arguments
    summary = generate_case_summary(text)

    if doc_type == "Statute":
        risk_analysis = "Risk analysis is not applicable for statutory legal documents."
        legal_arguments = "Legal arguments are not applicable for statutory documents."
    else:
        risk_analysis = predict_legal_risk(text, main_category)
        legal_arguments = generate_legal_arguments(text, main_category)

    # ── Step 10: Embeddings & similar cases
    try:
        embedding_vector = get_embedding(text)
        embedding = embedding_vector.tolist() if embedding_vector is not None else []
    except Exception:
        embedding = []

    query_text = summary if summary and len(summary) > 20 else text[:1000]
    try:
        similar_cases = retrieve_relevant_laws(query_text, category=main_category)
        # Normalize similar_cases for template compatibility
        normalized_similar = []
        for c in similar_cases:
            normalized_similar.append({
                "title": c.get("title") or c.get("case_name") or "Unknown Case",
                "summary": c.get("summary") or c.get("content", "")[:300],
                "main_category": c.get("main_category", main_category),
                "similarity": c.get("similarity", 0)
            })
        similar_cases = normalized_similar
    except Exception:
        similar_cases = []

    # ── Step 11: Section explanations
    section_explanations = explain_sections_with_ai(sections, text, detected_acts)

    # ── Step 12: Nature of dispute
    nature = analysis.get("nature_of_dispute", "")
    if not nature:
        nature = f"This document involves a {main_category} matter."

    # ── Step 13: Save to DB
    user_id = current_user.id if current_user.is_authenticated else None
    try:
        save_prediction(
            text=text,
            case_name=case_name,
            court=court,
            judge=judge,
            doc_type=doc_type,
            acts=detected_acts,
            main_category=main_category,
            nature_of_dispute=nature,
            source=source,
            confidence=confidence,
            subdivisions=subdivisions,
            user_id=user_id,
            court_decision=court_decision,
            summary=summary,
            embedding=embedding,
            created_at=datetime.now()
        )
    except Exception as e:
        print("DB save error:", e)

    return render_template(
        "analysis_result.html",
        main_category=analysis.get("main_category", main_category),
        case_name=case_name,
        petitioner=petitioner,
        respondent=respondent,
        citations=citations,
        dates=dates,
        timeline=timeline,
        amounts=amounts,
        entities=entities,
        subdivisions=subdivisions,
        nature=nature,
        legal_provisions=sections,
        section_explanations=section_explanations,
        confidence=confidence,
        source=source,
        court=court,
        judge=judge,
        court_decision=court_decision,
        summary=summary,
        risk_analysis=risk_analysis,
        legal_arguments=legal_arguments,
        acts=detected_acts,
        similar_cases=similar_cases,
        explanation=explanation,
        keywords=keywords
    )

# ─────────────────────────────────────────────
# ASK QUESTION
# ─────────────────────────────────────────────

@app.route("/ask_case_question", methods=["POST"])
def ask_case_question():
    question = request.form.get("question", "")
    case_text = request.form.get("case_text", "")
    q = question.lower()
    if any(word in q for word in ["what is", "define", "meaning", "explain"]):
        mode_instruction = "Answer using general legal knowledge."
    else:
        mode_instruction = "Answer using the case context if relevant."

    prompt = f"""You are an intelligent legal assistant.
{mode_instruction}
If both general knowledge and case context are relevant, combine them.

Case Context:
{case_text}

Question:
{question}

Provide a clear and helpful legal answer:"""
    answer = generate_llm_response(prompt)
    return jsonify({"answer": answer})

# ─────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────

@app.route("/search_cases", methods=["POST"])
def search_cases():
    query = request.form.get("query")
    if not query:
        return redirect(url_for("dashboard"))
    similar_cases = retrieve_relevant_laws(query)
    return render_template("search_results.html", query=query, results=similar_cases)

# ─────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────

@app.route("/save_note", methods=["POST"])
def save_note():
    try:
        data = request.get_json()
        note = data.get("note")
        case_name = data.get("case_name")
        if not note or not case_name:
            return jsonify({"status": "error", "message": "missing data"})
        result = predictions_collection.insert_one({
            "user_id": current_user.id,
            "case_name": case_name,
            "note": note,
            "type": "note",
            "created_at": datetime.now()
        })
        return jsonify({"status": "success", "id": str(result.inserted_id)})
    except Exception as e:
        print("SAVE NOTE ERROR:", e)
        return jsonify({"status": "error"})

@app.route("/view_note/<id>")
def view_note(id):
    doc = predictions_collection.find_one({"_id": ObjectId(id), "type": "note"})
    return render_template("view_note.html", doc=doc)

# ─────────────────────────────────────────────
# BOOKMARKS
# ─────────────────────────────────────────────

@app.route("/bookmarks")
def bookmarks():
    if not current_user.is_authenticated:
        documents = list(predictions_collection.find({"bookmarked": True}))
    else:
        documents = list(
            predictions_collection
            .find({"user_id": current_user.id, "bookmarked": True})
            .sort("_id", -1)
        )
    documents = clean_mongo_data(documents)
    return render_template("workspace/bookmark.html", documents=documents)

@app.route("/bookmark/<id>", methods=["POST"])
def bookmark_case(id):
    doc = predictions_collection.find_one({"_id": ObjectId(id)})
    if not doc:
        return jsonify({"status": "error", "message": "Document not found"})
    new_value = not doc.get("bookmarked", False)
    predictions_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"bookmarked": new_value}}
    )
    return jsonify({"status": "success", "bookmarked": new_value})

# ─────────────────────────────────────────────
# ASSISTANT / ASK
# ─────────────────────────────────────────────

@app.route("/workspace/assistant")
def assistant():
    return render_template("workspace/assistant.html")

@app.route("/ask", methods=["POST"])
def ask_route():
    data = request.get_json()
    question = data.get("question")
    answer = ask(question)
    return jsonify({"answer": answer})

# ─────────────────────────────────────────────
# STATIC PAGES
# ─────────────────────────────────────────────

@app.route("/case-understanding")
def case_understanding():
    return render_template("case.html")

@app.route("/domain-classification")
def domain_classification():
    return render_template("domain.html")

@app.route("/summarization")
def summarization():
    return render_template("summary.html")

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/terms")
def terms():
    return render_template("footer/terms.html")

@app.route("/privacy")
def privacy():
    return render_template("footer/privacy.html")

@app.route("/policy")
def policy():
    return render_template("footer/policy.html")

# ─────────────────────────────────────────────
# DASHBOARD & WORKSPACE
# ─────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    docs = list(predictions_collection.find({"user_id": current_user.id}).sort("_id", -1))
    total = len(docs)
    category_count = {}
    for d in docs:
        cat = d.get("main_category", "Unknown")
        category_count[cat] = category_count.get(cat, 0) + 1
    most_category = max(category_count, key=category_count.get) if category_count else "-"
    return render_template("workspace.html", documents=docs, total=total, most_category=most_category)

@app.route("/workspace")
def workspace():
    if current_user.is_authenticated:
        user_id = current_user.id
        recent_cases = list(predictions_collection.find({"user_id": user_id}).sort("_id", -1).limit(5))
        total_cases = predictions_collection.count_documents({"user_id": user_id})
        bookmarks_count = predictions_collection.count_documents({"user_id": user_id, "bookmarked": True})
        docs = list(predictions_collection.find({"user_id": user_id}))
    else:
        recent_cases = list(predictions_collection.find({}).sort("_id", -1).limit(5))
        total_cases = 0
        bookmarks_count = 0
        docs = []

    category_count = {}
    for d in docs:
        cat = d.get("main_category", "Unknown")
        category_count[cat] = category_count.get(cat, 0) + 1
    most_category = max(category_count, key=category_count.get) if category_count else "-"
    return render_template(
        "workspace/workspace.html",
        recent_cases=recent_cases,
        total_cases=total_cases,
        bookmarks_count=bookmarks_count,
        most_category=most_category
    )

# ─────────────────────────────────────────────
# CASES LIST
# ─────────────────────────────────────────────

@app.route("/cases")
def cases():
    search = request.args.get("search", "")
    page = int(request.args.get("page", 1))
    per_page = 10
    query = {"user_id": current_user.id} if current_user.is_authenticated else {}

    if search:
        conditions = [
            {"case_name": {"$regex": search, "$options": "i"}},
            {"main_category": {"$regex": search, "$options": "i"}},
            {"text": {"$regex": search, "$options": "i"}}
        ]
        if search.isdigit():
            conditions.append({"confidence": {"$gte": int(search)}})
        query["$or"] = conditions

    total = predictions_collection.count_documents(query)
    docs = list(
        predictions_collection.find(query)
        .sort("created_at", -1)
        .skip((page - 1) * per_page)
        .limit(per_page)
    )
    for d in docs:
        if "created_at" in d and d["created_at"]:
            d["formatted_date"] = d["created_at"].strftime("%d %b %Y, %I:%M %p")
        else:
            d["formatted_date"] = "-"

    total_pages = (total + per_page - 1) // per_page

    pipeline = [
        {"$match": query},
        {"$group": {"_id": "$main_category", "count": {"$sum": 1}}}
    ]
    data = list(predictions_collection.aggregate(pipeline))
    category_percent = {}
    for d in data:
        if d["_id"]:
            percent = round((d["count"] / total) * 100, 1) if total > 0 else 0
            category_percent[d["_id"]] = percent

    pipeline2 = [
        {"$match": query},
        {"$group": {"_id": "$main_category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 1}
    ]
    result = list(predictions_collection.aggregate(pipeline2))
    most_category = result[0]["_id"] if result else "-"

    return render_template(
        "workspace/cases.html",
        documents=docs,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
        search=search,
        most_category=most_category,
        category_percent=category_percent
    )

# ─────────────────────────────────────────────
# CRUD OPERATIONS
# ─────────────────────────────────────────────

@app.route("/delete/<id>", methods=["POST"])
def delete(id):
    predictions_collection.delete_one({"_id": ObjectId(id)})
    return jsonify({"success": True})

@app.route("/view/<id>")
def view_prediction(id):
    document = predictions_collection.find_one({"_id": ObjectId(id)})
    return render_template("workspace/view_document.html", doc=document)

@app.route("/analyze/<id>")
def analyze_document(id):
    doc = predictions_collection.find_one({"_id": ObjectId(id)})
    if not doc:
        return redirect(url_for("dashboard"))
    result = analyze_subdivisions_llm(doc.get("text", ""), doc.get("main_category", "General Law"))
    subdivisions = result.get("subdivisions", [])
    return render_template("analysis_result.html", main_category=doc["main_category"], subdivisions=subdivisions)

@app.route("/update_status/<id>/<new_status>", methods=["POST"])
def update_status(id, new_status):
    if new_status not in ["Approved", "Rejected"]:
        return redirect(url_for("dashboard"))
    predictions_collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": new_status}})
    return redirect(url_for("dashboard"))

# ─────────────────────────────────────────────
# EXPORT PDF — fixed: return send_file always
# ─────────────────────────────────────────────

@app.route("/export/<id>")
def export_pdf(id):
    doc = predictions_collection.find_one({"_id": ObjectId(id)})
    if not doc:
        return redirect(url_for("dashboard"))

    file_path = f"/tmp/report_{id}.pdf"
    pdf = SimpleDocTemplate(file_path)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Case Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Case Name: {doc.get('case_name', 'N/A')}", styles["Normal"]))
    elements.append(Paragraph(f"Main Category: {doc.get('main_category', 'N/A')}", styles["Normal"]))
    elements.append(Paragraph(f"Court: {doc.get('court', 'N/A')}", styles["Normal"]))
    elements.append(Paragraph(f"Judge: {doc.get('judge', 'N/A')}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {doc.get('confidence', 'N/A')}%", styles["Normal"]))
    elements.append(Paragraph(f"Decision: {doc.get('court_decision', 'N/A')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    if doc.get("summary"):
        elements.append(Paragraph("<b>Summary:</b>", styles["Heading3"]))
        elements.append(Paragraph(doc["summary"], styles["Normal"]))
        elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Detected Legal Issues:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 6))

    subdivisions = doc.get("subdivisions", [])
    if subdivisions:
        sub_list = []
        for sub in subdivisions:
            sub_text = (
                f"{sub.get('title', 'Issue')} "
                f"({sub.get('law', 'N/A')}) — "
                f"Severity: {sub.get('severity', 'Medium')}"
            )
            sub_list.append(ListItem(Paragraph(sub_text, styles["Normal"])))
        elements.append(ListFlowable(sub_list))
    else:
        elements.append(Paragraph("No subdivisions detected.", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Original Text (excerpt):</b>", styles["Heading3"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(str(doc.get("text", ""))[:3000], styles["Normal"]))

    pdf.build(elements)
    return send_file(file_path, as_attachment=True, download_name=f"case_report_{id}.pdf")

# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)