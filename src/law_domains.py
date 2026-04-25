LAW_DOMAIN_MAP = {

    # Criminal
    "Indian Penal Code": "Criminal Law",
    "IPC": "Criminal Law",
    "Bharatiya Nyaya Sanhita": "Criminal Law",
    "BNS": "Criminal Law",
    "Code of Criminal Procedure": "Criminal Law",
    "CrPC": "Criminal Law",
    "BNSS": "Criminal Law",

    # Child protection — highest priority
    "POCSO Act": "Child Protection (POCSO)",
    "Protection of Children from Sexual Offences Act": "Child Protection (POCSO)",

    # Cyber
    "Information Technology Act": "Cyber Law",
    "IT Act": "Cyber Law",

    # Property
    "Transfer of Property Act": "Property Law",
    "Registration Act": "Property Law",
    "Indian Easements Act": "Property Law",

    # Civil procedural
    "Code of Civil Procedure": "Civil Law",
    "CPC": "Civil Law",

    # Contract
    "Indian Contract Act": "Contract Law",
    "Specific Relief Act": "Contract Law",

    # Consumer
    "Consumer Protection Act": "Consumer Protection",

    # Corporate / company
    "Companies Act": "Company Law",
    "Insolvency and Bankruptcy Code": "Corporate Law",
    "IBC": "Corporate Law",

    # Banking
    "Negotiable Instruments Act": "Banking Law",
    "Banking Regulation Act": "Banking Law",

    # Family
    "Hindu Marriage Act": "Family Law",
    "Hindu Succession Act": "Family Law",
    "Hindu Adoption and Maintenance Act": "Family Law",
    "Domestic Violence Act": "Family Law",
    "Special Marriage Act": "Family Law",
    "Muslim Personal Law": "Family Law",

    # Labour
    "Industrial Disputes Act": "Labour Law",
    "Factories Act": "Labour Law",
    "Minimum Wages Act": "Labour Law",
    "Employees Provident Fund Act": "Labour Law",
    "Payment of Gratuity Act": "Labour Law",

    # Environmental
    "Environment Protection Act": "Environmental Law",
    "Forest Act": "Environmental Law",
    "Wildlife Protection Act": "Environmental Law",
    "Forest Conservation Act": "Environmental Law",
    "Water Pollution Act": "Environmental Law",
    "Air Pollution Act": "Environmental Law",

    # Arbitration
    "Arbitration and Conciliation Act": "Arbitration Law",

    # Tax
    "Income Tax Act": "Tax Law",
    "GST Act": "Tax Law",
    "Goods and Services Tax Act": "Tax Law",
    "Customs Act": "Tax Law",

    # Insurance
    "Insurance Act": "Insurance Law",
    "IRDA Act": "Insurance Law",

    # Competition
    "Competition Act": "Competition Law",

    # Constitutional
    "Constitution of India": "Constitutional Law",

    # Administrative
    "Administrative Tribunals Act": "Administrative Law",

    # Intellectual Property
    "Copyright Act": "Intellectual Property Law",
    "Trademark Act": "Intellectual Property Law",
    "Patents Act": "Intellectual Property Law",
    "Designs Act": "Intellectual Property Law",
    "Geographical Indications Act": "Intellectual Property Law",

    # Motor Vehicles
    "Motor Vehicles Act": "Civil Law",

    # NDPS
    "NDPS Act": "Criminal Law",
    "Narcotic Drugs and Psychotropic Substances Act": "Criminal Law",
}

# ─────────────────────────────────────────────────────────────────────
# KEYWORD_DOMAIN_MAP
# Rules:
#   1. Multi-word phrases are strongly preferred (less bleed)
#   2. Single generic words removed where they appear in multiple domains
#   3. Each keyword should be unique to its domain
# ─────────────────────────────────────────────────────────────────────

KEYWORD_DOMAIN_MAP = {

    "Criminal Law": [
        "criminal conspiracy",
        "cheating and fraud",
        "criminal breach of trust",
        "bail application",
        "suspension of sentence",
        "charge sheet",
        "first information report",
        "sessions court",
        "conviction",
        "acquittal",
        "accused",
        "offence",
        "murder",
        "kidnapping",
        "assault",
        "theft",
        "robbery",
        "extortion",
        "forgery",
        "dacoity",
        "culpable homicide",
        "grievous hurt",
        "wrongful confinement",
        "criminal intimidation",
        "abetment",
    ],

    "Child Protection (POCSO)": [
        "pocso",
        "protection of children from sexual offences",
        "special court pocso",
        "child victim",
        "minor victim",
        "sexual assault on child",
        "sexual abuse of minor",
        "child pornography",
        "penetrative sexual assault",
        "aggravated sexual assault",
        "child witness",
        "in camera trial",
    ],

    "Cyber Law": [
        "cyber crime",
        "hacking",
        "data breach",
        "online fraud",
        "unauthorized access to computer",
        "digital fraud",
        "cyber attack",
        "phishing",
        "cyber stalking",
        "electronic evidence",
        "information technology act",
        "computer related offence",
        "social media fraud",
        "identity theft online",
    ],

    "Property Law": [
        "sale deed",
        "title dispute",
        "partition of property",
        "ancestral property",
        "joint family property",
        "coparcenary",
        "land dispute",
        "ownership dispute",
        "property transfer",
        "registry of property",
        "gift deed",
        "will and succession",
        "adverse possession",
        "easement rights",
        "mortgage",
        "eviction from property",
        "encroachment",
        "boundary dispute",
    ],

    "Contract Law": [
        "breach of contract",
        "specific performance",
        "contractual obligation",
        "void contract",
        "voidable contract",
        "offer and acceptance",
        "consideration",
        "contract breached",
        "terms of agreement",
        "non-performance of contract",
        "agreement not honoured",
        "contract not fulfilled",
        "liquidated damages",
        "penalty clause",
        "novation",
        "indemnity",
    ],

    "Family Law": [
        "divorce petition",
        "matrimonial dispute",
        "child custody",
        "alimony",
        "maintenance of wife",
        "maintenance of children",
        "cruelty by spouse",
        "desertion",
        "restitution of conjugal rights",
        "judicial separation",
        "mutual consent divorce",
        "domestic violence",
        "dowry harassment",
        "guardianship of minor",
        "adoption of child",
        "hindu marriage act",
        "muslim divorce",
        "spousal rights",
    ],

    "Banking Law": [
        "cheque bounce",
        "cheque dishonour",
        "negotiable instruments act",
        "section 138",
        "loan default",
        "bank recovery",
        "non performing asset",
        "debt recovery tribunal",
        "sarfaesi act",
        "banking fraud",
        "financial liability",
        "overdraft",
        "letter of credit",
        "bank guarantee",
    ],

    "Consumer Protection": [
        "consumer complaint",
        "deficiency in service",
        "unfair trade practice",
        "consumer forum",
        "consumer court",
        "national consumer disputes redressal commission",
        "district consumer forum",
        "state consumer commission",
        "defective goods",
        "product liability",
        "delay in possession",
        "builder consumer dispute",
        "medical negligence consumer",
        "insurance claim deficiency",
        "service provider complaint",
        "compensation to consumer",
        "misleading advertisement",
    ],

    "Labour Law": [
        "wrongful termination",
        "industrial dispute",
        "labour court",
        "workman",
        "reinstatement",
        "back wages",
        "retrenchment",
        "strike lockout",
        "trade union",
        "provident fund dispute",
        "gratuity dispute",
        "employment dispute",
        "minimum wages",
        "factory worker rights",
        "unfair labour practice",
    ],

    "Corporate Law": [
        "insolvency",
        "bankruptcy",
        "corporate insolvency resolution",
        "national company law tribunal",
        "nclt",
        "corporate fraud",
        "shareholder dispute",
        "company mismanagement",
        "winding up",
        "liquidation",
        "resolution plan",
        "insolvency and bankruptcy code",
        "ibc proceedings",
    ],

    "Company Law": [
        "companies act",
        "director liability",
        "board of directors",
        "annual general meeting",
        "memorandum of association",
        "articles of association",
        "oppression and mismanagement",
        "minority shareholder",
        "company secretary",
        "corporate governance",
        "share allotment",
        "debentures",
    ],

    "Environmental Law": [
        "environmental pollution",
        "pollution control board",
        "environment impact assessment",
        "forest conservation",
        "wildlife protection",
        "ecological damage",
        "hazardous waste",
        "environmental clearance",
        "green tribunal",
        "national green tribunal",
        "deforestation",
        "water pollution",
        "air pollution",
    ],

    "Arbitration Law": [
        "arbitration award",
        "arbitral tribunal",
        "setting aside award",
        "section 34 arbitration",
        "section 37 arbitration",
        "conciliation",
        "dispute resolution clause",
        "arbitration agreement",
        "enforcement of award",
        "international arbitration",
        "domestic arbitration",
        "arbitrator appointment",
    ],

    "Tax Law": [
        "income tax assessment",
        "tax demand",
        "tax evasion",
        "income tax return",
        "gst dispute",
        "goods and services tax",
        "tax tribunal",
        "income tax appellate tribunal",
        "itat",
        "tax refund",
        "tax penalty",
        "customs duty",
        "excise duty",
        "tax avoidance",
    ],

    "Insurance Law": [
        "insurance claim",
        "claim repudiation",
        "insurance policy",
        "premium payment",
        "insurer liability",
        "life insurance dispute",
        "motor accident claim",
        "health insurance claim",
        "fire insurance",
        "marine insurance",
        "insurance fraud",
        "irdai",
        "insurance ombudsman",
    ],

    "Competition Law": [
        "competition commission of india",
        "cci",
        "anti competitive agreement",
        "abuse of dominant position",
        "cartel",
        "price fixing",
        "market dominance",
        "monopoly",
        "predatory pricing",
        "merger control",
        "combination regulation",
    ],

    "Constitutional Law": [
        "fundamental rights",
        "writ petition",
        "article 14",
        "article 19",
        "article 21",
        "article 32",
        "article 226",
        "constitution bench",
        "constitutional validity",
        "judicial review",
        "habeas corpus",
        "mandamus",
        "certiorari",
        "prohibition writ",
        "quo warranto",
        "directive principles",
        "constitutional amendment",
        "separation of powers",
    ],

    "Administrative Law": [
        "administrative action",
        "quasi judicial authority",
        "government order challenged",
        "public authority",
        "natural justice",
        "audi alteram partem",
        "administrative tribunal",
        "central administrative tribunal",
        "service matter",
        "government employee dispute",
        "regulatory authority",
        "administrative decision",
        "delegated legislation",
        "ultra vires",
    ],

    "Intellectual Property Law": [
        "copyright infringement",
        "trademark infringement",
        "patent infringement",
        "intellectual property rights",
        "passing off",
        "ip dispute",
        "brand protection",
        "licensing agreement ip",
        "trade secret",
        "design infringement",
        "geographical indication",
        "royalty dispute",
        "counterfeit goods",
    ],

    "Civil Law": [
        "civil suit",
        "plaintiff",
        "decree",
        "injunction order",
        "permanent injunction",
        "temporary injunction",
        "suit for recovery",
        "declaratory suit",
        "civil court",
        "appeal civil",
        "code of civil procedure",
        "order 7 rule 11",
        "res judicata",
        "limitation act",
    ],
}