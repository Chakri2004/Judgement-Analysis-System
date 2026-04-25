import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
    """
    Extract clean text from legal PDF (removes headers, page numbers, noise)
    """
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if not text:
                continue

            text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.I)

            text = re.sub(r'CRL\.[A-Z\.]+\s*\d+\/\d+', '', text)

            text = re.sub(r'\s+', ' ', text)

            full_text += text.strip() + "\n"

    return full_text