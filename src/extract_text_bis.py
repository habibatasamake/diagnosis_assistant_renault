import re

SYNONYMS = {
    "cause": ["Cause", "Explanation", "Explication"],
    "remedy": ["Remedy", "Action", "Mesure"]
}

def extract_triplets_from_text(text):
    triplets = []
    section_pattern = r"\b\d+(?:\.\d+){0,4}\.?"  # supporte 1 à 5 niveaux

    # === PARTIE 1 : Cas structurés avec numérotation
    blocks = re.findall(
        rf"{section_pattern} [A-Z]{{2,}}-\d{{3}}.*?(?=\n{section_pattern} [A-Z]{{2,}}-\d{{3}}|\Z)",
        text,
        re.DOTALL
    )

    for block in blocks:
        triplet = {"symptom": None, "cause": None, "remedy": None}

        # Symptom = toute la ligne contenant le code
        symptom_match = re.search(r"^.*?([A-Z]{2,}-\d{3}).*?$", block, flags=re.MULTILINE)
        if symptom_match:
            triplet["symptom"] = symptom_match.group(0).strip()

        # Cause
        for cause_kw in SYNONYMS["cause"]:
            match = re.search(rf"\(?{cause_kw}\)?[:：]?\s*(.*?)(?=\n|$)", block, flags=re.IGNORECASE)
            if match:
                triplet["cause"] = match.group(1).strip()
                break

        # Remedy (accumuler toutes les variantes)
        remedy_texts = []
        for rem_kw in SYNONYMS["remedy"]:
            matches = re.findall(rf"\(?{rem_kw}\s*\d*\)?[:：]?\s*(.*?)(?=\n|$)", block, flags=re.IGNORECASE)
            remedy_texts.extend([m.strip() for m in matches])
        if remedy_texts:
            triplet["remedy"] = "\n".join(remedy_texts)

        if sum(1 for v in triplet.values() if v and v.strip()) >= 2:
            triplets.append(triplet)

    # === PARTIE 2 : Cas simples avec seulement [A-Z]{2,}-\d{3}
    simple_blocks = re.findall(r"([A-Z]{2,}-\d{3}.*?)(?=([A-Z]{2,}-\d{3})|\Z)", text, flags=re.DOTALL)

    for raw_block, _ in simple_blocks:
        triplet = {"symptom": None, "cause": None, "remedy": None}

        # Symptom = première ligne entière
        lines = raw_block.strip().split("\n")
        if lines:
            triplet["symptom"] = lines[0].strip()

        # Cause via synonymes
        for cause_kw in SYNONYMS["cause"]:
            match = re.search(rf"\(?{cause_kw}\)?[:：]?\s*(.*?)(?=\n|\(?({'|'.join(SYNONYMS['remedy'])})|\Z)", raw_block, re.DOTALL | re.IGNORECASE)
            if match:
                triplet["cause"] = match.group(1).strip()
                break

        # Remedy via synonymes
        remedy_lines = []
        for rem_kw in SYNONYMS["remedy"]:
            matches = re.findall(rf"\(?{rem_kw}\s*\d*\)?[:：]?\s*(.*?)(?=\n|\(?({'|'.join(SYNONYMS['remedy'])})|\Z)", raw_block, re.DOTALL | re.IGNORECASE)
            remedy_lines.extend([m[0].strip() for m in matches])

        if remedy_lines:
            triplet["remedy"] = "\n".join(remedy_lines)

        # Si pas de match mais lignes restantes : prendre comme fallback
        if not triplet["remedy"]:
            fallback = "\n".join(
                l for l in lines[1:] if all(kw.lower() not in l.lower() for kw in SYNONYMS["cause"])
            ).strip()
            if fallback:
                triplet["remedy"] = fallback

        if sum(1 for v in triplet.values() if v and v.strip()) >= 2:
            triplets.append(triplet)

    return triplets


def convert_pdf_to_json_with_progress(pdf_path):
    """
    Convert a PDF file to a JSON object with its text content by page,
    and display a progress bar during processing.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: A JSON-like dictionary with the file name and text content.
    """
    json_object = {"FileName": "Doc fournisseur", "Text": []}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page in tqdm(pdf.pages, total=total_pages, desc="Processing PDF pages"):
                json_object["Text"].append({
                    "PageNumber": page.page_number,
                    "Raw Content": page.extract_text(),
                    "Clean Content": ""
                })
    except FileNotFoundError:
        raise Exception(f"File not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"An error occurred while processing the PDF: {e}")

    return json_object


class ExtracteurSCR:
    def __init__(self, path, meta):
        self.SYNONYMS = {
            "cause": ["Cause", "Explanation", "Explication"],
            "remedy": ["Remedy", "Action", "Mesure"]
        }
        self.path = path
        self.meta = meta
        self.liste_triplets_scr = []
        self.parsed_doc = None

    def parser_doc(self):
      self.parsed_doc = convert_pdf_to_json_with_progress(self.path)
    
    def extract_triplets_from_text(self):
      liste_triplets_scr = []
      for page in self.parsed_doc['Text']:
        triplets = extract_triplets_from_text(page['Raw Content'])
        liste_triplets_scr.extend(triplets)
      return liste_triplets_scr  


extracteur = ExtracteurSCR(path_to_doc_fournisseur, doc_fournisseur)
extracteur.parser_doc()
triplet = extracteur.extract_triplets_from_text()
