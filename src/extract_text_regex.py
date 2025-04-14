import re
import os
import json
import pdfplumber
import pandas as pd
from tqdm import tqdm

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


def convert_pdf_to_json_with_progress(pdf_path, num_pages=None):
    """
    Convert a PDF file to a JSON object with its text content by page,
    and display a progress bar during processing.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: A JSON-like dictionary with the file name and text content.
    """
    json_object = {"FileName": pdf_path, "Text": []}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            # Déterminer les pages à traiter
            if num_pages is None:
                pages_to_process = range(total_pages)  # toutes les pages
            elif isinstance(num_pages, tuple) and len(num_pages) == 2:
                start, end = num_pages
                if not (1 <= start <= end <= total_pages):
                    raise ValueError(f"Page range must be between 1 and {total_pages}")
                pages_to_process = range(start - 1, end)  # index 0-based
            else:
                raise ValueError("num_pages must be None or a tuple like (start, end)")
            
            for i in tqdm(pages_to_process, total=total_pages, desc="Processing PDF pages"):
                page = pdf.pages[i]
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
    def __init__(self, path, meta=None):
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
      for i, page in enumerate(self.parsed_doc['Text']):
        tmp = {}
        triplets = extract_triplets_from_text(page['Raw Content'])
        tmp['FileName'] = self.parsed_doc['FileName']
        tmp['PageNumber'] = page['PageNumber']
        tmp['Triplets'] = triplets
        liste_triplets_scr.append(tmp)
      self.liste_triplets_scr = liste_triplets_scr
      return liste_triplets_scr


    def export_triplets(self, output_path, format="csv"):
        if not self.liste_triplets_scr:
            print("Aucune donnée à exporter. Appelle extract_triplets_from_text() d'abord.")
            return

        # Aplatir tous les triplets de toutes les pages
        flat_data = []
        for page_data in self.liste_triplets_scr:
            for triplet in page_data['Triplets']:
                flat_data.append({
                    "FileName": page_data['FileName'],
                    "PageNumber": page_data['PageNumber'],
                    #"Code Erreur": triplet.get("symptom", ""),
                    "Symptom": triplet.get("symptom", ""),
                    "Cause": triplet.get("cause", ""),
                    "Remedy": triplet.get("remedy", "")
                })

        # Export CSV
        if format.lower() == "csv":
            df = pd.DataFrame(flat_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Exporté au format CSV : {output_path}")

        # Export JSON
        elif format.lower() == "json":
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(flat_data, f, ensure_ascii=False, indent=2)
            print(f"Exporté au format JSON : {output_path}")

        else:
            print("Format non supporté. Choisis 'csv' ou 'json'.")



def process_folder_v2(folder_path, output_dir="export_triplets", max_files=None, export_format="csv"):
    os.makedirs(output_dir, exist_ok=True)
    csv_paths = []
    file_count = 0

    for filename in tqdm(os.listdir(folder_path), desc="Traitement des PDF"):
        if filename.endswith(".pdf"):
            file_count += 1
            if max_files and file_count > max_files:
                break

            filepath = os.path.join(folder_path, filename)
            try:
                extracteur = ExtracteurSCR(filepath)
                extracteur.parser_doc()
                liste_triplets_scr = extracteur.extract_triplets_from_text()

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"triplets_{base_name}.{export_format}")
                extracteur.export_triplets(output_path, format=export_format)
                csv_paths.append(output_path)

            except Exception as e:
                print(f"Erreur lors du traitement de {filename} : {e}")

    return csv_paths

    
extracteur = ExtracteurSCR(path_to_doc_fournisseur)
extracteur.parser_doc()
triplet = extracteur.extract_triplets_from_text()
folder_path = "/content/drive/MyDrive/pdf_file/data_pdf/simple" # Remplacer par le chemin réel de votre dossier

extracteur.export_triplets(output_path= folder_path + "/triplets.csv", format="csv")
path_to_dataframe = pd.read_csv(folder_path + "/triplets.csv")
df = pd.DataFrame(path_to_dataframe)
nb = len(df)
