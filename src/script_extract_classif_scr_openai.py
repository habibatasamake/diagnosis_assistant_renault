import pandas as pd
from PyPDF2 import PdfReader
from json_repair import repair_json
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import openai

class ExtracteurSCROpenAI:
    def __init__(self, url, openai_api_key):
        """
        - url : chemin complet vers le PDF (ex : "/content/drive/MyDrive/chemin/vers/doc.pdf")
        - openai_api_key : clé API OpenAI (doit être définie pour accéder à GPT-4)
        """
        self.url = url
        self.pdf_reader = PdfReader(self.url)
        openai.api_key = openai_api_key
        self.df_scr = pd.DataFrame()

    def extract_equipment(self):
        """
        Extrait le nom de l'équipement depuis la première page du document.
        """
        first_page_text = self.pdf_reader.pages[0].extract_text()
        prompt = f"""
{first_page_text}

What is the equipment this document is dealing with? Return only the response.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    def alimentation_df(self, data_json, page_num, equipment):
        """
        Transforme la sortie JSON en DataFrame.
        Gère le cas où chaque item est un dict ou une liste.
        """
        rows = []
        for defect in data_json:
            if isinstance(defect, dict):
                symptom = defect.get("symptom", "Unknown")
                cause = defect.get("cause", "Unknown")
                remedy = defect.get("remedy", "Unknown")
            elif isinstance(defect, list):
                symptom = defect[0] if len(defect) > 0 else "Unknown"
                cause = defect[1] if len(defect) > 1 else "Unknown"
                remedy = defect[2] if len(defect) > 2 else "Unknown"
            else:
                symptom, cause, remedy = "Unknown", "Unknown", "Unknown"
            row = {
                "URL": self.url.rsplit("/", 1)[-1],
                "equipment": equipment,
                "page": page_num,
                "symptom": symptom,
                "cause": cause,
                "remedy": remedy,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def process_page(self, page_num, equipment):
        """
        Traite une page du PDF pour extraire les défauts (SCR) via le prompt.
        """
        page_text = self.pdf_reader.pages[page_num - 1].extract_text()
        if not page_text.strip():
            return pd.DataFrame([], columns=["URL", "equipment", "page", "symptom", "cause", "remedy"])
        
        prompt = f"""
{page_text}

Extract all defects and their associated causes and remedies from the provided text. Return a JSON array.
There can be more than one line for a single defect if it has different causes and if a cause has different remedies,
one line should represent a unique group of a symptom, a cause, and a remedy.

* symptom: A description of the defect. Include any error codes mentioned.
* cause: A possible explanation for the defect.
* remedy: The suggested solution or troubleshooting steps.

If a defect lacks one or more of these components (symptom, cause, or remedy), include the missing information as "Unknown".

Example JSON format:
[
    {{
        "symptom": "PNT1-166 Linear Potentiometer Unstable",
        "cause": "During Auto Calibration, the feedback from the linear potentiometer revealed large fluctuations.",
        "remedy": "Change the applicator and repair the malfunctioning linear potentiometer."
    }}
]
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000
        )
        generated_text = response.choices[0].message.content
        repaired_json = repair_json(generated_text)
        try:
            data_json = json.loads(repaired_json)
        except json.JSONDecodeError:
            data_json = []
        return self.alimentation_df(data_json, page_num, equipment)

    def extract_defects(self, start_page=0, end_page=0):
        """
        Extrait les défauts SCR du document entre start_page et end_page.
        """
        if end_page == 0:
            end_page = len(self.pdf_reader.pages)
        equipment = self.extract_equipment()
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(lambda page: self.process_page(page, equipment),
                             range(start_page, end_page + 1)),
                total=end_page - start_page + 1,
                desc="Extraction SCR",
                leave=False
            ))
        self.df_scr = pd.concat(results, ignore_index=True)
        self.df_scr = self.df_scr[["URL", "equipment", "page", "symptom", "cause", "remedy"]]
        self.df_scr.sort_values(by="page", ascending=True, inplace=True)
        return self.df_scr

    def classify_document(self, start_page=0, end_page=0):
        """
        Classe le document pour vérifier s'il est structuré explicitement en sections
        "Symptoms", "Causes", "Remedies". Extraction du texte des pages entre start_page et end_page 
        (si end_page=0, utilisation de l'ensemble du document) et renvoie True si la réponse contient "yes".
        """
        if end_page == 0:
            end_page = len(self.pdf_reader.pages)
        texts = []
        for i in range(start_page, end_page):
            text = self.pdf_reader.pages[i].extract_text()
            if text and text.strip():
                texts.append(text)
        full_text = "\n\n".join(texts)
        prompt = f"""
Here is an excerpt from a technical document:

{full_text}

Based on the text above, determine whether this document is structured in a way that facilitates the extraction of defects and their associated causes and remedies using a simple regex extraction. In particular, check if the document clearly delineates sections or markers corresponding to:
* symptom: a description of the defect (including any error codes),
* cause: a possible explanation for the defect,
* remedy: the suggested solution or troubleshooting steps.

Answer only "Yes" or "No".
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
