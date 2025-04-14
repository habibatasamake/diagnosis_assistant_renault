# script_extracteur_scr_mistral_local.py

import pandas as pd
from PyPDF2 import PdfReader
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from json_repair import repair_json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ExtracteurSCRMistralLocal:
    def __init__(self, url, model_id="mistralai/Mistral-7B-Instruct-v0.2", load_in_4bit=True):
        """
        Initialise l'extracteur :
         - url : chemin complet vers le PDF (ex : "/content/drive/MyDrive/chemin/vers/doc.pdf")
         - model_id : identifiant du modèle sur Hugging Face.
         - load_in_4bit : booléen pour charger le modèle en 4 bits.
        """
        self.url = url
        self.pdf_reader = PdfReader(self.url)
        # Charger tokenizer et modèle en 4 bits
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        # Créer le pipeline de génération
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=1024,
            do_sample=False  # réponses déterministes
        )
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
        response = self.pipe(prompt, max_new_tokens=100, temperature=0.1)[0]["generated_text"]
        return response.strip()

    def alimentation_df(self, data_json, page_num, equipment):
        """
        Transforme la sortie JSON en DataFrame.
        Gère le cas où chaque item est soit un dict, soit une liste.
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
        response = self.pipe(prompt, max_new_tokens=2000, temperature=0.1)[0]["generated_text"]
        repaired_json = repair_json(response)
        try:
            data_json = json.loads(repaired_json)
        except json.JSONDecodeError:
            # Si la réparation ne suffit pas, on retourne une liste vide
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
                executor.map(lambda page: self.process_page(page, equipment), range(start_page, end_page + 1)),
                total=end_page - start_page + 1,
                desc="Extraction SCR",
                leave=False
            ))
        self.df_scr = pd.concat(results, ignore_index=True)
        self.df_scr = self.df_scr[["URL", "equipment", "page", "symptom", "cause", "remedy"]]
        self.df_scr.sort_values(by="page", ascending=True, inplace=True)
        return self.df_scr

