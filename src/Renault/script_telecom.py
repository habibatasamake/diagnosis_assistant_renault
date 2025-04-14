# !pip install --upgrade google-cloud-aiplatform
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
import base64
from google.cloud import storage
import pandas as pd
from PyPDF2 import PdfReader
import io
import json
from json_repair import repair_json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

os.environ['http_proxy'] = "http://iac-proxy.cnz.renault.gcp:80"
os.environ['https_proxy'] = "http://iac-proxy.cnz.renault.gcp:80"
os.environ['no_proxy'] = "127.0.0.1,localhost,metadata.google.internal,googleapis.com"
    
class Extracteur_SCR:
    def __init__(self, url):
        self.url = "doc/" + url
        self.pdf_reader = PdfReader(self.url) # Initialisation du "reader"
        self.df_scr = pd.DataFrame() # Initialisation du dataframe de sortie
        # Initialisation des paramètres de génération :
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0
        }
        self.safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            )
        ]

    def extract_equipment(self, nom_modele_equipement="gemini-1.5-flash-002"):
        """
        Extract the equipment name from the first page of the PDF document.
        """
        modele_equipement = GenerativeModel(nom_modele_equipement)
        responses_equipment = modele_equipement.generate_content(
            [
                self.pdf_reader.pages[0].extract_text(), # Extraction du nom de l'équipement à partir de la première page du document (hypothèse à challenger ?)
                "What is the equipment this document is dealing with? Return only the response."
            ],
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            stream=False
        )
        return responses_equipment.text.strip()

    def alimentation_df(self, data_json, page_num, equipment):
        """
        Populate a DataFrame with defect data extracted from JSON.
        """
        rows = []
        for defect in data_json:
            row = {
                "URL": self.url.rsplit("/")[1],
                "equipment": equipment,
                "page": page_num,
                "symptom": defect.get("symptom", "Unknown"),
                "cause": defect.get("cause", "Unknown"),
                "remedy": defect.get("remedy", "Unknown"),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def extract_defects(self, nom_modele_equipement="gemini-1.5-flash-002", nom_modele_scr="gemini-1.5-flash-002", start_page=0, end_page=0):
        """
        Extract defects from the PDF document and return a DataFrame.
        """
        data_frames = []
        equipment = self.extract_equipment(nom_modele_equipement)
        modele_scr = GenerativeModel(nom_modele_scr)

        # Par défaut, travaille sur toutes les pages du document :
        if end_page == 0:
            end_page = len(self.pdf_reader.pages)

        def process_page(page_num):
            """
            Extract defects from a single page of the PDF.
            """
            page = self.pdf_reader.pages[page_num - 1]
            page_text = page.extract_text()

            # Passe les pages sans texte :
            if not page_text.strip():
                return pd.DataFrame([], columns=["URL", "equipment", "page", "symptom", "cause", "remedy"])

            # Configuration d'un agent extrayant les défauts sous la forme (symptôme/cause/remède) :
            responses_scr = modele_scr.generate_content(
                [
                    page_text,
                    """
                    Extract all defects and their associated causes and remedies from the provided text. Return a JSON array.  
                    There can be more than one line for a single defect if it has different causes and if a cause has different remedies, 
                    one line should represent a unique group of a symptom, a cause, and a remedy.

                    * **symptom:** A description of the defect. Include any error codes mentioned.
                    * **cause:** A possible explanation for the defect.
                    * **remedy:** The suggested solution or troubleshooting steps.

                    If a defect lacks one or more of these components (symptom, cause, or remedy), include the missing information as "Unknown".

                    The expected JSON format is the following :
                    [
                      {
                        "symptom": "PNT1-166 Linear Potentiometer Unstable",
                        "cause": "During Auto Calibration, the feedback from the linear potentiometer revealed large fluctuations...",
                        "remedy": "Change the applicator and repair the malfunctioning linear potentiometer..."
                      },
                      {
                        "symptom": "PRIO-377 PMIO ReStart Failed %d",
                        "cause": "Card failed to Start properly.",
                        "remedy": "Please check LEDs on PMIO card... If problem persists please contact Fanuc."
                      },
                      {
                        "symptom": "HOST-178 Router Address Not Defined",
                        "cause": "The router does not have an address listed in the local host table.",
                        "remedy": "If your network has a router, then define an address for it..."
                      }
                    ]
                    """
                ],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=False,
            )

            # Répare le json si le format n'est pas correct :
            repaired_json = repair_json(responses_scr.text)
            data_json = json.loads(repaired_json)

            # Mise sous forme de dataframe
            return self.alimentation_df(data_json, page_num, equipment)

        # Parallélisation par pages de l'extraction des défauts :
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_page, range(start_page, end_page + 1)),
                total=end_page - start_page + 1,
                desc="Lecture Pages",
                leave=False
            ))

        # Concaténation des résultats :
        self.df_scr = pd.concat(results, ignore_index=True)
        self.df_scr = self.df_scr[["URL", "equipment", "page", "symptom", "cause", "remedy"]]
        self.df_scr.sort_values(by="page", ascending=True, inplace=True)

        return self.df_scr
