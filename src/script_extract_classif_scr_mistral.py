import os
import pandas as pd
from PyPDF2 import PdfReader
from json_repair import repair_json
import json
from tqdm import tqdm
from llama_cpp import Llama


class Extracteur_SCR_Mistral:
    def __init__(self, url, model_path="~/llama_models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=4096):
        self.url = url
        self.pdf_reader = PdfReader(self.url)
        self.df_scr = pd.DataFrame()

        self.model_path = os.path.expanduser(model_path)
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            use_mlock=True,
            verbose=False
        )

    def query_llm(self, prompt, max_tokens=1024):
        response = self.llm(
            f"[INST] {prompt.strip()} [/INST]",
            max_tokens=max_tokens,
            stop=["</s>"]
        )
        return response["choices"][0]["text"].strip()

    def extract_equipment(self):
        first_page_text = self.pdf_reader.pages[0].extract_text()
        prompt = f"""
{first_page_text}

What is the equipment this document is dealing with? Return only the response.
"""
        return self.query_llm(prompt, max_tokens=100)

    def alimentation_df(self, data_json, page_num, equipment):
        rows = []
        for defect in data_json:
            row = {
                "URL": self.url.rsplit("/", 1)[-1],
                "equipment": equipment,
                "page": page_num,
                "symptom": defect.get("symptom", "Unknown"),
                "cause": defect.get("cause", "Unknown"),
                "remedy": defect.get("remedy", "Unknown"),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def process_page(self, page_num, equipment):
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
    {
        "symptom": "PNT1-166 Linear Potentiometer Unstable",
        "cause": "During Auto Calibration, the feedback from the linear potentiometer revealed large fluctuations.",
        "remedy": "Change the applicator and repair the malfunctioning linear potentiometer."
    }
]
"""
        response = self.query_llm(prompt, max_tokens=2000)
        repaired_json = repair_json(response)
        try:
            data_json = json.loads(repaired_json)
        except json.JSONDecodeError:
            data_json = []
        return self.alimentation_df(data_json, page_num, equipment)

    def extract_defects(self, start_page=0, end_page=0):
        if end_page == 0:
            end_page = len(self.pdf_reader.pages)
        equipment = self.extract_equipment()
        results = []
        for page in tqdm(range(start_page, end_page + 1), desc="Extraction SCR"):
            df_page = self.process_page(page, equipment)
            results.append(df_page)
        self.df_scr = pd.concat(results, ignore_index=True)
        self.df_scr = self.df_scr[["URL", "equipment", "page", "symptom", "cause", "remedy"]]
        self.df_scr.sort_values(by="page", ascending=True, inplace=True)
        return self.df_scr

    def classify_document_par_chunks(self, start_page=0, end_page=0, chunk_size_chars=3000):
        if end_page == 0:
            end_page = len(self.pdf_reader.pages)

        texts = []
        for i in range(start_page, end_page):
            text = self.pdf_reader.pages[i].extract_text()
            if text and text.strip():
                texts.append(text)
        full_text = "\n\n".join(texts)

        chunks = [full_text[i:i + chunk_size_chars] for i in range(0, len(full_text), chunk_size_chars)]

        results = []
        for idx, chunk in enumerate(tqdm(chunks, desc="Classification SCR chunks"), 1):
            prompt = f"""
Here is an excerpt from a technical document:

{chunk}

Based on the text above, determine whether this excerpt is structured in a way that facilitates the extraction of defects and their associated causes and remedies using a simple regex extraction. In particular, check if the excerpt clearly delineates sections or markers corresponding to:
* symptom: a description of the defect (including any error codes),
* cause: a possible explanation for the defect,
* remedy: the suggested solution or troubleshooting steps.

Answer only \"Yes\" or \"No\".
"""
            response = self.query_llm(prompt, max_tokens=10).lower()
            results.append("yes" in response)

        yes_count = sum(results)
        total_chunks = len(results)
        return f"{yes_count}/{total_chunks} chunks structured SCR"
