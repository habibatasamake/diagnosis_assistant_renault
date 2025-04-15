from sentence_transformers import SentenceTransformer, util
import pandas as pd

class SemanticSymptomSearcher:
    def __init__(self, df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2"):
        self.df = df.copy()
        self.model = SentenceTransformer(model_name)
        self.symptom_embeddings = None
        self._prepare()

    def _prepare(self):
        self._preprocess()
        self._encode_symptoms()

    def _preprocess(self):
        self.df = self.df.dropna(subset=['Symptom'])  # Enlever les NaN
        self.df['Symptom'] = self.df['Symptom'].astype(str).str.strip()
        self.df['Remedy'] = self.df['Remedy'].fillna('').astype(str).str.strip()

    def _encode_symptoms(self):
        self.symptoms = self.df['Symptom'].tolist()
        self.symptom_embeddings = self.model.encode(self.symptoms, convert_to_tensor=True)

    def search(self, query: str, top_k: int = 10):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.symptom_embeddings)[0]
        top_results = cos_scores.topk(k=top_k)

        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            row = self.df.iloc[idx.item()]
            results.append({
                "Symptom": row['Symptom'],
                "Remedy": row['Remedy'],
                "Score": round(score.item(), 4)
            })
        return results
