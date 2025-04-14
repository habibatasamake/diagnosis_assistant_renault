import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch

# 📁 Dossier local pour stocker le modèle
local_dir = os.path.expanduser("~/Documents/LLM_model")
os.makedirs(local_dir, exist_ok=True)
print(f"📂 Dossier du modèle : {local_dir}")

# 🔧 Configuration pour quantification 4-bit (nf4) avec bitsandbytes
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 🧠 ID du modèle à télécharger
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# 📥 Télécharger et sauvegarder le tokenizer
print("⏳ Téléchargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(local_dir)

# 📥 Télécharger et sauvegarder le modèle (peut prendre quelques minutes)
print("⏳ Téléchargement du modèle quantifié 4-bit (nf4)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=quant_config
)
model.save_pretrained(local_dir)

print("\n✅ Modèle Mistral-7B-Instruct téléchargé et prêt à l'usage local.")
print(f"📌 Utilisez ce chemin pour charger le modèle : '{local_dir}'")
