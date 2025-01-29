import re
import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm
import json

class TreeNodeV2:
    """
    Classe représentant un nœud dans un arbre hiérarchique.
    Permet d'extraire des causes et remèdes à partir de son contenu.
    """
    def __init__(self, name, content=None):
        self.name = name
        self.content = content if content else ""
        self.children = []
        self.cause = None
        self.remedy = None

    def add_child(self, node):
        self.children.append(node)

    def find_or_create_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        new_child = TreeNodeV2(name)
        self.add_child(new_child)
        return new_child

    def is_child(self, name):
        """Vérifie si un nœud est un enfant direct basé sur sa hiérarchie."""
        return self._is_child(self.name, name)

    @staticmethod
    def _is_child(name1, name2):
        """Méthode statique pour vérifier la relation parent-enfant."""
        return name1 in name2 and name2.count('.') == name1.count('.') + 1

    def extract_cause_remedy(self):
        """Analyse le contenu pour extraire les causes et remèdes."""
        cause_match = re.search(r"Cause:\s*(.*?)(\n|$)", self.content, re.DOTALL)
        remedy_match = re.search(r"Remedy:\s*(.*?)(\n|$)", self.content, re.DOTALL)

        if cause_match:
            self.cause = cause_match.group(1).strip()
            self.content = self.content.replace(cause_match.group(0), "").strip()
        if remedy_match:
            self.remedy = remedy_match.group(1).strip()
            self.content = self.content.replace(remedy_match.group(0), "").strip()

    def to_dict(self):
        """Convertit le nœud et ses enfants en dictionnaire JSON."""
        node_dict = {
            "name": self.name,
            "content": self.content,
            "children": [child.to_dict() for child in self.children]
        }
        if self.cause:
            node_dict["cause"] = self.cause
        if self.remedy:
            node_dict["remedy"] = self.remedy
        return node_dict

    def to_string(self, level=0):
        """Affiche l'arbre sous forme textuelle."""
        ret = "  " * level + f"{self.name}: {self.content}\n"
        if self.cause:
            ret += "  " * (level + 1) + f"Cause: {self.cause}\n"
        if self.remedy:
            ret += "  " * (level + 1) + f"Remedy: {self.remedy}\n"
        for child in self.children:
            ret += child.to_string(level + 1)
        return ret


class PDFProcessor:
    """
    Classe pour convertir un PDF en JSON avec extraction des causes et remèdes.
    """
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.json_data = {"FileName": "Doc fournisseur", "Text": []}

    def convert_with_pdfplumber(self):
        """Convertit un PDF en JSON en utilisant pdfplumber."""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in tqdm(pdf.pages, total=len(pdf.pages), desc="Processing PDF"):
                    raw_content = page.extract_text()
                    clean_content_tree = self._build_tree(raw_content).to_dict()
                    self.json_data["Text"].append({
                        "PageNumber": page.page_number,
                        "Raw Content": raw_content,
                        "Clean Content": clean_content_tree
                    })
        except FileNotFoundError:
            raise Exception(f"File not found: {self.pdf_path}")
        except Exception as e:
            raise Exception(f"Error processing PDF: {e}")
        return self.json_data

    def convert_with_pymupdf(self):
        """Convertit un PDF en JSON en utilisant PyMuPDF."""
        try:
            pdf_document = fitz.open(self.pdf_path)
            for page_num in tqdm(range(pdf_document.page_count), desc="Processing PDF"):
                page = pdf_document[page_num]
                raw_content = page.get_text("text")
                clean_content_tree = self._build_tree(raw_content).to_dict()
                self.json_data["Text"].append({
                    "PageNumber": page_num + 1,
                    "Raw Content": raw_content,
                    "Clean Content": clean_content_tree
                })
            pdf_document.close()
        except FileNotFoundError:
            raise Exception(f"File not found: {self.pdf_path}")
        except Exception as e:
            raise Exception(f"Error processing PDF: {e}")
        return self.json_data

    @staticmethod
    def _build_tree(text):
        """Construit un arbre hiérarchique à partir du texte structuré."""
        root = TreeNodeV2("root")
        nodes_by_level = {"root": root}

        for line in text.split("\n"):
            line = re.sub(r"\s+", " ", line.strip())
            match = re.match(r"^(\d+(\.\d+)*\.?)\s+(.*)", line)

            if match:
                section, _, content = match.groups()
                section = section.strip(".")
                level = section.count(".")

                parent_section = ".".join(section.split(".")[:-1]) if level > 0 else "root"
                parent_node = nodes_by_level.get(parent_section, root)

                if TreeNodeV2._is_child(parent_section, section):
                    new_node = parent_node.find_or_create_child(section)
                    new_node.content = content
                    new_node.extract_cause_remedy()
                    nodes_by_level[section] = new_node
                else:
                    print(f"Avertissement : '{section}' ne peut pas être ajouté comme enfant de '{parent_section}'.")
            else:
                last_section = list(nodes_by_level.keys())[-1] if nodes_by_level else "root"
                last_node = nodes_by_level.get(last_section, root)
                last_node.content += f"\n{line.strip()}"
                last_node.extract_cause_remedy()

        return root


class JSONHandler:
    """
    Classe pour gérer la sauvegarde des fichiers JSON.
    """
    @staticmethod
    def save_json_to_file(obj, file_path):
        """Sauvegarde un objet JSON dans un fichier."""
        try:
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(obj, json_file, indent=4, ensure_ascii=False)
            print(f"JSON saved successfully to {file_path}")
        except Exception as e:
            raise Exception(f"Failed to save JSON to file: {e}")


# Exemple d'utilisation
if __name__ == "__main__":
    pdf_path = "/Users/habibatasamake/Desktop/MS_IA/Projet_Fil_rouge/Projet_Renault/data/doc-R-30iB.pdf"  # Remplace par ton chemin réel
    output_path = "tree_with_cause_remedy.json"

    # Choix du convertisseur
    processor = PDFProcessor(pdf_path)
    json_data = processor.convert_with_pdfplumber()  # Ou processor.convert_with_pymupdf()

    # Sauvegarde en fichier JSON
    JSONHandler.save_json_to_file(json_data, output_path)