import os
import re
import unicodedata
from typing import Dict, List

import numpy as np


def read_files_from_directory(directory_path: str) -> List[Dict[str, str]]:
    documents = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    documents.append({filename: clean_text(content)})
            except Exception as e:
                print(f"Could not read file {filename}: {e}")

    return documents


def clean_text(text: str) -> str:
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        return np.zeros_like(x)  
    return (x - x_min) / (x_max - x_min)
