import os
import pandas as pd
import glob
import string
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from scrape_text import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
TIER1 = "clones/tier1"
TIER2 = "clones/tier2"
TIER3 = "clones/tier3"
TIER4 = "clones/tier4"

tier1_files = glob.glob(os.path.join(TIER1, "*.html"))
tier2_files = glob.glob(os.path.join(TIER2, "*.html"))
tier3_files = glob.glob(os.path.join(TIER3, "*.html"))
tier4_files = glob.glob(os.path.join(TIER4, "*.html"))


def preprocess_text(text):
    """Lowercases, removes punctuation, and splits text into words."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()

def build_vocab(data):
    vocabulary = set()
    for text_in_site in data.values():
        words = preprocess_text(text_in_site)
        vocabulary.update(words)

    return sorted(vocabulary)

def bow(data, vocab):
    matrix = {}
    for index, text in data.items():
        words = preprocess_text(text)
        matrix_row = [words.count(word) for word in vocab]
        matrix[index] = matrix_row
    return matrix

text = {file: extract_text(file) for file in tier3_files}

vocab = build_vocab(text)
bow = bow(text, vocab)
print(bow)

df = pd.DataFrame.from_dict(bow, orient="index")

df.to_csv("combined_features.csv", index=True, header=False)

print("✅ Features saved to combined_features.csv")


 

""" df = pd.DataFrame.from_dict(combined_features, orient="index")

df.to_csv("combined_features.csv", index=True, header=False)

print("✅ Features saved to combined_features.csv") """

