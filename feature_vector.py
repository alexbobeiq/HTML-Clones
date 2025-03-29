import os
import cv2
import pandas as pd
import glob
import string
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

# --- TEXTUAL FEATURE EXTRACTION ---

def preprocess_text(text):
    """Lowercases, removes punctuation, and splits text into words."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()

def build_vocab(data):
    """Build vocabulary from text content."""
    vocabulary = set()
    for text_in_site in data.values():
        words = preprocess_text(text_in_site)
        vocabulary.update(words)
    return sorted(vocabulary)

def bow(data, vocab):
    """Create Bag-of-Words (BoW) representation for text."""
    matrix = {}
    for index, text in data.items():
        words = preprocess_text(text)
        matrix_row = [words.count(word) for word in vocab]
        matrix[index] = matrix_row
    return matrix

text = {file: extract_text(file) for file in tier3_files}
vocab = build_vocab(text)
bow = bow(text, vocab)

# --- TEXTURE FEATURE EXTRACTION WITH ORB ---

def orb_descriptor_histogram(image):
    """Extracts ORB descriptors and creates a histogram of feature descriptors."""
    orb = cv2.ORB_create(nfeatures=500)  # Limit features for efficiency
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    if descriptors is None:  
        return np.zeros(32)  # If no descriptors found, return zero vector
    
    hist, _ = np.histogram(descriptors.ravel(), bins=32, range=(0, 256), density=True)
    return hist

screenshots = glob.glob(os.path.join(SCREENSHOT_DIR, "*.png"))
texture_features = {}

for screenshot in screenshots:
    image = cv2.imread(screenshot, cv2.IMREAD_GRAYSCALE)
    histo = orb_descriptor_histogram(image)
    
    # Convert screenshot filename to corresponding HTML path
    texture_features["clones/tier3\\" + os.path.basename(screenshot).replace("_", ".").replace(".png", ".html")] = histo

# --- COMBINE TEXTUAL AND TEXTURE FEATURES ---

common_keys = set(bow.keys()) & set(texture_features.keys())
combined_features = {key: np.concatenate((bow[key], texture_features[key])) for key in common_keys}

# Save to CSV
df = pd.DataFrame.from_dict(combined_features, orient="index")
df.to_csv("combined_features3.csv", index=True, header=False)

print("✅ Features saved to combined_features3.csv")
