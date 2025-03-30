import os
import cv2
import pandas as pd
import glob
import string
import numpy as np
from skimage.feature import local_binary_pattern
from scrape_text import extract_text
from scrape_screenshots import screenshots_for_tiers
from sklearn.feature_extraction.text import TfidfVectorizer

TIER1 = "clones/tier1"
TIER2 = "clones/tier2"
TIER3 = "clones/tier3"
TIER4 = "clones/tier4"

tier1_files = glob.glob(os.path.join(TIER1, "*.html"))
tier2_files = glob.glob(os.path.join(TIER2, "*.html"))
tier3_files = glob.glob(os.path.join(TIER3, "*.html"))
tier4_files = glob.glob(os.path.join(TIER4, "*.html"))

# Extragere Features folosind metoda TF_IDF

def preprocess_text(text):
    """Lowercases and removes punctuation from text."""
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def extract_tfidf_features(data):
    """Extracts TF-IDF features from text content."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(data.values())
    return {key: tfidf_matrix[i].toarray()[0] for i, key in enumerate(data.keys())}

# Extragere features folosind histogram LBP

def lbp_histogram(image, num_points=24, radius=3, bins=32):
    """Extracts Local Binary Pattern (LBP) histogram."""
    lbp = local_binary_pattern(image, num_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    return hist


def extract_features(file):
    if file == 1:
        SCREENSHOT_DIR = "screenshots1"
        tier = tier1_files
        path = "clones/tier1\\"
    elif file == 2:
        SCREENSHOT_DIR = "screenshots2"
        tier = tier2_files
        path = "clones/tier2\\"
    elif file == 3:
        SCREENSHOT_DIR = "screenshots3"
        tier = tier3_files
        path = "clones/tier3\\"
    elif file == 4:
        SCREENSHOT_DIR = "screenshots4"
        tier = tier4_files
        path = "clones/tier4\\"
        
    text = {file: extract_text(file) for file in tier}

    #daca nu exista fisier se creaza
    if (not os.path.isdir(SCREENSHOT_DIR)) or not os.listdir(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        screenshots_for_tiers(tier)

    tfidf_features = extract_tfidf_features(text)

    screenshots = glob.glob(os.path.join(SCREENSHOT_DIR, "*.png"))
    texture_features = {}

    for screenshot in screenshots:
        image = cv2.imread(screenshot, cv2.IMREAD_GRAYSCALE)
        histo = lbp_histogram(image)
        
        texture_features[path + os.path.basename(screenshot).replace("_", ".").replace(".png", ".html")] = histo # cheia este calea catre documetul html

    # se combina datele texuale cu cele ce provin de la imagini
    common_keys = set(tfidf_features.keys()) & set(texture_features.keys())
    combined_features = {key: np.concatenate((tfidf_features[key], texture_features[key])) for key in common_keys}

    df = pd.DataFrame.from_dict(combined_features, orient="index")
    shape_text = len(next(iter(tfidf_features.values())))
    shape_img = len(next(iter(texture_features.values())))

    df.to_csv("combined_features.csv", index=True, header=False)

    print("✅ Features saved to combined_features4.csv")
    return shape_text, shape_img