import os
import glob
from bs4 import BeautifulSoup


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


def extract_text(path):
    with open(path, "r", encoding="utf") as file:
        html = file.read()

    soup = BeautifulSoup(html, "html.parser")
    elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "li"])
    text = " ".join([el.get_text() for el in elements])

    return text.strip()
