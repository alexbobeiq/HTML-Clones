import os
import glob
import re
from bs4 import BeautifulSoup


# extragere text din site
def extract_text(path):
    with open(path, "r", encoding="utf") as file:
        html = file.read()

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) # elimina linkurile din site
    text = text.replace('\n', ' ').replace('\r', '') # elimina spatiile 
    return text.strip()
