import os
import glob
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

SCREENSHOT_DIR = "screenshots4"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
TIER1 = "clones/tier1"
TIER2 = "clones/tier2"
TIER3 = "clones/tier3"
TIER4 = "clones/tier4"

tier1_files = glob.glob(os.path.join(TIER1, "*.html"))
tier2_files = glob.glob(os.path.join(TIER2, "*.html"))
tier3_files = glob.glob(os.path.join(TIER3, "*.html"))
tier4_files = glob.glob(os.path.join(TIER4, "*.html"))

# Functie pentru a face un screenshot a unei pagini cu selenium
def take_screenshot(url, filename):
    options = Options()
    options.add_argument('--headless')

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        path = os.path.join(SCREENSHOT_DIR, filename)
        driver.set_window_size(1200, 900)

        driver.save_screenshot(path)
        print(f"screenshot taken: {path}")
        return path
    except Exception as e:
        print(f"error: {e}")
    finally:
        driver.quit()

def screenshots_for_tiers(directory):
    for file in directory:
        url = f"file:///{os.path.abspath(file)}" # path-ul ce trebuie deschis 
        domain = os.path.basename(file)
        filename = domain.replace(".", "_").replace("_html", ".png") # numele si extensia imaginii
        take_screenshot(url, filename)


