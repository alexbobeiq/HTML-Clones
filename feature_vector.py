import os
import pandas as pd
import glob
import torch
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

tier1_files = glob.glob(os.path.join(TIER1, "*.html"))
tier2_files = glob.glob(os.path.join(TIER2, "*.html"))
tier3_files = glob.glob(os.path.join(TIER3, "*.html"))
tier4_files = glob.glob(os.path.join(TIER4, "*.html"))

text = {file: extract_text(file) for file in tier2_files}

vectorizer = TfidfVectorizer(max_features=500)
text_features = vectorizer.fit_transform(text.values())

text_features = text_features.toarray()
text_feature_dict = {file: text_features[i] for i, file in enumerate(text.keys())}


model = models.resnet50(pretrained = True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = transforms.Compose([
    transforms.Resize((1200,900)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_image_features(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img)
    return features.squeeze().numpy()

ss_files = glob.glob("screenshots/*.png")
image_features = {file:extract_image_features(file) for file in ss_files}


combined_features = {}

for file in text.keys():  
    text_vec = text_feature_dict.get(file, np.zeros(384))  
    img_file = f"screenshots/{file.replace('.html', '.png')}"  
    img_vec = image_features.get(img_file, np.zeros(2048)) 
    
    combined_features[file] = np.concatenate([img_vec]) 

df = pd.DataFrame.from_dict(combined_features, orient="index")

df.to_csv("combined_features.csv", index=True, header=False)

print("✅ Features saved to combined_features.csv")

