import clip
import os
import torch
import json
from tqdm import tqdm
from PIL import Image

default_labels = [
  'Wood Texture',
  'Metal',
  'Frosted Glass',
  '2.5D Isometric Modern Architecture',
  '2.5D Isometric Ancient-Style Architecture',
  '3D Architectural Scenes',
  '3D Ancient-Style Architecture',
  '3D Paper Cuttings Style'
  'Marble',
  'Ceramic Tiles',
  'Ground',
  'Long Pile Plush Material',
  'Ice and Snow Material',
  'Inflatable Material',
  'Acid Glass Material',
  'Illustration, Thick Painted',
  'Illustration, Line Drawing Style',
  'Illustration, Flat Style',
  "Illustration, 2.5D",
  'Illustration, 3D',
  'Illustration, Paper Cutting Style',
  'Illustration, Diffuse Light',
  'Illustration, National Trend (Chinese style)',
  'Illustration, Texture',
  'Illustration, Graffiti',
  'Portrait',
  'CG Characters',
  'Cartoon 3D Characters',
  'Realistic 3D Characters',
  'Blindbox',
  'Thick Painted Characters',
  'National Trend 3D Characters',
  'National Trend Illustration Characters',
  'Digital Humans'
]

class Classify():
    def __init__(self,labels=default_labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model2, preprocess = clip.load("ViT-L/14", device=self.device)  #RN50x64   
        self.clip = model2
        self.preprocess = preprocess
        self.loadLabelPool(labels)
        
    def normalized(self, a, axis=-1, order=2):
        import numpy as np  # pylint: disable=import-outside-toplevel
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)
    
    def loadLabelPool(self,labels):
        self.labels = labels
        label_tokens = clip.tokenize(labels).to(self.device)
        with torch.no_grad():
            self.label_features = self.clip.encode_text(label_tokens)
        
    def matchLabel(self,pil_image):
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            similarity_scores = (image_features @ self.label_features.t()).squeeze(0)
            # best_label_idx = similarity_scores.argmax().item()
            # matched_label = self.labels[best_label_idx] 
            if torch.max(similarity_scores).item()<5.53:
                matched_labels = []
            else:
                top_label_indices = similarity_scores.argsort(descending=True)[:5]
                matched_labels = [self.labels[idx] for idx in top_label_indices]
        return matched_labels
        
    def predict(self,src_path,rst_path):
        with open(src_path, 'r+') as f:
            data_map = json.load(f)
        for item in tqdm(data_map):
            img_path = item['local_path']
            pil_image = Image.open(img_path)
            matched_labels = self.matchLabel(pil_image)
            pil_image.close()
            item['class'] = matched_labels
        with open(rst_path, 'w+') as f:
            data_map = json.dump(data_map,f,indent=2)
            