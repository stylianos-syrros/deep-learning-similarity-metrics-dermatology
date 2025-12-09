import torch
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image

# Καθορισμός του μονοπατιού της εικόνας στον τοπικό υπολογιστή
image_path = r"C:\Users\steli\DIPLOMA\bcc\CASE001\0.jpg"
image = Image.open(image_path)

# Φορτώνουμε τον feature extractor και το μοντέλο
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Προεπεξεργασία της εικόνας
inputs = feature_extractor(images=image, return_tensors="pt")

# Προώθηση της εικόνας μέσω του μοντέλου
outputs = model(**inputs)

# Η έξοδος του μοντέλου περιέχει το τελευταίο κρυφό κατάσταση (last_hidden_state)
# Το CLS token είναι το πρώτο token στην ακολουθία
cls_token = outputs.last_hidden_state[:, 0, :]
# Τα spatial tokens είναι τα υπόλοιπα tokens στην ακολουθία
spatial_tokens = outputs.last_hidden_state[:, 1:, :]

# Τυπώνουμε το CLS token και το σχήμα των spatial tokens
print("CLS Token:", cls_token)
print("Spatial Tokens:", spatial_tokens)
