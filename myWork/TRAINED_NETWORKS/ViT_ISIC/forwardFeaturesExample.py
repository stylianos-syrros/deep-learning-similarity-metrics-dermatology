import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

# Έλεγχος για GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", "GPU" if torch.cuda.is_available() else "CPU")

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Φόρτωση του ViT μοντέλου
model_name = 'vit_base_patch16_224'
model = timm.create_model(model_name, pretrained=True)
model.eval()
model.to(device)

# Προσθήκη της συνάρτησης forward_features στη κλάση του μοντέλου
def forward_features(self, x):
    # Διαχωρισμός της εικόνας σε patches και μετατροπή σε διανύσματα
    x = self.patch_embed(x)  # patch_embed: μεθόδος που μετατρέπει τα patches σε διανύσματα

    # Προσθήκη του [CLS] token στην αρχή της ακολουθίας
    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # Αντιγραφή του [CLS] token για κάθε δείγμα στο batch
    x = torch.cat((cls_tokens, x), dim=1)

    # Προσθήκη των positional embeddings
    x = x + self.pos_embed
    x = self.pos_drop(x)

    # Διέλευση μέσω των στρωμάτων του μετασχηματιστή
    for blk in self.blocks:
        x = blk(x)
    
    # Επιστροφή των χαρακτηριστικών
    return x

# Προσθήκη της συνάρτησης στο μοντέλο
model.forward_features = forward_features.__get__(model, model.__class__)

# Φόρτωση και μετασχηματισμός της εικόνας
def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Προσθέτουμε την batch dimension και μεταφέρουμε στη συσκευή

# Εξαγωγή χαρακτηριστικών από το μοντέλο
def get_features(model, image_tensor):
    with torch.no_grad():
        features = model.forward_features(image_tensor)
    return features.cpu().numpy()

# Ορισμός διαδρομής της εικόνας
#image_path = r"C:\Users\steli\DIPLOMA\bcc\CASE001\1.jpg"
image_path = r"C:\Users\steli\DIPLOMA\images\yeezyCARBON.png"
# Φόρτωση και προεπεξεργασία της εικόνας
image_tensor = load_and_transform_image(image_path)

# Εξαγωγή των χαρακτηριστικών
features = get_features(model, image_tensor)

print("Shape of extracted features:", features.shape)
print("Extracted features:", features)
