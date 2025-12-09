import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from averageCosineSimilarity import calculate_cosine_similarity as avg_cosine_similarity
from flattenCosineSimilarity import calculate_cosine_similarity as flat_cosine_similarity

# Παράκαμψη της προειδοποίησης για symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Έλεγχος για GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", "GPU" if torch.cuda.is_available() else "CPU")

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Φόρτωση του VGG16 μοντέλου
weights = models.VGG16_Weights.DEFAULT
model = models.vgg16(weights=weights)
model.to(device)
model.eval()

# Φόρτωση και μετασχηματισμός της εικόνας
def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Προσθέτουμε την batch dimension και μεταφέρουμε στη συσκευή

# Συνάρτηση για την εξαγωγή χαρακτηριστικών από ένα συγκεκριμένο επίπεδο
def get_features_from_layer(model, image_tensor, layer_type, layer_index=None):
    with torch.no_grad():
        if layer_type == 'features':
            if layer_index is None or layer_index >= len(model.features):
                raise ValueError(f"Invalid layer index {layer_index} for layer type {layer_type}. "
                                 f"Valid indices are 0 to {len(model.features) - 1}.")
            truncated_model = torch.nn.Sequential(*list(model.features.children())[:layer_index+1])
            features = truncated_model(image_tensor)
        elif layer_type == 'avgpool':
            if layer_index is not None:
                raise ValueError(f"Layer index must be None for layer type {layer_type}.")
            features = model.features(image_tensor)
            features = model.avgpool(features)
        elif layer_type == 'classifier':
            if layer_index is None or layer_index >= len(model.classifier):
                raise ValueError(f"Invalid layer index {layer_index} for layer type {layer_type}. "
                                 f"Valid indices are 0 to {len(model.classifier) - 1}.")
            features = model.features(image_tensor)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
            truncated_model = torch.nn.Sequential(*list(model.classifier.children())[:layer_index+1])
            features = truncated_model(features)
        else:
            raise ValueError(f"Invalid layer type {layer_type}.")
    return features.squeeze().cpu().numpy()

# Λήψη των ονομάτων των επιπέδων
feature_layer_names = list(model.features._modules.values())
classifier_layer_names = list(model.classifier._modules.values())

# Συνδυασμός των ονομάτων των επιπέδων
layer_names = feature_layer_names + ['avgpool'] + classifier_layer_names

# Ορισμός διαδρομών των εικόνων
image_path1 = r"C:\Users\steli\DIPLOMA\bcc\CASE001\0.jpg"
image_path2 = r"C:\Users\steli\DIPLOMA\bcc\CASE001\1.jpg"

# Φόρτωση και προεπεξεργασία των εικόνων
image_tensor1 = load_and_transform_image(image_path1)
image_tensor2 = load_and_transform_image(image_path2)

# Επιλογή του επιπέδου από το οποίο θα εξάγουμε τα χαρακτηριστικά
layer_type = 'features'  # 'features', 'avgpool', or 'classifier'
layer_index = 28  # Για τα classifier layers, χρησιμοποίησε το index στο classifier array

# Εξαγωγή των χαρακτηριστικών από το επιλεγμένο επίπεδο
features_1 = get_features_from_layer(model, image_tensor1, layer_type, layer_index)
features_2 = get_features_from_layer(model, image_tensor2, layer_type, layer_index)

# Εύρεση του ονόματος του επιπέδου
layer_name = ''
if layer_type == 'features':
    layer_name = str(model.features[layer_index])
elif layer_type == 'classifier':
    layer_name = str(model.classifier[layer_index])
elif layer_type == 'avgpool':
    layer_name = 'avgpool'

# Εκτύπωση του ονόματος του επιπέδου
print(f"Layer name: {layer_name}")

# Υπολογισμός της cosine similarity
avg_cosine_sim = avg_cosine_similarity(features_1, features_2)
flat_cosine_sim = flat_cosine_similarity(features_1.reshape(1, -1), features_2.reshape(1, -1))

# Υπολογισμός της απόστασης ως 1 - cosine similarity
avg_distance = 1- avg_cosine_sim 
flat_distance = 1 - flat_cosine_sim
print(f"Η απόσταση μεταξύ των δύο εικόνων με avg είναι: {avg_distance}")
print(f"Η απόσταση μεταξύ των δύο εικόνων με flatten είναι: {flat_distance}")
