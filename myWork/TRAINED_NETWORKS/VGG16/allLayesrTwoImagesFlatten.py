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
            truncated_model = torch.nn.Sequential(*list(model.features.children())[:layer_index+1])
            features = truncated_model(image_tensor)
        elif layer_type == 'avgpool':
            features = model.features(image_tensor)
            features = model.avgpool(features)
        elif layer_type == 'classifier':
            features = model.features(image_tensor)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
            truncated_model = torch.nn.Sequential(*list(model.classifier.children())[:layer_index+1])
            features = truncated_model(features)
    return features.cpu().numpy().flatten()

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

print("Features")
# Διατρέχει όλα τα επίπεδα των χαρακτηριστικών
for layer_index in range(len(model.features)):
    layer_type = 'features'
    features_1 = get_features_from_layer(model, image_tensor1, layer_type, layer_index)
    features_2 = get_features_from_layer(model, image_tensor2, layer_type, layer_index)
    similarity = flat_cosine_similarity(features_1, features_2)
    distance = 1 - similarity
    layer_name = str(model.features[layer_index])
    print(f"Layer name: {layer_name}, Distance: {distance}")

# Υπολογισμός για το avgpool
print("Avgpool")
layer_type = 'avgpool'
features_1 = get_features_from_layer(model, image_tensor1, layer_type)
features_2 = get_features_from_layer(model, image_tensor2, layer_type) 
similarity = flat_cosine_similarity(features_1, features_2)
distance = 1 - similarity
print(f"Layer name: avgpool, Distance: {distance}")

print("Classifier")
# Διατρέχει όλα τα επίπεδα του classifier
for layer_index in range(len(model.classifier)):
    layer_type = 'classifier'
    features_1 = get_features_from_layer(model, image_tensor1, layer_type, layer_index)
    features_2 = get_features_from_layer(model, image_tensor2, layer_type, layer_index)
    similarity = flat_cosine_similarity(features_1, features_2)
    distance = 1 - similarity
    layer_name = str(model.classifier[layer_index])
    print(f"Layer name: {layer_name}, Distance: {distance}")

