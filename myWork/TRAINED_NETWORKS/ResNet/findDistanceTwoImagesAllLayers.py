import os
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from averageCosineSimilarity import calculate_cosine_similarity as avg_cosine_similarity
import numpy as np
import random

# Ορισμός seed για αναπαραγωγιμότητα
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)


# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.6459899, 0.52058024, 0.51453681], std=[0.14522511, 0.15518147, 0.16543107])
    transforms.Normalize(mean=[0.65886277, 0.45850426, 0.41027424], std=[0.14796351, 0.14156736, 0.13876683])
])

# Load the ResNet50 model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Adjust the final layer to match your output classes
model.to(device)
model.eval()

# Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0).to(device)  # Add batch dimension and move to device

# Function to extract features from a specific layer
def get_features_from_layer(model, image_tensor, layer_name):
    with torch.no_grad():
        x = model.conv1(image_tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        if layer_name == 'layer3':
            x = model.layer3(model.layer2(model.layer1(x)))
        elif layer_name == 'avgpool':
            x = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(x)))))
            x = torch.flatten(x, 1)
        elif layer_name == 'fc':
            x = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(x)))))
            x = torch.flatten(x, 1)
            x = model.fc(x)
        else:
            raise ValueError("Invalid layer name")
        
        return x

# Function to calculate and print the distances
def calculate_and_print_distances(img_path1, img_path2):
    img_tensor1 = preprocess_image(img_path1)
    img_tensor2 = preprocess_image(img_path2)

    layer_names = ['layer3', 'avgpool', 'fc']
    
    for layer in layer_names:
        features1 = get_features_from_layer(model, img_tensor1, layer)
        features2 = get_features_from_layer(model, img_tensor2, layer)

        # Print the feature dimensions
        print(f"Layer: {layer}, Feature dimensions: {features1.shape}")

        if layer == 'layer3':
            # Calculate and print cosine similarity using cosine_similarity
            cosine_dist = 1 - cosine_similarity(features1.cpu().numpy().reshape(1, -1), features2.cpu().numpy().reshape(1, -1))
            print(f"Layer: {layer}, Cosine Similarity Distance: {cosine_dist.flatten()}")

            # Calculate and print cosine similarity using avg_cosine_similarity
            avg_cosine_dist = 1 - avg_cosine_similarity(features1.cpu().numpy(), features2.cpu().numpy())
            print(f"Layer: {layer}, Average Cosine Similarity Distance: {avg_cosine_dist.flatten()}")
        else:
            # Calculate cosine similarity distance for avgpool and fc
            distance = 1 - cosine_similarity(features1.cpu().numpy(), features2.cpu().numpy())
            print(f"Layer: {layer}, Distance: {distance.flatten()}")

# Paths to the two images
image_path1 = r"C:\Users\steli\DIPLOMA\images\yeezyCARBON.png"
image_path2 = r"C:\Users\steli\DIPLOMA\images\yeezyGRANITE.png"

# Calculate and print distances
calculate_and_print_distances(image_path1, image_path2)
