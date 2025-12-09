import os
import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Ορισμός της συσκευής
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Μετασχηματισμός των εικόνων με τα ίδια mean και std που χρησιμοποιήθηκαν κατά την εκπαίδευση
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6459899, 0.52058024, 0.51453681], std=[0.14522511, 0.15518147, 0.16543107])
])

def load_and_preprocess_image(img_path):
    """Φορτώνει και προεπεξεργάζεται μια εικόνα."""
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0).to(device)

def extract_features(img_tensor, model, layer_name):
    """Εξάγει τα χαρακτηριστικά από συγκεκριμένο επίπεδο ενός μοντέλου."""
    with torch.no_grad():
        x = model.conv1(img_tensor)
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
            raise ValueError(f"Invalid layer name: {layer_name}")

    return x.cpu().numpy()

def calculate_distance(features1, features2):
    """Υπολογίζει την απόσταση μεταξύ δύο συνόλων χαρακτηριστικών χρησιμοποιώντας cosine similarity."""
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
    distance = 1 - similarity[0][0]
    return distance

def calculate_distances_for_cases(base_dir, model_dir, output_file):
    """Υπολογίζει τις αποστάσεις για όλα τα cases και τις αποθηκεύει σε ένα αρχείο Excel."""
    results = {}
    # Φιλτράρισμα μόνο για τα μοντέλα των πρώτων 3 εποχών
    model_paths = [os.path.join(model_dir, f'resnet50_model_epoch_{i}.pth') for i in range(1, 26)]

    for epoch, model_path in enumerate(model_paths, start=1):
        print(f"Loading model from {model_path}")
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to(device)
        model.eval()

        for case in sorted(os.listdir(base_dir)):
            case_dir = os.path.join(base_dir, case)
            if os.path.isdir(case_dir):
                print(f"Processing case: {case}, Epoch: {epoch}")
                images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]

                img_tensor1 = load_and_preprocess_image(images[0])
                img_tensor2 = load_and_preprocess_image(images[1])
                img_tensor3 = load_and_preprocess_image(images[2])

                features1_layer3 = extract_features(img_tensor1, model, 'layer3')
                features2_layer3 = extract_features(img_tensor2, model, 'layer3')
                features3_layer3 = extract_features(img_tensor3, model, 'layer3')

                dist_layer3_cosine_1 = calculate_distance(features1_layer3, features2_layer3)
                dist_layer3_cosine_2 = calculate_distance(features1_layer3, features3_layer3)
                avg_dist_layer3_cosine = (dist_layer3_cosine_1 + dist_layer3_cosine_2) / 2

                features1_avgpool = extract_features(img_tensor1, model, 'avgpool')
                features2_avgpool = extract_features(img_tensor2, model, 'avgpool')
                features3_avgpool = extract_features(img_tensor3, model, 'avgpool')

                dist_avgpool_1 = calculate_distance(features1_avgpool, features2_avgpool)
                dist_avgpool_2 = calculate_distance(features1_avgpool, features3_avgpool)
                avg_dist_avgpool = (dist_avgpool_1 + dist_avgpool_2) / 2

                features1_fc = extract_features(img_tensor1, model, 'fc')
                features2_fc = extract_features(img_tensor2, model, 'fc')
                features3_fc = extract_features(img_tensor3, model, 'fc')

                dist_fc_1 = calculate_distance(features1_fc, features2_fc)
                dist_fc_2 = calculate_distance(features1_fc, features3_fc)
                avg_dist_fc = (dist_fc_1 + dist_fc_2) / 2

                if case not in results:
                    results[case] = {}
                
                results[case][f'Epoch_{epoch}_layer3_cosine'] = avg_dist_layer3_cosine
                results[case][f'Epoch_{epoch}_avgpool'] = avg_dist_avgpool
                results[case][f'Epoch_{epoch}_fc'] = avg_dist_fc

        print(f'Distance 1 : {avg_dist_layer3_cosine} , Distance 2 : {avg_dist_avgpool} , Distance 3 : {avg_dist_fc}')

    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_excel(output_file, index=True, index_label="Case")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    base_dir = r"C:\Users\steli\DIPLOMA\bcc"
    model_dir = r"D:\Diploma\ResNet50_models_ISIC"
    output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_ResNet_Distances_All_Epochs.xlsx"
    calculate_distances_for_cases(base_dir, model_dir, output_file)
