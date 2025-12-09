import os
import torch
import numpy as np
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import logging

# Ορισμός του Logger
logging.basicConfig(filename=r'D:\Diploma\VGG16_ISIC_LOG\VGG16_evaluation_log.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Ελέγχει αν η GPU είναι διαθέσιμη και θέτει την συσκευή
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU with CUDA")
else:
    print("Using CPU")

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6459899, 0.52058024, 0.51453681], std=[0.14522511, 0.15518147, 0.16543107])
])

# Φόρτωση των δεδομένων από τον φάκελο
test_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Φόρτωση του VGG16 μοντέλου
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)  # Αλλαγή του τελευταίου fully connected layer για 3 κατηγορίες
model.to(device)

# Ορισμός του loss function
criterion = torch.nn.CrossEntropyLoss()

# Φάκελος αποθήκευσης μοντέλων
model_dir = r"D:\Diploma\VGG16_models_ISIC"

# Αξιολόγηση του μοντέλου στο test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = corrects.double() / len(test_loader.dataset)
    return test_loss, test_acc.item()

# Αξιολόγηση όλων των μοντέλων
for epoch in range(1, 26):
    model_path = os.path.join(model_dir, f'vgg16_model_epoch_{epoch}.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f'Epoch {epoch}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}')
    logging.info(f'Epoch {epoch}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}')

# Αξιολόγηση του best_vgg16_model.pth
best_model_path = os.path.join(model_dir, 'best_vgg16_model.pth')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = evaluate_model(model, test_loader, criterion)
print(f'Best Model: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}')
logging.info(f'Best Model: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}')

# Αξιολόγηση του last_checkpoint.pth
last_checkpoint_path = os.path.join(model_dir, 'last_checkpoint.pth')
checkpoint = torch.load(last_checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = evaluate_model(model, test_loader, criterion)
print(f'Last Checkpoint: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}')
logging.info(f'Last Checkpoint: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}')
