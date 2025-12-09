import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm
import logging

# Ορισμός του Logger
logging.basicConfig(filename=r'D:\Diploma\ISIC_log\evaluate_log.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Έλεγχος για GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU with CUDA")
    logging.info("Using GPU with CUDA")
else:
    print("Using CPU")
    logging.info("Using CPU")

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6459899, 0.52058024, 0.51453681], std=[0.14522511, 0.15518147, 0.16543107])
])

# Φόρτωση των δεδομένων από τους φακέλους
test_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Φόρτωση του ViT μοντέλου
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)
model.to(device)

# Ορισμός του loss function
criterion = nn.CrossEntropyLoss()

# Αξιολόγηση του μοντέλου στο test set
def evaluate_model(model, test_loader, criterion, model_path):
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        logging.error(f"No model found at {model_path}")
        return

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

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
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    # Καθορισμός του path του αρχείου .pth
    model_path = r"D:\Diploma\ViT_models_ISIC\vit_model_epoch_25.pth"
    evaluate_model(model, test_loader, criterion, model_path)
