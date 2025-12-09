import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import logging

# Διαδρομή για τον φάκελο καταγραφής
log_dir = r'D:\Diploma\ResNet50_ISIC_LOG'

# Δημιουργία του φακέλου αν δεν υπάρχει
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Ρύθμιση του Logger
logging.basicConfig(filename=os.path.join(log_dir, 'ResNet50_training_from_start_log.log'), 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Έλεγχος για GPU
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

# Φόρτωση των δεδομένων από τους φακέλους
train_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\train"
test_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"

full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Δημιουργία των indices για κάθε κατηγορία στο training set
train_indices = {cls: [] for cls in range(3)}
for idx, (_, label) in enumerate(full_train_dataset):
    train_indices[label].append(idx)

# Επιλογή του 10% για validation και το υπόλοιπο 90% για training από κάθε κατηγορία
val_indices = []
train_indices_final = []
for cls in range(3):
    cls_train_indices = train_indices[cls]
    print(f'Total indices for class {cls}: {len(cls_train_indices)}')
    cls_train, cls_val = train_test_split(cls_train_indices, test_size=0.1, random_state=42)
    print(f'Class {cls}: {len(cls_train)} training indices, {len(cls_val)} validation indices')
    train_indices_final.extend(cls_train)
    val_indices.extend(cls_val)

train_dataset = Subset(full_train_dataset, train_indices_final)
val_dataset = Subset(full_train_dataset, val_indices)

# Δημιουργία DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Εκτύπωση του πλήθους εικόνων στα datasets
print(f'Training dataset size: {len(train_dataset)} images')
print(f'Validation dataset size: {len(val_dataset)} images')
print(f'Test dataset size: {len(test_dataset)} images')

# Εκτύπωση του πλήθους των κατηγοριών
print(f'Number of classes: {len(full_train_dataset.classes)}')

# Φάκελος αποθήκευσης μοντέλων
model_dir = r"D:\Diploma\ResNet50_models_ISIC"
os.makedirs(model_dir, exist_ok=True)

# Φόρτωση του ResNet50 μοντέλου
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # Αλλαγή του τελευταίου fully connected layer για 3 κατηγορίες
model.to(device)

# Ορισμός του loss function και του optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Εκπαίδευση του μοντέλου
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()  # Έναρξη του timer για κάθε epoch
        print(f"Epoch {epoch}/{num_epochs}")
        logging.info(f"Epoch {epoch}/{num_epochs}")

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        epoch_duration = time.time() - epoch_start_time  # Διάρκεια του epoch
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Epoch duration: {epoch_duration:.2f} seconds')  

        logging.info(f'Epoch {epoch}/{num_epochs} completed')
        logging.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        logging.info(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        logging.info(f'Epoch duration: {epoch_duration:.2f} seconds')      
        
        # Save model for each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_val_loss': val_loss,  # Τρέχουσα απόδοση του μοντέλου
            'epoch_val_acc': val_acc,    # Τρέχουσα ακρίβεια του μοντέλο
            'best_val_loss': best_val_loss,
        }, os.path.join(model_dir, f'resnet50_model_epoch_{epoch}.pth'))
        logging.info(f'Model for epoch {epoch} saved')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(model_dir, 'best_resnet50_model.pth'))
            print('Best model saved')
            logging.info('Best model saved')


    total_duration = time.time() - start_time  # Συνολική διάρκεια
    print(f'Best val Loss: {best_val_loss:.4f}')
    print(f'Total training duration: {total_duration:.2f} seconds')
    logging.info(f'Best val Loss: {best_val_loss:.4f}')
    logging.info(f'Total training duration: {total_duration:.2f} seconds')

# Κλήση της συνάρτησης εκπαίδευσης
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

# Τελική αξιολόγηση στο test set
checkpoint = torch.load(os.path.join(model_dir, 'best_resnet50_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])

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
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
                                                               
# Κλήση της συνάρτησης αξιολόγησης
evaluate_model(model, test_loader, criterion)
