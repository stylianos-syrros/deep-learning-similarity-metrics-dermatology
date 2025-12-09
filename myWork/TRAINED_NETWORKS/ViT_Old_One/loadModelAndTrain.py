import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm

# Παράκαμψη της προειδοποίησης για symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Καταστολή συγκεκριμένης προειδοποίησης
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

# Ελέγχει αν η GPU είναι διαθέσιμη και θέτει την συσκευή
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT απαιτεί σταθερό μέγεθος εισόδου
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7438, 0.5865, 0.5869], std=[0.0804, 0.1076, 0.1202])
])

# Φόρτωση των δεδομένων
train_dir = r"C:\Users\steli\DIPLOMA\myProgramms\TRAINED_NETWORKS\Skin_Cancer_Dataset\SkinCancerISIC\Train"
test_dir = r"C:\Users\steli\DIPLOMA\myProgramms\TRAINED_NETWORKS\Skin_Cancer_Dataset\SkinCancerISIC\Test"

# Φόρτωση του train dataset και διαχωρισμός του σε train και validation sets
full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Φόρτωση του test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

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

# Ορισμός του καταλόγου αποθήκευσης
model_dir = r"D:\Diploma\models"
os.makedirs(model_dir, exist_ok=True)

# Φόρτωση του καλύτερου μοντέλου
best_model_path = os.path.join(model_dir, "best_model.pth")
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(full_train_dataset.classes))
model.load_state_dict(torch.load(best_model_path))
model.to(device)

# Ορισμός συνάρτησης απώλειας και βελτιστοποιητή
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Συνάρτηση εκπαίδευσης
def train_model(model, train_loader, val_loader, criterion, optimizer, start_epoch=13, num_epochs=25, best_acc=0.0):
    start_time = time.time()  # Έναρξη του συνολικού timer

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()  # Έναρξη του timer για κάθε epoch
        print(f"Epoch {epoch}/{num_epochs}")

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

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        epoch_duration = time.time() - epoch_start_time  # Διάρκεια του epoch
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Epoch duration: {epoch_duration:.2f} seconds')

        # Αποθήκευση του μοντέλου κάθε εποχής
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch}.pth'))

        # Αποθήκευση του καλύτερου μοντέλου
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))

    total_duration = time.time() - start_time  # Συνολική διάρκεια
    print(f'Best val Acc: {best_acc:.4f}')
    print(f'Total training duration: {total_duration:.2f} seconds')

# Εκτέλεση της εκπαίδευσης
train_model(model, train_loader, val_loader, criterion, optimizer, start_epoch=13, num_epochs=25, best_acc=0.05)#################

# Τελική αξιολόγηση στο test set
model.load_state_dict(torch.load(best_model_path))

def evaluate_model(model, test_loader):
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    return test_acc.item()

# Αξιολόγηση με το test set
test_accuracy = evaluate_model(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.4f}')
