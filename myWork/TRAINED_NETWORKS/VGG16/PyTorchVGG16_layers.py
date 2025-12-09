import torch
import torchvision.models as models

# Φόρτωση του προεκπαιδευμένου μοντέλου VGG16
model = models.vgg16(pretrained=True)

# Εκτύπωση των ονομάτων των επιπέδων του VGG16
print("Features:")
for i, layer in enumerate(model.features):
    print(f"Layer {i + 1}: {layer}")

print("\nAvgpool:")
print(f"Layer: {model.avgpool}")

print("\nClassifier:")
for i, layer in enumerate(model.classifier):
    print(f"Layer {i + 1}: {layer}")
