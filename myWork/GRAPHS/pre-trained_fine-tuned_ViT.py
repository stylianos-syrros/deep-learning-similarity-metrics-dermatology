# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Data for ViT Pre-trained Network
layers = ['Embedding Token', 'CLS Token']
correlation_values = [0.3151, 0.3637]

# Plotting the Pre-trained Network Correlation Values
plt.figure(figsize=(10, 6))
plt.barh(layers, correlation_values, color='#87CEEB', edgecolor='black', linewidth=0.4)
plt.xlabel('Correlation')
plt.title('Correlation Values for Pre-trained ViT Layers')
plt.xlim([0, 0.8])
plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.tight_layout()
plt.show()

# Data for ViT Fine-Tuned Network using CLS Token
epochs = [f'Epoch_{i}' for i in range(1, 26)]
correlation_values_epochs = [
    0.4741, 0.6349, 0.7126, 0.6619, 0.5836, 0.4950, 0.3859, 0.3921, 0.4247, 0.4199,
    0.4355, 0.3873, 0.5514, 0.5390, 0.4822, 0.5208, 0.5246, 0.4258, 0.5301, 0.4490,
    0.5088, 0.3696, 0.2440, 0.5093, 0.4385
]

# Plotting the Fine-Tuned Network Correlation Values Across Epochs
plt.figure(figsize=(14, 6))
plt.bar(epochs, correlation_values_epochs, color='#87CEEB')
plt.ylabel('Correlation')
plt.title('Correlation Values for CLS Token Across Epochs')
plt.xticks(rotation=90)
plt.ylim([0, 0.8])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
