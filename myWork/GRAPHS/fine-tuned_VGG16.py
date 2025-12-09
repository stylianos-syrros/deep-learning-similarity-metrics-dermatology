import matplotlib.pyplot as plt

# Data for Epoch avgpool correlation values
epochs = [f"Epoch {i}" for i in range(1, 26)]
avgpool_correlation = [
    0.760547124, 0.611774308, 0.632766194, 0.611331458, 0.702900066,
    0.673481138, 0.348654724, 0.577243414, 0.517150582, 0.528706236,
    0.405222577, 0.511229046, 0.541428924, 0.529880218, 0.479601048,
    0.427797325, 0.504629156, 0.367946037, 0.379450759, 0.543632439,
    0.377362937, 0.485836374, 0.518353816, 0.491187674, 0.477563297
]

# Create bar chart for avgpool correlations across epochs
plt.figure(figsize=(12, 6))
plt.bar(epochs, avgpool_correlation, color='skyblue')
plt.xlabel('Epoch')
plt.ylabel('Correlation')
plt.title('Correlation Values for avgpool Layer Across Epochs')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Data for Epoch ReLU (classifier.4) correlation values
relu_correlation = [
    0.762251469, 0.592763523, 0.576477949, 0.55388117, 0.536621278,
    0.63292197, 0.318743091, 0.597456525, 0.270827168, 0.278531254,
    0.222015237, 0.269554497, 0.419511414, 0.519997137, 0.413034434,
    0.217827177, 0.350273853, 0.139436894, 0.257196017, 0.480073757,
    0.114058603, 0.14336012, 0.301717668, 0.193560047, 0.33082905
]

# Create bar chart for ReLU (classifier.4) correlations across epochs
plt.figure(figsize=(12, 6))
plt.bar(epochs, relu_correlation, color='skyblue')
plt.xlabel('Epoch')
plt.ylabel('Correlation')
plt.title('Correlation Values for ReLU Layer Across Epochs')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
