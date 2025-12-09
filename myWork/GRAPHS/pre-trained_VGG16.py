import matplotlib.pyplot as plt

# Data for pre-trained correlation values
layers = [
    "block5_conv3 (ASCS)",
    "block5_relu3 (ASCS)",
    "avgpool (ASCS)",
    "block5_conv3 (FCS)",
    "block5_relu3 (FCS)",
    "avgpool (FCS)",
    "ReLU (FCS)",
    "fc3 (FCS)"
]
correlation_values = [
    0.668574388,
    0.533618863,
    0.389153714,
    0.726930823,
    0.740873932,
    0.751843165,
    0.775115432,
    0.584569444
]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.barh(layers, correlation_values, color='skyblue')
plt.xlabel('Correlation')
plt.title('Correlation Values for Pre-trained VGG16 Layers')
plt.gca().invert_yaxis()  # Invert y-axis for better readability

# Display the chart
plt.tight_layout()
plt.show()
