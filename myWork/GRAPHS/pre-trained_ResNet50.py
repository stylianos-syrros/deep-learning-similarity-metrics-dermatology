import matplotlib.pyplot as plt

# Data for pre-trained correlation values
layers = [
    "conv5_block3_out (FCS)",
    "conv5_block3_out (ASCS)",
    "L2 Pooling (FCS)",
    "L2 Pooling HW (FCS)",
    "avgpool (FCS)",
    "Fully Connected (FCS)"
]
correlation_values = [
    0.8277,
    0.8333,
    0.8164,
    0.8116,
    0.8241,
    0.4438
]
# Δημιουργία Bar Chart
plt.figure(figsize=(10, 6))
plt.barh(layers, correlation_values, color='skyblue')
plt.xlabel('Correlation')
plt.title('Correlation Values for Pre-trained ResNet50 Layers')
plt.show()
