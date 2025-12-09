import matplotlib.pyplot as plt

# Δεδομένα για Epochs - conv5_block3_3_conv
epochs = [f'Epoch {i}' for i in range(1, 26)]
correlations_conv = [
    0.3026, 0.4680, 0.4659, 0.4871, 0.3934, 0.4312, 0.5528, 0.6625, 0.6687, 0.6172, 
    0.4963, 0.5505, 0.6662, 0.6248, 0.6625, 0.6900, 0.6715, 0.6200, 0.6294, 0.5874,
    0.5627, 0.5934, 0.6407, 0.6156, 0.5555
]

# Δημιουργία Bar Chart για το conv5_block3_3_conv
plt.figure(figsize=(12, 6))
plt.bar(epochs, correlations_conv, color='skyblue')
plt.xlabel('Epoch')
plt.ylabel('Correlation')
plt.title('Correlation Values for conv5_block3_3_conv Layer Across Epochs')
plt.xticks(rotation=45)
plt.show()


# Δεδομένα για Epochs - conv5_block3_3_bn
correlations_bn = [
    0.5037, 0.4902, 0.5214, 0.5333, 0.5356, 0.5586, 0.6892, 0.6547, 0.6606, 0.5854,
    0.4917, 0.5020, 0.6610, 0.6337, 0.6699, 0.6939, 0.6951, 0.6573, 0.6376, 0.5963,
    0.5679, 0.5947, 0.6334, 0.6311, 0.5906
]

# Δημιουργία Bar Chart για το conv5_block3_3_bn
plt.figure(figsize=(12, 6))
plt.bar(epochs, correlations_bn, color='skyblue')
plt.xlabel('Epoch')
plt.ylabel('Correlation')
plt.title('Correlation Values for conv5_block3_3_bn Layer Across Epochs')
plt.xticks(rotation=45)
plt.show()
