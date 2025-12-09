import matplotlib.pyplot as plt

# Data for the bar chart
models = ['Original DISTS', 'Fine-Tuned (1st stage) DISTS (Epoch 10)', 'Fine-Tuned (2nd satge) DISTS (Epoch 7)']
correlations = [0.850210983, 0.856371422, 0.928304919]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(models, correlations, color=['skyblue', 'lightgreen', 'salmon'])

# Adding the correlation values above the bars
for bar, corr in zip(bars, correlations):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,
             f'{corr:.3f}', ha='center', va='bottom', fontsize=12, color='black')

# Chart title and labels
plt.title('Correlation Values for DISTS and Fine-Tuned Versions', fontsize=16)
plt.ylabel('Correlation', fontsize=14)
plt.ylim(0.0, 1.0)  # Adjusting y-axis to highlight differences

# Display the chart
plt.tight_layout()
plt.show()
