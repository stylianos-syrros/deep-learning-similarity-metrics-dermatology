import matplotlib.pyplot as plt

# Data for LPIPS variants and DISTS
models = ['DISTS', 'First stage F\DISTS ASCS', 'DISTS Weighted ASCS']
correlations = [0.715707105, 0.751712162, 0.771565475, 0.850210983, 0.763656203, 0.788318933]

# Creating the bar chart
plt.figure(figsize=(10, 6))
plt.barh(models, correlations, color='skyblue')
plt.xlabel('Correlation')
plt.title('Correlation Values for LPIPS and DISTS Variants')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()
