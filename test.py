import matplotlib.pyplot as plt
import numpy as np

# Data for comparison
metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'PR AUC', 'F1 Score']
no_cv_values = [0.516, 0.757, 0.745, 0.721, 0.836, 0.816, 0.732]  # No cross-validation
cv_values = [0.418, 0.807, 0.799, 0.778, 0.892, 0.880, 0.788]  # Cross-validation

# Positions for the bars
x = np.arange(len(metrics))

# Width of the bars
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, no_cv_values, width, label='No Cross Validation', color='tab:blue')
bars2 = ax.bar(x + width/2, cv_values, width, label='Cross Validation', color='tab:orange')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance: Cross Validation vs No Cross Validation')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

