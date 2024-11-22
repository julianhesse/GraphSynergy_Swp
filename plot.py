import matplotlib.pyplot as plt
import numpy as np

# Metrics for cross-validation (6 folds, including the last one)
cross_validation_metrics = {
    'loss': [0.40, 0.41, 0.40, 0.40, 0.40, 0.40],
    'accuracy': [0.82, 0.82, 0.82, 0.82, 0.82, 0.82],
    'precision': [0.81, 0.81, 0.81, 0.81, 0.82, 0.82],
    'recall': [0.79, 0.79, 0.79, 0.79, 0.79, 0.79],
    'roc_auc': [0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
    'pr_auc': [0.89, 0.89, 0.89, 0.89, 0.89, 0.89],
    'f1_score': [0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
}

# No-cross-validation metrics
no_cross_validation_metrics = {
    'loss': 0.52,
    'accuracy': 0.76,
    'precision': 0.75,
    'recall': 0.73,
    'roc_auc': 0.84,
    'pr_auc': 0.82,
    'f1_score': 0.74,
}

# Averaging the cross-validation metrics
averaged_metrics = {metric: np.mean(values) for metric, values in cross_validation_metrics.items()}

# Prepare the plot
metrics = list(no_cross_validation_metrics.keys())
cross_validation_values = [averaged_metrics[metric] for metric in metrics]
no_cross_validation_values = [no_cross_validation_metrics[metric] for metric in metrics]

x = np.arange(len(metrics))  # Number of metrics
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width / 2, cross_validation_values, width, label='Cross-Validation (6 Folds)', color='blue', alpha=0.7)
bars2 = ax.bar(x + width / 2, no_cross_validation_values, width, label='No Cross-Validation', color='red', alpha=0.7)

# Add text and labels
ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Values', fontsize=12)
ax.set_title('Performance Comparison: Cross-Validation vs No Cross-Validation', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, fontsize=10)
ax.legend(fontsize=10)

# Annotate values on the bars
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
