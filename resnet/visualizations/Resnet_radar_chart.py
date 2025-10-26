import matplotlib.pyplot as plt
import numpy as np

# Data for all models
models = ['Baseline ResNet50', 'GeM Pooling', 'Concat Pooling', 'ECA Attention']

# Metrics: Accuracy, Precision, Recall, F1-Score (all as percentages)
baseline_metrics = [88.98, 84.48, 77.15, 80.03]
gem_metrics = [87.50, 82.00, 75.00, 75.51]  # Using peak values
concat_metrics = [86.80, 80.50, 74.00, 74.69]  # Using peak values
eca_metrics = [85.20, 78.90, 72.00, 72.82]  # Using peak values

# Metric labels
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
num_vars = len(categories)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Create figure
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'), dpi=100)

# Plot data for each model
colors = ['#9966FF', '#FF6384', '#36A2EB', '#4BC0C0']
line_styles = ['-', '-', '-', '-']

for idx, (model, metrics, color, style) in enumerate(zip(
    models, 
    [baseline_metrics, gem_metrics, concat_metrics, eca_metrics],
    colors,
    line_styles
)):
    values = metrics + metrics[:1]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2.5, label=model, 
            color=color, linestyle=style, markersize=8, 
            markeredgecolor='white', markeredgewidth=2, alpha=0.8)
    ax.fill(angles, values, alpha=0.15, color=color)

# Customize the plot
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(20, 101, 20))
ax.set_yticklabels([f'{i}%' for i in range(20, 101, 20)], fontsize=10)

# Add grid
ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)

# Add title
plt.title('Test Metrics Overview - Model Comparison\nRadar Chart', 
          fontsize=16, fontweight='bold', pad=30, y=1.08)

# Customize legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
          fontsize=11, framealpha=0.95, edgecolor='gray', 
          fancybox=True, shadow=True)

# Tight layout
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the figure
# plt.savefig('test_metrics_radar_chart.png', dpi=300, bbox_inches='tight')