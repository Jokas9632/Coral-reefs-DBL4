import matplotlib.pyplot as plt
import numpy as np

# Data from ResNet variants
models = ['Baseline\nResNet50', 'GeM\nPooling', 'Concat\nPooling', 'ECA\nAttention']
unhealthy_precision = [78.21, 65.52, 77.78, 73.91]
unhealthy_recall = [58.10, 45.24, 33.33, 40.48]

# Calculate F2 scores: F2 = 5 * (precision * recall) / (4 * precision + recall)
def calculate_f2(precision, recall):
    return (5 * precision * recall) / (4 * precision + recall)

unhealthy_f2 = [calculate_f2(p, r) for p, r in zip(unhealthy_precision, unhealthy_recall)]

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ResNet Architecture Variants: Unhealthy Coral Detection Comparison', 
             fontsize=18, fontweight='bold', y=0.98)

# Color scheme - colorblind-safe palette
colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']  # Blue, Orange, Teal, Purple
x = np.arange(len(models))

# Plot 1: Unhealthy Class - Precision
ax1 = axes[0]
bars1 = ax1.bar(x, unhealthy_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax1.set_title('Unhealthy Class - Precision', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_precision[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Plot 2: Unhealthy Class - Recall
ax2 = axes[1]
bars2 = ax2.bar(x, unhealthy_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax2.set_title('Unhealthy Class - Recall', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_recall[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Plot 3: Unhealthy Class - F2-Score
ax3 = axes[2]
bars3 = ax3.bar(x, unhealthy_f2, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('F2-Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('Unhealthy Class - F2-Score', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=10)
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_f2[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Add overall note
fig.text(0.5, 0.02, 
         'Note: F2-Score emphasizes recall over precision (β=2). Baseline ResNet50 significantly outperforms variants on Unhealthy coral detection (61.24% F2).',
         ha='center', fontsize=11, style='italic', color='#6c757d', wrap=True)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.show()

# Optional: Save the figure
# plt.savefig('resnet_variants_unhealthy_comparison.png', dpi=300, bbox_inches='tight')