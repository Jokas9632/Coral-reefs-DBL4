import matplotlib.pyplot as plt
import numpy as np

# Data from ResNet variants
models = ['Baseline\nResNet50', 'GeM\nPooling', 'Concat\nPooling', 'ECA\nAttention']
healthy_precision = [90.75, 91.35, 89.89, 90.81]
healthy_recall = [96.21, 96.05, 98.42, 97.63]
healthy_f1 = [93.40, 93.64, 93.96, 94.10]
unhealthy_precision = [78.21, 65.52, 77.78, 73.91]
unhealthy_recall = [58.10, 45.24, 33.33, 40.48]
unhealthy_f1 = [66.67, 53.52, 46.67, 52.31]

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('ResNet Architecture Variants: Coral Health Classification Comparison', 
             fontsize=18, fontweight='bold', y=0.98)

# Color scheme - colorblind-safe palette
colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']  # Blue, Orange, Teal, Purple
x = np.arange(len(models))

# Plot 1: Healthy Class - Precision
ax1 = axes[0, 0]
bars1 = ax1.bar(x, healthy_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax1.set_title('Healthy Class - Precision', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{healthy_precision[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Plot 2: Healthy Class - Recall
ax2 = axes[0, 1]
bars2 = ax2.bar(x, healthy_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax2.set_title('Healthy Class - Recall', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{healthy_recall[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Plot 3: Healthy Class - F1-Score
ax3 = axes[0, 2]
bars3 = ax3.bar(x, healthy_f1, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('Healthy Class - F1-Score', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=10)
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{healthy_f1[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Plot 4: Unhealthy Class - Precision
ax4 = axes[1, 0]
bars4 = ax4.bar(x, unhealthy_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax4.set_title('Unhealthy Class - Precision', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=10)
ax4.set_ylim(0, 100)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_precision[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Plot 5: Unhealthy Class - Recall
ax5 = axes[1, 1]
bars5 = ax5.bar(x, unhealthy_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax5.set_title('Unhealthy Class - Recall', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(models, fontsize=10)
ax5.set_ylim(0, 100)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars5):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_recall[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Plot 6: Unhealthy Class - F1-Score
ax6 = axes[1, 2]
bars6 = ax6.bar(x, unhealthy_f1, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax6.set_title('Unhealthy Class - F1-Score', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(models, fontsize=10)
ax6.set_ylim(0, 100)
ax6.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars6):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_f1[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Add overall note
fig.text(0.5, 0.02, 
         'Note: All variants perform similarly on Healthy coral (~93-94% F1), but Baseline ResNet50 significantly outperforms others on Unhealthy coral detection (66.67% F1).',
         ha='center', fontsize=11, style='italic', color='#6c757d', wrap=True)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.show()

# Optional: Save the figure
# plt.savefig('resnet_variants_comparison_plots.png', dpi=300, bbox_inches='tight')