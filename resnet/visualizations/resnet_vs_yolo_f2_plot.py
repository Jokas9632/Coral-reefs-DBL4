import matplotlib.pyplot as plt
import numpy as np

# Data from training logs
models = ['ResNet50', 'YOLOv11']
healthy_precision = [90.75, 46.67]
healthy_recall = [96.21, 10.14]
unhealthy_precision = [78.21, 88.48]
unhealthy_recall = [58.10, 98.35]

# Calculate F2 scores (F2 = 5 * precision * recall / (4 * precision + recall))
def calculate_f2(precision, recall):
    """Calculate F2 score which weighs recall higher than precision"""
    if precision + recall == 0:
        return 0
    return (5 * precision * recall) / (4 * precision + recall)

healthy_f2 = [calculate_f2(p, r) for p, r in zip(healthy_precision, healthy_recall)]
unhealthy_f2 = [calculate_f2(p, r) for p, r in zip(unhealthy_precision, unhealthy_recall)]

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('ResNet50 vs YOLOv11: Coral Health Classification Comparison', 
             fontsize=18, fontweight='bold', y=0.98)

# Color scheme
colors = ['#3b82f6', '#ef4444']  # Blue and Red
bar_width = 0.35
x = np.arange(len(models))

# Plot 1: Healthy Class - Precision
ax1 = axes[0, 0]
bars1 = ax1.bar(x, healthy_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax1.set_title('Healthy Class - Precision', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{healthy_precision[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=11, color='white')

# Plot 2: Healthy Class - Recall
ax2 = axes[0, 1]
bars2 = ax2.bar(x, healthy_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax2.set_title('Healthy Class - Recall', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{healthy_recall[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=11, color='white')

# Plot 3: Healthy Class - F2-Score
ax3 = axes[0, 2]
bars3 = ax3.bar(x, healthy_f2, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('F2-Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('Healthy Class - F2-Score', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=11)
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{healthy_f2[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=11, color='white')

# Plot 4: Unhealthy Class - Precision
ax4 = axes[1, 0]
bars4 = ax4.bar(x, unhealthy_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax4.set_title('Unhealthy Class - Precision', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=11)
ax4.set_ylim(0, 100)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_precision[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=11, color='white')

# Plot 5: Unhealthy Class - Recall
ax5 = axes[1, 1]
bars5 = ax5.bar(x, unhealthy_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax5.set_title('Unhealthy Class - Recall', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(models, fontsize=11)
ax5.set_ylim(0, 100)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars5):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_recall[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=11, color='white')

# Plot 6: Unhealthy Class - F2-Score
ax6 = axes[1, 2]
bars6 = ax6.bar(x, unhealthy_f2, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_ylabel('F2-Score (%)', fontsize=12, fontweight='bold')
ax6.set_title('Unhealthy Class - F2-Score', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(models, fontsize=11)
ax6.set_ylim(0, 100)
ax6.grid(axis='y', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars6):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{unhealthy_f2[i]:.2f}%', ha='center', va='center', 
            fontweight='bold', fontsize=11, color='white')

# Add overall note
fig.text(0.5, 0.02, 
         'Note: F2-Score emphasizes recall over precision. ResNet50 excels at Healthy coral detection, while YOLOv11 shows strong performance on Unhealthy coral but struggles with Healthy samples.',
         ha='center', fontsize=11, style='italic', color='#6c757d', wrap=True)

plt.tight_layout(rect=[0, 0.04, 1, 0.98])
plt.show()

# Optional: Save the figure
# plt.savefig('resnet_vs_yolov11_comparison_f2_plots.png', dpi=300, bbox_inches='tight')