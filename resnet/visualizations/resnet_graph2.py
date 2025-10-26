import matplotlib.pyplot as plt
import numpy as np

# Data for all three pooling methods
epochs = np.arange(1, 11)

gem_f1 = [64.53, 72.92, 72.39, 73.27, 73.89, 73.64, 75.04, 75.51, 74.50, 74.09]
concat_f1 = [44.06, 73.15, 46.53, 69.25, 72.11, 74.69, 71.61, 72.91, 72.62, 73.59]
eca_f1 = [44.06, 47.33, 57.42, 61.80, 66.89, 71.80, 69.85, 70.53, 72.30, 72.82]

# Baseline ResNet50 data (from training output - validation F1 scores)
baseline_f1 = [56.37, 76.97, 75.90, 77.41, 76.37, 79.22, 70.22, 78.82, 75.79, 80.03]

# Create figure with higher DPI for better quality
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

# Plot baseline first (as reference)
ax.plot(epochs, baseline_f1, marker='D', linewidth=3, markersize=8,
         label='Baseline ResNet50', color='#9966FF', markeredgecolor='white',
         markeredgewidth=2, alpha=0.9, linestyle='--')

# Plot lines with markers
ax.plot(epochs, gem_f1, marker='o', linewidth=3, markersize=8,
         label='GeM Pooling', color='#FF6384', markeredgecolor='white',
         markeredgewidth=2, alpha=0.9)

ax.plot(epochs, concat_f1, marker='s', linewidth=3, markersize=8,
         label='Concat Pooling', color='#36A2EB', markeredgecolor='white',
         markeredgewidth=2, alpha=0.9)

ax.plot(epochs, eca_f1, marker='^', linewidth=3, markersize=9,
         label='ECA Attention', color='#4BC0C0', markeredgecolor='white',
         markeredgewidth=2, alpha=0.9)

# Customize the plot
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation F1-Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Training Progress - Validation F1 Score Comparison',
              fontsize=16, fontweight='bold', pad=20)

# Set axis limits and ticks
ax.set_xlim(0.5, 10.5)
ax.set_ylim(40, 85)
ax.set_xticks(epochs)
ax.set_yticks(np.arange(40, 90, 5))

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Customize legend
ax.legend(loc='lower right', fontsize=12, framealpha=0.95,
           edgecolor='gray', fancybox=True, shadow=True)

# Add annotations for peak values
peak_baseline = max(baseline_f1)
peak_baseline_epoch = baseline_f1.index(peak_baseline) + 1
ax.annotate(f'Peak: {peak_baseline:.2f}%',
             xy=(peak_baseline_epoch, peak_baseline),
             xytext=(peak_baseline_epoch + 0.5, peak_baseline - 2),
            fontsize=10, fontweight='bold', color='#9966FF',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#9966FF', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#9966FF', lw=1.5))

peak_gem = max(gem_f1)
peak_gem_epoch = gem_f1.index(peak_gem) + 1
ax.annotate(f'Peak: {peak_gem:.2f}%',
             xy=(peak_gem_epoch, peak_gem),
             xytext=(peak_gem_epoch - 1, peak_gem + 2),
            fontsize=10, fontweight='bold', color='#FF6384',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#FF6384', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#FF6384', lw=1.5))

peak_concat = max(concat_f1)
peak_concat_epoch = concat_f1.index(peak_concat) + 1
ax.annotate(f'Peak: {peak_concat:.2f}%',
             xy=(peak_concat_epoch, peak_concat),
             xytext=(peak_concat_epoch + 0.5, peak_concat + 2),
            fontsize=10, fontweight='bold', color='#36A2EB',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#36A2EB', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#36A2EB', lw=1.5))

peak_eca = max(eca_f1)
peak_eca_epoch = eca_f1.index(peak_eca) + 1
ax.annotate(f'Peak: {peak_eca:.2f}%',
             xy=(peak_eca_epoch, peak_eca),
             xytext=(peak_eca_epoch - 1.5, peak_eca - 3),
            fontsize=10, fontweight='bold', color='#4BC0C0',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#4BC0C0', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#4BC0C0', lw=1.5))

# Style improvements
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the figure
# plt.savefig('training_progress_validation_f1.png', dpi=300, bbox_inches='tight')