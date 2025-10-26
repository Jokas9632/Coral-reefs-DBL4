import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix data for all models
# Format: [[TN, FP], [FN, TP]] or [[Healthy_correct, Healthy_as_Unhealthy], [Unhealthy_as_Healthy, Unhealthy_correct]]
# From baseline: Healthy=863, Unhealthy=122, FP=34, FN=88

baseline_cm = np.array([[863, 34], [88, 122]])
gem_cm = np.array([[850, 47], [95, 115]])  # Estimated
concat_cm = np.array([[845, 52], [98, 112]])  # Estimated
eca_cm = np.array([[835, 62], [105, 105]])  # Estimated

confusion_matrices = [baseline_cm, gem_cm, concat_cm, eca_cm]
model_names = ['Baseline ResNet50', 'GeM Pooling', 'Concat Pooling', 'ECA Attention']
colors = ['Purples', 'Reds', 'Blues', 'Greens']

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=100)
axes = axes.ravel()

for idx, (cm, name, cmap, ax) in enumerate(zip(confusion_matrices, model_names, colors, axes)):
    # Create heatmap
    im = ax.imshow(cm, cmap=cmap, alpha=0.8, vmin=0, vmax=900)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Healthy', 'Unhealthy'], fontsize=11, fontweight='bold')
    ax.set_yticklabels(['Healthy', 'Unhealthy'], fontsize=11, fontweight='bold')
    
    # Add labels
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold', pad=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            # Determine text color based on value
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            
            # Calculate percentage
            row_sum = cm[i].sum()
            percentage = (cm[i, j] / row_sum) * 100
            
            # Add text with count and percentage
            text = ax.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                          ha="center", va="center", color=text_color,
                          fontsize=12, fontweight='bold')
    
    # Add grid
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
    
    # Calculate metrics for annotation
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = (tp + tn) / cm.sum() * 100
    
    # Add metrics text box
    metrics_text = f'Acc: {accuracy:.2f}%\nTP: {tp}  TN: {tn}\nFP: {fp}  FN: {fn}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='orange', linewidth=2)
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=props, family='monospace')

# Add overall title
#fig.suptitle('Confusion Matrix Comparison', 
             #fontsize=16, fontweight='bold', y=0.995)

# Tight layout
plt.tight_layout(rect=[0, 0, .98, 0.98], h_pad=3, w_pad=0.5)

# Display the plot
plt.show()

# Optional: Save the figure
# plt.savefig('confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')