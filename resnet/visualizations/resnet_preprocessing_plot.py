import matplotlib.pyplot as plt
import numpy as np

# Data from CLAHE experiment
epochs = [1, 2, 3, 4]

# Healthy class metrics
healthy_precision = [0.8807, 0.8448, 0.8415, 0.8241]
healthy_recall = [0.9796, 1.0000, 0.7041, 0.9082]
healthy_f1 = [0.9275, 0.9159, 0.7667, 0.8641]

# Unhealthy class metrics
unhealthy_precision = [0.7500, 1.0000, 0.1714, 0.0000]
unhealthy_recall = [0.3158, 0.0526, 0.3158, 0.0000]
unhealthy_f1 = [0.4444, 0.1000, 0.2222, 0.0000]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training Progression: CLAHE Preprocessing Experiment', fontsize=16, fontweight='bold')

# Plot 1: Precision
axes[0].plot(epochs, healthy_precision, marker='o', linewidth=2, markersize=8, 
             label='Healthy', color='#2ecc71')
axes[0].plot(epochs, unhealthy_precision, marker='s', linewidth=2, markersize=8, 
             label='Unhealthy', color='#e74c3c')
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Precision', fontsize=12, fontweight='bold')
axes[0].set_title('Precision Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xticks(epochs)
axes[0].set_ylim(0, 1.05)
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].legend(loc='best', fontsize=11)
axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

# Plot 2: Recall
axes[1].plot(epochs, healthy_recall, marker='o', linewidth=2, markersize=8, 
             label='Healthy', color='#2ecc71')
axes[1].plot(epochs, unhealthy_recall, marker='s', linewidth=2, markersize=8, 
             label='Unhealthy', color='#e74c3c')
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Recall', fontsize=12, fontweight='bold')
axes[1].set_title('Recall Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xticks(epochs)
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].legend(loc='best', fontsize=11)
axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

# Highlight best epoch (Epoch 1)
for ax in axes:
    ax.axvline(x=1, color='gold', linestyle='--', alpha=0.4, linewidth=2, label='Best Model')

plt.tight_layout()
plt.savefig('training_progression_clahe.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("TRAINING PROGRESSION SUMMARY")
print("="*70)
print("\nHealthy Class Performance:")
print(f"  Precision: {min(healthy_precision):.4f} - {max(healthy_precision):.4f} (Δ {max(healthy_precision)-min(healthy_precision):.4f})")
print(f"  Recall:    {min(healthy_recall):.4f} - {max(healthy_recall):.4f} (Δ {max(healthy_recall)-min(healthy_recall):.4f})")

print("\nUnhealthy Class Performance:")
print(f"  Precision: {min(unhealthy_precision):.4f} - {max(unhealthy_precision):.4f} (Δ {max(unhealthy_precision)-min(unhealthy_precision):.4f})")
print(f"  Recall:    {min(unhealthy_recall):.4f} - {max(unhealthy_recall):.4f} (Δ {max(unhealthy_recall)-min(unhealthy_recall):.4f})")

print("\n✅ Best Model: Epoch 1 (F1 macro avg: 0.6860)")
print("="*70)