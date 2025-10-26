import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Your confusion matrices (2x2 numpy arrays)
baseline_cm = np.array([[863, 34], [88, 122]])
gem_cm = np.array([[850, 47], [95, 115]])
concat_cm = np.array([[845, 52], [98, 112]])
eca_cm = np.array([[835, 62], [105, 105]])

confusion_matrices = [baseline_cm, gem_cm, concat_cm, eca_cm]
model_names = ['Baseline ResNet50', 'GeM Pooling', 'Concat Pooling', 'ECA Attention']
class_labels = ['Healthy', 'Unhealthy']

fig, axes = plt.subplots(2, 2, figsize=(9, 6))

for cm, name, ax in zip(confusion_matrices, model_names, axes.ravel()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', ax=ax, colorbar=False, values_format='d')  # plot without colorbar here
    ax.set_title(name, fontsize=14, fontweight='bold')
    # Use disp.im_ attribute to get the AxesImage for the colorbar
    cbar = fig.colorbar(disp.im_, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
