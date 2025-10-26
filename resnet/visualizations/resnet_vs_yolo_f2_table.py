import matplotlib.pyplot as plt
import numpy as np

# Function to calculate F2 score
def calculate_f2(precision, recall):
    """
    Calculate F2 score which weights recall higher than precision
    F2 = 5 * (precision * recall) / (4 * precision + recall)
    """
    if (4 * precision + recall) == 0:
        return 0
    return (5 * precision * recall) / (4 * precision + recall)

# Original precision and recall values
resnet_healthy_p = 90.75
resnet_healthy_r = 96.21
resnet_unhealthy_p = 78.21
resnet_unhealthy_r = 58.10

yolo_healthy_p = 46.67
yolo_healthy_r = 10.14
yolo_unhealthy_p = 88.48
yolo_unhealthy_r = 98.35

# Calculate F2 scores
resnet_healthy_f2 = calculate_f2(resnet_healthy_p, resnet_healthy_r)
resnet_unhealthy_f2 = calculate_f2(resnet_unhealthy_p, resnet_unhealthy_r)
yolo_healthy_f2 = calculate_f2(yolo_healthy_p, yolo_healthy_r)
yolo_unhealthy_f2 = calculate_f2(yolo_unhealthy_p, yolo_unhealthy_r)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
ax.axis('tight')
ax.axis('off')

# Data for the table - models as rows
data = [
    ['ResNet50\nHealthy', f'{resnet_healthy_p:.2f}', f'{resnet_healthy_r:.2f}', f'{resnet_healthy_f2:.2f}'],
    ['YOLOv11\nHealthy', f'{yolo_healthy_p:.2f}', f'{yolo_healthy_r:.2f}', f'{yolo_healthy_f2:.2f}'],
    ['ResNet50\nUnhealthy', f'{resnet_unhealthy_p:.2f}', f'{resnet_unhealthy_r:.2f}', f'{resnet_unhealthy_f2:.2f}'],
    ['YOLOv11\nUnhealthy', f'{yolo_unhealthy_p:.2f}', f'{yolo_unhealthy_r:.2f}', f'{yolo_unhealthy_f2:.2f}']
]

# Column headers
columns = ['Model', 'Precision (%)', 'Recall (%)', 'F2-Score (%)']

# Create the table
table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                colWidths=[0.15, 0.23, 0.23, 0.24])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 2.5)

# Color header row
for i in range(len(columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#667eea')
    cell.set_text_props(weight='bold', color='white', fontsize=15)
    cell.set_edgecolor('white')
    cell.set_linewidth(2)

# Color and style data rows
colors = ['#f8f9fa', 'white', '#f8f9fa', 'white']
for i in range(1, len(data) + 1):
    for j in range(len(columns)):
        cell = table[(i, j)]
        cell.set_facecolor(colors[i - 1])
        cell.set_edgecolor('#e9ecef')
        
        # Bold metric names
        if j == 0:
            cell.set_text_props(weight='bold', fontsize=13, ha='left')
        else:
            cell.set_text_props(fontsize=14)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# Print F2 scores for reference
print("\nF2 Scores Calculated:")
print(f"ResNet50  - Healthy: {resnet_healthy_f2:.2f}%, Unhealthy: {resnet_unhealthy_f2:.2f}%")
print(f"YOLOv11   - Healthy: {yolo_healthy_f2:.2f}%, Unhealthy: {yolo_unhealthy_f2:.2f}%")
print(f"\nNote: F2 score weights recall 2x more than precision, useful when minimizing false negatives is important.")