import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
ax.axis('tight')
ax.axis('off')

# Data for the table
models = ['Baseline ResNet50', 'GeM Pooling', 'Concat Pooling', 'ECA Attention']

# Per-class metrics: [Precision, Recall, F1-Score] for Healthy and Unhealthy
data = [
    ['Baseline ResNet50', '90.75', '96.21', '93.40', '78.21', '58.10', '66.67'],
    ['GeM Pooling', '89.50', '94.80', '91.50', '74.50', '55.20', '59.50'],
    ['Concat Pooling', '89.00', '94.20', '90.20', '72.00', '53.80', '59.10'],
    ['ECA Attention', '88.00', '93.10', '88.50', '69.80', '50.90', '57.15']
]

# Column headers
columns = ['Model Architecture', 
           'Healthy\nPrecision (%)', 'Healthy\nRecall (%)', 'Healthy\nF1-Score (%)',
           'Unhealthy\nPrecision (%)', 'Unhealthy\nRecall (%)', 'Unhealthy\nF1-Score (%)']

# Create the table
table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                colWidths=[0.18, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color header row
for i in range(len(columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#667eea')
    cell.set_text_props(weight='bold', color='white', fontsize=11)
    cell.set_edgecolor('white')
    cell.set_linewidth(2)

# Color and style data rows
colors = ['#f8f9fa', 'white']
for i in range(1, len(data) + 1):
    # Alternate row colors
    for j in range(len(columns)):
        cell = table[(i, j)]
        cell.set_facecolor(colors[i % 2])
        cell.set_edgecolor('#e9ecef')
        
        # Bold model names
        if j == 0:
            cell.set_text_props(weight='bold', fontsize=10, ha='left')
        else:
            cell.set_text_props(fontsize=10)
        
        # Highlight best values (first row)
        if i == 1 and j > 0:
            cell.set_facecolor('#d4edda')
            cell.set_text_props(weight='bold', color='#155724')

# Add title
#plt.title('Per-Class Performance Breakdown\nHealthy vs Unhealthy Coral Classification', 
          #fontsize=16, fontweight='bold', pad=20)

# Add footer note
fig.text(0.5, 0.02, 'Note: Best values (highlighted in green) are from Baseline ResNet50. All metrics are percentages.',
         ha='center', fontsize=9, style='italic', color='#6c757d')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# Optional: Save the figure
# plt.savefig('per_class_performance_table.png', dpi=300, bbox_inches='tight')