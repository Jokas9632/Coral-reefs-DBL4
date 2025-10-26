import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
ax.axis('tight')
ax.axis('off')

# Data for the table
models = ['Baseline ResNet50', 'GeM Pooling', 'Concat Pooling', 'ECA Attention']

# Per-class metrics: [Precision, Recall, F1-Score] for Healthy and Unhealthy
# From actual test results
data = [
    ['Baseline ResNet50', '90.75', '96.21', '93.40', '78.21', '58.10', '66.67'],
    ['GeM Pooling', '91.35', '96.05', '93.64', '65.52', '45.24', '53.52'],
    ['Concat Pooling', '89.89', '98.42', '93.96', '77.78', '33.33', '46.67'],
    ['ECA Attention', '90.81', '97.63', '94.10', '73.91', '40.48', '52.31']
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
        
        # Highlight best and worst values per column
        if j > 0:  # Skip model name column
            col_values = [float(row[j]) for row in data]
            max_value = max(col_values)
            min_value = min(col_values)
            current_value = float(data[i-1][j])
            
            if current_value == max_value:
                cell.set_facecolor('#d4edda')
                cell.set_text_props(weight='bold', color='#155724')
            elif current_value == min_value:
                cell.set_facecolor('#f8d7da')
                cell.set_text_props(weight='bold', color='#721c24')

# Add title
#plt.title('Per-Class Performance Breakdown\nHealthy vs Unhealthy Coral Classification', 
          #fontsize=16, fontweight='bold', pad=20)

# Add footer note
#fig.text(0.5, 0.02, 'Note: Best values per metric (green) and worst values per metric (red) show performance range across models.',
         #ha='center', fontsize=9, style='italic', color='#6c757d')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# Optional: Save the figure
# plt.savefig('per_class_performance_table.png', dpi=300, bbox_inches='tight')