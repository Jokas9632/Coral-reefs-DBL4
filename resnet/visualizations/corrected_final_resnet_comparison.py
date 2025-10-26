import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
ax.axis('tight')
ax.axis('off')

# Function to calculate F2 score
def calculate_f2(precision, recall):
    """Calculate F2 score: F2 = (1 + β²) * (precision * recall) / (β² * precision + recall)
    where β = 2 (weights recall higher than precision)"""
    beta = 2
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

# Per-class metrics: [Model/Experiment Name, Healthy Precision, Healthy Recall, Unhealthy Precision, Unhealthy Recall]
# F2-Score will be calculated
# For Cost-Sensitive (FN=7×FP): TP=162, FN=57, total unhealthy=219
# Unhealthy: Precision=41.97%, Recall=73.97%
# Need to calculate healthy metrics from confusion matrix
# From validation set size of 589: TN+FP+FN+TP = 589
# TN+FP = 589-219 = 370 (total healthy)
# If unhealthy precision = TP/(TP+FP) = 162/(162+FP) = 0.4197
# Solving: 162 = 0.4197(162+FP) → FP ≈ 224
# TN = 370 - 224 = 146
# Healthy Precision = TN/(TN+FN) = 146/(146+57) = 71.92%
# Healthy Recall = TN/(TN+FP) = 146/(146+224) = 39.46%

precision_recall_data = [
    ['Baseline ResNet50', 90.75, 96.21, 78.21, 58.10],
    ['GeM Pooling', 91.35, 96.05, 65.52, 45.24],
    ['Concat Pooling', 89.89, 98.42, 77.78, 33.33],
    ['ECA Attention', 90.81, 97.63, 73.91, 40.48],
    ['CLAHE\nPreprocessing', 88.07, 97.96, 75.00, 31.58],
    ['Threshold Adjustment\n(Postprocessing)', 79.46, 95.14, 50.20, 58.40],
    ['Cost Sensitive\nDecision Making\n(Postprocessing)', 71.92, 39.46, 41.97, 73.97]  # Updated with FN=7×FP results
]

# Calculate F2 scores and format data for display
data = []
for row in precision_recall_data:
    model = row[0]
    healthy_prec = row[1]
    healthy_rec = row[2]
    unhealthy_prec = row[3]
    unhealthy_rec = row[4]
    
    healthy_f2 = calculate_f2(healthy_prec, healthy_rec)
    unhealthy_f2 = calculate_f2(unhealthy_prec, unhealthy_rec)
    
    data.append([
        model,
        f'{healthy_prec:.2f}',
        f'{healthy_rec:.2f}',
        f'{healthy_f2:.2f}',
        f'{unhealthy_prec:.2f}',
        f'{unhealthy_rec:.2f}',
        f'{unhealthy_f2:.2f}'
    ])

# Column headers
columns = ['Model/Experiment', 
           'Healthy\nPrecision (%)', 'Healthy\nRecall (%)', 'Healthy\nF2-Score (%)',
           'Unhealthy\nPrecision (%)', 'Unhealthy\nRecall (%)', 'Unhealthy\nF2-Score (%)']

# Create the table
table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.8)  # Increased scale to accommodate multi-line text

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
        
        # Bold model names and enable text wrapping
        if j == 0:
            cell.set_text_props(weight='bold', fontsize=10, ha='center', va='center')
        else:
            cell.set_text_props(fontsize=10)
        
        # Add special styling for CLAHE row (preprocessing experiment)
        if i == 5:  # CLAHE row
            if j == 0:
                cell.set_facecolor('#e7f3ff')
                cell.set_text_props(weight='bold', fontsize=10, ha='center', va='center', color='#0056b3')
            else:
                cell.set_facecolor('#e7f3ff')
        
        # Add special styling for Threshold adjustment row (postprocessing experiment)
        if i == 6:  # Threshold row
            if j == 0:
                cell.set_facecolor('#fff3e0')
                cell.set_text_props(weight='bold', fontsize=10, ha='center', va='center', color='#e65100')
            else:
                cell.set_facecolor('#fff3e0')
        
        # Add special styling for Cost-Sensitive row (postprocessing experiment)
        if i == 7:  # Cost-Sensitive row
            if j == 0:
                cell.set_facecolor('#fff3e0')
                cell.set_text_props(weight='bold', fontsize=10, ha='center', va='center', color='#e65100')
            else:
                cell.set_facecolor('#fff3e0')
        
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
#fig.text(0.5, 0.02, 'Note: F2-Score weights recall 2x higher than precision. Best values (green) and worst values (red). Blue=Preprocessing, Orange=Postprocessing.',
         #ha='center', fontsize=9, style='italic', color='#6c757d')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# Optional: Save the figure
# plt.savefig('per_class_performance_table_f2_complete.png', dpi=300, bbox_inches='tight')