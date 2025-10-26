import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Data from the document
thresholds = ['Baseline\n(0.5)', 'Threshold\n0.45', 'Threshold\n0.40', 
              'Threshold\n0.35', 'Threshold\n0.30', 'Threshold\n0.25', 
              'Conservative\n(0.2)']
miss_rates = [68.8, 64.8, 59.2, 56.8, 52.8, 48.8, 40.0]
recalls = [31.2, 35.2, 40.8, 43.2, 47.2, 51.2, 60.0]
false_negatives = [86, 81, 74, 71, 66, 61, 50]
true_positives = [39, 44, 51, 54, 59, 64, 75]

# Color mapping based on severity
colors = ['#ef4444', '#ef4444', '#f97316', '#f97316', '#f59e0b', '#eab308', '#22c55e']
statuses = ['Critical', 'Critical', 'Poor', 'Poor', 'Moderate', 'Acceptable', 'Good']

# Create figure with custom size
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#fef9f3')
ax.set_facecolor('white')

# Create horizontal bars
y_pos = np.arange(len(thresholds))
bars = ax.barh(y_pos, miss_rates, color=colors, edgecolor='white', linewidth=2, height=0.7)

# Add value labels at the end of bars
for i, (bar, miss_rate) in enumerate(zip(bars, miss_rates)):
    width = bar.get_width()
    ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, 
            f'{miss_rate}%', 
            ha='left', va='center', fontsize=12, fontweight='bold', color='#1f2937')

# Reference lines
ax.axvline(x=50, color='#f59e0b', linestyle='--', linewidth=2, alpha=0.7, label='Moderate Threshold: 50%')
ax.axvline(x=30, color='#22c55e', linestyle='--', linewidth=2, alpha=0.7, label='Target: <30%')

# Labels and title
ax.set_xlabel('Miss Rate (%)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('')
ax.set_yticks(y_pos)
ax.set_yticklabels(thresholds, fontsize=11, fontweight='500')
ax.set_xlim(0, 80)

# Grid
ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1)
ax.set_axisbelow(True)

# Title
fig.suptitle('Miss Rate Comparison: Critical Conservation Metric', 
             fontsize=18, fontweight='bold', y=0.98, color='#1f2937')
ax.text(0.5, 1.02, 'Percentage of unhealthy coral missed by the detection system (False Negative Rate)',
        transform=ax.transAxes, ha='center', fontsize=11, color='#6b7280', style='italic')

# Legend for reference lines
ax.legend(loc='lower right', fontsize=10, framealpha=0.9, edgecolor='gray')

# Add severity zone legend
legend_elements = [
    mpatches.Patch(facecolor='#ef4444', label='Critical (>60%)', edgecolor='white', linewidth=1),
    mpatches.Patch(facecolor='#f97316', label='Poor (50-60%)', edgecolor='white', linewidth=1),
    mpatches.Patch(facecolor='#eab308', label='Acceptable (40-50%)', edgecolor='white', linewidth=1),
    mpatches.Patch(facecolor='#22c55e', label='Good (<40%)', edgecolor='white', linewidth=1)
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
          title='Severity Zones', title_fontsize=10, framealpha=0.95, edgecolor='gray')

plt.tight_layout(rect=[0, 0.15, 1, 0.96])

# Add comparison cards at the bottom
fig.text(0.15, 0.12, '🚨 Baseline (0.5)', fontsize=12, fontweight='bold', color='#991b1b')
fig.text(0.15, 0.09, '68.8% Miss Rate', fontsize=16, fontweight='bold', color='#dc2626')
fig.text(0.15, 0.06, 'Missing: 86 corals', fontsize=9, color='#dc2626')
fig.text(0.15, 0.04, 'Detecting: 39 corals', fontsize=9, color='#16a34a')
fig.text(0.15, 0.01, 'Out of 125 diseased corals', fontsize=8, color='#6b7280', style='italic')

fig.text(0.42, 0.12, '✅ Conservative (0.2)', fontsize=12, fontweight='bold', color='#166534')
fig.text(0.42, 0.09, '40.0% Miss Rate', fontsize=16, fontweight='bold', color='#16a34a')
fig.text(0.42, 0.06, 'Missing: 50 corals', fontsize=9, color='#dc2626')
fig.text(0.42, 0.04, 'Detecting: 75 corals', fontsize=9, color='#16a34a')
fig.text(0.42, 0.01, 'Out of 125 diseased corals', fontsize=8, color='#6b7280', style='italic')

fig.text(0.70, 0.12, '📈 Improvement', fontsize=12, fontweight='bold', color='#1e40af')
fig.text(0.70, 0.09, '-28.8% Absolute', fontsize=16, fontweight='bold', color='#2563eb')
fig.text(0.70, 0.06, '36 more diseased corals detected', fontsize=9, color='#2563eb')
fig.text(0.70, 0.04, '42% fewer missed cases', fontsize=9, color='#6b7280')
fig.text(0.70, 0.01, 'Conservation impact', fontsize=8, color='#6b7280', style='italic')

# Add conservation alert box
alert_box = Rectangle((0.02, 0.89), 0.96, 0.06, 
                       transform=fig.transFigure,
                       facecolor='#fee2e2', edgecolor='#dc2626', linewidth=2)
fig.patches.append(alert_box)

fig.text(0.04, 0.935, '⚠️ Conservation Priority', fontsize=11, fontweight='bold', 
         color='#991b1b', transform=fig.transFigure)
fig.text(0.04, 0.91, 
         'Missing diseased coral allows disease to spread, leading to ecosystem collapse. Lower miss rates are critical for effective conservation intervention.',
         fontsize=9, color='#374151', transform=fig.transFigure, wrap=True)

plt.savefig('miss_rate_comparison.png', dpi=300, bbox_inches='tight', facecolor='#fef9f3')
print("✅ Figure saved as 'miss_rate_comparison.png'")
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("MISS RATE ANALYSIS SUMMARY")
print("="*70)
print(f"\n{'Threshold':<20} {'Miss Rate':<15} {'Recall':<10} {'FN':<8} {'TP':<8} {'Status'}")
print("-"*70)
for i in range(len(thresholds)):
    thresh_clean = thresholds[i].replace('\n', ' ')
    print(f"{thresh_clean:<20} {miss_rates[i]:>6.1f}%        {recalls[i]:>5.1f}%    {false_negatives[i]:>3}     {true_positives[i]:>3}     {statuses[i]}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print(f"• Baseline (0.5) misses {false_negatives[0]}/125 diseased corals ({miss_rates[0]}%)")
print(f"• Conservative (0.2) misses {false_negatives[-1]}/125 diseased corals ({miss_rates[-1]}%)")
print(f"• Improvement: {false_negatives[0] - false_negatives[-1]} more diseased corals detected")
print(f"• Relative improvement: {((false_negatives[0] - false_negatives[-1])/false_negatives[0]*100):.1f}% fewer missed cases")
print("\n💡 Recommendation: Deploy with threshold ≤ 0.25 for conservation-appropriate detection")
print("="*70)