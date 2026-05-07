import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

# Input F1-Scores
results = {
    'Random Forest': {'Music': 0.76, 'Coffee': 0.72, 'Perfume': 0.74, 'Overall': 0.74},
    'Naïve Bayes': {'Music': 0.68, 'Coffee': 0.65, 'Perfume': 0.62, 'Overall': 0.65},
    '1D-CNN': {'Music': 0.82, 'Coffee': 0.79, 'Perfume': 0.81, 'Overall': 0.81}
}

# Prepare data for plotting
models = list(results.keys())
categories = list(results['Random Forest'].keys())

data = np.array([[results[model][cat] for cat in categories] for model in models])

x = np.arange(len(categories))  
width = 0.25  

# Generate the Graph
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, data[0], width, label='Random Forest', color='#1f77b4')
rects2 = ax.bar(x, data[1], width, label='Naïve Bayes', color='#ff7f0e')
rects3 = ax.bar(x + width, data[2], width, label='1D-CNN', color='#2ca02c')

# Labels and title
ax.set_ylabel('F1-Score')
ax.set_title('Comparative AI Performance on Cognitive State Decoding (EEG)', pad=20, fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Target line and legend
ax.axhline(y=0.75, color='r', linestyle='--', alpha=0.7, label='Target (75%)')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.show()