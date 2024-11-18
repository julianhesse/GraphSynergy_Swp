import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('data/DrugCombDB/drug_combination_processed.csv')


print(data.head())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='drug1_db', y='synergy', hue='synergistic', data=data)
plt.title('Drug Combination Synergy')
plt.xlabel('Drug  Database ID')
plt.ylabel('Synergy Score')
plt.legend(title='Synergistic')
plt.show()
