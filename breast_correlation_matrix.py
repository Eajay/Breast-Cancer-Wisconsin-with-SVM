import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
# change Bare Nuclei from string to float
data["Bare Nuclei"] = pd.to_numeric(data["Bare Nuclei"], errors='coerce')
data.drop(['Sample code number'], axis=1, inplace=True)
correlations = data.corr()
plt.subplots(figsize=(5, 5))
sns.heatmap(correlations, annot=True, vmax=1, square=True, cmap="Blues")
plt.show()