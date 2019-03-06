import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
import matplotlib.pyplot as plt

column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
# change Bare Nuclei from string to float
data["Bare Nuclei"] = pd.to_numeric(data["Bare Nuclei"], errors='coerce')

del data['Sample code number']
data.hist()
plt.show()
