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
# split samples into inputs and labels
label = data.values[:, -1]
data = data.values[:, 1:-1]
data = data.astype(np.float64)


# normalize the data
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(data)

training_size = 547
validation_size = 68
testing_size = 68

training_data = data[:training_size, :]
training_labels = label[:training_size]
validation_data = data[training_size:training_size+validation_size, :]
validation_labels = label[training_size:training_size+validation_size]
testing_data = data[training_size+validation_size:, :]
testing_labels = label[training_size+validation_size:]

pca = decomposition.PCA(n_components=2)
training_data = pca.fit_transform(training_data)
fig = plt.figure()
plt.title('Scatter Plot')
plt.xlabel('value x')
plt.ylabel('value y')
data_2_x = []
data_2_y = []
data_4_x = []
data_4_y = []
for i in range(training_size):
    if training_labels[i] == 2:
        data_2_x.append(training_data[i][0])
        data_2_y.append(training_data[i][1])
    else:
        data_4_x.append(training_data[i][0])
        data_4_y.append(training_data[i][1])
plt.scatter(data_2_x, data_2_y, c='b', marker='o', label='2')
plt.scatter(data_4_x, data_4_y, c='r', marker='o', label='4')

plt.legend(bbox_to_anchor=[1, 1])
plt.show()

