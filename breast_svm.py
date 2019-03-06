import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
import time

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

print(data[:1, :])
# normalize the data
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(data)
print(data[:1, :])

training_size = 547
validation_size = 68
testing_size = 68

training_data = data[:training_size, :]
training_labels = label[:training_size]
validation_data = data[training_size:training_size+validation_size, :]
validation_labels = label[training_size:training_size+validation_size]
testing_data = data[training_size+validation_size:, :]
testing_labels = label[training_size+validation_size:]
# kernel: rbf, linear, poly(polynomial), sigmoid
start1 = time.time()
clf = SVC(kernel='linear', C=1)
print(clf.fit(training_data, training_labels))
end1 = time.time()
# n_support_: how many support vectors each
# support_: index of support vectors in inputs
# support_vectors_: all support vectors each
start2 = time.time()
res = clf.predict(testing_data)
end2 = time.time()

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


for i in range(testing_size):
    if testing_labels[i] == 4 and res[i] == 4:
        true_positive += 1
    elif testing_labels[i] == 2 and res[i] == 2:
        true_negative += 1
    elif testing_labels[i] == 4 and res[i] == 2:
        false_negative += 1
    else:
        false_positive += 1
print("true positive: ", true_positive)
print("true negative: ", true_negative)
print("false positive: ", false_positive)
print("false negative: ", false_negative)
print("True Positive Rate: ", true_positive/(true_positive+false_negative))
print("True Negative Rate: ", true_negative/(true_negative+false_positive))

print("The time consume in svm training: ", end1 - start1, "s")
print("The average time consume in one test: ", (end2 - start2)/testing_size, "s")


