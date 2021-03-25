import os
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import perceptron
import plt_vis

s = os.path.join('https://archive.ics.uci.edu', 'ml',
	'machine-learning-databases',
	'iris', 'iris.datа')
print('URL:', s)

iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(df.tail())
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='щетинистый')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')
plt.xlabel('длина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.show()

ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Эпохи')
# plt.ylabel('Количество обновлений')
# plt.show()

plt_vis.plot_decision_regions(X, y, classifier=ppn)
plot.xlabel('длина чашелистика [см]')
plot.ylabel('длина лепестка [см]')
plot.legend(loc='upper left')
plot.show()