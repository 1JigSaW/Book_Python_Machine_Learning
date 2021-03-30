import os
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import perceptron
import plt_vis
import adaline

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

# plt_vis.plot_decision_regions(X, y, classifier=ppn)
# plot.xlabel('длина чашелистика [см]')
# plot.ylabel('длина лепестка [см]')
# plot.legend(loc='upper left')
# plot.show()

fig, ах= plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = adaline.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ах[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ах[0].set_xlabel ('Эпохи')
ax[0].set_ylabel('log(Cyммa квадратичных ошибок)')
ax[0].set_title('Adaline - скорость обучения 0.01')
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, у)
ax[1].plot(range(1, len(ada2.cost ) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('Cyммa квадратичных ошибок')
ax[1].set_title('Adaline - скорость обучения 0.0001')
plt.show()