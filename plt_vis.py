from matplotlib.colors import ListedColormap
import numpy as np
def plot_decision_regions(X, y, classifier, resolution=0.02):
	#настроить генератор маркеров и карту цветов
	markers = ('s','x', 'o', '4', '4')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
		np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xxl, хх2, Z, alpha=0.3, cmap=cmap)
	plt.xlim(xxl.min(), xxl.max())
	plt.ylim(xx2.min(), xx2.max())
	# вывести образцы по классам
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, О], у=Х[у == cl, 1],
			alpha=0.8, c=colors [idx], marker=markers[idx],
			label=cl, edgecolor='Ьlack')