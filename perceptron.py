import numpy as np

class Perceptron(object):
	'''Классификация на основе персептрона.
	Параметры
	_________
	eta:float
		Скорость обучения
	n_iter:int
		Проходы по обучающему набору данных
	random_state:int
		Начальное значение генератора 
		случайных чисел для инициализации 
		случайными весами
	Атрибуты
	________
	w_: list
		Веса после подгонки
	errors_: list
		Количество неправильных классификаций в каждой эпохе
	'''
	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self, X, y):
		'''Подгоняет к обучающим данным
		Параметры
		_________
		X: массив, форма = [n_examples, n_features]
			Обучающие векторы, где n_examples - количество образцов
			и n_features - количество признаков
		y: массив, форма = [n_examples]
			Целевые значения
		Возвращает
		__________
		self: object
		'''
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
		self.errors_ = []

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	def net_input(self, X):
		'''Вычисляет общий вход'''
		return np.dot(X, self.w_p[1:]) + self.w_[0]

	def predict(self, X):
		'''Возвращает метку класса после единичного шага'''
		return np.where(self.net_input(X) >= 0.0, 1, -1)